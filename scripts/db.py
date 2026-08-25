"""
db.py — Single source of truth for arena_scores.db schema, indices, and connections.

Owns:
    - DDL for arena_results and training_checkpoints
    - Indices for hot query paths (timestamp/score range filters, model grouping)
    - PRAGMA tuning (WAL, synchronous, busy_timeout)
    - Connection factory + context manager

Why centralize:
    arena.py and arena_ui.py previously duplicated CREATE TABLE blocks. A drift
    between the two would silently corrupt assumptions (missing column, missing
    table on a fresh rig). This module is now the only place schema lives.

Usage:
    from db import get_conn, connection_scope

    with connection_scope() as conn:
        conn.execute("SELECT ...")

    # Or, when caller manages lifecycle (long-lived loops):
    conn = get_conn()
    try:
        ...
    finally:
        conn.close()
"""

import logging
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from config import (
    ARENA_DB_PATH,
    BACKUPS_DIR,
    CHAT_HISTORY_PATH,
    CHROMA_DIR,
    DATA_DIR,
    NEWS_CACHE_DIR,
    SEANCES_DIR,
)

log = logging.getLogger(__name__)


# Tables come first (idempotent CREATE), then column migrations (ALTER TABLE
# for existing DBs), then indices. Indices have to run LAST because some of
# them reference columns added via the migration step — running them before
# the ALTER fails with "no such column" on any existing DB.
TABLE_DDL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS arena_results (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp             TEXT,
        question              TEXT,
        model_key             TEXT,
        model_display         TEXT,
        model_type            TEXT,
        response              TEXT,
        score                 INTEGER,
        response_time_seconds REAL,
        reading               REAL,
        role                  TEXT,
        tags                  TEXT,
        tag_source            TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS training_checkpoints (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        name      TEXT,
        model_key TEXT,
        notes     TEXT
    )""",
    # One row per successful rag.add_to_collection call. Lets the Stats
    # tab and trainer report attribute curated-RAG contributions back to
    # the model that produced them — including Big-3 tab adds (which
    # don't go through scoring and so have no arena_results row).
    """CREATE TABLE IF NOT EXISTS rag_contributions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,
        model_key     TEXT,
        model_display TEXT,
        model_type    TEXT,
        source        TEXT,
        question      TEXT,
        round_id      TEXT
    )""",
)

# Columns added after the initial schema — applied via ALTER TABLE on existing
# DBs so old installs pick them up without a full wipe. (column_name, ddl_type)
_ARENA_RESULTS_ADDED_COLUMNS = (
    ("model_type",     "TEXT"),
    # persona_lean: which persona the response leaned on, derived at log
    # time by a keyword scan (see persona_lean.detect_persona_lean). One of
    # PERSONA_LEAN_LABELS — the user-defined PERSONA_NAMES plus "mixed"
    # and "none" for ambiguous / no-hit responses.
    ("persona_lean",   "TEXT"),
    # Token counts + cost in USD. Locals are hardcoded to 0; cloud models
    # use litellm.cost_per_token from the LiteLLM response usage block.
    ("input_tokens",   "INTEGER"),
    ("output_tokens",  "INTEGER"),
    ("input_cost",     "REAL"),
    ("output_cost",    "REAL"),
    # round_id: shared identifier for every model that answered the same
    # Ask click. Generated at on_ask time so the trainer report can group
    # peers by round_id instead of (timestamp_minute, question). Pairwise
    # tallying broke when scoring took long enough to cross a minute
    # boundary; round_id is bulletproof.
    ("round_id",       "TEXT"),
    # cost_reading: same shape as `reading` but with an extra log penalty
    # on USD cost. Computed at log time via config.compute_reading_with_cost.
    # Locals always have cost_reading == reading (cost=0 → log(e)=1).
    ("cost_reading",   "REAL"),
)

# Indices — stats tab filters/aggregates on timestamp and score; tagger
# filters on tags; leaderboard may group by model_type to split cloud vs local.
INDEX_DDL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_arena_results_timestamp    ON arena_results(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_arena_results_score        ON arena_results(score)",
    "CREATE INDEX IF NOT EXISTS idx_arena_results_tags         ON arena_results(tags)",
    "CREATE INDEX IF NOT EXISTS idx_arena_results_model        ON arena_results(model_display)",
    "CREATE INDEX IF NOT EXISTS idx_arena_results_model_type   ON arena_results(model_type)",
    "CREATE INDEX IF NOT EXISTS idx_arena_results_persona_lean ON arena_results(persona_lean)",
    "CREATE INDEX IF NOT EXISTS idx_arena_results_round_id      ON arena_results(round_id)",
    "CREATE INDEX IF NOT EXISTS idx_training_checkpoints_ts    ON training_checkpoints(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_rag_contributions_model    ON rag_contributions(model_display)",
    "CREATE INDEX IF NOT EXISTS idx_rag_contributions_ts       ON rag_contributions(timestamp)",
)

# Kept for backwards compatibility with any external caller that imported it.
SCHEMA_DDL: tuple[str, ...] = TABLE_DDL + INDEX_DDL

# Schema DDL is process-global state — only run once per interpreter.
_SCHEMA_INITIALIZED = False


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Enable WAL + sane concurrency defaults.

    WAL: arena_ui (writer) and stats refresh (reader) share the DB across
    Gradio's worker threads. Without WAL the reader blocks behind the writer.

    busy_timeout: prevents `database is locked` errors on brief contention.
    synchronous=NORMAL: safe with WAL; durability cost is bounded to one
    transaction on crash, which is acceptable for arena scoring data.
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
    except sqlite3.DatabaseError as e:
        log.warning("Failed to apply DB PRAGMAs (continuing with defaults): %s", e)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    try:
        # 1. Tables (idempotent — CREATE IF NOT EXISTS is a no-op on existing DBs).
        for stmt in TABLE_DDL:
            conn.execute(stmt)

        # 2. Column migrations — add columns that were introduced after the
        #    initial schema. PRAGMA table_info returns
        #    (cid, name, type, notnull, dflt_value, pk).
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(arena_results)").fetchall()
        }
        for col_name, col_type in _ARENA_RESULTS_ADDED_COLUMNS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE arena_results ADD COLUMN {col_name} {col_type}")
                log.info("Added %s column to arena_results", col_name)

        # 3. Indices — must run AFTER the column migration so that indices
        #    referencing newly-added columns (e.g. idx_arena_results_model_type)
        #    don't fail with "no such column" on an existing DB.
        for stmt in INDEX_DDL:
            conn.execute(stmt)

        conn.commit()
        _SCHEMA_INITIALIZED = True
    except sqlite3.DatabaseError:
        log.exception("Failed to initialize arena_scores.db schema")
        raise


def get_conn() -> sqlite3.Connection:
    """Return a PRAGMA-tuned, schema-initialized SQLite connection.

    Caller owns the connection lifecycle. For one-shot use prefer
    connection_scope().
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ARENA_DB_PATH, timeout=10.0)
    _apply_pragmas(conn)
    _ensure_schema(conn)
    return conn


@contextmanager
def connection_scope():
    """Context-manager wrapper around get_conn() that always closes the
    connection, even on exception. Use this for short-lived operations."""
    conn = get_conn()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            log.warning("Error closing SQLite connection", exc_info=True)


def log_rag_contribution(model_key: str,
                         model_display: str,
                         model_type: str,
                         source: str,
                         question: str,
                         round_id: str | None = None) -> None:
    """Insert one row recording that this model's response was added to RAG.

    Called from the Arena / Research / Big 3 click handlers after a
    successful rag.add_to_collection. Never raises — RAG ingest is the
    source of truth; this row is just attribution metadata, so a write
    failure logs and moves on rather than failing the user's click.
    """
    try:
        with connection_scope() as conn:
            conn.execute(
                """INSERT INTO rag_contributions
                   (timestamp, model_key, model_display, model_type, source,
                    question, round_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), model_key, model_display,
                 model_type, source, question, round_id),
            )
            conn.commit()
    except sqlite3.DatabaseError:
        log.exception(
            "log_rag_contribution failed for model %s (source=%s)",
            model_display, source,
        )


def backup_state(label: str = "wipe") -> dict:
    """Snapshot arena_scores.db, chat_history.json, and chroma/ into
    data/backups/<label>_YYYYMMDD_HHMMSS/.

    Used by the Stats-tab "Yes, Backup the database and chats" wipe path so
    the user can roll back. Each piece is independent — a missing file is
    reported but never blocks the others. ChromaDB is rebuildable from docs/
    but is still copied so a wiped collection can be restored verbatim.

    Returns: {"status": "success"|"partial"|"error",
              "path": str | None,
              "items": list[str],         # what got copied
              "missing": list[str],       # paths that didn't exist
              "errors": list[str],        # per-item failures
              "message": str}
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BACKUPS_DIR / f"{label}_{stamp}"
    result = {
        "status":  "success",
        "path":    None,
        "items":   [],
        "missing": [],
        "errors":  [],
        "message": "",
    }

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.exception("backup_state: could not create %s", target)
        result.update(status="error", message=f"Could not create backup dir: {e}")
        return result

    def _copy_file(src: Path, label_for_msg: str) -> None:
        if not src.exists():
            result["missing"].append(label_for_msg)
            return
        try:
            shutil.copy2(src, target / src.name)
            result["items"].append(label_for_msg)
        except OSError as e:
            log.warning("backup_state: copy %s failed: %s", src, e)
            result["errors"].append(f"{label_for_msg}: {e}")

    def _copy_tree(src: Path, label_for_msg: str) -> None:
        if not src.exists() or not src.is_dir():
            result["missing"].append(label_for_msg)
            return
        try:
            shutil.copytree(src, target / src.name, dirs_exist_ok=True)
            result["items"].append(label_for_msg)
        except OSError as e:
            log.warning("backup_state: copytree %s failed: %s", src, e)
            result["errors"].append(f"{label_for_msg}: {e}")

    _copy_file(ARENA_DB_PATH,     "arena_scores.db")
    _copy_file(CHAT_HISTORY_PATH, "chat_history.json")
    _copy_tree(CHROMA_DIR,        "chroma/")
    # Included so the time-based Rollback control can restore
    # deleted seance files and news docs; the full-Wipe path also
    # benefits — nothing lost by a bigger snapshot.
    _copy_tree(SEANCES_DIR,       "seances/")
    _copy_tree(NEWS_CACHE_DIR,    "news/")

    result["path"] = str(target)
    if result["errors"]:
        result["status"] = "partial" if result["items"] else "error"

    parts = []
    if result["items"]:
        parts.append("copied " + ", ".join(result["items"]))
    if result["missing"]:
        parts.append("missing " + ", ".join(result["missing"]))
    if result["errors"]:
        parts.append("errors " + "; ".join(result["errors"]))
    result["message"] = f"Backup → {target.name}: " + " · ".join(parts) if parts else f"Backup → {target.name}: nothing to copy"
    return result


def wipe_arena_db() -> dict:
    """Delete all rows from arena_results and training_checkpoints.

    Schema, columns, and indices are preserved — only the row data is removed.
    AUTOINCREMENT counters are reset so the next row starts at id=1 again.

    ChromaDB is NOT touched here. To rebuild RAG context, run a fresh ingest
    (the Ingest Documents button already drops + recreates the collection).

    Never raises — errors are surfaced via the returned dict so a UI button
    can render the failure cleanly.
    """
    result = {
        "status": "success",
        "rows_deleted": 0,
        "checkpoints_deleted": 0,
        "message": "",
    }
    try:
        with connection_scope() as conn:
            (rows,) = conn.execute("SELECT COUNT(*) FROM arena_results").fetchone()
            (cps,)  = conn.execute("SELECT COUNT(*) FROM training_checkpoints").fetchone()
            try:
                (rag_rows,) = conn.execute("SELECT COUNT(*) FROM rag_contributions").fetchone()
            except sqlite3.OperationalError:
                rag_rows = 0
            conn.execute("DELETE FROM arena_results")
            conn.execute("DELETE FROM training_checkpoints")
            try:
                conn.execute("DELETE FROM rag_contributions")
            except sqlite3.OperationalError:
                pass
            # sqlite_sequence holds AUTOINCREMENT row counts; deleting the
            # entries resets ids to 1 on the next insert. The table may not
            # exist if no AUTOINCREMENT row has ever been written.
            try:
                conn.execute(
                    "DELETE FROM sqlite_sequence "
                    "WHERE name IN ('arena_results', 'training_checkpoints', 'rag_contributions')"
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
    except sqlite3.DatabaseError as e:
        log.exception("wipe_arena_db failed")
        result.update(status="error", message=f"DB error: {e}")
        return result

    result.update(
        rows_deleted=rows,
        checkpoints_deleted=cps,
        rag_contributions_deleted=rag_rows,
        message=(
            f"Deleted {rows} arena_results rows, {cps} training_checkpoints, "
            f"{rag_rows} rag_contributions. Schema preserved."
        ),
    )
    return result


def count_arena_rows_before(cutoff_iso: str) -> dict:
    """Preview arena_results + rag_contributions row counts strictly
    before cutoff_iso. training_checkpoints are milestones and are
    NOT counted or purged — deleting them would corrupt the Δ Reading
    columns on the Stats tab."""
    result = {"arena_rows": 0, "rag_rows": 0, "error": None}
    try:
        with connection_scope() as conn:
            (arena_rows,) = conn.execute(
                "SELECT COUNT(*) FROM arena_results WHERE timestamp < ?",
                (cutoff_iso,),
            ).fetchone()
            try:
                (rag_rows,) = conn.execute(
                    "SELECT COUNT(*) FROM rag_contributions WHERE timestamp < ?",
                    (cutoff_iso,),
                ).fetchone()
            except sqlite3.OperationalError:
                rag_rows = 0
        result.update(arena_rows=arena_rows, rag_rows=rag_rows)
    except sqlite3.DatabaseError as e:
        log.exception("count_arena_rows_before failed")
        result["error"] = f"DB error: {e}"
    return result


def list_backups() -> list[dict]:
    """Return backup snapshots on disk, newest first. Each entry:
        {name, path, created_at (ISO), items (list of file/dir names)}
    Includes both wipe_* and rollback_* snapshots (naming pattern
    established by backup_state)."""
    out: list[dict] = []
    if not BACKUPS_DIR.exists():
        return out
    for p in BACKUPS_DIR.iterdir():
        if not p.is_dir():
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        items = []
        try:
            for child in p.iterdir():
                items.append(child.name)
        except OSError:
            pass
        out.append({
            "name":       p.name,
            "path":       str(p),
            "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(
                timespec="seconds",
            ),
            "items":      sorted(items),
        })
    out.sort(key=lambda d: d["name"], reverse=True)
    return out


def restore_backup(backup_name: str) -> dict:
    """Restore files from a specific backup snapshot into their
    original locations. Reverse of backup_state — copies
    arena_scores.db, chat_history.json, chroma/, seances/, news/
    from the backup directory back to the live paths.

    Existing live files are OVERWRITTEN. Directories that don't
    exist in the backup are LEFT ALONE (partial backups are fine).

    Never raises. Returns:
        {status: "success"|"partial"|"error",
         restored: list[str],
         missing: list[str],
         errors:  list[str],
         message: str}
    """
    result = {
        "status":   "success",
        "restored": [],
        "missing":  [],
        "errors":   [],
        "message":  "",
    }
    src_root = BACKUPS_DIR / backup_name
    if not src_root.exists() or not src_root.is_dir():
        result.update(
            status="error",
            message=f"Backup `{backup_name}` not found in {BACKUPS_DIR}",
        )
        return result

    def _restore_file(src_name: str, dst: Path, label: str) -> None:
        src = src_root / src_name
        if not src.exists():
            result["missing"].append(label)
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            result["restored"].append(label)
        except OSError as e:
            log.warning("restore_backup: copy %s → %s failed: %s",
                        src, dst, e)
            result["errors"].append(f"{label}: {e}")

    def _restore_tree(src_name: str, dst: Path, label: str) -> None:
        src = src_root / src_name
        if not src.exists() or not src.is_dir():
            result["missing"].append(label)
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            result["restored"].append(label)
        except OSError as e:
            log.warning("restore_backup: copytree %s → %s failed: %s",
                        src, dst, e)
            result["errors"].append(f"{label}: {e}")

    _restore_file("arena_scores.db",   ARENA_DB_PATH,     "arena_scores.db")
    _restore_file("chat_history.json", CHAT_HISTORY_PATH, "chat_history.json")
    _restore_tree("chroma",            CHROMA_DIR,        "chroma/")
    _restore_tree("seances",           SEANCES_DIR,       "seances/")
    _restore_tree("news",              NEWS_CACHE_DIR,    "news/")

    if result["errors"]:
        result["status"] = "partial" if result["restored"] else "error"
    parts = []
    if result["restored"]:
        parts.append("restored " + ", ".join(result["restored"]))
    if result["missing"]:
        parts.append("missing " + ", ".join(result["missing"]))
    if result["errors"]:
        parts.append("errors " + "; ".join(result["errors"]))
    result["message"] = " · ".join(parts) or "no-op"
    return result


def purge_arena_rows_before(cutoff_iso: str) -> dict:
    """Delete arena_results + rag_contributions rows with timestamp
    strictly less than cutoff_iso. training_checkpoints preserved.
    Returns a status dict; never raises."""
    result = {"status": "success", "arena_rows": 0, "rag_rows": 0,
              "message": ""}
    try:
        with connection_scope() as conn:
            (arena_rows,) = conn.execute(
                "SELECT COUNT(*) FROM arena_results WHERE timestamp < ?",
                (cutoff_iso,),
            ).fetchone()
            try:
                (rag_rows,) = conn.execute(
                    "SELECT COUNT(*) FROM rag_contributions WHERE timestamp < ?",
                    (cutoff_iso,),
                ).fetchone()
            except sqlite3.OperationalError:
                rag_rows = 0
            conn.execute(
                "DELETE FROM arena_results WHERE timestamp < ?",
                (cutoff_iso,),
            )
            try:
                conn.execute(
                    "DELETE FROM rag_contributions WHERE timestamp < ?",
                    (cutoff_iso,),
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        result.update(
            arena_rows=arena_rows, rag_rows=rag_rows,
            message=(
                f"Deleted {arena_rows} arena_results + {rag_rows} "
                f"rag_contributions rows with timestamp < {cutoff_iso}. "
                "Checkpoints preserved."
            ),
        )
    except sqlite3.DatabaseError as e:
        log.exception("purge_arena_rows_before failed")
        result.update(status="error", message=f"DB error: {e}")
    return result
