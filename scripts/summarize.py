"""
summarize.py — Historical reports of arena_results.

Two report flavors, both windowed to the **last 24 hours from the most recent
scored prompt**. This keeps reports focused on the current training cycle and
prevents older 0-3-scale rows or wiped-but-not-deleted experiments from
polluting the view.

  generate_summary()         → plain-text per-question detail report
                                  (question + every model's response + score)
  generate_trainer_report()  → per-model Taught / Learned counts for the
                                  trainer dashboard

Both write to data/summaries/<kind>_YYYYMMDD_HHMMSS.txt and return a status
dict suitable for CLI exit-coding or UI status display:
    {"status": "success"|"warning"|"error",
     "path": str | None,
     "message": str,
     ...kind-specific fields}

Usage:
    python scripts/summarize.py            # summary report
    python scripts/summarize.py --trainer  # trainer report
    # or click the matching button on the Stats tab in arena_ui
"""

import argparse
import logging
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta

from config import SUMMARIES_DIR
from db import connection_scope

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# All reports look back this many hours from MAX(timestamp). "From the last
# prompt" rather than "from now" so a report generated days later still
# reflects what happened in the final 24h of activity.
WINDOW_HOURS = 24


def _latest_and_cutoff(conn) -> tuple[str, str] | None:
    """Return (latest_iso, cutoff_iso) for the 24h window, or None if no
    scored rows exist. Both are ISO 8601 strings comparable as TEXT in SQL.
    """
    row = conn.execute(
        "SELECT MAX(timestamp) FROM arena_results WHERE score >= 0"
    ).fetchone()
    if not row or not row[0]:
        return None
    latest = row[0]
    try:
        latest_dt = datetime.fromisoformat(latest)
    except ValueError:
        log.warning("Could not parse latest timestamp %r — skipping window filter", latest)
        return None
    cutoff = (latest_dt - timedelta(hours=WINDOW_HOURS)).isoformat()
    return latest, cutoff


# ── Summary report (per-question detail) ──────────────────────────────────────

def generate_summary() -> dict:
    """Per-question detail report for the last WINDOW_HOURS hours."""
    result = {"status": "success", "path": None, "rows": 0, "questions": 0, "message": ""}

    try:
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        result.update(status="error", message=f"Could not create summaries dir: {e}")
        return result

    try:
        with connection_scope() as conn:
            window = _latest_and_cutoff(conn)
            if window is None:
                result.update(status="warning",
                              message="No scored arena results to summarize yet")
                return result
            latest, cutoff = window
            rows = conn.execute("""
                SELECT timestamp, question, model_display, model_type, response,
                       score, response_time_seconds, reading, role, tags,
                       persona_lean, input_tokens, output_tokens,
                       input_cost, output_cost
                FROM arena_results
                WHERE score >= 0 AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (cutoff,)).fetchall()
    except Exception as e:
        result.update(status="error", message=f"DB query failed: {e}")
        return result

    if not rows:
        result.update(
            status="warning",
            message=f"No scored rows in the last {WINDOW_HOURS}h (latest: {latest[:19]})",
        )
        return result

    grouped: OrderedDict = OrderedDict()
    for row in rows:
        ts, question = row[0], row[1]
        key = (ts[:16], question)   # YYYY-MM-DDTHH:MM
        grouped.setdefault(key, []).append(row)

    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUMMARIES_DIR / f"summary_{ts_now}.txt"
    bar = "─" * 78
    sep = "·" * 78

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(bar + "\n")
            f.write(f"  MR. OUIJA — HISTORICAL SUMMARY (last {WINDOW_HOURS}h)\n")
            f.write(f"  Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"  Window:    {cutoff[:19]} → {latest[:19]}\n")
            f.write(f"  Questions: {len(grouped)}    Scored responses: {len(rows)}\n")
            f.write(bar + "\n\n")

            for idx, ((ts_min, question), responses) in enumerate(grouped.items(), 1):
                f.write(f"[{idx}]  {ts_min}\n")
                f.write("QUESTION:\n")
                f.write(f"{question}\n\n")
                f.write(sep + "\n")

                for row in responses:
                    (_, _, model_display, model_type, response,
                     score, rt, reading, role, tags,
                     persona_lean, in_tok, out_tok, in_cost, out_cost) = row
                    mt = model_type or "?"
                    header = (
                        f"  {model_display}  ({mt})  ·  score={score}  ·  "
                        f"{rt}s  ·  Reading={reading}"
                    )
                    f.write(header + "\n")
                    meta = []
                    if role:
                        meta.append(f"role={role}")
                    if persona_lean:
                        meta.append(f"persona={persona_lean}")
                    if tags:
                        meta.append(f"tags={tags}")
                    total_cost = (in_cost or 0.0) + (out_cost or 0.0)
                    if (in_tok or out_tok or total_cost):
                        meta.append(
                            f"tokens={in_tok or 0}/{out_tok or 0}"
                            f" cost=${total_cost:.4f}"
                        )
                    if meta:
                        f.write("  " + "  ·  ".join(meta) + "\n")
                    f.write("\n")
                    f.write((response or "").rstrip() + "\n")
                    f.write("\n" + sep + "\n")

                f.write("\n")
    except OSError as e:
        result.update(status="error", message=f"Failed to write {out_path.name}: {e}")
        return result

    result.update(
        path=str(out_path),
        rows=len(rows),
        questions=len(grouped),
        message=(
            f"Wrote {out_path.name}  ·  {len(grouped)} questions, "
            f"{len(rows)} scored responses (last {WINDOW_HOURS}h)"
        ),
    )
    return result


# ── Trainer report (per-model Taught / Learned counts) ────────────────────────

def generate_trainer_report() -> dict:
    """Per-model Taught / Learned counts for the trainer dashboard.

    PAIRWISE tally: for each round (grouped by round_id when present, else
    falling back to the legacy (ts[:16], question) bucket for older rows),
    every ordered pair (A, B):
        A.score >= B.score → A.taught  += 1
        A.score <= B.score → A.learned += 1
    Ties count for BOTH columns — A tied with B means A taught B (and B
    taught A) AND A learned from B (and B learned from A). The user's rule
    is "every rating of equal or lesser value should be learned".

    Single-model rounds contribute no pairwise data but the answering
    model still appears in the report with N>0 / 0 / 0 so the user can
    see *every* model that responded in the window.
    """
    result = {
        "status": "success", "path": None, "models": 0, "questions": 0, "message": "",
    }

    try:
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        result.update(status="error", message=f"Could not create summaries dir: {e}")
        return result

    try:
        with connection_scope() as conn:
            window = _latest_and_cutoff(conn)
            if window is None:
                result.update(status="warning", message="No scored arena results yet")
                return result
            latest, cutoff = window
            rows = conn.execute("""
                SELECT timestamp, question, model_display, model_type, score, round_id
                FROM arena_results
                WHERE score >= 0 AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (cutoff,)).fetchall()
            # Per-model RAG contributions in the same window. Counts every
            # successful Score + Add To RAG (Arena/Research) and Add To RAG
            # (Big 3), grouped by model_display so trainer-report column
            # joins cleanly to the rest of the per-model stats.
            try:
                rag_counts_rows = conn.execute("""
                    SELECT model_display, COUNT(*)
                    FROM rag_contributions
                    WHERE timestamp >= ?
                    GROUP BY model_display
                """, (cutoff,)).fetchall()
            except Exception:
                # Legacy DB without rag_contributions table — fall through
                # with zeros so the report still renders.
                rag_counts_rows = []
            rag_adds: dict = {r[0]: r[1] for r in rag_counts_rows}
    except Exception as e:
        result.update(status="error", message=f"DB query failed: {e}")
        return result

    if not rows:
        result.update(
            status="warning",
            message=f"No scored rows in the last {WINDOW_HOURS}h (latest: {latest[:19]})",
        )
        return result

    # Group responses by round_id when available (every modern row carries
    # one — generated at Ask time so co-asked peers stay together no matter
    # how long the user takes to score them). Legacy rows with NULL
    # round_id fall back to the old (minute, question) bucket so reports
    # over historical data still work.
    by_round: dict = defaultdict(list)
    for ts, question, model_display, model_type, score, round_id in rows:
        key = round_id if round_id else ("legacy", ts[:16], question)
        by_round[key].append((model_display, model_type or "?", score))

    # Pre-seed every model that appears anywhere in the window with 0/0 so
    # solo-round models still show up on the report. Without this a model
    # that only answered single-model questions would silently drop off.
    taught:       dict = defaultdict(int)
    learned:      dict = defaultdict(int)
    types:        dict = {}
    answer_count: dict = defaultdict(int)
    for _ts, _q, model_display, model_type, _score, _rid in rows:
        types[model_display] = model_type or "?"
        answer_count[model_display] += 1
        taught.setdefault(model_display, 0)
        learned.setdefault(model_display, 0)

    # Tally taught / learned per model — PAIRWISE. For each ordered pair
    # (A, B) within a round (i != j):
    #   A.score >= B.score → A taught B  (+1)
    #   A.score <= B.score → A learned from B (+1)
    # Ties count for BOTH sides — the user's rule is "every rating of equal
    # or lesser value should be learned". A round of N models where one is
    # alone at the top credits Taught += N-1.
    counted_questions = 0

    for responses in by_round.values():
        if len(responses) < 2:
            # Need at least two models to have a teacher/learner relationship.
            continue
        counted_questions += 1
        for i, (m_a, _t_a, s_a) in enumerate(responses):
            for j, (_, _, s_b) in enumerate(responses):
                if i == j:
                    continue
                if s_a >= s_b:
                    taught[m_a]  += 1
                if s_a <= s_b:
                    learned[m_a] += 1

    # Order: net teachers first, then by raw wins. Every model in the
    # window appears, even if it only saw single-model rounds.
    all_models = sorted(
        types.keys(),
        key=lambda m: (taught[m] - learned[m], taught[m], answer_count[m]),
        reverse=True,
    )

    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUMMARIES_DIR / f"trainer_report_{ts_now}.txt"
    bar = "─" * 78

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(bar + "\n")
            f.write(f"  MR. OUIJA — TRAINER REPORT (last {WINDOW_HOURS}h)\n")
            f.write(f"  Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"  Window:    {cutoff[:19]} → {latest[:19]}\n")
            f.write(f"  Questions: {counted_questions} (multi-model)    "
                    f"Scored responses: {len(rows)}    Models seen: {len(all_models)}\n")
            f.write(bar + "\n\n")

            # Surface any models that contributed to RAG but weren't scored
            # in this window (e.g. Big 3 tab contributions for a cloud
            # model the user didn't pit in the arena). They wouldn't appear
            # otherwise — counted_questions / taught / learned are scoring-
            # based.
            all_models = list(all_models) + [
                m for m in rag_adds.keys() if m not in types
            ]

            f.write(f"  {'Model':<26}  {'Type':<7}  {'N':>4}  {'Taught':>7}  {'Learned':>7}  {'Δ':>5}  {'RAG':>4}\n")
            f.write(f"  {'-' * 78}\n")
            for m in all_models:
                t = taught[m]
                l = learned[m]
                delta = t - l
                delta_str = f"+{delta}" if delta >= 0 else str(delta)
                f.write(
                    f"  {m:<26}  {types.get(m, '?'):<7}  "
                    f"{answer_count[m]:>4}  {t:>7}  {l:>7}  {delta_str:>5}  "
                    f"{rag_adds.get(m, 0):>4}\n"
                )

            f.write("\n")
            f.write("  RAG     = Score + Add To RAG (Arena/Research) + Add To RAG (Big 3)\n")
            f.write("            successful contributions to the mr_ouija ChromaDB collection\n")
            f.write("  N       = total scored answers this model gave in the window\n")
            f.write("  Taught  = pairwise wins-or-ties: peers this model scored >= against\n")
            f.write("            (one round of N models where this model is alone at the top\n")
            f.write("            credits Taught += N-1, not +1)\n")
            f.write("  Learned = pairwise losses-or-ties: peers this model scored <= against\n")
            f.write("  Δ       = Taught − Learned (positive = net teacher, negative = net learner)\n")
            f.write("  Note    = ties count for BOTH columns (the user's rule: every rating of\n")
            f.write("            equal or lesser value should be learned by a model). Rounds are\n")
            f.write("            grouped by round_id when present, else by (minute, question).\n")
            f.write("            Models with N>0 but Taught=Learned=0 only saw single-model rounds.\n")
    except OSError as e:
        result.update(status="error", message=f"Failed to write {out_path.name}: {e}")
        return result

    result.update(
        path=str(out_path),
        models=len(all_models),
        questions=counted_questions,
        message=(
            f"Wrote {out_path.name}  ·  {len(all_models)} models, "
            f"{counted_questions} multi-model questions (last {WINDOW_HOURS}h)"
        ),
    )
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agent Ouija historical reports")
    parser.add_argument("--trainer", action="store_true",
                        help="Generate trainer report (Taught/Learned counts) "
                             "instead of the per-question summary")
    args = parser.parse_args()

    if args.trainer:
        result = generate_trainer_report()
    else:
        result = generate_summary()

    if result["status"] == "error":
        log.error(result["message"])
        sys.exit(1)
    if result["status"] == "warning":
        log.warning(result["message"])
        sys.exit(2)
    log.info(result["message"])
    print(f"\n  ✅ {result['message']}")
    print(f"     Path: {result['path']}\n")


if __name__ == "__main__":
    main()
