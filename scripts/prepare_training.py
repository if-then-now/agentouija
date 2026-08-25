"""
prepare_training.py — Fine-tuning data export

Exports scored arena responses as JSONL for LoRA fine-tuning via Unsloth.
Includes score, weight, response time, and The Reading in each record.

Score weights (0-4 baseball):
    4 = home run    → 1.0
    3 = extra bases → 0.75
    2 = base hit    → 0.50
    1 = contact     → 0.25
    0 = whiff       → excluded

Usage:
    python scripts/prepare_training.py                    # score >= 1
    python scripts/prepare_training.py --min-score 2      # base hits and above
    python scripts/prepare_training.py --min-score 3      # home runs only
    python scripts/prepare_training.py --tag "your-tag"   # filter by tag
    python scripts/prepare_training.py --model llama3.2   # filter by model
    python scripts/prepare_training.py --since-checkpoint # since last checkpoint only
    python scripts/prepare_training.py --stats            # leaderboard only
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from config import (
    SCORE_LABELS,
    SCORE_MIN_EXPORT,
    SCORE_WEIGHTS,
    TRAINING_DIR,
)
from db import connection_scope

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)


def get_last_checkpoint_ts(conn) -> str | None:
    row = conn.execute(
        "SELECT timestamp FROM training_checkpoints ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def print_stats(conn: sqlite3.Connection):
    rows = conn.execute("""
        SELECT model_display,
               ROUND(AVG(score), 2)                            AS avg_score,
               ROUND(AVG(response_time_seconds), 2)            AS avg_time,
               ROUND(AVG(reading), 4)                          AS avg_reading,
               COUNT(*)                                        AS total,
               SUM(CASE WHEN score = 4 THEN 1 ELSE 0 END)    AS home_runs,
               SUM(CASE WHEN score = 3 THEN 1 ELSE 0 END)    AS extra_bases,
               SUM(CASE WHEN score = 2 THEN 1 ELSE 0 END)    AS base_hits,
               SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END)    AS contact,
               SUM(CASE WHEN score = 0 THEN 1 ELSE 0 END)    AS whiffs
        FROM arena_results
        WHERE score >= 0
        GROUP BY model_display
        ORDER BY avg_reading DESC
    """).fetchall()

    if not rows:
        print("No scored data yet.")
        return

    print(f"\n{'─'*80}")
    print(f"  Arena leaderboard — 0-4 baseball · The Reading")
    print(f"{'─'*80}")
    print(f"  {'Model':<20} {'Score':>6} {'Time':>7} {'Reading':>9} {'Total':>6} {'HR':>4} {'XB':>4} {'Hit':>4} {'1B':>4} {'W':>4}")
    print(f"  {'─'*75}")
    for model, avg_score, avg_time, avg_reading, total, hr, xb, hit, contact, whiff in rows:
        print(f"  {model:<20} {avg_score:>6} {avg_time:>6}s {avg_reading:>9} {total:>6} {hr:>4} {xb:>4} {hit:>4} {contact:>4} {whiff:>4}")
    print(f"{'─'*80}\n")


def export_jsonl(conn, min_score, tag_filter, model_filter, since_checkpoint) -> Path | None:
    query  = "SELECT question, model_display, response, score, response_time_seconds, reading, role, tags FROM arena_results WHERE score >= ?"
    params = [min_score]

    if since_checkpoint:
        cp_ts = get_last_checkpoint_ts(conn)
        if cp_ts:
            query += " AND timestamp > ?"
            params.append(cp_ts)
            log.info("Filtering since checkpoint: %s", cp_ts)
        else:
            log.warning("No checkpoint found — exporting all records")

    if tag_filter:
        query += " AND tags LIKE ?"
        params.append(f"%{tag_filter}%")

    if model_filter:
        query += " AND (model_key = ? OR model_display LIKE ?)"
        params.extend([model_filter, f"%{model_filter}%"])

    query += " ORDER BY reading DESC, score DESC, timestamp DESC"
    rows = conn.execute(query, params).fetchall()

    if not rows:
        log.warning("No rows matched filters.")
        return None

    try:
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("Could not create %s: %s", TRAINING_DIR, e)
        return None

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = TRAINING_DIR / f"training_{ts}.jsonl"

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for question, model_display, response, score, resp_time, reading, role, tags in rows:
                record = {
                    "prompt":                question,
                    "response":              response,
                    "score":                 score,
                    "score_label":           SCORE_LABELS.get(score, str(score)),
                    "weight":                SCORE_WEIGHTS.get(score, 0.0),
                    "response_time_seconds": resp_time,
                    "reading":               reading,
                    "model":                 model_display,
                    "role":                  role,
                    "tags":                  tags,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        log.error("Failed to write %s: %s", out_path, e)
        return None

    log.info("Exported %d records → %s", len(rows), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export arena scores as JSONL for fine-tuning")
    parser.add_argument("--min-score",        type=int, default=SCORE_MIN_EXPORT)
    parser.add_argument("--tag",              default=None)
    parser.add_argument("--model",            default=None)
    parser.add_argument("--since-checkpoint", action="store_true",
                        help="Export only records since last training checkpoint")
    parser.add_argument("--stats",            action="store_true")
    args = parser.parse_args()

    try:
        with connection_scope() as conn:
            if args.stats:
                print_stats(conn)
                return
            print_stats(conn)
            out = export_jsonl(conn, args.min_score, args.tag, args.model, args.since_checkpoint)
            if out:
                print(f"  Output: {out}\n")
    except sqlite3.DatabaseError:
        log.exception("Database error during export")
        raise


if __name__ == "__main__":
    main()