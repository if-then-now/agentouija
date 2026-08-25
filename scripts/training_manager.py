"""
training_manager.py — Training checkpoint management

Marks training events in arena_scores.db so the stats tab can show
score deltas before and after each fine-tuning run.

The checkpoint marks the moment a model was actually updated — not just
when data was exported. Run this after deploying a new fine-tuned model.

Future: wire to Windows Task Scheduler or cron for automated checkpoints.

Usage:
    # Mark a checkpoint for all models
    python scripts/training_manager.py --checkpoint "LoRA run 1 — assignment 1 dictionary swap"

    # Mark a checkpoint for a specific model only
    python scripts/training_manager.py --checkpoint "Llama fine-tune v1" --model llama3.2

    # Add notes
    python scripts/training_manager.py --checkpoint "LoRA run 2" --notes "50 examples, lr=2e-4"

    # List all checkpoints
    python scripts/training_manager.py --list

    # Show leaderboard delta since last checkpoint
    python scripts/training_manager.py --delta
"""

import argparse
import logging
import sqlite3
from datetime import datetime

from db import connection_scope

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)


def add_checkpoint(name: str, model_key: str | None = None,
                   notes: str | None = None) -> dict:
    """Record a training checkpoint and return a status dict.

    Suitable for both CLI (via main()) and in-process callers (Gradio UI).
    Never raises — errors are logged and surfaced in result["message"].
    """
    ts = datetime.now().isoformat()
    try:
        with connection_scope() as conn:
            conn.execute(
                "INSERT INTO training_checkpoints (timestamp, name, model_key, notes) VALUES (?, ?, ?, ?)",
                (ts, name, model_key, notes),
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        log.exception("Failed to record checkpoint %r", name)
        return {"status": "error", "message": f"DB error: {e}", "timestamp": ts, "name": name}

    scope = f"model: {model_key}" if model_key else "all models"
    return {
        "status":    "success",
        "name":      name,
        "scope":     scope,
        "timestamp": ts,
        "notes":     notes,
        "message":   f"Checkpoint '{name}' logged at {ts[:19]} ({scope})",
    }


def list_checkpoints() -> None:
    with connection_scope() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, name, model_key, notes FROM training_checkpoints ORDER BY id DESC"
        ).fetchall()

    if not rows:
        print("\n  No checkpoints yet.\n")
        return

    print(f"\n{'─'*70}")
    print(f"  Training checkpoints")
    print(f"{'─'*70}")
    for cp_id, ts, name, model_key, notes in rows:
        print(f"  [{cp_id}] {ts[:19]}  |  {name}")
        print(f"       Scope: {model_key or 'all models'}")
        if notes:
            print(f"       Notes: {notes}")
    print(f"{'─'*70}\n")


def show_delta() -> None:
    with connection_scope() as conn:
        cp_row = conn.execute(
            "SELECT timestamp, name FROM training_checkpoints ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if not cp_row:
            print("\n  No checkpoints yet. Run --checkpoint first.\n")
            return

        cp_ts, cp_name = cp_row
        print(f"\n  Last checkpoint: {cp_name} ({cp_ts[:19]})\n")

        def get_stats(where, params):
            return conn.execute(f"""
                SELECT model_display,
                       ROUND(AVG(score), 2)        AS avg_score,
                       ROUND(AVG(reading), 4)      AS avg_reading,
                       COUNT(*)                    AS total
                FROM arena_results
                WHERE score >= 0 {where}
                GROUP BY model_display
                ORDER BY avg_reading DESC
            """, params).fetchall()

        after  = get_stats("AND timestamp > ?",  (cp_ts,))
        before = get_stats("AND timestamp <= ?", (cp_ts,))

    prev_map = {r[0]: r for r in before}

    if not after:
        print("  No data since last checkpoint yet.\n")
        return

    print(f"{'─'*65}")
    print(f"  {'Model':<20} {'Before Reading':>15} {'After Reading':>14} {'Δ Reading':>10}")
    print(f"  {'─'*60}")
    for model, avg_score, avg_reading, total in after:
        prev = prev_map.get(model)
        if prev:
            delta = round(avg_reading - prev[2], 4)
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            print(f"  {model:<20} {prev[2]:>15} {avg_reading:>14} {arrow} {delta:>8}")
        else:
            print(f"  {model:<20} {'—':>15} {avg_reading:>14} {'new':>10}")
    print(f"{'─'*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Agent Ouija training checkpoint manager")
    parser.add_argument("--checkpoint", default=None,
                        help="Name/description of this training checkpoint")
    parser.add_argument("--model",      default=None,
                        help="Limit checkpoint to a specific model key")
    parser.add_argument("--notes",      default=None,
                        help="Optional notes (e.g. hyperparameters, dataset size)")
    parser.add_argument("--list",       action="store_true",
                        help="List all checkpoints")
    parser.add_argument("--delta",      action="store_true",
                        help="Show Reading delta since last checkpoint")
    args = parser.parse_args()

    if args.checkpoint:
        result = add_checkpoint(args.checkpoint, args.model, args.notes)
        if result["status"] == "error":
            print(f"\n  ❌ {result['message']}\n")
            return
        print(f"\n  ✅ Checkpoint logged")
        print(f"     Name:  {result['name']}")
        print(f"     Scope: {result['scope']}")
        print(f"     Time:  {result['timestamp']}")
        if result.get("notes"):
            print(f"     Notes: {result['notes']}")
        print()
    elif args.list:
        list_checkpoints()
    elif args.delta:
        show_delta()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
