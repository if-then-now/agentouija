"""
tagger.py — Hybrid auto-tagging for arena_scores.db

Layer 1: Keyword matching — runs at every log_score() call.
         Reads from data/TAGS_keywords.json (gitignored, personal).
         Edit that file to tune tags — no code change needed.

Layer 2: LLM classifier — runs on-demand for rows still tagged "general".

Usage:
    python scripts/tagger.py --mode keyword   # backfill keyword tags
    python scripts/tagger.py --mode llm       # LLM classify untagged rows
    python scripts/tagger.py --mode both      # run both in sequence
"""

import argparse
import json
import logging
import sqlite3

import requests

from config import DEFAULT_MODEL, OLLAMA_BASE_URL, TAGS_KEYWORDS_PATH
from db import connection_scope

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)


def load_keyword_map() -> dict:
    """Load KEYWORD_TAG_MAP from TAGS_keywords.json.
    Keys prefixed with '_' are metadata comments — skipped.
    Falls back to empty map with warning if file missing or malformed.
    """
    if not TAGS_KEYWORDS_PATH.exists():
        log.warning(
            "TAGS_keywords.json not found at %s — keyword tagging disabled.",
            TAGS_KEYWORDS_PATH,
        )
        return {}
    try:
        with open(TAGS_KEYWORDS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.error(
            "Failed to load %s (%s): %s — keyword tagging disabled.",
            TAGS_KEYWORDS_PATH, type(e).__name__, e,
        )
        return {}
    if not isinstance(raw, dict):
        log.error(
            "%s does not contain a JSON object at top level — keyword tagging disabled.",
            TAGS_KEYWORDS_PATH,
        )
        return {}
    keyword_map = {k: v for k, v in raw.items() if not k.startswith("_")}
    log.info("Loaded %d tag categories from %s", len(keyword_map), TAGS_KEYWORDS_PATH)
    return keyword_map


def tag_by_keywords(question: str, keyword_map: dict) -> str:
    """Return the best-matching tag for a question, or 'general'.

    Scoring rule:
      - Count case-insensitive substring matches per category.
      - Highest-count category wins.
      - Ties broken by insertion order in keyword_map (so the user can
        manually prioritize by ordering categories in TAGS_keywords.json).
      - Zero matches across all categories → "general".

    Defensive: empty / whitespace-only keywords are dropped. They are
    silently truthy in Python's substring test (`"" in s` is always True),
    so a typo'd entry like "" or " " would otherwise tag every prompt
    with whichever category contained it.
    """
    if not question:
        return "general"
    q_lower = question.lower()

    best_tag    = "general"
    best_hits   = 0
    for tag, keywords in keyword_map.items():
        if tag == "general" or not keywords:
            continue
        # Skip empty / whitespace-only keywords — they would match anything.
        valid = [kw for kw in keywords if isinstance(kw, str) and kw.strip()]
        if not valid:
            continue
        hits = sum(1 for kw in valid if kw.lower() in q_lower)
        if hits > best_hits:
            best_tag, best_hits = tag, hits
    return best_tag


def backfill_keyword_tags(conn: sqlite3.Connection, keyword_map: dict) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT id, question FROM arena_results")
    rows = cursor.fetchall()
    if not rows:
        log.info("No arena_results rows to tag")
        return 0
    updates = [
        (tag_by_keywords(question, keyword_map), "keyword", row_id)
        for row_id, question in rows
    ]
    cursor.executemany(
        "UPDATE arena_results SET tags = ?, tag_source = ? WHERE id = ?",
        updates,
    )
    conn.commit()
    log.info("Keyword backfill: updated %d rows", len(updates))
    return len(updates)


def classify_with_llm(question: str, available_tags: list) -> str:
    tag_list = ", ".join(t for t in available_tags if t != "general")
    prompt = (
        f"Classify this question into exactly one of these tags: {tag_list}.\n"
        f"If none fit, respond with: general\n"
        f"Respond with only the tag name, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json().get("response", "general").strip().lower()
    except requests.RequestException as e:
        log.warning("LLM classifier HTTP error (%s): %s", type(e).__name__, e)
        return "general"
    except ValueError as e:
        # JSON decode error from a malformed Ollama response
        log.warning("LLM classifier returned non-JSON: %s", e)
        return "general"

    for tag in available_tags:
        if tag.lower() == result:
            return tag
    return "general"


def backfill_llm_tags(conn: sqlite3.Connection, keyword_map: dict) -> int:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, question FROM arena_results WHERE tags = 'general' OR tags IS NULL"
    )
    rows = cursor.fetchall()
    if not rows:
        log.info("No 'general' rows to classify")
        return 0
    available_tags = list(keyword_map.keys())
    updates: list[tuple[str, str, int]] = []
    for row_id, question in rows:
        tag = classify_with_llm(question, available_tags)
        updates.append((tag, "llm_classifier", row_id))
        log.info("Row %d → %s", row_id, tag)
    cursor.executemany(
        "UPDATE arena_results SET tags = ?, tag_source = ? WHERE id = ?",
        updates,
    )
    conn.commit()
    log.info("LLM classification: updated %d rows", len(updates))
    return len(updates)


def main():
    parser = argparse.ArgumentParser(description="Hybrid auto-tagger for arena results")
    parser.add_argument("--mode", choices=["keyword", "llm", "both"], default="keyword")
    args = parser.parse_args()

    keyword_map = load_keyword_map()
    if not keyword_map:
        log.error("Cannot proceed without keyword map. Check %s", TAGS_KEYWORDS_PATH)
        return

    with connection_scope() as conn:
        if args.mode in ("keyword", "both"):
            backfill_keyword_tags(conn, keyword_map)
        if args.mode in ("llm", "both"):
            backfill_llm_tags(conn, keyword_map)


if __name__ == "__main__":
    main()
