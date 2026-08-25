"""prefs.py — Persistent UI preferences for Agent Ouija.

Simple JSON file at data/preferences.json. Loaded on UI init,
updated via .change() handlers as the user toggles things in the
Control Center / Executive Boardroom. Reset button in the Data
tab clears the file, next launch (or click) falls back to
DEFAULTS below.

Keys stored today:
  energy_level      — MYSTICISM_LEVELS[N] string, or None to fall
                      back to MYSTICISM_LEVELS[0]
  boardroom_toggle  — bool, "🌐 Always Use Boardroom Mode"
  sr_with_jr        — bool, "Sr. Execs. with selected Jr. Execs"
  last_executive    — display name of the last-picked Executive
                      Model, so a fresh launch re-selects whatever
                      the user had before instead of falling back
                      to Claude Sonnet.

Cheap to load — a few hundred bytes of JSON. Never raises; a
corrupt file is logged and treated as "use defaults."
"""

from __future__ import annotations

import json
import logging

from config import DATA_DIR

log = logging.getLogger(__name__)

PREFS_PATH = DATA_DIR / "preferences.json"

DEFAULTS: dict = {
    "energy_level":     None,   # caller substitutes MYSTICISM_LEVELS[0]
    "boardroom_toggle": True,   # boardroom mode ON out of the box
    "sr_with_jr":       False,
    "last_executive":   None,   # None → caller falls back to
                                # _load_default_exec / Claude Sonnet
                                # for Medium+ tiers.
}


def load() -> dict:
    """Return the current preferences dict, backfilled with defaults
    for any keys the on-disk file is missing. Never raises."""
    if not PREFS_PATH.exists():
        return dict(DEFAULTS)
    try:
        data = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.warning(
            "preferences.json unreadable (%s) — using defaults", e,
        )
        return dict(DEFAULTS)
    out = dict(DEFAULTS)
    if isinstance(data, dict):
        for k in DEFAULTS:
            if k in data:
                out[k] = data[k]
    return out


def save(prefs: dict) -> None:
    """Persist the full preferences dict to disk. Called after any
    single-field update via update()."""
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(
            json.dumps(prefs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        log.warning("Could not save preferences: %s", e)


def update(**changes) -> dict:
    """Merge changes into current prefs, save, return the new dict.
    Wired as a lambda into gr.Component.change() handlers."""
    prefs = load()
    prefs.update(changes)
    save(prefs)
    return prefs


def reset() -> None:
    """Delete the prefs file. Next load() returns DEFAULTS."""
    try:
        PREFS_PATH.unlink(missing_ok=True)
    except OSError as e:
        log.warning("Could not delete preferences.json: %s", e)
