"""
persona_lean.py — Detect which persona a response leaned on.

Cheap, deterministic, runs at log time. No extra LLM call. Counts
case-insensitive substring matches per persona bucket using markers
configured in config.PERSONA_MARKERS, then picks the highest non-zero
bucket. Ties → "mixed". All zero → "none".

To tune detection, edit PERSONA_MARKERS in config.py — no code change here.

Returned value is one of config.PERSONA_LEAN_LABELS so the column stays
constrained to a small known set and stats panels can group on it.
"""

import logging
from typing import Optional

from config import PERSONA_MARKERS, PERSONA_NAMES

log = logging.getLogger(__name__)


def detect_persona_lean(response: Optional[str]) -> str:
    """Return the persona this response leaned on, or 'none'."""
    if not response or not response.strip():
        return "none"

    text = response.lower()
    counts: dict[str, int] = {}
    for persona, markers in PERSONA_MARKERS.items():
        if persona not in PERSONA_NAMES:
            # Stays aligned with config: only the canonical personas count.
            continue
        if not markers:
            counts[persona] = 0
            continue
        counts[persona] = sum(1 for m in markers if m and m.lower() in text)

    if not counts:
        return "none"

    max_hits = max(counts.values())
    if max_hits == 0:
        return "none"

    leaders = [p for p, c in counts.items() if c == max_hits]
    if len(leaders) > 1:
        return "mixed"
    return leaders[0]
