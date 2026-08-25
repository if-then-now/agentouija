"""
serper_search.py — Thin wrapper around the Serper.dev Google Search API.

Used by the Research tab to fetch real-time web results and format them
as a context block injectable into any model's system prompt — letting
locals + clouds both answer with grounded facts (unlike The Big 3 tab,
which relies on each provider's native web-search tool and so only
covers Claude / GPT-4o / Gemini).

API contract:
    POST https://google.serper.dev/search
        headers: X-API-KEY, Content-Type: application/json
        body:    {"q": "<query>"}
    response: JSON with "organic" array of {title, link, snippet}, plus
              optional "answerBox" / "knowledgeGraph" for direct answers.

Setup:
    1. Sign up at serper.dev (free tier: 2,500 searches/month).
    2. Drop SERPER_API_KEY=... into .env on Brainy.
    3. Restart arena_ui.py.

Behaviour:
    - search(query) → formatted context string or None on any failure.
    - Never raises — failure cases (missing key, network down, malformed
      response) all degrade to None so the caller treats it as "search
      gave us nothing, run the model without web context".
"""

import json
import logging
import os
from typing import Optional

import requests

from config import (
    SERPER_API_KEY_ENV,
    SERPER_ENDPOINT,
    SERPER_TIMEOUT,
    SERPER_TOP_K,
)

log = logging.getLogger(__name__)


def _format_results(payload: dict, top_k: int) -> Optional[str]:
    """Render Serper response into a context block. Returns None if there
    isn't enough material to be useful (no organic results AND no answer
    box / knowledge graph)."""
    lines: list[str] = []

    answer_box = payload.get("answerBox") or {}
    if answer_box.get("answer") or answer_box.get("snippet"):
        ab_text = answer_box.get("answer") or answer_box.get("snippet")
        ab_title = answer_box.get("title") or "Featured answer"
        ab_link  = answer_box.get("link", "")
        lines.append(f"[ANSWER BOX] {ab_title}")
        if ab_link:
            lines.append(f"  URL: {ab_link}")
        lines.append(f"  {ab_text}")
        lines.append("")

    knowledge = payload.get("knowledgeGraph") or {}
    if knowledge.get("description"):
        lines.append(f"[KNOWLEDGE] {knowledge.get('title', '')}")
        lines.append(f"  {knowledge['description']}")
        if knowledge.get("descriptionLink"):
            lines.append(f"  URL: {knowledge['descriptionLink']}")
        lines.append("")

    organic = payload.get("organic") or []
    for i, item in enumerate(organic[:top_k], 1):
        title   = (item.get("title") or "Untitled").strip()
        link    = (item.get("link") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        lines.append(f"[{i}] {title}")
        if link:
            lines.append(f"    URL: {link}")
        if snippet:
            lines.append(f"    {snippet}")
        lines.append("")

    text = "\n".join(lines).strip()
    return text or None


def search(query: str, top_k: int = SERPER_TOP_K) -> Optional[str]:
    """Run a Serper search and return a formatted context string, or None.

    None signals "no usable context" — missing key, network failure,
    empty result set, or any parse error. Callers should treat None as
    "run the model without web context" rather than as an error.
    """
    if not query or not query.strip():
        return None

    api_key = os.environ.get(SERPER_API_KEY_ENV, "").strip()
    if not api_key:
        log.warning(
            "%s not set in environment — Research tab will run without web context",
            SERPER_API_KEY_ENV,
        )
        return None

    try:
        resp = requests.post(
            SERPER_ENDPOINT,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            data=json.dumps({"q": query.strip()}),
            timeout=SERPER_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        log.warning("Serper HTTP error (%s): %s", type(e).__name__, e)
        return None
    except ValueError as e:
        log.warning("Serper response was not JSON: %s", e)
        return None

    try:
        return _format_results(payload, top_k)
    except Exception as e:
        log.warning("Serper response parse failed (%s): %s", type(e).__name__, e)
        return None
