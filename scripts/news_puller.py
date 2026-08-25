"""news_puller.py — pull top headlines from NewsAPI (https://newsapi.org)
and cache them into docs/news/ so the main RAG pipeline picks them up on
the next ingest. Free tier is 100 requests/day; every pull_* function is
exactly 1 request.

Three category functions:
    pull_sports_news()     baseball-centric per config.NEWSAPI_SPORTS_QUERY
    pull_world_news()      international / general top headlines
    pull_national_news()   US top headlines
    pull_all_news()        all three  →  3 requests

Each writes a Markdown file to docs/news/YYYY-MM-DD_<category>.md and
updates data/news_state.json with the timestamp + article count for the
UI's status display. Set the API key via .env (NEWSAPI_KEY=…).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from config import (
    NEWSAPI_ENDPOINT,
    NEWSAPI_KEY_ENV,
    NEWSAPI_PAGE_SIZE,
    NEWSAPI_SPORTS_QUERY,
    NEWSAPI_TIMEOUT,
    NEWS_CACHE_DIR,
    NEWS_STATE_PATH,
)

log = logging.getLogger(__name__)


# ── Request layer ────────────────────────────────────────────────────────────


def _api_key() -> Optional[str]:
    return os.getenv(NEWSAPI_KEY_ENV) or None


def _request(path: str, params: dict) -> dict:
    """Thin wrapper around NewsAPI. Never raises; returns {'error': str}
    on any failure so callers can surface a friendly message."""
    key = _api_key()
    if not key:
        return {"error": f"{NEWSAPI_KEY_ENV} not set — add it to .env"}
    params = dict(params)
    params["apiKey"] = key
    url = NEWSAPI_ENDPOINT.rstrip("/") + "/" + path.lstrip("/")
    try:
        r = requests.get(url, params=params, timeout=NEWSAPI_TIMEOUT)
    except requests.RequestException as e:
        return {"error": f"HTTP error: {type(e).__name__}: {e}"}
    try:
        data = r.json()
    except ValueError:
        return {"error": f"Non-JSON response (HTTP {r.status_code})"}
    if r.status_code != 200 or data.get("status") != "ok":
        return {
            "error": f"NewsAPI: {data.get('message') or f'HTTP {r.status_code}'}"
        }
    return data


# ── Category pulls (1 credit each) ───────────────────────────────────────────


def pull_sports_news() -> dict:
    """Baseball-centric per NEWSAPI_SPORTS_QUERY. Uses /v2/everything so
    the query text actually filters (unlike /top-headlines which mixes
    all sports)."""
    result = _request("everything", {
        "q":        NEWSAPI_SPORTS_QUERY,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": NEWSAPI_PAGE_SIZE,
    })
    return _write_category("sports", result,
                             heading=f"Sports News — {NEWSAPI_SPORTS_QUERY}")


def pull_world_news() -> dict:
    """/v2/top-headlines with category=general — the mix of major
    outlets NewsAPI classifies as 'general' news."""
    result = _request("top-headlines", {
        "category": "general",
        "language": "en",
        "pageSize": NEWSAPI_PAGE_SIZE,
    })
    return _write_category("world", result, heading="World / General News")


def pull_national_news() -> dict:
    """/v2/top-headlines with country=us — US national headlines."""
    result = _request("top-headlines", {
        "country":  "us",
        "pageSize": NEWSAPI_PAGE_SIZE,
    })
    return _write_category("national", result, heading="US National News")


def pull_all_news() -> dict:
    """All three categories in sequence. 3 API credits total."""
    return {
        "sports":   pull_sports_news(),
        "world":    pull_world_news(),
        "national": pull_national_news(),
    }


# ── Cache writer ─────────────────────────────────────────────────────────────


def _write_category(category: str, result: dict, heading: str) -> dict:
    """Persist a NewsAPI response as a Markdown file under
    docs/news/YYYY-MM-DD_<category>.md and update the state file."""
    if "error" in result:
        return {"category": category, "articles": 0,
                "error": result["error"]}
    articles = result.get("articles") or []
    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    fname = f"{date}_{category}.md"
    path = NEWS_CACHE_DIR / fname

    lines = [
        f"# {heading}",
        f"_Pulled {datetime.now().isoformat(timespec='seconds')} · "
        f"{len(articles)} article(s)_",
        "",
    ]
    for a in articles:
        title = a.get("title") or "(untitled)"
        src   = (a.get("source") or {}).get("name") or "?"
        pub   = a.get("publishedAt") or ""
        desc  = (a.get("description") or "").strip()
        url   = a.get("url") or ""
        lines.append(f"## {title}")
        lines.append(f"_{src}  ·  {pub}_")
        if desc:
            lines.append(desc)
        if url:
            lines.append(f"[read more]({url})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    _bump_state(category, len(articles), fname)
    log.info("News pull '%s' → %d article(s) in %s", category, len(articles), fname)
    return {"category": category, "articles": len(articles),
            "path": str(path)}


# ── State (last-pull timestamps) ─────────────────────────────────────────────


def _load_state() -> dict:
    if not NEWS_STATE_PATH.exists():
        return {}
    try:
        return json.loads(NEWS_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _bump_state(category: str, article_count: int, filename: str) -> None:
    state = _load_state()
    state[category] = {
        "last_pull":     datetime.now().isoformat(timespec="seconds"),
        "articles":      article_count,
        "last_filename": filename,
    }
    NEWS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEWS_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def state_summary() -> str:
    """Markdown table of last-pull times per category. Rendered on the
    Data tab so the user can see freshness at a glance."""
    state = _load_state()
    if not state:
        return (
            "_No news pulls yet. Set `NEWSAPI_KEY` in `.env`, then click "
            "one of the pull buttons above._"
        )
    rows = ["| Category | Last pull | Articles |",
            "|----------|-----------|---------:|"]
    for cat in ("sports", "world", "national"):
        info = state.get(cat)
        if info:
            rows.append(
                f"| {cat.title()} | {info.get('last_pull', '?')} | "
                f"{info.get('articles', 0)} |"
            )
        else:
            rows.append(f"| {cat.title()} | _never_ | 0 |")
    return "\n".join(rows)


def credit_estimate() -> str:
    """Static markdown block explaining NewsAPI credit usage. Rendered
    on the Data tab beside the pull buttons so users know what they're
    burning against the 100/day free-tier cap."""
    return (
        "**Credit usage (NewsAPI free tier = 100 requests/day)**\n\n"
        "- Each category pull = **1 credit**\n"
        "- \"Pull All\" (sports + world + national) = **3 credits**\n"
        "- Typical daily budget (1 scheduled + a couple manual pulls) = "
        "**3-6 credits/day**\n"
        "- Free tier holds ~15-30 full-pull days easily."
    )
