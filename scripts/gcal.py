"""gcal.py — Google Calendar integration for Agent Ouija.

Three surfaces, in decreasing complexity:

  READ                 list_upcoming_events(n, days_ahead) → list[dict]
                       refresh_calendar_cache()             writes upcoming
                                                            events into
                                                            docs/calendar/…
                                                            so the main RAG
                                                            can retrieve
                                                            them per question.
  ADD  (form-based)    add_event(title, date_iso, start_hm, end_hm, desc)
                       Straight API call — no LLM in the loop.
  DELETE (form-based)  delete_event(event_id)
                       Same. The UI puts a "type YES" confirmation on top.

Auth is standard Google OAuth 2.0 (installed-app flow). See config.py
for the one-time setup steps. `python scripts\\gcal.py --auth` runs the
browser flow interactively and writes the refresh token to
data/gcal_token.json — after that, the app opens Calendar sessions
without further prompts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent))

from config import (
    GCAL_CACHE_DIR,
    GCAL_CREDENTIALS_FILE,
    GCAL_SCOPES,
    GCAL_TOKEN_FILE,
)

log = logging.getLogger(__name__)


# ── Auth ─────────────────────────────────────────────────────────────────────


def _get_service():
    """Return an authorized Calendar service, or None if not set up.

    Never raises AND never blocks — if the token is missing or a
    silent refresh fails, returns None so callers surface a
    friendly "click Authorize" message. The old code path called
    InstalledAppFlow.run_local_server() here, which opens a
    browser window on the machine running the code and BLOCKS
    until the user completes the OAuth consent. When the caller
    is on mobile (or a background Gradio worker), that popup can
    never be answered and the whole handler freezes. The
    explicit authorize_interactive() function below is now the
    only path that runs the browser flow — the user clicks a
    dedicated Authorize button to trigger it deliberately.

    Testing-mode note: Google enforces a 7-day refresh-token
    expiry on OAuth apps still in "Testing" status. Expect a
    weekly re-auth click until the app is published."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as e:
        log.warning(
            "Google Calendar libs missing (%s). "
            "Run: pip install google-api-python-client google-auth-oauthlib",
            e,
        )
        return None

    creds = None
    if GCAL_TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(GCAL_TOKEN_FILE), GCAL_SCOPES,
            )
        except Exception as e:
            log.warning(
                "Could not read gcal_token.json (%s) — click 🔐 Authorize "
                "Google Calendar in the Data tab to re-auth.", e,
            )
            creds = None

    if creds and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Persist refreshed credentials so the next
                # process launch doesn't have to refresh again.
                try:
                    GCAL_TOKEN_FILE.write_text(
                        creds.to_json(), encoding="utf-8",
                    )
                except OSError as e:
                    log.warning("Could not save refreshed token: %s", e)
            except Exception as e:
                log.warning(
                    "Silent token refresh failed (%s) — click 🔐 "
                    "Authorize Google Calendar in the Data tab.", e,
                )
                creds = None
        else:
            log.warning(
                "Token invalid and no refresh token — click 🔐 "
                "Authorize Google Calendar in the Data tab.",
            )
            creds = None

    if not creds:
        return None

    return build("calendar", "v3", credentials=creds,
                  cache_discovery=False)


def authorize_interactive() -> dict:
    """Run the OAuth browser flow interactively. THIS BLOCKS while
    the browser popup on the local machine (Brainy) waits for the
    user to consent — only call this from a deliberate user
    action (a button click) after making sure the user is
    actually at the Brainy console / RDP'd in.

    Returns a status dict:
        {"status": "success"|"error", "message": str}

    Never raises. On success, refreshes the token file at
    GCAL_TOKEN_FILE so subsequent _get_service() calls succeed
    without prompting."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        return {
            "status":  "error",
            "message": (
                f"Google auth libs missing ({e}). "
                "Run: pip install google-api-python-client "
                "google-auth-oauthlib"
            ),
        }

    if not GCAL_CREDENTIALS_FILE.exists():
        return {
            "status":  "error",
            "message": (
                f"No gcal_credentials.json at `{GCAL_CREDENTIALS_FILE}`. "
                "Download the OAuth client JSON (Desktop app) from the "
                "Google Cloud Console, save it there, then click "
                "Authorize again."
            ),
        }

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(GCAL_CREDENTIALS_FILE), GCAL_SCOPES,
        )
        # port=0 asks the OS to pick a free port for the loopback
        # callback. open_browser=True is the whole point — user
        # is at Brainy and will click through the popup.
        creds = flow.run_local_server(port=0, open_browser=True)
    except Exception as e:
        return {
            "status":  "error",
            "message": (
                f"OAuth flow failed: {type(e).__name__}: {e}. "
                "If the browser didn't open, run "
                "`python scripts\\gcal.py --auth` from a terminal "
                "on Brainy instead."
            ),
        }

    try:
        GCAL_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        GCAL_TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    except OSError as e:
        return {
            "status":  "error",
            "message": f"Could not save token to {GCAL_TOKEN_FILE}: {e}",
        }

    return {
        "status":  "success",
        "message": (
            "Authorized. Token cached at "
            f"`{GCAL_TOKEN_FILE.name}` — expect a re-auth in ~7 days "
            "while the OAuth app is in Google's Testing status."
        ),
    }


def is_authorized() -> bool:
    """Quick check without triggering the browser flow. Looks for
    the token file AND does a silent refresh probe so a stale
    (expired-refresh-token) file doesn't misreport as authorized."""
    if not GCAL_TOKEN_FILE.exists():
        return False
    # _get_service() no longer blocks — it only silent-refreshes
    # and returns None if that fails. Cheap way to probe.
    return _get_service() is not None


# ── Display helpers ──────────────────────────────────────────────────────────


def _parse_iso(s: str) -> Optional[datetime]:
    """Parse an ISO 8601 string (with or without timezone). Returns None
    on failure — datetime.fromisoformat handles Python 3.11+'s Z suffix
    for us."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_date(dt: datetime) -> str:
    # e.g. "Wed Jul 30" — stripping the leading zero from the day
    # manually since %-d is Unix-only and %#d is Windows-only.
    return f"{dt.strftime('%a %b')} {dt.day}"


def _fmt_time(dt: datetime) -> str:
    # e.g. "3:00 PM" — same reason we don't use %-I.
    hour_12 = dt.hour % 12 or 12
    return f"{hour_12}:{dt.strftime('%M %p')}"


def format_event_window(event: dict) -> str:
    """Render a Google Calendar event's start/end as a compact,
    human-friendly string.

    Timed event  →  "Wed Jul 30 · 3:00 PM → 4:00 PM"
    All-day 1d   →  "Wed Jul 30 (all day)"
    Multi-day    →  "Wed Jul 30 → Fri Aug 1"
    Cross-day    →  "Wed Jul 30 3:00 PM → Thu Jul 31 9:00 AM"
    """
    start_dt_iso = (event.get("start", {}) or {}).get("dateTime")
    end_dt_iso   = (event.get("end", {})   or {}).get("dateTime")
    start_d_iso  = (event.get("start", {}) or {}).get("date")
    end_d_iso    = (event.get("end", {})   or {}).get("date")

    # All-day event (Google returns 'date' not 'dateTime')
    if start_d_iso:
        s = _parse_iso(start_d_iso)
        e = _parse_iso(end_d_iso) if end_d_iso else None
        if not s:
            return start_d_iso or "?"
        # Google's `end.date` is exclusive — a 1-day all-day event has
        # end = start+1 day. Show as single-day; multi-day when > 1 day.
        if e and (e - s).days > 1:
            return f"{_fmt_date(s)} → {_fmt_date(e - timedelta(days=1))}"
        return f"{_fmt_date(s)} (all day)"

    s = _parse_iso(start_dt_iso)
    e = _parse_iso(end_dt_iso) if end_dt_iso else None
    if not s:
        return start_dt_iso or "?"

    if e and (s.date() != e.date()):
        # spans midnight — show date on both sides
        return (f"{_fmt_date(s)} {_fmt_time(s)} → "
                f"{_fmt_date(e)} {_fmt_time(e)}")

    if e:
        return f"{_fmt_date(s)} · {_fmt_time(s)} → {_fmt_time(e)}"
    return f"{_fmt_date(s)} · {_fmt_time(s)}"


# ── Read ─────────────────────────────────────────────────────────────────────


def list_upcoming_events(count: int = 20, days_ahead: int = 14) -> dict:
    """Return the next `count` events from the primary calendar in the
    next `days_ahead` days. Wrapped in a dict so callers can distinguish
    'not set up' / 'API error' / 'no events' cleanly."""
    service = _get_service()
    if service is None:
        return {"error": "Google Calendar not authorized. "
                          "Run `python scripts/gcal.py --auth` from the terminal."}
    now   = datetime.now(timezone.utc).isoformat()
    later = (datetime.now(timezone.utc)
              + timedelta(days=days_ahead)).isoformat()
    try:
        resp = service.events().list(
            calendarId="primary",
            timeMin=now, timeMax=later,
            maxResults=count,
            singleEvents=True, orderBy="startTime",
        ).execute()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"events": resp.get("items", [])}


def refresh_calendar_cache(count: int = 40, days_ahead: int = 30) -> dict:
    """Pull upcoming events and dump them as a Markdown file under
    docs/calendar/YYYY-MM-DD_upcoming.md so the RAG pipeline retrieves
    them when the user asks a calendar-shaped question. Rebuilds the
    file each call — one file per day."""
    result = list_upcoming_events(count=count, days_ahead=days_ahead)
    if "error" in result:
        return result
    events = result.get("events", [])
    GCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    path = GCAL_CACHE_DIR / f"{date}_upcoming.md"
    lines = [
        f"# Upcoming Calendar Events — {date}",
        f"_Pulled {datetime.now().isoformat(timespec='seconds')} · "
        f"{len(events)} event(s) in the next {days_ahead} days_",
        "",
    ]
    if not events:
        lines.append("_No upcoming events._")
    for e in events:
        summary = e.get("summary") or "(no title)"
        loc     = e.get("location", "")
        desc    = e.get("description", "")
        lines.append(f"## {summary}")
        lines.append(f"_{format_event_window(e)}_")
        if loc:
            lines.append(f"Location: {loc}")
        if desc:
            lines.append(desc.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"events": len(events), "path": str(path)}


# ── Write (add + delete) ─────────────────────────────────────────────────────


def add_event(title: str,
              date_iso: str,
              start_hm: str,
              end_hm: str,
              description: str = "",
              location: str = "") -> dict:
    """Insert a new event on the primary calendar.

    date_iso     — 'YYYY-MM-DD'
    start_hm     — 'HH:MM' (24-hour, local time)
    end_hm       — 'HH:MM' (24-hour, local time)
    description  — optional free text
    location     — optional
    Timezone: uses the machine's local timezone (whatever Windows says).
    """
    service = _get_service()
    if service is None:
        return {"error": "Google Calendar not authorized. "
                          "Run `python scripts/gcal.py --auth` from the terminal."}
    if not title or not title.strip():
        return {"error": "Title is required."}
    try:
        start_dt = datetime.strptime(f"{date_iso} {start_hm}", "%Y-%m-%d %H:%M")
        end_dt   = datetime.strptime(f"{date_iso} {end_hm}",   "%Y-%m-%d %H:%M")
    except ValueError as e:
        return {"error": f"Bad date/time format ({e}). "
                          "Use YYYY-MM-DD for date, HH:MM for time."}
    if end_dt <= start_dt:
        return {"error": "End time must be after start time."}

    # Local timezone label — Google accepts IANA names OR just an offset.
    # We use dateTime without timezone and let Google interpret in the
    # calendar's default timezone. That matches user expectations
    # ("3pm today" = 3pm in my zone).
    local_tz_offset = datetime.now(timezone.utc).astimezone().strftime("%z")
    # Format Google expects: 2026-07-28T15:00:00-05:00
    def _fmt(dt):
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + (
            f"{local_tz_offset[:3]}:{local_tz_offset[3:]}"
            if local_tz_offset else ""
        )

    body = {
        "summary":     title.strip(),
        "description": description or "",
        "location":    location or "",
        "start":       {"dateTime": _fmt(start_dt)},
        "end":         {"dateTime": _fmt(end_dt)},
    }
    try:
        created = service.events().insert(
            calendarId="primary", body=body,
        ).execute()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {
        "id":      created.get("id"),
        "title":   created.get("summary"),
        "htmlLink": created.get("htmlLink"),
        "when":    f"{start_dt} → {end_dt}",
    }


def delete_event(event_id: str) -> dict:
    """Delete an event by id. The UI paints a 'type YES to confirm'
    guard around this so a stray click doesn't nuke something."""
    service = _get_service()
    if service is None:
        return {"error": "Google Calendar not authorized."}
    if not event_id or not event_id.strip():
        return {"error": "No event selected."}
    try:
        service.events().delete(
            calendarId="primary", eventId=event_id.strip(),
        ).execute()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"deleted": event_id.strip()}


# ── CLI (--auth trigger) ─────────────────────────────────────────────────────


def _main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    parser = argparse.ArgumentParser(description="Google Calendar helper")
    parser.add_argument("--auth", action="store_true",
                        help="Run the OAuth browser flow to authorize "
                              "Agent Ouija against your Google account. "
                              "Writes data/gcal_token.json on success.")
    parser.add_argument("--list", action="store_true",
                        help="Print upcoming events and exit (smoke test).")
    args = parser.parse_args()

    if args.auth:
        service = _get_service()
        if service:
            print("✓ Google Calendar authorized. Token saved to",
                   GCAL_TOKEN_FILE)
        else:
            print("✗ Auth failed. Check the logs above.")
            sys.exit(1)
        return

    if args.list:
        result = list_upcoming_events()
        if "error" in result:
            print("Error:", result["error"])
            sys.exit(1)
        events = result.get("events", [])
        print(f"{len(events)} upcoming event(s):")
        for e in events:
            when = (e.get("start", {}).get("dateTime")
                     or e.get("start", {}).get("date", "?"))
            print(f"  {when}  {e.get('summary', '(no title)')}")
        return

    parser.print_help()


if __name__ == "__main__":
    _main()
