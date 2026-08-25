"""emailer.py — Send a file as an email attachment.

Powers the "📧 Email Last Edit" buttons on the Agent Ouija and
Files tabs. Uses stdlib smtplib + email.message — no third-party
dep, works with any SMTP provider that speaks plain STARTTLS on
port 587 (Gmail with an App Password, Outlook, Fastmail, Proton
bridge, etc.).

Config comes entirely from .env:
    SMTP_HOST       server hostname
    SMTP_PORT       usually 587 (STARTTLS) or 465 (SSL)
    SMTP_USER       login username (usually your email address)
    SMTP_PASSWORD   app password / SMTP password
    EMAIL_FROM      From: header (defaults to SMTP_USER when blank)
    EMAIL_TO        default recipient — can be overridden at call time

Graceful degradation: is_available() returns False when any of the
required env vars are missing, so callers can gate the UI without
having to import smtplib themselves. send_file() never raises;
returns a {status, message} dict.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


REQUIRED_ENV = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_TO")


def is_available() -> bool:
    """True when every required SMTP env var is set. Callers use
    this to short-circuit into a friendly UI message instead of
    firing an SMTP call that would fail with an auth error."""
    return all(os.environ.get(k) for k in REQUIRED_ENV)


def _guess_mime(path: Path) -> tuple[str, str]:
    """(maintype, subtype) for the attachment. Falls back to
    'application/octet-stream' when the extension isn't in the
    mimetypes DB — recipient's mail client picks a default icon."""
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or "/" not in mime:
        return ("application", "octet-stream")
    maintype, _, subtype = mime.partition("/")
    return (maintype, subtype)


def send_file(
    path: str | Path,
    *,
    subject: Optional[str]  = None,
    body:    Optional[str]  = None,
    to:      Optional[str]  = None,
) -> dict:
    """Send a single file as an attachment. Thin wrapper around
    send_files() — kept for the callers that only ever email one
    file at a time (e.g. the Email Last Edit button)."""
    return send_files(
        [path] if path else [],
        subject=subject, body=body, to=to,
    )


def send_files(
    paths: list,
    *,
    subject: Optional[str]  = None,
    body:    Optional[str]  = None,
    to:      Optional[str]  = None,
) -> dict:
    """Send one or more files in a single email. Never raises;
    returns a status dict:
        {"status": "success"|"error", "message": str}

    to defaults to the EMAIL_TO env var. subject / body default to
    a reasonable auto-generated string when None. Files that don't
    exist are logged and skipped — if the result is zero
    attachments, the whole call fails."""
    if not paths:
        return {"status": "error", "message": "No file(s) to send."}
    resolved: list[Path] = []
    missing: list[str] = []
    for raw in paths:
        if not raw:
            continue
        p = Path(raw)
        if p.exists() and p.is_file():
            resolved.append(p)
        else:
            missing.append(str(p))
    if not resolved:
        return {
            "status": "error",
            "message": (
                f"None of the file(s) were found: "
                f"{', '.join(missing) if missing else '(empty list)'}"
            ),
        }
    if not is_available():
        missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
        return {
            "status": "error",
            "message": (
                "SMTP not configured — missing env vars: "
                + ", ".join(missing)
                + ". Add them to `.env` (see .env.example for a "
                "Gmail App Password walkthrough)."
            ),
        }

    host      = os.environ["SMTP_HOST"]
    port      = int(os.environ.get("SMTP_PORT", "587"))
    user      = os.environ["SMTP_USER"]
    password  = os.environ["SMTP_PASSWORD"]
    from_addr = os.environ.get("EMAIL_FROM") or user
    to_addr   = to or os.environ["EMAIL_TO"]

    names = [p.name for p in resolved]
    if subject is None:
        if len(resolved) == 1:
            subject = f"Agent Ouija — {names[0]}"
        else:
            subject = f"Agent Ouija — {len(resolved)} files"
    if body is None:
        body = (
            f"Attached: {', '.join(names)}\n\n"
            "Sent from Agent Ouija via the 📧 Send File button. If "
            "you didn't request this, someone with access to your "
            "rig did."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg.set_content(body)

    total_bytes = 0
    read_errors: list[str] = []
    for p in resolved:
        try:
            with p.open("rb") as fh:
                data = fh.read()
        except OSError as e:
            read_errors.append(f"{p.name}: {e}")
            continue
        maintype, subtype = _guess_mime(p)
        msg.add_attachment(data, maintype=maintype, subtype=subtype,
                            filename=p.name)
        total_bytes += len(data)

    if not msg.iter_attachments() or total_bytes == 0:
        return {
            "status": "error",
            "message": (
                "Could not read any of the requested file(s): "
                + "; ".join(read_errors)
            ),
        }

    # Two connection paths: port 465 → implicit TLS, everything
    # else → STARTTLS upgrade. Most providers document 587.
    try:
        ctx = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ctx,
                                   timeout=30) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                s.ehlo()
                s.starttls(context=ctx)
                s.ehlo()
                s.login(user, password)
                s.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        return {
            "status": "error",
            "message": (
                f"SMTP auth rejected: {e.smtp_code} "
                f"{e.smtp_error.decode('utf-8', 'replace') if isinstance(e.smtp_error, bytes) else e.smtp_error}. "
                "For Gmail, use an App Password, not your account "
                "password. Generate one at "
                "https://myaccount.google.com/apppasswords ."
            ),
        }
    except (smtplib.SMTPException, OSError) as e:
        return {
            "status": "error",
            "message": (
                f"SMTP send failed ({type(e).__name__}): {e}"
            ),
        }

    log.info(
        "Email sent: %s → %s via %s:%s (%d file(s), %d bytes total)",
        from_addr, to_addr, host, port, len(resolved), total_bytes,
    )
    tail = ""
    if missing:
        tail = f" (skipped missing: {', '.join(missing)})"
    if read_errors:
        tail += f" (read errors: {'; '.join(read_errors)})"
    listing = ", ".join(f"`{n}`" for n in names)
    return {
        "status":  "success",
        "message": (
            f"📧 Emailed {len(resolved)} file(s) to `{to_addr}`: "
            f"{listing}.{tail}"
        ),
    }
