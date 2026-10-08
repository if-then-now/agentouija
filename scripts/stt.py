"""
stt.py — Speech-to-text for Agent Ouija's microphone input.

The Agent Ouija tab's mic button records audio into a temp file via
Gradio's `gr.Audio(sources=["microphone"], type="filepath")`. This
module turns that file into text that gets appended to the question
textbox so the user can review (or edit) before clicking Ask.

Provider dispatch, driven by config.STT_PROVIDER:
    "openai"          → openai Whisper API (default; uses existing key)
    "faster-whisper"  → local CTranslate2 Whisper on the RTX 5080
    "off" / anything  → is_available() returns False; mic UI hidden

Mobile browsers (iOS Safari / Chrome) require HTTPS for
`navigator.mediaDevices.getUserMedia`. On Rahul's Tailscale rig this is
served by `tailscale serve` fronting the Gradio port with a trusted
cert — plumbing is already documented in SETUP.md §4b.

Design goals (same as tts.py):
    - Never raise into the UI; return None on any failure.
    - Lazy-load deps; cache load errors so repeat calls stay cheap.
    - Degrade gracefully — missing API key or missing dep hides the UI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from config import (
    STT_LOCAL_COMPUTE,
    STT_LOCAL_DEVICE,
    STT_LOCAL_MODEL,
    STT_MAX_SECONDS,
    STT_OPENAI_MODEL,
    STT_PROVIDER,
)

log = logging.getLogger(__name__)


# ── Capability probe ─────────────────────────────────────────────────────────
# Called once at UI-build time to decide whether to render the mic button.
# Keeps the Agent Ouija tab clean on installs that don't want voice input.


def is_available() -> bool:
    """Return True iff the configured provider can plausibly succeed.

    This is a cheap probe — we check the obvious preconditions (env var
    for openai, importability for faster-whisper) but don't actually
    load a model. If the probe says True and the real call later fails,
    transcribe() still returns None and the UI stays functional.
    """
    provider = (STT_PROVIDER or "").lower()
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if provider == "faster-whisper":
        try:
            import faster_whisper  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False
    return False


# ── OpenAI Whisper backend ───────────────────────────────────────────────────
_OPENAI_CLIENT     = None
_OPENAI_LOAD_ERR   = None


def _get_openai_client():
    """Lazy OpenAI client singleton. Mirrors tts._get_openai_client so the
    two don't share state (TTS failures shouldn't disable STT and vice
    versa — different endpoints, different failure modes)."""
    global _OPENAI_CLIENT, _OPENAI_LOAD_ERR
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if _OPENAI_LOAD_ERR is not None:
        raise _OPENAI_LOAD_ERR

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        _OPENAI_LOAD_ERR = RuntimeError(
            "OPENAI_API_KEY not set — mic input needs it. Add the key to "
            ".env, or set STT_PROVIDER=faster-whisper for a local alternative."
        )
        raise _OPENAI_LOAD_ERR

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        _OPENAI_LOAD_ERR = ImportError(
            f"openai sdk not installed: pip install openai ({e})"
        )
        raise _OPENAI_LOAD_ERR

    _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _transcribe_openai(audio_path: Path) -> Optional[str]:
    """Call OpenAI's audio.transcriptions.create. Returns the text or None."""
    try:
        client = _get_openai_client()
    except Exception as e:
        log.warning("STT (OpenAI) unavailable: %s", e)
        return None

    try:
        with open(audio_path, "rb") as fh:
            response = client.audio.transcriptions.create(
                model=STT_OPENAI_MODEL,
                file=fh,
            )
    except Exception as e:
        log.warning("STT (OpenAI) call failed (%s: %s)", type(e).__name__, e)
        return None

    text = getattr(response, "text", None)
    if not text:
        return None
    return text.strip()


# ── faster-whisper (local) backend ───────────────────────────────────────────
_LOCAL_MODEL      = None
_LOCAL_LOAD_ERR   = None


def _get_local_model():
    """Lazy-load the faster-whisper model. The first call hits the HF
    model download if it's not cached; subsequent calls are instant.

    Model stays resident until the process exits — that's fine because
    the installed-size footprint on small.en is <~500 MB and GPU VRAM
    hit while idle is modest."""
    global _LOCAL_MODEL, _LOCAL_LOAD_ERR
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL
    if _LOCAL_LOAD_ERR is not None:
        raise _LOCAL_LOAD_ERR

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        _LOCAL_LOAD_ERR = ImportError(
            f"faster-whisper not installed: pip install faster-whisper ({e})"
        )
        raise _LOCAL_LOAD_ERR

    try:
        _LOCAL_MODEL = WhisperModel(
            STT_LOCAL_MODEL,
            device=STT_LOCAL_DEVICE,
            compute_type=STT_LOCAL_COMPUTE,
        )
        log.info("Loaded faster-whisper model=%s device=%s compute=%s",
                 STT_LOCAL_MODEL, STT_LOCAL_DEVICE, STT_LOCAL_COMPUTE)
        return _LOCAL_MODEL
    except Exception as e:
        _LOCAL_LOAD_ERR = e
        log.warning("Failed to load faster-whisper (%s): %s",
                    type(e).__name__, e)
        raise


def _transcribe_local(audio_path: Path) -> Optional[str]:
    """Run faster-whisper on the recording. Returns concatenated segment
    text or None on failure."""
    try:
        model = _get_local_model()
    except Exception:
        return None

    try:
        segments, _info = model.transcribe(str(audio_path), beam_size=5)
        parts = [seg.text for seg in segments if seg.text]
    except Exception as e:
        log.warning("STT (faster-whisper) failed (%s: %s)",
                    type(e).__name__, e)
        return None

    text = " ".join(p.strip() for p in parts).strip()
    return text or None


# ── Length guard ─────────────────────────────────────────────────────────────
# Reject obviously-too-long recordings before paying for a transcription.
# Uses the WAV header where possible, falls back to file-size heuristic
# so Gradio's webm/ogg mic captures aren't rejected outright.


def _duration_seconds(audio_path: Path) -> Optional[float]:
    """Return recording length in seconds, or None if we can't tell."""
    try:
        import wave
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate   = wf.getframerate() or 1
            return frames / float(rate)
    except Exception:
        pass
    # Non-WAV: no cheap duration probe without pulling in ffmpeg. Fall
    # back to file-size heuristic (~16 KB/s @ 128 kbps mp3 / opus).
    try:
        size = audio_path.stat().st_size
        return size / 16000.0
    except OSError:
        return None


# ── Public entry point ───────────────────────────────────────────────────────


def transcribe(audio_filepath: Optional[str]) -> Optional[str]:
    """Transcribe a recording file to text. Returns text or None.

    Called from the Agent Ouija tab's mic-input handler. The caller
    (Gradio) hands us a filesystem path to a temp file containing the
    user's recording. We route through the configured backend and return
    cleaned text; None signals "nothing usable" and the handler leaves
    the question textbox unchanged.

    Never raises — every failure path is logged and surfaced as None.
    """
    if not audio_filepath:
        return None

    path = Path(audio_filepath)
    if not path.exists() or path.stat().st_size == 0:
        return None

    duration = _duration_seconds(path)
    if duration is not None and duration > STT_MAX_SECONDS:
        log.warning(
            "Recording longer than STT_MAX_SECONDS (%.1fs > %ds) — rejecting",
            duration, STT_MAX_SECONDS,
        )
        return None

    provider = (STT_PROVIDER or "").lower()
    if provider == "openai":
        return _transcribe_openai(path)
    if provider == "faster-whisper":
        return _transcribe_local(path)

    log.debug("STT_PROVIDER=%r — mic input not configured", STT_PROVIDER)
    return None
