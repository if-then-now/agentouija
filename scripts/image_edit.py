"""image_edit.py — Instruction-based image editing via Google's
Gemini 2.5 Flash Image (a.k.a. "Nano Banana") model.

User writes something like:
    "add a heart between the two people, dog.png"

The Ask For Revelations handler detects the image-file reference
+ edit intent, calls edit_image() with the referenced image and
the full instruction, and drops the result into docs/media/ as
`<stem>_edited_<timestamp>.<ext>` so the Files tab picks it up
on the next refresh.

Cloud-only (Gemini API, reuses GEMINI_API_KEY from .env). Local
Stable Diffusion is a separate follow-up.

Graceful degradation: if google-genai isn't installed or the API
key is missing, edit_image() returns None + logs a warning so the
UI can show a friendly error instead of crashing.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import GEMINI_IMAGE_MODELS, MEDIA_DIR

log = logging.getLogger(__name__)


# GEMINI_IMAGE_MODELS is a fallback chain of model IDs to try in
# order. Google has been renaming this checkpoint every few months
# (gemini-2.0-flash-exp-image-generation →
# gemini-2.5-flash-image-preview → gemini-2.5-flash-image →
# whatever's current), and different accounts see different
# availability. First name that succeeds wins and is cached for
# subsequent calls this process.
_LAST_WORKING_MODEL: Optional[str] = None

# Extensions we're willing to edit. Others fall through to the
# normal vision-attach path (describe / analyze) instead.
IMAGE_EDIT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def is_available() -> bool:
    """Cheap probe the UI can call to decide whether to route to
    image-edit or fall through to normal LLM. Checks both the SDK
    and the API key without making a network call."""
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


def edit_image(input_path: Path, instruction: str) -> Optional[Path]:
    """Send the input image + instruction to Gemini, save the
    returned image next to the source in docs/media/, return the
    new path. Returns None (never raises) on:
      - google-genai not installed
      - GEMINI_API_KEY missing
      - API refuses / errors
      - Response contains no image parts

    Caller shows a friendly UI message on None."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.warning(
            "GEMINI_API_KEY not set — image edit skipped. Add it to .env.",
        )
        return None
    if not input_path.exists():
        log.warning("Image edit source does not exist: %s", input_path)
        return None

    try:
        from google import genai
    except ImportError:
        log.warning(
            "google-genai not installed — image edit skipped. "
            "Run: pip install google-genai",
        )
        return None

    try:
        from PIL import Image
    except ImportError:
        log.warning(
            "Pillow not installed — needed to decode the source image "
            "for Gemini. Run: pip install Pillow",
        )
        return None

    global _LAST_WORKING_MODEL
    try:
        client = genai.Client(api_key=api_key)
        pil_img = Image.open(input_path)
    except Exception as e:
        log.warning(
            "Gemini client / Image open failed (%s): %s",
            type(e).__name__, e,
        )
        return None

    # Try model IDs in order. Cache the first that succeeds so
    # subsequent calls in the same process skip the fallback dance.
    candidates = (
        [_LAST_WORKING_MODEL] if _LAST_WORKING_MODEL else []
    ) + [m for m in GEMINI_IMAGE_MODELS if m != _LAST_WORKING_MODEL]
    resp = None
    last_error: Optional[Exception] = None
    for model_id in candidates:
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=[instruction, pil_img],
            )
            _LAST_WORKING_MODEL = model_id
            if model_id != candidates[0]:
                log.info(
                    "Image edit: using %s (previous candidates 404'd)",
                    model_id,
                )
            break
        except Exception as e:
            last_error = e
            log.warning(
                "Gemini image edit failed on %s (%s): %s",
                model_id, type(e).__name__, e,
            )
            # If the error isn't a "not found," don't keep trying
            # other models — same error will just repeat.
            err_txt = str(e).lower()
            if "not found" not in err_txt and "not_found" not in err_txt:
                break

    if resp is None:
        log.warning(
            "All Gemini image-edit model candidates failed. "
            "Run `python scripts/image_edit.py --list` to see what "
            "your key can actually reach. Last error: %s", last_error,
        )
        return None

    # The response returns a Candidate whose Content has zero or more
    # Parts — some text, some inline_data (image). We want the first
    # image part; text parts are model commentary we drop for now.
    try:
        parts = (
            resp.candidates[0].content.parts
            if resp.candidates else []
        )
    except (IndexError, AttributeError) as e:
        log.warning("Gemini response missing candidates/parts: %s", e)
        return None

    image_bytes: Optional[bytes] = None
    mime: str = "image/png"
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        if data:
            image_bytes = data
            mime = getattr(inline, "mime_type", mime) or mime
            break

    if not image_bytes:
        log.warning(
            "Gemini image edit returned no image bytes — likely refused "
            "or produced text-only. Instruction: %r", instruction[:120],
        )
        return None

    ext = _mime_to_ext(mime, input_path.suffix)
    stem = input_path.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = MEDIA_DIR / f"{stem}_edited_{ts}{ext}"

    try:
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
    except OSError as e:
        log.warning("Could not save edited image to %s: %s", output_path, e)
        return None

    log.info(
        "Image edit: %s → %s (%d bytes)",
        input_path.name, output_path.name, len(image_bytes),
    )
    return output_path


def _mime_to_ext(mime: str, fallback_ext: str) -> str:
    """Map Gemini's response mime type to a file extension. Falls
    back to the input's extension if the mime is unrecognized so
    the user's filename convention (`.jpg`) is preserved when the
    model returns image/jpeg."""
    m = {
        "image/png":  ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }
    ext = m.get((mime or "").lower())
    if ext:
        return ext
    return fallback_ext.lower() if fallback_ext else ".png"


def list_reachable_models() -> list[str]:
    """Return the full list of Gemini models the current
    GEMINI_API_KEY can reach, per Google's ListModels endpoint.
    Empty list on any failure. Useful for diagnosing 404s on the
    image-edit path when Google renames the checkpoint."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []
    try:
        from google import genai
    except ImportError:
        return []
    try:
        client = genai.Client(api_key=api_key)
        out = []
        for m in client.models.list():
            name = getattr(m, "name", None) or ""
            supported = getattr(m, "supported_actions", None) \
                or getattr(m, "supported_generation_methods", None) \
                or []
            out.append(f"{name}  [{','.join(supported)}]")
        return out
    except Exception as e:
        log.warning("list_reachable_models failed: %s: %s",
                    type(e).__name__, e)
        return []


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s — %(message)s")
    ap = argparse.ArgumentParser(
        description="Diagnostics for the Agent Ouija image editor.",
    )
    ap.add_argument("--list", action="store_true",
                     help="List Gemini models the key can reach.")
    args = ap.parse_args()
    if args.list:
        models = list_reachable_models()
        if not models:
            print("No models reachable. Check GEMINI_API_KEY in .env.")
        else:
            print(f"Reachable models ({len(models)}):")
            for m in models:
                print(f"  {m}")
    else:
        ap.print_help()
