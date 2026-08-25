"""
media_cost_estimator.py — Rough per-model cost estimates for images + videos
uploaded via the arena_ui.py upload button.

Purpose: tell the user, BEFORE they include a picture/video in a question,
roughly what each cloud model would charge to process it. Cost is a rough
input-side estimate — actual token accounting varies with the model's
completion length and (for OpenAI) the "detail" mode the caller picks.
Locals are marked as unsupported (Ollama vision-capable variants exist but
are not the models in the default lineup).

Ratings map max-across-cloud-models cost to an intuition bucket:
    Low       < $0.01
    Medium    $0.01 – $0.10
    High      $0.10 – $1.00
    Very High > $1.00

APIs:
    classify_file(path) → 'document' / 'image' / 'video' / 'unknown'
    estimate_file(path) → dict describing per-model cost + overall rating
    format_report(analyses) → markdown-formatted status string
"""

import logging
from pathlib import Path
from typing import Optional

from config import AVAILABLE_MODELS

log = logging.getLogger(__name__)

# ── File classification ───────────────────────────────────────────────────────
IMAGE_EXTS    = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS    = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
DOCUMENT_EXTS = {".md", ".txt", ".pdf", ".docx", ".doc",
                 ".html", ".htm", ".json", ".csv", ".rtf"}


def classify_file(path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:    return "image"
    if ext in VIDEO_EXTS:    return "video"
    if ext in DOCUMENT_EXTS: return "document"
    return "unknown"


# ── Image dimensions ─────────────────────────────────────────────────────────
def _image_dimensions(path) -> tuple[int, int]:
    """Try PIL for exact dimensions; fall back to a size-based guess.
    PIL usually ships transitively with the ML deps; if not, the fallback
    keeps the estimator working without adding a new hard dependency."""
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        # File-size heuristic — better than nothing when PIL is absent.
        try:
            size = Path(path).stat().st_size
        except OSError:
            return 1024, 1024
        if size < 200_000:    return 640, 480
        if size < 1_000_000:  return 1280, 720
        if size < 4_000_000:  return 1920, 1080
        return 3840, 2160


# ── Token estimation per provider ────────────────────────────────────────────
# Rough formulas based on published rate cards / SDK docs. Not exact —
# the estimator is for planning, not billing.
def _image_tokens(provider: str, width: int, height: int) -> Optional[int]:
    p = (provider or "").lower()
    if p == "anthropic":
        # Claude: (w × h) / 750 per Anthropic's rule of thumb
        return max(300, (width * height) // 750)
    if p == "openai":
        # GPT-4o high-detail: 85 base + 170 per 512×512 tile
        import math
        tiles = max(1, math.ceil(width / 512)) * max(1, math.ceil(height / 512))
        return 85 + 170 * tiles
    if p == "google":
        # Gemini: ~258 tokens for images up to 384×384; increases with tiles.
        # Rough: 258 base + 258 per 384x384 tile beyond first.
        import math
        tiles = max(1, math.ceil(width / 384)) * max(1, math.ceil(height / 384))
        return 258 * tiles
    return None


def _video_estimated_image_count(path) -> int:
    """Very rough: assume ~3 frames sampled per MB of typical H.264/H.265."""
    try:
        size_mb = Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return 30
    return max(10, int(size_mb * 3))


# ── Per-file estimate ────────────────────────────────────────────────────────
def _rating_from_cost(cost_usd: float) -> str:
    if cost_usd < 0.01:  return "Low"
    if cost_usd < 0.10:  return "Medium"
    if cost_usd < 1.00:  return "High"
    return "Very High"


def _per_model_costs(image_or_video_kind: str,
                     width: int, height: int, frame_count: int = 1) -> dict:
    """Return {model_key: {tokens, cost, supported, note}} for every model
    in AVAILABLE_MODELS. Locals + non-vision providers get supported=False."""
    try:
        from litellm import cost_per_token
    except Exception:
        cost_per_token = None

    results: dict = {}
    for key, cfg in AVAILABLE_MODELS.items():
        display = cfg.get("display_name", key)
        if cfg.get("type") != "cloud":
            results[key] = {
                "display":   display,
                "supported": False,
                "tokens":    0,
                "cost":      0.0,
                "note":      "local — vision requires a multimodal Ollama variant",
            }
            continue

        provider = cfg.get("provider", "")
        per_frame_tokens = _image_tokens(provider, width, height)
        if per_frame_tokens is None:
            results[key] = {
                "display":   display,
                "supported": False,
                "tokens":    0,
                "cost":      0.0,
                "note":      f"vision unsupported by {provider or 'this provider'}",
            }
            continue

        tokens = per_frame_tokens * max(1, frame_count)
        if cost_per_token is None:
            cost = 0.0
            note = "litellm not installed — cost unknown"
        else:
            try:
                input_cost, _ = cost_per_token(
                    model=cfg["name"], prompt_tokens=tokens, completion_tokens=0,
                )
                cost = float(input_cost)
                note = ""
            except Exception as e:
                cost = 0.0
                note = f"cost lookup failed: {type(e).__name__}"

        results[key] = {
            "display":   display,
            "supported": True,
            "tokens":    tokens,
            "cost":      cost,
            "note":      note,
        }
    return results


def estimate_file(path) -> dict:
    """Return a per-file cost/rating summary or None for non-media files."""
    kind = classify_file(path)
    p = Path(path)
    if kind not in ("image", "video"):
        return {"kind": kind, "path": str(p), "name": p.name}

    if kind == "image":
        w, h = _image_dimensions(p)
        per_model = _per_model_costs("image", w, h, frame_count=1)
        max_cost  = max((r["cost"] for r in per_model.values() if r["supported"]),
                        default=0.0)
        return {
            "kind":       "image",
            "path":       str(p),
            "name":       p.name,
            "dimensions": f"{w}×{h}",
            "size_kb":    round(p.stat().st_size / 1024, 1),
            "per_model":  per_model,
            "max_cost":   max_cost,
            "rating":     _rating_from_cost(max_cost),
        }

    # video — treat as N sampled frames at 1280×720 default
    frames  = _video_estimated_image_count(p)
    per_model = _per_model_costs("video", 1280, 720, frame_count=frames)
    max_cost  = max((r["cost"] for r in per_model.values() if r["supported"]),
                    default=0.0)
    return {
        "kind":         "video",
        "path":         str(p),
        "name":         p.name,
        "size_mb":      round(p.stat().st_size / (1024 * 1024), 1),
        "est_frames":   frames,
        "per_model":    per_model,
        "max_cost":     max_cost,
        "rating":       _rating_from_cost(max_cost),
        "assumption":   "assumes ~3 frames sampled per MB at 1280×720",
    }


# ── Markdown report ──────────────────────────────────────────────────────────
_RATING_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Very High": "🔴"}


def format_report(analyses: list[dict]) -> str:
    """Format a list of per-file analyses as a Markdown status block.
    Non-media entries are skipped."""
    media = [a for a in analyses if a.get("kind") in ("image", "video")]
    if not media:
        return ""

    lines = ["\n### 🎨 Media cost preview (before any model calls)\n"]
    for a in media:
        icon = _RATING_ICON.get(a["rating"], "•")
        header = f"**{icon} {a['name']}**  ·  {a['rating']}"
        if a["kind"] == "image":
            header += f"  ·  {a['dimensions']}  ·  {a['size_kb']} KB"
        else:
            header += (f"  ·  video {a['size_mb']} MB  ·  "
                       f"~{a['est_frames']} frames est.")
        lines.append(header)

        # Only show cloud models — locals are always $0 and add noise.
        cloud_rows = [
            (r["display"], r["tokens"], r["cost"], r["note"])
            for r in a["per_model"].values() if r["supported"]
        ]
        if cloud_rows:
            lines.append("")
            lines.append("| Model | Est. tokens | Est. cost |")
            lines.append("|-------|-------------|-----------|")
            for display, tokens, cost, _ in cloud_rows:
                lines.append(f"| {display} | {tokens:,} | ${cost:.4f} |")
        unsupported = [
            r["display"] for r in a["per_model"].values() if not r["supported"]
        ]
        if unsupported:
            lines.append(f"\n_Not supported for this file: {', '.join(unsupported)}_")
        if a.get("assumption"):
            lines.append(f"\n_{a['assumption']}_")
        lines.append("")

    lines.append(
        "_These are input-side rough estimates only. Actual cost varies "
        "with prompt length, completion length, and (OpenAI) the "
        "`detail=low/high` mode you choose when attaching the image._"
    )
    return "\n".join(lines)
