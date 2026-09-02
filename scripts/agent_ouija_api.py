"""agent_ouija_api.py — HTTP sidecar exposing Agent Ouija's Boardroom
programmatically.

Runs alongside `arena_ui.py`:
    python scripts/arena_ui.py         # port 7860 (Gradio UI)
    python scripts/agent_ouija_api.py  # port 7861 (this API)

Exposed endpoints:
    GET  /health            → {"ok": true, "models_loaded": N}
    POST /api/executive     → run the Boardroom flow, return one synthesised answer

The API is deliberately narrow — one endpoint, one purpose. Meant for
downstream orchestrator apps (Master Ouija being the first) that want
Boardroom synthesis without the Gradio UI overhead. Localhost-only by
default; if you expose it beyond localhost, wire an auth token.

Bypasses Agent Ouija's own persona system (system_core.md +
enhanced_persona / episode files). Callers supply their OWN tone_prompt
so a downstream app like Master Ouija can inject a game-specific
narrative voice per call without persona layering interfering.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Make sibling scripts importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Reuse Agent Ouija's low-level model callers + registry.
# arena.py's helpers are terminal-friendly (no Gradio dependency).
from arena import _query_cloud, _query_local, _empty_result
from config import (
    ARENA_MODELS,
    AVAILABLE_MODELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("agent_ouija_api")


# ── Tier → model-key sets ─────────────────────────────────────────────────────
# Mirrors arena_ui.py's _exec_choices_for() semantics without dragging in the
# UI module. Kept as a single function so if Agent Ouija's tier definitions
# ever move (e.g. new tier), we change one place.

def _peer_and_exec_for(tier: str) -> tuple[list[str], str]:
    """Return (peer_model_keys, executive_model_key) for a boardroom tier.

    junior  → locals fan out, first local execs (fast, free, offline-capable)
    senior  → locals + clouds fan out, Claude Sonnet execs
    c_suite → web-search-capable clouds only fan out, Claude Sonnet execs
    """
    locals_   = [m for m in ARENA_MODELS
                  if AVAILABLE_MODELS.get(m, {}).get("type") == "local"]
    clouds_   = [m for m in ARENA_MODELS
                  if AVAILABLE_MODELS.get(m, {}).get("type") == "cloud"]
    web_caps  = [m for m in ARENA_MODELS
                  if AVAILABLE_MODELS.get(m, {}).get("web_search", False)]

    if tier == "junior":
        peers = locals_
        exec_key = locals_[0] if locals_ else ""
    elif tier == "senior":
        peers = locals_ + clouds_
        exec_key = "claude-sonnet" if "claude-sonnet" in AVAILABLE_MODELS else (
            clouds_[0] if clouds_ else (locals_[0] if locals_ else "")
        )
    elif tier == "c_suite":
        peers = web_caps
        exec_key = "claude-sonnet" if "claude-sonnet" in web_caps else (
            web_caps[0] if web_caps else ""
        )
    else:
        raise HTTPException(400, f"Unknown tier: {tier!r} (expected junior|senior|c_suite)")

    if not peers:
        raise HTTPException(500, f"No peer models available for tier {tier!r}")
    if not exec_key:
        raise HTTPException(500, f"No executive model available for tier {tier!r}")
    return peers, exec_key


# ── Message assembly ──────────────────────────────────────────────────────────
# Bypasses Agent Ouija's persona system entirely. The caller's tone_prompt
# becomes the whole system prompt for peer + exec runs.

def _peer_messages(question: str, tone_prompt: str) -> list[dict]:
    system = tone_prompt or (
        "You are one of several LLMs answering a question. Be helpful, "
        "accurate, and concise. Another model will synthesise all answers "
        "into a final response."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": question},
    ]


def _exec_messages(question: str, peer_answers: list[dict],
                    tone_prompt: str) -> list[dict]:
    """Compose the executive's synthesis prompt from peer answers.

    Peer answers are labelled by their display name so the exec can
    reference them internally without leaking model identifiers to the
    end user (its final response should read as one voice).
    """
    system = (
        (tone_prompt or "You are a synthesising executive.")
        + "\n\nYou will receive a question plus the answers of several "
        "other models. Produce a single unified response that reflects "
        "the best of what the peer models offered. Write in a single "
        "coherent voice — do not name or reference the peer models."
    )

    # Build the peer-answer block. Trim excessively long peer answers so
    # the exec's context budget doesn't blow up.
    peer_blocks = []
    for a in peer_answers:
        text = (a.get("text") or "").strip()
        if not text:
            continue
        peer_blocks.append(f"— {a['display_name']} —\n{text[:2000]}")
    peer_block = "\n\n".join(peer_blocks) or "(no peer answers available)"

    user = (
        f"QUESTION\n{question}\n\n"
        f"PEER ANSWERS\n{peer_block}\n\n"
        "SYNTHESIS\n"
        "Produce a single unified response — one voice, coherent tone. "
        "Draw on the peer answers as source material but do not name them."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ── Model call thin wrappers ──────────────────────────────────────────────────

def _run_one_model(model_key: str, messages: list[dict],
                     max_tokens: Optional[int] = None) -> dict:
    """Route to _query_local or _query_cloud based on model type.

    max_tokens (optional) is threaded through so callers can cap the
    output length — Master Ouija's Terse/Standard/Rich response-length
    presets rely on this. None = model default (~1024 for cloud,
    unbounded for local).
    """
    cfg = AVAILABLE_MODELS.get(model_key)
    if not cfg:
        return {**_empty_result(0.0), "display_name": model_key, "model_key": model_key}

    start = time.perf_counter()
    try:
        if cfg.get("type") == "local":
            result = _query_local(cfg, messages, max_tokens=max_tokens)
        else:
            result = _query_cloud(model_key, cfg, messages, max_tokens=max_tokens)
    except Exception as e:                              # noqa: BLE001
        log.warning("Model %s unavailable (%s): %s", model_key, type(e).__name__, e)
        result = _empty_result(round(time.perf_counter() - start, 3))
    else:
        result["elapsed"] = round(time.perf_counter() - start, 3)

    result["display_name"] = cfg.get("display_name", model_key)
    result["model_key"]    = model_key
    return result


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agent Ouija — Executive Boardroom API",
    version="1.0",
    description=(
        "HTTP interface into Agent Ouija's Boardroom flow. One endpoint: "
        "POST /api/executive. Peer models fan out, executive synthesises."
    ),
)


class ExecutiveRequest(BaseModel):
    question:       str = Field(..., description="The prompt to run through the boardroom.")
    tier:           str = Field("senior",
                                  description="junior | senior | c_suite")
    boardroom_mode: bool = Field(True,
                                  description="True = fan-out + synth. "
                                              "False = just call the exec model alone (faster, cheaper).")
    tone_prompt:    Optional[str] = Field(
        None,
        description="System-prompt supplement supplied by the caller. "
                    "Overrides Agent Ouija's own persona system for this call. "
                    "Use this to inject Master Ouija's game-specific narrative voice.",
    )
    max_tokens:     Optional[int] = Field(
        None,
        description="Advisory — passed to callers who honor it (locals via Ollama, "
                    "clouds via LiteLLM). Not a hard clamp on peer answers.",
    )


class ExecutiveMeta(BaseModel):
    models_used:    list[str]
    tier:           str
    time_s:         float
    cost_usd:       float
    tokens_in:      int
    tokens_out:     int
    peers_ran:      int
    peers_succeeded: int


class ExecutiveResponse(BaseModel):
    answer:  str
    meta:    ExecutiveMeta


@app.get("/health")
def health():
    return {
        "ok": True,
        "models_loaded": len(AVAILABLE_MODELS),
        "arena_models":  len(ARENA_MODELS),
    }


@app.post("/api/executive", response_model=ExecutiveResponse)
def executive(req: ExecutiveRequest):
    start = time.perf_counter()
    peers, exec_key = _peer_and_exec_for(req.tier)

    # Peer answers stay short regardless of the caller's target — they
    # feed into the exec's synthesis, not the user, and a tight cap
    # keeps the fan-out fast + the exec's context tidy. The exec
    # itself honours the caller's max_tokens verbatim.
    peer_cap = min(req.max_tokens, 400) if req.max_tokens else 400
    exec_cap = req.max_tokens                    # None = library default

    # ── Single-model shortcut ────────────────────────────────────────────────
    if not req.boardroom_mode:
        result = _run_one_model(
            exec_key,
            _peer_messages(req.question, req.tone_prompt or ""),
            max_tokens=exec_cap,
        )
        return _build_response(
            answer      = result.get("text") or "",
            tier        = req.tier,
            time_s      = round(time.perf_counter() - start, 3),
            results     = [result],
            exec_used   = [exec_key],
            peers_ran   = 0,
            peers_succ  = 0,
        )

    # ── Boardroom mode: fan out to peers, then run exec on the pile ─────────
    peer_msgs = _peer_messages(req.question, req.tone_prompt or "")
    peer_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(peers), 10)) as pool:
        futs = {
            pool.submit(_run_one_model, m, peer_msgs, peer_cap): m
            for m in peers
        }
        for fut in as_completed(futs):
            peer_results.append(fut.result())

    peers_succeeded = sum(1 for r in peer_results if (r.get("text") or "").strip())

    exec_msgs   = _exec_messages(req.question, peer_results, req.tone_prompt or "")
    exec_result = _run_one_model(exec_key, exec_msgs, max_tokens=exec_cap)

    return _build_response(
        answer      = exec_result.get("text") or "",
        tier        = req.tier,
        time_s      = round(time.perf_counter() - start, 3),
        results     = peer_results + [exec_result],
        exec_used   = [exec_key],
        peers_ran   = len(peers),
        peers_succ  = peers_succeeded,
    )


def _build_response(
    answer:     str,
    tier:       str,
    time_s:     float,
    results:    list[dict],
    exec_used:  list[str],
    peers_ran:  int,
    peers_succ: int,
) -> ExecutiveResponse:
    total_cost = sum(
        float(r.get("input_cost", 0.0) or 0.0) + float(r.get("output_cost", 0.0) or 0.0)
        for r in results
    )
    total_in   = sum(int(r.get("input_tokens", 0)  or 0) for r in results)
    total_out  = sum(int(r.get("output_tokens", 0) or 0) for r in results)
    models_used = [r.get("display_name") or r.get("model_key") or "?" for r in results
                    if (r.get("text") or "").strip()]

    return ExecutiveResponse(
        answer = answer or "*(no answer synthesised — all peers failed)*",
        meta   = ExecutiveMeta(
            models_used     = models_used,
            tier            = tier,
            time_s          = time_s,
            cost_usd        = round(total_cost, 6),
            tokens_in       = total_in,
            tokens_out      = total_out,
            peers_ran       = peers_ran,
            peers_succeeded = peers_succ,
        ),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Ouija Executive API sidecar")
    parser.add_argument("--host", default=os.getenv("AGENT_OUIJA_API_HOST", "127.0.0.1"),
                         help="Bind host. Defaults to 127.0.0.1 (localhost-only). "
                              "Set to 0.0.0.0 to expose beyond the local rig "
                              "(add auth first!).")
    parser.add_argument("--port", type=int,
                         default=int(os.getenv("AGENT_OUIJA_API_PORT", "7861")))
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    log.info("Agent Ouija API starting on http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
