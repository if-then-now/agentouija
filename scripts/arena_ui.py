"""
arena_ui.py — Gradio web UI for Agent Ouija

Runs at http://localhost:7860
    python scripts/arena_ui.py

Features:
    - Model checkboxes — select/deselect any model for the session
    - Unified personas (defined in PERSONA_NAMES) load as one prompt;
      no role selection — the LLM picks its register from question context
    - Blind Mode — hides model label only; order is ALWAYS randomized
    - Slideshow scoring — one response at a time, score then advance
    - 0-4 baseball scoring: 0=whiff, 1=contact, 2=base hit, 3=extra bases, 4=home run
    - Response time displayed per model
    - The Reading — composite metric: (score/SCORE_MAX) / log(time + e)
    - Stats tab — leaderboard: avg score, avg time, avg Reading, delta vs checkpoint
    - Ouija board icon in header (assets/ouija.png — 512x512 recommended)
    - ITN logo at footer (assets/itn_logo.png)
"""

import base64
import logging
import os
import glob
import json
import threading
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import gradio as gr
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

# .env location resolves to ~/.config/agentouija/.env when it
# exists (hardened default), else the legacy repo-root file. See
# config.env_file_path() for the ordering. Import config early
# via the same sys.path shim install.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (   # noqa: E402
    env_file_path as _env_file_path,
    is_encrypted_env as _is_encrypted_env,
)


def _load_env_or_prompt():
    """Load env vars from either plaintext .env or an encrypted
    .env.enc pair. Encrypted path prompts on stdin (hidden via
    getpass) up to 3 times, or reads AGENT_OUIJA_ENV_PASSPHRASE
    when the app is launched headless (cron / service). Falls
    back to plain load_dotenv when nothing is encrypted."""
    env_path = _env_file_path()
    if not _is_encrypted_env():
        load_dotenv(env_path)
        return

    import getpass
    from env_crypto import (
        load_into_environ as _env_load_str,
        read_encrypted as _env_decrypt,
    )
    try:
        from cryptography.fernet import InvalidToken
    except ImportError:
        raise SystemExit(
            "❌ Agent Ouija: cryptography package missing but "
            "an encrypted .env is present. Run "
            "`pip install cryptography` and relaunch."
        )

    # Non-interactive escape hatch — user pre-loads the passphrase
    # into an env var for scheduled tasks or CI-adjacent runs.
    env_pass = os.environ.get("AGENT_OUIJA_ENV_PASSPHRASE")
    if env_pass:
        pt = _env_decrypt(env_path, env_pass)
        _env_load_str(pt)
        return

    for attempt in range(1, 4):
        try:
            pw = getpass.getpass(
                f"🔐 Agent Ouija — .env passphrase "
                f"(attempt {attempt}/3, or set "
                "AGENT_OUIJA_ENV_PASSPHRASE and relaunch): ",
            )
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(
                "\n❌ Agent Ouija: no passphrase provided. Aborting."
            )
        try:
            pt = _env_decrypt(env_path, pw)
        except InvalidToken:
            print("   Wrong passphrase.")
            continue
        _env_load_str(pt)
        return
    raise SystemExit(
        "❌ Agent Ouija: 3 wrong passphrases. Aborting for safety."
    )


_load_env_or_prompt()

from config import (
    ACTIVE_VOICE_PATH,
    ARENA_MODELS,
    AVAILABLE_MODELS,
    BASE_DIR,
    BILLING_LINKS,
    CHAT_HISTORY_PATH,
    CONVERSATION_HISTORY_WINDOW,
    CUSTOM_ACTIVE_FLAG,
    CUSTOM_PROMPTS_DIR,
    DATA_DIR,
    DEFAULT_PERSONA_MODE,
    DOCS_DIR,
    ENHANCED_ACTIVE_FLAG,
    ITN_LOGO_PATH,
    MEDIA_DIR,
    MEDIA_MAX_MODELS,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    OUIJA_ICON_PATH,
    PERSONA_REINFORCEMENT,
    PRIVATE_CORPUS_ACTIVE_FLAG,
    PRIVATE_CORPUS_EPISODES_DIR,
    REFUSAL_LABEL,
    REFUSAL_SCORE,
    SCORE_LABELS,
    SEANCES_DIR,
    TTS_VOICE,
    TTS_VOICES_DIR,
    compute_reading,
    compute_reading_with_cost,
    label_for_score,
)
from db import (
    backup_state,
    connection_scope,
    count_arena_rows_before,
    list_backups,
    log_rag_contribution,
    purge_arena_rows_before,
    restore_backup,
    wipe_arena_db,
)
from ingest import run_ingestion
from media_cost_estimator import classify_file, estimate_file, format_report
from notebook_runner import make_globals, run_cells_for_models
from persona_lean import detect_persona_lean
import prefs
from prompt_loader import get_prompt_loader
from rag import (
    add_to_collection as rag_add,
    count_chroma_before,
    get_private_context,
    get_rag_context,
    purge_chroma_before,
    wipe_chroma_collection,
)
from serper_search import search as serper_search
from stt import is_available as _stt_available, transcribe as _stt_transcribe
from tts import set_active_voice, synth_to_wav
from summarize import generate_summary, generate_trainer_report
from tagger import load_keyword_map, tag_by_keywords
from training_manager import add_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)


# ── HTTP session (shared, thread-safe) ────────────────────────────────────────
# Reused across the Gradio worker thread pool to avoid per-call TCP handshakes
# when fanning out to multiple local models.
_HTTP = requests.Session()
_HTTP.mount("http://",  HTTPAdapter(pool_connections=8, pool_maxsize=16))
_HTTP.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=16))

# ── Cancellation flag ─────────────────────────────────────────────────────────
# Set by the ✋ Cancel button. run_models / run_research_models /
# run_big3_models poll this between as_completed iterations and abandon any
# still-pending futures. In-flight HTTP calls to LiteLLM can't be aborted
# from Python (LiteLLM is synchronous under the hood), so already-started
# model queries continue — but their results are discarded once the flag
# is set. Cleared at the top of every fresh Ask so a prior cancel doesn't
# poison the next call.
_ao_cancel_flag = threading.Event()


# ── System prompts (per persona mode + model type) ────────────────────────────
# Locals always get the base stack (system_core + enhanced_persona_main only
# if uploaded). Clouds get the enhanced stack when either their persona field
# is "enhanced" OR the global ENHANCED_ACTIVE_FLAG is set. Not cached — the
# flag file can flip mid-session from the Control Center / Persona tab, and
# file reads are cheap.


def get_system_prompt_for(model_key: str) -> str:
    cfg = AVAILABLE_MODELS.get(model_key, {})
    mode = cfg.get("persona", DEFAULT_PERSONA_MODE)
    is_local = cfg.get("type") == "local"
    # Per-model opt-out of the enhanced persona stack — for cloud
    # models like Groq whose per-minute token cap can't fit the
    # ~4k tokens of persona overlay + regular context. Config
    # field: "skip_enhanced_persona": True.
    force_base = bool(cfg.get("skip_enhanced_persona", False))
    return get_prompt_loader(
        mode, is_local=is_local, force_base=force_base,
    ).load()


# ── Asset loading ─────────────────────────────────────────────────────────────
# Both icons are PNG. Missing assets degrade gracefully — UI always launches.

def load_img_b64(path: Path, mime: str = "image/png") -> str:
    """Return base64 data URI or empty string if file missing. Never raises."""
    try:
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            return f"data:{mime};base64,{data}"
    except Exception as e:
        log.warning("Could not load asset %s: %s", path, e)
    log.warning("Asset not found — using fallback: %s", path)
    return ""

# Ouija icon — expects assets/ouija.png (512x512 recommended)
OUIJA_B64    = load_img_b64(OUIJA_ICON_PATH)
# ITN logo   — expects assets/itn_logo.png
ITN_LOGO_B64 = load_img_b64(ITN_LOGO_PATH)

OUIJA_HTML = (
    f'<img src="{OUIJA_B64}" style="width:32px;height:32px;object-fit:contain" alt="Agent Ouija" />'
    if OUIJA_B64 else '<span style="font-size:24px" title="Agent Ouija">🔮</span>'
)
ITN_LOGO_HTML = (
    f'<img src="{ITN_LOGO_B64}" style="height:64px;opacity:0.7" alt="ITN" />'
    if ITN_LOGO_B64 else '<span style="font-size:28px;color:#999;letter-spacing:0.05em">ITN</span>'
)


# ── DB ────────────────────────────────────────────────────────────────────────
# Schema, indices, and PRAGMA tuning are owned by db.py. Connections are
# short-lived and use the connection_scope() context manager so they always
# close even on exception.

def log_notebook_score(code, model_key, model_display, response, score,
                       elapsed, round_id=None):
    """Persist a Notebook-tab scoring event into arena_results.

    Differs from log_score (the Arena path) in three ways:
      - skips keyword tagging (running tagger over Python code produces
        weird matches — e.g. baseball tag on every snippet). Tags hard-
        coded to "notebook" so these rows can be filtered out of training
        exports or panels later if desired.
      - input/output tokens + cost = 0. We don't intercept the litellm
        call inside the user's notebook code, so we can't observe usage.
      - tag_source = "notebook" mirrors the same convention.
    Everything else mirrors log_score so the leaderboard / persona /
    trainer report can include or exclude these rows by SQL filter.
    """
    reading      = compute_reading(score, elapsed)
    cost_reading = compute_reading_with_cost(score, elapsed, 0.0)
    model_type   = AVAILABLE_MODELS.get(model_key, {}).get("type", "unknown")
    persona_lean = detect_persona_lean(response)
    try:
        with connection_scope() as conn:
            conn.execute(
                """INSERT INTO arena_results
                   (timestamp, question, model_key, model_display, model_type, response,
                    score, response_time_seconds, reading, tags, tag_source,
                    persona_lean, input_tokens, output_tokens, input_cost, output_cost,
                    round_id, cost_reading)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), code, model_key, model_display,
                 model_type, response, score, elapsed, reading,
                 "notebook", "notebook",
                 persona_lean, 0, 0, 0.0, 0.0,
                 round_id, cost_reading),
            )
            conn.commit()
    except sqlite3.DatabaseError:
        log.exception("Failed to log notebook score for model %s", model_key)


def log_score(question, model_key, model_display, response, score,
              response_time_seconds, keyword_map,
              input_tokens=0, output_tokens=0,
              input_cost=0.0, output_cost=0.0,
              round_id=None,
              tag_source="keyword"):
    """Persist a scored arena/research row.

    tag_source defaults to "keyword" (regular Arena scoring). The Research
    tab passes "research" so the Stats tab can group those rows separately
    in the future. Notebook scoring uses log_notebook_score which forces
    tag_source="notebook" — same column, three orthogonal values.
    """
    # Research-tab rows keep the per-row "tags" column meaningful by
    # still running the keyword tagger on the question text.
    tag          = tag_by_keywords(question, keyword_map)
    reading      = compute_reading(score, response_time_seconds)
    cost_reading = compute_reading_with_cost(
        score, response_time_seconds, (input_cost or 0.0) + (output_cost or 0.0)
    )
    model_type   = AVAILABLE_MODELS.get(model_key, {}).get("type", "unknown")
    persona_lean = detect_persona_lean(response)
    try:
        with connection_scope() as conn:
            conn.execute(
                """INSERT INTO arena_results
                   (timestamp, question, model_key, model_display, model_type, response,
                    score, response_time_seconds, reading, tags, tag_source,
                    persona_lean, input_tokens, output_tokens, input_cost, output_cost,
                    round_id, cost_reading)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), question, model_key, model_display,
                 model_type, response, score, response_time_seconds, reading,
                 tag, tag_source,
                 persona_lean, input_tokens, output_tokens, input_cost, output_cost,
                 round_id, cost_reading),
            )
            conn.commit()
    except sqlite3.DatabaseError:
        log.exception("Failed to log score for model %s", model_key)


# ── Model querying ────────────────────────────────────────────────────────────
# Helpers return a dict carrying text + token counts + USD cost so log_score
# can persist all of it. Locals have cost=0.0 by design (Ollama is unbilled).


def _query_local(model_cfg: dict, messages: list) -> dict:
    resp = _HTTP.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":    model_cfg["name"],
            "messages": messages,
            "stream":   False,
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    try:
        text = payload["message"]["content"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Unexpected Ollama response shape: {payload!r}") from e
    return {
        "text":          text,
        "input_tokens":  int(payload.get("prompt_eval_count") or 0),
        "output_tokens": int(payload.get("eval_count") or 0),
        "input_cost":    0.0,
        "output_cost":   0.0,
    }


def _web_search_kwargs(provider: str) -> dict:
    """Provider-specific kwargs to enable the LiteLLM web-search path.

    Empty dict if the provider isn't recognized — caller treats that as
    "this provider can't web-search" and the model returns nothing useful.
    """
    p = (provider or "").lower()
    if p == "anthropic":
        # Claude's native web search tool. Max 5 lookups per call keeps
        # latency and per-call cost bounded.
        return {"tools": [{
            "type":     "web_search_20250305",
            "name":     "web_search",
            "max_uses": 5,
        }]}
    if p == "openai":
        return {"web_search_options": {"search_context_size": "medium"}}
    if p == "google":
        return {"tools": [{"googleSearch": {}}]}
    return {}


def _parse_rate_limit_wait(msg: str) -> Optional[float]:
    """Extract 'try again in Xs' from a rate-limit error message.
    Groq, OpenAI, Anthropic all emit variants of this hint. Also
    matches minutes ("in 1m30s") — normalises to seconds. Returns
    None if no hint can be extracted."""
    import re
    # Groq: "Please try again in 17.08s"
    # OpenAI: "Please try again in 12ms" (throttle probes — treat as small)
    # Anthropic: "retry after 30 seconds"
    patterns = [
        r"try again in\s+([\d.]+)\s*s\b",
        r"retry after\s+([\d.]+)\s*second",
        r"in\s+([\d.]+)\s*s\b",
    ]
    for pat in patterns:
        m = re.search(pat, msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _is_rate_limit(exc: Exception) -> bool:
    """LiteLLM's RateLimitError has moved between litellm.exceptions
    and litellm top-level across versions; also some providers wrap
    it. Check by isinstance where possible, then fall back to class
    name — the class name has been stable at RateLimitError across
    every litellm version we've shipped against."""
    try:
        from litellm.exceptions import RateLimitError as _RL1
        if isinstance(exc, _RL1):
            return True
    except Exception:
        pass
    try:
        from litellm import RateLimitError as _RL2
        if isinstance(exc, _RL2):
            return True
    except Exception:
        pass
    return type(exc).__name__ == "RateLimitError"


# Max we're willing to block a Gradio worker thread waiting on a
# rate-limit retry. Groq's TPM hints are usually 5-30 s; anything
# longer we surface as a failure so the user can pick a different
# executive or retry manually.
_RATE_LIMIT_MAX_WAIT_S    = 30.0
# Default wait when the provider says we're rate-limited but
# doesn't include a "try again in Xs" hint. Half of Groq's TPM
# window is a reasonable-blind guess.
_RATE_LIMIT_DEFAULT_WAIT  = 15.0
# How many retries to attempt after the first failure. Two hops
# past the first attempt keeps worst-case wait under 60 s.
_RATE_LIMIT_MAX_RETRIES   = 2


def _query_cloud(model_key: str, model_cfg: dict, messages: list,
                 web_search: bool = False) -> dict:
    # Lazy import — litellm is heavy and unused while cloud models are disabled.
    from litellm import completion as litellm_completion
    from litellm import cost_per_token

    env_key = model_cfg.get("env_key", "")
    if not env_key:
        raise RuntimeError(f"Cloud model {model_key} is missing 'env_key' in config")
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} is not set in environment — add it to .env")

    extra: dict = {}
    model_name = model_cfg["name"]
    if web_search and model_cfg.get("web_search", False):
        provider = (model_cfg.get("provider") or "").lower()
        extra = _web_search_kwargs(provider)
        # OpenAI's web_search_options is only supported by the dedicated
        # search-preview model variants — vanilla gpt-4o errors with
        # "Web search options not supported with this model". Swap the
        # model name only for the Research-tab path; the same model_cfg
        # is used unchanged by Arena and Notebook (no web_search).
        if provider == "openai":
            model_name = "openai/gpt-4o-search-preview"

    # Pre-flight token count. When the model has a
    # `max_input_tokens` set in config (Groq's free-tier TPM cap is
    # the main case), estimate the assembled prompt size via
    # litellm.token_counter BEFORE calling the API. Oversize
    # calls guaranteed to 429 with "Request too large" — much
    # cleaner to skip client-side than eat the round-trip + retry
    # marathon. Estimation failures fall through to the API call
    # (fail-open) so a broken token counter doesn't kill peers
    # that would have worked.
    max_in = model_cfg.get("max_input_tokens")
    if max_in:
        try:
            from litellm import token_counter as _token_counter
            approx = _token_counter(model=model_name, messages=messages)
        except Exception:
            # Fallback: chars/4 heuristic across every message's
            # text content. Ignores image bytes in multimodal
            # messages but those aren't the token-hog on Groq
            # anyway (Groq is text-only right now).
            approx = 0
            for m in messages:
                c = m.get("content", "")
                if isinstance(c, str):
                    approx += len(c) // 4
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict):
                            t = part.get("text", "")
                            approx += len(t) // 4
        if approx > max_in:
            raise RuntimeError(
                f"Prompt too large for {model_key} (~{approx:,} tokens > "
                f"{max_in:,} config cap). Skipped client-side to avoid a "
                f"guaranteed provider 429. Biggest wins on this app: "
                f"turn OFF Enhanced Persona and Private Corpus on the "
                f"Persona tab (persona stack + corpus chunks are the "
                f"main token hog on peer calls). Also works: fewer "
                f"referenced docs, shorter conversation history, or "
                f"raise `max_input_tokens` in config.py after upgrading "
                f"to a higher tier."
            )

    def _do_completion():
        return litellm_completion(
            model=model_name,
            messages=messages,
            api_key=api_key,
            max_tokens=2048 if web_search else 1024,
            **extra,
        )

    # Rate-limit retry loop. LiteLLM raises RateLimitError when the
    # provider says slow-down; we parse the hinted wait, sleep, and
    # retry up to _RATE_LIMIT_MAX_RETRIES times. Groq's free TPM
    # tier is the main source of these; hitting an exec-level rate
    # limit used to silently fail the whole boardroom synthesis.
    #
    # Robustness: _is_rate_limit checks the exception class across
    # litellm's shifting import paths, then falls back to name check.
    # If the provider doesn't include a "try again in Xs" hint we
    # default to _RATE_LIMIT_DEFAULT_WAIT seconds — long enough for
    # a TPM window to have moved, short enough to feel responsive.
    #
    # BUT: "Request too large" (single payload > TPM cap) is NOT a
    # transient throttle — retrying the same payload guarantees the
    # same error. Detect it and bail immediately without burning
    # 45s across three doomed attempts.
    resp = None
    last_rl_error: Optional[Exception] = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            resp = _do_completion()
            break
        except Exception as e:
            if not _is_rate_limit(e):
                raise
            last_rl_error = e
            msg = str(e).lower()
            oversize = (
                "request too large" in msg
                or "reduce your message size" in msg
                or "requested " in msg and "limit " in msg
                and (
                    "tpm" in msg
                    or "tokens per minute" in msg
                )
            )
            if oversize:
                log.warning(
                    "Model %s: request exceeds per-minute token cap in a "
                    "single call — not retrying (retry would guarantee "
                    "the same error). Cut the peer set / question / "
                    "history, or pick a different exec: %s",
                    model_key, e,
                )
                raise
            if attempt == _RATE_LIMIT_MAX_RETRIES:
                log.warning(
                    "Model %s rate-limited on final attempt (%d/%d) — "
                    "giving up: %s",
                    model_key, attempt + 1, _RATE_LIMIT_MAX_RETRIES + 1, e,
                )
                raise
            hinted = _parse_rate_limit_wait(str(e))
            pause = hinted if hinted is not None else _RATE_LIMIT_DEFAULT_WAIT
            if pause > _RATE_LIMIT_MAX_WAIT_S:
                log.warning(
                    "Model %s rate-limit wait %.1fs > cap %.0fs — giving up",
                    model_key, pause, _RATE_LIMIT_MAX_WAIT_S,
                )
                raise
            pause += 0.5   # small buffer past the provider's hint
            log.warning(
                "Model %s rate-limited (attempt %d/%d, hinted=%s) — "
                "sleeping %.1fs then retrying",
                model_key, attempt + 1, _RATE_LIMIT_MAX_RETRIES + 1,
                hinted, pause,
            )
            time.sleep(pause)
    if resp is None:
        # Defensive — every branch above either sets resp or raises.
        raise last_rl_error or RuntimeError("rate-limit retry loop exited without a response")

    usage = getattr(resp, "usage", None) or {}
    prompt_tokens     = int(getattr(usage, "prompt_tokens",     usage.get("prompt_tokens", 0))     or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", usage.get("completion_tokens", 0)) or 0)

    try:
        input_cost, output_cost = cost_per_token(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as e:
        # LiteLLM's cost registry doesn't cover every model — OpenRouter
        # OSS routes in particular (hermes-3-llama-3.1-70b etc.) throw
        # here because there's no per-token price in the local table.
        # That's expected + harmless (we record 0.0 and move on), so
        # downgrade the log level for known coverage-gap providers to
        # keep the console readable. Genuinely unexpected coverage
        # gaps (Anthropic / OpenAI / Gemini / Groq) still warn.
        provider_lc = (model_cfg.get("provider") or "").lower()
        if provider_lc in ("openrouter", "hermes3-cloud"):
            log.info("cost_per_token unavailable for %s (%s) — "
                      "recording 0.0 (expected for OpenRouter routes)",
                      model_name, type(e).__name__)
        else:
            log.warning("cost_per_token failed for %s (%s) — recording 0.0",
                        model_name, type(e).__name__)
        input_cost = output_cost = 0.0

    return {
        "text":          resp.choices[0].message.content,
        "input_tokens":  prompt_tokens,
        "output_tokens": completion_tokens,
        "input_cost":    float(input_cost),
        "output_cost":   float(output_cost),
    }


def _empty_payload(elapsed: float, error: str | None = None) -> dict:
    return {
        "text":          None,
        "elapsed":       elapsed,
        "input_tokens":  0,
        "output_tokens": 0,
        "input_cost":    0.0,
        "output_cost":   0.0,
        "error":         error,
    }


def _brief_error(exc: Exception, max_len: int = 220) -> str:
    """Surface the useful bit of a LiteLLM / requests / Ollama exception in
    a form short enough to fit in a status line. Recognises the two failure
    patterns we hit constantly with a long enhanced-persona stack:
        * provider context-length rejects → "context: <requested>/<limit> tok"
        * provider TPM rate limits         → "rate limit: <requested>/<limit> TPM"
    Falls back to `<ClassName>: <first N chars of str(exc), whitespace collapsed>`.
    """
    import re
    msg  = " ".join(str(exc).split())
    kind = type(exc).__name__
    m = re.search(r"maximum context length is (\d+).*?requested about (\d+)", msg)
    if m:
        limit, req = int(m.group(1)), int(m.group(2))
        return f"context: {req//1000}K / {limit//1000}K tokens"
    m = re.search(r"Please reduce the length", msg)
    if m:
        return "context length exceeded (provider default limit)"
    m = re.search(r"tokens per min \(TPM\): Limit (\d+), Requested (\d+)", msg)
    if m:
        limit, req = int(m.group(1)), int(m.group(2))
        return f"rate limit: {req//1000}K / {limit//1000}K TPM"
    if len(msg) > max_len:
        msg = msg[:max_len].rstrip() + "…"
    return f"{kind}: {msg}"


def _collect_referenced_docs(question: str, char_budget: int = 40_000) -> str:
    """Look for document filenames from docs/ mentioned in the question
    and load their raw text as an extra system-message block. This is
    the "specific-file reference" path — bypasses RAG's semantic-search
    when the user names a file directly.

    Media (docs/media/) is handled separately via `_collect_media_paths`.
    """
    if not question or not DOCS_DIR.exists():
        return ""
    q_lower = question.lower()
    hits: list[tuple[str, str]] = []
    budget = char_budget
    for p in DOCS_DIR.iterdir():
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name.lower() not in q_lower:
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.warning("Could not read referenced doc %s: %s", p, e)
            continue
        if len(content) > budget:
            content = content[:budget] + "\n\n[...truncated...]"
        budget -= len(content)
        hits.append((p.name, content))
        if budget <= 0:
            break
    if not hits:
        return ""
    parts = []
    for name, content in hits:
        parts.append(f"===== FILE: {name} =====\n{content}\n===== END: {name} =====")
    return "\n\n".join(parts)


def _open_media_folder_in_explorer() -> str:
    """Open docs/media/ in the host OS's native file manager.
    Runs on the SERVER (Brainy), so useful when the user is at
    Brainy directly or RDP'd in — pointless from a phone. Cross-
    platform best-effort; never raises."""
    import platform
    import subprocess
    try:
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        system = platform.system()
        if system == "Windows":
            os.startfile(str(MEDIA_DIR))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(MEDIA_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(MEDIA_DIR)])
        return f"📁 Opened `{MEDIA_DIR}` in the host file manager."
    except Exception as e:
        return (f"❌ Couldn't open folder: {type(e).__name__}: {e}. "
                f"Path is `{MEDIA_DIR}`.")


def _collect_referenced_files(question: str) -> list[Path]:
    """Return every on-disk file in docs/ or docs/media/ whose
    filename appears in the question. Used by the 📧 Send File
    button to decide what to attach. Both directories are
    searched; media files sort after documents so a mixed
    reference list stays predictable. De-duped by path."""
    if not question:
        return []
    q_lower = question.lower()
    seen: set[Path] = set()
    out: list[Path] = []
    for base in (DOCS_DIR, MEDIA_DIR):
        if not base.exists():
            continue
        for p in sorted(base.iterdir()):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.name.lower() in q_lower and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _collect_media_paths(question: str) -> list[Path]:
    """Return the docs/media/ files whose names appear in the question,
    ordered as they appear on disk. Used to attach images/videos to the
    LLM call for vision-capable models."""
    if not question or not MEDIA_DIR.exists():
        return []
    q_lower = question.lower()
    return [
        p for p in sorted(MEDIA_DIR.iterdir())
        if p.is_file() and p.name.lower() in q_lower
    ]


def _media_to_content_part(path: Path) -> Optional[dict]:
    """Build a LiteLLM-compatible content part for one media file.
    Returns a dict with type='image_url' and a data-URI so it works
    without needing a hosted URL. None on read failure."""
    import base64
    ext = path.suffix.lower()
    mime = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
        ".bmp":  "image/bmp",
        # videos: only some providers (Gemini) accept these; LiteLLM will
        # either forward or 400 depending on model.
        ".mp4":  "video/mp4",
        ".mov":  "video/quicktime",
        ".webm": "video/webm",
    }.get(ext)
    if not mime:
        return None
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as e:
        log.warning("Could not read media %s: %s", path, e)
        return None
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}"},
    }


def query_model(model_key: str, question: str,
                rag_context: str | None = None,
                private_context: str | None = None,
                web_search: bool = False,
                web_context: str | None = None,
                history: list[dict] | None = None,
                grounded: bool = False) -> tuple[str, dict]:
    """Return (model_key, payload_dict).

    System prompt is selected per-model from AVAILABLE_MODELS[...]["persona"].
    rag_context is injected only for models with "rag": True.
    private_context is injected only for models with "private_corpus": True
    AND the master USE_PRIVATE_CORPUS toggle is on.
    web_search enables the provider's native web-search tool for cloud
    models with "web_search": True; ignored otherwise. Locals can't web-
    search — they return through the normal Ollama path either way.
    web_context is a Serper-fetched search-result block injected into
    EVERY selected model (locals + clouds) — used by the Research tab so
    any model can answer with web-grounded facts. Unconditional injection
    by design — that's the whole point of the Research tab.

    File attachments (documents + media):
    - If the question mentions a filename from `docs/`, that file's raw
      text is prepended as an extra system message ("Referenced files")
      — direct-reference path that bypasses RAG's semantic match.
    - If the question mentions a filename from `docs/media/`, that media
      is base64-encoded and attached to the user message as an image_url
      content part (cloud models only; locals get a text-only prompt with
      a note that the media couldn't be attached).
    """
    model_cfg = AVAILABLE_MODELS.get(model_key)
    if model_cfg is None:
        log.warning("Unknown model_key %s — skipping", model_key)
        return model_key, _empty_payload(-1.0)

    system_prompt = get_system_prompt_for(model_key)
    messages = [{"role": "system", "content": system_prompt}]

    # Current date/time — inject on every call so models can interpret
    # "today", "yesterday", "this week", "tomorrow morning" etc. against
    # the actual moment they're being asked, not their training cutoff.
    # ~30 tokens; negligible cost, big utility win for calendar +
    # freshness questions.
    _now_local = datetime.now().astimezone()
    messages.append({
        "role": "system",
        "content": (
            f"Current date and time: "
            f"{_now_local.strftime('%A, %B %d, %Y at %I:%M %p %Z')} "
            f"(ISO: {_now_local.isoformat(timespec='seconds')}). "
            "Use this when the user says 'today', 'tomorrow', "
            "'yesterday', 'this week', 'next Monday' etc."
        ),
    })

    # Direct-file reference — load exact contents when the question names
    # a document. Applies to every model regardless of rag/private flags.
    doc_context = _collect_referenced_docs(question)
    if doc_context:
        messages.append({
            "role": "system",
            "content": (
                "The user referenced these files by name. Contents follow — "
                "answer directly from them if the question is about them:\n\n"
                f"{doc_context}"
            ),
        })
    if rag_context and model_cfg.get("rag", False):
        messages.append({"role": "system",
                         "content": f"Knowledge base context:\n\n{rag_context}"})
    if private_context and model_cfg.get("private_corpus", False):
        messages.append({"role": "system",
                         "content": f"Persona corpus context:\n\n{private_context}"})
    if web_context:
        messages.append({"role": "system",
                         "content": (
                             "Real-time web search results (cite URLs in your "
                             "answer when relevant; flag if results disagree):"
                             f"\n\n{web_context}"
                         )})

    # Grounded-to-RAG mode — Ask the Lore. Strict system directive:
    # answer only from the context blocks above (knowledge base +
    # private corpus + web + referenced docs); if the answer isn't
    # there, refuse rather than fall back on training data.
    # Locals are less reliable at honouring closed-book instructions
    # than clouds, but the persona reinforcement + this directive
    # are the strongest signal we can give them from here.
    if grounded:
        messages.append({"role": "system", "content": (
            "STRICT GROUNDING: Answer ONLY from the context blocks "
            "supplied above (knowledge base / private corpus / web "
            "search / referenced files). Do NOT use any information "
            "from your training data. If the supplied context does "
            "not contain the answer, reply exactly: "
            "**\"Not in the lore.\"** and briefly say what related "
            "material you did see. Do not guess. Do not hedge with "
            "outside knowledge."
        )})

    # Conversation history — prior (question, answer) turns injected as
    # alternating user/assistant messages so follow-ups like "we're
    # under 25" resolve against what "25" meant a turn ago. Callers
    # pre-slice to CONVERSATION_HISTORY_WINDOW turns; anything older is
    # RAG-tagged elsewhere.
    if history:
        for turn in history:
            q_prev = (turn.get("q") or "").strip()
            a_prev = (turn.get("a") or "").strip()
            if not q_prev or not a_prev:
                continue
            messages.append({"role": "user",      "content": q_prev})
            messages.append({"role": "assistant", "content": a_prev})

    user_text = (
        f"{PERSONA_REINFORCEMENT}\n\n{question}"
        if PERSONA_REINFORCEMENT
        else question
    )

    # Media attachment path. Cloud vision-capable models get a
    # multimodal content list; locals get a text-only prompt with a
    # heads-up so their response doesn't just say "I don't see any file".
    media_paths = _collect_media_paths(question)
    is_cloud = model_cfg.get("type") == "cloud"
    if media_paths and is_cloud:
        content_parts: list[dict] = [{"type": "text", "text": user_text}]
        attached_names: list[str] = []
        for p in media_paths:
            part = _media_to_content_part(p)
            if part is not None:
                content_parts.append(part)
                attached_names.append(p.name)
        if len(content_parts) > 1:
            log.info(
                "Attaching %d media file(s) to %s: %s",
                len(attached_names), model_key, ", ".join(attached_names),
            )
            messages.append({"role": "user", "content": content_parts})
        else:
            # None of the referenced media had a supported MIME type —
            # fall back to text-only.
            messages.append({"role": "user", "content": user_text})
    elif media_paths and not is_cloud:
        # Locals can't do vision on the current lineup. Give the model a
        # hint so its answer explains WHY it can't see the file rather
        # than pretending to.
        note = (
            "\n\n[Note: this local model cannot read image/video files. "
            f"Referenced media on disk: {', '.join(p.name for p in media_paths)}. "
            "Use a cloud model (Claude / GPT-4o / Gemini) to actually see these.]"
        )
        messages.append({"role": "user", "content": user_text + note})
    else:
        messages.append({"role": "user", "content": user_text})

    start = time.perf_counter()
    try:
        kind = model_cfg.get("type")
        if kind == "local":
            payload = _query_local(model_cfg, messages)
        elif kind == "cloud":
            payload = _query_cloud(model_key, model_cfg, messages, web_search=web_search)
        else:
            log.warning("Model %s has unknown type %r — skipping", model_key, kind)
            return model_key, _empty_payload(-1.0)

        elapsed = round(time.perf_counter() - start, 3)
        payload["elapsed"] = elapsed
        return model_key, payload

    except requests.Timeout:
        elapsed = round(time.perf_counter() - start, 3)
        log.warning("Model %s timed out after %.1fs", model_key, OLLAMA_TIMEOUT)
        return model_key, _empty_payload(
            elapsed, error=f"timeout after {OLLAMA_TIMEOUT:.0f}s",
        )
    except requests.RequestException as e:
        elapsed = round(time.perf_counter() - start, 3)
        log.warning("Model %s HTTP error (%s): %s", model_key, type(e).__name__, e)
        return model_key, _empty_payload(elapsed, error=_brief_error(e))
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 3)
        log.warning("Model %s unavailable (%s): %s", model_key, type(e).__name__, e)
        return model_key, _empty_payload(elapsed, error=_brief_error(e))


def run_models(question: str, selected_models: list, progress=None,
                history: list[dict] | None = None,
                grounded: bool = False) -> list:
    """Query all selected models concurrently. Always shuffles results.

    Each returned entry is a dict with the full payload + display name so
    the Gradio state can carry token/cost data forward into log_score().
    Every entry shares the same round_id so the trainer report can group
    peers by round rather than by minute-of-log timestamp.

    progress: optional gr.Progress() instance. Updated as each future
    completes — shows which model JUST finished, the running count, and
    which models are still in flight. Detailed style matches the Notebook
    tab's progress bar.
    """
    results = []
    if not selected_models:
        return results

    round_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{random.randint(0, 0xFFFF):04x}"

    needs_rag = any(
        AVAILABLE_MODELS.get(m, {}).get("rag", False) for m in selected_models
    )
    rag_context = get_rag_context(question) if needs_rag else None
    if needs_rag:
        if rag_context:
            log.info("RAG context retrieved (%d chars)", len(rag_context))
        else:
            log.info("RAG retrieval returned nothing — cloud models will run without context")

    # Private corpus context — fetched ONCE per round and shared across the
    # fan-out, same pattern as the main RAG context. get_private_context
    # already gates on USE_PRIVATE_CORPUS, so the check here only matters
    # for whether any selected model has opted in via its per-model flag.
    needs_private = any(
        AVAILABLE_MODELS.get(m, {}).get("private_corpus", False) for m in selected_models
    )
    private_context = get_private_context(question) if needs_private else None
    if needs_private:
        if private_context:
            log.info("Private corpus context retrieved (%d chars)", len(private_context))
        else:
            log.info(
                "Private corpus returned nothing — opted-in models will run "
                "without private context (toggle off or collection empty)"
            )

    total = len(selected_models)
    pending: set = set(selected_models)
    done = 0
    if progress is not None:
        names = ", ".join(
            AVAILABLE_MODELS.get(k, {}).get("display_name", k) for k in selected_models
        )
        progress(0.0, desc=f"Querying {total} models concurrently · {names}")

    with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
        futures = {
            executor.submit(query_model, m, question, rag_context,
                            private_context, False, None, history,
                            grounded): m
            for m in selected_models
        }
        for future in as_completed(futures):
            if _ao_cancel_flag.is_set():
                # Cancel pending futures; already-running ones can't be
                # aborted mid-request but their results won't be added.
                for f in futures:
                    f.cancel()
                log.info("run_models: cancelled — %d results collected", len(results))
                break
            model_key, payload = future.result()
            display = AVAILABLE_MODELS.get(model_key, {}).get("display_name", model_key)
            results.append({
                "model_key":     model_key,
                "model_display": display,
                "response":      payload["text"],
                "elapsed":       payload["elapsed"],
                "input_tokens":  payload["input_tokens"],
                "output_tokens": payload["output_tokens"],
                "input_cost":    payload["input_cost"],
                "output_cost":   payload["output_cost"],
                "round_id":      round_id,
                "error":         payload.get("error"),
            })
            pending.discard(model_key)
            done += 1
            if progress is not None:
                if pending:
                    pending_names = ", ".join(sorted(
                        AVAILABLE_MODELS.get(k, {}).get("display_name", k) for k in pending
                    ))
                    desc = (
                        f"✓ {display} done ({done}/{total}) · "
                        f"waiting on: {pending_names}"
                    )
                else:
                    desc = f"✓ all {total} models done"
                progress(done / total, desc=desc)

    # Always randomize — blind mode controls label visibility only
    random.shuffle(results)
    return results


def _format_boardroom_prompt(question: str, peer_results: list) -> str:
    """Build the synthesis prompt for the executive. Only responses where
    the peer actually returned text are included — unavailable models
    contribute nothing to the executive's context."""
    peer_blocks = []
    for r in peer_results:
        text = (r.get("response") or "").strip()
        if not text:
            continue
        peer_blocks.append(f"[{r['model_display']}]\n{text}")
    peer_section = "\n\n".join(peer_blocks) if peer_blocks else "(no peer responses available)"
    return (
        "You are acting as the Executive summarizer for a boardroom of AI "
        "models. Below is the user's original question and each model's "
        "response. Your job is to synthesize these into ONE coherent "
        "answer for the user. Note where the models agreed, flag where "
        "they diverged with a brief mention, and produce the best "
        "combined response — do not just list what each said.\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"BOARDROOM RESPONSES:\n{peer_section}\n\n"
        "Provide your synthesized executive answer:"
    )


def _run_executive(question: str,
                   peer_results: list,
                   executive_key: str,
                   web_context: str | None = None,
                   web_search: bool = False,
                   progress=None,
                   history: list[dict] | None = None,
                   grounded: bool = False) -> dict:
    """Query the executive with the peer boardroom responses as context.

    Callers run the peers themselves (via run_models / run_research_models
    / run_big3_models) so this function only owns the synthesis step —
    that keeps the caller in control of Serper / native-search wiring.

    Returns:
        {
            "executive_key":         str,
            "executive_display":     str,
            "executive_response":    str | None,
            "executive_elapsed":     float,
            "executive_input_cost":  float,
            "executive_output_cost": float,
            "peers_answered":        int,
            "peers_total":           int,
        }
    """
    cfg = AVAILABLE_MODELS.get(executive_key)
    if cfg is None:
        return {
            "executive_key":         executive_key,
            "executive_display":     executive_key,
            "executive_response":    None,
            "executive_elapsed":     0.0,
            "executive_input_cost":  0.0,
            "executive_output_cost": 0.0,
            "peers_answered":        0,
            "peers_total":           len(peer_results),
        }

    display = cfg.get("display_name", executive_key)
    if progress is not None:
        progress(0.9, desc=f"👑 {display} synthesizing…")

    prompt = _format_boardroom_prompt(question, peer_results)
    _, payload = query_model(
        executive_key, prompt,
        rag_context=None, private_context=None,
        web_search=web_search, web_context=web_context,
        history=history, grounded=grounded,
    )
    return {
        "executive_key":         executive_key,
        "executive_display":     display,
        "executive_response":    payload["text"],
        "executive_elapsed":     payload["elapsed"],
        "executive_input_cost":  payload.get("input_cost",  0.0),
        "executive_output_cost": payload.get("output_cost", 0.0),
        "peers_answered":        len([r for r in peer_results if r.get("response")]),
        "peers_total":           len(peer_results),
    }


def run_research_models(question: str, selected_models: list, progress=None,
                          history: list[dict] | None = None,
                          grounded: bool = False) -> list:
    """Run the question through Serper FIRST, then fan-out to every
    selected model with the search results as web_context. Returns the
    same result-dict shape as run_models so the Arena slideshow / scoring
    machinery can be reused.

    Unlike run_big3_models (native per-provider web search, cloud-only),
    this works for every model — locals + clouds — because the search is
    done OUTSIDE the model call and injected as context. The trade-off
    is no model-side citation handling; the model has to format URLs
    from the snippets we hand it.
    """
    results = []
    if not selected_models:
        return results

    round_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{random.randint(0, 0xFFFF):04x}"

    # ONE Serper call per question, shared across the fan-out. Same
    # fetch-once pattern as RAG.
    if progress is not None:
        progress(0.0, desc="Fetching web search results from Serper…")
    web_context = serper_search(question)
    if web_context:
        log.info("Serper context retrieved (%d chars)", len(web_context))
    else:
        log.info(
            "Serper returned nothing — models will run without web context "
            "(check SERPER_API_KEY in .env, or the query may have no results)"
        )

    total = len(selected_models)
    pending: set = set(selected_models)
    done = 0
    if progress is not None:
        names = ", ".join(
            AVAILABLE_MODELS.get(k, {}).get("display_name", k) for k in selected_models
        )
        progress(0.05, desc=f"Querying {total} models with web context · {names}")

    with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
        futures = {
            executor.submit(
                query_model, m, question, None, None, False, web_context,
                history, grounded,
            ): m
            for m in selected_models
        }
        for future in as_completed(futures):
            if _ao_cancel_flag.is_set():
                for f in futures:
                    f.cancel()
                log.info("run_*_models: cancelled — %d results collected",
                         len(results))
                break
            model_key, payload = future.result()
            display = AVAILABLE_MODELS.get(model_key, {}).get("display_name", model_key)
            results.append({
                "model_key":     model_key,
                "model_display": display,
                "response":      payload["text"],
                "elapsed":       payload["elapsed"],
                "input_tokens":  payload["input_tokens"],
                "output_tokens": payload["output_tokens"],
                "input_cost":    payload["input_cost"],
                "output_cost":   payload["output_cost"],
                "round_id":      round_id,
                "error":         payload.get("error"),
            })
            pending.discard(model_key)
            done += 1
            if progress is not None:
                if pending:
                    pending_names = ", ".join(sorted(
                        AVAILABLE_MODELS.get(k, {}).get("display_name", k) for k in pending
                    ))
                    desc = (
                        f"✓ {display} done ({done}/{total}) · "
                        f"waiting on: {pending_names}"
                    )
                else:
                    desc = f"✓ all {total} models done"
                progress(done / total, desc=desc)

    random.shuffle(results)
    return results


def run_big3_models(question: str, selected_models: list, progress=None,
                      history: list[dict] | None = None,
                      grounded: bool = False) -> list:
    """Query each selected model with web_search=True, no scoring.

    No RAG or private corpus injection (research is for fact-finding,
    not persona embellishment). No shuffle either — research output is
    presented in completion order so the user can compare what each
    provider's search returned in a stable layout.
    """
    results = []
    if not selected_models:
        return results

    total = len(selected_models)
    pending: set = set(selected_models)
    done = 0
    if progress is not None:
        names = ", ".join(
            AVAILABLE_MODELS.get(k, {}).get("display_name", k) for k in selected_models
        )
        progress(0.0, desc=f"Researching across {total} models · {names}")

    with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
        futures = {
            executor.submit(
                query_model, m, question, None, None, True, None, history,
                grounded,
            ): m
            for m in selected_models
        }
        for future in as_completed(futures):
            if _ao_cancel_flag.is_set():
                for f in futures:
                    f.cancel()
                log.info("run_*_models: cancelled — %d results collected",
                         len(results))
                break
            model_key, payload = future.result()
            display = AVAILABLE_MODELS.get(model_key, {}).get("display_name", model_key)
            results.append({
                "model_key":     model_key,
                "model_display": display,
                "response":      payload["text"],
                "elapsed":       payload["elapsed"],
                "input_tokens":  payload["input_tokens"],
                "output_tokens": payload["output_tokens"],
                "input_cost":    payload["input_cost"],
                "output_cost":   payload["output_cost"],
                "error":         payload.get("error"),
            })
            pending.discard(model_key)
            done += 1
            if progress is not None:
                if pending:
                    pending_names = ", ".join(sorted(
                        AVAILABLE_MODELS.get(k, {}).get("display_name", k) for k in pending
                    ))
                    desc = (
                        f"✓ {display} done ({done}/{total}) · "
                        f"waiting on: {pending_names}"
                    )
                else:
                    desc = f"✓ all {total} research lookups done"
                progress(done / total, desc=desc)

    return results


# ── Stats / leaderboard ───────────────────────────────────────────────────────

def get_last_checkpoint_ts(conn) -> str | None:
    row = conn.execute(
        "SELECT timestamp FROM training_checkpoints ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def _safe_stats_panel(builder, label: str) -> str:
    """Run a stats-builder function; on DB failure return a visible error
    rather than letting the stats tab crash."""
    try:
        return builder()
    except sqlite3.DatabaseError as e:
        log.exception("%s query failed", label)
        return f"_{label} unavailable: {type(e).__name__}_"


# Two tables on the Stats tab. Each row is a list of typed numerics
# (float / int / None) so gr.Dataframe's click-to-sort treats them
# numerically. Headers and row shape kept in lockstep with the SQL below.
HIST_HEADERS = [
    "Model", "N",
    "Avg Score", "Avg Arena", "Avg Notebook",
    "Avg Time (s)", "The Reading", "Cost Reading",
    "HR (4)", "XB (3)", "Hit (2)", "1B (1)", "Whiff (0)",
    "Cost (USD)", "RAG Adds",
]

SESSION_HEADERS = [
    "Model", "N",
    "Avg Score", "Avg Arena", "Avg Notebook",
    "Avg Time (s)", "The Reading", "Cost Reading", "Δ Reading",
    "HR (4)", "XB (3)", "Hit (2)", "1B (1)", "Whiff (0)",
    "Cost (USD)", "RAG Adds",
]


def _build_stats_tables_inner() -> tuple[str, list[list], str, list[list]]:
    """Return (hist_caption, hist_rows, session_caption, session_rows).

    Two side-by-side perspectives on the arena:
      - Historical: every model, all-time, no checkpoint filter. Always
        populated as long as the DB has any scored row — survives a
        fresh checkpoint that would leave the session table empty.
      - Current Session: rows logged since the last training checkpoint
        (or all-time when no checkpoint exists yet). Carries a Δ Reading
        column against the pre-checkpoint baseline so the user can see
        whether the current session is trending up or down.
    """
    with connection_scope() as conn:
        last_cp = get_last_checkpoint_ts(conn)

        def query_period(where_clause: str, params: tuple,
                         rag_where_clause: str = "",
                         rag_params: tuple = ()) -> list[tuple]:
            # avg_arena / avg_notebook discriminate by tag_source. Notebook
            # rows are logged with tag_source='notebook' (see log_notebook_score);
            # arena rows use 'keyword' (or NULL on legacy data) — coerced
            # with COALESCE so pre-tag_source rows count as arena.
            # rag_adds joins rag_contributions for the same model_display,
            # windowed by the same checkpoint filter.
            return conn.execute(f"""
                SELECT a.model_display,
                       COUNT(*)                                           AS total,
                       ROUND(AVG(a.score), 2)                             AS avg_score,
                       ROUND(AVG(CASE WHEN COALESCE(a.tag_source, '') != 'notebook'
                                      THEN a.score END), 2)               AS avg_arena,
                       ROUND(AVG(CASE WHEN a.tag_source = 'notebook'
                                      THEN a.score END), 2)               AS avg_notebook,
                       ROUND(AVG(a.response_time_seconds), 2)             AS avg_time,
                       ROUND(AVG(a.reading), 4)                           AS avg_reading,
                       ROUND(AVG(a.cost_reading), 6)                      AS avg_cost_reading,
                       SUM(CASE WHEN a.score = 4 THEN 1 ELSE 0 END)       AS home_runs,
                       SUM(CASE WHEN a.score = 3 THEN 1 ELSE 0 END)       AS extra_bases,
                       SUM(CASE WHEN a.score = 2 THEN 1 ELSE 0 END)       AS base_hits,
                       SUM(CASE WHEN a.score = 1 THEN 1 ELSE 0 END)       AS contact,
                       SUM(CASE WHEN a.score = 0 THEN 1 ELSE 0 END)       AS whiffs,
                       COALESCE(SUM(a.input_cost),  0.0)
                         + COALESCE(SUM(a.output_cost), 0.0)              AS total_cost,
                       COALESCE((
                           SELECT COUNT(*) FROM rag_contributions r
                           WHERE r.model_display = a.model_display
                           {rag_where_clause}
                       ), 0)                                              AS rag_adds
                FROM arena_results a
                WHERE a.score >= 0 AND a.response_time_seconds >= 0
                {where_clause}
                GROUP BY a.model_display
            """, params + rag_params).fetchall()

        hist_raw     = query_period("", ())
        if last_cp:
            session_raw  = query_period(
                "AND a.timestamp > ?",  (last_cp,),
                "AND r.timestamp > ?",  (last_cp,),
            )
            previous_raw = query_period(
                "AND a.timestamp <= ?", (last_cp,),
                "AND r.timestamp <= ?", (last_cp,),
            )
        else:
            # No checkpoint yet — current session is "everything so far".
            session_raw  = hist_raw
            previous_raw = []

    prev_map = {r[0]: r for r in previous_raw}

    # Indices into the raw SQL tuple (same shape for hist / session / prev):
    #   0 model, 1 total, 2 avg_score, 3 avg_arena, 4 avg_notebook,
    #   5 avg_time, 6 avg_reading, 7 avg_cost_reading,
    #   8 hr, 9 xb, 10 hit, 11 contact, 12 whiff, 13 total_cost, 14 rag_adds
    _AVG_READING = 6

    def _format_row(r: tuple, *, include_delta: bool) -> list:
        (model, total, avg_score, avg_arena, avg_notebook,
         avg_time, avg_reading, avg_cost_reading,
         hr, xb, hit, contact, whiff, total_cost, rag_adds) = r
        cost = round(total_cost or 0.0, 4)
        if include_delta:
            prev = prev_map.get(model)
            if prev and prev[_AVG_READING] is not None and avg_reading is not None:
                delta = round(avg_reading - prev[_AVG_READING], 4)
            else:
                delta = None
            return [model, total, avg_score, avg_arena, avg_notebook,
                    avg_time, avg_reading, avg_cost_reading, delta,
                    hr, xb, hit, contact, whiff, cost, rag_adds or 0]
        return [model, total, avg_score, avg_arena, avg_notebook,
                avg_time, avg_reading, avg_cost_reading,
                hr, xb, hit, contact, whiff, cost, rag_adds or 0]

    hist_rows    = [_format_row(r, include_delta=False) for r in hist_raw]
    session_rows = [_format_row(r, include_delta=True)  for r in session_raw]

    # Default sort: Avg Score desc for historical, The Reading desc for
    # session. avg_score is r[2] in both; The Reading shifts from r[4] to
    # r[6] in session row layout due to the two new split columns.
    hist_rows.sort(   key=lambda r: (r[2] is None, -(r[2] or 0)))   # r[2] = avg_score
    session_rows.sort(key=lambda r: (r[6] is None, -(r[6] or 0)))   # r[6] = avg_reading

    from config import SCORE_MAX  # local import — keeps module-level free
    hist_caption = (
        f"### Historical Stats\n"
        f"All-time per-model scoring summary (Avg Score on the 0-{SCORE_MAX} "
        f"baseball scale). _Click any column header to sort. Default order: "
        f"Avg Score descending._"
    )
    cp_tag = f"since checkpoint: {last_cp[:10]}" if last_cp else "all time — no checkpoints yet"
    session_caption = (
        f"### Current Session Stats\n"
        f"Rows logged {cp_tag}. Δ Reading compares this session's average "
        f"Reading against the pre-checkpoint baseline. _Click any column "
        f"header to sort. Default order: The Reading descending._"
    )

    if not hist_rows:
        hist_rows = [["No scored results yet"] + [None] * (len(HIST_HEADERS) - 1)]
    if not session_rows:
        session_rows = [["No session data yet"] + [None] * (len(SESSION_HEADERS) - 1)]

    return hist_caption, hist_rows, session_caption, session_rows


def _build_checkpoint_log_inner() -> str:
    """Render the checkpoint log with a per-session token-cost column.

    Session cost for checkpoint X = sum(input_cost + output_cost) of every
    arena_results row logged after the previous checkpoint and at-or-before
    X. The very first checkpoint counts every row up to it.

    LAG() is a SQLite window function (available since 3.25) — gives the
    previous checkpoint's timestamp inline so each row's cost subquery
    can range over (prev_ts, this_ts] without a Python loop.
    """
    with connection_scope() as conn:
        rows = conn.execute(
            """
            WITH bounds AS (
                SELECT id, timestamp, name, model_key, notes,
                       LAG(timestamp) OVER (ORDER BY id) AS prev_ts
                FROM training_checkpoints
            )
            SELECT b.timestamp, b.name, b.model_key, b.notes,
                   COALESCE((
                       SELECT SUM(COALESCE(input_cost, 0)
                                + COALESCE(output_cost, 0))
                       FROM arena_results
                       WHERE timestamp <= b.timestamp
                         AND (b.prev_ts IS NULL OR timestamp > b.prev_ts)
                   ), 0.0) AS session_cost
            FROM bounds b
            ORDER BY b.id DESC
            LIMIT 20
            """
        ).fetchall()
    if not rows:
        return "No training checkpoints yet. Use the **Mark Training Checkpoint** button above."
    lines = ["**Training checkpoints**\n"]
    lines.append("| Date | Name | Model | Notes | Session Cost (USD) |")
    lines.append("|------|------|-------|-------|--------------------|")
    for ts, name, model_key, notes, session_cost in rows:
        cost_val = round(session_cost or 0.0, 4)
        cost_str = f"${cost_val:.4f}" if cost_val else "$0"
        lines.append(
            f"| {ts[:10]} | {name} | {model_key or 'all'} | "
            f"{notes or '—'} | {cost_str} |"
        )
    return "\n".join(lines)


def _build_persona_stats_inner() -> str:
    """One table per persona in PERSONA_NAMES.

    Each table shows every model's N / avg score / avg reading for
    responses that leaned on THAT persona. Reads PERSONA_NAMES from
    config so adding a persona over there auto-adds a table here. The
    "mixed" and "none" buckets are summarized as a trailing line for
    context but don't get their own tables (they're noise, not signal).
    """
    from config import PERSONA_NAMES, SCORE_MAX

    sections: list[str] = ["**Persona lean** — per-persona breakdown of model performance\n"]

    with connection_scope() as conn:
        for persona in PERSONA_NAMES:
            rows = conn.execute("""
                SELECT model_display,
                       COUNT(*)               AS n,
                       ROUND(AVG(score), 2)   AS avg_score,
                       ROUND(AVG(reading), 4) AS avg_reading
                FROM arena_results
                WHERE score >= 0 AND persona_lean = ?
                GROUP BY model_display
                ORDER BY avg_reading DESC, n DESC
            """, (persona,)).fetchall()

            block = [f"### {persona.title()}"]
            if not rows:
                block.append(f"_No responses have leaned on {persona.title()} yet._")
            else:
                block.append(f"| Model | N | Avg Score (0-{SCORE_MAX}) | Avg Reading |")
                block.append("|-------|---|--------------|-------------|")
                for model, n, avg_score, avg_reading in rows:
                    block.append(f"| {model} | {n} | {avg_score} | {avg_reading} |")
            sections.append("\n".join(block))

        # Tail line: how many responses landed in the synthesized buckets.
        # Useful diagnostic — if "none" dominates, PERSONA_MARKERS need tuning.
        other = dict(conn.execute("""
            SELECT persona_lean, COUNT(*)
            FROM arena_results
            WHERE score >= 0 AND persona_lean IN ('mixed', 'none')
            GROUP BY persona_lean
        """).fetchall())

    if other:
        bits = []
        if other.get("mixed"):
            bits.append(f"{other['mixed']} mixed")
        if other.get("none"):
            bits.append(f"{other['none']} no clear lean")
        sections.append(f"_Uncategorized: {', '.join(bits)} — tune `PERSONA_MARKERS` in config.py to shift these._")

    return "\n\n".join(sections)


def _build_refusals_inner() -> str:
    """Per-model count of Valid Refusal scores.

    These rows are excluded from every other panel (Reading averages,
    leaderboard, persona stats, trainer report, training-data export)
    because their stored score is REFUSAL_SCORE (< 0). This panel is
    where the user actually sees them.
    """
    with connection_scope() as conn:
        rows = conn.execute(
            """SELECT model_display, COALESCE(model_type, '?') AS mtype,
                      COUNT(*) AS n
               FROM arena_results
               WHERE score = ?
               GROUP BY model_display, mtype
               ORDER BY n DESC, model_display ASC""",
            (REFUSAL_SCORE,),
        ).fetchall()

    if not rows:
        return ("**Safety refusals** — none logged yet.  \n"
                f"_Pick **{REFUSAL_LABEL}** in the score radio when a model "
                "declines a task on its own safety policy. Refusals are excluded "
                "from training and from every Reading / Avg-Score stat._")

    lines = [
        "**Safety refusals** — responses scored as valid model refusals.  \n"
        "_Excluded from training data and from all Reading / Avg-Score stats._\n",
        "| Model | Type | Refusals |",
        "|-------|------|----------|",
    ]
    for model, mtype, n in rows:
        lines.append(f"| {model} | {mtype} | {n} |")
    return "\n".join(lines)


def _build_tag_stats_inner() -> str:
    with connection_scope() as conn:
        rows = conn.execute("""
            SELECT tags, COUNT(*) AS n, ROUND(AVG(score), 2) AS avg_score,
                   ROUND(AVG(reading), 4) AS avg_reading
            FROM arena_results
            WHERE score >= 0 AND tags IS NOT NULL
            GROUP BY tags ORDER BY n DESC LIMIT 20
        """).fetchall()
    if not rows:
        return "No tag data yet."
    from config import SCORE_MAX
    lines = ["**Top tags**\n"]
    lines.append(f"| Tag | Count | Avg Score (0-{SCORE_MAX}) | Avg Reading |")
    lines.append("|-----|-------|-----------|-------------|")
    for tag, n, avg_score, avg_reading in rows:
        lines.append(f"| {tag} | {n} | {avg_score} | {avg_reading} |")
    return "\n".join(lines)


def build_stats_tables() -> tuple[str, list[list], str, list[list]]:
    """Safe wrapper around the two-table builder. Returns the same shape
    on DB failure so the Stats tab degrades cleanly instead of crashing."""
    try:
        return _build_stats_tables_inner()
    except sqlite3.DatabaseError as e:
        log.exception("Stats tables query failed")
        err_hist    = [["error"] + [None] * (len(HIST_HEADERS) - 1)]
        err_session = [["error"] + [None] * (len(SESSION_HEADERS) - 1)]
        return (
            f"_Historical stats unavailable: {type(e).__name__}_",
            err_hist,
            f"_Current session stats unavailable: {type(e).__name__}_",
            err_session,
        )

def build_checkpoint_log() -> str:
    return _safe_stats_panel(_build_checkpoint_log_inner, "Checkpoint log")

def build_persona_stats() -> str:
    return _safe_stats_panel(_build_persona_stats_inner, "Persona lean")

def build_refusals() -> str:
    return _safe_stats_panel(_build_refusals_inner, "Safety refusals")

def build_tag_stats() -> str:
    return _safe_stats_panel(_build_tag_stats_inner, "Tag stats")


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    keyword_map = load_keyword_map()

    def on_ask(question, local_display_list, cloud_display_list, blind_mode,
               s_results, s_idx, progress=gr.Progress()):
        # Validation paths: leave retry_btn state untouched (gr.update() = no-op).
        if not question.strip():
            return (s_results, s_idx, "Enter a question first.", "",
                    gr.update(value="—"), gr.update(interactive=False), gr.update())
        model_display_list = (local_display_list or []) + (cloud_display_list or [])
        if not model_display_list:
            return (s_results, s_idx, "Select at least one model.", "",
                    gr.update(value="—"), gr.update(interactive=False), gr.update())
        media_error = _media_rule_error(question, len(model_display_list))
        if media_error:
            return (s_results, s_idx, media_error, "",
                    gr.update(value="—"), gr.update(interactive=False), gr.update())

        model_key_map = {AVAILABLE_MODELS[m]["display_name"]: m
                         for m in ARENA_MODELS if m in AVAILABLE_MODELS}
        selected = [model_key_map[d] for d in model_display_list if d in model_key_map]
        results  = run_models(question, selected, progress=progress)

        available   = [r for r in results if r["response"] is not None]
        unavailable = [r for r in results if r["response"] is None]

        if not available:
            return (s_results, s_idx, "No models responded. Check Ollama.", "",
                    gr.update(value="—"), gr.update(interactive=False), gr.update(interactive=False))

        status_parts = []
        if unavailable:
            status_parts.append("⚠️ Unavailable: " + " · ".join(
                f"{r['model_display']} ({r.get('error') or 'no response'})"
                for r in unavailable
            ))
        status_parts.append(f"Showing 1 of {len(available)}")

        first = available[0]
        label = ("Model A" if blind_mode else first["model_display"]) + f"  ·  {first['elapsed']}s"

        # Successful ask: retry becomes available from this point on.
        return (
            available, 0,
            " · ".join(status_parts),
            first["response"],
            gr.update(value=label),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def _prompt_references_media(question: str) -> list[str]:
        """Return filenames from docs/media/ that appear in the question
        text (case-insensitive substring match on full filename+ext).
        Catches all three ways a user typically mentions a file: bare
        name (`screenshot.png`), relative path (`docs/media/…`), or
        absolute path — all three contain the filename."""
        if not question or not MEDIA_DIR.exists():
            return []
        q_lower = question.lower()
        return [
            p.name for p in MEDIA_DIR.iterdir()
            if p.is_file() and p.name.lower() in q_lower
        ]

    def _media_rule_error(question: str, selected_count: int) -> Optional[str]:
        """Return an enforcement banner if the question mentions any
        media file AND the current selection exceeds MEDIA_MAX_MODELS.
        None otherwise (either no media referenced, or selection is
        within the limit)."""
        if selected_count <= MEDIA_MAX_MODELS:
            return None
        refs = _prompt_references_media(question)
        if not refs:
            return None
        files = ", ".join(f"`{r}`" for r in refs)
        return (
            f"🚨 **One-model rule: media detected in prompt.** "
            f"Your question references {files}. Image and video tokens "
            f"cost 10–1000× more than text tokens, so Agent Ouija refuses "
            f"to fan out a media prompt across the boardroom. You have "
            f"**{selected_count} models checked** — uncheck all but one "
            f"and re-Ask."
        )

    def _media_policy_banner(selected_names: list[str]) -> str:
        """Return the one-model-rule status banner shown ahead of the
        media cost preview. Explains that per-query media processing
        should route through a single model, and reports whether the
        user's current selection satisfies that."""
        n = len(selected_names)
        if n == MEDIA_MAX_MODELS:
            return (
                f"✅ **One model checked: {selected_names[0]}** — media "
                f"processing is OK to proceed with this model."
            )
        if n == 0:
            return (
                "⚠️ **No models checked.** Pick exactly one model before "
                "referencing this media in a question — the one-model "
                "rule applies to any prompt that includes a picture "
                "or video."
            )
        return (
            f"🚨 **One-model rule for media files.** Image and video "
            f"tokens cost 10–1000× more than text tokens, so Agent Ouija's "
            f"convention is that a single model — not the whole "
            f"boardroom — processes each media prompt. You currently "
            f"have **{n} models checked** "
            f"({', '.join(selected_names)}). Uncheck all but one before "
            f"including this media in a question."
        )

    def on_upload(files, primary_selection=None, secondary_selection=None):
        """Sort uploaded files into docs/ (documents) or docs/media/
        (images + videos), auto-run RAG ingestion on the document set,
        and return a Markdown status block with a per-file media cost
        analysis so the user knows what each cloud model would charge
        before including a picture or video in a question.

        Media files are NOT auto-ingested — the pipeline is text-only,
        and the vision-side cost is what the user has to decide about
        first. The cost preview is informational; media stays parked in
        docs/media/ until the user manually references it in a prompt.

        primary_selection / secondary_selection are the currently-checked
        display names on the calling tab (Arena/Research/Boardroom have
        two checkbox groups, Big 3 has one — the second arg is left
        None on Big 3). Used to render the one-model-rule banner when
        the upload contains any media file.

        `files` is the list of TemporaryFile-like paths Gradio passes
        after an UploadButton click.
        """
        if not files:
            return "_No files uploaded._"

        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)

        moved_docs: list[str]  = []
        moved_media: list[dict] = []
        errors: list[str]       = []
        unknown: list[str]      = []

        for f in files:
            src_path = getattr(f, "name", None) or str(f)
            src      = Path(src_path)
            kind     = classify_file(src)
            try:
                if kind == "document":
                    dest = DOCS_DIR / src.name
                    shutil.copy2(src, dest)
                    moved_docs.append(src.name)
                elif kind in ("image", "video"):
                    dest = MEDIA_DIR / src.name
                    shutil.copy2(src, dest)
                    moved_media.append(estimate_file(dest))
                else:
                    unknown.append(src.name)
            except Exception as e:
                errors.append(f"{src.name}: {type(e).__name__}: {e}")

        parts: list[str] = []
        if moved_docs:
            log.info("Upload: ingesting %d document(s)", len(moved_docs))
            ing = run_ingestion()
            if ing.get("status") == "error":
                parts.append(f"❌ Ingestion failed: {ing.get('message')}")
            else:
                # Backtick each filename so it renders as a code span —
                # easier to click-and-drag select for copy/paste into a
                # question. Same treatment for media below.
                doc_list = "\n".join(f"- `{n}`" for n in moved_docs)
                parts.append(
                    f"✅ **Uploaded {len(moved_docs)} document(s)** and "
                    f"rebuilt the RAG collection "
                    f"({ing.get('documents', 0)} chunks total).\n\n"
                    f"Auto-ingested:\n{doc_list}"
                )
        if moved_media:
            selected = (primary_selection or []) + (secondary_selection or [])
            parts.append(_media_policy_banner(selected))
            media_list = "\n".join(f"- `{m['name']}`" for m in moved_media)
            parts.append(
                f"🎨 **Parked {len(moved_media)} media file(s)** in "
                f"`docs/media/`. Reference them by name in a prompt "
                f"(e.g. *\"Analyze `{moved_media[0]['name']}` and tell me…\"*):"
                f"\n\n{media_list}"
            )
            parts.append(format_report(moved_media))
        if unknown:
            unk_list = "\n".join(f"- `{n}`" for n in unknown)
            parts.append(
                f"⚠️ Ignored (unknown file type):\n{unk_list}"
            )
        if errors:
            parts.append("❌ Errors:\n- " + "\n- ".join(errors))

        return "\n\n".join(parts) if parts else "_No supported files in upload._"

    def on_summary():
        """Generate a plain-text historical report into data/summaries/.
        Windowed to the last 24h from the latest scored prompt."""
        log.info("Summary requested via UI")
        result = generate_summary()
        status  = result.get("status", "error")
        message = result.get("message", "Unknown summary error")
        if status == "error":
            return f"❌ **Summary failed:** {message}"
        if status == "warning":
            return f"⚠️ {message}"
        return f"✅ {message}"

    def on_trainer_report():
        """Generate the per-model Taught / Learned report (last 24h)."""
        log.info("Trainer report requested via UI")
        result = generate_trainer_report()
        status  = result.get("status", "error")
        message = result.get("message", "Unknown trainer-report error")
        if status == "error":
            return f"❌ **Trainer report failed:** {message}"
        if status == "warning":
            return f"⚠️ {message}"
        return f"✅ {message}"

    def on_retry(blind_mode, s_results, s_idx):
        """Re-step through the current question's responses for re-scoring.

        Resets the slideshow to position 0 after re-shuffling so the blind
        order is fresh. Each re-score writes a new row to arena_results — the
        user accepted that historic data is disposable, so we don't try to
        update prior rows in place.
        """
        if not s_results:
            return (s_results, s_idx, "Nothing to retry yet — ask a question first.",
                    "", gr.update(value="—"), gr.update(interactive=False),
                    gr.update(value=None), gr.update(interactive=False))

        shuffled = list(s_results)
        random.shuffle(shuffled)
        first = shuffled[0]
        label = ("Model A" if blind_mode else first["model_display"]) + f"  ·  {first['elapsed']}s"

        return (
            shuffled, 0,
            f"Retrying — showing 1 of {len(shuffled)}",
            first["response"],
            gr.update(value=label),
            gr.update(interactive=False),   # score_btn disabled until a score is selected
            gr.update(value=None),          # clear score_radio
            gr.update(interactive=True),    # retry_btn stays available
        )

    def _show_slide(s_results, idx, blind_mode, status_msg):
        """Helper — render the slideshow at position idx with no score state
        change. Used by on_previous / on_next so they share rendering.
        Executive slides bypass blind mode — their tier label IS the
        useful identifier, not a hidden model.
        """
        item = s_results[idx]
        if item.get("kind") == "executive":
            label = f"{item['model_display']}  ·  {item['elapsed']}s"
        else:
            label = (f"Model {chr(65 + idx)}" if blind_mode else item["model_display"]) + f"  ·  {item['elapsed']}s"
        return (
            s_results, idx,
            status_msg,
            item["response"],
            gr.update(value=label),
            gr.update(interactive=False),   # score_btn disabled until radio is touched
            gr.update(value=None),          # clear score_radio
            gr.update(interactive=True),    # retry_btn stays available
        )

    def on_previous(blind_mode, s_results, s_idx):
        """Step back one position in the slideshow without scoring.

        No-op at position 0 — the status message tells the user.
        """
        if not s_results:
            return (s_results, s_idx, "Nothing to navigate yet — ask a question first.",
                    "", gr.update(value="—"), gr.update(interactive=False),
                    gr.update(value=None), gr.update(interactive=False))
        if s_idx <= 0:
            return _show_slide(s_results, 0, blind_mode,
                               f"Already at the first response (1 of {len(s_results)}).")
        new_idx = s_idx - 1
        return _show_slide(s_results, new_idx, blind_mode,
                           f"Showing {new_idx + 1} of {len(s_results)}")

    def on_next(blind_mode, s_results, s_idx):
        """Step forward one position in the slideshow without scoring.

        Pairs with Previous so the user can scroll back and forth. To
        record a score AND advance, use Score And Next Response instead.
        """
        if not s_results:
            return (s_results, s_idx, "Nothing to navigate yet — ask a question first.",
                    "", gr.update(value="—"), gr.update(interactive=False),
                    gr.update(value=None), gr.update(interactive=False))
        last_idx = len(s_results) - 1
        if s_idx >= last_idx:
            return _show_slide(s_results, last_idx, blind_mode,
                               f"Already at the last response ({len(s_results)} of {len(s_results)}).")
        new_idx = s_idx + 1
        return _show_slide(s_results, new_idx, blind_mode,
                           f"Showing {new_idx + 1} of {len(s_results)}")

    def on_skip_round():
        """Abandon the current round of responses without scoring.

        Clears slideshow state so the user can ask a new prompt
        immediately. Any rows already scored in this round stay in the DB
        — only the unscored remainder is dropped.
        """
        return (
            [], 0,
            "Round skipped — ready for the next prompt.",
            "",
            gr.update(value="—"),
            gr.update(interactive=False),  # score_btn off
            gr.update(value=None),          # clear radio
            gr.update(interactive=False),  # retry_btn off
        )

    def on_checkpoint():
        """Record a training checkpoint with an auto-generated timestamp name.

        Returns (status_text, refreshed_checkpoint_log_markdown) so the
        checkpoint table updates immediately without needing Refresh Stats.
        """
        name = f"Manual checkpoint {datetime.now():%Y-%m-%d %H:%M:%S}"
        log.info("Checkpoint requested via UI: %s", name)
        result = add_checkpoint(name)
        new_log = build_checkpoint_log()
        if result.get("status") == "error":
            return f"❌ **Checkpoint failed:** {result.get('message')}", new_log
        return f"✅ {result.get('message')}", new_log

    # ── Wipe Database (three-way confirm) ─────────────────────────────────────
    # Click flow:
    #   wipe_btn         → arms the three confirm buttons (hides wipe_btn)
    #   yes_nobackup_btn → wipes immediately, no snapshot
    #   yes_backup_btn   → snapshots arena_scores.db + chat_history.json + chroma/
    #                      into data/backups/wipe_<stamp>/ FIRST, then wipes
    #   no_btn           → restores initial visibility, no DB writes
    # Each terminal branch restores the initial button visibility and refreshes
    # all three Stats markdowns inline.

    _ARM_VISIBILITY = (
        gr.update(visible=False),   # wipe_btn
        gr.update(visible=True),    # yes_nobackup_btn
        gr.update(visible=True),    # yes_backup_btn
        gr.update(visible=True),    # no_btn
    )
    _IDLE_VISIBILITY = (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )

    def _perform_wipe() -> tuple[str, bool]:
        """Run the three-way wipe and return (summary_msg, all_ok)."""
        sql_result    = wipe_arena_db()
        chroma_result = wipe_chroma_collection()

        chat_deleted = False
        chat_error   = None
        if CHAT_HISTORY_PATH.exists():
            try:
                CHAT_HISTORY_PATH.unlink()
                chat_deleted = True
            except OSError as e:
                chat_error = str(e)
                log.warning("Could not delete %s: %s", CHAT_HISTORY_PATH, e)

        parts = []
        if sql_result.get("status") == "error":
            parts.append(f"❌ SQL: {sql_result.get('message')}")
        else:
            parts.append(
                f"{sql_result['rows_deleted']} scored rows + "
                f"{sql_result['checkpoints_deleted']} checkpoints"
            )

        if chroma_result.get("status") == "error":
            parts.append(f"❌ ChromaDB: {chroma_result.get('message')}")
        else:
            parts.append(f"{chroma_result.get('chunks_deleted', 0)} RAG chunks")

        if chat_deleted:
            parts.append("chat_history.json")
        elif chat_error:
            parts.append(f"❌ chat_history.json: {chat_error}")

        all_ok = (
            sql_result.get("status") == "success"
            and chroma_result.get("status") == "success"
            and chat_error is None
        )
        return ", ".join(parts), all_ok

    def on_wipe_request():
        return (
            *_ARM_VISIBILITY,
            "⚠️ **Are you sure?** This will permanently delete all generated history:\n"
            "- All scored rows and training checkpoints (`arena_scores.db`)\n"
            "- All RAG chunks (ChromaDB collection — rebuilt by **Ingest Documents**)\n"
            "- Chat history file (`chat_history.json`) if it exists\n\n"
            "**Preserved:** source docs in `docs/`, persona files in `prompts/`, "
            "saved summaries and training exports.\n\n"
            "**Backup** copies the database, chat history, and ChromaDB into "
            "`data/backups/wipe_<timestamp>/` before deleting.",
        )

    def on_wipe_no():
        return (
            *_IDLE_VISIBILITY,
            "Wipe cancelled.",
            gr.update(),  # hist caption unchanged
            gr.update(),  # hist df unchanged
            gr.update(),  # session caption unchanged
            gr.update(),  # session df unchanged
            gr.update(),  # checkpoint log unchanged
            gr.update(),  # refusals unchanged
            gr.update(),  # tag stats unchanged
        )

    def on_wipe_yes_nobackup():
        log.info("Full wipe confirmed via UI (no backup)")
        wipe_parts, all_ok = _perform_wipe()
        prefix  = "✅ Wiped:" if all_ok else "⚠️ Partial wipe:"
        summary = (
            f"{prefix} {wipe_parts}. "
            f"Click **Ingest Documents** on the Arena tab to rebuild RAG context."
        )
        h_cap, h_rows, s_cap, s_rows = build_stats_tables()
        return (
            *_IDLE_VISIBILITY,
            summary,
            h_cap, h_rows, s_cap, s_rows,
            build_checkpoint_log(),
            build_refusals(),
            build_tag_stats(),
        )

    def on_wipe_yes_backup():
        log.info("Full wipe confirmed via UI (with backup)")
        backup_result = backup_state(label="wipe")
        if backup_result.get("status") == "error":
            return (
                *_IDLE_VISIBILITY,
                f"❌ Backup failed — wipe aborted. {backup_result.get('message')}",
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        wipe_parts, wipe_ok = _perform_wipe()
        backup_path = backup_result.get("path", "?")
        prefix      = "✅ Wiped:" if wipe_ok else "⚠️ Partial wipe:"
        backup_tag  = "✅" if backup_result.get("status") == "success" else "⚠️"
        summary = (
            f"{backup_tag} Backup → `{backup_path}` ({backup_result.get('message')}). "
            f"{prefix} {wipe_parts}. "
            f"Click **Ingest Documents** on the Arena tab to rebuild RAG context."
        )
        h_cap, h_rows, s_cap, s_rows = build_stats_tables()
        return (
            *_IDLE_VISIBILITY,
            summary,
            h_cap, h_rows, s_cap, s_rows,
            build_checkpoint_log(),
            build_refusals(),
            build_tag_stats(),
        )

    def on_score(question, current_q, blind_mode, score_str, s_results, s_idx):
        # After Ask, the question textbox clears; the question we
        # actually asked lives in ao_current_q_state. Fall back to
        # it so scoring after Ask logs the right question instead
        # of an empty string.
        effective_q = ((question or "").strip()
                        or (current_q or "").strip())
        if not s_results:
            return (s_results, s_idx, "Ask a question first.", "",
                    gr.update(value="—"), gr.update(interactive=False),
                    gr.update(), gr.update(interactive=False))
        if score_str is None:
            return (s_results, s_idx, "Select a score first.",
                    s_results[s_idx]["response"], gr.update(),
                    gr.update(interactive=True), gr.update(), gr.update())

        current   = s_results[s_idx]
        score_val = int(score_str.split(" — ")[0])

        log_score(
            effective_q, current["model_key"], current["model_display"],
            current["response"], score_val, current["elapsed"], keyword_map,
            input_tokens=current.get("input_tokens", 0),
            output_tokens=current.get("output_tokens", 0),
            input_cost=current.get("input_cost", 0.0),
            output_cost=current.get("output_cost", 0.0),
            round_id=current.get("round_id"),
        )

        reading = compute_reading(score_val, current["elapsed"])

        # Re-shuffle remaining before advancing — always random order
        remaining = s_results[s_idx + 1:]
        random.shuffle(remaining)
        new_results = s_results[:s_idx + 1] + remaining

        next_idx = s_idx + 1
        if next_idx >= len(new_results):
            # All done — score_btn disabled, retry_btn stays enabled so the
            # user can re-step through before asking the next question.
            return (new_results, next_idx,
                    f"✅ All {len(new_results)} responses scored. Use **Retry Scoring** to re-review or ask another question.",
                    "", gr.update(value="—"), gr.update(interactive=False),
                    gr.update(value=None), gr.update(interactive=True))

        nxt   = new_results[next_idx]
        if nxt.get("kind") == "executive":
            label = f"{nxt['model_display']}  ·  {nxt['elapsed']}s"
        else:
            label = (f"Model {chr(65 + next_idx)}" if blind_mode else nxt["model_display"]) + f"  ·  {nxt['elapsed']}s"

        return (
            new_results, next_idx,
            f"Logged: {label_for_score(score_val)}  ·  {current['elapsed']}s  ·  Reading: {reading}  —  Showing {next_idx + 1} of {len(new_results)}",
            nxt["response"],
            gr.update(value=label),
            gr.update(interactive=True),
            gr.update(value=None),
            gr.update(interactive=True),
        )

    def on_score_and_rag(question, current_q, blind_mode, score_str,
                          s_results, s_idx):
        """Same as on_score, but ALSO ingests the current Q+A into the main
        ChromaDB collection so it becomes retrievable context for future
        queries. Locals only benefit if their per-model 'rag' flag is True
        (flip them in config.py — see CLAUDE.md).

        current_q falls in from ao_current_q_state — the question the
        user actually asked in this seance, preserved after the Ask
        textbox cleared. Without this fallback, hitting Score+RAG
        after Ask lands a "no question provided" from rag_add because
        `question` (from ao_question) is empty.
        """
        effective_q = ((question or "").strip()
                        or (current_q or "").strip())
        if not s_results:
            return (s_results, s_idx, "Ask a question first.", "",
                    gr.update(value="—"), gr.update(interactive=False),
                    gr.update(), gr.update(interactive=False))
        if score_str is None:
            return (s_results, s_idx, "Select a score first.",
                    s_results[s_idx]["response"], gr.update(),
                    gr.update(interactive=True), gr.update(), gr.update())

        current   = s_results[s_idx]
        score_val = int(score_str.split(" — ")[0])

        log_score(
            effective_q, current["model_key"], current["model_display"],
            current["response"], score_val, current["elapsed"], keyword_map,
            input_tokens=current.get("input_tokens", 0),
            output_tokens=current.get("output_tokens", 0),
            input_cost=current.get("input_cost", 0.0),
            output_cost=current.get("output_cost", 0.0),
            round_id=current.get("round_id"),
        )
        reading = compute_reading(score_val, current["elapsed"])

        # Add the Q+A to the main RAG collection. Metadata carries the model
        # + score so curated chunks can be filtered or audited later.
        rag_result = rag_add(
            question=effective_q,
            response=current["response"],
            metadata={
                "model_key":     current["model_key"],
                "model_display": current["model_display"],
                "score":         score_val,
                "round_id":      current.get("round_id"),
                "elapsed":       current["elapsed"],
            },
        )
        if rag_result.get("status") == "success":
            log_rag_contribution(
                model_key=current["model_key"],
                model_display=current["model_display"],
                model_type=AVAILABLE_MODELS.get(current["model_key"], {}).get("type", "unknown"),
                source="arena",
                question=effective_q,
                round_id=current.get("round_id"),
            )
        rag_tag = "RAG ✓" if rag_result.get("status") == "success" \
                  else f"RAG ❌ ({rag_result.get('message', 'unknown')})"

        remaining = s_results[s_idx + 1:]
        random.shuffle(remaining)
        new_results = s_results[:s_idx + 1] + remaining
        next_idx = s_idx + 1

        if next_idx >= len(new_results):
            return (new_results, next_idx,
                    f"✅ All {len(new_results)} responses scored. {rag_tag}. "
                    f"Use **Retry Scoring** to re-review or ask another question.",
                    "", gr.update(value="—"), gr.update(interactive=False),
                    gr.update(value=None), gr.update(interactive=True))

        nxt   = new_results[next_idx]
        if nxt.get("kind") == "executive":
            label = f"{nxt['model_display']}  ·  {nxt['elapsed']}s"
        else:
            label = (f"Model {chr(65 + next_idx)}" if blind_mode else nxt["model_display"]) + f"  ·  {nxt['elapsed']}s"
        return (
            new_results, next_idx,
            f"Logged: {label_for_score(score_val)}  ·  {current['elapsed']}s  ·  "
            f"Reading: {reading}  ·  {rag_tag}  —  Showing {next_idx + 1} of {len(new_results)}",
            nxt["response"],
            gr.update(value=label),
            gr.update(interactive=True),
            gr.update(value=None),
            gr.update(interactive=True),
        )

    # Partition into Local / Cloud so the UI can show each tier in its own
    # row with its own label. Order is preserved from ARENA_MODELS so the
    # user can shape the row layout by reordering config.
    local_choices = [AVAILABLE_MODELS[m]["display_name"]
                     for m in ARENA_MODELS
                     if m in AVAILABLE_MODELS
                     and AVAILABLE_MODELS[m].get("type") == "local"]
    cloud_choices = [AVAILABLE_MODELS[m]["display_name"]
                     for m in ARENA_MODELS
                     if m in AVAILABLE_MODELS
                     and AVAILABLE_MODELS[m].get("type") == "cloud"]
    # Score radio: 0-4 baseball, then a Valid Refusal entry at the bottom.
    # The parser in on_score does int(label.split(" — ")[0]) which works for
    # both "0 — whiff" and "-2 — valid refusal".
    score_choices = (
        [f"{k} — {v}" for k, v in SCORE_LABELS.items()]
        + [f"{REFUSAL_SCORE} — {REFUSAL_LABEL}"]
    )

    with gr.Blocks(title="Agent Ouija") as demo:

        gr.HTML(f"""
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=IM+Fell+English:ital@0;1&display=swap" rel="stylesheet">
        <style>
            .gradio-container {{ font-family: 'IM Fell English', serif !important; }}
            h1, h2, h3 {{ font-family: 'Cinzel', serif !important; font-weight: 400 !important; text-transform: uppercase; letter-spacing: 0.08em; }}
            .gr-button {{ font-family: 'Cinzel', serif !important; font-weight: 400 !important; letter-spacing: 0.06em; text-transform: uppercase; }}
            .gr-tab-nav button {{ font-family: 'Cinzel', serif !important; font-weight: 400 !important; text-transform: uppercase; letter-spacing: 0.05em; }}
            label span,
            .block-label,
            .block-label > span,
            .block-label span,
            [class*="block-label"],
            [class*="block-label"] span,
            [class*="block_label"] {{ font-family: 'Cinzel', serif !important; font-weight: 400 !important; text-transform: uppercase; letter-spacing: 0.04em; }}

            /* Force normal weight on every descendant of a block-label — Gradio's
               CheckboxGroup / Radio nest the visible text one level deeper than
               Textbox, so the outer 400 doesn't always reach the inner span and
               the browser falls back to a synthesized bold (the "blurry" look). */
            .block-label *,
            [class*="block-label"] * {{ font-weight: 400 !important; }}
            strong, b {{ font-weight: 400 !important; text-transform: uppercase; letter-spacing: 0.04em; }}

            /* Stats tab tables — readable numerics. Overrides body Cinzel/IM Fell
               styling inside table cells only, keeps the ouija aesthetic on chrome. */
            .gradio-container table {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                font-variant-numeric: tabular-nums;
                font-size: 13.5px;
            }}
            .gradio-container table th,
            .gradio-container table td {{
                font-family: inherit !important;
                letter-spacing: 0 !important;
                text-transform: none !important;
                font-weight: 400 !important;
            }}
            .gradio-container table th {{ font-weight: 600 !important; }}
            .gradio-container table strong,
            .gradio-container table b {{
                font-weight: 600 !important;
                text-transform: none !important;
                letter-spacing: 0 !important;
            }}

            /* Model checkbox groups — 5 across so each tier fits one row.
               Same layout for the Arena tab and the Notebook tab pickers. */
            #model_checks_local .wrap,
            #model_checks_local > div > div:not(:first-child),
            #model_checks_cloud .wrap,
            #model_checks_cloud > div > div:not(:first-child),
            #nb_local_checks .wrap,
            #nb_local_checks > div > div:not(:first-child),
            #nb_cloud_checks .wrap,
            #nb_cloud_checks > div > div:not(:first-child),
            #res_local_checks .wrap,
            #res_local_checks > div > div:not(:first-child),
            #res_cloud_checks .wrap,
            #res_cloud_checks > div > div:not(:first-child),
            #big3_checks .wrap,
            #big3_checks > div > div:not(:first-child),
            #boardroom_local_checks .wrap,
            #boardroom_local_checks > div > div:not(:first-child),
            #boardroom_cloud_checks .wrap,
            #boardroom_cloud_checks > div > div:not(:first-child),
            #files_local_checks .wrap,
            #files_local_checks > div > div:not(:first-child),
            #files_cloud_checks .wrap,
            #files_cloud_checks > div > div:not(:first-child) {{
                display: grid !important;
                grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
                gap: 6px 12px !important;
            }}

            /* Theme toggle row — flush right and compact, sits under the
               header without taking a full row of vertical space. */
            #theme_toggle_row {{
                justify-content: flex-end !important;
                margin: -8px 0 8px !important;
            }}

            /* Agent Ouija response Markdown — coloured blockquote for the
               question line so it reads as visually distinct from the
               answer without needing a monospace separator bar. Colour
               works in both light and dark themes; theme-aware
               background/border overrides sit below. */
            #ao_response {{
                min-height: 240px;
                max-height: 800px;
                overflow-y: auto;
                padding: 12px 16px;
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 8px;
                background: rgba(128, 128, 128, 0.03);
            }}
            #ao_response blockquote {{
                color: #7c3aed;
                border-left: 4px solid #7c3aed !important;
                padding: 8px 12px !important;
                margin: 12px 0 !important;
                background: rgba(124, 58, 237, 0.08);
                border-radius: 4px;
                font-weight: 600;
            }}
            body.dark #ao_response blockquote,
            .dark #ao_response blockquote {{
                color: #c4b5fd;
                border-left-color: #c4b5fd !important;
                background: rgba(196, 181, 253, 0.10);
            }}

            /* Ouija header icon backdrop — white in light mode, dark
               gray in dark mode. Gradio toggles a `.dark` class on
               <body> when the dark theme is active. */
            .ouija-icon-bg {{ background: #ffffff; }}
            body.dark .ouija-icon-bg,
            .dark .ouija-icon-bg {{ background: #3a3a3a; }}

            /* Hide the main tab bar entirely — Agent Ouija is the only
               user-visible tab; the other four (Dev Workshop / Files /
               Persona / Stats) are reached via the Control Center's nav
               buttons, and each has a "🔮 Return to Agent Ouija" button
               at the top so users always have a way back.

               Multiple selectors because Gradio's tab-bar class name has
               moved across versions (`.tab-nav`, `.tabs-nav`, and the
               ARIA-standard `[role="tablist"]`). One of these three will
               match. */
            #ao_main_tabs .tab-nav,
            #ao_main_tabs .tabs-nav,
            #ao_main_tabs [role="tablist"] {{ display: none !important; }}

            /* File-listing scroll container — FIXED height on both the
               Documents and Media panels so the dropdown/rename/delete
               controls beneath line up horizontally regardless of how
               many files each column has. Uses `height` (not
               `max-height`) so the box stays exactly 180px whether it
               holds 0 files or 50. Roughly a 5-line preview; scrolls
               when there are more.
               Inner selectors force the Gradio-rendered <div> children
               to also honor the fixed height — otherwise Markdown's
               internal wrappers shrink to their content and the outer
               container collapses. */
            .file-listing-scroll,
            .file-listing-scroll > div,
            .file-listing-scroll > div > div {{
                height: 180px !important;
                min-height: 180px !important;
                max-height: 180px !important;
            }}
            .file-listing-scroll {{
                overflow-y: auto !important;
                border: 1px solid rgba(128,128,128,0.25);
                border-radius: 6px;
                padding: 8px 12px !important;
                margin-bottom: 8px;
                background: rgba(255,255,255,0.4);
                box-sizing: border-box !important;
            }}
            body.dark .file-listing-scroll,
            .dark .file-listing-scroll {{ background: rgba(0,0,0,0.2); }}

            /* Executive-mode buttons — purple gradient across all four
               tabs (Arena, Research, Big 3, Boardroom) so the executive
               workflow reads consistently. Overrides both light and dark
               Gradio theme variants; explicit !important since Gradio's
               variant="primary" default has higher CSS specificity. */
            #arena_exec_btn button,
            #res_exec_btn button,
            #big3_exec_btn button,
            #boardroom_btn button {{
                background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
                color: #ffffff !important;
                border: none !important;
                font-family: 'Cinzel', serif !important;
                letter-spacing: 0.06em !important;
            }}
            #arena_exec_btn button:hover,
            #res_exec_btn button:hover,
            #big3_exec_btn button:hover,
            #boardroom_btn button:hover {{
                background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
                filter: brightness(1.05);
            }}

            /* On Arena + Research the exec button sits next to the
               Executive dropdown in the same Row. Stretch the button to
               match the dropdown's height (label + input) so the row
               reads symmetrically instead of a squat button and a tall
               dropdown. */
            #arena_exec_row, #res_exec_row {{
                align-items: stretch !important;
            }}
            #arena_exec_btn, #res_exec_btn {{
                align-self: stretch !important;
            }}
            #arena_exec_btn button,
            #res_exec_btn button {{
                min-height: 78px !important;
                height: 100% !important;
            }}

            /* Question + Response textareas — readable typing/reading font,
               override Cinzel/IM Fell English serif globally. Explicit
               overflow + max-height keeps long prompts from cutting off or
               pushing the Score button below the fold. */
            .gradio-container textarea {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
                font-size: 15px !important;
                line-height: 1.5 !important;
                letter-spacing: 0 !important;
                text-transform: none !important;
                overflow-y: auto !important;
                max-height: 55vh !important;
                resize: vertical;
            }}
        </style>
        <div style="display:flex;align-items:center;gap:12px;padding:12px 0 8px">
            <div class="ouija-icon-bg" style="width:36px;height:36px;flex-shrink:0;border-radius:6px;padding:2px">
                {OUIJA_HTML}
            </div>
            <div style="flex:1">
                <div style="font-size:18px;font-weight:600;line-height:1.2;font-family:'Cinzel',serif">Agent Ouija</div>
                <div style="font-size:12px;color:#888;font-family:'IM Fell English',serif;font-style:italic">local edge AI · model evaluation · The Reading</div>
            </div>
        </div>
        """)

        # Theme toggle as a real gr.Button — Gradio's HTML component
        # sanitizes inline onclick handlers, so the button needs to go
        # through Gradio's js= client-side event path. Flips
        # ?__theme=light/dark in the URL and reloads. Persists via URL
        # so a bookmark "always dark" works on Brainy, "always light"
        # on Big without touching config.py. No-param visits respect
        # the user's system preference.
        with gr.Row(elem_id="theme_toggle_row"):
            theme_btn = gr.Button("🌓 Toggle Theme", size="sm", scale=0)

        theme_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
                const url = new URL(window.location);
                const cur = url.searchParams.get('__theme');
                const next = cur === 'dark' ? 'light' : 'dark';
                url.searchParams.set('__theme', next);
                // Remember the choice so future no-param visits (e.g.
                // launcher opens localhost:7860, browser bookmarks
                // without the param, PWA installs) land on the same
                // theme instead of drifting to the OS system pref.
                try { localStorage.setItem('agent_ouija_theme', next); }
                catch (e) { /* private mode */ }
                window.location = url.toString();
            }""",
        )

        # First-visit / no-param theme alignment.
        #
        # If the URL has no `?__theme=`, check localStorage — if the
        # user has toggled before, respect their choice. Otherwise
        # default to dark (the app's designed home). Either way we
        # redirect ONCE with the theme param so Gradio's native
        # theme system takes over from there. Fixes the "Brainy
        # sometimes opens light" report — Windows 11's auto-day/night
        # was flipping the OS system preference under Gradio, so
        # no-param visits inherited whichever mode Windows was in
        # at load time.
        #
        # If URL already has `?__theme=`, sync it INTO localStorage
        # so a URL-driven visit updates the stored preference too.
        # Safe now that the question textbox uses on-demand Refresh
        # instead of auto-restore — a one-time reload can't wipe
        # in-flight typing because there is no in-flight typing on
        # a fresh mount.
        demo.load(
            fn=None,
            js="""() => {
                const url = new URL(window.location);
                const urlTheme = url.searchParams.get('__theme');
                if (urlTheme) {
                    try { localStorage.setItem('agent_ouija_theme', urlTheme); }
                    catch (e) {}
                    return;
                }
                let stored = null;
                try { stored = localStorage.getItem('agent_ouija_theme'); }
                catch (e) {}
                const target = (stored === 'light') ? 'light' : 'dark';
                url.searchParams.set('__theme', target);
                window.location = url.toString();
            }""",
        )

        # Question textbox draft persistence.
        #
        # Every keystroke in ao_question is saved to localStorage. On
        # page mount we DON'T auto-restore — that behaviour surprised
        # users when a stale draft from a previous session showed up
        # unbidden. Instead, a small `🔄 Refresh` button in the Ask
        # row pulls the saved draft into the textbox on demand (see
        # the button's click handler further down for the read side).
        # Draft is cleared when the user clicks any Ask button —
        # they've committed the question and don't need it back.
        demo.load(
            fn=None,
            js=r"""
            () => {
              const STORAGE_KEY = 'agent_ouija_question_draft';
              const findQ = () =>
                document.querySelector(
                  '#ao_question textarea, #ao_question input'
                );

              const attach = () => {
                const q = findQ();
                if (!q) return false;

                // Save every keystroke.
                q.addEventListener('input', () => {
                  try { localStorage.setItem(STORAGE_KEY, q.value); }
                  catch (e) {}
                });

                // Clear the draft when the user clicks any Ask button.
                for (const askId of ['ao_ask_btn', 'ao_ask_lore_btn']) {
                  const host = document.getElementById(askId);
                  if (!host) continue;
                  const btn = host.querySelector('button') || host;
                  btn.addEventListener('click', () => {
                    try { localStorage.removeItem(STORAGE_KEY); }
                    catch (e) {}
                  });
                }
                return true;
              };

              // Gradio may mount the textbox after this hook fires; retry
              // for a couple seconds so the listeners land regardless.
              if (attach()) return;
              let tries = 0;
              const t = setInterval(() => {
                if (attach() || ++tries > 40) clearInterval(t);
              }, 100);
            }
            """,
        )

        with gr.Tabs(elem_id="ao_main_tabs") as ao_tabs:

            with gr.Tab("Agent Ouija", id="agent_ouija"):
                # Four dropdown labels but only three numeric tiers — the
                # two Medium variants share tier index 1 (both use Serper
                # web context for peers). Difference is default check
                # state on the Models section: Agile pre-checks clouds
                # only (fast); Synergistic pre-checks clouds + locals
                # (slower but more perspectives).
                # Prefix each level label with a coloured flame emoji
                # so the dropdown reads at a glance without opening
                # the dropdown or reading the description underneath.
                # 🟡 yellow = low, 🟠 orange = medium/agile, 🔴 red =
                # medium/synergistic, 🔵 blue = high. Combined with
                # the fire emoji to reinforce "energy". Every string
                # comparison downstream still resolves because it
                # references MYSTICISM_LEVELS[N] by index.
                MYSTICISM_LEVELS = [
                    "🟡🔥 Low Energy — Locals Only",
                    "🟠🔥 Medium Energy — Agile",
                    "🔴🔥 Medium Energy — Synergistic",
                    "🔵🔥 High Energy — Native Web Search",
                ]
                _EXEC_TIER_LABELS = ["Junior Execs", "Senior Execs", "The C-Suite"]

                LEVEL_DESCRIPTIONS = {
                    MYSTICISM_LEVELS[0]:
                        "🕯 **Low Energy** — Locals only, no web search. Free but slower.",
                    MYSTICISM_LEVELS[1]:
                        "🌫 **Medium Energy — Agile** — Cloud models pre-checked, "
                        "locals unchecked (fast). All checkboxes are still "
                        "editable — mix as you like. Serper web search "
                        "injected into every checked model's prompt.",
                    MYSTICISM_LEVELS[2]:
                        "🌫 **Medium Energy — Synergistic** — Both clouds AND "
                        "locals pre-checked (slower, more perspectives). "
                        "All checkboxes editable. Serper web search "
                        "injected into every checked model.",
                    MYSTICISM_LEVELS[3]:
                        "🔥 **High Energy** — Only models with native web-search tools "
                        "(Claude, GPT-4o, Gemini). Real-time provider-side search.",
                }

                LOCAL_KEYS  = [m for m in ARENA_MODELS
                               if m in AVAILABLE_MODELS
                               and AVAILABLE_MODELS[m].get("type") == "local"]
                CLOUD_KEYS  = [m for m in ARENA_MODELS
                               if m in AVAILABLE_MODELS
                               and AVAILABLE_MODELS[m].get("type") == "cloud"]
                WEB_CAPABLE = {m for m in ARENA_MODELS
                               if m in AVAILABLE_MODELS
                               and AVAILABLE_MODELS[m].get("web_search", False)}

                def _level_index(level_str: str) -> int:
                    """Numeric tier: 0=Low, 1=Medium (both variants), 2=High.
                    Executive-tier constraint + runner dispatch both use
                    this — the two Medium variants collapse into the same
                    index because they route through the same runner and
                    have the same eligibility rules."""
                    if level_str == MYSTICISM_LEVELS[0]:
                        return 0
                    if level_str in (MYSTICISM_LEVELS[1], MYSTICISM_LEVELS[2]):
                        return 1
                    if level_str == MYSTICISM_LEVELS[3]:
                        return 2
                    return 0

                def _model_eligible_at(model_key: str, level_str: str) -> bool:
                    """A model is pickable as a PEER at the given energy level.
                    Low = locals only. Medium (either variant) = all models.
                    High = only web-search-capable models."""
                    idx  = _level_index(level_str)
                    cfg  = AVAILABLE_MODELS.get(model_key, {})
                    kind = cfg.get("type")
                    if idx == 0:  return kind == "local"
                    if idx == 1:  return True
                    return model_key in WEB_CAPABLE  # High Energy

                def _default_check_at(model_key: str, level_str: str) -> bool:
                    """Whether the checkbox should default to CHECKED when
                    tier switches to `level_str`. Only applies at the tier
                    transition — user can freely uncheck/check afterwards."""
                    cfg = AVAILABLE_MODELS.get(model_key, {})
                    is_local = cfg.get("type") == "local"
                    if level_str == MYSTICISM_LEVELS[0]:      # Low
                        return is_local
                    if level_str == MYSTICISM_LEVELS[1]:      # Medium - Agile
                        return not is_local                    # clouds checked, locals unchecked
                    if level_str == MYSTICISM_LEVELS[2]:      # Medium - Synergistic
                        return True                            # all checked
                    # High Energy — only web-capable clouds
                    return model_key in WEB_CAPABLE

                def _default_level() -> str:
                    """Pick the initial Mysticism Level. Prefers the
                    last one the user picked (persisted via prefs.py)
                    when that value maps to a current level; falls
                    back to MYSTICISM_LEVELS[0] otherwise.

                    Tolerant of label churn — when we added the
                    🟡🔥 / 🟠🔥 / 🔴🔥 / 🔵🔥 prefixes and swapped a
                    hyphen for an em-dash, exact-match saves like
                    'Medium Energy - Agile' stopped matching. Now
                    we fuzzy-match by stripping non-ASCII (emojis
                    + em-dashes) and comparing lower-cased. On a
                    fuzzy hit we also migrate the pref forward so
                    the next launch is an exact match."""
                    saved = prefs.load().get("energy_level")
                    if not saved:
                        return MYSTICISM_LEVELS[0]
                    if saved in MYSTICISM_LEVELS:
                        return saved

                    def _canon(s: str) -> str:
                        # ASCII-only, lowercase, dash-normalised.
                        # "🟠🔥 Medium Energy — Agile" → "medium energy - agile"
                        # Order matters: replace the em-dash with a
                        # hyphen BEFORE stripping non-ASCII (em-dash
                        # is ord 8212, so the strip would eat it
                        # otherwise and leave a double space in the
                        # spot where the delimiter used to be).
                        pre = (s or "").replace("—", "-")
                        stripped = "".join(
                            c for c in pre if ord(c) < 128
                        )
                        # Collapse repeated whitespace so an emoji-
                        # adjacent space + regular space don't leave
                        # gaps like "medium energy  agile".
                        return " ".join(stripped.lower().split())

                    saved_key = _canon(saved)
                    for level in MYSTICISM_LEVELS:
                        if _canon(level) == saved_key:
                            # Migrate the pref so future launches
                            # match exactly and skip this branch.
                            try:
                                prefs.update(energy_level=level)
                            except Exception as e:
                                log.warning(
                                    "prefs migration failed: %s", e,
                                )
                            return level
                    return MYSTICISM_LEVELS[0]

                def _default_boardroom_on() -> bool:
                    # Second arg to .get is the fallback when the
                    # key is absent from prefs.json (fresh install).
                    # True as of this build — Executive Boardroom
                    # is the intended default flow for a new user.
                    # Anyone who explicitly toggled it OFF gets
                    # False from their saved prefs; only truly-
                    # never-touched-it prefs see the new True.
                    return bool(prefs.load().get("boardroom_toggle", True))

                def _default_sr_with_jr() -> bool:
                    return bool(prefs.load().get("sr_with_jr", False))

                # ── Header row: + New Seance + current seance name ────────
                # State: current seance name. "New Seance" until saved or
                # loaded. Rename in the Seances accordion updates this.
                ao_seance_name_state = gr.State("New Seance")
                # Path (filename in data/seances/) of the currently loaded
                # seance. Empty when the session is a fresh "New Seance" —
                # no auto-save. When a seance is loaded from disk (or the
                # user Save-Seances one), the path lives here and every
                # subsequent Ask auto-writes into it.
                ao_seance_path_state = gr.State("")
                # Thread accumulation: everything BELOW the current
                # question. Prepended to on every fresh Ask.
                ao_thread_below_state = gr.State("")
                # The question currently at the top of the response
                # window (used to render the thread + to shift into
                # thread_below on the next Ask).
                ao_current_q_state = gr.State("")
                # Structured conversation history — list of {q, a, ts}
                # dicts (oldest first). Last CONVERSATION_HISTORY_WINDOW
                # entries are injected into every model call as
                # alternating user/assistant messages so follow-ups
                # ("we're under 25") resolve against prior context.
                # Persisted into the seance JSON alongside s_results.
                ao_history_state = gr.State([])

                with gr.Row():
                    ao_new_seance_btn = gr.Button(
                        "＋ New Seance", scale=0, size="sm",
                        variant="secondary",
                    )
                    # Recover last unsaved chat from the quick-chat
                    # autosave file (_autosave_new_seance.json). Hidden
                    # by default; demo.load below shows it only when a
                    # recent autosave exists — surfaces the recovery
                    # affordance only when there's something to recover.
                    # Fixes the mobile pain point: iOS Safari suspends
                    # background tabs mid-response, and the user returns
                    # to a fresh session losing their chat. The autosave
                    # is written server-side after each Ask, so the
                    # response IS preserved; this button pulls it back.
                    ao_recover_btn = gr.Button(
                        "🔄 Recover last chat",
                        scale=0, size="sm",
                        variant="primary",
                        visible=False,
                        elem_id="ao_recover_btn",
                    )
                    ao_seance_name_display = gr.Markdown(
                        "🕯 **Current Seance:** New Seance",
                        elem_id="ao_seance_name_display",
                    )

                # ── Section 1 — Question & Response ───────────────────────
                with gr.Accordion("💬 Question & Response", open=True):
                    ao_energy_label = gr.Markdown(
                        "<span style='opacity:0.6;font-size:0.85em'>"
                        f"🔮 Energy: {_default_level()}"
                        "</span>",
                        elem_id="ao_energy_label",
                    )
                    ao_question = gr.Textbox(
                        label="Question",
                        placeholder="What would you like to know?",
                        lines=3, max_lines=10,
                        elem_id="ao_question",
                    )
                    # Mic input — only rendered when scripts/stt.py reports
                    # a usable backend (API key present / dep installed).
                    # Record, stop, we transcribe, text lands in the
                    # question box; user still has to click Ask so a
                    # misheard phrase can be corrected before firing.
                    # Mobile Safari/Chrome require HTTPS → use Tailscale
                    # serve to front localhost:7860 (SETUP.md §4b).
                    _ao_mic_enabled = _stt_available()
                    ao_mic = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎙️ Speak your question",
                        visible=_ao_mic_enabled,
                        elem_id="ao_mic",
                    )
                    with gr.Row():
                        ao_ask_btn = gr.Button(
                            "Ask For Revelations", variant="primary", scale=3,
                            elem_id="ao_ask_btn",
                        )
                        # Same Ask flow, but with the STRICT GROUNDING
                        # directive layered into every model call —
                        # answers come only from RAG / doc / web
                        # context supplied to the model, or the model
                        # returns "Not in the lore." Tooltip is
                        # attached by a demo.load(js=…) shim below
                        # because Gradio 6 doesn't expose a title prop
                        # on Button.
                        ao_ask_lore_btn = gr.Button(
                            "📖 Ask the Lore", variant="secondary", scale=2,
                            elem_id="ao_ask_lore_btn",
                        )
                        ao_cancel_btn = gr.Button(
                            "✋ Cancel", variant="stop", scale=1,
                        )
                        # Upload lives here (was in Control Center) so users
                        # can attach files without opening the accordion.
                        # on_upload already routes documents → docs/ and
                        # images/videos → docs/media/ via classify_file.
                        ao_upload_btn = gr.UploadButton(
                            "📂 Upload Files",
                            file_count="multiple",
                            file_types=[".md", ".txt", ".pdf", ".docx", ".doc",
                                        ".html", ".htm", ".json", ".csv", ".rtf",
                                        "image", "video"],
                            scale=1,
                        )
                        ao_save_seance_btn = gr.Button(
                            "💾 Save Seance", variant="secondary", scale=1,
                        )
                    ao_status = gr.Markdown("")

                    with gr.Group():
                        ao_model_label = gr.Markdown("—")
                        # Markdown (not Textbox) so we can color the
                        # question lines differently from the answer.
                        # _fmt_thread_top wraps the question in a
                        # blockquote whose colour is set by the CSS
                        # rules on #ao_response in the head styles.
                        # Answer text renders in the default theme
                        # colour. Scroll behaviour + max-height are
                        # also handled in the CSS block.
                        ao_response = gr.Markdown(
                            value="",
                            label="Response",
                            elem_id="ao_response",
                            elem_classes=["ao-response-md"],
                        )
                        # Inline Add-to-RAG affordance — sits right under
                        # the response so the user can preserve the good
                        # answer with a single click, without opening
                        # the Scoring Controls accordion. Wires to the
                        # same on_ao_add_to_rag handler as the
                        # accordion's "Add To RAG (no score)" button.
                        with gr.Row():
                            ao_inline_rag_btn = gr.Button(
                                "💾 Add To RAG",
                                size="sm",
                                variant="secondary",
                                scale=0,
                            )
                            # Reference any file from docs/ or
                            # docs/media/ by name in the current
                            # question, click this, and it ships as
                            # an email attachment via the SMTP config
                            # in `.env`. Covers the image-edit output
                            # too (just reference the timestamped
                            # filename). Behind an SMTP config;
                            # degrades to a "not configured" message
                            # when the env vars are missing.
                            ao_send_file_btn = gr.Button(
                                "📧 Send File",
                                size="sm",
                                variant="secondary",
                                scale=0,
                            )
                            # Manual draft restore — pulls the last
                            # auto-saved question (localStorage) into
                            # the textbox. Companion to the auto-save
                            # keystroke handler set up in demo.load
                            # further up. Placed alongside Add To RAG
                            # and Send File so all three post-response
                            # affordances live in the same row.
                            ao_refresh_draft_btn = gr.Button(
                                "🔄 Refresh",
                                size="sm",
                                variant="secondary",
                                scale=0,
                                elem_id="ao_refresh_draft_btn",
                            )
                        ao_response_audio = gr.Audio(
                            label="Agent Ouija speaks",
                            autoplay=True, visible=False, interactive=False,
                        )
                    # Inline escalation prompts — shown after an executive
                    # slide runs; hidden otherwise. When a Junior exec just
                    # produced an answer, both "→ Senior" and "→ C-Suite"
                    # buttons show. Senior → only C-Suite. C-Suite → neither.
                    with gr.Row(visible=False) as ao_escalate_row:
                        ao_escalate_senior_btn = gr.Button(
                            "🔼 Would you like the Senior Executives to look at this?",
                            variant="secondary",
                            visible=False,
                        )
                        ao_escalate_csuite_btn = gr.Button(
                            "🔼 Would you like the C-Suite to look at this?",
                            variant="secondary",
                            visible=False,
                        )
                    ao_last_exec_tier_state = gr.State("")   # "" | Low/Med/High

                # ── Section 2 — Executive Boardroom ──────────────────────
                with gr.Accordion("👑 Executive Boardroom", open=False):
                    gr.Markdown(
                        "**Two ways to use boardroom mode:**\n"
                        "- **Checkbox ON** → every **Ask For Revelations** "
                        "returns one synthesized executive answer instead of "
                        "individual model responses. Uses whichever tier "
                        "button was clicked last (defaults to the lowest "
                        "tier your Mysticism Level allows if none has been "
                        "clicked yet).\n"
                        "- **Checkbox OFF (default)** → individual model "
                        "answers appear in the slideshow as normal. You can "
                        "still trigger a one-off boardroom answer by "
                        "clicking any tier button below — that fires the "
                        "current question through that tier's executive."
                    )
                    ao_boardroom_toggle = gr.Checkbox(
                        value=_default_boardroom_on(),
                        label="🌐 Always Use Boardroom Mode "
                              "(every Ask → single synthesized answer)",
                    )
                    ao_sr_with_jr_toggle = gr.Checkbox(
                        value=_default_sr_with_jr(),
                        label="Sr. Execs. with selected Jr. Execs (local models) "
                              "— when Senior tier is used, include locals as peers "
                              "instead of clouds-only",
                    )
                    gr.Markdown("**Executive Tier** — must be ≥ Mysticism Level. Clicking a tier button fires the question through that tier's executive right away.")
                    with gr.Row():
                        ao_exec_junior_btn = gr.Button(
                            _EXEC_TIER_LABELS[0], variant="secondary",
                            elem_id="ao_exec_junior_btn",
                        )
                        ao_exec_senior_btn = gr.Button(
                            _EXEC_TIER_LABELS[1], variant="secondary",
                            elem_id="ao_exec_senior_btn",
                        )
                        ao_exec_csuite_btn = gr.Button(
                            _EXEC_TIER_LABELS[2], variant="secondary",
                            elem_id="ao_exec_csuite_btn",
                        )
                    # Match the exec tier + dropdown to whichever Mysticism
                    # Level the user is restoring to. Previously this was
                    # hardcoded to Low, which caused a crash on locals-only
                    # rigs when a stale "Claude Sonnet" pref met a Low-tier
                    # locals-only choice list (Gradio's Dropdown preprocess
                    # rejects any value not in current choices). Aligning
                    # initial state with the saved level eliminates the
                    # mismatch class entirely.
                    _initial_exec_level = _default_level()
                    ao_exec_tier_state = gr.State(_initial_exec_level)
                    ao_exec_tier_display = gr.Markdown(
                        f"Selected: **{_EXEC_TIER_LABELS[_level_index(_initial_exec_level)]}** "
                        f"({_initial_exec_level})"
                    )
                    # Default executive persisted at data/default_executive.txt.
                    # Presence + value pre-selects the dropdown on UI load.
                    DEFAULT_EXEC_PATH = DATA_DIR / "default_executive.txt"

                    def _load_default_exec() -> Optional[str]:
                        if DEFAULT_EXEC_PATH.exists():
                            try:
                                v = DEFAULT_EXEC_PATH.read_text(encoding="utf-8").strip()
                                return v or None
                            except OSError:
                                return None
                        return None

                    # Compute the initial exec choices from the SAVED level's
                    # tier, not a hardcoded Low. Mirrors _exec_choices_for
                    # (defined later — same math, kept in sync).
                    def _exec_keys_for_tier(tier: str) -> list:
                        i = _level_index(tier)
                        if i == 0:
                            return list(LOCAL_KEYS)
                        if i == 1:
                            return list(LOCAL_KEYS) + list(CLOUD_KEYS)
                        return [m for m in CLOUD_KEYS if m in WEB_CAPABLE]

                    _saved_default_exec = _load_default_exec()
                    _initial_exec_choices = [
                        AVAILABLE_MODELS[m]["display_name"]
                        for m in _exec_keys_for_tier(_initial_exec_level)
                    ]
                    # Priority: explicit default_executive.txt → last-used
                    # from prefs → Claude Sonnet fallback (Medium+ only) →
                    # first available. Each candidate must be IN choices
                    # so the initial value can never mismatch the initial
                    # choices list.
                    _initial_exec_value: Optional[str] = None
                    if _saved_default_exec and _saved_default_exec in _initial_exec_choices:
                        _initial_exec_value = _saved_default_exec
                    else:
                        _last_used = prefs.load().get("last_executive")
                        if _last_used and _last_used in _initial_exec_choices:
                            _initial_exec_value = _last_used
                        else:
                            _claude_display = AVAILABLE_MODELS.get("claude-sonnet", {}).get("display_name")
                            if (_level_index(_initial_exec_level) >= 1
                                    and _claude_display in _initial_exec_choices):
                                _initial_exec_value = _claude_display
                            elif _initial_exec_choices:
                                _initial_exec_value = _initial_exec_choices[0]

                    # allow_custom_value=True prevents Gradio from raising
                    # when a stale/mismatched value slips through from the
                    # client (race between choices-update and value-update
                    # on tier switches). Handlers below snap invalid values
                    # to a valid one, so the app degrades gracefully instead
                    # of crashing the queue.
                    ao_exec_model = gr.Dropdown(
                        choices=_initial_exec_choices,
                        value=_initial_exec_value,
                        label="👑 Executive Model",
                        allow_custom_value=True,
                    )
                    ao_default_exec_toggle = gr.Checkbox(
                        value=(_saved_default_exec == _initial_exec_value
                               and _initial_exec_value is not None),
                        label="💾 Make this model the Default Executive (remembered across sessions)",
                    )
                    # Groq-as-executive warning. Groq's free tier caps
                    # token throughput per minute per model — a
                    # boardroom synthesis over multiple peer responses
                    # blows past that limit and lands on rate-limit
                    # or model-not-found errors mid-synthesis. Warning
                    # is hidden by default; on_exec_model_change flips
                    # visibility whenever the user picks Groq.
                    _groq_display = (
                        AVAILABLE_MODELS.get("groq", {}).get("display_name")
                        or "Groq"
                    )
                    ao_groq_exec_warning = gr.Markdown(
                        "⚠️ **Heads-up on `" + _groq_display + "`:** "
                        "Groq's free-tier caps at 8k tokens per minute "
                        "per model, which this app blows through easily "
                        "on exec synthesis (peer answers + question + "
                        "history all sum in). Also unreliable as a "
                        "**peer** when Enhanced Persona or Private "
                        "Corpus is on — the persona stack + corpus "
                        "chunks push a normal peer call past the cap "
                        "too. Testing shows Groq handles peer calls "
                        "fine with both toggles OFF (Persona tab); use "
                        "**Claude Sonnet** or **GPT-4o** for exec "
                        "regardless, or upgrade to Groq Dev Tier for "
                        "higher TPM if you like the LPU speed.",
                        visible=(_initial_exec_value == _groq_display),
                    )

                # ── Section 2b — The Assistant (Calendar + News) ─────────
                # External APIs the boardroom can lean on, in their own
                # collapsible so the Executive Boardroom section stays
                # focused on tier + exec-model selection. Google Calendar
                # (read/add/delete) + NewsAPI (daily pulls into RAG) both
                # live here now. Full News controls were moved from the
                # Data tab; Calendar was moved out of Boardroom.
                with gr.Accordion("🤖 The Assistant", open=False):
                    gr.Markdown(
                        "External APIs the boardroom can lean on. Two "
                        "surfaces:\n"
                        "- **Google Calendar** — add / delete events via "
                        "a plain form (no LLM tool-use). Read-my-"
                        "calendar questions work via the same RAG "
                        "pipeline as documents — click **Refresh "
                        "Calendar into RAG** below and ask.\n"
                        "- **NewsAPI daily pulls** — cache Sports / "
                        "World / National headlines into `docs/news/` "
                        "so any model with `rag: True` can quote or "
                        "summarise recent news.\n\n"
                        "**One-time setup:**\n"
                        "1. Get a NewsAPI key at https://newsapi.org "
                        "and add `NEWSAPI_KEY=…` to `.env`.\n"
                        "2. Create a Google Cloud project → enable "
                        "*Google Calendar API* → create an OAuth 2.0 "
                        "*Desktop* client → download the client-"
                        "secrets JSON as `data/gcal_credentials.json`.\n"
                        "3. Click **🔐 Authorize Google Calendar** "
                        "below (while at Brainy / RDP'd in) to complete "
                        "the OAuth handshake."
                    )

                    # Try importing gcal lazily so the Assistant section
                    # renders even if google-api-python-client isn't
                    # installed yet (the buttons just report the missing
                    # dep in their status output).
                    try:
                        from gcal import (
                            authorize_interactive as _gcal_authorize,
                            is_authorized as _gcal_is_authorized,
                            list_upcoming_events as _gcal_list_events,
                            refresh_calendar_cache as _gcal_refresh_cache,
                            add_event as _gcal_add_event,
                            delete_event as _gcal_delete_event,
                            format_event_window as _gcal_fmt_window,
                        )
                        _gcal_available = True
                    except ImportError as e:
                        log.warning("gcal module unavailable: %s", e)
                        _gcal_available = False
                        def _gcal_authorize():
                            return {"status": "error",
                                    "message": "gcal module not importable"}
                        def _gcal_is_authorized(): return False
                        def _gcal_list_events(**k): return {"error": "gcal module not importable"}
                        def _gcal_refresh_cache(**k): return {"error": "gcal module not importable"}
                        def _gcal_add_event(**k):   return {"error": "gcal module not importable"}
                        def _gcal_delete_event(id): return {"error": "gcal module not importable"}
                        def _gcal_fmt_window(e):    return "?"

                    def _gcal_status():
                        if not _gcal_available:
                            return "🕯 Calendar: module load failed (see logs)."
                        if not _gcal_is_authorized():
                            return ("🕯 Calendar: **not authorised** — "
                                    "click **🔐 Authorize Google Calendar** "
                                    "below while sitting at Brainy (RDP is "
                                    "fine). Google's Testing-mode OAuth "
                                    "expires refresh tokens every 7 days, "
                                    "so this re-auth click is a weekly "
                                    "ritual until the app is published.")
                        return "🕯 Calendar: 🟢 authorised."

                    gcal_status_md = gr.Markdown(_gcal_status())

                    with gr.Row():
                        gcal_authorize_btn = gr.Button(
                            "🔐 Authorize Google Calendar",
                            variant="secondary",
                        )
                        gcal_refresh_btn = gr.Button(
                            "🔄 Refresh Calendar into RAG",
                            variant="primary",
                        )
                        gcal_list_btn = gr.Button(
                            "📋 Show upcoming (below)",
                        )
                    gcal_result_md = gr.Markdown("")

                    with gr.Accordion("➕ Add Event", open=False):
                        gr.Markdown(
                            "Fill in title + date + start/end times. Dates "
                            "use `YYYY-MM-DD`, times use 24-hour `HH:MM`. "
                            "Timezone follows your OS locale."
                        )
                        _today_iso = datetime.now().strftime("%Y-%m-%d")
                        _now_hm    = datetime.now().strftime("%H:%M")
                        gcal_add_title = gr.Textbox(
                            label="Title *",
                            placeholder="e.g. Dentist appointment",
                        )
                        with gr.Row():
                            gcal_add_date = gr.Textbox(
                                label="Date (YYYY-MM-DD) *",
                                value=_today_iso,
                            )
                            gcal_add_start = gr.Textbox(
                                label="Start (HH:MM) *",
                                value=_now_hm,
                            )
                            gcal_add_end = gr.Textbox(
                                label="End (HH:MM) *",
                                value=_now_hm,
                            )
                        gcal_add_location = gr.Textbox(
                            label="Location (optional)",
                        )
                        gcal_add_desc = gr.Textbox(
                            label="Description (optional)",
                            lines=2, max_lines=6,
                        )
                        gcal_add_btn = gr.Button(
                            "➕ Add Event", variant="primary",
                        )
                        gcal_add_result = gr.Markdown("")

                    with gr.Accordion("🗑 Delete Event", open=False):
                        gr.Markdown(
                            "Pick from the upcoming events, then type "
                            "`YES` in the confirmation box to enable "
                            "the delete."
                        )
                        with gr.Row():
                            gcal_del_dropdown = gr.Dropdown(
                                choices=[],
                                label="Event to delete",
                                interactive=True,
                                scale=3,
                            )
                            gcal_del_refresh_btn = gr.Button(
                                "🔄", scale=1,
                            )
                        gcal_del_confirm = gr.Textbox(
                            label="Type YES to confirm",
                            placeholder="YES",
                        )
                        gcal_del_btn = gr.Button(
                            "🗑 Delete", variant="stop",
                        )
                        gcal_del_result = gr.Markdown("")

                    # ── Handlers ──────────────────────────────────────

                    def on_gcal_refresh_cache(progress=gr.Progress()):
                        progress(0, desc="Contacting Google Calendar…")
                        r = _gcal_refresh_cache()
                        if "error" in r:
                            return (f"❌ {r['error']}", _gcal_status())
                        # Re-ingest so the cached events land in ChromaDB.
                        progress(0.7, desc="Re-ingesting…")
                        try:
                            run_ingestion()
                        except Exception as e:
                            log.warning("Re-ingest after calendar refresh failed: %s", e)
                        progress(1.0, desc="Done")
                        return (
                            f"✓ Refreshed — **{r['events']}** upcoming event(s) "
                            f"cached to `{r['path']}` and ingested into RAG.",
                            _gcal_status(),
                        )

                    def on_gcal_list_upcoming():
                        r = _gcal_list_events()
                        if "error" in r:
                            return f"❌ {r['error']}"
                        events = r.get("events", [])
                        if not events:
                            return "_No upcoming events._"
                        # Wrap the datetime window in backticks so it
                        # renders in a monospace font — dates + times
                        # line up cleanly and are much easier to scan.
                        lines = [f"**{len(events)} upcoming event(s):**", ""]
                        for e in events:
                            lines.append(
                                f"- `{_gcal_fmt_window(e)}` — "
                                f"**{e.get('summary', '(no title)')}**"
                            )
                        return "\n".join(lines)

                    def on_gcal_add(title, date_iso, start_hm, end_hm,
                                     location, description):
                        r = _gcal_add_event(
                            title=title, date_iso=date_iso,
                            start_hm=start_hm, end_hm=end_hm,
                            description=description, location=location,
                        )
                        if "error" in r:
                            return f"❌ {r['error']}"
                        link = (f" · [open]({r['htmlLink']})"
                                if r.get("htmlLink") else "")
                        return (f"✓ Added **{r.get('title')}** "
                                f"({r.get('when')}){link}.")

                    def _gcal_delete_choices():
                        """Build (label, id) pairs for the delete dropdown."""
                        r = _gcal_list_events()
                        if "error" in r:
                            return []
                        events = r.get("events", [])
                        choices = []
                        for e in events:
                            title = e.get("summary", "(no title)")
                            eid   = e.get("id", "")
                            if eid:
                                choices.append(
                                    (f"{_gcal_fmt_window(e)} — {title}", eid)
                                )
                        return choices

                    def on_gcal_del_refresh():
                        choices = _gcal_delete_choices()
                        value = choices[0][1] if choices else None
                        return gr.update(choices=choices, value=value)

                    def on_gcal_delete(event_id, confirm):
                        if (confirm or "").strip().upper() != "YES":
                            return "❌ Type **YES** in the confirmation box first."
                        if not event_id:
                            return "❌ Pick an event to delete."
                        r = _gcal_delete_event(event_id)
                        if "error" in r:
                            return f"❌ {r['error']}"
                        return f"🗑 Deleted event `{r.get('deleted', event_id)}`."

                    def on_gcal_authorize():
                        """Deliberate click → run the OAuth browser popup
                        on Brainy. Blocks the UI thread while the popup
                        is open, which is fine because the USER kicked
                        it off knowing they're at Brainy. Never runs on
                        its own — that's the point of moving away from
                        the auto-popup in _get_service()."""
                        r = _gcal_authorize()
                        if r.get("status") == "success":
                            return (f"✓ {r.get('message', 'Authorized.')}",
                                    _gcal_status())
                        return (f"❌ {r.get('message', 'Auth failed.')}",
                                _gcal_status())

                    gcal_authorize_btn.click(
                        on_gcal_authorize,
                        outputs=[gcal_result_md, gcal_status_md],
                    )
                    gcal_refresh_btn.click(
                        on_gcal_refresh_cache,
                        outputs=[gcal_result_md, gcal_status_md],
                    )
                    gcal_list_btn.click(
                        on_gcal_list_upcoming,
                        outputs=[gcal_result_md],
                        queue=False,
                    )
                    gcal_add_btn.click(
                        on_gcal_add,
                        inputs=[gcal_add_title, gcal_add_date,
                                gcal_add_start, gcal_add_end,
                                gcal_add_location, gcal_add_desc],
                        outputs=[gcal_add_result],
                    )
                    gcal_del_refresh_btn.click(
                        on_gcal_del_refresh,
                        outputs=[gcal_del_dropdown],
                        queue=False,
                    )
                    gcal_del_btn.click(
                        on_gcal_delete,
                        inputs=[gcal_del_dropdown, gcal_del_confirm],
                        outputs=[gcal_del_result],
                    )

                    # ── 📰 News Pulls (NewsAPI → RAG) ─────────────────
                    # Moved here from the Data tab so both Assistant
                    # APIs sit together. Cache Sports / World /
                    # National headlines into docs/news/; auto re-
                    # ingest so the new files land in ChromaDB
                    # immediately, not on next launch.
                    with gr.Accordion("📰 News Pulls", open=False):
                        from news_puller import (
                            pull_sports_news, pull_world_news,
                            pull_national_news, pull_all_news,
                            state_summary, credit_estimate,
                        )
                        gr.Markdown(
                            "Pull the latest news from NewsAPI and "
                            "cache it into `docs/news/` so the main "
                            "RAG picks it up on the next ingest — "
                            "any model with `rag: True` can then "
                            "quote / summarise recent headlines. Set "
                            "`NEWSAPI_KEY` in `.env` first."
                        )
                        gr.Markdown(credit_estimate())
                        news_state_md = gr.Markdown(state_summary())
                        with gr.Row():
                            news_sports_btn   = gr.Button("⚾ Sports (baseball)", variant="primary")
                            news_world_btn    = gr.Button("🌍 World",             variant="primary")
                            news_national_btn = gr.Button("🇺🇸 National",          variant="primary")
                            news_all_btn      = gr.Button("📰 Pull All (3 credits)",
                                                           variant="secondary")
                        news_result_md = gr.Markdown("")

                        def _news_pull_wrap(fn):
                            """Wrap a pull_* function to return (result_md, state_md)."""
                            def handler(progress=gr.Progress()):
                                progress(0, desc="Contacting NewsAPI…")
                                r = fn()
                                progress(1.0, desc="Done")
                                # Kick a re-ingest so the new files land in
                                # ChromaDB immediately, not on next launch.
                                try:
                                    run_ingestion()
                                except Exception as e:
                                    log.warning("Re-ingest after news pull failed: %s", e)
                                return (_format_news_result(r), state_summary())
                            return handler

                        def _news_pull_all(progress=gr.Progress()):
                            progress(0, desc="Pulling 3 categories (3 credits)…")
                            r = pull_all_news()
                            progress(0.9, desc="Re-ingesting…")
                            try:
                                run_ingestion()
                            except Exception as e:
                                log.warning("Re-ingest after news pull failed: %s", e)
                            progress(1.0, desc="Done")
                            return (_format_news_result(r), state_summary())

                        def _format_news_result(r):
                            if isinstance(r, dict) and "articles" in r:
                                if r.get("error"):
                                    return f"❌ **{r.get('category', '?')}**: {r['error']}"
                                return (f"✓ **{r.get('category', '?')}** — "
                                        f"{r['articles']} article(s) written to "
                                        f"`{r.get('path', '?')}`.")
                            # dict of category → result
                            if isinstance(r, dict):
                                lines = []
                                for cat, sub in r.items():
                                    if sub.get("error"):
                                        lines.append(f"❌ **{cat}**: {sub['error']}")
                                    else:
                                        lines.append(
                                            f"✓ **{cat}** — {sub.get('articles', 0)} "
                                            "article(s)"
                                        )
                                return "\n\n".join(lines)
                            return "?"

                        news_sports_btn.click(
                            _news_pull_wrap(pull_sports_news),
                            outputs=[news_result_md, news_state_md],
                        )
                        news_world_btn.click(
                            _news_pull_wrap(pull_world_news),
                            outputs=[news_result_md, news_state_md],
                        )
                        news_national_btn.click(
                            _news_pull_wrap(pull_national_news),
                            outputs=[news_result_md, news_state_md],
                        )
                        news_all_btn.click(
                            _news_pull_all,
                            outputs=[news_result_md, news_state_md],
                        )

                # ── Section 3 — Scoring Controls ─────────────────────────
                with gr.Accordion("🎯 Scoring Controls", open=False):
                    ao_score_radio = gr.Radio(
                        choices=score_choices,
                        label="Score This Response",
                        interactive=True, value=None,
                    )
                    with gr.Row():
                        ao_score_btn     = gr.Button("Score And Next Response",
                                                      interactive=False)
                        ao_score_rag_btn = gr.Button("Score + Add To RAG",
                                                      interactive=False)
                        ao_rag_only_btn  = gr.Button("Add To RAG (no score)")
                        ao_retry_btn     = gr.Button("Retry Scoring",
                                                      interactive=False)
                    with gr.Row():
                        ao_prev_btn = gr.Button("◀ Previous")
                        ao_next_btn = gr.Button("Next ▶")
                        ao_skip_btn = gr.Button("Skip To New Prompt")
                    ao_blind_toggle = gr.Checkbox(
                        value=False, label="Blind mode (hide model name)",
                    )

                # ── Section 4 — The Models ───────────────────────────────
                # Initial per-model checkbox state must reflect the
                # persisted Mysticism Level, not a hardcoded Low-Energy
                # default. Otherwise loading with Medium-Agile checks
                # locals + greys clouds until the user re-toggles the
                # dropdown, which sends the wrong models on the first
                # Ask. Compute via the same _default_check_at +
                # _model_eligible_at helpers that on_energy_change uses
                # at runtime.
                _initial_level         = _default_level()
                _initial_level_locals  = [_default_check_at(k, _initial_level)
                                           and _model_eligible_at(k, _initial_level)
                                           for k in LOCAL_KEYS]
                _initial_level_clouds  = [_default_check_at(k, _initial_level)
                                           and _model_eligible_at(k, _initial_level)
                                           for k in CLOUD_KEYS]
                _initial_local_master  = all(_initial_level_locals) if _initial_level_locals else False
                _initial_cloud_master  = all(_initial_level_clouds) if _initial_level_clouds else False

                with gr.Accordion("🤖 The Models", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Local Models")
                            ao_local_master = gr.Checkbox(
                                value=_initial_local_master,
                                label="**All Locals** (master toggle)",
                            )
                            ao_local_cbs = []
                            for k, checked in zip(LOCAL_KEYS,
                                                    _initial_level_locals):
                                cfg = AVAILABLE_MODELS[k]
                                cb = gr.Checkbox(
                                    value=checked,
                                    label=f"{cfg['display_name']} — {cfg.get('description', '')}",
                                    interactive=_model_eligible_at(k, _initial_level),
                                )
                                ao_local_cbs.append(cb)
                        with gr.Column():
                            gr.Markdown("### Cloud Models")
                            ao_cloud_master = gr.Checkbox(
                                value=_initial_cloud_master,
                                label="**All Clouds** (master toggle)",
                            )
                            ao_cloud_cbs = []
                            for k, checked in zip(CLOUD_KEYS,
                                                    _initial_level_clouds):
                                cfg = AVAILABLE_MODELS[k]
                                cb = gr.Checkbox(
                                    value=checked,
                                    label=f"{cfg['display_name']} — {cfg.get('description', '')}",
                                    interactive=_model_eligible_at(k, _initial_level),
                                )
                                ao_cloud_cbs.append(cb)

                # ── Section 5 — Control Center ───────────────────────────
                with gr.Accordion("⚙ Control Center", open=False):
                    ao_energy_dropdown = gr.Dropdown(
                        choices=MYSTICISM_LEVELS,
                        value=_default_level(),
                        label="🔮 Mysticism Level",
                    )
                    ao_level_description = gr.Markdown(
                        LEVEL_DESCRIPTIONS[_default_level()]
                    )
                    # Enhanced-persona toggle moved to the Persona tab —
                    # all persona controls live in one place now.
                    gr.Markdown("**Navigate:**")
                    with gr.Row():
                        ao_nav_stats   = gr.Button("📊 Data",          variant="secondary")
                        ao_nav_files   = gr.Button("📁 Files",         variant="secondary")
                        ao_nav_dev     = gr.Button("🛠 Dev Workshop",  variant="secondary")
                        ao_nav_pers    = gr.Button("🎭 Persona",       variant="secondary")
                    ao_voice_toggle = gr.Checkbox(
                        value=False,
                        label="🔊 Voice (Agent Ouija speaks)",
                    )

                # ── Slideshow state ──────────────────────────────────────
                s_results_ao = gr.State([])
                s_idx_ao     = gr.State(0)

                # ── Handlers ─────────────────────────────────────────────

                def _energy_label_html(level: str) -> str:
                    return (
                        "<span style='opacity:0.6;font-size:0.85em'>"
                        f"🔮 Energy: {level}"
                        "</span>"
                    )

                def _refresh_model_interactivity(level: str):
                    """Return per-checkbox gr.update(interactive=…) matching
                    tier eligibility. Non-eligible checkboxes are also
                    forced to unchecked so a stale value doesn't leak into
                    on_ask."""
                    local_updates = []
                    for k in LOCAL_KEYS:
                        eligible = _model_eligible_at(k, level)
                        checked  = _default_check_at(k, level) and eligible
                        local_updates.append(gr.update(
                            interactive=eligible,
                            value=checked,
                        ))
                    cloud_updates = []
                    for k in CLOUD_KEYS:
                        eligible = _model_eligible_at(k, level)
                        checked  = _default_check_at(k, level) and eligible
                        cloud_updates.append(gr.update(
                            interactive=eligible,
                            value=checked,
                        ))
                    return local_updates, cloud_updates

                def _exec_choices_for(exec_tier: str):
                    idx = _level_index(exec_tier)
                    if idx == 0:
                        keys = LOCAL_KEYS
                    elif idx == 1:
                        keys = LOCAL_KEYS + CLOUD_KEYS
                    else:
                        keys = [m for m in CLOUD_KEYS if m in WEB_CAPABLE]
                    return [AVAILABLE_MODELS[m]["display_name"] for m in keys]

                def _default_exec_value(exec_tier: str) -> Optional[str]:
                    """Pick the default executive model for a tier.

                    Priority order:
                    1. prefs.last_executive — the most-recent exec
                       the user actually picked (auto-saved on every
                       dropdown change; no checkbox click needed).
                       Feels like "the app remembers".
                    2. data/default_executive.txt — the explicit
                       "Make this the Default Executive" checkbox.
                       Wins over last_executive when both point at
                       valid-at-tier models; the checkbox is a
                       stronger commitment.
                    3. Claude Sonnet for Medium/High tiers (fallback
                       chosen after in-house testing showed Groq's
                       TPM cap crashes synthesis on boardroom-length
                       peer sets).
                    4. First available for Low tier.

                    Each fallback still requires the candidate to be
                    IN the current tier's choices — moving from
                    Senior to Low with Claude saved will land on a
                    local instead of a mismatched cloud."""
                    choices = _exec_choices_for(exec_tier)
                    if not choices:
                        return None

                    # 2 (checkbox default) beats 1 (last used) when
                    # BOTH exist and are valid — explicit choice
                    # wins over passive habit.
                    saved = _load_default_exec()
                    if saved and saved in choices:
                        return saved

                    # 1 — last-used from prefs.
                    last_used = prefs.load().get("last_executive")
                    if last_used and last_used in choices:
                        return last_used

                    claude_display = AVAILABLE_MODELS.get("claude-sonnet", {}).get("display_name")
                    idx = _level_index(exec_tier)
                    if idx >= 1 and claude_display in choices:
                        return claude_display
                    return choices[0]

                # Three exec-tier buttons ↔ three numeric tiers (0/1/2).
                # Mysticism has FOUR labels but two Medium variants share
                # tier index 1; Senior Execs canonically stores the Agile
                # label but treats either the same way at runtime.
                _EXEC_TIER_MYSTICISM = [
                    MYSTICISM_LEVELS[0],   # Junior  → Low
                    MYSTICISM_LEVELS[1],   # Senior  → Medium (Agile as canonical)
                    MYSTICISM_LEVELS[3],   # C-Suite → High
                ]

                def _exec_tier_button_updates(current_level: str,
                                              selected_exec_tier: str):
                    """Grey out exec-tier buttons whose tier < current peer
                    tier. Highlight the currently-selected exec tier."""
                    peer_idx = _level_index(current_level)
                    updates = []
                    for i, tier_lvl in enumerate(_EXEC_TIER_MYSTICISM):
                        eligible = i >= peer_idx
                        # Selection matches by numeric tier so either
                        # Medium variant highlights the Senior button.
                        is_selected = (
                            _level_index(tier_lvl) == _level_index(selected_exec_tier)
                        )
                        updates.append(gr.update(
                            interactive=eligible,
                            variant=("primary" if (eligible and is_selected)
                                     else "secondary"),
                        ))
                    return updates

                def on_energy_change(new_level, cur_exec_tier):
                    peer_idx = _level_index(new_level)
                    # Persist the picked level so the next launch
                    # opens on the same energy. Fused into the handler
                    # (not a .then chain) because Gradio 6 was silent-
                    # skipping the follow-up save on some rigs.
                    try:
                        prefs.update(energy_level=new_level)
                    except Exception as e:
                        log.warning("prefs.update(energy_level) failed: %s", e)
                    # Force exec tier to ≥ peer tier.
                    new_exec_tier = (cur_exec_tier
                                     if _level_index(cur_exec_tier) >= peer_idx
                                     else new_level)
                    exec_choices = _exec_choices_for(new_exec_tier)
                    local_updates, cloud_updates = _refresh_model_interactivity(new_level)
                    tier_btn_updates = _exec_tier_button_updates(new_level, new_exec_tier)
                    tier_display = (
                        f"Selected: **{_EXEC_TIER_LABELS[_level_index(new_exec_tier)]}** "
                        f"({new_exec_tier})"
                    )
                    return (
                        LEVEL_DESCRIPTIONS[new_level],
                        _energy_label_html(new_level),
                        *local_updates,
                        *cloud_updates,
                        gr.update(choices=exec_choices,
                                  value=_default_exec_value(new_exec_tier)),
                        new_exec_tier,
                        tier_display,
                        *tier_btn_updates,
                    )

                def _make_exec_tier_click(tier_str):
                    def handler(current_level):
                        exec_choices = _exec_choices_for(tier_str)
                        tier_btn_updates = _exec_tier_button_updates(current_level, tier_str)
                        tier_display = (
                            f"Selected: **{_EXEC_TIER_LABELS[_level_index(tier_str)]}** "
                            f"({tier_str})"
                        )
                        return (
                            tier_str,
                            tier_display,
                            gr.update(choices=exec_choices,
                                      value=_default_exec_value(tier_str)),
                            *tier_btn_updates,
                        )
                    return handler

                def on_local_master(master, level):
                    """Master toggle → set every eligible local child to
                    the master's value. Non-eligible children stay off."""
                    return [
                        gr.update(value=(master and _model_eligible_at(k, level)),
                                  interactive=_model_eligible_at(k, level))
                        for k in LOCAL_KEYS
                    ]

                def on_cloud_master(master, level):
                    return [
                        gr.update(value=(master and _model_eligible_at(k, level)),
                                  interactive=_model_eligible_at(k, level))
                        for k in CLOUD_KEYS
                    ]

                def _run_executive_as_slide(question, peers, exec_key, tier_str,
                                             progress, history=None,
                                             grounded=False):
                    """Run the exec on `peers` at the given tier and return a
                    slide dict that plugs into the same slideshow model as
                    peer answers."""
                    e_idx = _level_index(tier_str)
                    web_context = None
                    web_search  = False
                    if e_idx == 1:
                        web_context = serper_search(question)
                    elif e_idx == 2:
                        web_search = True
                    try:
                        er = _run_executive(
                            question, peers, exec_key,
                            web_context=web_context, web_search=web_search,
                            progress=progress, history=history,
                            grounded=grounded,
                        )
                    except Exception as e:
                        log.warning("Executive synthesis failed: %s", e)
                        return None
                    if not er.get("executive_response"):
                        return None
                    tier_label = _EXEC_TIER_LABELS[e_idx]
                    return {
                        "model_key":     exec_key,
                        "model_display": f"👑 {tier_label} — {er['executive_display']}",
                        "response":      er["executive_response"],
                        "elapsed":       er["executive_elapsed"],
                        "input_tokens":  0,
                        "output_tokens": 0,
                        "input_cost":    er["executive_input_cost"],
                        "output_cost":   er["executive_output_cost"],
                        "round_id":      (peers[0].get("round_id")
                                          if peers else None),
                        "kind":          "executive",
                        "exec_tier":     tier_str,
                    }

                def _escalation_updates(exec_tier_str: str):
                    """Return (row_visible, sr_btn_visible, cs_btn_visible)
                    tuple. Junior exec → both escalation options. Senior exec
                    → only C-Suite. C-Suite or no exec → row hidden."""
                    if not exec_tier_str:
                        return (gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False))
                    idx = _level_index(exec_tier_str)
                    if idx == 0:  # Junior exec
                        return (gr.update(visible=True),
                                gr.update(visible=True),
                                gr.update(visible=True))
                    if idx == 1:  # Senior exec
                        return (gr.update(visible=True),
                                gr.update(visible=False),
                                gr.update(visible=True))
                    return (gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False))

                def _slide_label_for(slide, blind_mode, idx):
                    if slide.get("kind") == "executive":
                        return f"{slide['model_display']}  ·  {slide['elapsed']}s"
                    return (
                        (f"Model {chr(65 + idx)}" if blind_mode
                         else slide["model_display"])
                        + f"  ·  {slide['elapsed']}s"
                    )

                _NO_ESCALATE = (gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False))

                def _fmt_thread_top(question: str, response: str) -> str:
                    """Format the current Q+response block for the response
                    Markdown. Question renders as a blockquote so the CSS
                    on #ao_response can colour it distinctly from the
                    answer. Answer text stays plain markdown so any
                    formatting the model returned (headings, lists, code
                    blocks) still renders inline."""
                    if not question:
                        return response or ""
                    # Escape any accidental markdown/HTML in the question
                    # itself so a question containing '#' or '<' doesn't
                    # turn into a heading or eat the layout. A single-
                    # line replacement of angle brackets + a leading
                    # backslash on '#' is enough for the usual cases.
                    q_safe = (question or "").replace("<", "&lt;").replace(">", "&gt;")
                    q_safe = q_safe.replace("\n", "  \n> ")  # keep multi-line qs inside the blockquote
                    return (
                        f"> 🔮 **Q:** {q_safe}\n\n"
                        f"{response or ''}"
                    )

                def _render_thread(question, response, thread_below):
                    """Assemble the response-window value: current Q+A at
                    top, prior thread below. Latest on top."""
                    top = _fmt_thread_top(question, response)
                    if thread_below:
                        return f"{top}\n\n{thread_below}".rstrip()
                    return top.rstrip()

                def _ao_reformat_display(current_q, thread_below,
                                          current_response):
                    """Called after slideshow-nav handlers to re-wrap the
                    now-set raw response in the thread format. The nav
                    handlers themselves don't know about thread state, so
                    this happens via a .then() chain."""
                    return _render_thread(current_q, current_response,
                                            thread_below)

                def _ao_empty_return(status_msg):
                    # Trailing slots must match the Ask output list exactly:
                    #   ao_current_q_state, ao_thread_below_state,
                    #   ao_question, ao_history_state.
                    return ([], 0, status_msg, "",
                            gr.update(value="—"),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            "",
                            *_NO_ESCALATE,
                            "",             # ao_current_q_state
                            "",             # ao_thread_below_state
                            gr.update(),    # ao_question (leave as-is)
                            gr.update(),    # ao_history_state (leave as-is)
                            )

                # Keywords that mark an edit intent when combined with
                # an image reference. Anything else with an image ref
                # (e.g. "describe dog.png") falls through to the normal
                # vision-attach path so a caption / analysis still works.
                _EDIT_INTENT_KEYWORDS = (
                    "edit", "add ", "remove", "delete", "change",
                    "replace", "put ", "insert", "place ",
                    "make it", "make them", "make the",
                    "erase", "fix", "combine", "merge",
                    "overlay", "cover", "draw", "paint",
                    "highlight", "circle ", "meme",
                    "caption", "text at", "text on",
                    "text to", "text overlay",
                )

                def _try_image_edit_shortcut(
                    question, prev_current_q, prev_thread_below,
                    prev_s_results, prev_s_idx, prev_history,
                ):
                    """If the question references an image AND uses an
                    edit-intent verb, run Gemini Nano Banana on it and
                    return a single-slide result. None means the caller
                    should proceed with the normal LLM fan-out."""
                    try:
                        from image_edit import (
                            IMAGE_EDIT_EXTS, edit_image, is_available,
                        )
                    except ImportError:
                        return None

                    media = _collect_media_paths(question)
                    imgs  = [p for p in media
                             if p.suffix.lower() in IMAGE_EDIT_EXTS]
                    if not imgs:
                        return None
                    q_lower = question.lower()
                    if not any(kw in q_lower for kw in _EDIT_INTENT_KEYWORDS):
                        return None

                    if not is_available():
                        return _ao_empty_return(
                            "🎨 Image edit requested but Gemini Nano Banana "
                            "isn't available. Ensure GEMINI_API_KEY is in "
                            "`.env` and `pip install google-genai Pillow` "
                            "has run."
                        )

                    input_path = imgs[0]
                    if len(imgs) > 1:
                        log.info(
                            "Image edit: multiple images referenced (%s), "
                            "using first: %s",
                            [p.name for p in imgs], input_path.name,
                        )

                    t0 = time.perf_counter()
                    output_path = edit_image(input_path, question)
                    elapsed = round(time.perf_counter() - t0, 3)
                    if output_path is None:
                        return _ao_empty_return(
                            f"🎨 Image edit failed for `{input_path.name}`. "
                            "Check the console for a Gemini error (invalid "
                            "API key, model refusal, or no image in the "
                            "response). Instruction was: "
                            f"_{question[:200]}_"
                        )

                    body = (
                        f"🎨 **Edited image saved.**\n\n"
                        f"- Source: `{input_path.name}`\n"
                        f"- Output: `{output_path.name}`\n"
                        f"- Location: `docs/media/`\n"
                        f"- Instruction: _{question}_\n\n"
                        "Refresh the Files tab or reload the page to see "
                        "the new file in the Media column."
                    )
                    slide = {
                        "model_key":     "gemini-image-edit",
                        "model_display": "🎨 Gemini Image Edit",
                        "response":      body,
                        "elapsed":       elapsed,
                        "input_tokens":  0,
                        "output_tokens": 0,
                        "input_cost":    0.0,
                        "output_cost":   0.0,
                        "error":         None,
                    }
                    new_results = [slide]
                    label = _slide_label_for(slide, False, 0)

                    # Thread accumulation identical to the normal Ask
                    # path so follow-ups still have context.
                    if prev_current_q and prev_s_results:
                        try:
                            old_resp = prev_s_results[prev_s_idx].get(
                                "response", "",
                            )
                        except (IndexError, KeyError, TypeError):
                            old_resp = ""
                        old_top = _fmt_thread_top(prev_current_q, old_resp)
                        new_thread_below = (
                            f"{old_top}\n\n{prev_thread_below or ''}"
                        ).rstrip()
                    else:
                        new_thread_below = prev_thread_below or ""

                    display = _render_thread(
                        question, body, new_thread_below,
                    )
                    new_history = list(prev_history or [])
                    new_history.append({
                        "q":  question,
                        "a":  body,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    })

                    return (
                        new_results, 0,
                        f"🎨 Image edit complete in {elapsed:.1f}s — "
                        f"`{output_path.name}` saved to docs/media/.",
                        display,
                        gr.update(value=label),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        "",                      # last_exec_tier (none)
                        *_NO_ESCALATE,
                        question,                # ao_current_q_state
                        new_thread_below,        # ao_thread_below_state
                        "",                      # ao_question (clear)
                        new_history,             # ao_history_state
                    )

                def on_ao_ask(question, level, boardroom_on, exec_tier_str,
                              exec_model_display, blind_mode, sr_with_jr,
                              prev_current_q, prev_thread_below,
                              prev_s_results, prev_s_idx, prev_history,
                              grounded,
                              *cb_values, progress=gr.Progress()):
                    """Primary ask flow. When boardroom_on, the peers run
                    but s_results holds a SINGLE executive slide (no peer
                    slideshow). When boardroom_on is off, s_results holds
                    the peers; the executive is only triggered later by
                    clicking a tier button, which then reuses these cached
                    peers.

                    Thread accumulation: the previous top entry (prior
                    question + whichever slide was showing when Ask was
                    hit) gets prepended into thread_below; the new
                    question + first slide becomes the new top. The
                    question textbox clears so the next prompt starts
                    fresh."""
                    # Clear any stale cancel flag from a previous run —
                    # user might have hit ✋ Cancel then relaunched.
                    _ao_cancel_flag.clear()
                    if not question.strip():
                        return _ao_empty_return("Enter a question first.")

                    # Image-edit short-circuit. If the question names
                    # an image in docs/media/ AND uses an edit-intent
                    # verb, we bypass the model fan-out and route to
                    # Gemini's Nano Banana image editor. Result is a
                    # single slide referencing the newly-saved file.
                    _edit_slide = _try_image_edit_shortcut(
                        question, prev_current_q, prev_thread_below,
                        prev_s_results, prev_s_idx, prev_history,
                    )
                    if _edit_slide is not None:
                        return _edit_slide

                    n_local = len(LOCAL_KEYS)
                    local_bools = cb_values[:n_local]
                    cloud_bools = cb_values[n_local:]
                    selected_keys = (
                        [k for k, on in zip(LOCAL_KEYS, local_bools) if on] +
                        [k for k, on in zip(CLOUD_KEYS, cloud_bools) if on]
                    )
                    # Sr Execs preference: with Senior tier + toggle OFF,
                    # prefer clouds. But only if there ARE clouds in the
                    # selection — otherwise dropping locals would empty
                    # the peer set and the user would see a confusing
                    # "no eligible models" error. Locals stay in when
                    # that's all the user picked.
                    if (boardroom_on
                            and _level_index(exec_tier_str) == 1
                            and not sr_with_jr):
                        cloud_only = [
                            k for k in selected_keys
                            if AVAILABLE_MODELS.get(k, {}).get("type") != "local"
                        ]
                        if cloud_only:
                            selected_keys = cloud_only
                    # Tier eligibility filter — defense against stale
                    # checkbox state.
                    selected_keys = [k for k in selected_keys
                                     if _model_eligible_at(k, level)]

                    if not selected_keys:
                        return _ao_empty_return(
                            "No eligible models selected. Check at least one "
                            "model in the 🤖 The Models section (the tier "
                            "greys out ineligible ones automatically)."
                        )
                    media_error = _media_rule_error(question, len(selected_keys))
                    if media_error:
                        return _ao_empty_return(media_error)

                    # Slice the last-N turns as inline conversation
                    # history; older turns are RAG-aged elsewhere.
                    hist = (prev_history or [])[-CONVERSATION_HISTORY_WINDOW:]

                    idx = _level_index(level)
                    if idx == 0:
                        results = run_models(question, selected_keys,
                                              progress=progress, history=hist,
                                              grounded=grounded)
                    elif idx == 1:
                        results = run_research_models(question, selected_keys,
                                                       progress=progress,
                                                       history=hist,
                                                       grounded=grounded)
                    else:
                        results = run_big3_models(question, selected_keys,
                                                    progress=progress,
                                                    history=hist,
                                                    grounded=grounded)

                    available   = [r for r in results if r.get("response") is not None]
                    unavailable = [r for r in results if r.get("response") is None]

                    # If the user hit Cancel while peers were running,
                    # discard everything and show a clear "cancelled"
                    # state instead of a half-populated slideshow.
                    if _ao_cancel_flag.is_set():
                        _ao_cancel_flag.clear()
                        return _ao_empty_return(
                            "🛑 Cancelled — no answers saved from this round."
                        )

                    if not available:
                        return _ao_empty_return("No models responded.")

                    status_parts = []
                    if grounded:
                        # Ask the Lore path — status area gets a short
                        # explanation up front so the user sees why the
                        # answer might be terse or refuse ("Not in the
                        # lore"). Replaces the earlier one-time modal;
                        # this shows on every Lore response and is
                        # always the same three-point pitch.
                        status_parts.append(
                            "📖 **Ask the Lore** — RAG-only, no web "
                            "or reasoning beyond retrieved chunks "
                            "(edge-AI test + minimal tokens). Use "
                            "**Ask For Revelations** for reasoning-"
                            "open answers."
                        )
                    if unavailable:
                        # Show a brief reason per model — context length,
                        # rate-limit, timeout, etc. — so the user knows why
                        # the model dropped out rather than just "gone".
                        detailed = []
                        for r in unavailable:
                            err = r.get("error") or "no response"
                            detailed.append(f"**{r['model_display']}** ({err})")
                        status_parts.append("⚠️ Unavailable: " + " · ".join(detailed))

                    if boardroom_on:
                        # Mode 1 — synthesize into a single exec slide.
                        model_key_map = {AVAILABLE_MODELS[m]["display_name"]: m
                                         for m in ARENA_MODELS if m in AVAILABLE_MODELS}
                        exec_key = model_key_map.get(exec_model_display)
                        if not exec_key:
                            return _ao_empty_return(
                                f"Executive '{exec_model_display}' not configured.")
                        exec_slide = _run_executive_as_slide(
                            question, available, exec_key,
                            exec_tier_str, progress, history=hist,
                            grounded=grounded,
                        )
                        if exec_slide is None:
                            # Graceful degrade — exec failed (rate
                            # limit, context length, refusal, etc.)
                            # but peers succeeded. Fall through to
                            # peer mode with the question KEPT in
                            # the textbox so the user can just
                            # change the Executive Model and re-Ask
                            # without retyping. exec_failed drives
                            # the final ao_question output.
                            log.warning(
                                "Executive synthesis returned None "
                                "for %s — falling back to peer mode; "
                                "question preserved in the textbox.",
                                exec_model_display,
                            )
                            new_results = available
                            first = available[0]
                            last_exec_tier = ""
                            exec_failed = True
                            status_parts.append(
                                f"⚠️ **Executive `{exec_model_display}` "
                                "failed** — peer answers shown as "
                                "fallback. Change the Executive Model "
                                "in the Boardroom accordion and click "
                                "Ask For Revelations again to retry "
                                "synthesis without retyping."
                            )
                        else:
                            new_results = [exec_slide]
                            first = exec_slide
                            last_exec_tier = exec_tier_str
                            exec_failed = False
                    else:
                        # Mode 2 — peer slideshow, no exec yet.
                        new_results = available
                        first = available[0]
                        last_exec_tier = ""
                        exec_failed = False

                    status_parts.append(f"Showing 1 of {len(new_results)}")
                    label = _slide_label_for(first, blind_mode, 0)
                    escalate_updates = _escalation_updates(last_exec_tier)

                    # Roll the previous top into the thread — if there was
                    # anything showing before this Ask, capture it now.
                    if prev_current_q and prev_s_results:
                        try:
                            old_resp = prev_s_results[prev_s_idx].get("response", "")
                        except (IndexError, KeyError, TypeError):
                            old_resp = ""
                        old_top = _fmt_thread_top(prev_current_q, old_resp)
                        new_thread_below = (
                            f"{old_top}\n\n{prev_thread_below or ''}"
                        ).rstrip()
                    else:
                        new_thread_below = prev_thread_below or ""

                    display = _render_thread(
                        question, first["response"], new_thread_below,
                    )

                    # Extend history with the just-completed turn. The
                    # answer we store is the visible slide's text — for
                    # boardroom mode that's the exec synthesis, for peer
                    # mode it's the first peer's response. On exec
                    # failure (boardroom-mode + exec collapsed to peer
                    # fallback) we skip the append so a retry can add
                    # a clean single history entry.
                    new_history = list(prev_history or [])
                    if not exec_failed:
                        new_history.append({
                            "q":  question,
                            "a":  first.get("response") or "",
                            "ts": datetime.now().isoformat(timespec="seconds"),
                        })

                    # Only clear the textbox on a clean success. If
                    # the exec fell back (Groq TPM etc.), keep the
                    # question in place so the user can change the
                    # Executive Model and re-Ask without retyping.
                    question_out = question if exec_failed else ""

                    return (
                        new_results, 0,
                        " · ".join(status_parts),
                        display,
                        gr.update(value=label),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        last_exec_tier,
                        *escalate_updates,
                        question,                # ao_current_q_state
                        new_thread_below,        # ao_thread_below_state
                        question_out,            # ao_question (keep on exec fail)
                        new_history,             # ao_history_state
                    )

                def on_ao_add_to_rag(question, current_q, s_res, s_ix):
                    if not s_res:
                        return gr.update(), "Ask a question first."
                    # After Ask, the textbox clears; the question we
                    # actually asked lives in ao_current_q_state. Fall
                    # back to it so Add To RAG works right after Ask.
                    effective_q = (question or "").strip() or (current_q or "").strip()
                    if not effective_q:
                        return gr.update(), (
                            "Empty question — nothing to attribute this RAG "
                            "add to. Type or reload a question first."
                        )
                    question = effective_q
                    current = s_res[s_ix]
                    result = rag_add(
                        question=question,
                        response=current["response"],
                        metadata={
                            "model_key":     current["model_key"],
                            "model_display": current["model_display"],
                            "source":        "agent_ouija_rag_only",
                            "elapsed":       current["elapsed"],
                        },
                    )
                    if result.get("status") == "success":
                        log_rag_contribution(
                            model_key=current["model_key"],
                            model_display=current["model_display"],
                            model_type=AVAILABLE_MODELS.get(current["model_key"], {}).get("type", "unknown"),
                            source="agent_ouija",
                            question=question,
                            round_id=current.get("round_id"),
                        )
                        return gr.update(), (
                            f"✓ Added `{current['model_display']}` response to RAG."
                        )
                    return gr.update(), (
                        f"❌ RAG add failed: {result.get('message')}"
                    )

                # ── Wire — energy change cascades to everything ──────────
                energy_change_outputs = [
                    ao_level_description,
                    ao_energy_label,
                    *ao_local_cbs,
                    *ao_cloud_cbs,
                    ao_exec_model,
                    ao_exec_tier_state,
                    ao_exec_tier_display,
                    ao_exec_junior_btn,
                    ao_exec_senior_btn,
                    ao_exec_csuite_btn,
                ]
                ao_energy_dropdown.change(
                    on_energy_change,
                    inputs=[ao_energy_dropdown, ao_exec_tier_state],
                    outputs=energy_change_outputs,
                )

                # Persist the two Executive Boardroom checkboxes on
                # every toggle so the next launch opens with the same
                # boardroom mode + sr-with-jr choice. Named functions
                # (not lambdas) because Gradio 6 was silently skipping
                # some outputs=[] lambda handlers.
                def _persist_boardroom(v):
                    try:
                        prefs.update(boardroom_toggle=bool(v))
                        log.info("prefs: boardroom_toggle=%s persisted",
                                 bool(v))
                    except Exception as e:
                        log.warning("prefs.update(boardroom_toggle) failed: %s", e)
                    return None

                def _persist_sr_with_jr(v):
                    try:
                        prefs.update(sr_with_jr=bool(v))
                        log.info("prefs: sr_with_jr=%s persisted", bool(v))
                    except Exception as e:
                        log.warning("prefs.update(sr_with_jr) failed: %s", e)
                    return None

                ao_boardroom_toggle.change(
                    _persist_boardroom,
                    inputs=[ao_boardroom_toggle],
                    queue=False,
                )
                ao_sr_with_jr_toggle.change(
                    _persist_sr_with_jr,
                    inputs=[ao_sr_with_jr_toggle],
                    queue=False,
                )

                # Master checkboxes cascade to children.  All the change
                # handlers below are pure-UI (no model calls) — queue=False
                # so they don't get stuck behind a slow / errored Ask.
                ao_local_master.change(
                    on_local_master,
                    inputs=[ao_local_master, ao_energy_dropdown],
                    outputs=ao_local_cbs,
                    queue=False,
                )
                ao_cloud_master.change(
                    on_cloud_master,
                    inputs=[ao_cloud_master, ao_energy_dropdown],
                    outputs=ao_cloud_cbs,
                    queue=False,
                )

                # Voice toggle → audio player visibility (only one audio
                # now — exec answers ride in the same response window).
                ao_voice_toggle.change(
                    fn=lambda v: gr.update(visible=v, value=None),
                    inputs=[ao_voice_toggle],
                    outputs=[ao_response_audio],
                    queue=False,
                )

                # Mic input → transcribe + append into the question box.
                # We append (with a space separator) rather than replace
                # so a user can speak-then-type, or append a second
                # thought without losing the first. If transcription
                # returns None (missing key, API error, empty record),
                # the textbox stays as-is. We also clear the recording
                # after transcription so the next click starts fresh;
                # Gradio's gr.Audio otherwise holds onto the clip until
                # the user manually clears it.
                def _ao_transcribe_mic(audio_path, current_q):
                    if not audio_path:
                        return gr.update(), gr.update()
                    text = _stt_transcribe(audio_path)
                    if not text:
                        return gr.update(), gr.update(value=None)
                    cur = (current_q or "").rstrip()
                    merged = f"{cur} {text}".strip() if cur else text
                    return (
                        gr.update(value=merged),
                        gr.update(value=None),
                    )

                if _ao_mic_enabled:
                    ao_mic.stop_recording(
                        _ao_transcribe_mic,
                        inputs=[ao_mic, ao_question],
                        outputs=[ao_question, ao_mic],
                        queue=True,
                    )

                def _ao_maybe_synth(voice_on, s_results, s_idx):
                    """Speak only the CURRENT slide's raw response — not
                    the assembled thread. The thread textbox contains
                    every prior Q+A too and TTS would otherwise read the
                    lot (truncated at TTS_MAX_CHARS but still noisy)."""
                    if not voice_on or not s_results:
                        return None
                    try:
                        text = s_results[s_idx].get("response", "")
                    except (IndexError, KeyError, TypeError):
                        return None
                    if not text:
                        return None
                    path = synth_to_wav(text)
                    return str(path) if path else None

                # Score radio → enable/disable score buttons.
                def _ao_toggle_score(x):
                    s = gr.update(interactive=x is not None)
                    return s, s
                ao_score_radio.change(
                    fn=_ao_toggle_score,
                    inputs=[ao_score_radio],
                    outputs=[ao_score_btn, ao_score_rag_btn],
                    queue=False,
                )

                _ao_ask_outputs = [
                    s_results_ao, s_idx_ao, ao_status, ao_response,
                    ao_model_label, ao_score_btn, ao_retry_btn,
                    ao_last_exec_tier_state,
                    ao_escalate_row, ao_escalate_senior_btn,
                    ao_escalate_csuite_btn,
                    ao_current_q_state, ao_thread_below_state, ao_question,
                    ao_history_state,
                ]
                _ao_slideshow_outputs = [
                    s_results_ao, s_idx_ao, ao_status, ao_response,
                    ao_model_label, ao_score_btn, ao_score_radio,
                    ao_retry_btn,
                ]

                # Auto-save helper — quiet side-effect after each Ask.
                #
                # Two paths:
                #   (a) Named seance (path already set — either loaded
                #       from disk or manually Saved this session):
                #       overwrites the same file on disk so every
                #       follow-up question is preserved without the
                #       user having to click Save again.
                #   (b) Fresh "New Seance" (no path yet): auto-saves
                #       to `_autosave_new_seance.json` for the FIRST
                #       THREE turns as a safety net — quick chats get
                #       preserved without a manual Save. After 3
                #       turns we stop; the user is expected to hit
                #       Save Seance if they want the thread kept.
                #       The `_` prefix keeps the autosave file out of
                #       the Seances dropdown (see _list_seances).
                _AUTOSAVE_NEW_FILE  = "_autosave_new_seance.json"
                _AUTOSAVE_NEW_LIMIT = 3
                def _ao_autosave(seance_path, seance_name, question,
                                  level, s_results, s_idx,
                                  last_exec_tier, current_q, history):
                    if not s_results:
                        return
                    if not seance_path:
                        # New Seance path — quick-chat safety net.
                        turn_count = len(history or [])
                        if turn_count > _AUTOSAVE_NEW_LIMIT:
                            return   # user needs to Save manually now
                        seance_path = _AUTOSAVE_NEW_FILE
                    p = SEANCES_DIR / seance_path
                    existing = {}
                    if p.exists():
                        try:
                            existing = json.loads(p.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            existing = {}
                    # Preserve the original timestamp — it's part of the
                    # dropdown label (`{name} · {timestamp}`), and if
                    # auto-save keeps bumping it, the label the dropdown
                    # remembers stops matching the file on disk and
                    # Load fails with "Seance not found". "updated"
                    # tracks when the last write happened separately.
                    now_iso = datetime.now().isoformat(timespec="seconds")
                    data = {
                        "id":             existing.get("id",
                                                        seance_path.replace(".json", "")),
                        "name":           (seance_name
                                            or existing.get("name")
                                            or "unnamed"),
                        "timestamp":      existing.get("timestamp", now_iso),
                        "updated":        now_iso,
                        "question":       (current_q
                                            or question
                                            or existing.get("question", "")),
                        "energy_level":   level or existing.get("energy_level", ""),
                        "last_exec_tier": last_exec_tier or "",
                        "s_idx":          int(s_idx or 0),
                        "s_results":      s_results,
                        "history":        history or existing.get("history", []),
                    }
                    try:
                        p.write_text(
                            json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        log.info("Auto-saved seance to %s", p.name)
                    except OSError as e:
                        log.warning("Auto-save failed for %s: %s", p, e)

                # Hidden States that pin the `grounded` flag per
                # button. Ask For Revelations → False (normal fluent
                # answer). Ask the Lore → True (strict grounding).
                _grounded_off_state = gr.State(False)
                _grounded_on_state  = gr.State(True)

                _ao_ask_inputs = [
                    ao_question, ao_energy_dropdown, ao_boardroom_toggle,
                    ao_exec_tier_state, ao_exec_model, ao_blind_toggle,
                    ao_sr_with_jr_toggle,
                    ao_current_q_state, ao_thread_below_state,
                    s_results_ao, s_idx_ao, ao_history_state,
                    _grounded_off_state,
                    *ao_local_cbs, *ao_cloud_cbs,
                ]
                _ao_ask_lore_inputs = [
                    ao_question, ao_energy_dropdown, ao_boardroom_toggle,
                    ao_exec_tier_state, ao_exec_model, ao_blind_toggle,
                    ao_sr_with_jr_toggle,
                    ao_current_q_state, ao_thread_below_state,
                    s_results_ao, s_idx_ao, ao_history_state,
                    _grounded_on_state,
                    *ao_local_cbs, *ao_cloud_cbs,
                ]

                _ask_event_1 = ao_ask_btn.click(
                    on_ao_ask,
                    inputs=_ao_ask_inputs,
                    outputs=_ao_ask_outputs,
                )
                _ask_event_2 = _ask_event_1.then(
                    _ao_maybe_synth,
                    inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                    outputs=[ao_response_audio],
                )
                _autosave_inputs = [
                    ao_seance_path_state, ao_seance_name_state,
                    ao_question, ao_energy_dropdown,
                    s_results_ao, s_idx_ao,
                    ao_last_exec_tier_state, ao_current_q_state,
                    ao_history_state,
                ]
                _ask_event_1.then(
                    _ao_autosave,
                    inputs=_autosave_inputs,
                    outputs=[],
                    queue=False,
                )

                # Ask the Lore — mirrors the Ask button chain but
                # threads _grounded_on_state through the same handler
                # so every peer + exec call gets the STRICT GROUNDING
                # directive layered in.
                _lore_event_1 = ao_ask_lore_btn.click(
                    on_ao_ask,
                    inputs=_ao_ask_lore_inputs,
                    outputs=_ao_ask_outputs,
                )
                _lore_event_2 = _lore_event_1.then(
                    _ao_maybe_synth,
                    inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                    outputs=[ao_response_audio],
                )
                _lore_event_1.then(
                    _ao_autosave,
                    inputs=_autosave_inputs,
                    outputs=[],
                    queue=False,
                )

                # Cancel button — sets the module-level flag that all three
                # run_* helpers poll between as_completed iterations. Pending
                # futures get cancelled, already-in-flight ones can't be
                # aborted mid-request (LiteLLM is synchronous) but their
                # results are discarded when on_ao_ask sees the flag after
                # run_models returns. Also uses Gradio's `cancels=` to abort
                # any queued (not-yet-started) ask events. queue=False so
                # the cancel itself doesn't have to wait behind the Ask.
                def on_ao_cancel():
                    _ao_cancel_flag.set()
                    return ("🛑 Cancelling — pending model queries dropped; "
                            "any in-flight ones finish but their results "
                            "will be discarded.")
                ao_cancel_btn.click(
                    on_ao_cancel,
                    outputs=[ao_status],
                    cancels=[_ask_event_1, _ask_event_2,
                              _lore_event_1, _lore_event_2],
                    queue=False,
                )

                # 🔄 Refresh — pulls the last-typed draft out of
                # localStorage and puts it back in ao_question. Manual
                # counterpart to the auto-save keystroke handler above.
                # JS-only handler; no Python round-trip needed. Fires
                # an input event so Gradio's state notices the change.
                ao_refresh_draft_btn.click(
                    fn=None,
                    js=r"""
                    () => {
                      const STORAGE_KEY = 'agent_ouija_question_draft';
                      const q = document.querySelector(
                        '#ao_question textarea, #ao_question input'
                      );
                      if (!q) return;
                      try {
                        const saved = localStorage.getItem(STORAGE_KEY);
                        if (saved) {
                          q.value = saved;
                          q.dispatchEvent(new Event('input', {bubbles: true}));
                        }
                      } catch (e) { /* localStorage denied */ }
                    }
                    """,
                    queue=False,
                )

                # Exec-tier buttons — first update tier state (highlights
                # the button, refreshes the model dropdown), then fire a
                # one-off boardroom ask at that specific tier. The tier
                # state is remembered for the "Always Use Boardroom Mode"
                # checkbox path too.
                _exec_tier_button_outputs = [
                    ao_exec_tier_state, ao_exec_tier_display, ao_exec_model,
                    ao_exec_junior_btn, ao_exec_senior_btn, ao_exec_csuite_btn,
                ]

                def _make_forced_boardroom_ask(forced_tier):
                    """Run FRESH peers at the given tier, then synthesize
                    with an exec at that tier. Peer set is fixed by tier
                    — the user's checkbox selection and any cached
                    slides are ignored so each tier button shows what
                    that tier actually thinks of the question.

                      Junior  → all locals
                      Senior  → all clouds  (+ locals if sr_with_jr on)
                      C-Suite → 3 web-capable clouds

                    The same handler wires the top-row tier buttons AND
                    the escalation buttons at the bottom of an exec
                    answer — both want a fresh answer at that tier.
                    """
                    def handler(question, level, _boardroom_on, _tier_state,
                                exec_model, blind, sr_with_jr,
                                current_q, thread_below,
                                s_results, s_idx, last_exec_tier,
                                prev_history, grounded,
                                *cbs, progress=gr.Progress()):
                        _ao_cancel_flag.clear()
                        q = ((question or "").strip()
                             or (current_q or "").strip())
                        if not q:
                            return _ao_empty_return("Ask a question first.")

                        # Inline conversation memory — last N turns.
                        hist = (prev_history or [])[-CONVERSATION_HISTORY_WINDOW:]

                        tier_idx = _level_index(forced_tier)
                        if tier_idx == 0:
                            peer_keys = list(LOCAL_KEYS)
                            runner    = run_models
                        elif tier_idx == 1:
                            peer_keys = (list(LOCAL_KEYS) + list(CLOUD_KEYS)
                                          if sr_with_jr
                                          else list(CLOUD_KEYS))
                            runner    = run_research_models
                        else:
                            peer_keys = [m for m in CLOUD_KEYS
                                          if m in WEB_CAPABLE]
                            runner    = run_big3_models

                        media_error = _media_rule_error(q, len(peer_keys))
                        if media_error:
                            return _ao_empty_return(media_error)

                        results = runner(q, peer_keys, progress=progress,
                                          history=hist, grounded=grounded)
                        available = [r for r in results
                                     if r.get("response") is not None]
                        unavailable = [r for r in results
                                       if r.get("response") is None]

                        if _ao_cancel_flag.is_set():
                            _ao_cancel_flag.clear()
                            return _ao_empty_return(
                                "🛑 Cancelled — no answers saved from this round."
                            )
                        if not available:
                            return _ao_empty_return(
                                "Peers all failed. Check API keys / "
                                "context limits."
                            )

                        # Exec = current dropdown value if valid at tier,
                        # else _default_exec_value picks a sensible one.
                        model_key_map = {
                            AVAILABLE_MODELS[m]["display_name"]: m
                            for m in ARENA_MODELS if m in AVAILABLE_MODELS
                        }
                        exec_key = model_key_map.get(exec_model)
                        if not exec_key:
                            default_display = _default_exec_value(forced_tier)
                            exec_key = model_key_map.get(default_display)
                        if not exec_key:
                            return _ao_empty_return(
                                "No valid executive at this tier."
                            )
                        exec_slide = _run_executive_as_slide(
                            q, available, exec_key, forced_tier, progress,
                            history=hist, grounded=grounded,
                        )
                        if exec_slide is None:
                            # Graceful degrade — same treatment as
                            # on_ao_ask. Peers succeeded but the
                            # exec fell over; show the peers instead
                            # of nuking state, keep the question in
                            # the textbox so the user can change
                            # the Executive Model and re-click a
                            # tier button.
                            log.warning(
                                "Tier-button exec synthesis returned "
                                "None (tier=%s exec_key=%s) — falling "
                                "back to peer view.",
                                forced_tier, exec_key,
                            )
                            new_results = list(available)
                            new_idx = 0
                            label = _slide_label_for(
                                new_results[0], blind, new_idx,
                            )
                            escalate_updates = _NO_ESCALATE
                            exec_failed = True
                        else:
                            new_results = list(available) + [exec_slide]
                            new_idx = len(new_results) - 1
                            label = _slide_label_for(
                                exec_slide, blind, new_idx,
                            )
                            escalate_updates = _escalation_updates(
                                forced_tier,
                            )
                            exec_failed = False

                        # Thread accumulation — previous top rolls into
                        # thread_below just like a fresh Ask does.
                        if current_q and s_results:
                            try:
                                prev_resp = s_results[s_idx].get("response", "")
                            except (IndexError, KeyError, TypeError):
                                prev_resp = ""
                            prev_top = _fmt_thread_top(current_q, prev_resp)
                            new_thread_below = (
                                f"{prev_top}\n\n{thread_below or ''}"
                            ).rstrip()
                        else:
                            new_thread_below = thread_below or ""

                        status_parts = []
                        if unavailable:
                            detailed = [
                                f"**{r['model_display']}** "
                                f"({r.get('error') or 'no response'})"
                                for r in unavailable
                            ]
                            status_parts.append(
                                "⚠️ Unavailable: " + " · ".join(detailed)
                            )
                        if exec_failed:
                            status_parts.append(
                                f"⚠️ **Executive `{exec_model}` failed** "
                                "— peer answers shown as fallback. "
                                "Change the Executive Model and click "
                                "a tier button again to retry synthesis."
                            )
                        else:
                            status_parts.append(
                                f"👑 {_EXEC_TIER_LABELS[tier_idx]} — "
                                f"{len(available)} peer(s) — showing "
                                f"{new_idx + 1} of {len(new_results)}."
                            )

                        # Only add a history entry on success (real
                        # exec response). On failure we don't have a
                        # canonical "answer" for this turn yet, so
                        # let the retry populate history.
                        new_history = list(prev_history or [])
                        top_response = (
                            new_results[new_idx].get("response") or ""
                        )
                        if not exec_failed:
                            new_history.append({
                                "q":  q,
                                "a":  top_response,
                                "ts": datetime.now().isoformat(timespec="seconds"),
                            })

                        return (
                            new_results, new_idx,
                            " · ".join(status_parts),
                            _render_thread(q, top_response, new_thread_below),
                            gr.update(value=label),
                            gr.update(interactive=True),
                            gr.update(interactive=True),
                            "" if exec_failed else forced_tier,
                            *escalate_updates,
                            q,                 # ao_current_q_state
                            new_thread_below,  # ao_thread_below_state
                            (q if exec_failed else ""),   # ao_question
                            new_history,       # ao_history_state
                        )
                    return handler

                _tier_button_ask_inputs = [
                    ao_question, ao_energy_dropdown, ao_boardroom_toggle,
                    ao_exec_tier_state, ao_exec_model, ao_blind_toggle,
                    ao_sr_with_jr_toggle,
                    ao_current_q_state, ao_thread_below_state,
                    s_results_ao, s_idx_ao, ao_last_exec_tier_state,
                    ao_history_state, _grounded_off_state,
                    *ao_local_cbs, *ao_cloud_cbs,
                ]

                for btn, tier_str in (
                    (ao_exec_junior_btn, MYSTICISM_LEVELS[0]),
                    (ao_exec_senior_btn, MYSTICISM_LEVELS[1]),
                    (ao_exec_csuite_btn, MYSTICISM_LEVELS[3]),
                ):
                    _tier_ev = btn.click(
                        _make_exec_tier_click(tier_str),
                        inputs=[ao_energy_dropdown],
                        outputs=_exec_tier_button_outputs,
                    ).then(
                        _make_forced_boardroom_ask(tier_str),
                        inputs=_tier_button_ask_inputs,
                        outputs=_ao_ask_outputs,
                    )
                    _tier_ev.then(
                        _ao_maybe_synth,
                        inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                        outputs=[ao_response_audio],
                    )
                    _tier_ev.then(
                        _ao_autosave,
                        inputs=_autosave_inputs,
                        outputs=[],
                        queue=False,
                    )

                # Escalation buttons at the end of any exec answer use
                # the same handler as the top-row tier buttons — a fresh
                # peer run at the escalation tier, then the exec at that
                # tier synthesizes.
                for esc_btn, esc_tier in (
                    (ao_escalate_senior_btn, MYSTICISM_LEVELS[1]),
                    (ao_escalate_csuite_btn, MYSTICISM_LEVELS[3]),
                ):
                    _esc_ev = esc_btn.click(
                        _make_exec_tier_click(esc_tier),
                        inputs=[ao_energy_dropdown],
                        outputs=_exec_tier_button_outputs,
                    ).then(
                        _make_forced_boardroom_ask(esc_tier),
                        inputs=_tier_button_ask_inputs,
                        outputs=_ao_ask_outputs,
                    )
                    _esc_ev.then(
                        _ao_maybe_synth,
                        inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                        outputs=[ao_response_audio],
                    )
                    _esc_ev.then(
                        _ao_autosave,
                        inputs=_autosave_inputs,
                        outputs=[],
                        queue=False,
                    )

                # Slideshow handlers return raw response into ao_response;
                # we chain a reformat step that wraps it in the thread
                # display (Q header + thread below). Order matters:
                # reformat first so it can read the raw response the
                # handler just set, then synth (which reads the now-
                # formatted display — clean_for_speech drops the
                # separator unicode so TTS still sounds OK).
                _reformat_inputs  = [ao_current_q_state,
                                     ao_thread_below_state,
                                     ao_response]
                _reformat_outputs = [ao_response]

                ao_prev_btn.click(
                    on_previous,
                    inputs=[ao_blind_toggle, s_results_ao, s_idx_ao],
                    outputs=_ao_slideshow_outputs,
                ).then(_ao_reformat_display,
                       inputs=_reformat_inputs,
                       outputs=_reformat_outputs
                ).then(_ao_maybe_synth,
                       inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                       outputs=[ao_response_audio])
                ao_next_btn.click(
                    on_next,
                    inputs=[ao_blind_toggle, s_results_ao, s_idx_ao],
                    outputs=_ao_slideshow_outputs,
                ).then(_ao_reformat_display,
                       inputs=_reformat_inputs,
                       outputs=_reformat_outputs
                ).then(_ao_maybe_synth,
                       inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                       outputs=[ao_response_audio])
                ao_skip_btn.click(on_skip_round, outputs=_ao_slideshow_outputs)
                ao_retry_btn.click(
                    on_retry,
                    inputs=[ao_blind_toggle, s_results_ao, s_idx_ao],
                    outputs=_ao_slideshow_outputs,
                ).then(_ao_reformat_display,
                       inputs=_reformat_inputs,
                       outputs=_reformat_outputs
                ).then(_ao_maybe_synth,
                       inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                       outputs=[ao_response_audio])
                ao_score_btn.click(
                    on_score,
                    inputs=[ao_question, ao_current_q_state,
                            ao_blind_toggle, ao_score_radio,
                            s_results_ao, s_idx_ao],
                    outputs=_ao_slideshow_outputs,
                ).then(_ao_reformat_display,
                       inputs=_reformat_inputs,
                       outputs=_reformat_outputs
                ).then(_ao_maybe_synth,
                       inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                       outputs=[ao_response_audio])
                ao_score_rag_btn.click(
                    on_score_and_rag,
                    inputs=[ao_question, ao_current_q_state,
                            ao_blind_toggle, ao_score_radio,
                            s_results_ao, s_idx_ao],
                    outputs=_ao_slideshow_outputs,
                ).then(_ao_reformat_display,
                       inputs=_reformat_inputs,
                       outputs=_reformat_outputs
                ).then(_ao_maybe_synth,
                       inputs=[ao_voice_toggle, s_results_ao, s_idx_ao],
                       outputs=[ao_response_audio])
                ao_rag_only_btn.click(
                    on_ao_add_to_rag,
                    inputs=[ao_question, ao_current_q_state,
                            s_results_ao, s_idx_ao],
                    outputs=[ao_response, ao_status],
                )
                # Same handler, but wired to the inline button under
                # the response textbox — for one-click "keep this"
                # without opening the Scoring Controls accordion.
                ao_inline_rag_btn.click(
                    on_ao_add_to_rag,
                    inputs=[ao_question, ao_current_q_state,
                            s_results_ao, s_idx_ao],
                    outputs=[ao_response, ao_status],
                )

                def _on_ao_send_file(question, current_q):
                    """Email every file in docs/ or docs/media/ whose
                    name appears in the current question. Falls back
                    to ao_current_q_state when ao_question is empty
                    (same coalesce pattern the Add To RAG buttons
                    use)."""
                    from emailer import send_files
                    effective_q = ((question or "").strip()
                                    or (current_q or "").strip())
                    if not effective_q:
                        return (
                            "📧 Reference a filename in the question "
                            "first (e.g. `spec.pdf` or `dog.png`), "
                            "then click 📧 Send File."
                        )
                    paths = _collect_referenced_files(effective_q)
                    if not paths:
                        return (
                            "📧 No matching file in `docs/` or "
                            "`docs/media/`. Reference the exact "
                            "filename (case-insensitive) in your "
                            "question and click 📧 Send File again."
                        )
                    r = send_files([str(p) for p in paths])
                    icon = "✓" if r.get("status") == "success" else "❌"
                    return f"{icon} {r.get('message', '')}"

                ao_send_file_btn.click(
                    _on_ao_send_file,
                    inputs=[ao_question, ao_current_q_state],
                    outputs=[ao_status],
                    queue=False,
                )

                ao_upload_btn.upload(
                    on_upload,
                    inputs=[ao_upload_btn],
                    outputs=[ao_status],
                )

                # ── Save Seance ──────────────────────────────────────────
                # Serialise the current question + slideshow state to a
                # timestamped JSON in data/seances/. Reload from the
                # hidden Seances tab.
                def on_ao_save_seance(question, level, s_results, s_idx,
                                       last_exec_tier, seance_name,
                                       seance_path, current_q, history):
                    """Save under the current seance name (from the header
                    state). Three paths:
                      - Loaded seance (seance_path set) → update that
                        file in place. No new timestamped duplicate.
                      - Fresh seance whose NAME matches an on-disk
                        file → overwrite the existing match instead of
                        creating a same-name-different-timestamp dup.
                      - Otherwise → create a new timestamped file with
                        an auto-slugified name.

                    Auto-save already writes on every Ask when a
                    seance is loaded; this manual button just makes
                    the intent-to-save explicit.
                    """
                    if not s_results:
                        return ("Nothing to save — ask a question first.",
                                seance_name,
                                f"🕯 **Current Seance:** {seance_name}",
                                seance_path or "")
                    SEANCES_DIR.mkdir(parents=True, exist_ok=True)
                    ts_iso  = datetime.now().isoformat(timespec="seconds")
                    ts_file = datetime.now().strftime("%Y-%m-%d_%H%M%S")

                    # Determine the name: keep the header-state name
                    # when it's user-set; otherwise slugify the
                    # question. Falls back to "seance" if both are
                    # empty (defensive).
                    if seance_name and seance_name != "New Seance":
                        name = seance_name
                    else:
                        source = (question or current_q
                                    or "seance")
                        slug = "".join(c if c.isalnum() else "_"
                                        for c in source)[:40].strip("_")
                        name = slug or "seance"

                    # Path resolution — three cases (see docstring).
                    if seance_path:
                        fname = seance_path
                    else:
                        # Find any existing file whose `name` field
                        # inside its JSON matches. Same-name dedup —
                        # keeps the original timestamp so seance
                        # dropdown labels stay stable.
                        existing_match = None
                        for p in SEANCES_DIR.glob("*.json"):
                            try:
                                d = json.loads(
                                    p.read_text(encoding="utf-8"),
                                )
                            except (OSError, json.JSONDecodeError):
                                continue
                            if d.get("name") == name:
                                existing_match = p
                                break
                        if existing_match:
                            fname = existing_match.name
                        else:
                            fname = f"{ts_file}_{name}.json"

                    # Preserve original timestamp when overwriting an
                    # existing file — that field is part of the
                    # dropdown label and shifting it would break the
                    # user's mental map of which seance is which.
                    target = SEANCES_DIR / fname
                    existing_data = {}
                    if target.exists():
                        try:
                            existing_data = json.loads(
                                target.read_text(encoding="utf-8"),
                            )
                        except (OSError, json.JSONDecodeError):
                            pass

                    data = {
                        "id":             existing_data.get("id",
                                                             f"{ts_file}_{name}"),
                        "name":           name,
                        "timestamp":      existing_data.get("timestamp",
                                                             ts_iso),
                        "updated":        ts_iso,
                        "question":       (current_q
                                            or question
                                            or existing_data.get("question", "")),
                        "energy_level":   level or existing_data.get("energy_level", ""),
                        "last_exec_tier": last_exec_tier or "",
                        "s_idx":          int(s_idx or 0),
                        "s_results":      s_results,
                        "history":        history or existing_data.get("history", []),
                    }
                    try:
                        target.write_text(
                            json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        # Status message differentiates so users can
                        # tell an in-place save from a fresh save.
                        if seance_path or (target.exists() and existing_data):
                            action = "Updated"
                        else:
                            action = "Saved"
                        return (f"💾 {action} seance `{name}`.",
                                name,
                                f"🕯 **Current Seance:** {name}",
                                fname)
                    except OSError as e:
                        return (f"❌ Save failed: {e}",
                                seance_name,
                                f"🕯 **Current Seance:** {seance_name}",
                                seance_path or "")

                ao_save_seance_btn.click(
                    on_ao_save_seance,
                    inputs=[ao_question, ao_energy_dropdown,
                            s_results_ao, s_idx_ao, ao_last_exec_tier_state,
                            ao_seance_name_state, ao_seance_path_state,
                            ao_current_q_state, ao_history_state],
                    outputs=[ao_status, ao_seance_name_state,
                             ao_seance_name_display, ao_seance_path_state],
                    queue=False,
                )

                # Default Executive persistence. Toggle checkbox on →
                # save current dropdown value to DEFAULT_EXEC_PATH.
                # Toggle off → remove the file. Dropdown change re-syncs
                # the checkbox based on whether the new value == saved
                # default. Also runs once on page load via demo.load().
                def on_default_exec_toggle(save_it, current_display):
                    if save_it and current_display:
                        DEFAULT_EXEC_PATH.parent.mkdir(parents=True, exist_ok=True)
                        DEFAULT_EXEC_PATH.write_text(current_display.strip(),
                                                     encoding="utf-8")
                        return f"💾 `{current_display}` saved as Default Executive."
                    if not save_it:
                        try:
                            DEFAULT_EXEC_PATH.unlink()
                        except FileNotFoundError:
                            pass
                        return "Default Executive cleared."
                    return ""

                def on_exec_model_change(current_display):
                    saved = _load_default_exec()
                    groq_display = (
                        AVAILABLE_MODELS.get("groq", {}).get("display_name")
                        or "Groq"
                    )
                    # Auto-persist the last-picked exec so a fresh
                    # launch re-selects it. Distinct from the
                    # "Make this the Default Executive" checkbox —
                    # this fires on EVERY pick, no click needed.
                    # Handler runs on programmatic updates too (e.g.
                    # tier change repopulating the dropdown); we
                    # only persist non-empty picks that map to a
                    # REAL model. Guards against persisting stale
                    # / mismatched values (e.g. "Claude Sonnet" coming
                    # through on a locals-only rig) which would then
                    # crash a subsequent launch.
                    _known_displays = {
                        cfg.get("display_name")
                        for cfg in AVAILABLE_MODELS.values()
                        if cfg.get("display_name")
                    }
                    if current_display and current_display in _known_displays:
                        try:
                            prefs.update(last_executive=current_display)
                        except Exception as e:
                            log.warning(
                                "prefs.update(last_executive) failed: %s", e,
                            )
                    return (
                        gr.update(value=(saved == current_display
                                          and current_display is not None)),
                        # Show Groq warning only when Groq is picked.
                        gr.update(visible=(current_display == groq_display)),
                    )

                ao_default_exec_toggle.change(
                    on_default_exec_toggle,
                    inputs=[ao_default_exec_toggle, ao_exec_model],
                    outputs=[ao_status],
                )
                ao_exec_model.change(
                    on_exec_model_change,
                    inputs=[ao_exec_model],
                    outputs=[ao_default_exec_toggle, ao_groq_exec_warning],
                )

                # ── Section 6 — Seances ───────────────────────────────────
                # Saved boardroom sessions. Load restores the slideshow;
                # rename edits the name in place. Fully local, gitignored.
                SEANCES_DIR.mkdir(parents=True, exist_ok=True)

                def _list_seances() -> list[dict]:
                    if not SEANCES_DIR.exists():
                        return []
                    out = []
                    for p in sorted(SEANCES_DIR.glob("*.json"), reverse=True):
                        # Skip the quick-chat autosave slot — it's a
                        # recovery file, not a proper seance the user
                        # picked a name for. The `_` prefix is our
                        # convention for internal seance files.
                        if p.name.startswith("_"):
                            continue
                        try:
                            data = json.loads(p.read_text(encoding="utf-8"))
                            data["_path"] = str(p)
                            out.append(data)
                        except (OSError, json.JSONDecodeError) as e:
                            log.warning("Could not read seance %s: %s", p, e)
                    return out

                def _seance_label(s: dict) -> str:
                    name = s.get("name") or s.get("id") or "seance"
                    ts   = s.get("timestamp", "?")
                    return f"{name}  ·  {ts}"

                _init_seances_choices = [_seance_label(s) for s in _list_seances()]

                with gr.Accordion("🕯 Seances", open=False):
                    gr.Markdown(
                        "Every **💾 Save Seance** writes to `data/seances/` "
                        "under whichever name is shown in the header. Load "
                        "restores the slideshow and updates the header. "
                        "Rename edits the current label — nothing is deleted "
                        "unless you use 🗑."
                    )
                    ao_seances_dropdown = gr.Dropdown(
                        choices=_init_seances_choices,
                        value=(_init_seances_choices[0]
                               if _init_seances_choices else None),
                        label="Saved Seances (newest first)",
                        interactive=True,
                    )
                    ao_seances_preview = gr.Textbox(
                        label="Preview",
                        lines=6, max_lines=20,
                        interactive=False,
                    )
                    ao_seances_rename_input = gr.Textbox(
                        label="Rename selected seance to",
                        placeholder="new-seance-name",
                    )
                    with gr.Row():
                        ao_seances_load_btn    = gr.Button("↩ Load", variant="primary")
                        ao_seances_delete_btn  = gr.Button("🗑 Delete", variant="stop")
                        ao_seances_refresh_btn = gr.Button("🔄 Refresh")
                        ao_seances_rename_btn  = gr.Button(
                            "✎ Rename", variant="secondary",
                        )
                    ao_seances_status = gr.Markdown("")

                def _preview_seance_by_label(label):
                    if not label:
                        return ""
                    for s in _list_seances():
                        if _seance_label(s) == label:
                            slides = s.get("s_results", []) or []
                            head = (
                                f"Question: {s.get('question', '')}\n"
                                f"Energy Level: {s.get('energy_level', '?')}\n"
                                f"Slides: {len(slides)}\n\n"
                            )
                            body = ""
                            for r in slides[:3]:
                                resp = (r.get('response') or '').strip()
                                snip = resp[:200] + ("…" if len(resp) > 200 else "")
                                body += f"— {r.get('model_display', '?')} — {snip}\n\n"
                            if len(slides) > 3:
                                body += f"…and {len(slides) - 3} more slide(s)."
                            return head + body
                    return "(seance not found — click Refresh)"

                def on_ao_seances_refresh():
                    fresh = _list_seances()
                    choices = [_seance_label(s) for s in fresh]
                    value   = choices[0] if choices else None
                    return (gr.update(choices=choices, value=value),
                            _preview_seance_by_label(value),
                            f"🔄 {len(choices)} seance(s) on disk.",
                            "")   # clear rename input

                def on_ao_seances_delete(label):
                    if not label:
                        return (gr.update(), "", "Nothing selected.", "")
                    for s in _list_seances():
                        if _seance_label(s) == label:
                            try:
                                Path(s["_path"]).unlink(missing_ok=True)
                                status = f"🗑 Deleted `{Path(s['_path']).name}`."
                            except OSError as e:
                                status = f"Delete failed: {e}"
                            break
                    else:
                        status = "Seance not found on disk."
                    dd_upd, prev_upd, _refresh_status, _rn = on_ao_seances_refresh()
                    return (dd_upd, prev_upd, status, "")

                def _reconstruct_thread_below(history):
                    """Assemble the formatted thread string from a
                    saved history list. Uses every turn EXCEPT the
                    last one — the last is what shows at the top of
                    the response window via _fmt_thread_top. Empty
                    result when there's 0-1 turns."""
                    if not history or len(history) < 2:
                        return ""
                    parts = []
                    for turn in reversed(history[:-1]):
                        q = turn.get("q", "")
                        a = turn.get("a", "")
                        if not q and not a:
                            continue
                        parts.append(_fmt_thread_top(q, a))
                    return "\n\n".join(parts).rstrip()

                def on_ao_seances_load(label):
                    """Restore Agent Ouija's slideshow state from the picked
                    seance and update the seance name header. Also returns
                    the filename so ao_seance_path_state starts pointing at
                    the on-disk file for auto-save-after-ask.

                    Question textbox loads BLANK so the user can just type
                    a follow-up. The answer window shows the full
                    reconstructed thread — current Q+A at top, all
                    prior history below — so nothing is lost visually.
                    ao_current_q_state and ao_thread_below_state are
                    also populated so the next Ask threads correctly
                    (loaded question rolls into thread_below on top of
                    prior history)."""
                    empty_ret = (
                        [], 0, "Pick a seance first.", "",
                        gr.update(value="—"),
                        "",                     # ao_question stays blank
                        "New Seance",
                        "🕯 **Current Seance:** New Seance",
                        "",                     # ao_seance_path_state
                        [],                     # ao_history_state
                        "",                     # ao_current_q_state
                        "",                     # ao_thread_below_state
                    )
                    if not label:
                        return empty_ret
                    for s in _list_seances():
                        if _seance_label(s) == label:
                            s_results = s.get("s_results", []) or []
                            history   = s.get("history", []) or []
                            fname = Path(s["_path"]).name
                            if not s_results:
                                return (
                                    [], 0, "Seance file is empty.", "",
                                    gr.update(value="—"),
                                    "", "New Seance",
                                    "🕯 **Current Seance:** New Seance",
                                    "", [], "", "",
                                )
                            idx = min(int(s.get("s_idx", 0) or 0),
                                       len(s_results) - 1)
                            slide = s_results[idx]
                            lbl = f"{slide['model_display']}  ·  {slide['elapsed']}s"
                            name = s.get("name") or s.get("id") or "loaded seance"

                            # The last-asked question — either the last
                            # history entry's `q`, or the seance's
                            # top-level `question` field for old
                            # seances that pre-date the history feature.
                            loaded_q = (
                                (history[-1].get("q") if history else "")
                                or s.get("question", "")
                                or ""
                            )
                            # Rebuild thread_below from history[:-1].
                            # Old seances with no history → empty
                            # thread_below (only current Q+A shows).
                            reconstructed_below = _reconstruct_thread_below(history)
                            display_text = _render_thread(
                                loaded_q,
                                slide.get("response", ""),
                                reconstructed_below,
                            )
                            return (
                                s_results, idx,
                                f"↩ Loaded seance `{name}` — showing slide "
                                f"{idx + 1} of {len(s_results)} · "
                                f"{len(history)} turn(s) of memory. "
                                "Type a follow-up to resume the "
                                "conversation. Auto-save is on.",
                                display_text,
                                gr.update(value=lbl),
                                "",                       # ao_question blank
                                name,
                                f"🕯 **Current Seance:** {name}",
                                fname,
                                history,
                                loaded_q,                 # ao_current_q_state
                                reconstructed_below,      # ao_thread_below_state
                            )
                    return (
                        [], 0, "Seance not found (deleted?).", "",
                        gr.update(value="—"),
                        "", "New Seance",
                        "🕯 **Current Seance:** New Seance",
                        "", [], "", "",
                    )

                # Quick-chat recovery — loads _autosave_new_seance.json
                # into the current session (same shape as seance load
                # but keeps the header "New Seance" and the path state
                # empty so the 3-turn autosave rule still applies to
                # any further turns). Mobile Safari suspends background
                # tabs; the response completes server-side but the
                # client comes back to a fresh session. This restores.
                _AUTOSAVE_RECOVERY_WINDOW_SECONDS = 3600   # 1 hour

                def _autosave_recent() -> bool:
                    """True iff _autosave_new_seance.json exists AND
                    was written in the last _AUTOSAVE_RECOVERY_WINDOW
                    seconds. Used by demo.load to decide whether to
                    surface the 🔄 Recover button."""
                    p = SEANCES_DIR / _AUTOSAVE_NEW_FILE
                    if not p.exists():
                        return False
                    try:
                        age = time.time() - p.stat().st_mtime
                        return age <= _AUTOSAVE_RECOVERY_WINDOW_SECONDS
                    except OSError:
                        return False

                def on_ao_recover_autosave():
                    """Restore the quick-chat autosave into the current
                    session. Header stays 'New Seance' so the same
                    3-turn autosave rule keeps protecting further turns.
                    Return shape matches on_ao_seances_load + a
                    visibility update to hide the Recover button after
                    successful load."""
                    empty_ret = (
                        [], 0, "No autosaved chat to recover.", "",
                        gr.update(value="—"),
                        "",                       # ao_question blank
                        "New Seance",
                        "🕯 **Current Seance:** New Seance",
                        "",                       # ao_seance_path_state
                        [],                       # ao_history_state
                        "",                       # ao_current_q_state
                        "",                       # ao_thread_below_state
                        gr.update(visible=False), # ao_recover_btn (hide)
                    )
                    p = SEANCES_DIR / _AUTOSAVE_NEW_FILE
                    if not p.exists():
                        return empty_ret
                    try:
                        s = json.loads(p.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as e:
                        log.warning("Autosave unreadable: %s", e)
                        return empty_ret
                    s_results = s.get("s_results", []) or []
                    history   = s.get("history", []) or []
                    if not s_results:
                        return empty_ret
                    idx = min(int(s.get("s_idx", 0) or 0),
                               len(s_results) - 1)
                    slide = s_results[idx]
                    lbl = f"{slide['model_display']}  ·  {slide['elapsed']}s"
                    loaded_q = (
                        (history[-1].get("q") if history else "")
                        or s.get("question", "")
                        or ""
                    )
                    reconstructed_below = _reconstruct_thread_below(history)
                    display_text = _render_thread(
                        loaded_q,
                        slide.get("response", ""),
                        reconstructed_below,
                    )
                    return (
                        s_results, idx,
                        f"🔄 Recovered autosave — {len(history)} turn(s) "
                        f"of memory · showing slide {idx + 1} of "
                        f"{len(s_results)}. Continue typing to resume; "
                        "hit 💾 Save Seance to keep it under a name.",
                        display_text,
                        gr.update(value=lbl),
                        "",                       # ao_question blank
                        "New Seance",             # header stays New Seance
                        "🕯 **Current Seance:** New Seance",
                        "",                       # keep path empty
                        history,
                        loaded_q,
                        reconstructed_below,
                        gr.update(visible=False), # hide Recover button
                    )

                def on_ao_seances_rename(selected_label, new_name,
                                          current_seance_name):
                    """Rename the on-disk file for the selected seance to
                    the given name. If the renamed seance is the current
                    one (matched by name field in the JSON), also update
                    the header state."""
                    new_name = (new_name or "").strip()
                    if not selected_label:
                        return (gr.update(), "",
                                "Pick a seance first.", "",
                                current_seance_name,
                                f"🕯 **Current Seance:** {current_seance_name}")
                    if not new_name:
                        return (gr.update(), "",
                                "Type a new name first.", "",
                                current_seance_name,
                                f"🕯 **Current Seance:** {current_seance_name}")
                    for s in _list_seances():
                        if _seance_label(s) == selected_label:
                            old_path = Path(s["_path"])
                            new_data = dict(s)
                            new_data.pop("_path", None)
                            new_data["name"] = new_name
                            try:
                                old_path.write_text(
                                    json.dumps(new_data, ensure_ascii=False,
                                                indent=2),
                                    encoding="utf-8",
                                )
                                status = (
                                    f"✎ Renamed `{s.get('name') or s.get('id')}` "
                                    f"→ `{new_name}`."
                                )
                            except OSError as e:
                                status = f"Rename failed: {e}"
                                break
                            # If the renamed seance is what the user has
                            # loaded, update the header state too.
                            was_current = (
                                current_seance_name
                                and current_seance_name in (
                                    s.get("name"), s.get("id"),
                                )
                            )
                            fresh = _list_seances()
                            choices = [_seance_label(x) for x in fresh]
                            first = choices[0] if choices else None
                            new_current = (
                                new_name if was_current else current_seance_name
                            )
                            return (
                                gr.update(choices=choices, value=first),
                                _preview_seance_by_label(first),
                                status,
                                "",
                                new_current,
                                f"🕯 **Current Seance:** {new_current}",
                            )
                    return (gr.update(), "",
                            "Seance not found on disk.", "",
                            current_seance_name,
                            f"🕯 **Current Seance:** {current_seance_name}")

                ao_seances_dropdown.change(
                    _preview_seance_by_label,
                    inputs=[ao_seances_dropdown],
                    outputs=[ao_seances_preview],
                    queue=False,
                )
                ao_seances_refresh_btn.click(
                    on_ao_seances_refresh,
                    outputs=[ao_seances_dropdown, ao_seances_preview,
                             ao_seances_status, ao_seances_rename_input],
                    queue=False,
                )
                ao_seances_delete_btn.click(
                    on_ao_seances_delete,
                    inputs=[ao_seances_dropdown],
                    outputs=[ao_seances_dropdown, ao_seances_preview,
                             ao_seances_status, ao_seances_rename_input],
                    queue=False,
                )
                ao_seances_load_btn.click(
                    on_ao_seances_load,
                    inputs=[ao_seances_dropdown],
                    outputs=[s_results_ao, s_idx_ao, ao_status, ao_response,
                             ao_model_label, ao_question,
                             ao_seance_name_state, ao_seance_name_display,
                             ao_seance_path_state, ao_history_state,
                             ao_current_q_state, ao_thread_below_state],
                    queue=False,
                )
                # 🔄 Recover last chat — restores the quick-chat
                # autosave into the current session. See on_ao_recover
                # above for the return-shape rationale.
                ao_recover_btn.click(
                    on_ao_recover_autosave,
                    outputs=[s_results_ao, s_idx_ao, ao_status, ao_response,
                             ao_model_label, ao_question,
                             ao_seance_name_state, ao_seance_name_display,
                             ao_seance_path_state, ao_history_state,
                             ao_current_q_state, ao_thread_below_state,
                             ao_recover_btn],
                    queue=False,
                )
                # On page mount, surface the 🔄 Recover button only
                # when a recent quick-chat autosave exists. This fires
                # after every mount — including the mobile "come back
                # to browser" reload that lost the client's state
                # while the server-side autosave completed. Threshold
                # in _AUTOSAVE_RECOVERY_WINDOW_SECONDS above.
                def _on_load_show_recover():
                    return gr.update(visible=_autosave_recent())
                demo.load(
                    _on_load_show_recover,
                    outputs=[ao_recover_btn],
                    queue=False,
                )
                ao_seances_rename_btn.click(
                    on_ao_seances_rename,
                    inputs=[ao_seances_dropdown, ao_seances_rename_input,
                            ao_seance_name_state],
                    outputs=[ao_seances_dropdown, ao_seances_preview,
                             ao_seances_status, ao_seances_rename_input,
                             ao_seance_name_state, ao_seance_name_display],
                    queue=False,
                )

                # ── ＋ New Seance — reset everything to a fresh session ────
                def on_ao_new_seance():
                    return (
                        [],   # s_results_ao
                        0,    # s_idx_ao
                        "New Seance",   # ao_seance_name_state
                        "🕯 **Current Seance:** New Seance",  # display
                        "",   # ao_question
                        "",   # ao_response
                        gr.update(value="—"),  # ao_model_label
                        "",   # ao_thread_below_state
                        "",   # ao_current_q_state
                        "🕯 Fresh seance — ready.",   # ao_status
                        "",   # ao_seance_path_state — clear auto-save target
                        [],   # ao_history_state — fresh memory
                    )
                ao_new_seance_btn.click(
                    on_ao_new_seance,
                    outputs=[
                        s_results_ao, s_idx_ao,
                        ao_seance_name_state, ao_seance_name_display,
                        ao_question, ao_response, ao_model_label,
                        ao_thread_below_state, ao_current_q_state,
                        ao_status, ao_seance_path_state, ao_history_state,
                    ],
                    queue=False,
                )


            with gr.Tab("Dev Workshop", id="dev_workshop"):
                _dev_return_btn = gr.Button("🔮 Return to Agent Ouija",
                                             variant="secondary")
                gr.Markdown(
                    "### Dev Workshop\n"
                    "Two collapsible sections: **📓 Notebook** for course-style "
                    "programmatic prompts (Python cells against arena "
                    "models), and **🖥️ CLI** for running PowerShell commands "
                    "and launching Claude Code from inside Agent Ouija.\n\n"
                    "Click either accordion header to hide the section you're "
                    "not using — the layout collapses to just what you need."
                )

                with gr.Accordion("📓 Notebook", open=False):
                    gr.Markdown(
                        "Write Python and run it against any subset of the "
                        "arena's models. **No personas, no "
                        "PERSONA_REINFORCEMENT, no auto-RAG** — prompts go "
                        "to the model verbatim. Mirrors the course's Colab "
                        "/ Jupyter pattern.\n\n"
                        "**Cells** — split with `# %%` on its own line.\n\n"
                        "**Per-model checkboxes** drive iteration:\n"
                        "- One model checked → code runs once; state persists "
                        "across Run clicks.\n"
                        "- Several checked → code runs once per model on an "
                        "isolated copy of state. Slideshow shows one response "
                        "at a time so you can read and score each.\n\n"
                        "**Variables injected into scope on every run:**\n"
                        "- `MODEL` — LiteLLM-format id "
                        "(e.g. `\"openai/gpt-4o\"`, `\"ollama_chat/llama3.2\"`).\n"
                        "- `MODEL_KEY` — config key (e.g. `\"gpt-4o\"`).\n"
                        "- `SELECTED_MODELS` — list of all checked keys.\n\n"
                        "**Helpers in scope:** `ask(prompt, model, system, "
                        "temperature, max_tokens) → str`, `ask_all(prompt, …) → "
                        "dict`, `get_rag(question) → str | None`.\n\n"
                        "API keys come from `.env` — no Colab `userdata.get(...)` "
                        "setup needed."
                    )

                    # Model picker — matches the Agent Ouija tab shape:
                    # per-model Checkboxes with an All Locals / All
                    # Clouds master toggle per column. Default: GPT-4o
                    # pre-checked (the course uses it as its default).
                    _nb_gpt4o_default = "gpt-4o" in CLOUD_KEYS
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Local Models")
                            nb_local_master = gr.Checkbox(
                                value=False,
                                label="**All Locals** (master toggle)",
                            )
                            nb_local_cbs = []
                            for k in LOCAL_KEYS:
                                cfg = AVAILABLE_MODELS[k]
                                cb = gr.Checkbox(
                                    value=False,
                                    label=f"{cfg['display_name']} — {cfg.get('description', '')}",
                                    interactive=True,
                                )
                                nb_local_cbs.append(cb)
                        with gr.Column():
                            gr.Markdown("### Cloud Models")
                            nb_cloud_master = gr.Checkbox(
                                value=False,
                                label="**All Clouds** (master toggle)",
                            )
                            nb_cloud_cbs = []
                            for k in CLOUD_KEYS:
                                cfg = AVAILABLE_MODELS[k]
                                cb = gr.Checkbox(
                                    value=(k == "gpt-4o" and _nb_gpt4o_default),
                                    label=f"{cfg['display_name']} — {cfg.get('description', '')}",
                                    interactive=True,
                                )
                                nb_cloud_cbs.append(cb)

                    # Master cascades — flip every child to the
                    # master's value. No energy-level gate on this
                    # tab (no Mysticism dropdown here), so every
                    # child is always interactive.
                    def _nb_master_cascade_local(m):
                        return [gr.update(value=bool(m)) for _ in LOCAL_KEYS]

                    def _nb_master_cascade_cloud(m):
                        return [gr.update(value=bool(m)) for _ in CLOUD_KEYS]

                    nb_local_master.change(
                        _nb_master_cascade_local,
                        inputs=[nb_local_master],
                        outputs=nb_local_cbs,
                        queue=False,
                    )
                    nb_cloud_master.change(
                        _nb_master_cascade_cloud,
                        inputs=[nb_cloud_master],
                        outputs=nb_cloud_cbs,
                        queue=False,
                    )

                    # State containers — same shape as Arena slideshow so the
                    # Previous / Next / Skip / Retry helpers can be reused.
                    nb_state     = gr.State(make_globals())
                    nb_s_results = gr.State([])
                    nb_s_idx     = gr.State(0)

                    nb_code = gr.Code(
                        value=(
                            "# %%\n"
                            "# Course-style: call LiteLLM directly. The MODEL\n"
                            "# variable is injected based on which checkbox above\n"
                            "# is ticked; the same code runs once per model.\n"
                            "from litellm import completion\n"
                            "from typing import List, Dict\n"
                            "\n"
                            "def generate_response(messages: List[Dict]) -> str:\n"
                            "    response = completion(\n"
                            "        model=MODEL,\n"
                            "        messages=messages,\n"
                            "        max_tokens=1024,\n"
                            "    )\n"
                            "    return response.choices[0].message.content\n"
                            "\n"
                            "messages = [\n"
                            "    {\"role\": \"system\", \"content\": \"You are concise.\"},\n"
                            "    {\"role\": \"user\",   \"content\": \"Write a function "
                            "to swap the keys and values in a dictionary.\"},\n"
                            "]\n"
                            "\n"
                            "print(generate_response(messages))\n"
                        ),
                        language="python",
                        lines=20,
                        label="Code (use # %% to split into cells)",
                    )

                    with gr.Row():
                        nb_blind_toggle = gr.Checkbox(
                            value=True, label="Blind mode (hide model)"
                        )
                        nb_run_btn      = gr.Button("Run", variant="primary")
                        nb_reset_btn    = gr.Button("Reset State")

                    nb_status = gr.Markdown(
                        "_Pick at least one model checkbox, then click **Run**. "
                        "While the model is thinking the orange Gradio progress "
                        "bar shows up at the top of the page._"
                    )

                    # Slideshow output — mirrors the Arena Group, including the
                    # same response Textbox + score radio + button layout.
                    with gr.Group():
                        nb_model_label = gr.Markdown("—")
                        nb_response_box = gr.Textbox(
                            label="Response",
                            lines=15,
                            max_lines=40,
                            interactive=False,
                        )

                    nb_score_radio = gr.Radio(
                        choices=score_choices,
                        label="Score This Response",
                        interactive=True,
                        value=None,
                    )
                    with gr.Row():
                        nb_score_btn = gr.Button("Score And Next Response",
                                                 interactive=False)
                        nb_retry_btn = gr.Button("Retry Scoring", interactive=False)
                    with gr.Row():
                        nb_prev_btn = gr.Button("◀ Previous")
                        nb_next_btn = gr.Button("Next ▶")
                        nb_skip_btn = gr.Button("Skip To New Prompt")

                    nb_score_radio.change(
                        fn=lambda x: gr.update(interactive=x is not None),
                        inputs=[nb_score_radio],
                        outputs=[nb_score_btn],
                    )

                    # ── Handlers ──────────────────────────────────────────────

                    def _nb_label(item, idx, blind_mode):
                        """Slideshow label — blind shows 'Model A/B/C…', otherwise
                        the model's real display name. Time is always shown."""
                        if blind_mode:
                            return f"**Model {chr(65 + idx)}**  ·  {item['elapsed']}s"
                        return f"**{item['model_display']}**  ·  {item['elapsed']}s"

                    def on_notebook_run(code, state, blind_mode,
                                        *cb_values, progress=gr.Progress()):
                        """Run code per checked model, with the same orange
                        Gradio progress bar Arena uses. Returns first slide of
                        the per-model response slideshow. Always shuffles results
                        so blind mode is meaningful — blind toggle only controls
                        the label visibility.

                        cb_values order: LOCAL_KEYS booleans followed by
                        CLOUD_KEYS booleans — same shape the Agent Ouija
                        tab's Ask handler uses. Individual per-model
                        checkboxes replaced the old CheckboxGroups so
                        the UI matches AO tab's model panel.
                        """
                        n_local = len(LOCAL_KEYS)
                        local_bools = cb_values[:n_local]
                        cloud_bools = cb_values[n_local:]
                        selected_keys = (
                            [k for k, on in zip(LOCAL_KEYS, local_bools) if on] +
                            [k for k, on in zip(CLOUD_KEYS, cloud_bools) if on]
                        )
                        if not selected_keys:
                            return (
                                [], 0,
                                "⚠️ Pick at least one model checkbox before running.",
                                "",
                                gr.update(value="—"),
                                gr.update(interactive=False),
                                gr.update(value=None),
                                gr.update(interactive=False),
                                state,
                            )

                        progress(0, desc="Starting…")
                        results, new_state = run_cells_for_models(
                            code, state, selected_keys, progress=progress
                        )

                        if not results:
                            return (
                                [], 0,
                                "No output produced — check the code editor.",
                                "",
                                gr.update(value="—"),
                                gr.update(interactive=False),
                                gr.update(value=None),
                                gr.update(interactive=False),
                                new_state,
                            )

                        # Always randomize so blind mode is meaningful. Matches
                        # the Arena tab's "blind controls labels only" rule.
                        random.shuffle(results)

                        first = results[0]
                        status = (
                            f"Ran {len(results)} model(s). Showing 1 of "
                            f"{len(results)} — score and advance, or use "
                            f"Previous / Next to browse."
                        )
                        return (
                            results, 0,
                            status,
                            first["response"],
                            gr.update(value=_nb_label(first, 0, blind_mode)),
                            gr.update(interactive=False),     # score_btn until radio touched
                            gr.update(value=None),             # clear radio
                            gr.update(interactive=True),       # retry_btn live
                            new_state,
                        )

                    def on_notebook_reset():
                        return (
                            make_globals(),
                            [], 0,
                            "_State reset — all variables cleared._",
                            "",
                            gr.update(value="—"),
                            gr.update(interactive=False),
                            gr.update(value=None),
                            gr.update(interactive=False),
                        )

                    def _nb_show_slide(s_results, idx, status_msg, blind_mode):
                        item  = s_results[idx]
                        return (
                            s_results, idx,
                            status_msg,
                            item["response"],
                            gr.update(value=_nb_label(item, idx, blind_mode)),
                            gr.update(interactive=False),
                            gr.update(value=None),
                            gr.update(interactive=True),
                        )

                    def on_notebook_previous(blind_mode, s_results, s_idx):
                        if not s_results:
                            return (s_results, s_idx,
                                    "Nothing to navigate — click Run first.",
                                    "", gr.update(value="—"),
                                    gr.update(interactive=False),
                                    gr.update(value=None),
                                    gr.update(interactive=False))
                        if s_idx <= 0:
                            return _nb_show_slide(s_results, 0,
                                                  f"Already at the first response "
                                                  f"(1 of {len(s_results)}).",
                                                  blind_mode)
                        new_idx = s_idx - 1
                        return _nb_show_slide(s_results, new_idx,
                                              f"Showing {new_idx + 1} of {len(s_results)}",
                                              blind_mode)

                    def on_notebook_next(blind_mode, s_results, s_idx):
                        if not s_results:
                            return (s_results, s_idx,
                                    "Nothing to navigate — click Run first.",
                                    "", gr.update(value="—"),
                                    gr.update(interactive=False),
                                    gr.update(value=None),
                                    gr.update(interactive=False))
                        last = len(s_results) - 1
                        if s_idx >= last:
                            return _nb_show_slide(s_results, last,
                                                  f"Already at the last response "
                                                  f"({len(s_results)} of {len(s_results)}).",
                                                  blind_mode)
                        new_idx = s_idx + 1
                        return _nb_show_slide(s_results, new_idx,
                                              f"Showing {new_idx + 1} of {len(s_results)}",
                                              blind_mode)

                    def on_notebook_skip():
                        return (
                            [], 0,
                            "Round skipped — edit the code and Run when ready.",
                            "",
                            gr.update(value="—"),
                            gr.update(interactive=False),
                            gr.update(value=None),
                            gr.update(interactive=False),
                        )

                    def on_notebook_retry(blind_mode, s_results, s_idx):
                        """Re-step from position 0 without re-running models.
                        Re-shuffles the response order so blind labels A/B/C…
                        don't always map to the same models across retries."""
                        if not s_results:
                            return (s_results, s_idx,
                                    "Nothing to retry — click Run first.",
                                    "", gr.update(value="—"),
                                    gr.update(interactive=False),
                                    gr.update(value=None),
                                    gr.update(interactive=False))
                        shuffled = list(s_results)
                        random.shuffle(shuffled)
                        return _nb_show_slide(shuffled, 0,
                                              f"Retrying — showing 1 of {len(shuffled)}",
                                              blind_mode)

                    def on_notebook_score(code, blind_mode, score_str, s_results, s_idx):
                        """Persist this slide's score and advance. Notebook rows
                        land in arena_results tagged 'notebook' so they can be
                        filtered out of training exports if desired.
                        """
                        if not s_results:
                            return (s_results, s_idx,
                                    "Click Run first.", "",
                                    gr.update(value="—"),
                                    gr.update(interactive=False),
                                    gr.update(value=None),
                                    gr.update(interactive=False))
                        if score_str is None:
                            return (s_results, s_idx,
                                    "Select a score first.",
                                    s_results[s_idx]["response"],
                                    gr.update(),
                                    gr.update(interactive=True),
                                    gr.update(),
                                    gr.update())

                        current   = s_results[s_idx]
                        score_val = int(score_str.split(" — ")[0])
                        log_notebook_score(
                            code=code,
                            model_key=current["model_key"],
                            model_display=current["model_display"],
                            response=current["response"],
                            score=score_val,
                            elapsed=current["elapsed"],
                            round_id=current.get("round_id"),
                        )
                        reading = compute_reading(score_val, current["elapsed"])

                        next_idx = s_idx + 1
                        if next_idx >= len(s_results):
                            return (
                                s_results, next_idx,
                                f"✅ All {len(s_results)} responses scored — "
                                f"{label_for_score(score_val)} · Reading: {reading}.",
                                "",
                                gr.update(value="—"),
                                gr.update(interactive=False),
                                gr.update(value=None),
                                gr.update(interactive=True),
                            )

                        nxt = s_results[next_idx]
                        return (
                            s_results, next_idx,
                            f"Logged: {label_for_score(score_val)} · Reading: {reading} "
                            f"— showing {next_idx + 1} of {len(s_results)}",
                            nxt["response"],
                            gr.update(value=_nb_label(nxt, next_idx, blind_mode)),
                            gr.update(interactive=False),
                            gr.update(value=None),
                            gr.update(interactive=True),
                        )

                    # Output bundle used by every nav / score / run handler so
                    # signatures stay consistent.
                    _nb_outputs = [
                        nb_s_results, nb_s_idx, nb_status, nb_response_box,
                        nb_model_label, nb_score_btn, nb_score_radio, nb_retry_btn,
                    ]

                    nb_run_btn.click(
                        on_notebook_run,
                        inputs=[nb_code, nb_state, nb_blind_toggle,
                                *nb_local_cbs, *nb_cloud_cbs],
                        outputs=_nb_outputs + [nb_state],
                    )
                    nb_reset_btn.click(
                        on_notebook_reset,
                        outputs=[nb_state] + _nb_outputs,
                    )
                    nb_prev_btn.click(
                        on_notebook_previous,
                        inputs=[nb_blind_toggle, nb_s_results, nb_s_idx],
                        outputs=_nb_outputs,
                    )
                    nb_next_btn.click(
                        on_notebook_next,
                        inputs=[nb_blind_toggle, nb_s_results, nb_s_idx],
                        outputs=_nb_outputs,
                    )
                    nb_skip_btn.click(
                        on_notebook_skip,
                        outputs=_nb_outputs,
                    )
                    nb_retry_btn.click(
                        on_notebook_retry,
                        inputs=[nb_blind_toggle, nb_s_results, nb_s_idx],
                        outputs=_nb_outputs,
                    )
                    nb_score_btn.click(
                        on_notebook_score,
                        inputs=[nb_code, nb_blind_toggle, nb_score_radio,
                                nb_s_results, nb_s_idx],
                        outputs=_nb_outputs,
                    )

                with gr.Accordion("🖥️ CLI", open=False):
                    gr.Markdown(
                        "Run PowerShell commands from inside Agent Ouija, or "
                        "launch **Claude Code** in a fresh terminal window "
                        "for interactive work.\n\n"
                        "**One-shot commands** — type a command and click "
                        "**Run**; stdout + stderr append to the history box "
                        "below. `cd <path>` is treated as a built-in and "
                        "updates the working directory used by subsequent "
                        "runs.\n\n"
                        "**Launch Claude Code** — spawns a new PowerShell "
                        "window in the current working directory and starts "
                        "`claude`. Claude Code needs an interactive TTY so "
                        "it can't run inside this textbox.\n\n"
                        "_Commands run with a 60-second timeout. Output is "
                        "truncated at 20 KB per run to keep the UI snappy._"
                    )

                    _cli_startup = f"[cwd → {BASE_DIR}]\n"
                    cli_cwd_state = gr.State(str(BASE_DIR))
                    cli_history_state = gr.State(_cli_startup)

                    cli_cwd_display = gr.Markdown(
                        f"**Working directory:** `{BASE_DIR}`"
                    )
                    cli_command = gr.Textbox(
                        label="PowerShell command",
                        placeholder='e.g. git status  |  ollama list  |  cd scripts',
                        lines=1,
                    )
                    with gr.Row():
                        cli_run_btn    = gr.Button("▶ Run", variant="primary")
                        cli_launch_btn = gr.Button("🚀 Launch Claude Code",
                                                    variant="secondary")
                        cli_clear_btn  = gr.Button("Clear Output")
                    cli_output = gr.Textbox(
                        label="Output",
                        value=_cli_startup,
                        lines=18,
                        max_lines=40,
                        interactive=False,
                    )

                    _CLI_MAX_OUTPUT = 20_000
                    _CLI_TIMEOUT_SEC = 60

                    def _cwd_line(cwd: str) -> str:
                        return f"**Working directory:** `{cwd}`"

                    def on_cli_run(command, cwd, history):
                        cmd = (command or "").strip()
                        if not cmd:
                            return (history, history, cwd,
                                    _cwd_line(cwd), "")

                        # Built-in: cd <path> updates state without spawning.
                        if cmd.lower() == "cd" or cmd.lower().startswith("cd "):
                            target = cmd[2:].strip() or str(BASE_DIR)
                            # Strip surrounding quotes if user pasted them.
                            if len(target) >= 2 and target[0] == target[-1] and target[0] in ('"', "'"):
                                target = target[1:-1]

                            has_glob = any(ch in target for ch in "*?[")
                            if has_glob:
                                pattern = target
                                if not Path(pattern).is_absolute():
                                    pattern = str(Path(cwd) / target)
                                matches = sorted(
                                    m for m in glob.glob(pattern)
                                    if Path(m).is_dir()
                                )
                                if not matches:
                                    entry = (
                                        f"$ {cmd}\n"
                                        f"[cd error] no directories match: {target}\n"
                                    )
                                elif len(matches) > 1:
                                    listed = "\n  ".join(matches)
                                    entry = (
                                        f"$ {cmd}\n"
                                        f"[cd error] {len(matches)} matches "
                                        "— be more specific:\n  "
                                        f"{listed}\n"
                                    )
                                else:
                                    cwd = str(Path(matches[0]).resolve())
                                    entry = f"$ {cmd}\n[cwd → {cwd}]\n"
                            else:
                                candidate = Path(target)
                                if not candidate.is_absolute():
                                    candidate = Path(cwd) / target
                                try:
                                    resolved = candidate.resolve(strict=True)
                                except (OSError, RuntimeError) as exc:
                                    entry = (
                                        f"$ {cmd}\n"
                                        f"[cd error] {type(exc).__name__}: {exc}\n"
                                    )
                                else:
                                    if not resolved.is_dir():
                                        entry = (
                                            f"$ {cmd}\n"
                                            f"[cd error] not a directory: {resolved}\n"
                                        )
                                    else:
                                        cwd = str(resolved)
                                        entry = f"$ {cmd}\n[cwd → {cwd}]\n"

                            new_history = entry + "\n" + history if history else entry
                            return (new_history, new_history, cwd,
                                    _cwd_line(cwd), "")

                        try:
                            proc = subprocess.run(
                                ["powershell", "-NoProfile", "-Command", cmd],
                                cwd=cwd,
                                capture_output=True,
                                text=True,
                                timeout=_CLI_TIMEOUT_SEC,
                            )
                            stdout = proc.stdout or ""
                            stderr = proc.stderr or ""
                            combined = stdout
                            if stderr:
                                combined += ("\n" if combined else "") + stderr
                            if len(combined) > _CLI_MAX_OUTPUT:
                                combined = combined[:_CLI_MAX_OUTPUT] + "\n… [output truncated]"
                            entry = f"$ {cmd}\n{combined}".rstrip() + "\n"
                            if proc.returncode != 0:
                                entry += f"[exit code {proc.returncode}]\n"
                        except subprocess.TimeoutExpired:
                            entry = f"$ {cmd}\n[timeout after {_CLI_TIMEOUT_SEC}s]\n"
                        except FileNotFoundError:
                            entry = (
                                f"$ {cmd}\n"
                                "[error] powershell not found on PATH\n"
                            )
                        except Exception as exc:
                            entry = f"$ {cmd}\n[{type(exc).__name__}] {exc}\n"

                        new_history = entry + "\n" + history if history else entry
                        return (new_history, new_history, cwd,
                                _cwd_line(cwd), "")

                    def on_cli_launch_claude_code(cwd, history):
                        try:
                            # `start powershell` returns immediately; the new
                            # window stays open (-NoExit) with `claude` running
                            # inside it so the user can interact with the TUI.
                            subprocess.Popen(
                                f'start powershell -NoExit -Command "cd \'{cwd}\'; claude"',
                                shell=True,
                            )
                            entry = (
                                "$ claude\n"
                                f"[launched Claude Code in new PowerShell window at {cwd}]\n"
                            )
                        except Exception as exc:
                            entry = f"$ claude\n[{type(exc).__name__}] {exc}\n"
                        new_history = entry + "\n" + history if history else entry
                        return new_history, new_history

                    cli_run_btn.click(
                        on_cli_run,
                        inputs=[cli_command, cli_cwd_state, cli_history_state],
                        outputs=[cli_output, cli_history_state,
                                 cli_cwd_state, cli_cwd_display, cli_command],
                    )
                    cli_command.submit(
                        on_cli_run,
                        inputs=[cli_command, cli_cwd_state, cli_history_state],
                        outputs=[cli_output, cli_history_state,
                                 cli_cwd_state, cli_cwd_display, cli_command],
                    )
                    cli_launch_btn.click(
                        on_cli_launch_claude_code,
                        inputs=[cli_cwd_state, cli_history_state],
                        outputs=[cli_output, cli_history_state],
                    )
                    cli_clear_btn.click(
                        lambda cwd: (f"[cwd → {cwd}]\n", f"[cwd → {cwd}]\n"),
                        inputs=[cli_cwd_state],
                        outputs=[cli_output, cli_history_state],
                    )

            with gr.Tab("Files", id="files"):
                _files_return_btn = gr.Button("🔮 Return to Agent Ouija",
                                               variant="secondary")

                with gr.Accordion("📁 File Manager", open=False):
                    gr.Markdown(
                        "### File Repository\n"
                        "Everything currently ingested (or parked for reference) "
                        "in one place. Filenames render as `code spans` — "
                        "click-drag to select and paste into a prompt. Rename "
                        "or delete files from here; document changes "
                        "auto-trigger a RAG re-ingest so ChromaDB stays in "
                        "sync.\n\n"
                        "The **👥 Models** panel below controls the one-model "
                        "rule when uploading pictures or videos. Click the "
                        "header to collapse it once you've configured your "
                        "selection."
                    )

                    # Model picker — matches the Agent Ouija tab shape:
                    # per-model Checkboxes with an All Locals / All
                    # Clouds master toggle per column. Default:
                    # Gemini pre-checked (image editor + strong
                    # vision model; matches the expected file-tab
                    # workflow). Fall back to first cloud if Gemini
                    # isn't in ARENA_MODELS.
                    _files_gemini_default = (
                        "gemini" in CLOUD_KEYS
                    )
                    _files_first_cloud_fallback = (
                        CLOUD_KEYS[0] if CLOUD_KEYS else None
                    )
                    with gr.Accordion("🤖 The Models", open=False):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Local Models")
                                files_local_master = gr.Checkbox(
                                    value=False,
                                    label="**All Locals** (master toggle)",
                                )
                                files_local_cbs = []
                                for k in LOCAL_KEYS:
                                    cfg = AVAILABLE_MODELS[k]
                                    cb = gr.Checkbox(
                                        value=False,
                                        label=f"{cfg['display_name']} — {cfg.get('description', '')}",
                                        interactive=True,
                                    )
                                    files_local_cbs.append(cb)
                            with gr.Column():
                                gr.Markdown("### Cloud Models")
                                files_cloud_master = gr.Checkbox(
                                    value=False,
                                    label="**All Clouds** (master toggle)",
                                )
                                files_cloud_cbs = []
                                for k in CLOUD_KEYS:
                                    cfg = AVAILABLE_MODELS[k]
                                    if _files_gemini_default:
                                        checked = (k == "gemini")
                                    else:
                                        checked = (k == _files_first_cloud_fallback)
                                    cb = gr.Checkbox(
                                        value=checked,
                                        label=f"{cfg['display_name']} — {cfg.get('description', '')}",
                                        interactive=True,
                                    )
                                    files_cloud_cbs.append(cb)

                        # Master cascades — flip every child to the
                        # master's value. No energy gate on this tab.
                        def _files_master_cascade_local(m):
                            return [gr.update(value=bool(m)) for _ in LOCAL_KEYS]

                        def _files_master_cascade_cloud(m):
                            return [gr.update(value=bool(m)) for _ in CLOUD_KEYS]

                        files_local_master.change(
                            _files_master_cascade_local,
                            inputs=[files_local_master],
                            outputs=files_local_cbs,
                            queue=False,
                        )
                        files_cloud_master.change(
                            _files_master_cascade_cloud,
                            inputs=[files_cloud_master],
                            outputs=files_cloud_cbs,
                            queue=False,
                        )

                    repo_status = gr.Markdown("")

                    with gr.Row():
                        repo_refresh_btn = gr.Button("🔄 Refresh Listing")
                        repo_upload_btn  = gr.UploadButton(
                            "📂 Upload Files",
                            file_count="multiple",
                            file_types=[".md", ".txt", ".pdf", ".docx", ".doc",
                                        ".html", ".htm", ".json", ".csv", ".rtf",
                                        "image", "video"],
                        )

                    def _list_dir(folder):
                        if not folder.exists():
                            return []
                        return sorted(
                            p for p in folder.iterdir()
                            if p.is_file() and not p.name.startswith(".")
                        )

                    def _fmt_size(size: int) -> str:
                        for unit, threshold in (("B", 1024),
                                                ("KB", 1024**2),
                                                ("MB", 1024**3)):
                            if size < threshold:
                                div = threshold // 1024
                                return (f"{size} {unit}" if unit == "B"
                                        else f"{size/div:.1f} {unit}")
                        return f"{size/(1024**3):.2f} GB"

                    def _format_listing(paths, empty_msg: str) -> str:
                        if not paths:
                            return f"_{empty_msg}_"
                        lines = []
                        for p in paths:
                            try:
                                size = _fmt_size(p.stat().st_size)
                            except OSError:
                                size = "?"
                            lines.append(f"- `{p.name}` — {size}")
                        return "\n".join(lines)

                    def _refresh_state():
                        """Return (docs_md, docs_update, media_md, media_update)
                        for handler outputs. Raw lists are computed inline at
                        build time below to avoid picking values off Gradio's
                        update object."""
                        docs  = _list_dir(DOCS_DIR)
                        # DOCS_DIR contains a media subdirectory that _list_dir
                        # excludes automatically (only files, not dirs). Good.
                        media = _list_dir(MEDIA_DIR)
                        doc_names   = [p.name for p in docs]
                        media_names = [p.name for p in media]
                        return (
                            _format_listing(docs,  "No documents yet — click Upload Files."),
                            gr.update(choices=doc_names,
                                      value=doc_names[0] if doc_names else None),
                            _format_listing(media, "No media yet — upload an image or video."),
                            gr.update(choices=media_names,
                                      value=media_names[0] if media_names else None),
                        )

                    # Raw initial values — separate from _refresh_state so we
                    # never depend on gr.update()'s internal shape.
                    _init_docs        = _list_dir(DOCS_DIR)
                    _init_media       = _list_dir(MEDIA_DIR)
                    _init_doc_names   = [p.name for p in _init_docs]
                    _init_media_names = [p.name for p in _init_media]

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📄 Documents  ·  `docs/`")
                            docs_listing_md = gr.Markdown(
                                _format_listing(_init_docs,
                                                "No documents yet — click Upload Files."),
                                elem_classes=["file-listing-scroll"],
                            )
                            docs_dropdown   = gr.Dropdown(
                                choices=_init_doc_names,
                                value=_init_doc_names[0] if _init_doc_names else None,
                                label="Select a document",
                                interactive=True,
                            )
                            docs_rename_box = gr.Textbox(
                                label="Rename to",
                                placeholder="new-name.ext",
                            )
                            with gr.Row():
                                docs_rename_btn = gr.Button("Rename")
                                docs_delete_btn = gr.Button("Delete", variant="stop")
                        with gr.Column():
                            gr.Markdown("### 🎨 Media  ·  `docs/media/`")
                            media_listing_md = gr.Markdown(
                                _format_listing(_init_media,
                                                "No media yet — upload an image or video."),
                                elem_classes=["file-listing-scroll"],
                            )
                            media_dropdown   = gr.Dropdown(
                                choices=_init_media_names,
                                value=_init_media_names[0] if _init_media_names else None,
                                label="Select a media file",
                                interactive=True,
                            )
                            media_rename_box = gr.Textbox(
                                label="Rename to",
                                placeholder="new-name.ext",
                            )
                            with gr.Row():
                                media_rename_btn = gr.Button("Rename")
                                media_delete_btn = gr.Button("Delete", variant="stop")

                    # ── Handlers ──────────────────────────────────────────────

                    def on_repo_refresh():
                        docs_md, docs_dd, media_md, media_dd = _refresh_state()
                        return (docs_md, docs_dd, media_md, media_dd,
                                "🔄 Listing refreshed.")

                    def _autofill_rename(name):
                        """Prefill the rename textbox with the currently
                        selected filename so users can just edit part of it."""
                        return gr.update(value=name or "")

                    def _rename_in(folder, current_name, new_name,
                                   auto_reingest: bool):
                        if not current_name:
                            return "❌ Select a file first."
                        new_name = (new_name or "").strip()
                        if not new_name:
                            return "❌ Type a new name first."
                        src = folder / current_name
                        dst = folder / new_name
                        if not src.exists():
                            return f"❌ `{current_name}` not found in {folder.name}/."
                        if dst.exists():
                            return f"❌ `{new_name}` already exists — pick a different name."
                        try:
                            src.rename(dst)
                        except OSError as e:
                            return f"❌ Rename failed: {e}"
                        reingest_note = ""
                        if auto_reingest:
                            ing = run_ingestion()
                            if ing.get("status") == "error":
                                reingest_note = (
                                    f"\n\n⚠️ Re-ingest failed: {ing.get('message')}"
                                )
                            else:
                                reingest_note = (
                                    f"\n\n🔄 RAG re-ingested "
                                    f"({ing.get('documents', 0)} chunks total)."
                                )
                        return f"✅ Renamed `{current_name}` → `{new_name}`.{reingest_note}"

                    def _delete_in(folder, current_name, auto_reingest: bool):
                        if not current_name:
                            return "❌ Select a file first."
                        target = folder / current_name
                        if not target.exists():
                            return f"❌ `{current_name}` not found."
                        try:
                            target.unlink()
                        except OSError as e:
                            return f"❌ Delete failed: {e}"
                        reingest_note = ""
                        if auto_reingest:
                            ing = run_ingestion()
                            if ing.get("status") == "error":
                                reingest_note = (
                                    f"\n\n⚠️ Re-ingest failed: {ing.get('message')}"
                                )
                            else:
                                reingest_note = (
                                    f"\n\n🔄 RAG re-ingested "
                                    f"({ing.get('documents', 0)} chunks total)."
                                )
                        return f"🗑️ Deleted `{current_name}`.{reingest_note}"

                    def on_docs_rename(current, new_name):
                        status = _rename_in(DOCS_DIR, current, new_name, auto_reingest=True)
                        docs_md, docs_dd, media_md, media_dd = _refresh_state()
                        return docs_md, docs_dd, media_md, media_dd, status

                    def on_docs_delete(current):
                        status = _delete_in(DOCS_DIR, current, auto_reingest=True)
                        docs_md, docs_dd, media_md, media_dd = _refresh_state()
                        return docs_md, docs_dd, media_md, media_dd, status

                    def on_media_rename(current, new_name):
                        status = _rename_in(MEDIA_DIR, current, new_name, auto_reingest=False)
                        docs_md, docs_dd, media_md, media_dd = _refresh_state()
                        return docs_md, docs_dd, media_md, media_dd, status

                    def on_media_delete(current):
                        status = _delete_in(MEDIA_DIR, current, auto_reingest=False)
                        docs_md, docs_dd, media_md, media_dd = _refresh_state()
                        return docs_md, docs_dd, media_md, media_dd, status

                    # ── Wirings ───────────────────────────────────────────────

                    _repo_outputs = [
                        docs_listing_md, docs_dropdown,
                        media_listing_md, media_dropdown,
                        repo_status,
                    ]

                    repo_refresh_btn.click(
                        on_repo_refresh,
                        outputs=_repo_outputs,
                    )
                    # Wrapper: convert the per-model boolean state
                    # into (local_displays, cloud_displays) lists so
                    # on_upload's existing signature keeps working.
                    def _files_on_upload(files, *cb_values):
                        n_local = len(LOCAL_KEYS)
                        local_bools = cb_values[:n_local]
                        cloud_bools = cb_values[n_local:]
                        local_displays = [
                            AVAILABLE_MODELS[k]["display_name"]
                            for k, on in zip(LOCAL_KEYS, local_bools) if on
                        ]
                        cloud_displays = [
                            AVAILABLE_MODELS[k]["display_name"]
                            for k, on in zip(CLOUD_KEYS, cloud_bools) if on
                        ]
                        return on_upload(files, local_displays, cloud_displays)

                    repo_upload_btn.upload(
                        _files_on_upload,
                        inputs=[repo_upload_btn,
                                *files_local_cbs, *files_cloud_cbs],
                        outputs=[repo_status],
                    ).then(
                        on_repo_refresh,
                        outputs=_repo_outputs,
                    )

                    # Autofill the rename textbox from the selected filename
                    # so it's easy to edit part of the name.
                    docs_dropdown.change(_autofill_rename,
                                         inputs=[docs_dropdown],
                                         outputs=[docs_rename_box])
                    media_dropdown.change(_autofill_rename,
                                          inputs=[media_dropdown],
                                          outputs=[media_rename_box])

                    docs_rename_btn.click(
                        on_docs_rename,
                        inputs=[docs_dropdown, docs_rename_box],
                        outputs=_repo_outputs,
                    )
                    docs_delete_btn.click(
                        on_docs_delete,
                        inputs=[docs_dropdown],
                        outputs=_repo_outputs,
                    )
                    media_rename_btn.click(
                        on_media_rename,
                        inputs=[media_dropdown, media_rename_box],
                        outputs=_repo_outputs,
                    )
                    media_delete_btn.click(
                        on_media_delete,
                        inputs=[media_dropdown],
                        outputs=_repo_outputs,
                    )


                with gr.Accordion("💬 Work With File(s)", open=False):
                    # ── Work With File(s) — ask about uploaded content ────────

                    gr.Markdown("---\n### 💬 Work With File(s)")
                    gr.Markdown(
                        "Ask the checked model(s) about anything in your file "
                        "repository. Reference documents or media by name "
                        "(e.g. *\"Summarize `spec.pdf` and flag any risks.\"*). "
                        "The one-model rule enforced on the other tabs applies "
                        "here too — if your prompt mentions a picture or video "
                        "and you have more than one model checked, the run is "
                        "refused with the policy banner in the status area.\n\n"
                        "**Editing pictures:** only **Gemini (Nano Banana)** "
                        "can actually EDIT images. If your prompt references "
                        "an image filename + an edit verb "
                        "(*\"add a heart between the two people, wedding.png\"*), "
                        "the request routes to Gemini regardless of which "
                        "models you have checked — your selection is bypassed "
                        "for that call and Gemini's response drops into the "
                        "preview panel below."
                    )

                    files_question = gr.Textbox(
                        label="Question",
                        placeholder=("Ask about your files — reference filenames "
                                     "in backticks or plain text."),
                        lines=5, max_lines=20,
                    )
                    with gr.Row():
                        files_work_btn = gr.Button("Work With File(s)", variant="primary")

                    files_response = gr.Textbox(
                        label="Response",
                        lines=15, max_lines=40,
                        interactive=False,
                    )
                    # Image-edit affordances — preview stays empty
                    # until an edit runs; the Email button ships the
                    # last-edit image to the address in EMAIL_TO.
                    # Note: no show_download_button — Gradio 6.19
                    # doesn't accept that kwarg on gr.Image. Users
                    # can right-click the preview to save.
                    files_edit_preview = gr.Image(
                        label="🎨 Edited output preview",
                        visible=False,
                        interactive=False,
                    )
                    with gr.Row():
                        # Reference a filename in the Work With
                        # File(s) question below and this ships it
                        # via SMTP. Handy on mobile where opening
                        # a folder on Brainy is pointless.
                        files_send_file_btn = gr.Button(
                            "📧 Send File",
                            variant="secondary",
                            scale=0,
                        )

                    def on_files_work(question, *cb_values,
                                      progress=gr.Progress()):
                        # cb_values order matches the click wiring:
                        # LOCAL_KEYS booleans followed by CLOUD_KEYS
                        # booleans. Same shape the AO tab uses now
                        # that this tab's model panel copies AO's
                        # per-model-checkbox format.
                        n_local = len(LOCAL_KEYS)
                        local_bools = cb_values[:n_local]
                        cloud_bools = cb_values[n_local:]
                        local_displays = [
                            AVAILABLE_MODELS[k]["display_name"]
                            for k, on in zip(LOCAL_KEYS, local_bools) if on
                        ]
                        cloud_displays = [
                            AVAILABLE_MODELS[k]["display_name"]
                            for k, on in zip(CLOUD_KEYS, cloud_bools) if on
                        ]

                        # Returns 3 values: (response_text,
                        # status_text, preview_update).
                        _hide_preview = gr.update(visible=False, value=None)
                        if not question.strip():
                            return "", "Enter a question first.", _hide_preview

                        # Image-edit short-circuit — same detection as
                        # the main Ask flow: image filename in prompt +
                        # edit-intent keyword → Gemini Nano Banana
                        # instead of LLM fan-out. Runs BEFORE the
                        # media-rule check so a picture prompt with
                        # multiple models still routes to the edit
                        # path (edit uses one model regardless of
                        # what's checked here).
                        try:
                            from image_edit import (
                                IMAGE_EDIT_EXTS,
                                edit_image as _do_edit_image,
                                is_available as _edit_is_available,
                            )
                            _edit_paths = [
                                p for p in _collect_media_paths(question)
                                if p.suffix.lower() in IMAGE_EDIT_EXTS
                            ]
                            _q_lower = question.lower()
                            if _edit_paths and any(
                                kw in _q_lower for kw in _EDIT_INTENT_KEYWORDS
                            ):
                                if not _edit_is_available():
                                    return (
                                        "",
                                        "🎨 Image edit requested but Gemini "
                                        "Nano Banana isn't available. Ensure "
                                        "GEMINI_API_KEY is in `.env` and "
                                        "`pip install google-genai Pillow` "
                                        "has run.",
                                        _hide_preview,
                                    )
                                src = _edit_paths[0]
                                # If Gemini isn't in the user's
                                # current selection, prepend a
                                # heads-up. The edit still runs — the
                                # short-circuit uses Gemini regardless
                                # of what's checked — but the user
                                # should know their model choice was
                                # bypassed for this call.
                                _gemini_dn = (
                                    AVAILABLE_MODELS.get("gemini", {})
                                    .get("display_name") or "Gemini"
                                )
                                _selected = (local_displays or []) + (cloud_displays or [])
                                _bypass_note = ""
                                if _gemini_dn not in _selected:
                                    _bypass_note = (
                                        f"ℹ️ **{_gemini_dn}** (Nano Banana) is "
                                        "the only model that edits images, so "
                                        "your current selection was bypassed "
                                        "for this call.\n\n"
                                    )
                                t0 = time.perf_counter()
                                out = _do_edit_image(src, question)
                                elapsed = round(time.perf_counter() - t0, 3)
                                if out is None:
                                    return (
                                        "",
                                        f"🎨 Image edit failed for "
                                        f"`{src.name}`. Check the console "
                                        "for a Gemini error.",
                                        _hide_preview,
                                    )
                                body = (
                                    f"{_bypass_note}"
                                    f"🎨 **Edited image saved.**\n\n"
                                    f"- Source: `{src.name}`\n"
                                    f"- Output: `{out.name}`\n"
                                    f"- Location: `docs/media/`\n"
                                    f"- Editor: **{_gemini_dn}** (Nano Banana)\n"
                                    f"- Instruction: _{question}_"
                                )
                                return (
                                    body,
                                    f"🎨 Image edit complete in {elapsed:.1f}s "
                                    f"— `{out.name}` saved (via {_gemini_dn}).",
                                    gr.update(visible=True, value=str(out)),
                                )
                        except ImportError:
                            pass

                        selected_displays = (local_displays or []) + (cloud_displays or [])
                        if not selected_displays:
                            return "", "Pick at least one model in the 👥 Models panel above.", _hide_preview
                        media_error = _media_rule_error(question, len(selected_displays))
                        if media_error:
                            return "", media_error, _hide_preview

                        display_to_key = {
                            AVAILABLE_MODELS[m]["display_name"]: m
                            for m in ARENA_MODELS if m in AVAILABLE_MODELS
                        }
                        selected_keys = [display_to_key[d] for d in selected_displays
                                         if d in display_to_key]
                        if not selected_keys:
                            return "", "Selected model(s) not in ARENA_MODELS.", _hide_preview

                        progress(0, desc="Working…")
                        results = run_models(question, selected_keys, progress=progress)

                        parts = []
                        ok_count = 0
                        for r in results:
                            header = f"--- {r['model_display']}  ·  {r['elapsed']}s ---"
                            if r.get("response"):
                                parts.append(f"{header}\n{r['response']}")
                                ok_count += 1
                            else:
                                parts.append(f"{header}\n(unavailable)")
                        status = f"✅ Ran {ok_count} of {len(results)} model(s)."
                        return (
                            "\n\n".join(parts) if parts else "(no response)",
                            status,
                            _hide_preview,
                        )

                    files_work_btn.click(
                        on_files_work,
                        inputs=[files_question,
                                *files_local_cbs, *files_cloud_cbs],
                        outputs=[files_response, repo_status,
                                  files_edit_preview],
                    )

                    def _on_files_send_file(question):
                        """Files-tab twin of ao_send_file — reads the
                        Work With File(s) textbox, ships every
                        referenced repo file as one email."""
                        from emailer import send_files
                        q = (question or "").strip()
                        if not q:
                            return (
                                "📧 Type a question that references a "
                                "filename (e.g. `spec.pdf` or "
                                "`dog.png`) first, then click 📧 Send "
                                "File."
                            )
                        paths = _collect_referenced_files(q)
                        if not paths:
                            return (
                                "📧 No matching file in `docs/` or "
                                "`docs/media/`. Reference the exact "
                                "filename and click 📧 Send File again."
                            )
                        r = send_files([str(p) for p in paths])
                        icon = "✓" if r.get("status") == "success" else "❌"
                        return f"{icon} {r.get('message', '')}"

                    files_send_file_btn.click(
                        _on_files_send_file,
                        inputs=[files_question],
                        outputs=[repo_status],
                        queue=False,
                    )

            with gr.Tab("Persona", id="persona"):
                _persona_return_btn = gr.Button("🔮 Return to Agent Ouija",
                                                 variant="secondary")
                gr.Markdown(
                    "### Persona\n"
                    "Vertical stack — one compact row for the required "
                    "`enhanced_persona_main.md`, then three full-width "
                    "panels beneath it.\n\n"
                    "1. **Enhanced Persona Main** — one file, required for "
                    "enhanced mode; new upload replaces the old.\n"
                    "2. **Enhanced Persona Files** — optional additional "
                    "personalities. `persona_<name>.md` routes to `prompts/`; "
                    "anything else routes to `prompts/episodes/`.\n"
                    "3. **Copyright Files** — reference material chunked into "
                    "the private ChromaDB collection (retrieved per question).\n"
                    "4. **Voice** — Piper voice picker + speaker-id support "
                    "for multi-speaker voices (libritts, vctk).\n\n"
                    "_All four folders are gitignored — nothing here leaves the rig._"
                )

                # ── Shared helpers ─────────────────────────────────────────
                def _persona_list_dir(folder):
                    if not folder.exists():
                        return []
                    return sorted(
                        p for p in folder.iterdir()
                        if p.is_file() and not p.name.startswith(".")
                    )

                def _persona_fmt_size(size: int) -> str:
                    for unit, threshold in (("B", 1024),
                                            ("KB", 1024**2),
                                            ("MB", 1024**3)):
                        if size < threshold:
                            div = threshold // 1024
                            return (f"{size} {unit}" if unit == "B"
                                    else f"{size/div:.1f} {unit}")
                    return f"{size/(1024**3):.2f} GB"

                def _persona_format_listing(paths, empty_msg: str) -> str:
                    if not paths:
                        return f"_{empty_msg}_"
                    lines = []
                    for p in paths:
                        try:
                            size = _persona_fmt_size(p.stat().st_size)
                        except OSError:
                            size = "?"
                        try:
                            rel = p.relative_to(BASE_DIR / "prompts")
                            display_name = str(rel).replace("\\", "/")
                        except ValueError:
                            display_name = p.name
                        lines.append(f"- `{display_name}` — {size}")
                    return "\n".join(lines)

                ENHANCED_MAIN_PATH = BASE_DIR / "prompts" / "enhanced_persona_main.md"
                ENHANCED_EPISODES_DIR = BASE_DIR / "prompts" / "episodes"

                (BASE_DIR / "prompts").mkdir(parents=True, exist_ok=True)
                ENHANCED_EPISODES_DIR.mkdir(parents=True, exist_ok=True)
                PRIVATE_CORPUS_EPISODES_DIR.mkdir(parents=True, exist_ok=True)
                TTS_VOICES_DIR.mkdir(parents=True, exist_ok=True)

                def _enhanced_files_paths():
                    prompts_dir = BASE_DIR / "prompts"
                    persona_files = sorted(prompts_dir.glob("persona_*.md"))
                    episode_files = (sorted(ENHANCED_EPISODES_DIR.iterdir())
                                     if ENHANCED_EPISODES_DIR.exists() else [])
                    episode_files = [p for p in episode_files
                                     if p.is_file() and not p.name.startswith(".")]
                    return persona_files + episode_files

                def _main_status_line() -> str:
                    if ENHANCED_MAIN_PATH.exists():
                        try:
                            size = _persona_fmt_size(ENHANCED_MAIN_PATH.stat().st_size)
                        except OSError:
                            size = "?"
                        return (
                            "**📜 Enhanced Persona Main:** 🟢 "
                            f"`enhanced_persona_main.md` — {size}"
                        )
                    return (
                        "**📜 Enhanced Persona Main:** ⚪ not uploaded — "
                        "required to give enhanced mode a base file"
                    )

                def _persona_status_line() -> str:
                    enhanced_on = "🟢 enhanced ON" if ENHANCED_ACTIVE_FLAG.exists() else "⚪ enhanced off"
                    corpus_on   = "🟢 corpus ON"   if PRIVATE_CORPUS_ACTIVE_FLAG.exists() else "⚪ corpus off"
                    return f"**{enhanced_on}**  ·  **{corpus_on}**"

                # Precompute initial state
                _init_enhanced     = _enhanced_files_paths()
                _init_corpus       = _persona_list_dir(PRIVATE_CORPUS_EPISODES_DIR)
                _init_enh_names    = [
                    (str(p.relative_to(BASE_DIR / "prompts")).replace("\\", "/"))
                    for p in _init_enhanced
                ]
                _init_corpus_names = [p.name for p in _init_corpus]

                # ── Section 1: Enhanced Persona Main ──────────────────────
                with gr.Accordion("📜 Enhanced Persona Main", open=False):
                    gr.Markdown(
                        "One required file — `prompts/enhanced_persona_main.md`. "
                        "Ships as the always-on layer both locals and clouds see "
                        "when enhanced mode is active. A new upload replaces the "
                        "existing file."
                    )
                    with gr.Row():
                        # gr.Markdown doesn't accept scale= across Gradio
                        # versions — wrap in a Column so the layout hint lives
                        # on the container instead.
                        with gr.Column(scale=3):
                            main_status_md = gr.Markdown(_main_status_line())
                        main_upload_btn = gr.UploadButton(
                            "📂 Upload Enhanced Main (.md)",
                            file_count="single",
                            file_types=[".md", ".txt"],
                            scale=1,
                        )
                        main_delete_btn = gr.Button(
                            "Delete Main", variant="stop", scale=1,
                        )

                # ── Section 2: Enhanced Persona Copyright Files — Prompts ─
                with gr.Accordion(
                    "📚 Enhanced Persona Copyright Files - Prompts",
                    open=False,
                ):
                    gr.Markdown(
                        "Always-on layered personalities. `persona_<name>.md` "
                        "files route to `prompts/` (top-level); anything else "
                        "routes to `prompts/episodes/`. Loaded into every cloud "
                        "model call when enhanced mode is on — locals stay on "
                        "the base file only."
                    )
                    enhanced_listing_md = gr.Markdown(
                        _persona_format_listing(
                            _init_enhanced,
                            "No enhanced persona files yet — upload persona_*.md or episode files.",
                        ),
                        elem_classes=["file-listing-scroll"],
                    )
                    with gr.Row():
                        enhanced_upload_btn = gr.UploadButton(
                            "📂 Upload Enhanced Files",
                            file_count="multiple",
                            file_types=[".md", ".txt"],
                            scale=2,
                        )
                        enhanced_dropdown = gr.Dropdown(
                            choices=_init_enh_names,
                            value=_init_enh_names[0] if _init_enh_names else None,
                            label="Delete file",
                            interactive=True,
                            scale=2,
                        )
                        enhanced_delete_btn = gr.Button(
                            "Delete", variant="stop", scale=1,
                        )

                # ── Section 3: Enhanced Persona Copyright Files — RAG ────
                with gr.Accordion(
                    "📖 Enhanced Persona Copyright Files - RAG",
                    open=False,
                ):
                    gr.Markdown(
                        "Retrieval-based reference material chunked into the "
                        "private ChromaDB collection (`Copyright/episodes/`). "
                        "The private-corpus toggle in Toggles & Actions "
                        "below (or in Agent Ouija's Control Center) turns "
                        "per-question retrieval on. Any file too big to fit "
                        "in the always-on Prompts stack belongs here — the "
                        "corpus surfaces only the ~3 most-relevant chunks "
                        "per question."
                    )
                    corpus_listing_md = gr.Markdown(
                        _persona_format_listing(
                            _init_corpus,
                            "No reference files yet — upload transcripts or character docs.",
                        ),
                        elem_classes=["file-listing-scroll"],
                    )
                    with gr.Row():
                        corpus_upload_btn = gr.UploadButton(
                            "📂 Upload Copyright Files",
                            file_count="multiple",
                            file_types=[".md", ".txt", ".pdf", ".docx", ".doc",
                                        ".html", ".htm", ".json", ".csv", ".rtf"],
                            scale=2,
                        )
                        corpus_dropdown = gr.Dropdown(
                            choices=_init_corpus_names,
                            value=_init_corpus_names[0] if _init_corpus_names else None,
                            label="Delete file",
                            interactive=True,
                            scale=2,
                        )
                        corpus_delete_btn = gr.Button(
                            "Delete", variant="stop", scale=1,
                        )

                # ── Section 4: Voice ──────────────────────────────────────
                # Helpers defined at the tab scope (outside the accordion)
                # so demo.load() and other refresh paths can still call
                # them. Python's `with` doesn't scope function defs, but
                # keeping them here makes the ordering obvious.
                def _available_voices() -> list[str]:
                    if not TTS_VOICES_DIR.is_dir():
                        return []
                    ids = []
                    for onnx in sorted(TTS_VOICES_DIR.glob("*.onnx")):
                        cfg = onnx.with_suffix(".onnx.json")
                        if cfg.exists():
                            ids.append(onnx.stem)
                    return ids

                def _current_voice_id_and_speaker() -> tuple[str, int]:
                    if ACTIVE_VOICE_PATH.exists():
                        try:
                            content = ACTIVE_VOICE_PATH.read_text(
                                encoding="utf-8").strip()
                        except OSError:
                            content = ""
                        if content:
                            if "|" in content:
                                name, _, sid = content.partition("|")
                                try:
                                    return name.strip(), int(sid.strip())
                                except ValueError:
                                    return name.strip(), 0
                            return content, 0
                    return TTS_VOICE, 0

                def _voice_status_line() -> str:
                    current, sid = _current_voice_id_and_speaker()
                    voices  = _available_voices()
                    on_disk = "🟢 loaded" if current in voices else "⚠️  missing files"
                    speaker_part = f" · speaker {sid}" if sid else ""
                    return (
                        f"**Active:** `{current}`{speaker_part} — {on_disk}  ·  "
                        f"{len(voices)} voice(s) available"
                    )

                _init_voice, _init_speaker = _current_voice_id_and_speaker()

                with gr.Accordion("🔊 Voice", open=False):
                    gr.Markdown(
                        "Piper `.onnx` + `.onnx.json` pair. Multi-speaker "
                        "voices (libritts, vctk) need a `Speaker ID` — 0 for "
                        "single-speaker voices.\n\n"
                        "Get files: [piper-samples](https://rhasspy.github.io/piper-samples/) "
                        "→ [piper-voices HF](https://huggingface.co/rhasspy/piper-voices)."
                    )
                    voice_status = gr.Markdown(_voice_status_line())
                    with gr.Row():
                        voice_dropdown = gr.Dropdown(
                            choices=_available_voices() or [_init_voice],
                            value=_init_voice,
                            label="Available voices",
                            interactive=True,
                            allow_custom_value=False,
                            scale=3,
                        )
                        voice_speaker_input = gr.Number(
                            value=_init_speaker,
                            label="Speaker ID (0 = default)",
                            precision=0,
                            minimum=0,
                            scale=1,
                        )
                    with gr.Row():
                        voice_set_btn = gr.Button(
                            "Set As Default", variant="primary",
                        )
                        voice_upload_btn = gr.UploadButton(
                            "📂 Upload voice files",
                            file_count="multiple",
                            file_types=[".onnx", ".json"],
                        )
                        voice_refresh_btn = gr.Button("🔄 Refresh")

                # ── Toggles + Preview / Re-ingest ─────────────────────────
                gr.Markdown("---\n### Toggles & Actions")
                gr.Markdown(
                    "_Note: **Groq** is excluded from both the enhanced "
                    "persona overlay and the private corpus even when "
                    "these toggles are on. Groq's free-tier 8k-TPM cap "
                    "can't fit the persona stack (~4k tokens) alongside "
                    "the usual peer prompt without hitting a `Request "
                    "too large` 429, so it stays on the base "
                    "system_core prompt to remain a viable peer. Change "
                    "the `skip_enhanced_persona` field on the Groq "
                    "entry in `config.py` if you'd rather Groq drop "
                    "out entirely._"
                )
                with gr.Row():
                    enhanced_active_toggle = gr.Checkbox(
                        value=ENHANCED_ACTIVE_FLAG.exists(),
                        label="Enable enhanced persona for all models",
                    )
                    corpus_active_toggle = gr.Checkbox(
                        value=PRIVATE_CORPUS_ACTIVE_FLAG.exists(),
                        label="Use private corpus in retrieval (per-question RAG)",
                    )
                with gr.Row():
                    persona_preview_btn = gr.Button(
                        "👁 Preview assembled system prompt",
                    )
                    corpus_reingest_btn = gr.Button(
                        "🔄 Re-ingest Copyright corpus", variant="primary",
                    )

                persona_status = gr.Markdown(_persona_status_line())
                persona_preview = gr.Textbox(
                    label="Assembled system prompt (with current toggles applied)",
                    lines=12, max_lines=40,
                    interactive=False,
                    visible=False,
                )
                corpus_ingest_output = gr.Textbox(
                    label="Ingest output",
                    lines=8, max_lines=20,
                    interactive=False,
                    visible=False,
                )

                # ── Refresh helpers ───────────────────────────────────────
                def _refresh_main():
                    return _main_status_line(), _persona_status_line()

                def _refresh_enhanced():
                    files = _enhanced_files_paths()
                    names = [
                        (str(p.relative_to(BASE_DIR / "prompts")).replace("\\", "/"))
                        for p in files
                    ]
                    return (
                        _persona_format_listing(
                            files,
                            "No enhanced persona files yet — upload persona_*.md or episode files.",
                        ),
                        gr.update(choices=names,
                                  value=names[0] if names else None),
                        _persona_status_line(),
                    )

                def _refresh_corpus():
                    files = _persona_list_dir(PRIVATE_CORPUS_EPISODES_DIR)
                    names = [p.name for p in files]
                    return (
                        _persona_format_listing(
                            files,
                            "No reference files yet — upload transcripts or character docs.",
                        ),
                        gr.update(choices=names,
                                  value=names[0] if names else None),
                        _persona_status_line(),
                    )

                # ── Handlers ──────────────────────────────────────────────

                def on_main_upload(file):
                    if file is None:
                        return _refresh_main()
                    src = Path(file.name)
                    try:
                        shutil.copyfile(src, ENHANCED_MAIN_PATH)
                    except OSError as e:
                        log.warning("Enhanced-main upload failed for %s: %s", src, e)
                    return _refresh_main()

                def on_main_delete():
                    try:
                        ENHANCED_MAIN_PATH.unlink(missing_ok=True)
                    except OSError as e:
                        log.warning("Enhanced-main delete failed: %s", e)
                    return _refresh_main()

                def on_enhanced_upload(files):
                    if not files:
                        return _refresh_enhanced()
                    for f in files:
                        src = Path(f.name)
                        if src.stem.startswith("persona_"):
                            target = (BASE_DIR / "prompts" / src.name)
                        else:
                            target = ENHANCED_EPISODES_DIR / src.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copyfile(src, target)
                        except OSError as e:
                            log.warning("Enhanced upload failed for %s: %s", src, e)
                    return _refresh_enhanced()

                def on_enhanced_delete(name):
                    if not name:
                        return _refresh_enhanced()
                    normalized = name.replace("\\", "/")
                    target = (BASE_DIR / "prompts" / normalized)
                    try:
                        target.unlink(missing_ok=True)
                    except OSError as e:
                        log.warning("Enhanced delete failed for %s: %s", target, e)
                    return _refresh_enhanced()

                def on_corpus_upload(files):
                    if not files:
                        return _refresh_corpus()
                    for f in files:
                        src = Path(f.name)
                        target = PRIVATE_CORPUS_EPISODES_DIR / src.name
                        try:
                            shutil.copyfile(src, target)
                        except OSError as e:
                            log.warning("Corpus upload failed for %s: %s", src, e)
                    return _refresh_corpus()

                def on_corpus_delete(name):
                    if not name:
                        return _refresh_corpus()
                    target = PRIVATE_CORPUS_EPISODES_DIR / name
                    try:
                        target.unlink(missing_ok=True)
                    except OSError as e:
                        log.warning("Corpus delete failed for %s: %s", target, e)
                    return _refresh_corpus()

                def on_enhanced_toggle(active):
                    if active:
                        ENHANCED_ACTIVE_FLAG.parent.mkdir(parents=True, exist_ok=True)
                        ENHANCED_ACTIVE_FLAG.write_text("on", encoding="utf-8")
                    else:
                        try:
                            ENHANCED_ACTIVE_FLAG.unlink()
                        except FileNotFoundError:
                            pass
                    return _persona_status_line()

                def on_corpus_toggle(active):
                    if active:
                        PRIVATE_CORPUS_ACTIVE_FLAG.parent.mkdir(parents=True, exist_ok=True)
                        PRIVATE_CORPUS_ACTIVE_FLAG.write_text("on", encoding="utf-8")
                    else:
                        try:
                            PRIVATE_CORPUS_ACTIVE_FLAG.unlink()
                        except FileNotFoundError:
                            pass
                    return _persona_status_line()

                def on_persona_preview():
                    text = get_prompt_loader(DEFAULT_PERSONA_MODE).load()
                    return gr.update(visible=True, value=text or "(empty)")

                def on_corpus_reingest():
                    try:
                        proc = subprocess.run(
                            [sys.executable,
                             str(BASE_DIR / "scripts" / "private_corpus.py"),
                             "--ingest"],
                            cwd=str(BASE_DIR),
                            capture_output=True, text=True, timeout=300,
                        )
                        combined = (proc.stdout or "") + (proc.stderr or "")
                        if proc.returncode != 0:
                            combined += f"\n[exit code {proc.returncode}]"
                    except subprocess.TimeoutExpired:
                        combined = "[timeout after 300s — ingest still running?]"
                    except Exception as e:
                        combined = f"[{type(e).__name__}] {e}"
                    return gr.update(visible=True, value=combined or "(no output)")

                def on_voice_upload(files):
                    if not files:
                        return (
                            gr.update(choices=_available_voices() or
                                      [_current_voice_id_and_speaker()[0]],
                                      value=_current_voice_id_and_speaker()[0]),
                            _voice_status_line(),
                        )
                    TTS_VOICES_DIR.mkdir(parents=True, exist_ok=True)
                    for f in files:
                        src = Path(f.name)
                        target = TTS_VOICES_DIR / src.name
                        try:
                            shutil.copyfile(src, target)
                        except OSError as e:
                            log.warning("Voice upload failed for %s: %s", src, e)
                    voices = _available_voices() or [_current_voice_id_and_speaker()[0]]
                    return (
                        gr.update(choices=voices,
                                  value=_current_voice_id_and_speaker()[0]),
                        _voice_status_line() + " — upload complete. Pick the "
                        "voice from the dropdown, set Speaker ID if it's "
                        "multi-speaker, then click **Set As Default**.",
                    )

                def on_voice_set(choice, speaker):
                    if not choice:
                        return _voice_status_line() + " — pick a voice first."
                    try:
                        speaker_int = int(speaker or 0)
                    except (TypeError, ValueError):
                        speaker_int = 0
                    set_active_voice(choice, speaker_id=speaker_int)
                    return _voice_status_line() + " — updated."

                def on_voice_refresh():
                    voices = _available_voices() or [_current_voice_id_and_speaker()[0]]
                    current, _ = _current_voice_id_and_speaker()
                    return (
                        gr.update(choices=voices, value=current),
                        _voice_status_line(),
                    )

                # ── Wire ──────────────────────────────────────────────────

                main_upload_btn.upload(
                    on_main_upload, inputs=[main_upload_btn],
                    outputs=[main_status_md, persona_status],
                )
                main_delete_btn.click(
                    on_main_delete,
                    outputs=[main_status_md, persona_status],
                )

                enhanced_upload_btn.upload(
                    on_enhanced_upload, inputs=[enhanced_upload_btn],
                    outputs=[enhanced_listing_md, enhanced_dropdown, persona_status],
                )
                enhanced_delete_btn.click(
                    on_enhanced_delete, inputs=[enhanced_dropdown],
                    outputs=[enhanced_listing_md, enhanced_dropdown, persona_status],
                )

                corpus_upload_btn.upload(
                    on_corpus_upload, inputs=[corpus_upload_btn],
                    outputs=[corpus_listing_md, corpus_dropdown, persona_status],
                )
                corpus_delete_btn.click(
                    on_corpus_delete, inputs=[corpus_dropdown],
                    outputs=[corpus_listing_md, corpus_dropdown, persona_status],
                )

                enhanced_active_toggle.change(
                    on_enhanced_toggle, inputs=[enhanced_active_toggle],
                    outputs=[persona_status],
                )
                corpus_active_toggle.change(
                    on_corpus_toggle, inputs=[corpus_active_toggle],
                    outputs=[persona_status],
                )

                persona_preview_btn.click(
                    on_persona_preview, outputs=[persona_preview],
                )
                corpus_reingest_btn.click(
                    on_corpus_reingest, outputs=[corpus_ingest_output],
                )

                voice_upload_btn.upload(
                    on_voice_upload, inputs=[voice_upload_btn],
                    outputs=[voice_dropdown, voice_status],
                )
                voice_set_btn.click(
                    on_voice_set,
                    inputs=[voice_dropdown, voice_speaker_input],
                    outputs=[voice_status],
                )
                voice_refresh_btn.click(
                    on_voice_refresh,
                    outputs=[voice_dropdown, voice_status],
                )

                # ── Refresh from disk on every page load ──────────────────
                # gr.Markdown / gr.Dropdown values are computed once at
                # Blocks build time. Uploads mutate them for the current
                # session, but a browser refresh reverts each component to
                # its build-time snapshot. demo.load() re-runs on every
                # page load so the four Persona sections always mirror what
                # is currently on disk.
                def _persona_reload_all():
                    main_status_new, persona_status_new = _refresh_main()
                    enh_md, enh_dd, _ = _refresh_enhanced()
                    cor_md, cor_dd, _ = _refresh_corpus()
                    current_voice, current_speaker = _current_voice_id_and_speaker()
                    voice_choices = _available_voices() or [current_voice]
                    return (
                        main_status_new,
                        enh_md, enh_dd,
                        cor_md, cor_dd,
                        _voice_status_line(),
                        gr.update(choices=voice_choices, value=current_voice),
                        current_speaker,
                        persona_status_new,
                        # Also refresh the toggle values in case the flag
                        # files were touched outside the UI.
                        ENHANCED_ACTIVE_FLAG.exists(),
                        PRIVATE_CORPUS_ACTIVE_FLAG.exists(),
                    )

                demo.load(
                    _persona_reload_all,
                    outputs=[
                        main_status_md,
                        enhanced_listing_md, enhanced_dropdown,
                        corpus_listing_md, corpus_dropdown,
                        voice_status, voice_dropdown, voice_speaker_input,
                        persona_status,
                        enhanced_active_toggle, corpus_active_toggle,
                    ],
                )

            with gr.Tab("Data", id="data"):
                _stats_return_btn = gr.Button("🔮 Return to Agent Ouija",
                                               variant="secondary")

                # ── 💰 Billing Details ─────────────────────────────
                with gr.Accordion("💰 Billing Details", open=True):
                    gr.Markdown(
                        "Cost breakdown from `arena_results` — every scored "
                        "row carries its own `input_cost` / `output_cost` "
                        "so these numbers are what has actually posted to "
                        "each provider across the rig's entire history. "
                        "Auto-loads on tab open; click Refresh Billing to "
                        "recompute after new answers land."
                    )

                    def _billing_summary_query():
                        """Produce four views of the same underlying
                        arena_results cost data:
                          - Lifetime grand total (headline)
                          - Since last training checkpoint (delta)
                          - Last 30 / 7 days (recent-activity roll-up)
                          - Per-model + per-provider tables (drill-down)
                        Historical, not session-scoped — the user asked
                        explicitly for costs beyond the most recent
                        training session."""
                        try:
                            with connection_scope() as conn:
                                # Lifetime + recency totals in one pass.
                                life = conn.execute(
                                    "SELECT COALESCE(SUM(input_cost + output_cost), 0.0),"
                                    " COALESCE(SUM(input_tokens + output_tokens), 0),"
                                    " COUNT(*), MIN(timestamp), MAX(timestamp)"
                                    " FROM arena_results"
                                ).fetchone()
                                last30 = conn.execute(
                                    "SELECT COALESCE(SUM(input_cost + output_cost), 0.0),"
                                    " COUNT(*) FROM arena_results"
                                    " WHERE timestamp >= datetime('now', '-30 days')"
                                ).fetchone()
                                last7 = conn.execute(
                                    "SELECT COALESCE(SUM(input_cost + output_cost), 0.0),"
                                    " COUNT(*) FROM arena_results"
                                    " WHERE timestamp >= datetime('now', '-7 days')"
                                ).fetchone()
                                # Since last checkpoint (delta the user
                                # would think of as "current session's
                                # progression"). Empty checkpoint table
                                # → equal to lifetime total.
                                cp = conn.execute(
                                    "SELECT MAX(timestamp) FROM training_checkpoints"
                                ).fetchone()
                                cp_ts = cp[0] if cp else None
                                if cp_ts:
                                    since_cp = conn.execute(
                                        "SELECT COALESCE(SUM(input_cost + output_cost), 0.0),"
                                        " COUNT(*) FROM arena_results"
                                        " WHERE timestamp > ?", (cp_ts,),
                                    ).fetchone()
                                else:
                                    since_cp = (life[0], life[2])
                                # Per-model rollup.
                                model_rows = conn.execute(
                                    "SELECT model_display, model_type, COUNT(*),"
                                    " COALESCE(SUM(input_cost + output_cost), 0.0),"
                                    " COALESCE(SUM(input_tokens + output_tokens), 0)"
                                    " FROM arena_results GROUP BY model_display, model_type"
                                    " ORDER BY 4 DESC"
                                ).fetchall()
                        except Exception as e:
                            return (
                                f"_Billing summary unavailable: "
                                f"{type(e).__name__}: {e}_"
                            )

                        if life[2] == 0:
                            return (
                                "_No scored responses yet — costs will "
                                "appear once the arena_results table has "
                                "rows (any Score And Next click writes one)._"
                            )

                        lifetime_cost   = float(life[0] or 0.0)
                        lifetime_toks   = int(life[1] or 0)
                        lifetime_rows   = int(life[2] or 0)
                        first_seen      = life[3] or "?"
                        last_seen       = life[4] or "?"
                        last30_cost     = float(last30[0] or 0.0)
                        last30_rows     = int(last30[1] or 0)
                        last7_cost      = float(last7[0] or 0.0)
                        last7_rows      = int(last7[1] or 0)
                        since_cp_cost   = float(since_cp[0] or 0.0)
                        since_cp_rows   = int(since_cp[1] or 0)

                        out: list[str] = []
                        out.append(
                            f"### 💰 Lifetime total: **${lifetime_cost:.4f}**  "
                            f"·  {lifetime_rows:,} rows  ·  "
                            f"{lifetime_toks:,} tokens"
                        )
                        out.append(
                            f"_First row: {first_seen}  ·  "
                            f"Latest row: {last_seen}_"
                        )
                        out.append("")
                        out.append("| Window | Cost (USD) | Rows |")
                        out.append("|--------|-----------:|-----:|")
                        out.append(
                            f"| Last 7 days | ${last7_cost:.4f} | "
                            f"{last7_rows:,} |"
                        )
                        out.append(
                            f"| Last 30 days | ${last30_cost:.4f} | "
                            f"{last30_rows:,} |"
                        )
                        cp_label = (
                            f"Since checkpoint {cp_ts}" if cp_ts
                            else "Since checkpoint (none set yet)"
                        )
                        out.append(
                            f"| {cp_label} | ${since_cp_cost:.4f} | "
                            f"{since_cp_rows:,} |"
                        )
                        out.append(
                            f"| **Lifetime** | **${lifetime_cost:.4f}** | "
                            f"**{lifetime_rows:,}** |"
                        )
                        out.append("")

                        # Per-model breakdown (drill-down)
                        out.append("**Per-model lifetime rollup**")
                        out.append("")
                        out.append(
                            "| Model | Type | Rows | Tokens | Cost (USD) |"
                        )
                        out.append(
                            "|-------|------|-----:|-------:|-----------:|"
                        )
                        # Provider is inferred from AVAILABLE_MODELS
                        # lookup by display_name → provider field.
                        display_to_provider = {
                            AVAILABLE_MODELS[k].get("display_name"):
                                AVAILABLE_MODELS[k].get("provider")
                                or ("Local" if AVAILABLE_MODELS[k].get("type") == "local"
                                     else "?")
                            for k in ARENA_MODELS if k in AVAILABLE_MODELS
                        }
                        by_provider: dict = {}
                        for r in model_rows:
                            disp, mtype, n, cost, tok = r
                            disp = disp or "?"
                            mtype = mtype or "?"
                            n = int(n or 0)
                            cost = float(cost or 0.0)
                            tok = int(tok or 0)
                            out.append(
                                f"| {disp} | {mtype} | {n:,} | {tok:,} | "
                                f"${cost:.4f} |"
                            )
                            prov = display_to_provider.get(disp, "?")
                            slot = by_provider.setdefault(
                                prov, {"cost": 0.0, "rows": 0, "tok": 0},
                            )
                            slot["cost"] += cost
                            slot["rows"] += n
                            slot["tok"]  += tok
                        out.append("")

                        # Per-provider breakdown (roll-up)
                        out.append("**Per-provider lifetime rollup**")
                        out.append("")
                        out.append("| Provider | Rows | Tokens | Cost (USD) |")
                        out.append("|----------|-----:|-------:|-----------:|")
                        for prov, agg in sorted(by_provider.items(),
                                                 key=lambda kv: -kv[1]["cost"]):
                            out.append(
                                f"| {prov} | {agg['rows']:,} | "
                                f"{agg['tok']:,} | ${agg['cost']:.4f} |"
                            )
                        return "\n".join(out)

                    # Auto-populate on tab load so users don't have to
                    # click Refresh Billing to see costs. The click
                    # handler on the button below recomputes on demand.
                    billing_summary_md = gr.Markdown(_billing_summary_query())
                    billing_refresh_btn = gr.Button(
                        "🔄 Refresh Billing", variant="primary",
                    )
                    gr.Markdown("**Provider dashboards** — click to open in a new tab:")
                    _bl_html = "\n".join(
                        f'- <a href="{url}" target="_blank" rel="noopener">{name}</a>'
                        for name, url in BILLING_LINKS.items()
                    )
                    gr.Markdown(_bl_html)

                    billing_refresh_btn.click(
                        _billing_summary_query,
                        outputs=[billing_summary_md],
                        queue=False,
                    )

                # ── 📊 Local Model Training Data (existing stats) ──
                with gr.Accordion("📊 Local Model Training Data", open=False):
                    with gr.Row():
                        refresh_btn    = gr.Button("Refresh Stats")
                        summary_btn    = gr.Button("Generate Summary")
                        trainer_btn    = gr.Button("Trainer Report")
                        checkpoint_btn = gr.Button("Mark Training Checkpoint")

                    stats_status         = gr.Markdown("")

                    # Two-table stats: historical baseline + current session.
                    # gr.Dataframe defaults to row_count=(1, "dynamic") which
                    # CAPS VISIBLE ROWS at 1 regardless of how many rows the
                    # value list contains. Setting it generously so every model
                    # in ARENA_MODELS fits even after future additions.
                    _max_rows = max(len(ARENA_MODELS) * 2, 20)
                    _hist_cap, _hist_rows, _sess_cap, _sess_rows = build_stats_tables()

                    hist_caption_md      = gr.Markdown(_hist_cap)
                    hist_df              = gr.Dataframe(
                        value=_hist_rows,
                        headers=HIST_HEADERS,
                        interactive=False,
                        wrap=True,
                        row_count=(_max_rows, "dynamic"),
                        column_count=(len(HIST_HEADERS), "fixed"),
                        elem_id="hist_df",
                    )

                    session_caption_md   = gr.Markdown(_sess_cap)
                    session_df           = gr.Dataframe(
                        value=_sess_rows,
                        headers=SESSION_HEADERS,
                        interactive=False,
                        wrap=True,
                        row_count=(_max_rows, "dynamic"),
                        column_count=(len(SESSION_HEADERS), "fixed"),
                        elem_id="session_df",
                    )

                    checkpoint_md        = gr.Markdown(build_checkpoint_log())
                    # Persona-lean stats panel is intentionally not rendered by
                    # default — persona names are user-defined (see PERSONA_NAMES
                    # in config.py) and out-of-the-box Agent Ouija ships with an
                    # empty tuple. The persona_lean column + detection at log
                    # time still run whenever PERSONA_NAMES is populated, so the
                    # data is collected for whoever wants to enable the panel.
                    # To turn it on: re-add `persona_md = gr.Markdown(
                    # build_persona_stats())` here + the refresh/wipe wirings.
                    refusals_md          = gr.Markdown(build_refusals())
                    tag_md               = gr.Markdown(build_tag_stats())

                    gr.Markdown("---\n### Reset UI Preferences")
                    gr.Markdown(
                        "Flip the Control Center + Executive Boardroom "
                        "back to their factory defaults — Mysticism Level "
                        "to **Low Energy**, both Boardroom checkboxes "
                        "OFF. Wipes `data/preferences.json` (light, "
                        "non-destructive; no scores or seances are "
                        "touched)."
                    )
                    with gr.Row():
                        reset_prefs_btn = gr.Button(
                            "↩ Reset UI Preferences",
                            variant="secondary",
                        )
                    reset_prefs_status = gr.Markdown("")

                    def on_reset_prefs():
                        prefs.reset()
                        return (
                            gr.update(value=MYSTICISM_LEVELS[0]),   # energy
                            gr.update(value=False),                  # boardroom
                            gr.update(value=False),                  # sr_with_jr
                            "✓ Preferences reset — Mysticism Level "
                            "back to **Low Energy**, both Boardroom "
                            "toggles OFF. `preferences.json` deleted.",
                        )

                    reset_prefs_btn.click(
                        on_reset_prefs,
                        outputs=[
                            ao_energy_dropdown,
                            ao_boardroom_toggle,
                            ao_sr_with_jr_toggle,
                            reset_prefs_status,
                        ],
                        queue=False,
                    )

                    gr.Markdown("---\n### Danger Zone")
                    with gr.Row():
                        wipe_btn              = gr.Button("Wipe Database", variant="stop")
                        wipe_yes_nobackup_btn = gr.Button("Yes, Don't backup the database and chats", variant="stop", visible=False)
                        wipe_yes_backup_btn   = gr.Button("Yes, Backup the database and chats",   variant="stop", visible=False)
                        wipe_no_btn           = gr.Button("No", visible=False)

                    def _refresh_stats():
                        h_cap, h_rows, s_cap, s_rows = build_stats_tables()
                        return (h_cap, h_rows, s_cap, s_rows,
                                build_checkpoint_log(),
                                build_refusals(),
                                build_tag_stats())

                    refresh_btn.click(
                        _refresh_stats,
                        outputs=[hist_caption_md, hist_df,
                                 session_caption_md, session_df,
                                 checkpoint_md, refusals_md, tag_md],
                    )
                    summary_btn.click(
                        on_summary,
                        outputs=[stats_status],
                    )
                    trainer_btn.click(
                        on_trainer_report,
                        outputs=[stats_status],
                    )
                    checkpoint_btn.click(
                        on_checkpoint,
                        outputs=[stats_status, checkpoint_md],
                    )
                    _wipe_outputs = [
                        wipe_btn, wipe_yes_nobackup_btn, wipe_yes_backup_btn, wipe_no_btn,
                        stats_status,
                        hist_caption_md, hist_df,
                        session_caption_md, session_df,
                        checkpoint_md, refusals_md, tag_md,
                    ]
                    wipe_btn.click(
                        on_wipe_request,
                        outputs=[wipe_btn, wipe_yes_nobackup_btn, wipe_yes_backup_btn,
                                 wipe_no_btn, stats_status],
                    )
                    wipe_no_btn.click(
                        on_wipe_no,
                        outputs=_wipe_outputs,
                    )
                    wipe_yes_nobackup_btn.click(
                        on_wipe_yes_nobackup,
                        outputs=_wipe_outputs,
                    )
                    wipe_yes_backup_btn.click(
                        on_wipe_yes_backup,
                        outputs=_wipe_outputs,
                    )

                    # ── Time-based rollback ──────────────────────────
                    # Delete rows/files strictly older than
                    #   now - (amount × unit)
                    # from the selected target. Preview first (counts
                    # only), then explicit YES + auto backup before
                    # the delete runs. Chroma chunks and doc-ingest
                    # rows without a created_at metadata field survive
                    # every purge; the wipe button above is the only
                    # way to remove them.
                    from config import NEWS_CACHE_DIR

                    gr.Markdown("---\n### Time-based Rollback")
                    gr.Markdown(
                        "Delete data older than a chosen cutoff. Preview "
                        "first — the button below runs a dry count "
                        "without touching disk. Purge is behind a typed "
                        "**YES** confirmation and always backs up "
                        "first into `data/backups/rollback_<stamp>/`.\n\n"
                        "**Note:** ChromaDB chunks written before this "
                        "feature (or docs ingested without a `created_at` "
                        "field) are not counted or purged by age — use "
                        "the full Wipe above to clear them."
                    )
                    with gr.Row():
                        rollback_amount = gr.Number(
                            value=6, minimum=1, maximum=999,
                            precision=0, label="Amount",
                            scale=0, min_width=110,
                        )
                        rollback_unit = gr.Dropdown(
                            choices=["Days", "Weeks", "Months"],
                            value="Weeks",
                            label="Unit",
                            scale=0, min_width=140,
                        )
                        rollback_target = gr.Dropdown(
                            choices=[
                                "Curated RAG (ChromaDB)",
                                "Arena scores (arena_scores.db)",
                                "Seances (JSON)",
                                "News docs",
                                "All of the above",
                            ],
                            value="All of the above",
                            label="Target",
                            scale=1, min_width=260,
                        )
                    rollback_preview_md = gr.Markdown(
                        "_Set an amount + unit + target, then click "
                        "**Preview** to see counts._"
                    )
                    with gr.Row():
                        rollback_preview_btn = gr.Button(
                            "🔍 Preview", variant="secondary",
                        )
                        rollback_confirm_input = gr.Textbox(
                            label="Type YES to confirm purge",
                            placeholder="YES",
                            scale=1,
                        )
                        rollback_purge_btn = gr.Button(
                            "🗑 Purge (backup first)",
                            variant="stop", scale=0,
                        )
                    rollback_status_md = gr.Markdown("")

                    def _rollback_cutoff_iso(amount, unit):
                        """Return an ISO-second timestamp `now - (amount×unit)`.
                        Months are treated as 30 days (calendar-accurate
                        month math needs dateutil; the built-in datetime
                        library doesn't support it out of the box and a
                        30-day approximation is fine for a housekeeping
                        control)."""
                        n = int(max(1, amount or 1))
                        per = {"Days": 1, "Weeks": 7, "Months": 30}.get(
                            unit, 7,
                        )
                        cutoff = (datetime.now()
                                    - timedelta(days=n * per))
                        return cutoff.isoformat(timespec="seconds")

                    def _rollback_count_seances(cutoff_iso):
                        if not SEANCES_DIR.exists():
                            return 0
                        n = 0
                        for p in SEANCES_DIR.glob("*.json"):
                            try:
                                d = json.loads(p.read_text(encoding="utf-8"))
                            except (OSError, json.JSONDecodeError):
                                continue
                            ts = d.get("updated") or d.get("timestamp") or ""
                            if ts and ts < cutoff_iso:
                                n += 1
                        return n

                    def _rollback_count_news(cutoff_iso):
                        if not NEWS_CACHE_DIR.exists():
                            return 0
                        # Filenames are YYYY-MM-DD_<category>.md; compare
                        # date-prefix to cutoff's date. String compare
                        # works because ISO dates sort lexically.
                        cutoff_date = cutoff_iso[:10]
                        n = 0
                        for p in NEWS_CACHE_DIR.glob("*.md"):
                            name = p.name
                            if len(name) < 10 or name[4] != "-" or name[7] != "-":
                                continue
                            if name[:10] < cutoff_date:
                                n += 1
                        return n

                    def _rollback_delete_seances(cutoff_iso):
                        deleted = 0
                        if not SEANCES_DIR.exists():
                            return deleted
                        for p in list(SEANCES_DIR.glob("*.json")):
                            try:
                                d = json.loads(p.read_text(encoding="utf-8"))
                            except (OSError, json.JSONDecodeError):
                                continue
                            ts = d.get("updated") or d.get("timestamp") or ""
                            if ts and ts < cutoff_iso:
                                try:
                                    p.unlink()
                                    deleted += 1
                                except OSError as e:
                                    log.warning(
                                        "Could not delete seance %s: %s",
                                        p.name, e,
                                    )
                        return deleted

                    def _rollback_delete_news(cutoff_iso):
                        deleted = 0
                        if not NEWS_CACHE_DIR.exists():
                            return deleted
                        cutoff_date = cutoff_iso[:10]
                        for p in list(NEWS_CACHE_DIR.glob("*.md")):
                            name = p.name
                            if len(name) < 10 or name[4] != "-" or name[7] != "-":
                                continue
                            if name[:10] < cutoff_date:
                                try:
                                    p.unlink()
                                    deleted += 1
                                except OSError as e:
                                    log.warning(
                                        "Could not delete news file %s: %s",
                                        p.name, e,
                                    )
                        return deleted

                    _ROLLBACK_TARGETS = (
                        "Curated RAG (ChromaDB)",
                        "Arena scores (arena_scores.db)",
                        "Seances (JSON)",
                        "News docs",
                        "All of the above",
                    )

                    def on_rollback_preview(amount, unit, target):
                        cutoff = _rollback_cutoff_iso(amount, unit)
                        lines = [
                            f"**Cutoff:** rows/files with timestamp before "
                            f"`{cutoff}` would be removed.\n",
                        ]
                        do_all = target == "All of the above"
                        if do_all or target == "Curated RAG (ChromaDB)":
                            r = count_chroma_before(cutoff)
                            if r.get("error"):
                                lines.append(f"- ChromaDB: ❌ {r['error']}")
                            else:
                                lines.append(
                                    f"- ChromaDB: **{r['matched']}** chunk(s) "
                                    f"of {r['total']} total would be deleted"
                                )
                        if do_all or target == "Arena scores (arena_scores.db)":
                            r = count_arena_rows_before(cutoff)
                            if r.get("error"):
                                lines.append(f"- Arena DB: ❌ {r['error']}")
                            else:
                                lines.append(
                                    f"- Arena DB: **{r['arena_rows']}** "
                                    f"arena_results + **{r['rag_rows']}** "
                                    "rag_contributions row(s) would be deleted "
                                    "(training_checkpoints preserved)"
                                )
                        if do_all or target == "Seances (JSON)":
                            n = _rollback_count_seances(cutoff)
                            lines.append(
                                f"- Seances: **{n}** file(s) would be deleted"
                            )
                        if do_all or target == "News docs":
                            n = _rollback_count_news(cutoff)
                            lines.append(
                                f"- News docs: **{n}** file(s) would be deleted"
                            )
                        lines.append(
                            "\n_Preview is read-only. Type **YES** in the "
                            "box, then click Purge to commit._"
                        )
                        return "\n".join(lines)

                    def on_rollback_purge(amount, unit, target, confirm):
                        if (confirm or "").strip().upper() != "YES":
                            return (
                                "❌ Type **YES** in the confirmation box first.",
                                gr.update(),
                            )
                        cutoff = _rollback_cutoff_iso(amount, unit)
                        backup_result = backup_state(label="rollback")
                        if backup_result.get("status") == "error":
                            return (
                                f"❌ Backup failed: {backup_result.get('message')}. "
                                "Nothing was purged.",
                                gr.update(),
                            )
                        summary_lines = [
                            f"✓ Backup written to `{backup_result.get('path')}`",
                            f"**Cutoff:** `{cutoff}`",
                        ]
                        do_all = target == "All of the above"
                        if do_all or target == "Curated RAG (ChromaDB)":
                            r = purge_chroma_before(cutoff)
                            summary_lines.append(
                                f"- ChromaDB: "
                                f"{'✓' if r['status'] == 'success' else '❌'} "
                                f"{r['message']}"
                            )
                        if do_all or target == "Arena scores (arena_scores.db)":
                            r = purge_arena_rows_before(cutoff)
                            summary_lines.append(
                                f"- Arena DB: "
                                f"{'✓' if r['status'] == 'success' else '❌'} "
                                f"{r['message']}"
                            )
                        if do_all or target == "Seances (JSON)":
                            n = _rollback_delete_seances(cutoff)
                            summary_lines.append(
                                f"- Seances: ✓ deleted {n} file(s)"
                            )
                        if do_all or target == "News docs":
                            n = _rollback_delete_news(cutoff)
                            summary_lines.append(
                                f"- News docs: ✓ deleted {n} file(s)"
                            )
                        return "\n".join(summary_lines), gr.update(value="")

                    rollback_preview_btn.click(
                        on_rollback_preview,
                        inputs=[rollback_amount, rollback_unit,
                                rollback_target],
                        outputs=[rollback_preview_md],
                    )
                    rollback_purge_btn.click(
                        on_rollback_purge,
                        inputs=[rollback_amount, rollback_unit,
                                rollback_target, rollback_confirm_input],
                        outputs=[rollback_status_md,
                                  rollback_confirm_input],
                    )

                    # ── Restore from a backup ─────────────────────────
                    # Backups accumulate in data/backups/ from either
                    # a full Wipe or a time-based Rollback. This
                    # control lists them newest-first, previews what
                    # each holds, and (with typed YES) copies the
                    # contents back into their live locations.
                    gr.Markdown("---\n### Restore From Backup")
                    gr.Markdown(
                        "Every Wipe and Rollback drops a snapshot into "
                        "`data/backups/<label>_<timestamp>/`. Restore "
                        "overwrites the live copies of `arena_scores.db`, "
                        "`chat_history.json`, `chroma/`, `seances/`, and "
                        "`news/` — whichever the snapshot contains. "
                        "Restart the app after restoring so the retriever "
                        "and DB handles pick up the swapped-in files."
                    )

                    def _backup_choices():
                        entries = list_backups()
                        return [
                            (
                                f"{b['name']} · {b['created_at']} · "
                                f"{len(b['items'])} item(s)",
                                b["name"],
                            )
                            for b in entries
                        ]

                    _initial_backup_choices = _backup_choices()
                    with gr.Row():
                        restore_dropdown = gr.Dropdown(
                            choices=_initial_backup_choices,
                            value=(_initial_backup_choices[0][1]
                                   if _initial_backup_choices else None),
                            label="Backup snapshot",
                            scale=3, min_width=360,
                        )
                        restore_refresh_btn = gr.Button(
                            "🔄 Refresh", variant="secondary",
                            scale=0,
                        )
                    restore_preview_md = gr.Markdown(
                        "_Pick a snapshot to preview its contents._"
                    )
                    with gr.Row():
                        restore_preview_btn = gr.Button(
                            "🔍 Preview", variant="secondary",
                        )
                        restore_confirm_input = gr.Textbox(
                            label="Type YES to confirm restore",
                            placeholder="YES",
                            scale=1,
                        )
                        restore_btn = gr.Button(
                            "♻ Restore",
                            variant="stop", scale=0,
                        )
                    restore_status_md = gr.Markdown("")

                    def on_restore_refresh():
                        choices = _backup_choices()
                        value = choices[0][1] if choices else None
                        return gr.update(choices=choices, value=value)

                    def on_restore_preview(backup_name):
                        if not backup_name:
                            return "_Pick a snapshot first._"
                        for b in list_backups():
                            if b["name"] == backup_name:
                                items = "\n".join(f"- `{i}`"
                                                   for i in b["items"])
                                return (
                                    f"**{b['name']}** — created "
                                    f"`{b['created_at']}`\n\n"
                                    f"Contents:\n{items}"
                                )
                        return f"❌ Backup `{backup_name}` not found."

                    def on_restore_run(backup_name, confirm):
                        if (confirm or "").strip().upper() != "YES":
                            return (
                                "❌ Type **YES** in the confirmation box first.",
                                gr.update(),
                            )
                        if not backup_name:
                            return "❌ Pick a snapshot first.", gr.update()
                        r = restore_backup(backup_name)
                        icon = {"success": "✓",
                                "partial": "⚠️",
                                "error":   "❌"}.get(r["status"], "?")
                        return (
                            f"{icon} Restore `{backup_name}` — {r['message']}. "
                            "Restart the app to pick up the swapped files.",
                            gr.update(value=""),
                        )

                    restore_refresh_btn.click(
                        on_restore_refresh,
                        outputs=[restore_dropdown],
                        queue=False,
                    )
                    restore_preview_btn.click(
                        on_restore_preview,
                        inputs=[restore_dropdown],
                        outputs=[restore_preview_md],
                        queue=False,
                    )
                    restore_btn.click(
                        on_restore_run,
                        inputs=[restore_dropdown, restore_confirm_input],
                        outputs=[restore_status_md, restore_confirm_input],
                    )

                # 📰 News Pulls moved to Agent Ouija → 🤖 The Assistant
                # accordion so both external APIs (Calendar + News)
                # sit together. Data tab now focuses on scores /
                # backups / rollback / disk usage.

                # ── 💾 Disk Usage ──────────────────────────────────
                with gr.Accordion("💾 Disk Usage", open=False):
                    gr.Markdown(
                        "Sizes of key on-disk assets (gitignored). Includes " +
                        "your document corpus, media, ChromaDB collections, " +
                        "voice files, and saved seances."
                    )
                    disk_usage_md = gr.Markdown("(click Refresh Sizes to scan)")
                    disk_refresh_btn = gr.Button("🔄 Refresh Sizes", variant="primary")

                    def _fmt_bytes(n: int) -> str:
                        if n < 1024: return f"{n} B"
                        if n < 1024**2: return f"{n/1024:.1f} KB"
                        if n < 1024**3: return f"{n/(1024**2):.1f} MB"
                        return f"{n/(1024**3):.2f} GB"

                    def _dir_size(p) -> int:
                        if not p.exists(): return 0
                        try:
                            if p.is_file(): return p.stat().st_size
                            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                        except (OSError, PermissionError):
                            return 0

                    def _disk_scan():
                        from config import (
                            CHROMA_DIR, DOCS_DIR, MEDIA_DIR,
                            PROMPTS_DIR, PRIVATE_CORPUS_DIR,
                            TTS_VOICES_DIR, ARENA_DB_PATH,
                            CHAT_HISTORY_PATH, SEANCES_DIR,
                            TRAINING_DIR, SUMMARIES_DIR, BACKUPS_DIR,
                        )
                        rows = [
                            ("Documents (docs/)",                DOCS_DIR),
                            ("Media (docs/media/)",              MEDIA_DIR),
                            ("Prompts (prompts/)",               PROMPTS_DIR),
                            ("Copyright (Copyright/)",           PRIVATE_CORPUS_DIR),
                            ("ChromaDB (data/chroma/)",          CHROMA_DIR),
                            ("Voices (voices/)",                 TTS_VOICES_DIR),
                            ("Seances (data/seances/)",          SEANCES_DIR),
                            ("Training exports (data/training/)", TRAINING_DIR),
                            ("Summaries (data/summaries/)",      SUMMARIES_DIR),
                            ("Backups (data/backups/)",          BACKUPS_DIR),
                            ("arena_scores.db",                  ARENA_DB_PATH),
                            ("chat_history.json",                CHAT_HISTORY_PATH),
                        ]
                        sized = [(label, _dir_size(p)) for label, p in rows]
                        total = sum(s for _, s in sized)
                        lines_out = ["| Location | Size |", "|----------|-----:|"]
                        for label, size in sized:
                            lines_out.append(f"| {label} | {_fmt_bytes(size)} |")
                        lines_out.append(f"| **TOTAL** | **{_fmt_bytes(total)}** |")
                        return "\n".join(lines_out)

                    disk_refresh_btn.click(
                        _disk_scan,
                        outputs=[disk_usage_md],
                        queue=False,
                    )

            # ── Tab navigation wiring ──────────────────────────────────
            # Control Center's nav buttons switch which (hidden) tab is
            # selected; Return buttons on each hidden tab send the user
            # back to Agent Ouija. queue=False so nav bypasses the model-
            # ask queue — otherwise a stuck / long-running Ask blocks
            # every subsequent click and the whole UI feels frozen until
            # a browser refresh.
            def _goto(tab_id):
                # gr.Tabs(selected=...) is Gradio's canonical way to
                # switch tabs programmatically (works across 4.x and 5.x).
                return lambda: gr.Tabs(selected=tab_id)

            ao_nav_stats.click(_goto("data"),         outputs=[ao_tabs], queue=False)
            ao_nav_files.click(_goto("files"),        outputs=[ao_tabs], queue=False)
            ao_nav_dev.click(  _goto("dev_workshop"), outputs=[ao_tabs], queue=False)
            ao_nav_pers.click( _goto("persona"),      outputs=[ao_tabs], queue=False)

            _dev_return_btn.click(     _goto("agent_ouija"), outputs=[ao_tabs], queue=False)
            _files_return_btn.click(   _goto("agent_ouija"), outputs=[ao_tabs], queue=False)
            _persona_return_btn.click( _goto("agent_ouija"), outputs=[ao_tabs], queue=False)
            _stats_return_btn.click(   _goto("agent_ouija"), outputs=[ao_tabs], queue=False)

        gr.HTML(f"""
        <div style="display:flex;justify-content:flex-end;align-items:center;
                    padding:16px 0 4px;border-top:1px solid #eee;margin-top:8px">
            {ITN_LOGO_HTML}
        </div>
        """)

        # Attach native browser tooltips to the Ask buttons. Gradio 6
        # doesn't expose a `title` prop on gr.Button, so we hop into
        # the rendered DOM once the app is loaded and set the
        # attribute by elem_id. Users get real hover text without a
        # separate hint line under the row.
        demo.load(
            fn=None, inputs=None, outputs=None,
            js=r"""
            () => {
              const tooltips = {
                "ao_ask_btn":
                  "Ask For Revelations — the chosen models answer freely, "
                  + "with RAG + web context injected. Fluent, may reason "
                  + "beyond the retrieved chunks.",
                "ao_ask_lore_btn":
                  "Ask the Lore — same models, but STRICTLY grounded to "
                  + "RAG / doc / web context. If the answer isn't in the "
                  + "supplied material, they say \"Not in the lore.\" "
                  + "instead of guessing.",
              };
              const attach = () => {
                for (const [id, text] of Object.entries(tooltips)) {
                  const host = document.getElementById(id);
                  if (!host) continue;
                  const btn = host.querySelector("button") || host;
                  btn.title = text;
                }
              };
              // Gradio may mount buttons after this hook fires; retry
              // for a couple seconds so the tooltip lands regardless.
              attach();
              let tries = 0;
              const t = setInterval(() => {
                attach();
                if (++tries > 20) clearInterval(t);
              }, 150);
            }
            """,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # inbrowser=True opens the default browser to the Gradio URL on launch
    # and pulls it to the front so the user doesn't have to alt-tab from
    # the console. On Windows the default browser focuses naturally; on
    # other OSes Gradio falls back to webbrowser.open() which is best-effort.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        inbrowser=True,
    )
