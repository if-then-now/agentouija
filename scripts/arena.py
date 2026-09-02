"""
arena.py — Terminal multi-model arena

Fans a single query to all ARENA_MODELS simultaneously via ThreadPoolExecutor.
Tracks response time per model. Computes The Reading composite metric.

Scoring — 0-4 baseball system:
    0 = whiff       — not what you needed
    1 = contact     — got something useful
    2 = base hit    — solid, did the job
    3 = extra bases — better than expected
    4 = home run    — couldn't tell it wasn't Claude

The Reading = (score / SCORE_MAX) / log(response_time + e)
    Fast + high score = strong Reading
    Slow + low score  = weak Reading

Usage:
    python scripts/arena.py
"""

import argparse
import logging
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter

from dotenv import load_dotenv

# Resolve .env to ~/.config/agentouija/.env (hardened default)
# when present, else fall back to the legacy repo-root file.
# Encrypted-.env installs prompt via getpass at process start.
from config import (
    env_file_path as _env_file_path,
    is_encrypted_env as _is_encrypted_env,
)


def _load_env_or_prompt():
    if not _is_encrypted_env():
        load_dotenv(_env_file_path())
        return
    import getpass
    import os as _os
    from env_crypto import (
        load_into_environ as _env_load_str,
        read_encrypted as _env_decrypt,
    )
    try:
        from cryptography.fernet import InvalidToken
    except ImportError:
        raise SystemExit(
            "❌ arena.py: cryptography missing but .env is "
            "encrypted. `pip install cryptography` and relaunch."
        )
    env_pass = _os.environ.get("AGENT_OUIJA_ENV_PASSPHRASE")
    if env_pass:
        _env_load_str(_env_decrypt(_env_file_path(), env_pass))
        return
    for i in range(1, 4):
        try:
            pw = getpass.getpass(f"🔐 .env passphrase (attempt {i}/3): ")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n❌ no passphrase provided.")
        try:
            _env_load_str(_env_decrypt(_env_file_path(), pw))
            return
        except InvalidToken:
            print("   Wrong passphrase.")
    raise SystemExit("❌ 3 wrong passphrases — aborting.")


_load_env_or_prompt()

from config import (
    ARENA_MODELS,
    AVAILABLE_MODELS,
    DEFAULT_PERSONA_MODE,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    PERSONA_REINFORCEMENT,
    REFUSAL_LABEL,
    REFUSAL_SCORE,
    SCORE_LABELS,
    SCORE_RANGE,
    compute_reading,
    compute_reading_with_cost,
    label_for_score,
)
from db import get_conn
from persona_lean import detect_persona_lean
from prompt_loader import get_prompt_loader
from rag import get_private_context, get_rag_context
from tagger import load_keyword_map, tag_by_keywords

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)


# ── HTTP session (shared, thread-safe) ────────────────────────────────────────
# urllib3 pools sockets per-host; reusing this Session across the thread-fan-out
# avoids TCP setup latency on every model call.
_HTTP = requests.Session()
_HTTP.mount("http://",  HTTPAdapter(pool_connections=8, pool_maxsize=16))
_HTTP.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=16))


# ── System-prompt cache (per persona mode) ───────────────────────────────────
# Locals and clouds use different prompts. Build each mode lazily on first
# request so a session with no cloud models doesn't pay the cost of reading
# every persona + episode file.
_SYSTEM_PROMPTS: dict[str, str] = {}


def get_system_prompt_for(model_key: str) -> str:
    cfg = AVAILABLE_MODELS.get(model_key, {})
    mode = cfg.get("persona", DEFAULT_PERSONA_MODE)
    force_base = bool(cfg.get("skip_enhanced_persona", False))
    # Cache key includes force_base so a model with the opt-out
    # stays on the base stack even if another peer already warmed
    # up the enhanced one this session.
    key = (mode, force_base)
    if key not in _SYSTEM_PROMPTS:
        _SYSTEM_PROMPTS[key] = get_prompt_loader(
            mode, force_base=force_base,
        ).load()
    return _SYSTEM_PROMPTS[key]


# ── Logging a scored result ───────────────────────────────────────────────────

def log_score(
    conn: sqlite3.Connection,
    question: str,
    model_key: str,
    model_display: str,
    response: str,
    score: int,
    response_time_seconds: float,
    keyword_map: dict,
    input_tokens: int = 0,
    output_tokens: int = 0,
    input_cost: float = 0.0,
    output_cost: float = 0.0,
    round_id: str | None = None,
) -> None:
    tag          = tag_by_keywords(question, keyword_map)
    reading      = compute_reading(score, response_time_seconds)
    cost_reading = compute_reading_with_cost(
        score, response_time_seconds, (input_cost or 0.0) + (output_cost or 0.0)
    )
    model_type   = AVAILABLE_MODELS.get(model_key, {}).get("type", "unknown")
    persona_lean = detect_persona_lean(response)
    try:
        conn.execute(
            """INSERT INTO arena_results
               (timestamp, question, model_key, model_display, model_type, response,
                score, response_time_seconds, reading, tags, tag_source,
                persona_lean, input_tokens, output_tokens, input_cost, output_cost,
                round_id, cost_reading)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), question, model_key, model_display,
             model_type, response, score, response_time_seconds, reading,
             tag, "keyword",
             persona_lean, input_tokens, output_tokens, input_cost, output_cost,
             round_id, cost_reading),
        )
        conn.commit()
    except sqlite3.DatabaseError:
        log.exception("Failed to log score for model %s", model_key)


# ── Model querying ────────────────────────────────────────────────────────────
# Both _query_* helpers return a dict with text + token counts + USD cost so
# the caller can log all of it. Locals carry costs of 0.0 by definition (no
# per-call billing on Ollama); cloud models source costs from LiteLLM's
# per-token price table.
#
# Shape:
#   {"text": str, "input_tokens": int, "output_tokens": int,
#    "input_cost": float, "output_cost": float}


def _query_local(model_cfg: dict, messages: list,
                  max_tokens: int | None = None) -> dict:
    """Call a local model via Ollama's /api/chat.

    max_tokens (optional) → Ollama's `options.num_predict`, capping the
    output. None = model default (unbounded until context fills).
    """
    body: dict = {
        "model":    model_cfg["name"],
        "messages": messages,
        "stream":   False,
    }
    if max_tokens:
        body["options"] = {"num_predict": int(max_tokens)}
    resp = _HTTP.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=body,
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


def _query_cloud(model_key: str, model_cfg: dict, messages: list,
                  max_tokens: int | None = None) -> dict:
    """Call a cloud model via LiteLLM.

    max_tokens (optional) — the max_tokens knob passed to LiteLLM.
    Default 1024 preserves the previous behaviour; the /api/executive
    sidecar overrides based on caller-supplied response length so
    downstream apps (Master Ouija) can enforce a shorter cap.
    """
    # Lazy import: litellm is heavy; only pay the cost if a cloud model is selected.
    from litellm import completion as litellm_completion
    from litellm import cost_per_token

    env_key = model_cfg.get("env_key", "")
    if not env_key:
        raise RuntimeError(f"Cloud model {model_key} is missing 'env_key' in config")
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} is not set in environment — add it to .env")

    resp = litellm_completion(
        model=model_cfg["name"],
        messages=messages,
        api_key=api_key,
        max_tokens=int(max_tokens) if max_tokens else 1024,
    )

    usage = getattr(resp, "usage", None) or {}
    prompt_tokens     = int(getattr(usage, "prompt_tokens",     usage.get("prompt_tokens", 0))     or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", usage.get("completion_tokens", 0)) or 0)

    # LiteLLM exposes per-model cost in USD/token; we split prompt vs
    # completion so the leaderboard can show both axes. Pricing errors
    # (unknown model, missing table entry) degrade to 0 — billing is
    # advisory here, not the source of truth.
    try:
        input_cost, output_cost = cost_per_token(
            model=model_cfg["name"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as e:
        log.warning("cost_per_token failed for %s (%s) — recording 0.0",
                    model_cfg["name"], type(e).__name__)
        input_cost = output_cost = 0.0

    return {
        "text":          resp.choices[0].message.content,
        "input_tokens":  prompt_tokens,
        "output_tokens": completion_tokens,
        "input_cost":    float(input_cost),
        "output_cost":   float(output_cost),
    }


def _empty_result(elapsed: float) -> dict:
    return {
        "text":          None,
        "elapsed":       elapsed,
        "input_tokens":  0,
        "output_tokens": 0,
        "input_cost":    0.0,
        "output_cost":   0.0,
    }


def query_model(model_key: str, question: str,
                rag_context: str | None = None,
                private_context: str | None = None) -> tuple[str, dict]:
    """Return (model_key, result_dict).

    result_dict shape:
      {"text": str|None, "elapsed": float,
       "input_tokens": int, "output_tokens": int,
       "input_cost": float, "output_cost": float}

    System prompt is chosen per-model from AVAILABLE_MODELS[...]["persona"]
    ("generic" or "enhanced"). rag_context is injected only for models
    with "rag": True. private_context is injected only for models with
    "private_corpus": True (and is only populated upstream when the
    USE_PRIVATE_CORPUS master switch is on).
    """
    model_cfg = AVAILABLE_MODELS.get(model_key)
    if model_cfg is None:
        log.warning("Unknown model_key %s — skipping", model_key)
        return model_key, _empty_result(-1.0)

    system_prompt = get_system_prompt_for(model_key)
    messages = [{"role": "system", "content": system_prompt}]
    if rag_context and model_cfg.get("rag", False):
        messages.append({"role": "system",
                         "content": f"Knowledge base context:\n\n{rag_context}"})
    if private_context and model_cfg.get("private_corpus", False):
        messages.append({"role": "system",
                         "content": f"Persona corpus context:\n\n{private_context}"})
    user_content = (
        f"{PERSONA_REINFORCEMENT}\n\n{question}"
        if PERSONA_REINFORCEMENT
        else question
    )
    messages.append({"role": "user", "content": user_content})

    start = time.perf_counter()
    try:
        kind = model_cfg.get("type")
        if kind == "local":
            payload = _query_local(model_cfg, messages)
        elif kind == "cloud":
            payload = _query_cloud(model_key, model_cfg, messages)
        else:
            log.warning("Model %s has unknown type %r — skipping", model_key, kind)
            return model_key, _empty_result(-1.0)

        elapsed = round(time.perf_counter() - start, 3)
        payload["elapsed"] = elapsed
        return model_key, payload

    except requests.Timeout:
        elapsed = round(time.perf_counter() - start, 3)
        log.warning("Model %s timed out after %.1fs", model_key, OLLAMA_TIMEOUT)
        return model_key, _empty_result(elapsed)
    except requests.RequestException as e:
        elapsed = round(time.perf_counter() - start, 3)
        log.warning("Model %s HTTP error (%s): %s", model_key, type(e).__name__, e)
        return model_key, _empty_result(elapsed)
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 3)
        log.warning("Model %s unavailable (%s): %s", model_key, type(e).__name__, e)
        return model_key, _empty_result(elapsed)


# ── Scoring prompt ────────────────────────────────────────────────────────────

def prompt_score(model_display: str) -> int:
    labels = "  ".join(f"{k}={v}" for k, v in SCORE_LABELS.items())
    valid_keys = ", ".join(str(i) for i in SCORE_RANGE)
    while True:
        raw = input(
            f"\nScore [{model_display}]  {labels}  "
            f"(r={REFUSAL_LABEL}, s=skip): "
        ).strip().lower()
        if raw == "s":
            return -1
        if raw == "r":
            return REFUSAL_SCORE
        if raw in [str(i) for i in SCORE_RANGE]:
            return int(raw)
        print(f"  Enter {valid_keys}, r, or s.")


# ── Arena run ─────────────────────────────────────────────────────────────────

def run_arena() -> None:
    keyword_map   = load_keyword_map()
    conn          = get_conn()
    active_models = [m for m in ARENA_MODELS if m in AVAILABLE_MODELS]

    if not active_models:
        log.error("No models in ARENA_MODELS are present in AVAILABLE_MODELS — aborting")
        conn.close()
        return

    print(f"\n{'─'*60}")
    print(f"  Agent Ouija")
    print(f"  Models: {', '.join(AVAILABLE_MODELS[m]['display_name'] for m in active_models)}")
    print(f"  Scoring: {' | '.join(f'{k}={v}' for k, v in SCORE_LABELS.items())}")
    print(f"  The Reading: (score/SCORE_MAX) / log(time + e)")
    print(f"{'─'*60}\n")

    try:
        while True:
            try:
                question = input("Question (q to quit): ").strip()
            except EOFError:
                break
            if question.lower() in ("q", "quit", "exit"):
                break
            if not question:
                continue

            # Round identifier — shared by every model that answered this
            # question. The trainer report groups peers by round_id so a
            # slow scoring pass that crosses a minute boundary still
            # treats co-asked responses as peers.
            round_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{random.randint(0, 0xFFFF):04x}"

            # Fetch RAG context once per question if any selected model wants it,
            # then share it across the fan-out so ChromaDB is hit just once.
            needs_rag = any(
                AVAILABLE_MODELS.get(m, {}).get("rag", False) for m in active_models
            )
            rag_context = get_rag_context(question) if needs_rag else None
            if needs_rag:
                if rag_context:
                    log.info("RAG context retrieved (%d chars)", len(rag_context))
                else:
                    log.info("RAG retrieval returned nothing — cloud models will run without context")

            # Private corpus context — same fetch-once pattern. Gating on
            # USE_PRIVATE_CORPUS happens inside get_private_context.
            needs_private = any(
                AVAILABLE_MODELS.get(m, {}).get("private_corpus", False) for m in active_models
            )
            private_context = get_private_context(question) if needs_private else None
            if needs_private and private_context:
                log.info("Private corpus context retrieved (%d chars)", len(private_context))

            print("\n  Querying models concurrently...\n")
            results: dict = {}

            with ThreadPoolExecutor(max_workers=len(active_models)) as executor:
                futures = {
                    executor.submit(query_model, m, question, rag_context, private_context): m
                    for m in active_models
                }
                for future in as_completed(futures):
                    model_key, payload = future.result()
                    results[model_key] = payload

            for model_key in active_models:
                payload       = results.get(model_key) or _empty_result(-1.0)
                response      = payload["text"]
                elapsed       = payload["elapsed"]
                model_display = AVAILABLE_MODELS[model_key]["display_name"]

                if response is None:
                    print(f"\n  [{model_display}] — unavailable ({elapsed}s), skipping")
                    continue

                print(f"\n{'─'*60}")
                print(f"  {model_display}  ·  {elapsed}s")
                print(f"{'─'*60}")
                print(response)

                score = prompt_score(model_display)
                if score == -1:
                    print("  Skipped — not logged")
                    continue

                reading = compute_reading(score, elapsed)
                log_score(
                    conn, question, model_key, model_display,
                    response, score, elapsed, keyword_map,
                    input_tokens=payload["input_tokens"],
                    output_tokens=payload["output_tokens"],
                    input_cost=payload["input_cost"],
                    output_cost=payload["output_cost"],
                    round_id=round_id,
                )
                cost_total = payload["input_cost"] + payload["output_cost"]
                cost_str   = f"${cost_total:.4f}" if cost_total else "$0"
                print(f"  Logged: {label_for_score(score)}  ·  {elapsed}s  ·  "
                      f"Reading: {reading}  ·  Cost: {cost_str}")

            print()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        conn.close()
        print("\nArena session ended.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agent Ouija terminal arena")
    parser.parse_args()  # no flags right now; leaves room for future args
    run_arena()


if __name__ == "__main__":
    main()
