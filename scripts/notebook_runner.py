"""
notebook_runner.py — Python notebook execution for the Notebook tab.

Owns the small surface area exposed to user code in the Notebook tab:

    ask(prompt, model=..., system=..., temperature=..., max_tokens=...) -> str
    ask_all(prompt, system=..., temperature=..., max_tokens=...) -> dict[key, str]
    get_rag(question) -> str | None

Design intent — this is for learning the programmatic API patterns the
course teaches. So unlike the Arena flow, NOTHING is injected on the
user's behalf:

    - No persona system prompt (system_core / persona_*.md never loaded here)
    - No PERSONA_REINFORCEMENT prepended to the user content
    - No automatic RAG injection (call get_rag() explicitly if needed)
    - No tagging, no scoring, no DB writes

Cell splitting uses the # %% convention (Spyder / VS Code interactive
windows). Variables persist across cells via a shared globals dict that
the caller stows in gr.State.

Execution is direct exec() — Brainy is a single-user local rig and the
user authored the code; we don't sandbox.
"""

import io
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import requests

from config import AVAILABLE_MODELS, OLLAMA_BASE_URL, OLLAMA_TIMEOUT
from rag import get_rag_context

CELL_MARKER = "# %%"

# LiteLLM routes Ollama calls via OLLAMA_API_BASE if it's set; otherwise
# it defaults to localhost:11434. Pin it to the configured base URL so
# the notebook tab's `completion(model="ollama_chat/...")` works the
# moment the user clicks Run — no extra setup cell required.
os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_BASE_URL)


# LiteLLM-format identifier expected by `litellm.completion(model=...)`.
# Cloud entries already carry the format in cfg["name"] (e.g.
# "openai/gpt-4o"). Locals are stored as the bare Ollama model name
# (e.g. "llama3.2") — prepend "ollama_chat/" so the call routes through
# Ollama's chat API.
LOCAL_LITELLM_PREFIX = "ollama_chat"


def litellm_name(model_key: str) -> str:
    """Return the LiteLLM-format model id for the given config key."""
    cfg = AVAILABLE_MODELS.get(model_key)
    if cfg is None:
        raise ValueError(
            f"Unknown model key {model_key!r}. "
            f"Available: {sorted(AVAILABLE_MODELS.keys())}"
        )
    if cfg.get("type") == "local":
        return f"{LOCAL_LITELLM_PREFIX}/{cfg['name']}"
    return cfg["name"]


# ── Direct model calls (NO persona / reinforcement / RAG injection) ───────────

def _query_local_simple(model_name: str, messages: list,
                        temperature: float | None,
                        max_tokens: int | None) -> str:
    options: dict = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        # Ollama uses num_predict for output token cap; mirror cloud "max_tokens".
        options["num_predict"] = max_tokens

    payload: dict = {"model": model_name, "messages": messages, "stream": False}
    if options:
        payload["options"] = options

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT
    )
    resp.raise_for_status()
    body = resp.json()
    try:
        return body["message"]["content"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Unexpected Ollama response shape: {body!r}") from e


def _query_cloud_simple(model_cfg: dict, messages: list,
                        temperature: float | None,
                        max_tokens: int | None) -> str:
    from litellm import completion as litellm_completion

    env_key = model_cfg.get("env_key", "")
    if not env_key:
        raise RuntimeError(f"Cloud model is missing 'env_key' in config")
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} is not set in environment — add it to .env")

    kwargs: dict = {"model": model_cfg["name"], "messages": messages, "api_key": api_key}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = litellm_completion(**kwargs)
    return resp.choices[0].message.content


# ── Public helpers (injected into the notebook scope) ─────────────────────────

def ask(prompt: str,
        model: str = "claude-sonnet",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = 1024) -> str:
    """Send a single prompt to one model. Returns the response text.

    No persona prompt is loaded — `system` is exactly what the model sees
    as its system message (or nothing if you leave it None). Defaults
    match the arena's claude slot so a course exercise can run immediately.
    """
    cfg = AVAILABLE_MODELS.get(model)
    if cfg is None:
        raise ValueError(
            f"Unknown model {model!r}. "
            f"Available: {sorted(AVAILABLE_MODELS.keys())}"
        )

    messages: list = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kind = cfg.get("type")
    if kind == "local":
        return _query_local_simple(cfg["name"], messages, temperature, max_tokens)
    if kind == "cloud":
        return _query_cloud_simple(cfg, messages, temperature, max_tokens)
    raise ValueError(f"Model {model!r} has unknown type {kind!r}")


def ask_all(prompt: str,
            system: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = 1024) -> dict:
    """Send the same prompt to every model in AVAILABLE_MODELS, concurrently.

    Returns {model_key: response_text}. On a per-model failure the value
    is "<error: ExceptionType: message>" so the iteration completes
    instead of breaking the whole notebook cell.
    """
    results: dict = {}
    keys = list(AVAILABLE_MODELS.keys())
    if not keys:
        return results

    def _run(k):
        try:
            return k, ask(prompt, model=k, system=system,
                          temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            return k, f"<error: {type(e).__name__}: {e}>"

    with ThreadPoolExecutor(max_workers=len(keys)) as ex:
        for fut in as_completed([ex.submit(_run, k) for k in keys]):
            k, v = fut.result()
            results[k] = v

    # Preserve config order in the returned dict for deterministic display.
    return {k: results[k] for k in keys if k in results}


def get_rag(question: str) -> str | None:
    """Retrieve top-k chunks from ChromaDB for the question, or None.

    Returns a formatted context string. Inject it into your prompt
    manually — the notebook never does this automatically.
    """
    return get_rag_context(question)


# ── Globals + execution ───────────────────────────────────────────────────────

def make_globals() -> dict:
    """Fresh globals dict pre-loaded with the notebook's helpers.

    Stored in gr.State and mutated in place by exec() on each Run click,
    so variables persist across runs until the user clicks Reset State.
    """
    return {
        "__name__":    "__notebook__",
        "__builtins__": __builtins__,
        "ask":         ask,
        "ask_all":     ask_all,
        "get_rag":     get_rag,
    }


def split_cells(code: str) -> list[str]:
    """Split code into cells at lines that begin with # %% (whitespace ok).

    A leading run of code BEFORE the first marker counts as cell 1. Empty
    cells are dropped so a trailing # %% doesn't produce a phantom cell.
    """
    if not code or not code.strip():
        return []
    cells: list[list[str]] = [[]]
    for line in code.splitlines():
        if line.lstrip().startswith(CELL_MARKER):
            cells.append([])
        else:
            cells[-1].append(line)
    return ["\n".join(lines) for lines in cells if any(s.strip() for s in lines)]


def run_cells(code: str, persisted_globals: dict) -> tuple[str, dict]:
    """Run each cell sequentially against the shared globals dict.

    Returns (markdown_output, updated_globals). On the first exception
    we stop running subsequent cells — the user fixes the error and
    re-runs. State from cells that ran successfully is preserved in
    persisted_globals (which is mutated in place).
    """
    cells = split_cells(code)
    if not cells:
        return "_Nothing to run — write some Python first._", persisted_globals

    sections: list[str] = []
    for i, cell_src in enumerate(cells, 1):
        sections.append(f"### Cell {i}")
        sections.append("```python\n" + cell_src + "\n```")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        traceback_text: str | None = None
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(cell_src, f"<cell-{i}>", "exec"), persisted_globals)
        except Exception:
            traceback_text = traceback.format_exc()

        out_text = stdout_buf.getvalue().rstrip()
        err_text = stderr_buf.getvalue().rstrip()

        if out_text:
            sections.append("**stdout**\n```\n" + out_text + "\n```")
        if err_text:
            sections.append("**stderr**\n```\n" + err_text + "\n```")
        if traceback_text:
            sections.append("**traceback**\n```\n" + traceback_text + "\n```")
            sections.append(f"_Stopped at cell {i} due to error. Fix and re-run._")
            break
        if not out_text and not err_text:
            sections.append("_no output_")

    return "\n\n".join(sections), persisted_globals


def _exec_cells_plain(code: str, scope: dict) -> str:
    """Execute the user's cells in scope, return a single plain-text
    transcript suitable for a Textbox response widget.

    Each cell's stdout/stderr/traceback is concatenated. Execution stops
    at the first cell that raises so the user can fix and re-run.
    """
    cells = split_cells(code)
    if not cells:
        return "(no code in editor)"

    chunks: list[str] = []
    for ci, cell_src in enumerate(cells, 1):
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        traceback_text: str | None = None
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(cell_src, f"<cell-{ci}>", "exec"), scope)
        except Exception:
            traceback_text = traceback.format_exc()

        out_text = stdout_buf.getvalue().rstrip()
        err_text = stderr_buf.getvalue().rstrip()

        parts: list[str] = []
        if len(cells) > 1:
            parts.append(f"--- Cell {ci} ---")
        if out_text:
            parts.append(out_text)
        if err_text:
            parts.append(f"[stderr]\n{err_text}")
        if traceback_text:
            parts.append(f"[traceback]\n{traceback_text}")
        if not (out_text or err_text or traceback_text):
            parts.append("(no output)")
        chunks.append("\n".join(parts))

        if traceback_text:
            chunks.append(f"_Execution stopped at cell {ci} due to error._")
            break

    return "\n\n".join(chunks)


def _new_round_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + f"{random.randint(0, 0xFFFF):04x}"


def run_cells_for_models(code: str,
                         persisted_globals: dict,
                         selected_keys: list[str],
                         progress=None) -> tuple[list[dict], dict]:
    """Run the notebook code once per selected model. Slideshow-ready output.

    Returns (results, updated_globals). Each result dict:
        {
            "model_key":     str,        # config key, e.g. "gpt-4o"
            "model_display": str,        # e.g. "GPT-4o"
            "response":      str,        # captured plain-text transcript
            "elapsed":       float,      # seconds for this model's run
            "round_id":      str,        # shared across this Run click
        }

    Per-iteration scope injection (visible to the user's code):
        MODEL_KEY        — config key
        MODEL            — LiteLLM model id ("openai/gpt-4o" or
                           "ollama_chat/llama3.2")
        SELECTED_MODELS  — list of all checked keys

    State semantics:
        - 1 model checked → runs against persisted_globals directly so
          state survives across Run clicks (course's iterative pattern).
        - N>1 checked → each iteration runs against a shallow COPY of
          persisted_globals so per-model output doesn't cross-pollute,
          and the outer state stays unchanged.

    progress: optional gr.Progress() instance. Called per model with
              progress(fraction, desc="Running <model>...") so the user
              sees the same Gradio loading bar Arena gets.
    """
    if not selected_keys:
        return [], persisted_globals

    use_shared_state = len(selected_keys) == 1
    round_id         = _new_round_id()
    total            = len(selected_keys)
    results: list[dict] = []

    for i, key in enumerate(selected_keys):
        cfg     = AVAILABLE_MODELS.get(key)
        display = (cfg or {}).get("display_name", key)

        if progress is not None:
            progress((i + 0.0) / total, desc=f"Running {display} ({i+1}/{total})…")

        if cfg is None:
            results.append({
                "model_key":     key,
                "model_display": display,
                "response":      f"<unknown model key {key!r}>",
                "elapsed":       0.0,
                "round_id":      round_id,
            })
            continue

        try:
            model_id = litellm_name(key)
        except ValueError as e:
            results.append({
                "model_key":     key,
                "model_display": display,
                "response":      f"<{type(e).__name__}: {e}>",
                "elapsed":       0.0,
                "round_id":      round_id,
            })
            continue

        scope = persisted_globals if use_shared_state else dict(persisted_globals)
        scope["MODEL_KEY"]       = key
        scope["MODEL"]           = model_id
        scope["SELECTED_MODELS"] = list(selected_keys)

        start    = time.perf_counter()
        response = _exec_cells_plain(code, scope)
        elapsed  = round(time.perf_counter() - start, 3)

        results.append({
            "model_key":     key,
            "model_display": display,
            "response":      response,
            "elapsed":       elapsed,
            "round_id":      round_id,
        })

    if progress is not None:
        progress(1.0, desc="Done")

    return results, persisted_globals
