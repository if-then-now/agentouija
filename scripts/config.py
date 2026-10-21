from pathlib import Path
import math
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# BASE_DIR resolves dynamically to the repo root — config.py sits under
# scripts/, so parent.parent gives the folder that contains scripts/, docs/,
# prompts/, data/, etc. This lets install.py + arena_ui.py work from any
# clone location — C:\AgentOuija\, C:\Users\<name>\Downloads\agentouija-main\,
# ~/agentouija on Linux, wherever. No env var, no user config — the code
# follows the checkout.
BASE_DIR               = Path(__file__).resolve().parent.parent

# .env location — preferred is out of the repo tree so a repo
# zip / clean / accidental delete can't leak or trash your API
# keys. env_file_path() falls back to the legacy repo-root
# location for older installs so no manual migration is needed
# to keep working. install.py's Full mode writes to (and
# migrates any legacy file into) the preferred path.
ENV_FILE_PREFERRED     = Path.home() / ".config" / "agentouija" / ".env"
ENV_FILE_LEGACY        = BASE_DIR / ".env"


def env_file_path() -> Path:
    """Return the .env location the app should read from. Prefers
    the out-of-tree location; falls back to the legacy in-tree
    file when only that one exists. For writes on a fresh install
    (neither present) the caller should default to
    ENV_FILE_PREFERRED — which is what install.py does.

    Also picks up encrypted installs — when a .env.enc pair sits
    next to a location but the plaintext is absent, that location
    is treated as "the .env is here" for pathing purposes.
    Callers use is_encrypted_env() to decide whether to
    load_dotenv() directly or prompt-and-decrypt first."""
    if (ENV_FILE_PREFERRED.exists()
            or ENV_FILE_PREFERRED.with_suffix(".env.enc").exists()):
        return ENV_FILE_PREFERRED
    if (ENV_FILE_LEGACY.exists()
            or ENV_FILE_LEGACY.with_suffix(".env.enc").exists()):
        return ENV_FILE_LEGACY
    return ENV_FILE_PREFERRED


def is_encrypted_env() -> bool:
    """True when the resolved .env location is stored as an
    encrypted .env.enc + .env.enc.meta pair (no plaintext file).
    Runtime callers use this to decide whether to prompt for a
    passphrase before loading."""
    path = env_file_path()
    return (path.with_suffix(".env.enc").exists()
            and path.with_suffix(".env.enc.meta").exists()
            and not path.exists())

SCRIPTS_DIR            = BASE_DIR / "scripts"
DOCS_DIR               = BASE_DIR / "docs"
MEDIA_DIR              = DOCS_DIR / "media"    # uploaded images + video, no auto-ingest

# One-model rule for media (pictures + videos). Cloud image/video tokens
# cost 10-1000x more than text tokens, so Agent Ouija's convention is to
# process media through a single model per query rather than fan out to
# every checked model. The upload handler emits a policy banner ahead of
# the cost preview; a future media-attachment feature will refuse to
# run when this limit is exceeded.
MEDIA_MAX_MODELS       = 1
PROMPTS_DIR            = BASE_DIR / "prompts"
# User-uploaded persona markdown files. When CUSTOM_ACTIVE_FLAG exists,
# prompt_loader swaps out system_core.md for the files in this folder.
CUSTOM_PROMPTS_DIR     = PROMPTS_DIR / "custom"
CUSTOM_ACTIVE_FLAG     = PROMPTS_DIR / "custom_active.flag"
# Presence of this flag flips every model in AVAILABLE_MODELS onto the
# "enhanced" persona stack at runtime — no per-model config edit needed.
# Managed by the Persona tab's "Enable enhanced persona for all models"
# toggle.
ENHANCED_ACTIVE_FLAG   = PROMPTS_DIR / "enhanced_active.flag"
DATA_DIR               = BASE_DIR / "data"
ASSETS_DIR             = BASE_DIR / "assets"          # ouija.svg, itn_logo.png

CHROMA_DIR             = DATA_DIR / "chroma"
CHAT_HISTORY_PATH      = DATA_DIR / "chat_history.json"
ARENA_DB_PATH          = DATA_DIR / "arena_scores.db"
TRAINING_DIR           = DATA_DIR / "training"
# Saved boardroom / slideshow sessions ("seances"). One JSON file per save,
# reload from Agent Ouija's Seances tab. Fully local, gitignored under
# data/.
SEANCES_DIR            = DATA_DIR / "seances"
SUMMARIES_DIR          = DATA_DIR / "summaries"
BACKUPS_DIR            = DATA_DIR / "backups"              # wipe-time snapshots
TAGS_KEYWORDS_PATH     = DATA_DIR / "TAGS_keywords.json"   # gitignored, personal
EPISODES_DIR           = PROMPTS_DIR / "episodes"          # <persona>_*.md files

OUIJA_ICON_PATH        = ASSETS_DIR / "ouija.png"
ITN_LOGO_PATH          = ASSETS_DIR / "itn_logo.png"

# ── Scoring — 0-4 baseball system ─────────────────────────────────────────────
# 0 = whiff       — excluded from training export
# 1 = contact     — weight 0.25
# 2 = base hit    — weight 0.50
# 3 = extra bases — weight 0.75
# 4 = home run    — weight 1.0  ("couldn't tell it wasn't Claude")
SCORE_LABELS = {
    0: "whiff",
    1: "contact",
    2: "base hit",
    3: "extra bases",
    4: "home run",
}
SCORE_WEIGHTS = {
    0: 0.0,
    1: 0.25,
    2: 0.50,
    3: 0.75,
    4: 1.0,
}
SCORE_MAX        = max(SCORE_LABELS)      # 4 — used by compute_reading + UI
SCORE_MIN_EXPORT = 1                       # scores below this excluded from training JSONL
SCORE_RANGE      = list(range(SCORE_MAX + 1))   # [0, 1, 2, 3, 4]

# Valid Refusal — model declined the task on its own safety policy.
# Stored as score = REFUSAL_SCORE (negative, so the existing
# "WHERE score >= 0" filters in leaderboard / persona / tag / trainer
# panels naturally exclude refusals from quality stats. Training export
# also excludes them via SCORE_MIN_EXPORT. They surface only on the
# dedicated Safety Refusals panel.
REFUSAL_SCORE = -2
REFUSAL_LABEL = "valid refusal"


def label_for_score(score: int) -> str:
    """Display label for a stored score value (baseball or refusal)."""
    if score in SCORE_LABELS:
        return SCORE_LABELS[score]
    if score == REFUSAL_SCORE:
        return REFUSAL_LABEL
    return f"score {score}"

# ── The Reading — composite metric ────────────────────────────────────────────
# Combines quality (score) and speed (response_time_seconds) into one number.
# Formula: (score / SCORE_MAX) / log(response_time + e)
# Fast + high score = strong Reading. Slow + low score = weak Reading.
# Log smooths the time penalty — 0.5s vs 1s matters less than 1s vs 10s.
def compute_reading(score: int, response_time_seconds: float) -> float:
    if response_time_seconds < 0 or score < 0:
        return 0.0
    return round((score / SCORE_MAX) / math.log(response_time_seconds + math.e), 4)


# Cost-adjusted Reading — same shape as Reading, with an extra log penalty
# on USD cost. Locals (cost=0) collapse the cost term to log(e)=1, so their
# Cost Reading equals their Reading exactly. Cloud models with positive cost
# pay a smooth penalty proportional to log(cost·1000).
# Formula: (score / SCORE_MAX) / log(time + e) / log(cost*1000 + e)
def compute_reading_with_cost(score: int,
                              response_time_seconds: float,
                              cost_usd: float) -> float:
    if response_time_seconds < 0 or score < 0:
        return 0.0
    cost_clamped = max(0.0, cost_usd or 0.0)
    time_penalty = math.log(response_time_seconds + math.e)
    cost_penalty = math.log(cost_clamped * 1000 + math.e)
    return round((score / SCORE_MAX) / time_penalty / cost_penalty, 6)

# ── Models ────────────────────────────────────────────────────────────────────
# Default rig policy: western-origin models only (US / EU / allied nations).
# Non-western models (Qwen, DeepSeek, Falcon) are commented out below — flip
# them on if that's your fit. Command R (Cohere, Canada) is a planned add.
#
# To add a local model:
#   1. ollama pull <model_name>
#   2. Uncomment the block below
#   3. Add the key to ARENA_MODELS
#   4. Restart arena_ui.py
#
# To add a cloud model:
#   1. Add API key to .env (gitignored): ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
#   2. Uncomment the block below
#   3. Add the key to ARENA_MODELS
#   4. Restart arena_ui.py
#
# Field contracts:
#   env_key         — for cloud models ONLY. Holds the NAME of the env
#                     var (e.g. "ANTHROPIC_API_KEY"), never the raw key.
#                     The key itself lives in .env, which is gitignored.
#   persona         — "generic"  → system_core.md only (cheap, locals)
#                     "enhanced" → system_core + all persona_*.md +
#                                  episodes/<persona>_*.md
#                     See prompt_loader.get_prompt_loader().
#   private_corpus  — when True AND USE_PRIVATE_CORPUS is True, the
#                     model receives extra context retrieved from the
#                     gitignored ChromaDB collection populated by
#                     scripts/private_corpus.py. Lets you toggle persona
#                     enhancement per model — flip a local off if it's
#                     not benefiting.
#   web_search      — when True, the model is eligible for the Research
#                     tab and queries it sends carry the provider's
#                     native web-search tool (Anthropic
#                     web_search_20250305, OpenAI web_search_options,
#                     Gemini googleSearch). Locals and providers without
#                     native search stay False.
AVAILABLE_MODELS = {
    # ── Local — western origin ────────────────────────────────────────────────
    "llama3.2": {
        "name":         "llama3.2",
        "type":         "local",
        "display_name": "Llama 3.2",
        "description":  "Meta — US, general purpose",
        "rag":          True,
        "persona":      "generic",
        "private_corpus": False,
    },
    "mistral": {
        "name":         "mistral",
        "type":         "local",
        "display_name": "Mistral 7B",
        "description":  "Mistral AI — France, European open source",
        "rag":          True,
        "persona":      "generic",
        "private_corpus": False,
    },
    "phi4": {
        "name":         "phi4",
        "type":         "local",
        "display_name": "Phi-4",
        "description":  "Microsoft Research — US, strong at small size",
        "rag":          True,
        "persona":      "generic",
        "private_corpus": False,
    },
    "gemma3": {
        "name":         "gemma3",
        "type":         "local",
        "display_name": "Gemma 3",
        "description":  "Google DeepMind — US, efficient mid-size",
        "rag":          True,
        "persona":      "generic",
        "private_corpus": False,
    },
    "hermes3": {
        "name":         "hermes3",
        "type":         "local",
        "display_name": "Hermes 3",
        "description":  "Nous Research — US, agentic instruction following",
        "rag":          True,
        "persona":      "generic",
        "private_corpus": False,
    },

    # ── Cloud models — uncomment + add API key to .env ────────────────────────
    # Cloud models carry "rag": True so query_model injects ChromaDB context
    # retrieved by scripts/rag.py. Locals have no "rag" key and default to
    # False via .get("rag", False). See CLAUDE.md "Cloud vs Local — Intentionally
    # Unequal" for the design rationale.
     "claude-sonnet": {
         "name":         "anthropic/claude-sonnet-4-6",
         "type":         "cloud",
         "provider":     "anthropic",
         "display_name": "Claude Sonnet",
         "description":  "Anthropic — US, via API key",
         "env_key":      "ANTHROPIC_API_KEY",
         "rag":          True,
         "web_search":   True,
         "persona":      "generic",
         "private_corpus": False,
     },
     "gpt-4o": {
         "name":         "openai/gpt-4o",
         "type":         "cloud",
         "provider":     "openai",
         "display_name": "GPT-4o",
         "description":  "OpenAI — US, via API key",
         "env_key":      "OPENAI_API_KEY",
         "rag":          True,
         "web_search":   True,
         "persona":      "generic",
         "private_corpus": False,
     },
     "gemini": {
         "name":         "gemini/gemini-2.5-flash",
         "type":         "cloud",
         "provider":     "google",
         "display_name": "Gemini",
         "description":  "Google — US, via API key",
         "env_key":      "GEMINI_API_KEY",
         "rag":          True,
         "web_search":   True,
         "persona":      "generic",
         "private_corpus": False,
     },
     "groq": {
         # Groq's catalog rotates — llama-3.3-70b-versatile was
         # deprecated Q3 2026 with a "model_not_found" 404 on the
         # /chat/completions endpoint. Switched to their GPT-OSS
         # 120B offering, which is Groq's current large-tier
         # workhorse. If THIS one gets deprecated too, list live
         # models via:  curl -H "Authorization: Bearer $GROQ_API_KEY"
         # https://api.groq.com/openai/v1/models  and update the
         # `name` here to any id in the response.
         "name":         "groq/openai/gpt-oss-120b",
         "type":         "cloud",
         "provider":     "groq",
         "display_name": "Groq",
         "description":  "Groq — US, ultra-fast inference on OSS models (LPU)",
         "env_key":      "GROQ_API_KEY",
         "rag":          True,
         "persona":      "generic",
         "private_corpus": False,
         # Groq free tier caps per-minute tokens per model at 8000.
         # A single request bigger than that returns "Request too
         # large" with no retry that helps. _query_cloud counts the
         # assembled prompt via litellm.token_counter BEFORE the
         # API call and skips cleanly when we'd exceed. 6500 leaves
         # ~1500 headroom for the response's own token budget
         # (max_tokens=1024 by default, plus completion metadata).
         # Bump to 25_000 / 200_000 / etc. if the user's on Dev
         # Tier — check https://console.groq.com/settings/billing
         # for the current per-model cap.
         "max_input_tokens": 6500,
         # Groq stays a peer contributor even when the global
         # Enhanced Persona toggle is on — the persona stack
         # (~4k tokens) blows past the 8k TPM cap on peer calls,
         # so we skip it here regardless of the flag. Other
         # models still get the enhanced stack when the toggle
         # is set. Private corpus is already opted-out via the
         # `"private_corpus": False` field above.
         "skip_enhanced_persona": True,
     },
     "hermes3-cloud": {
        "name":         "openrouter/nousresearch/hermes-3-llama-3.1-70b",
        "type":         "cloud",
        "provider":     "openrouter",
        "display_name": "Hermes 3 Cloud",
        "description":  "Nous Research via OpenRouter — 70B cloud version",
        "env_key":      "OPENROUTER_API_KEY",
        "rag":          True,
        "persona":      "generic",
        "private_corpus": False,
    },

    # ── Future — planned additions ────────────────────────────────────────────
    # "command-r": {
    #     "name":         "command-r",
    #     "type":         "local",
    #     "display_name": "Command R",
    #     "description":  "Cohere — Canada, enterprise RAG specialist",
    # },

    # ── Future — separate rig (non-western origin) ────────────────────────────
    # "qwen3":       Alibaba, China
    # "deepseek-r1": DeepSeek, China
    # "falcon":      TII, UAE
}
# Used only for command land query. Not For Arena
DEFAULT_MODEL = "llama3.2"

# Update ARENA_MODELS after each model is ready:
# Local:  add key after ollama pull + uncomment block above
# Cloud:  add key after adding API key to .env + uncomment block above
ARENA_MODELS = ["llama3.2", "mistral", "phi4", "gemma3","hermes3","claude-sonnet","gpt-4o","gemini","groq","hermes3-cloud"]

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_TIMEOUT  = 120.0
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Embedding & Vector DB ─────────────────────────────────────────────────────
EMBED_MODEL_NAME       = "BAAI/bge-small-en-v1.5"
CHROMA_COLLECTION_NAME = "mr_ouija"

# ── Private corpus — gitignored, copyrighted material ─────────────────────────
# Episode transcripts and enhanced persona files live in `Copyright/episodes/`,
# which is in .gitignore so source files never reach the repo. A separate
# ChromaDB collection (mr_ouija_private) holds embedded chunks so queries can
# retrieve persona-flavored context at runtime without inlining giant prompts.
#
# Workflow:
#   python scripts/private_corpus.py --ingest     # build/rebuild collection
#   python scripts/private_corpus.py --status     # per-persona chunk counts
#   python scripts/private_corpus.py --export     # portable JSON for migration
#   python scripts/private_corpus.py --import F   # restore on another rig
#   python scripts/private_corpus.py --wipe       # drop the private collection
#
# Gating:
#   USE_PRIVATE_CORPUS              — master kill switch
#   AVAILABLE_MODELS[m]["private_corpus"] — per-model opt-in. A model needs
#                                          BOTH flags True to receive private
#                                          context. Lets you turn off a
#                                          local that's not improving.
PRIVATE_CORPUS_DIR             = BASE_DIR / "Copyright"
PRIVATE_CORPUS_EPISODES_DIR    = PRIVATE_CORPUS_DIR / "episodes"
PRIVATE_CORPUS_COLLECTION_NAME = "mr_ouija_private"
PRIVATE_CORPUS_TOP_K           = 3
USE_PRIVATE_CORPUS             = False
# Runtime override. Presence of this file flips USE_PRIVATE_CORPUS on
# without a config edit / restart — the Persona tab creates/removes it.
PRIVATE_CORPUS_ACTIVE_FLAG     = DATA_DIR / "private_corpus_active.flag"

# ── TTS — Piper (local, free, MIT) ────────────────────────────────────────────
# Arena-tab voice toggle. Fully local, no cloud calls. Voice .onnx + .json
# files live under TTS_VOICES_DIR (gitignored under data/ once you create it
# or anywhere on disk if you point this elsewhere). Setup:
#
#     pip install piper-tts
#     mkdir voices
#     cd voices
#     curl -L -o en_GB-northern_english_male-medium.onnx \
#       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx
#     curl -L -o en_GB-northern_english_male-medium.onnx.json \
#       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json
#
# To swap voices later, change TTS_VOICE and drop the matching .onnx
# pair into voices/. Catalog + samples: https://rhasspy.github.io/piper-samples/
# One shipped voice — British Cori (high). The Persona tab will later
# expose a voice-upload flow so any Piper voice file the user drops in
# can become the runtime default; that plumbing lives there, not here.
TTS_VOICE         = "en_GB-cori-high"
TTS_VOICES_DIR    = BASE_DIR / "voices"
# Runtime-selectable active voice. Persona tab writes the currently-picked
# voice name to this file; tts.py reads it in _load_voice() and falls back
# to TTS_VOICE when the file is missing or points at a voice file that
# isn't actually on disk. Value is just the voice id, no extension.
ACTIVE_VOICE_PATH = TTS_VOICES_DIR / "active_voice.txt"
TTS_MAX_CHARS     = 2000      # truncate longer responses to keep synth snappy

# ── TTS provider dispatch ─────────────────────────────────────────────────────
# Which backend tts.synth_to_wav() dials. "piper" is the default (local,
# free, air-gap capable — matches the project's edge-AI story). "openai"
# routes to tts-1-hd using OPENAI_API_KEY for premium, consistent prosody
# across daily use. "elevenlabs" is reserved — the backend module isn't
# wired yet; selecting it logs a warning and degrades to Piper.
#
# Set the environment variable TTS_PROVIDER to override at launch time
# without a config edit — useful on Rahul's personal rig where the premium
# voice is wanted, while public installs keep the Piper default.
TTS_PROVIDER      = os.environ.get("TTS_PROVIDER", "piper").strip().lower()

# OpenAI TTS settings. Model "tts-1-hd" is the premium tier (slightly
# slower synth, markedly better prosody than "tts-1"). Voices (one of
# alloy / echo / fable / onyx / nova / shimmer) can be swapped via the
# TTS_OPENAI_VOICE env var at launch time without a config edit.
TTS_OPENAI_MODEL  = "tts-1-hd"
TTS_OPENAI_VOICE  = os.environ.get("TTS_OPENAI_VOICE", "onyx").strip().lower()

# ── STT — Microphone input (optional, personal-rig voice chat) ────────────────
# Transcribes mic recordings into the question textbox for daily voice
# conversations with the agent. Default provider is "openai" (uses the
# same OPENAI_API_KEY as everything else, no extra deps). "faster-whisper"
# runs locally on the RTX 5080 for an offline-capable path — requires
# `pip install faster-whisper` and is probed lazily. "off" (or missing
# API key / missing dep) hides the mic UI silently.
#
# Mobile Safari / Chrome require HTTPS for navigator.mediaDevices to
# expose the microphone. On the Tailscale rig this is covered by
# `tailscale serve` which fronts the Gradio port with a trusted cert.
STT_PROVIDER       = os.environ.get("STT_PROVIDER", "openai").strip().lower()
# OpenAI's current speech-to-text model id. gpt-4o-transcribe is their
# newer tier with better accuracy; whisper-1 is the budget option.
# Override via env for cost/quality tuning without a config edit.
STT_OPENAI_MODEL   = os.environ.get("STT_OPENAI_MODEL", "whisper-1").strip()
# faster-whisper on-rig model when STT_PROVIDER="faster-whisper".
# "small.en" is the latency sweet spot on an RTX 5080; "base.en" is
# faster/less accurate; "medium.en" is slower/more accurate.
STT_LOCAL_MODEL    = os.environ.get("STT_LOCAL_MODEL", "small.en").strip()
STT_LOCAL_DEVICE   = os.environ.get("STT_LOCAL_DEVICE", "cuda").strip()
STT_LOCAL_COMPUTE  = os.environ.get("STT_LOCAL_COMPUTE", "float16").strip()
# Hard upper bound on recording length sent to the transcriber. Keeps
# cost + latency bounded if the user forgets to stop recording. 2 min
# at 16 kHz mono WAV is ~3.8 MB; well under API payload limits.
STT_MAX_SECONDS    = 120
# Chat Mode: how many seconds of mic silence before we auto-stop the
# recording and fire Ask. Client-side (JS AudioContext RMS). Tune lower
# for snappier back-and-forth, higher if you need thinking pauses.
CHAT_MODE_SILENCE_SECONDS = 10

# ── NewsAPI (daily news pulls into RAG) ───────────────────────────────────────
# Free tier at https://newsapi.org — 100 requests/day.
# Each pull_* call in scripts/news_puller.py = 1 request.
# Manual full-pull (sports + world + national) = 3 requests.
# Daily scheduled + a couple manual pulls ≈ 3-6 requests/day, well under
# the free-tier ceiling.
NEWSAPI_KEY_ENV      = "NEWSAPI_KEY"
NEWSAPI_ENDPOINT     = "https://newsapi.org/v2/"
NEWSAPI_TIMEOUT      = 20
NEWSAPI_PAGE_SIZE    = 20
NEWS_CACHE_DIR       = DOCS_DIR / "news"       # each pull writes a .md here
NEWS_STATE_PATH      = DATA_DIR / "news_state.json"

# The "sports" pull uses NewsAPI's /v2/everything (search) endpoint with
# this query — baseball-centric per the user's ask. Change the query to
# broaden or narrow ("baseball OR NFL OR NBA" etc).
NEWSAPI_SPORTS_QUERY = "baseball OR MLB"

# ── Google Calendar (read + form-based add/delete) ────────────────────────────
# One-time setup:
#   1. Create a Google Cloud project + enable "Google Calendar API"
#   2. Create OAuth 2.0 client ID (type: Desktop app)
#   3. Download the client-secrets JSON, save as gcal_credentials.json in
#      the path below.
#   4. Run `python scripts\gcal.py --auth` once from the terminal — it
#      opens a browser, asks you to authorize the app on your Google
#      account, then writes the refresh token to gcal_token.json.
# After that, the app reads/writes your primary calendar without further
# prompts. Delete gcal_token.json to force re-auth (e.g. after revoking).
GCAL_CREDENTIALS_FILE = DATA_DIR / "gcal_credentials.json"
GCAL_TOKEN_FILE       = DATA_DIR / "gcal_token.json"
# `calendar.events` covers both read + write on your own calendar.
GCAL_SCOPES           = ["https://www.googleapis.com/auth/calendar.events"]
GCAL_CACHE_DIR        = DOCS_DIR / "calendar"  # RAG-facing dump

# ── Billing / usage dashboards ────────────────────────────────────────────────
# Links shown in the Data tab → 💰 Billing Details section. Each pair is a
# human-readable provider name → the URL where the user tracks their token
# usage / remaining credits on that provider. Config-driven so swapping to
# a different portal (or adding a new provider) is a one-line change here.
BILLING_LINKS = {
    "Anthropic (Claude)":  "https://console.anthropic.com/settings/usage",
    "OpenAI":              "https://platform.openai.com/usage",
    "Google AI (Gemini)":  "https://aistudio.google.com/apikey",
    "Groq":                "https://console.groq.com/settings/usage",
    "OpenRouter (Hermes)": "https://openrouter.ai/settings/credits",
    "Serper.dev":          "https://serper.dev/dashboard",
}

# ── Serper.dev (Google search API for the Research tab) ───────────────────────
# Powers the Research tab — one Serper call per question, results injected
# as web_context into every selected model's prompt. Lets locals + clouds
# both answer with web-grounded facts (unlike The Big 3 tab, which only
# uses providers with native search).
#
# Free tier: 2,500 searches/month. Paid: ~$5 / 1k. Sign up at serper.dev
# then drop the key into .env under SERPER_API_KEY (gitignored).
SERPER_API_KEY_ENV = "SERPER_API_KEY"
SERPER_ENDPOINT    = "https://google.serper.dev/search"
SERPER_TOP_K       = 6
SERPER_TIMEOUT     = 15

# ── Enhanced stack trimming ───────────────────────────────────────────────────
# When the enhanced persona stack (system_core + enhanced_persona_main +
# persona_<name>.md + episodes/*.md) grows past this many characters,
# prompt_loader drops episode files first — they're usually the biggest
# and least essential (reference material rather than character floor).
# Rough guide: ~4 chars ≈ 1 token, so 60_000 chars ≈ 15k tokens.  Hermes
# 3 Cloud caps at 131k tokens; leaving headroom for the question, RAG
# context, and the response itself.  Set to None to disable auto-trim.
MAX_ENHANCED_STACK_CHARS = 60_000

# ── Conversation memory (follow-ups) ──────────────────────────────────────────
# Number of prior (question, answer) turns to inject as user/assistant
# messages before the current question so follow-ups like "we're under
# 25" resolve against what "25" referred to a turn ago. Kept small
# because every peer + exec call re-sends these turns — 6 turns of
# ~500 tokens each = ~3k extra tokens per call, comfortable on all
# models. Turns beyond the window still exist in the seance JSON on
# disk; the age-out RAG layer surfaces them semantically when relevant.
CONVERSATION_HISTORY_WINDOW = 6

# ── Image editing (Gemini "Nano Banana") ─────────────────────────────────────
# Fallback chain of model IDs to try in order for the instruction-
# based image editor. Google has renamed this checkpoint several
# times as it graduated from experimental → preview → stable:
#   gemini-2.0-flash-exp-image-generation  (early 2025)
#   gemini-2.5-flash-image-preview         (mid 2025, "Nano Banana")
#   gemini-2.5-flash-image                 (stable, no -preview)
# Different accounts also see different availability by region.
# image_edit.py tries each in order, caches the first that works
# for the process lifetime, and stops on non-404 errors.
#
# Run `python scripts/image_edit.py --list` to see what your key
# can actually reach if all of these 404.
GEMINI_IMAGE_MODELS = (
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-exp-image-generation",
)

# ── Personas ──────────────────────────────────────────────────────────────────
# Agent Ouija ships with a blank persona canvas. The default voice is the
# mystical Agent Ouija tone defined in prompts_template/system_core.md.
# Every additional persona is fully user-defined — drop persona_<name>.md
# files into prompts/ and add the <name>s to PERSONA_NAMES below.
#
# PERSONA_REINFORCEMENT is prepended to every user message to keep local
# Ollama models from drifting out of character. Claude / GPT-4o / Gemini were
# RLHF'd to respect the system prompt strongly, but quantized Llama / Mistral
# / Phi / Gemma treat it as a soft suggestion — a brief reinforcement in the
# user turn pulls them back into the persona without bloating the system
# prompt. Set to "" (empty string) to disable.
PERSONA_REINFORCEMENT = (
    "[Agent Ouija: stay fully in character. Respond using the personas in your "
    "system prompt — pick the right voice for the question. Do not break "
    "character or describe yourself as an AI assistant.]"
)

# ── Persona loading mode ──────────────────────────────────────────────────────
# Each model in AVAILABLE_MODELS carries a "persona" field with one of these
# values. prompt_loader.get_prompt_loader(persona_mode) reads it.
#   "generic"  → just system_core.md (locals get a small, fast system prompt)
#   "enhanced" → system_core + all persona_*.md + episodes/<persona>_*.md
#                (cloud models get the full character stack, if defined)
PERSONA_MODES = ("generic", "enhanced")
DEFAULT_PERSONA_MODE = "generic"

# The personas listed here drive episode discovery: for each name the loader
# globs PROMPTS_DIR/episodes/<name>_*.md and concatenates the matches after the
# main persona_<name>.md file. Add a persona by dropping persona_<name>.md
# (and optionally episodes/<name>_*.md) into prompts/ and adding the name here.
# Empty tuple = "no custom personas defined yet" — enhanced-mode falls back to
# system_core only. See the Persona tab in the UI for a guided upload flow.
PERSONA_NAMES: tuple[str, ...] = ()

# ── Persona-lean detection ────────────────────────────────────────────────────
# arena optionally tracks which of your defined personas a response leaned on
# via a cheap keyword scan at log time. Configure per-persona keyword lists
# below (mapped by name — must match a name in PERSONA_NAMES). Detection
# counts case-insensitive substring matches per bucket; highest non-zero
# count wins, ties → "mixed", all zero → "none". Leave empty if you don't
# use personas or don't care about lean tracking.
#
# Example (uncomment + adapt for your own personas):
#   PERSONA_MARKERS = {
#       "wizard":   ["indeed", "alas", "the path", "wisdom", "ancient"],
#       "advisor":  ["deliverable", "stakeholder", "milestone", "kpi"],
#       "comedian": ["what's the deal", "so anyway", "you know what"],
#   }
PERSONA_MARKERS: dict[str, list[str]] = {}
# Stored values for arena_results.persona_lean — keep aligned with PERSONA_NAMES
# plus the two synthesized buckets.
PERSONA_LEAN_LABELS = list(PERSONA_NAMES) + ["mixed", "none"]