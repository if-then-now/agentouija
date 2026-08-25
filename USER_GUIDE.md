# Agent Ouija — User Guide

Comprehensive walkthrough of every feature, every tab, every concept.
For the abbreviated install path, see [SETUP.md](SETUP.md).

---

## What's new since this guide was first written

The tab structure has evolved. The four separate tabs this guide
was originally organized around (Arena, Notebook, Research, The
Big 3) have collapsed into a single **Agent Ouija** tab whose
Mysticism Level dropdown picks the mode:

- **Low Energy** — locals only (no web, no cloud)
- **Medium Energy — Agile** — Serper web-context injected into
  every checked model, cloud pre-checked
- **Medium Energy — Synergistic** — same as Agile but locals +
  clouds together
- **High Energy** — provider-native web search (Claude,
  GPT-4o-search-preview, Gemini googleSearch)

Additional capabilities that landed after this guide's original text:

- **Ask the Lore** — companion to Ask For Revelations that adds a
  STRICT GROUNDING directive so the models answer only from RAG /
  doc / web context and refuse with "Not in the lore" otherwise.
- **Executive Boardroom tiers** — Junior / Senior / C-Suite exec
  buttons synthesize the peer round into one answer.
- **Instruction-based image editing** — reference an image file
  in your prompt + an edit verb ("add a heart between the two
  people, wedding.png") and Google Gemini's Nano Banana model
  runs the edit locally in your session and saves the result to
  `docs/media/`. Files tab includes an inline preview and an Open
  Folder button.
- **Sticky UI preferences** — the Mysticism Level and both
  Executive Boardroom checkboxes persist across restarts via
  `data/preferences.json`. Reset button in the Data tab.
- **Historical cost breakdown** on the Data tab — lifetime total,
  last 7 / 30 days, since last checkpoint, per-model and per-
  provider rollups. Auto-loads on tab open.
- **Time-based rollback** on the Data tab — number + days/weeks/
  months + target (ChromaDB / arena_scores.db / seances / news
  docs / all) with preview-then-YES-then-purge, always backed up
  first into `data/backups/rollback_<stamp>/`.
- **Backup restore** on the Data tab — pick a snapshot, preview
  contents, restore with typed YES.
- **Google Calendar** — Read / Add / Delete events section on the
  Data tab. Uses OAuth (Testing status → weekly re-auth via a
  dedicated 🔐 Authorize Google Calendar button; no more mid-
  workflow freezes when the token expires).
- **NewsAPI daily pulls** — Sports / World / National categories
  cached into `docs/news/` and ingested into RAG.
- **Full historical Billing Details** panel showing every provider
  the rig has ever spent money with, auto-refreshed on tab open.

The sections below still walk through the original scoring /
training / persona / RAG concepts — those haven't changed.

---

## Table of Contents

1. [What Agent Ouija Is](#what-agent-ouija-is)
2. [Launching Agent Ouija](#launching-agent-ouija)
3. [The Four Work Tabs](#the-four-work-tabs)
   - [Arena](#1-arena--blind-multi-model-evaluation)
   - [Notebook](#2-notebook--programmatic-prompts)
   - [Research](#3-research--serper-web-search-for-all-10-models)
   - [The Big 3](#4-the-big-3--native-web-search)
4. [The Stats Tab](#the-stats-tab)
5. [Core Concepts](#core-concepts)
   - [The Scoring System](#the-scoring-system--04-baseball)
   - [The Reading + Cost Reading](#the-reading--cost-reading)
   - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
   - [Personas](#personas)
   - [Private Corpus](#private-corpus)
   - [Voice TTS](#voice-tts)
   - [Training Workflow](#training-workflow)
6. [Configuration Reference](#configuration-reference)
7. [Deployment Story](#deployment-story)
8. [Troubleshooting](#troubleshooting)

---

## What Agent Ouija Is

Agent Ouija is a **local edge AI assistant and multi-model evaluation arena**.
Two things at once:

- **A working tool** for federal/enterprise-style on-premise AI deployment
  — sovereign, air-gap capable, no vendor lock-in for inference.
- **A training rig** for fine-tuning local models against frontier cloud
  models — every scored response is signal that drives the next LoRA
  cycle.

The product vision is *purpose-built single-domain devices*. A medical
edge box that knows medical facts. A legal kiosk that knows legal facts.
A field-engineering tablet that knows your equipment. Same architecture,
different RAG store, same offline capability.

**Hardware tested:** RTX 5080, Windows 11 Pro. Smaller GPUs work — model
selection at install time lets you skip what won't fit. Even a Raspberry
Pi 5 can run a 3B quantized model + ChromaDB + Piper for a roaming
single-purpose unit.

---

## Launching Agent Ouija

After running `install.py` you have two equivalent launch paths:

- **Double-click `Agent Ouija.bat`** in the repo root, or the desktop
  shortcut `Agent Ouija.lnk` if you accepted that prompt during install.
- **Run the script directly:** `python scripts\arena_ui.py`.

Either way, the default browser opens automatically to
<http://localhost:7860> and pulls to the foreground. The console window
that launched the server stays open — closing it stops Agent Ouija.

### Header controls

Across the top of the UI:

- **Agent Ouija logo + title** (left).
- **🌓 Toggle Theme** button (right) — flips between light and dark mode
  by adding `?__theme=light` or `?__theme=dark` to the URL and
  reloading. With no parameter present, the page respects your system
  preference — which is why a freshly launched Agent Ouija comes up dark
  on a system in dark mode and light on a system in light mode. The URL
  param overrides system preference, so you can bookmark a specific URL
  per rig if you want a particular mode every time.

---

## The Four Work Tabs

The web UI exposes four work tabs — Arena · Notebook · Research · The
Big 3 — plus the analytics-focused Stats tab. Tab order left to right.

### 1. Arena — Blind Multi-Model Evaluation

The core feature. Pitch one question out to any subset of the 10 models
(5 local + 5 cloud) and score each response on the 0-4 baseball scale
plus Valid Refusal.

**Layout:**

- **Local Models / Cloud Models** checkboxes (all 10 ticked by default).
- **Question** textbox.
- **Blind mode** toggle (on by default) — hides the model name on each
  slide so you can't see who you're scoring.
- **🔊 Voice** toggle — when on, Piper TTS speaks each response as it
  appears. Auto-plays on every slide change.
- **Ask For Revelations** — pitches the question to every selected model
  in parallel. Orange progress bar narrates per-model completion:
  `✓ Llama 3.2 done (2/5) · waiting on: Claude Sonnet, GPT-4o`.
- **Ingest Documents** — runs the doc ingestion pipeline against your
  `docs/` folder. Drop new files in, click ingest, the main ChromaDB
  collection rebuilds.

**Response slideshow** (one model at a time):

- Model label (`Model A` while blind, real name while not) + elapsed
  time.
- Response textbox (read-only).
- 🔊 Audio player (visible only when Voice toggle is on).
- **Score** radio — five baseball ratings plus Valid Refusal:
  - `0` — **whiff**: missed entirely
  - `1` — **contact**: something useful
  - `2` — **base hit**: solid, did the job
  - `3` — **extra bases**: better than expected
  - `4` — **home run**: couldn't tell it wasn't Claude
  - `-2` — **valid refusal**: model declined on safety policy (excluded
    from training stats but tracked separately on the Refusals panel)
- **Score And Next Response** — log the score, advance to next slide.
- **Score + Add To RAG** — log the score AND ingest the Q+A pair into
  your main ChromaDB collection so future questions can retrieve it.
  Use this when the answer is good enough to become reference material.
- **Retry Scoring** — re-shuffle and re-step through the same responses.
- **◀ Previous / Next ▶** — navigate without scoring (read-only review).
- **Skip To New Prompt** — abandon the current round, start over.

**Persistence:** every scored response writes a row to
`data/arena_scores.db` (gitignored) with the score, elapsed time, The
Reading, the Cost Reading, token counts, persona-lean detection, and
`round_id` for trainer-report grouping. Score + Add To RAG additionally
writes to the `rag_contributions` table for per-model RAG attribution.

---

### 2. Notebook — Programmatic Prompts

Course-style Jupyter-like tab for learning programmatic API patterns.

**Layout:**

- Same **Local Models / Cloud Models** checkboxes as Arena.
- **Blind mode** toggle (on by default).
- Python code editor pre-seeded with a course-style `litellm.completion`
  call.
- **Run** button.
- **Reset State** — clears persistent variables.

**Cells:** split the editor with `# %%` on its own line (Spyder / VS Code
interactive window convention). Each cell runs independently in the
output.

**Per-model iteration:** the code runs **once per checked model**.
Before each iteration, three variables are injected into scope:

- `MODEL` — LiteLLM model id (`"openai/gpt-4o"` /
  `"ollama_chat/llama3.2"`) — drop straight into
  `litellm.completion(model=MODEL, …)`.
- `MODEL_KEY` — config key (`"gpt-4o"`).
- `SELECTED_MODELS` — list of all checked keys.

**Helpers in scope** (no imports needed):

- `ask(prompt, model='claude-sonnet', system=None, temperature=None,
  max_tokens=1024) → str`
- `ask_all(prompt, system=None, …) → dict` — fan-out to every model in
  config (ignores checkboxes).
- `get_rag(question) → str | None` — fetch ChromaDB context manually.

**State semantics:**

- 1 model checked → runs against persistent globals; variables survive
  across Run clicks.
- N>1 checked → each iteration runs against a shallow copy so per-model
  output doesn't cross-pollute.

**Scoring:** same slideshow + 0-4 baseball + Valid Refusal as Arena.
Notebook scores land in `arena_scores.db` with `tag_source="notebook"`.

---

### 3. Research — Serper Web Search for All 10 Models

Fetches one Google Search result block (via Serper.dev's API) and injects
it as system context into **every selected model** — locals + clouds, all
10. Lets you compare how each model reasons over the same web evidence.

**Layout:**

- Same Arena layout: Local + Cloud checkboxes, blind mode, voice
  toggle, question textbox, **Research The Web** button.
- Same slideshow + scoring + Score And Next / Score + Add To RAG /
  Previous / Next / Skip / Retry buttons.

**How it differs from Arena:**

- One Serper call per question (~$0.005 on the paid tier; free tier is
  2,500 searches/month).
- Web context block (search results) injected into every model's prompt
  unconditionally — no per-model `rag` flag needed.
- Scores write with `tag_source="research"` so the Stats tab can split
  them out.
- **Score + Add To RAG** metadata source is `"research_serper"`.

**Setup:** add `SERPER_API_KEY=your_key` to `.env`. Sign up at
[serper.dev](https://serper.dev) for the free tier key.

---

### 4. The Big 3 — Native Web Search

Three providers that ship their own web-search tool: Anthropic
(`web_search_20250305`), OpenAI (`gpt-4o-search-preview` with
`web_search_options`), Google (`googleSearch` grounding). Each provider
runs its own internal search and synthesizes a cited answer.

**Layout:**

- Three model checkboxes only (Claude Sonnet, GPT-4o, Gemini).
- Question textbox, **Search The Web** button.
- Slideshow + Previous / Next / Skip nav.
- **No scoring** — research is for finding facts, not training.
- **Add To RAG** is the only persistence action.

**When to use Research vs The Big 3:**

| | Research (Serper) | The Big 3 (native) |
|-|-|-|
| Models covered | All 10 | 3 (Claude, GPT-4o, Gemini) |
| Search source | Serper (Google) | Provider's own infra |
| Citations | URL list in context | Inline / structured |
| Comparison fairness | All models see same evidence | Each provider's search differs |
| Scoring | Yes | No |
| Cost | Serper + per-model | Per-model (search inflates token use) |

Use **Research** for training-signal collection ("which model reasons
best over this evidence"). Use **The Big 3** when you want the model's
own search judgment, not just its reasoning.

---

## The Stats Tab

Leaderboards, analytics, and the danger zone.

### Top-row buttons

- **Refresh Stats** — re-runs every query on demand.
- **Generate Summary** — writes
  `data/summaries/summary_<stamp>.txt` with the per-question detail for
  the last 24 hours (every scored response, full text, score, Reading).
- **Trainer Report** — writes `data/summaries/trainer_report_<stamp>.txt`
  with per-model Taught / Learned / RAG counts.
- **Mark Training Checkpoint** — inserts a row into
  `training_checkpoints` so subsequent rounds count as the
  "Current Session" view in the leaderboard.

### Tables

- **Historical Stats** (all-time): Model, N, Avg Score, Avg Arena, Avg
  Notebook, Avg Time, The Reading, Cost Reading, score-bucket counts,
  Cost (USD), **RAG Adds**.
- **Current Session Stats** (since last checkpoint): same columns plus
  **Δ Reading** vs pre-checkpoint baseline.
- Both DataFrames support click-to-sort on any column. Default order:
  Avg Score (historical) / The Reading (session).

### Below the leaderboards

- **Training checkpoints** — log of every checkpoint with date + name +
  notes.
- **Safety refusals** — per-model Valid Refusal counts.
- **Top tags** — most common question tags (from `TAGS_keywords.json`).

The persona-lean tables are intentionally not rendered by default.
Agent Ouija ships with an empty `PERSONA_NAMES` tuple — you configure
your own personas in `config.py`. The `persona_lean` column + detection
at log time still run whenever `PERSONA_NAMES` is populated, so the
data is collected. To turn the panel on, add `persona_md = gr.Markdown(
build_persona_stats())` back in `arena_ui.py` next to where the
checkpoint log is rendered (an inline comment marks the spot), plus
the refresh / wipe handler wirings.

### Danger Zone

**Wipe Database** opens a three-way confirmation:

- **Yes, Backup the database and chats** — snapshots `arena_scores.db`,
  `chat_history.json`, and the entire `chroma/` dir into
  `data/backups/wipe_<stamp>/` before deleting. If the backup fails,
  the wipe aborts so you never lose data unintentionally.
- **Yes, Don't backup the database and chats** — wipes immediately.
- **No** — cancel.

Wipe deletes all scored rows, all training checkpoints, all RAG
contributions, the ChromaDB collection, and the chat history. Source
files in `docs/`, `prompts/`, `Copyright/`, and your `.env` are
untouched.

---

## Core Concepts

### The Scoring System — 0-4 Baseball

| Score | Label | Training weight | Meaning |
|-------|-------|-----------------|---------|
| 0 | whiff | 0 (excluded) | Missed entirely |
| 1 | contact | 0.25 | Something useful |
| 2 | base hit | 0.50 | Solid, did the job |
| 3 | extra bases | 0.75 | Better than expected |
| 4 | home run | 1.0 | Couldn't tell it wasn't Claude |
| -2 | valid refusal | — (tracked separately) | Model declined on its safety policy |

The 4 is the bar. When a local hits 4 on an Architect-grade prompt with
the same RAG context everyone else got, the weight-level improvements
are doing the work — that's a successful training cycle.

Edit `SCORE_LABELS` / `SCORE_WEIGHTS` in `config.py` to change the scale
— `compute_reading()` uses `SCORE_MAX` as denominator so it tracks
automatically.

### The Reading + Cost Reading

**The Reading** combines quality (score) and speed (response time) into
one number:

```
Reading = (score / SCORE_MAX) / log(response_time_seconds + e)
```

Fast + high score = strong Reading. Slow + low score = weak Reading.
Log smooths the time penalty (0.5s vs 1s matters less than 1s vs 10s).

**Cost Reading** extends with a USD-cost penalty:

```
Cost Reading = (score / SCORE_MAX) / log(time + e) / log(cost·1000 + e)
```

For locals (cost = 0) the cost denominator collapses to `log(e) = 1` —
their Cost Reading equals their Reading exactly. Cloud models pay a
smooth log penalty proportional to actual USD spend.

Both are stored per-row in `arena_results.reading` /
`arena_results.cost_reading` and surfaced on the Stats tab.

### RAG (Retrieval-Augmented Generation)

Every model with `"rag": True` in `config.AVAILABLE_MODELS` receives an
extra system message with top-k chunks retrieved from the main ChromaDB
collection at query time. Defaults to True for every shipped model
(locals + clouds) — flip to False on any individual model to use it as a
no-RAG control.

**Two ways to fill the collection:**

1. **Ingest documents** — drop files into `docs/`, click **Ingest
   Documents** on the Arena tab. The pipeline reads + chunks +
   embeds + writes. Rebuilds the collection from scratch on every run so
   repeated clicks can't create duplicates.

2. **Curate from arena rounds** — click **Score + Add To RAG** in
   Arena/Research to ingest the Q+A pair after scoring. Every successful
   add writes an attribution row to `rag_contributions` so the Stats tab
   and trainer report can show per-model contribution counts.

The embedding model is `BAAI/bge-small-en-v1.5` (384-dim, ~130MB,
cached locally on first use). All write paths in `rag.py` pin this
model unconditionally to avoid dimension mismatches.

### Personas

Agent Ouija ships with a **blank persona canvas**. The default voice is
the mystical Agent Ouija tone in `prompts_template/system_core.md`.
Every additional persona is user-defined:

1. Drop `persona_<name>.md` files into `prompts/` (gitignored — your
   personas stay on your rig).
2. Optionally add `episodes/<name>_*.md` files for extended reference
   material.
3. Add each `<name>` to the `PERSONA_NAMES` tuple in `config.py`.

When multiple personas are defined they load together as a single
unified system prompt — no role selection, the LLM picks the register
from the question's context.

**Two modes** (per-model in `config.AVAILABLE_MODELS["persona"]`):

- `"generic"` (default) — just `prompts/system_core.md`. Locals use this
  to keep the system prompt small and inference fast.
- `"enhanced"` — `system_core.md` + each `persona_<name>.md` from
  `prompts/` + every `episodes/<name>_*.md` discovered by glob.

A short **persona reinforcement** string is prepended to every user
message — quantized locals deprioritize the system prompt and would
otherwise drift out of character. Set `PERSONA_REINFORCEMENT = ""` in
`config.py` to disable.

**Persona lean detection:** if you populate `PERSONA_MARKERS` in
`config.py` with keyword lists per persona, every scored response is
scanned and the detected lean (persona name / mixed / none) is stored
in `arena_results.persona_lean`. The dedicated Stats-tab panel that
breaks this out into per-persona tables is hidden by default (empty
`PERSONA_NAMES` means nothing to show); re-enabling it once you've
defined personas is a two-line change in `arena_ui.py`.

### Private Corpus

For copyrighted material (episode transcripts, character bios, etc.)
that should never reach the repo. A second ChromaDB collection
(`mr_ouija_private`) populated from `Copyright/episodes/`, gitignored.

**Per-model opt-in:** `"private_corpus": True` in
`AVAILABLE_MODELS` + `USE_PRIVATE_CORPUS = True` master switch in
`config.py`. A model needs BOTH flags to receive private context.

**Workflow:**

```powershell
# On dev rig (or wherever you build it)
python scripts\private_corpus.py --ingest    # build collection
python scripts\private_corpus.py --status    # per-persona chunk counts
python scripts\private_corpus.py --export    # → data/private_corpus_<stamp>.json

# Copy the JSON to the deploy rig (Brainy)
# On Brainy
python scripts\private_corpus.py --import data\private_corpus_<stamp>.json
# Flip USE_PRIVATE_CORPUS = True in config.py
# Flip "private_corpus": True on the models that should receive it
```

**Filename convention** for `Copyright/episodes/*.md`:
`{persona}_{anything}.md` — the prefix maps to the persona metadata.
If your `PERSONA_NAMES = ("wizard", "advisor")`, then
`wizard_council.md` and `advisor_kickoff.md` map to those personas.
Files without a matching prefix are tagged `persona="unknown"` and
still ingested.

### Voice TTS

Piper TTS (local, MIT-licensed, no cloud) wired to the Arena tab's 🔊
Voice toggle. Default voice: `en_GB-northern_english_male-medium`.

**Setup** (the installer handles this if you pass `--voice`):

```powershell
pip install piper-tts
cd C:\AgentOuija\voices
# Download the .onnx + .onnx.json pair from
# https://huggingface.co/rhasspy/piper-voices
```

**Swap voices** later: change `TTS_VOICE` in `config.py` and drop the
new `.onnx` pair into `voices/`. Voice catalog with audio samples:
<https://rhasspy.github.io/piper-samples/>

**Sanitization:** every response is run through `clean_for_speech()`
before synth — strips markdown, emoji, code blocks, headers, bullets,
links — so Piper speaks natural prose instead of reading "asterisk
asterisk bold asterisk asterisk".

**Failure modes degrade gracefully:** missing pip dep, missing voice
file, synth error — all silent no-op, the UI never crashes.

### Training Workflow

```
Pitch question → Score blind → (optional) Score + Add To RAG
→ Mark Training Checkpoint → Export JSONL → LoRA fine-tune locals
→ Re-checkpoint → Re-run same questions → Track Δ Reading in Stats
```

**Export training data:**

```powershell
# All scored responses since last checkpoint, score >= 1
python scripts\prepare_training.py --since-checkpoint --min-score 2

# Specific model
python scripts\prepare_training.py --model llama3.2

# Specific tag (set up in data\TAGS_keywords.json)
python scripts\prepare_training.py --tag "ARCHITECTURE"
```

Output: `data/training/training_<stamp>.jsonl` with one record per
scored response, including weight (from `SCORE_WEIGHTS`), the
question/response pair, response_time, Reading, model, tags.

**Mark a checkpoint** after deploying a new fine-tuned model so the
Δ-tracking starts fresh:

```powershell
python scripts\training_manager.py --checkpoint "LoRA run 1 — assignment 1" --notes "50 examples, lr=2e-4"
```

Or just click **Mark Training Checkpoint** on the Stats tab. The
**Current Session** column then re-runs the same questions and the
Δ Reading column shows whether the model is improving against
its pre-checkpoint baseline.

---

## Configuration Reference

### `scripts/config.py` — The Only Source of Truth

Single source of truth. Every path, model, score, persona setting lives
here. Adding a model is one config block + one ARENA_MODELS entry.
Nothing else.

**Key sections:**

| Section | What |
|---------|------|
| Paths | `BASE_DIR`, `DATA_DIR`, `CHROMA_DIR`, `BACKUPS_DIR`, etc. |
| Scoring | `SCORE_LABELS`, `SCORE_WEIGHTS`, `SCORE_MAX`, `REFUSAL_SCORE` |
| Reading | `compute_reading()`, `compute_reading_with_cost()` |
| Models | `AVAILABLE_MODELS` dict, `ARENA_MODELS` list, `DEFAULT_MODEL` |
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT` |
| RAG | `EMBED_MODEL_NAME`, `CHROMA_COLLECTION_NAME` |
| Private corpus | `PRIVATE_CORPUS_*`, `USE_PRIVATE_CORPUS` |
| TTS | `TTS_VOICE`, `TTS_VOICES_DIR`, `TTS_MAX_CHARS` |
| Serper | `SERPER_API_KEY_ENV`, `SERPER_ENDPOINT`, `SERPER_TOP_K` |
| Personas | `PERSONA_NAMES`, `PERSONA_MARKERS`, `PERSONA_REINFORCEMENT` |

**`AVAILABLE_MODELS` field contracts:**

| Field | Type | Notes |
|-------|------|-------|
| `name` | str | LiteLLM model id (cloud) or Ollama tag (local) |
| `type` | `"local"` / `"cloud"` | Routing in `query_model` |
| `provider` | str | Cloud only — `"anthropic"` / `"openai"` / `"google"` / `"groq"` / `"openrouter"` |
| `display_name` | str | Label in checkboxes + slideshow |
| `description` | str | Shown in installer prompts |
| `env_key` | str | Cloud only — NAME of env var, not the raw key |
| `rag` | bool | If True, retrieves from main ChromaDB |
| `private_corpus` | bool | If True AND master switch on, also retrieves private corpus |
| `persona` | `"generic"` / `"enhanced"` | Prompt loader mode |
| `web_search` | bool | Cloud only — eligible for The Big 3 tab + native search routing |

### `.env` — API Keys

Gitignored. One key per cloud provider plus Serper:

```bash
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
SERPER_API_KEY=
```

Leave any blank — the corresponding cloud model just won't work until
the key is filled in.

---

## Deployment Story

The headline use case: ship a local model + curated ChromaDB +
embedding model cache + Piper voice to a device with no internet.

**Roaming bundle** (rough sizes for a 3B Q4 local):

| Component | Size | Path |
|-----------|------|------|
| Llama 3.2 (3B Q4) | ~2 GB | `~/.ollama/models/` |
| Embedding model (bge-small) | ~130 MB | HF cache |
| ChromaDB with curated chunks | tens of MB | `data/chroma/` |
| Piper voice (medium) | ~60 MB | `voices/` |
| Ollama runtime | ~200 MB | system |
| Agent Ouija scripts | <1 MB | this repo |

Total: a few GB. Phone-sized. Raspberry Pi 5 territory.

**Domain-specific variants** the architecture supports out of the box:

- **Medical Agent Ouija** — ingest FDA pathways, clinical guidelines,
  drug interactions into `docs/`. Curate via Score + Add To RAG in the
  Arena. Local Llama 3.2 fine-tuned on a few hundred medical Q+A pairs.
  Air-gap-capable medical reference.
- **Legal Agent Ouija** — same architecture, different RAG. Case law,
  jurisdiction-specific statutes, contract templates.
- **Field Engineering Agent Ouija** — equipment manuals, safety
  procedures, troubleshooting flowcharts.

The cloud teachers stay at the dev rig. Their lessons live on in the
local's weights and the RAG store the device carries.

---

## Troubleshooting

**Arena says "Ollama unavailable"** — Ollama service isn't running.
Start the service via the Windows tray or `ollama serve`.

**Cloud model errors with "{KEY} is not set in environment"** —
edit `.env` and add the missing key. Restart `arena_ui.py` so
`load_dotenv()` re-reads.

**Voice toggle visible but no sound** —
1. Check the `arena_ui.py` terminal for a `Failed to load Piper
   voice` warning.
2. Verify both `voices/en_GB-northern_english_male-medium.onnx` AND
   `voices/en_GB-northern_english_male-medium.onnx.json` exist.
3. Run `python -c "import piper; print(piper.__version__)"` to
   verify the pip dep installed.
4. On Python 3.14+ the underlying onnxruntime wheel may be missing —
   drop to Python 3.11 or 3.12 if so.

**"Embedding dimension mismatch (384 vs 1536)"** when adding to RAG —
LlamaIndex's default `Settings.embed_model` lazily resolves to
OpenAI's 1536-dim. The current code pins bge-small unconditionally
in every RAG write path; if you see this in older history, pull the
latest dev.

**Trainer report shows 0 taught/learned for everyone** — every model
needs at least one PEER in the same round to have a teach/learn
relationship. The pre-seed step ensures models appear with N>0 / 0 / 0
when they only had single-model rounds.

**Stats RAG Adds column shows 0** — the data starts accumulating
from the commit that added the `rag_contributions` table forward.
Pre-existing RAG additions aren't retroactively counted.

**ChromaDB / Persona stats blank after wipe** — wipe drops the
collection and the cached retriever. Click **Ingest Documents** on the
Arena tab to rebuild. Persona stats need newly logged rows from the
post-wipe DB.

**Big 3 tab GPT-4o errors with "Web search options not supported"** —
The code routes OpenAI web-search through `gpt-4o-search-preview`. If
the error persists, your OpenAI account tier may not include access to
the search-preview model — check your usage page.

**Long-running issues:** read the trainer report
(`data/summaries/trainer_report_*.txt`) — the Δ column tells you which
models are improving and which are stalling. Models with N>0 but
Taught=Learned=0 only saw single-model rounds in the window; ask more
multi-model questions to get pairwise signal.

---

## Architecture Reference

### Source tree (committed)

```
agentouija/
├── CLAUDE.md              # Project intent / philosophy (single source of truth)
├── SETUP.md               # Quick install
├── USER_GUIDE.md          # This file
├── README.md              # (if present)
├── install.py             # Interactive installer
├── requirements.txt       # Python deps
├── .env.example           # Template, NOT real keys
├── .gitignore             # Gitignore everything personal
└── scripts/
    ├── config.py          # SINGLE SOURCE OF TRUTH
    ├── db.py              # arena_scores.db schema + WAL + helpers
    ├── rag.py             # Main + private ChromaDB retrieval
    ├── arena.py           # Terminal arena
    ├── arena_ui.py        # Gradio web UI (the main entry point)
    ├── notebook_runner.py # Per-cell exec for the Notebook tab
    ├── serper_search.py   # Serper.dev wrapper
    ├── tts.py             # Piper wrapper + speech sanitizer
    ├── tagger.py          # Keyword + LLM tagger for arena rows
    ├── persona_lean.py    # Persona detection from response text
    ├── prompt_loader.py   # System prompt assembly
    ├── ingest.py          # Document → ChromaDB pipeline
    ├── private_corpus.py  # Gitignored corpus manager
    ├── prepare_training.py # JSONL export for LoRA
    ├── training_manager.py # Checkpoint markers
    ├── summarize.py       # Summary + trainer reports
    ├── memory.py          # Chat history persistence
    ├── query.py           # Single-model CLI
    └── log_parser.py      # Captain's Log file parser
```

### Gitignored / personal

```
prompts/             # Personal system prompts and personas
docs/                # Personal documents to ingest
Copyright/           # Copyrighted material for private corpus
voices/              # Piper .onnx files
data/                # Everything generated at runtime
.env                 # Real API keys
assets/              # Licensed graphics
```

### Design principles

- **SOLID** — every script has one responsibility, open for extension,
  closed for modification.
- **Config as single source of truth** — no magic strings outside
  `config.py`.
- **Graceful degradation** — unavailable models, missing assets,
  missing API keys — all silent no-ops with logged warnings. UI
  never crashes on the absence of an optional piece.
- **Gitignore personal data** — assume this repo will be public one day.
- **Western models on Brainy** — federal consulting context; provenance
  matters. Non-western models (Qwen, DeepSeek, Falcon) live on a
  separate rig.
- **Local first** — cloud models are benchmarks, not dependencies.
