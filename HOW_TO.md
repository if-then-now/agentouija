# How to use Agent Ouija

A beginner-friendly walkthrough of every tab in the app. If you're
looking for install instructions, see [SETUP.md](SETUP.md); this
doc is what to do once the app is running at
<http://localhost:7860>.

Companion piece intended for
[www.ifthennow.com](https://www.ifthennow.com/) — feel free to link
directly from the site.

## Accessing from mobile

Agent Ouija runs on your host machine's `localhost` by default,
but it's designed to work great from a phone or tablet too — the
UI is responsive, and everything works exactly the same. The
recommended path is [Tailscale](https://tailscale.com/) (free
personal plan):

1. Install Tailscale on the host + your mobile device, sign into
   both with the same account, enable **MagicDNS + HTTPS** in
   the Tailscale admin console.
2. On the host, run once:
   `tailscale serve --bg --https=7860 http://localhost:7860`
3. Bookmark the resulting URL on your phone
   (e.g. `https://brainy.tail<xxxx>.ts.net:7860`). Real TLS
   cert — no browser warnings, no port-forwarding, no exposure
   to the public internet.

Full walkthrough with screenshots in [SETUP.md](SETUP.md#4b-mobile-access-via-tailscale-recommended).

## Where to get the API keys

Every key Agent Ouija reads is optional — the installer walks you
through and any blank one is simply skipped. Sources at a glance:

| Key | Where to get it |
|---|---|
| **ANTHROPIC_API_KEY** (Claude) | <https://console.anthropic.com/settings/keys> |
| **OPENAI_API_KEY** (GPT-4o) | <https://platform.openai.com/api-keys> |
| **GEMINI_API_KEY** (Gemini + Nano Banana image editor) | <https://aistudio.google.com/apikey> |
| **GROQ_API_KEY** (Groq LPU) | <https://console.groq.com/keys> |
| **OPENROUTER_API_KEY** (Hermes 3 Cloud) | <https://openrouter.ai/settings/keys> |
| **SERPER_API_KEY** (Serper.dev web search) | <https://serper.dev/api-key> |
| **NEWSAPI_KEY** (daily news pulls) | <https://newsapi.org/register> |
| **SMTP_*** (📧 Send File) | Gmail App Password: <https://myaccount.google.com/apppasswords> — any SMTP provider works too |
| **Google Calendar** | <https://console.cloud.google.com/> → new project → enable Calendar API → create OAuth client (Desktop) → download JSON → save as `data/gcal_credentials.json` |

Add or update a key later without re-running the whole installer:

```powershell
python install.py --add-api-key
```

Details on each service — free-tier limits, gotchas — in the tab
sections below (Google Calendar setup is walked through
step-by-step under Tab 1, and SMTP is covered under Tab 3).

## Table of contents

1. [Tab 1 — Agent Ouija](#tab-1--agent-ouija)
2. [Tab 2 — Dev Workshop](#tab-2--dev-workshop)
3. [Tab 3 — Files](#tab-3--files)
4. [Tab 4 — Persona](#tab-4--persona)
5. [Tab 5 — Data](#tab-5--data)

---

## Tab 1 — Agent Ouija

The main workspace. Everything you'd normally do — ask, get an
answer, save the thread, edit a picture — happens here. Five
collapsible sections top to bottom.

### 💬 Question & Response — the place to chat

Type your question in the textbox. Buttons on the row underneath:

- **Ask For Revelations** — Fires the question at every selected
  model. Each model answers in parallel; the response window shows
  them one at a time (slideshow). Fluent, open-ended reasoning —
  the models can use their training data plus whatever RAG /
  document context you've referenced.
- **📖 Ask the Lore** — Same shape as Ask For Revelations, but the
  models are constrained to answer ONLY from what's in the RAG
  database (your ingested documents + any private corpus). If the
  answer isn't there, they respond *"Not in the lore."* Great for
  testing edge-AI recall and minimising token spend. Every Lore
  response leads its status line with a short note reminding you
  what mode you're in.
- **✋ Cancel** — Aborts an in-flight round. Pending model calls
  are dropped; already-in-flight ones finish but their results are
  discarded.
- **📂 Upload Files** — Drag files (documents or pictures) here to
  add them to the repository. Documents go to `docs/` and get
  auto-ingested into RAG. Pictures + videos go to `docs/media/`
  and stay parked until you reference them by name.
- **💾 Save Seance** — Save the current thread (question, answers,
  history, seance name) so you can reload it later from the
  Seances section. Auto-saves on every subsequent Ask once loaded.

Two small buttons appear under each response:

- **💾 Add This to RAG** — Ingest the current answer into the RAG
  database so future questions can retrieve it as context. Useful
  when a model gave a particularly good answer you want to
  preserve.
- **📧 Send File** — Ships any file referenced in your question
  (from `docs/` or `docs/media/`) as an email attachment via SMTP.
  Reference the exact filename in the prompt (e.g.
  *"send `spec.pdf`"*) and click. Requires SMTP config in `.env`
  — see the Files section below for setup.

**File repository interaction:** any file in `docs/` or
`docs/media/` can be referenced by name in your prompt. The app
detects the filename and either injects the doc contents as
context, attaches the media to a vision-capable model, or (for
image edits) routes to Gemini.

### 👑 Executive Boardroom — LLMs working together

Turns the boardroom into a single-answer flow. Instead of showing
you five separate answers, one selected model synthesizes the
group into one response.

**Fields inside the accordion:**

- **🌐 Always Use Boardroom Mode checkbox** — When ON, every Ask
  automatically triggers boardroom mode (peers run, then the
  executive synthesizes). When OFF, Ask returns individual peer
  responses; you trigger boardroom mode on demand via the tier
  buttons.
- **Sr. Execs. with selected Jr. Execs checkbox** — Only affects
  Senior tier: when checked, local models are included as peers
  alongside cloud models. When unchecked, Senior runs clouds only.
- **Three tier buttons** — Junior Execs, Senior Execs, The C-Suite.
  Clicking any of them runs a fresh peer set at that tier and
  synthesizes with an executive matching the tier:
  - **Junior Execs** — Local models only, one local synthesizes.
  - **Senior Execs** — Cloud models, one cloud synthesizes.
  - **The C-Suite** — Web-capable cloud models (Claude, GPT-4o,
    Gemini), one of them synthesizes.
- **👑 Executive Model dropdown** — Which specific model plays
  executive at the currently-selected tier. Filtered to tier-
  eligible models.
- **💾 Make this the Default Executive** — Remembers your pick
  across sessions.

**Groq note:** Groq's free-tier per-minute token cap (8,000 tokens/
minute) can't fit boardroom synthesis on more than a couple peer
responses. A warning banner appears under the dropdown whenever
Groq is picked as the executive. **Recommendation:** use Claude
Sonnet or GPT-4o for Senior / C-Suite tiers on the free tier.
Groq works fine as a peer, and shines on shorter prompts — it's
just the exec synthesis that blows the cap. Groq's Dev Tier lifts
the cap to 25k+ TPM, at which point you can bump
`max_input_tokens` in `config.py` and use it for exec too.

### 🤖 The Assistant — Calendar and News reporter

Two external-API surfaces that plug into the same RAG the models
use to answer questions.

#### Google Calendar setup (step-by-step)

1. **Enable the Calendar API in Google Cloud.**
   - Go to <https://console.cloud.google.com/>. Log in with the
     Google account whose calendar you want Agent Ouija to read.
   - Create a new project (top-left dropdown → "New Project"). Name
     it something like "Agent Ouija".
   - Once you're in the project: open the hamburger menu → APIs &
     Services → Library. Search for "Google Calendar API" and
     click Enable.

2. **Configure the OAuth consent screen.**
   - Hamburger → APIs & Services → OAuth consent screen.
   - Pick **External** unless your Google account is on a Google
     Workspace. Fill in the required fields (app name, support
     email, developer email). Save.
   - Add your Google account as a **Test user** on the same page.

3. **Create OAuth credentials.**
   - APIs & Services → Credentials → "Create Credentials" → OAuth
     client ID.
   - Application type: **Desktop app**. Name it anything.
   - Click **Download JSON** on the created client. Save the file
     as `gcal_credentials.json` inside your Agent Ouija
     installation's `data/` folder (i.e., wherever `scripts/` is,
     go up one and into `data/`).

4. **Authorize inside Agent Ouija.**
   - In the Agent Ouija tab, open **🤖 The Assistant**. You'll see
     "🕯 Calendar: **not authorised**" and a **🔐 Authorize Google
     Calendar** button.
   - Click Authorize. A browser window opens; consent as your
     Google account. The token is cached in `data/gcal_token.json`
     for future sessions.
   - Weekly re-auth expected while the OAuth app is in Testing
     status — Google enforces a 7-day refresh-token expiry there.
     Just click Authorize again when it lapses.

5. **Use it.**
   - **🔄 Refresh Calendar into RAG** — Pulls your upcoming events
     and adds them to ChromaDB. Any question after that ("What's
     on my calendar this week?") retrieves the events through RAG.
   - **📋 Show upcoming** — Prints upcoming events inline without
     touching RAG.
   - **➕ Add Event** — Fill in the form (title / date / times),
     click, event lands in your Google Calendar.
   - **🗑 Delete Event** — Pick from a dropdown of upcoming events,
     type YES to confirm.

#### NewsAPI setup (step-by-step)

1. **Get a NewsAPI key.**
   - Sign up at <https://newsapi.org/register>. Free tier: 100
     requests/day.
   - Copy the API key from your dashboard.

2. **Add the key to your Agent Ouija install.**
   - Run `python install.py --add-api-key` OR edit `.env` directly
     (at `%USERPROFILE%\.config\agentouija\.env` after the security
     hardening commit — see SETUP.md for the exact location).
   - Set `NEWSAPI_KEY=<paste-your-key-here>` and save.

3. **Use it.**
   - Under 🤖 The Assistant, open the **📰 News Pulls** sub-accordion.
   - Four buttons: **⚾ Sports (baseball)**, **🌍 World**,
     **🇺🇸 National**, **📰 Pull All (3 credits)**.
   - Click any category. The headlines get cached to
     `docs/news/YYYY-MM-DD_<category>.md` and auto-ingested into
     RAG. Every credit = one API request against your daily 100
     free-tier allowance.
   - Any question like "What's happening in baseball today?" now
     retrieves cached headlines as RAG context.

### 🎯 Scoring Controls — Edge AI training

Score each response 0-4 (baseball scale) so Agent Ouija can build
a training dataset from your best answers:

- **0 — whiff** (excluded from training export)
- **1 — contact** (weight 0.25)
- **2 — base hit** (weight 0.50)
- **3 — extra bases** (weight 0.75)
- **4 — home run** (weight 1.0 — "couldn't tell it wasn't Claude")
- **Valid Refusal** — the model declined for its own safety policy
  (excluded from quality stats).

**Buttons:**
- **Score And Next** — Log the score, move to the next model's
  response in the round.
- **Score + Add To RAG** — Same, plus curate this response into
  ChromaDB so future questions can retrieve it.
- **Add To RAG (no score)** — Curate without scoring.
- **Retry Scoring** — Re-review the current round from the start.
- **Prev / Next / Skip** — Slideshow navigation.
- **Blind mode** — Hide the model name while scoring, so you
  score on quality alone (no anchoring to which vendor produced
  the answer).

**What "training" means here:**

- **RAG (Retrieval-Augmented Generation)** — Every high-scoring
  Q+A pair you Add To RAG becomes retrievable context for future
  questions. This is a runtime improvement — no training run
  required. The local models get sharper on your specific
  domain just by accumulating good examples.
- **LoRA (Low-Rank Adaptation) fine-tuning** — When you're ready
  for a bigger jump, `scripts/prepare_training.py` exports every
  scored row (score ≥ 1) as a JSONL file suitable for Unsloth /
  Axolotl / other LoRA trainers. Feed the JSONL into a fine-tune
  and you get a local model that speaks in your domain.

**The Reading** — a composite metric: `(score / SCORE_MAX) /
log(response_time + e)`. Rewards high scores AND fast responses.
Full breakdown per model on the Data tab.

**Training sessions:** click *Save Training Checkpoint* on the
Data tab after each LoRA run. Subsequent rows are attributed to
that checkpoint, so the Δ Reading column shows whether your fine-
tune actually improved things.

### 🤖 The Models — the workers

Ten checkboxes total — five locals on the left, five clouds on
the right. Check any combination you want in a round.

**Master toggles** at the top of each column flip every model on
or off at once, respecting tier eligibility (locals grey out at
High Energy, clouds grey out at Low).

**Upgrading a model:**

- **Local models:** In the terminal on Brainy: `ollama pull
  llama3.3` (or any model tag from <https://ollama.com/library>).
  Then in `config.py`, update the matching entry's `"name"` field
  to the new tag. Restart Agent Ouija. Delete the old model with
  `ollama rm llama3.2` when you're comfortable.
- **Cloud models:** Update the model id in that model's config
  entry. E.g. `"anthropic/claude-sonnet-4-6"` →
  `"anthropic/claude-sonnet-4-7"` when a new Sonnet ships. No pull
  needed — restart the app and it picks up.

**Any combination is valid.** One local + two clouds. All ten. Zero
clouds. Whatever you check will run. The 🎯 Media rule is the one
exception: prompts referencing a picture / video only run when
exactly one model is checked (cost-control).

### ⚙ Control Center — settings

- **🔮 Mysticism Level dropdown** — one of four modes:
  - **🟡🔥 Low Energy — Locals Only** — free but slower; pure
    local inference, no cloud, no web.
  - **🟠🔥 Medium Energy — Agile** — cloud models pre-checked,
    locals unchecked. Serper web search injected into every
    checked model's prompt. Best default for most questions.
  - **🔴🔥 Medium Energy — Synergistic** — clouds AND locals
    pre-checked. Slower (more calls) but you get every angle.
  - **🔵🔥 High Energy — Native Web Search** — only web-capable
    clouds (Claude / GPT-4o / Gemini) run, and they use their
    provider's native web search instead of Serper-fed context.
- **Navigation buttons** — jump to the four hidden tabs:
  - **📊 Data** — leaderboards, billing, backups, rollback, disk
    usage, wipe controls.
  - **📁 Files** — file repository, image editor, per-file Q+A.
  - **🛠 Dev Workshop** — Python notebook + PowerShell CLI.
  - **🎭 Persona** — persona files, voice picker, RAG toggles.
- **🔊 Voice checkbox** — Piper TTS reads the current answer aloud
  when checked. To change WHICH voice, go to the Persona tab's
  Voice section, upload / pick a `.onnx` file, click *Set As
  Default*.

### 🕯 Seances — saved threads

A "seance" is a saved conversation with all the state it takes to
resume — question history, model responses, chosen executive
tier, thread accumulation.

- **📋 Load** dropdown — every saved seance in `data/seances/`.
  Pick and load; question box comes up blank so you can type a
  follow-up, response window shows the full prior thread.
- **📝 Rename** — pick a seance, type a new name, click Rename.
  Auto-updates the label in the dropdown.
- **🗑 Delete** — remove a seance file. Confirms first.
- **💾 Save Seance** button (in the main Ask row) — saves the
  current thread. Existing seance? Overwrites in place. New?
  Auto-slugifies from the question if you haven't given it a
  name.
- **＋ New Seance** button (top of Agent Ouija tab) — clears
  everything for a fresh start.

Every seance is a JSON file in `data/seances/`, portable and
gitignored.

---

## Tab 2 — Dev Workshop

Two accordions. Both are shortcuts for programmatic work; you can
close either and just use the other.

### 📓 Notebook — run Python code

Cell-based Python runner against Agent Ouija's model set. Split
your code with `# %%` on its own line — each block becomes a cell.

Per-model checkboxes above the notebook drive iteration:
- **One model checked** → code runs once; state persists across
  Run clicks.
- **Multiple checked** → code runs once per model on an isolated
  copy of state, results show up in the slideshow so you can
  compare answers.

Variables injected into your code's scope on every run:
- `MODEL` — LiteLLM-format id (e.g. `"openai/gpt-4o"`,
  `"ollama_chat/llama3.2"`)
- `MODEL_KEY` — config key (`"gpt-4o"`)
- `SELECTED_MODELS` — list of all checked keys

Helpers in scope: `ask(prompt, model, system, temperature,
max_tokens) → str`, `ask_all(prompt, …) → dict`,
`get_rag(question) → str | None`.

API keys come from `.env` — no `userdata.get(...)` setup needed.

### 🖥️ CLI — PowerShell

Runs PowerShell commands with a persistent working directory
tracked in the UI. Built-in `cd` handles wildcards. Output capped
at 20 KB per command, 60-second timeout.

**🚀 Launch Claude Code** button spawns a detached PowerShell
window running `claude` from the current cwd — ideal for handing
off a piece of work to Claude Code interactively (the interactive
TTY can't render inside a Gradio textbox, hence the detached
window).

**Great use case:** the CLI + Notebook combo is the fastest way to
test snippets of code the LLMs suggest before pasting them into a
real file. Ask a model *"write me a quick script to X"*, paste
the answer into the Notebook, click Run, iterate.

---

## Tab 3 — Files

The **file repository** and everything you can do with the files
in it.

### 📁 File Manager

Two columns: **Documents** (`docs/`) and **Media** (`docs/media/`).

- **Documents** — anything ingestible (`.md`, `.txt`, `.pdf`,
  `.docx`, `.html`, `.json`, `.csv`, `.rtf`). Uploaded documents
  auto-trigger RAG re-ingestion so ChromaDB stays in sync.
- **Media** — images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`,
  `.gif`) and videos. Not auto-ingested — media stays parked
  until you reference it by name in a prompt.

Both columns are pinned to a 180-pixel fixed height so the layout
stays symmetric regardless of contents. Each supports:

- **📂 Upload** — multi-file upload; each file routes to the
  correct column by extension.
- **📝 Rename** — pick a file from the dropdown, type a new name,
  click Rename.
- **🗑 Delete** — pick + delete. Documents get an auto re-ingest;
  media doesn't (nothing to re-ingest).

**Photo editing (Gemini only, one-model rule):** Only Gemini's
Nano Banana model can actually edit images. When you reference
an image + an edit verb in your prompt, the request routes to
Gemini regardless of what other models you have checked —
your selection is bypassed for that call. This bypass exists
because per-image token cost is 10-1000× a text token, so the
app enforces a one-model rule to keep spend predictable.

**Photo analysis (any vision-capable model):** Simply referencing
an image without an edit verb ("describe cat.jpg") lands as a
normal vision call to whatever model you have checked. Cloud
vision models (Claude, GPT-4o, Gemini) all support this. The
Files-tab uploader shows a **per-model cost preview** (Low /
Medium / High / Very High) before you attach the image so you
know what a run would cost.

### 💬 Work With File(s) — Q+A on your repository

Section at the bottom of the tab. Uses the tab's own model panel
(same shape as the main Agent Ouija tab).

**Edit-intent keywords that trigger image editing:**

`edit`, `add `, `remove`, `delete`, `change`, `replace`, `put `,
`insert`, `place `, `make it`, `make them`, `make the`, `erase`,
`fix`, `combine`, `merge`, `overlay`, `cover`, `draw`, `paint`,
`highlight`, `circle `, `meme`, `caption`, `text at`, `text on`,
`text to`, `text overlay`.

**Example edit prompts:**
- *"add a heart between the two people, wedding.png"*
- *"put meme text 'not today' at the bottom of cat.jpg"*
- *"remove the traffic cones from street.png"*
- *"replace the sky with a sunset in landscape.jpg"*

**Example non-edit prompts (fall through to normal vision analysis):**
- *"describe what's in `screenshot.png`"*
- *"summarize the key data in `report.pdf`"*
- *"compare the wording in `contract-v1.pdf` and `contract-v2.pdf`"*

### 🎨 Edited output preview

After a successful image edit, a preview image appears in the
"Edited output preview" panel below the response. Right-click the
image to save it locally. The edited file also lives on disk at
`docs/media/<original-stem>_edited_<timestamp>.<ext>`.

**Image editing is also available in the Agent Ouija tab.** Same
detection rules — reference an image filename + edit verb in Ask
For Revelations and it routes to Gemini. The preview panel only
lives in the Files tab; in Agent Ouija you'll see the filename
in the response and can retrieve it via 📧 Send File.

### 📧 Send File — SMTP setup

Click 📧 Send File after referencing a file in your prompt to
email it as an attachment via SMTP.

**SMTP setup (Gmail example):**

1. **Generate a Gmail App Password.** At
   <https://myaccount.google.com/apppasswords>, sign in, select
   "Mail" as the app and your device. Google returns a 16-
   character password with no spaces. Copy it.

2. **Add SMTP config to `.env`.** Run `python install.py
   --add-api-key` or edit `.env` directly:
   ```
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your.address@gmail.com
   SMTP_PASSWORD=<the-16-char-app-password>
   EMAIL_FROM=your.address@gmail.com
   EMAIL_TO=your.address@gmail.com    # or a different destination
   ```

3. **Test it.** Reference a file in a prompt (e.g. type
   `spec.pdf`), click 📧 Send File. Status area confirms
   delivery.

**Other providers (Outlook / Fastmail / Proton bridge):** same
shape, different `SMTP_HOST` and `SMTP_PORT`. Outlook: `smtp-
mail.outlook.com` port 587. Proton bridge: `127.0.0.1` port 1025.
Port 465 handled automatically (implicit TLS).

**Multi-file:** if your prompt references three files, one email
lands with three attachments. Cleaner than three emails.

---

## Tab 4 — Persona — JARVIS for Agent Ouija

Turn your Agent Ouija into any voice / character you want. Three
levels of customisation, from minimal to full immersion.

### The three persona layers

Every model call starts with a **system prompt**. The Persona tab
lets you build that system prompt out of files:

1. **`system_core.md` (always)** — the base personality. Ships
   with a mystical Ouija voice ("the pointer moves, you speak,
   the pointer stops"). You can replace this with your own
   `system_core.md` — install.py preserves any existing file.

2. **`enhanced_persona_main.md` (optional)** — additional voice
   rules that layer on top of `system_core.md`. Uploaded via the
   **Enhanced Persona Main** row on the Persona tab. New upload
   replaces the existing file.

3. **`persona_<name>.md` + episodes (optional, enhanced-only)** —
   named character overlays. Uploaded via the **Enhanced Persona
   Files** section. `persona_<name>.md` files route to `prompts/`;
   anything else routes to `prompts/episodes/`. Multiple personas
   can layer together; the model picks the register from question
   context.

### Copyright-safe corpus

The **Copyright Files** panel is for reference material
(transcripts, quote collections, source books) that would be
copyright-risky to commit. Files go to `Copyright/episodes/`,
which is **gitignored by default** — nothing you drop here ever
lands in a public git commit.

Clicking **Re-ingest Copyright corpus** chunks the files into a
SEPARATE ChromaDB collection (`mr_ouija_private`), distinct from
the main RAG. Enable the **Use private corpus in retrieval**
toggle and any question you ask retrieves relevant chunks from
this collection as extra context.

The gitignore safeguards:
- `Copyright/` folder — gitignored entirely
- `prompts/` folder — gitignored entirely (personal personas)
- `data/chroma/` — gitignored (contains the private collection)
- `data/private_corpus_*.json` — gitignored (portable exports)

You can safely share the repo publicly without any of your
persona overlays or copyrighted reference material coming with it.

### Example persona setup

You could have Agent Ouija answer in a blended voice of, say,
**a comedian known for observational humor**, **a popular cutthroat
business executive**, and **the most famous wizard ever** —
combining their signature phrasings, timing, and worldview. The
enhanced stack loads all three `persona_*.md` files together;
the model picks whichever register fits your question. A
"how do I run a meeting?" question tilts toward the executive
voice; "explain what's weird about airline peanuts" leans
comedian; a life-advice question drifts wizard.

You'd:
1. Drop three `.md` files into `prompts/` — `persona_comedian.md`,
   `persona_executive.md`, `persona_wizard.md`. Each has voice
   rules, catchphrases, worldview notes.
2. Drop episode transcripts / quote files into
   `Copyright/episodes/` (any filenames — `comedian_ep_1.md`,
   `executive_quotes.md`, etc.). Those get chunked into the
   private ChromaDB collection when you click Re-ingest.
3. Turn on **Enable enhanced persona for all models** — every
   cloud model now gets the enhanced stack. (Locals stay on
   `system_core.md` only — quantised models drift under long
   persona stacks.)
4. Turn on **Use private corpus in retrieval** — the models get
   relevant transcript chunks alongside the persona stack.

### Voice section

Piper TTS voices as `.onnx` + `.onnx.json` files in `voices/`.

- **Voice dropdown** — pick from installed voices.
- **Speaker ID** — for multi-speaker voices (like `libritts`).
- **Upload** — drop a new voice pair; refresh the dropdown.
- **Set As Default** — writes `voices/active_voice.txt` so the
  next launch loads this voice.

Voices download from <https://huggingface.co/rhasspy/piper-voices>.

### Toggles

- **Enable enhanced persona for all models** — flips every cloud
  model onto the enhanced stack. Groq is auto-excluded because its
  free-tier TPM cap can't fit the persona overlay (see the
  Persona tab's inline note).
- **Use private corpus in retrieval** — enables the copyright
  ChromaDB collection.
- **Preview assembled prompt** — see exactly what system prompt
  every model would receive with the current toggles.

---

## Tab 5 — Data — accounting and training stats

Everything numeric about your Agent Ouija usage lives here.

### 💰 Billing Details

Auto-loads on tab open. Four views of the same underlying
`arena_results` cost data:

- **Lifetime total (headline)** — total dollars spent, total rows,
  total tokens, first-row / latest-row timestamps.
- **Window comparison table** — Last 7 days / Last 30 days /
  Since last checkpoint / Lifetime, in one glance.
- **Per-model breakdown** — every model, its row count, tokens,
  cost.
- **Per-provider roll-up** — Anthropic / OpenAI / Google / Groq /
  OpenRouter / Local, in one line each. Answers "how much did I
  spend at each provider total" instantly.

**🔄 Refresh Billing** re-runs the query on demand.

**Provider dashboards** — links open each cloud provider's usage
page in a new tab so you can cross-check.

### 📊 Historical + Current Session tables

Sortable leaderboards with the following columns:

- **Model** — display name.
- **Avg Score** — average score across all scored responses.
- **Avg Arena** — avg score on Arena-tab / Agent Ouija tab
  responses.
- **Avg Notebook** — avg score on Dev Workshop / Notebook
  responses.
- **Avg Time** — average response time in seconds.
- **The Reading** — the composite metric: `(score / SCORE_MAX) /
  log(time + e)`. Higher is better.
- **Cost Reading** — Reading adjusted for USD cost. Locals collapse
  to Reading (cost = 0); clouds pay a smooth log penalty.
- **Δ Reading** — change since the last training checkpoint. Green
  = improved after fine-tuning.
- **Score buckets** — count of home runs (4), extra bases (3),
  hits (2), contacts (1), whiffs (0).
- **Cost (USD)** — total spend on this model, across all rows.
- **RAG Adds** — count of Add-To-RAG events attributed to this
  model.

Click any column header to sort. Historical = every row ever;
Current Session = rows since the last checkpoint.

### 🎯 Training Checkpoints

Log a checkpoint after each LoRA fine-tune so the Δ Reading
column has something to compare against. Columns: Date, Name,
Model, Notes, Session Cost (USD).

**Add Checkpoint** button — pops up a form for name + notes,
records the current arena_results timestamp as the checkpoint
boundary.

### 🗑 Danger Zone — three-way wipe

- **Wipe Database** — reveals three confirmation buttons.
  - **Yes, backup first** — snapshots arena_scores.db,
    chat_history.json, chroma/, seances/, news/ to
    `data/backups/wipe_<timestamp>/` before wiping.
  - **Yes, no backup** — wipes without a backup.
  - **No** — cancels.

### ↩ Reset UI Preferences

Wipes `data/preferences.json` — flips Mysticism Level back to
Low Energy, both Executive Boardroom checkboxes off. Non-
destructive to anything else.

### ⏰ Time-based Rollback

Purge rows / files older than a chosen window. Amount +
unit (days/weeks/months) + target (ChromaDB, arena_scores.db,
Seances, News docs, All of the above).

- **🔍 Preview** — dry counts, no writes.
- **🗑 Purge (backup first)** — requires typed YES. Always
  snapshots to `data/backups/rollback_<timestamp>/` before
  deleting.

### ♻ Restore From Backup

- Dropdown lists every snapshot in `data/backups/` (newest first).
- **🔍 Preview** — shows the contents of the selected snapshot.
- **♻ Restore** — requires typed YES. Overwrites live copies.
  You have to restart the app afterward so DB handles + retriever
  caches pick up the restored files.

### 💾 Disk Usage

Shows per-folder disk usage for: docs, media, prompts, Copyright,
ChromaDB, voices, seances, training exports, summaries, backups,
arena_scores.db, chat_history.json.

Click **🔄 Refresh Sizes** to rescan.

---

## Getting help

- Command-line help: `python install.py --help`,
  `python uninstall.py --help`.
- Bug reports: <https://github.com/if-then-now/agentouija/issues>.
- Feedback / feature requests: same repo, discussions tab.
