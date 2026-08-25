# Agent Ouija — Setup

Quick install path for a fresh rig. The interactive installer handles every
step listed here; this doc is the human-readable companion for what's
happening under the hood. For a deep dive on every feature once it's
running, see [USER_GUIDE.md](USER_GUIDE.md).

---

## 1. Prerequisites

| Component | Minimum | Why |
|-----------|---------|-----|
| Windows 10/11 (or Linux/macOS) | — | Tested target |
| Python 3.10 - 3.13 | 3.11 recommended | Piper TTS may lag on 3.14+ |
| [Ollama](https://ollama.com/download) | Latest | Local model runtime |
| ~10 GB free disk | 30 GB recommended | Models + embeddings + ChromaDB |
| GPU recommended | RTX 3060+ | Local model inference speed |

Verify Ollama is on `PATH`:

```powershell
ollama --version
```

Verify Python:

```powershell
python --version
```

---

## 2. Clone the repo

```powershell
git clone https://github.com/if-then-now/agentouija.git C:\AgentOuija
cd C:\AgentOuija
git checkout dev
```

(`dev` is the active branch — `main` lags behind.)

**You can clone anywhere.** `C:\AgentOuija` is a suggestion, not a
requirement — the app resolves its data folders relative to
wherever `scripts/config.py` lives. `C:\Users\<you>\Downloads\
agentouija-main\` from a Zip download works just as well.

---

## 3. Run the installer

**Easy path — double-click the batch file:**

- **`Install Agent Ouija.bat`** — opens a terminal window and walks
  you through the interactive prompts. No typing required.
- **`Uninstall Agent Ouija.bat`** — matching double-click launcher
  for the uninstaller; asks for a level and requires typed **YES**
  before deleting anything.

Both .bat files sit at the repo root next to this SETUP.md. Both
prefer the Windows `py` launcher if it's installed; fall back to
`python` on PATH; print a helpful message if neither is available.

**Terminal path — same thing under the hood:**

```powershell
python install.py
```

At the top the installer asks you to pick a mode:

**[1] Full install** — one-shot: pulls every local model in
`config.AVAILABLE_MODELS`, prompts for every API key (leave blank to
skip any you don't have yet), downloads the default Piper voice,
runs `pip install`, creates the launcher .bat + desktop shortcut.
No per-component Y/N — one pass and done.

**[2] Custom install** — Y/N per phase and per component: skip
individual models, decline the voice download, refuse pip if you're
behind a proxy, etc.

Either mode walks the same 8 phases:

1. **Prerequisite checks** — Python version, Ollama on PATH.
2. **Folder structure** — creates `data/`, `voices/`, `prompts/`,
   `docs/`, `Copyright/episodes/`, `assets/`, `data/chroma/`,
   `data/training/`, `data/summaries/`, `data/backups/`.
3. **`.env` setup** — copies `.env.example` and prompts you for
   each API key. Blank keeps existing / skips. The file lives at
   **`%USERPROFILE%\.config\agentouija\.env`** (out of the repo
   tree so a repo zip / clean / accidental delete can't leak or
   trash your keys); ACL locked down to the current user +
   SYSTEM only via `icacls`. Legacy in-repo `.env` auto-migrated.

   **Where to get each key (all optional — skip any you don't
   want and add later via `python install.py --add-api-key`):**

   | Key | Where to get it | Notes |
   |---|---|---|
   | `ANTHROPIC_API_KEY` | <https://console.anthropic.com/settings/keys> | Claude Sonnet — recommended for the boardroom executive. |
   | `OPENAI_API_KEY` | <https://platform.openai.com/api-keys> | GPT-4o + `gpt-4o-search-preview` for The Big 3 web search. |
   | `GEMINI_API_KEY` | <https://aistudio.google.com/apikey> | Gemini 2.5 Flash + Nano Banana image editor. Free tier available. |
   | `GROQ_API_KEY` | <https://console.groq.com/keys> | Free tier caps at 8k TPM — Groq stays a peer, not exec. |
   | `OPENROUTER_API_KEY` | <https://openrouter.ai/settings/keys> | Hermes 3 70B via OpenRouter's Nous route. |
   | `SERPER_API_KEY` | <https://serper.dev/api-key> | Google Search API for Medium Energy web-grounded flows. Free 2,500/mo. |
   | `NEWSAPI_KEY` | <https://newsapi.org/register> | Daily news pulls into RAG. Free 100 requests/day. |
   | `SMTP_HOST` / `USER` / `PASSWORD` + `EMAIL_FROM` / `EMAIL_TO` | <https://myaccount.google.com/apppasswords> (Gmail App Password) | Powers 📧 Send File. Any SMTP provider works; Gmail needs an App Password not your login password. |
   | `GCAL_CREDENTIALS_FILE` (Google Calendar) | <https://console.cloud.google.com/> → new project → Enable Calendar API → OAuth Client ID (Desktop) → download JSON → save as `data/gcal_credentials.json`. Then click 🔐 Authorize Google Calendar inside the app. |

   None of these are stored in the repo — every key lives only
   in the gitignored `.env` at the user-config path above.
4. **Local model selection** — Full: pulls all. Custom: Y/N per model
   in `config.AVAILABLE_MODELS` with `type: "local"`. Downloads are
   2-8 GB each.
5. **Python dependencies** — `pip install -r requirements.txt`.
6. **Piper voice** — Full: auto-download. Custom: --voice flag or Y/N.
7. **Launcher** — `Agent Ouija.bat` + optional desktop shortcut.
8. **Git key-guard hook** — installs a `.git/hooks/pre-commit`
   that blocks any commit whose staged diff contains a non-blank
   `API_KEY` / `PASSWORD` / `TOKEN` assignment or an
   Anthropic/OpenAI-style `sk-…` token. Bypassable with
   `git commit --no-verify` if you ever need to.
9. **Smoke test** — imports the heavy stack so wheel/version problems
   show up here instead of on first browser hit.

### Adding an API key later

If you skipped a key in step 3 and want to add it later (Serper,
NewsAPI, a cloud provider you signed up for after the initial
install), re-open just the .env prompts without redoing any other
phase:

```powershell
python install.py --add-api-key
```

### Encrypting `.env` with a passphrase (laptop installs)

The installer offers this immediately after step 3 as an optional
extra layer — recommended for **laptop installs** where the disk
might not be encrypted at rest (BitLocker off), and skipped for
stationary desktop rigs. Uses PBKDF2-HMAC-SHA256 (200k iterations)
to derive a Fernet key from your passphrase; the key never
touches disk. Ciphertext + meta land next to the .env location as
`.env.enc` + `.env.enc.meta`; the plaintext file is deleted.

`arena_ui.py` prompts for the passphrase on every launch (hidden
input via `getpass`). For headless runs (scheduled tasks, systemd,
cron) set `AGENT_OUIJA_ENV_PASSPHRASE` in the environment before
starting the process instead.

Toggle after install:

```powershell
python install.py --encrypt-env   # plaintext → encrypted (or rotate passphrase)
python install.py --decrypt-env   # encrypted → plaintext
```

**A lost passphrase can't be recovered.** Store it in a password
manager. Every affected API key would have to be regenerated at
its provider if the passphrase is lost.

### Uninstall

The uninstaller mirrors the installer's mode idea — three levels:

```powershell
python uninstall.py                # interactive — pick a level
python uninstall.py --level=data   # data-only: wipe generated files, keep .env
python uninstall.py --level=keys   # data + .env
python uninstall.py --level=full   # data + .env + `ollama rm` on every local model
```

All destructive; a typed `YES` confirms unless you pass `--yes`.
The repo folder itself is NOT auto-deleted (PowerShell can't delete
the folder its script is running from) — a final line prints the
manual command.

### Installer flags

```powershell
python install.py --full         # skip mode prompt, go Full
python install.py --custom       # skip mode prompt, go Custom
python install.py --add-api-key  # only re-open .env prompts
python install.py --yes          # accept defaults at every prompt (CI)
python install.py --no-pip       # Custom mode: skip pip
python install.py --no-models    # Custom mode: skip ollama pulls
python install.py --no-env       # Custom mode: skip .env editing
python install.py --voice        # Custom mode: also grab Piper voice
```

Idempotent — re-run any time. Existing files are preserved.

---

## 4. Start the web UI

Two ways — pick whichever:

**A) Double-click the launcher** (created in step 3, Phase 7 of the
installer):

- `Agent Ouija.bat` in the repo root, OR
- The desktop shortcut `Agent Ouija.lnk` if you accepted the prompt.

**B) Run the script directly:**

```powershell
python scripts\arena_ui.py
```

Either way, the default browser opens automatically to
<http://localhost:7860> and pulls to the foreground. The console window
that launched the server has to stay open — closing it stops Agent Ouija.

**Light / dark mode:** click the **🌓 Toggle Theme** button in the
header. The choice persists in the URL as `?__theme=light` or
`?__theme=dark`, so bookmark the URL you prefer if you want a specific
mode every time you launch.

---

## 4b. Mobile access via Tailscale (recommended)

Agent Ouija runs on your local network by default at
<http://localhost:7860>. To use it from your phone / tablet /
another laptop with almost no friction, wire it through
[Tailscale](https://tailscale.com/) — a free personal-tier
VPN mesh that gives every device a stable HTTPS URL.

**One-time setup:**

1. **Install Tailscale** on the host running Agent Ouija AND on
   the mobile device. Sign into both with the same account.
   Free personal plan is plenty. <https://tailscale.com/download>
2. **Enable MagicDNS + HTTPS** in your Tailscale admin console
   (Settings → DNS → toggle both). One-time.
3. **Serve the app over HTTPS.** On the host:
   ```powershell
   tailscale serve --bg --https=7860 http://localhost:7860
   ```
   `tailscale serve status` should now show a URL like
   `https://brainy.tail<xxxx>.ts.net:7860`.
4. **Bookmark that URL on your phone.** Real cert (Tailscale-
   issued via Let's Encrypt), so no browser warnings, no
   `-Not Secure-` label. Full Agent Ouija UI on mobile — Ask
   For Revelations, Ask the Lore, image edits, Send File, all
   of it.

**Why Tailscale specifically:**

- Zero port-forwarding on your router. Nothing exposed to the
  public internet.
- Real HTTPS cert without you managing anything. Browsers treat
  it as a secure origin — that's what unlocks `navigator.
  mediaDevices` for microphone input, for example.
- Access works the same at home / coffee shop / hotel / on
  cellular. The tailnet routes over whatever underlying
  connection is available.
- Free for personal use (up to 3 users / 100 devices).

**Alternative — no Tailscale:** you can bookmark
`http://<hostname>:7860` from another device on your local WiFi
(same LAN), but that's HTTP-only and only works while you're
on the same network. For any "away from home" use, Tailscale is
the shortest path.

---

## 5. First-run smoke check

In the **Arena** tab:

1. Type a question in the textbox.
2. Make sure at least one model checkbox is ticked (the installer
   already pulled them but Ollama needs to be running — start the
   Ollama service if you haven't).
3. Click **Ask For Revelations**.
4. Watch the orange progress bar narrate per-model completion.

If you get an "Ollama unavailable" warning, the Ollama service isn't
running. Start it from the Windows tray or:

```powershell
ollama serve
```

If a cloud model fails with an "env not set" error, your `.env` is missing
its API key — re-run `python install.py` and fill it in this time, or
edit `.env` directly.

---

## 6. (Optional) Ingest your own knowledge base

Drop documents into `docs/` (any text/markdown/PDF), then click
**Ingest Documents** on the Arena tab. The pipeline chunks them, embeds
via BAAI/bge-small-en-v1.5, and stores in your local ChromaDB. Cloud
models with `"rag": True` (the default for the five clouds) will retrieve
relevant chunks on every Ask afterward.

For copyrighted material that shouldn't reach the repo, use
`Copyright/episodes/` + the private corpus workflow instead:

```powershell
python scripts\private_corpus.py --ingest
python scripts\private_corpus.py --status
```

See the **Private Corpus** section of [USER_GUIDE.md](USER_GUIDE.md#private-corpus)
for the full migration story.

---

## 7. What's next

You're up. The web UI is now a single-tab layout with four hidden
tabs reachable from Agent Ouija's Control Center:

- **Agent Ouija** — main flow: Ask For Revelations, Ask the Lore
  (RAG-grounded), Executive Boardroom (Junior / Senior / C-Suite
  synthesis tiers), Mysticism Level dropdown that toggles Low /
  Medium / High energy modes, seance save/load.
- **Dev Workshop** — collapsible **📓 Notebook** (course-style
  Python cells against arena models) and **🖥️ CLI** (PowerShell
  runner + Launch Claude Code button).
- **Files** — document + media repository with instruction-based
  image editing (reference an image file + edit verb → Gemini
  Nano Banana runs the edit and saves to `docs/media/`).
- **Persona** — 4 sections: enhanced main + files + Copyright
  corpus + voice picker.
- **Data** — historical cost breakdown (lifetime, 7d / 30d,
  per-provider), leaderboards, training checkpoints, backup
  restore, time-based rollback, NewsAPI pulls, Google Calendar
  authorization, and the three-way full wipe.

Walk through each in [USER_GUIDE.md](USER_GUIDE.md).
