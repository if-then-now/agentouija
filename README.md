# Agent Ouija

**Local edge AI assistant and multi-model evaluation arena.** Runs ten
models side by side — five local (via Ollama) and five cloud (Anthropic,
OpenAI, Google, Groq, OpenRouter) — behind one Gradio web UI on your
own machine.

Built as a working proof of concept for **air-gap-capable, sovereign
AI**: no vendor lock-in for inference, no data egress for anything you
don't explicitly send to a cloud provider, no black boxes between you
and your model outputs.

## What it does

- **Ask any question across ten models** and see who answers well
  under blind scoring. Local vs cloud comparison, cost tracked per
  call, response time tracked per call.
- **Boardroom mode** — one selected executive model synthesizes the
  other peers' answers into a single response. Junior (locals),
  Senior (clouds), C-Suite (web-capable clouds) tiers.
- **Grounded RAG** — every answer can be constrained to your own
  document corpus. Drop docs in `docs/`, they get chunked into
  ChromaDB, cited on demand.
- **Image editing** — reference an image in your prompt with an edit
  verb (*"add a heart between the two people, dog.png"*) and Google's
  Gemini Nano Banana runs the edit locally in your session.
- **Persona overlay** — layer a custom voice / character on top of
  every response. Ships with a mystical default; drop your own
  `.md` files to swap.
- **Google Calendar + NewsAPI integration** — read/write your calendar,
  cache daily headlines into RAG.
- **Voice output** — Piper TTS reads answers aloud (fully local).
- **Score + train** — 0-4 baseball scoring, per-model leaderboards,
  training-data export as JSONL for LoRA fine-tuning.
- **Works great on mobile** — reach your rig from a phone via
  [Tailscale](https://tailscale.com/) (free personal plan). One
  `tailscale serve` command on the host gives you a proper HTTPS
  URL like `https://brainy.tail<xxxx>.ts.net:7860` — real cert,
  no browser warnings, no port-forwarding. See SETUP.md for the
  30-second setup.

Stack: Ollama, LiteLLM, LlamaIndex, ChromaDB, Gradio, SQLite, Piper
TTS, cryptography.

## Getting started

**Requirements:**
- Windows 10/11 (Linux/macOS should work but tested primarily on Win)
- Python 3.11 – 3.13
- [Ollama](https://ollama.com/download) installed and on PATH
- ~30 GB free disk if you plan to pull all five local models

**One-time install:**

Clone the repo (any path works — `C:\AgentOuija\` is a suggestion,
not a requirement):

```powershell
git clone https://github.com/if-then-now/agentouija.git C:\AgentOuija
cd C:\AgentOuija
```

Then **double-click `Install Agent Ouija.bat`** in the repo root.
It'll walk you through:

1. **Prerequisite checks** — Python + Ollama on PATH.
2. **Folder setup** — creates `data/`, `docs/`, `prompts/`, etc.
3. **`.env` API keys** — prompts for each (all optional; skip any you
   don't have and add later via `python install.py --add-api-key`).
4. **Local model selection** — pulls the Ollama models you pick.
5. **`pip install -r requirements.txt`**.
6. **Piper voice download** (optional).
7. **Launcher** — generates `Agent Ouija.bat` for daily use.
8. **Git key-guard hook** — blocks accidental key commits.
9. **Smoke test** — imports every heavy dependency.

Two install modes:
- **Full** — one pass, installs every local model, prompts for every
  key. Recommended for a fresh rig.
- **Custom** — per-component Y/N. Skip individual models, decline
  the voice, refuse pip if you're behind a proxy.

**To run it after install:** double-click the generated
`Agent Ouija.bat`. Your default browser opens at
<http://localhost:7860>.

**To uninstall:** double-click `Uninstall Agent Ouija.bat`. Three
levels — data-only, data + keys, or full (data + keys + `ollama rm`
on every local model).

## Documentation

- **[SETUP.md](SETUP.md)** — install path, `.env` walkthrough,
  Tailscale HTTPS for mobile access, encrypted-`.env` option.
- **[USER_GUIDE.md](USER_GUIDE.md)** — feature reference.
- **[HOW_TO.md](HOW_TO.md)** — beginner-friendly walkthrough of every
  tab, with setup steps for Google Calendar, NewsAPI, SMTP.
- **[CLAUDE.md](CLAUDE.md)** — project intent + architecture, for
  developers reading the source.

## License

TBD.
