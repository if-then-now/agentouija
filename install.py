"""
install.py — Agent Ouija installer.

Three modes:

  FULL       Installs every local model, prompts for every API key
             (skip individually if you don't have one yet), downloads
             the default Piper voice, sets up the launcher + desktop
             shortcut. One decision at the start, then it runs.

  CUSTOM     Y/N per component. Skip individual models, decline the
             voice download, refuse pip if you're behind a proxy, etc.
             Same phases as Full but each one asks first.

  ADD KEY    Just re-opens the API-key prompts on an existing install
             so you can drop in a key you didn't have earlier. Nothing
             else runs. Ideal for adding Serper or NewsAPI later.

Every install phase is idempotent — safe to re-run. Existing files
are preserved (custom system_core.md, non-template .env keys, already-
present Piper voices, already-pulled Ollama models all short-circuit).

Usage:
    python install.py                # Interactive — asks Full vs Custom
    python install.py --full         # Skip the mode prompt
    python install.py --custom       # Skip the mode prompt
    python install.py --add-api-key  # Only re-run .env prompting
    python install.py --yes          # Accept every default (CI-friendly)
    python install.py --no-pip       # Custom mode: skip pip install
    python install.py --no-models    # Custom mode: skip Ollama pulls
    python install.py --no-env       # Custom mode: skip .env prompts
    python install.py --voice        # Custom mode: also grab the voice

Features requiring API keys (all in `.env`, all optional):
    ANTHROPIC_API_KEY   — Claude cloud model
    OPENAI_API_KEY      — GPT-4o cloud model
    GEMINI_API_KEY      — Gemini cloud model + image editor (Nano Banana)
    GROQ_API_KEY        — Groq Llama 3.3 70B (inference speed benchmark)
    OPENROUTER_API_KEY  — Hermes 3 Cloud (agentic instruction model)
    SERPER_API_KEY      — Google Search API for the web-grounded flows
    NEWSAPI_KEY         — Daily news pulls (100 requests/day free tier)

Google Calendar auth is a separate one-time step from within the app
(see the Data tab → 🔐 Authorize Google Calendar) after install.
"""

import argparse
import shutil
import os
import subprocess
import sys
from typing import Optional
import urllib.error
import urllib.request
from pathlib import Path

ROOT             = Path(__file__).resolve().parent
SCRIPTS          = ROOT / "scripts"
REQS             = ROOT / "requirements.txt"
ENV_SAMPLE       = ROOT / ".env.example"
PROMPTS_TEMPLATE = ROOT / "prompts_template"

# .env locations — preferred is out of the repo tree so a repo
# zip / clean / accidental delete can't leak or trash your API
# keys. Legacy path is the old repo-root file; the installer
# migrates it into ENV_FILE_PREFERRED and then leaves the legacy
# copy alone (Windows ACL applied to both if they exist).
ENV_FILE_LEGACY    = ROOT / ".env"
ENV_FILE_PREFERRED = Path.home() / ".config" / "agentouija" / ".env"


def _resolve_env_file() -> Path:
    """Where to read/write the .env file. Prefers the out-of-tree
    location; falls back to the legacy in-tree file for older
    installs so re-running the installer on an existing setup
    keeps working. Matches config.env_file_path() exactly — the
    installer and the running app must agree on the same path."""
    if ENV_FILE_PREFERRED.exists():
        return ENV_FILE_PREFERRED
    if ENV_FILE_LEGACY.exists():
        return ENV_FILE_LEGACY
    return ENV_FILE_PREFERRED

# Voice download — one voice ships by default (Cori, British female, high
# quality). Additional voices will be uploadable via the Persona tab later.
# Each entry maps a filename to its Hugging Face URL under rhasspy/piper-voices.
VOICE_FILES = {
    "en_GB-cori-high": {
        "onnx":
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            "en/en_GB/cori/high/en_GB-cori-high.onnx",
        "json":
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            "en/en_GB/cori/high/en_GB-cori-high.onnx.json",
    },
}

# ── Console helpers ──────────────────────────────────────────────────────────

def say(msg: str = "") -> None:
    print(msg)

def header(title: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def ok(msg: str)   -> None: print(f"  ✅ {msg}")
def warn(msg: str) -> None: print(f"  ⚠️  {msg}")
def err(msg: str)  -> None: print(f"  ❌ {msg}")

def ask_yn(prompt: str, default: bool = True, *, yes: bool = False) -> bool:
    if yes:
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"  {prompt} {suffix} ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    Please answer y or n.")

def ask_str(prompt: str, default: str = "", *, yes: bool = False) -> str:
    if yes:
        return default
    raw = input(f"  {prompt}{' ['+default+']' if default else ''}: ").strip()
    return raw or default


# ── Prereq checks ────────────────────────────────────────────────────────────

def check_python() -> bool:
    major, minor = sys.version_info.major, sys.version_info.minor
    say(f"  Python: {sys.version.split()[0]} ({sys.executable})")
    if (major, minor) < (3, 10):
        err("Python 3.10 or newer is required.")
        return False
    if (major, minor) >= (3, 14):
        warn(
            "Python 3.14+ detected — Piper TTS depends on onnxruntime which "
            "may not yet have a wheel for this version. Voice toggle may be "
            "unavailable until onnxruntime catches up."
        )
    ok("Python version OK")
    return True


def check_ollama() -> bool:
    path = shutil.which("ollama")
    if not path:
        err(
            "Ollama not found on PATH. Install from https://ollama.com/download "
            "and re-run this installer."
        )
        return False
    say(f"  Ollama: {path}")
    ok("Ollama detected")
    return True


# ── Folder structure ─────────────────────────────────────────────────────────

def ensure_folders() -> None:
    # Pull paths from config so adding/renaming a folder there flows here.
    sys.path.insert(0, str(SCRIPTS))
    from config import (   # noqa: E402  — import after path manipulation
        ASSETS_DIR,
        BACKUPS_DIR,
        CHROMA_DIR,
        DATA_DIR,
        DOCS_DIR,
        PRIVATE_CORPUS_EPISODES_DIR,
        PROMPTS_DIR,
        SUMMARIES_DIR,
        TRAINING_DIR,
        TTS_VOICES_DIR,
    )
    folders = [
        DATA_DIR, CHROMA_DIR, TRAINING_DIR, SUMMARIES_DIR, BACKUPS_DIR,
        DOCS_DIR, PROMPTS_DIR, PRIVATE_CORPUS_EPISODES_DIR,
        ASSETS_DIR, TTS_VOICES_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    ok(f"Folder structure ready under {DATA_DIR.parent}")

    # Seed the default mystical system_core.md from prompts_template/ if the
    # user's prompts/ dir doesn't have one yet. Existing files are preserved
    # so a customized system_core.md is never clobbered.
    template = PROMPTS_TEMPLATE / "system_core.md"
    target   = PROMPTS_DIR / "system_core.md"
    if template.exists() and not target.exists():
        target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        ok(f"Seeded default {target.name} (mystical Agent Ouija voice)")
    elif target.exists():
        ok(f"{target.name} already present — preserving current text")


# ── .env handling ────────────────────────────────────────────────────────────

def _prompt_new_passphrase(*, yes: bool) -> Optional[str]:
    """Prompt (twice, hidden) for a new .env passphrase. Returns
    None on cancel / empty input / mismatch after 3 attempts.
    yes=True short-circuits — you can't set a passphrase in a
    non-interactive install; the caller falls back to plaintext
    with a warning."""
    if yes:
        warn("--yes mode: skipping passphrase prompt (plaintext .env)")
        return None
    import getpass
    for attempt in range(1, 4):
        try:
            pw1 = getpass.getpass(
                f"  New .env passphrase (attempt {attempt}/3, "
                "empty = cancel encryption): ",
            )
        except (EOFError, KeyboardInterrupt):
            warn("Cancelled — leaving .env in plaintext")
            return None
        if not pw1:
            warn("Empty passphrase — skipping encryption "
                 "(.env stays plaintext)")
            return None
        if len(pw1) < 8:
            print("  Too short (need 8+ chars). Try again.")
            continue
        try:
            pw2 = getpass.getpass("  Confirm passphrase: ")
        except (EOFError, KeyboardInterrupt):
            warn("Cancelled — leaving .env in plaintext")
            return None
        if pw1 == pw2:
            return pw1
        print("  Passphrases don't match. Try again.")
    warn("Too many mismatches — leaving .env in plaintext")
    return None


def _read_encrypted_for_edit(env_path: Path) -> Optional[str]:
    """Prompt for the existing passphrase and return the decrypted
    plaintext for editing. None on user-cancel or 3 wrong attempts.
    Called by --add-api-key / --encrypt-env / --decrypt-env when
    they need to touch an already-encrypted file."""
    import getpass
    sys.path.insert(0, str(SCRIPTS))
    from env_crypto import read_encrypted
    try:
        from cryptography.fernet import InvalidToken
    except ImportError:
        err("cryptography missing — run `pip install cryptography`")
        return None
    for attempt in range(1, 4):
        try:
            pw = getpass.getpass(
                f"  Existing .env passphrase (attempt {attempt}/3): ",
            )
        except (EOFError, KeyboardInterrupt):
            return None
        if not pw:
            return None
        try:
            return read_encrypted(env_path, pw)
        except InvalidToken:
            print("  Wrong passphrase.")
    return None


def _maybe_encrypt_env(*, yes: bool, is_full: bool) -> None:
    """After ensure_env_file wrote the plaintext to disk, offer
    to encrypt it with a passphrase. Recommended for laptop
    installs — a stolen device with the disk unencrypted (no
    BitLocker) can't read the file without the passphrase.
    Recommended-against for a desktop rig you never move.

    is_full drives the default answer: Full mode defaults to No
    (don't want to surprise users with an extra prompt they must
    respond to for the install to complete); Custom mode defaults
    to No too but explicitly asks."""
    sys.path.insert(0, str(SCRIPTS))
    from env_crypto import (
        is_encrypted_at, write_encrypted,
    )

    target = _resolve_env_file()
    if is_encrypted_at(target):
        # Already encrypted; nothing to do unless user is
        # rotating the passphrase (handled by --encrypt-env).
        ok(f"{target.name} already encrypted — leaving as-is")
        return
    if not target.exists():
        warn(f"No .env at {target} to encrypt — skipping")
        return

    say("")
    say("  🔐 Optional: encrypt .env with a passphrase.")
    say("     Recommended for **laptop installs** where the disk")
    say("     might not be encrypted at rest (BitLocker off, etc.)")
    say("     — a stolen or lost device can't read your API keys")
    say("     without the passphrase.")
    say("     Not needed on a stationary desktop rig you trust.")
    say("     If you say yes, arena_ui.py will prompt for the")
    say("     passphrase every time it starts (or set")
    say("     AGENT_OUIJA_ENV_PASSPHRASE for headless launches).")

    if not ask_yn("  Encrypt .env now?", default=False, yes=yes):
        say("  Leaving .env in plaintext. Toggle later with:")
        say("    python install.py --encrypt-env")
        return

    passphrase = _prompt_new_passphrase(yes=yes)
    if not passphrase:
        return

    try:
        plaintext = target.read_text(encoding="utf-8")
        write_encrypted(target, plaintext, passphrase)
        ok(f"Encrypted {target.name} → {target.name}.enc + .meta")
        warn(
            "Store your passphrase somewhere secure (password "
            "manager). No recovery — a lost passphrase means "
            "starting over with fresh API keys."
        )
    except Exception as e:
        err(f"Encrypt failed: {type(e).__name__}: {e}")


def run_encrypt_env(args) -> int:
    """`--encrypt-env` subcommand — take a plaintext .env and
    encrypt it, or rotate the passphrase on an already-encrypted
    one. No other install phases run."""
    header("Agent Ouija — Encrypt .env")
    sys.path.insert(0, str(SCRIPTS))
    from env_crypto import (
        is_encrypted_at, write_encrypted,
    )
    target = _resolve_env_file()

    if is_encrypted_at(target):
        say(f"  {target.name} is already encrypted. Rotating "
            "passphrase — enter current one first.")
        plaintext = _read_encrypted_for_edit(target)
        if plaintext is None:
            err("Cancelled or 3 wrong attempts — passphrase not rotated")
            return 1
    else:
        if not target.exists():
            err(f"No .env at {target} — nothing to encrypt")
            return 1
        plaintext = target.read_text(encoding="utf-8")

    new_pw = _prompt_new_passphrase(yes=args.yes)
    if not new_pw:
        return 1
    try:
        write_encrypted(target, plaintext, new_pw)
        ok(f"Encrypted {target.name} — plaintext removed")
        say("")
        say("  arena_ui.py will now prompt for the passphrase on")
        say("  every launch. For headless runs, set the environment")
        say("  variable AGENT_OUIJA_ENV_PASSPHRASE and relaunch.")
        return 0
    except Exception as e:
        err(f"Encrypt failed: {type(e).__name__}: {e}")
        return 1


def run_decrypt_env(args) -> int:
    """`--decrypt-env` subcommand — reverse encrypt_env. Prompts
    for the current passphrase, writes plaintext back to the .env
    file, deletes the .env.enc + .env.enc.meta pair."""
    header("Agent Ouija — Decrypt .env (back to plaintext)")
    sys.path.insert(0, str(SCRIPTS))
    from env_crypto import (
        ciphertext_paths, is_encrypted_at,
    )
    target = _resolve_env_file()
    if not is_encrypted_at(target):
        err(f"No encrypted .env at {target}.enc — nothing to decrypt")
        return 1

    plaintext = _read_encrypted_for_edit(target)
    if plaintext is None:
        err("Cancelled or 3 wrong attempts — .env stays encrypted")
        return 1
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(plaintext, encoding="utf-8")
        _lock_env_permissions(target)
        ct_path, meta_path = ciphertext_paths(target)
        ct_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        ok(f"Decrypted → {target}. .enc + .enc.meta removed.")
        warn("Plaintext .env is now on disk. Only do this if the "
             "underlying filesystem is encrypted at rest, or if "
             "you're about to re-encrypt with a new passphrase.")
        return 0
    except Exception as e:
        err(f"Decrypt failed: {type(e).__name__}: {e}")
        return 1


def ensure_env_file(*, yes: bool) -> None:
    """Read from the resolved .env location (preferred out-of-tree
    path, fallback to legacy repo-root), prompt for each key from
    the template, write back to the PREFERRED location, migrate
    any legacy repo-root file into it, and lock permissions.

    Handles the encrypted case transparently — if the resolved
    .env is a .env.enc pair, we prompt for the existing passphrase
    to decrypt for editing, then re-encrypt on save with the same
    passphrase. Rotating the passphrase is a separate --encrypt-env
    subcommand."""
    if not ENV_SAMPLE.exists():
        warn(".env.example missing — skipping .env setup")
        return

    sys.path.insert(0, str(SCRIPTS))
    from env_crypto import (
        is_encrypted_at, write_encrypted,
    )

    # Bootstrap: if neither .env (plain or encrypted) exists yet,
    # seed the preferred location from the template.
    ENV_FILE_PREFERRED.parent.mkdir(parents=True, exist_ok=True)
    if (not ENV_FILE_PREFERRED.exists()
            and not ENV_FILE_LEGACY.exists()
            and not is_encrypted_at(ENV_FILE_PREFERRED)
            and not is_encrypted_at(ENV_FILE_LEGACY)):
        ENV_FILE_PREFERRED.write_text(
            ENV_SAMPLE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        ok(f"Created {ENV_FILE_PREFERRED} from template")

    # Migration: if only the legacy .env exists (older install),
    # move its contents into the preferred location. Skip when
    # the legacy is encrypted — that's a rare edge case not worth
    # auto-migrating; user can decrypt then re-encrypt at the new
    # location manually.
    if (ENV_FILE_LEGACY.exists()
            and not ENV_FILE_PREFERRED.exists()
            and not is_encrypted_at(ENV_FILE_PREFERRED)):
        say(
            "\n  Found legacy `.env` at the repo root. Copying it "
            f"to {ENV_FILE_PREFERRED} so a repo delete / zip / "
            "upload can't leak your keys."
        )
        ENV_FILE_PREFERRED.write_text(
            ENV_FILE_LEGACY.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        ok(f"Migrated legacy .env → {ENV_FILE_PREFERRED}")

    target = _resolve_env_file()

    # If the target is encrypted, decrypt in memory so we can
    # merge new prompts, then re-encrypt on save.
    reencrypt_pass: Optional[str] = None
    if is_encrypted_at(target):
        say(
            f"\n  {target.name} is encrypted. Enter your passphrase "
            "so the installer can prompt for any missing keys."
        )
        plaintext_str = _read_encrypted_for_edit(target)
        if plaintext_str is None:
            err("Skipping .env update — could not decrypt")
            return
        existing = _parse_env_str(plaintext_str)
        # We'll re-encrypt with the SAME passphrase on save.
        # Rotation is a separate --encrypt-env command.
        import getpass
        try:
            reencrypt_pass = getpass.getpass(
                "  Re-enter passphrase for re-encrypt (same one): "
            )
        except (EOFError, KeyboardInterrupt):
            err("Cancelled — .env unchanged")
            return
    else:
        existing = _parse_env(target)

    template = _parse_env(ENV_SAMPLE)

    new_lines: list[str] = []
    say("\n  Enter API keys (leave blank to skip; existing values preserved):")
    for key in template:
        cur = existing.get(key, "")
        if cur and not yes:
            keep = ask_yn(f"  {key} already set — keep?", default=True, yes=yes)
            if keep:
                new_lines.append(f"{key}={cur}")
                continue
        val = ask_str(f"  {key}", default=cur, yes=yes)
        new_lines.append(f"{key}={val}")

    # Preserve any non-template keys the user may have added themselves
    extras = [(k, v) for k, v in existing.items() if k not in template]
    if extras:
        new_lines.append("")
        new_lines.append("# Keys not in the .env.example template — preserved as-is")
        for k, v in extras:
            new_lines.append(f"{k}={v}")

    # Header comment from template, if any
    head_lines = []
    for ln in ENV_SAMPLE.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            head_lines.append(ln)
        else:
            break
    out = "\n".join(head_lines + new_lines).rstrip() + "\n"

    # Encrypted → write back encrypted at the same location.
    if reencrypt_pass is not None:
        try:
            write_encrypted(target, out, reencrypt_pass)
            ok(f"{target.name} re-encrypted at {target}")
        except Exception as e:
            err(f"Re-encrypt failed — writing plaintext: {e}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(out, encoding="utf-8")
        _lock_env_permissions(target.with_suffix(".env.enc"))
        _lock_env_permissions(target.with_suffix(".env.enc.meta"))
        return

    # Plaintext path — write to the preferred location.
    ENV_FILE_PREFERRED.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE_PREFERRED.write_text(out, encoding="utf-8")
    ok(f"{ENV_FILE_PREFERRED} updated")

    _lock_env_permissions(ENV_FILE_PREFERRED)
    if ENV_FILE_LEGACY.exists() and ENV_FILE_LEGACY != ENV_FILE_PREFERRED:
        # If a legacy copy still exists, lock it too. User can
        # delete it manually when they're done vetting the migration.
        _lock_env_permissions(ENV_FILE_LEGACY)
        warn(
            f"Legacy .env still at {ENV_FILE_LEGACY}. Safe to delete "
            "once you've confirmed the app runs from the new location."
        )


def _lock_env_permissions(path: Path) -> None:
    """Lock a .env down so only the current user + SYSTEM can
    read it. Windows uses icacls; POSIX uses chmod 600. Best-
    effort — logs and continues on failure so the install
    doesn't wedge on a permission tweak."""
    if not path.exists():
        return
    if sys.platform == "win32":
        try:
            rc = subprocess.call(
                ["icacls", str(path),
                 "/inheritance:r",
                 "/grant:r", f"{os.environ.get('USERNAME', '%USERNAME%')}:(F)",
                 "/grant:r", "SYSTEM:(F)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if rc == 0:
                ok(f"Locked {path.name} ACL to current user + SYSTEM only")
            else:
                warn(f"icacls returned {rc} on {path} — permissions unchanged")
        except FileNotFoundError:
            warn("icacls not found on PATH — skipping .env ACL lockdown "
                 "(only relevant if you share this machine with other users)")
    else:
        try:
            path.chmod(0o600)
            ok(f"Set {path.name} mode 0600 (owner-read-write only)")
        except OSError as e:
            warn(f"chmod on {path} failed: {e}")


def _parse_env(path: Path) -> dict:
    """Tiny KEY=VAL parser — preserves order via dict-insertion."""
    if not path.exists():
        return {}
    return _parse_env_str(path.read_text(encoding="utf-8"))


def _parse_env_str(text: str) -> dict:
    """Same as _parse_env, but from an in-memory string. Used by
    the encrypted-.env path (decrypt-in-memory → parse → prompt →
    re-encrypt) so we never write the plaintext to disk in
    between."""
    out: dict = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, _, v = ln.partition("=")
        out[k.strip()] = v.strip()
    return out


# ── Model selection + ollama pull ────────────────────────────────────────────

def select_and_pull_models(*, yes: bool, install_all: bool = False) -> None:
    """Y/N per-model prompt (Custom) or fire-and-forget "install every
    local model in config.AVAILABLE_MODELS" (Full). install_all=True
    skips the per-model prompts and pulls the whole set."""
    sys.path.insert(0, str(SCRIPTS))
    from config import AVAILABLE_MODELS   # noqa: E402

    locals_in_config = [
        (k, cfg) for k, cfg in AVAILABLE_MODELS.items()
        if cfg.get("type") == "local"
    ]
    if not locals_in_config:
        warn("No local models defined in config.AVAILABLE_MODELS — nothing to pull")
        return

    if install_all:
        say(f"\n  Full install — pulling every local model ({len(locals_in_config)}). "
            "Downloads are 2-8 GB each.")
        selected = list(locals_in_config)
    else:
        say("\n  Y/N for each local model. Selected models will be pulled via "
            "`ollama pull` (downloads are 2-8 GB each; skip ones you don't want).")
        selected = []
        for key, cfg in locals_in_config:
            display = cfg.get("display_name", key)
            desc    = cfg.get("description",  "")
            say(f"\n  {display} — {desc}")
            if ask_yn(f"  Install {key}?", default=True, yes=yes):
                selected.append((key, cfg))

    if not selected:
        warn("No local models selected — Arena will only work with cloud keys")
        return

    say(f"\n  Pulling {len(selected)} model(s)…")
    for key, cfg in selected:
        name = cfg["name"]
        say(f"\n  >>> ollama pull {name}")
        rc = subprocess.call(["ollama", "pull", name])
        if rc == 0:
            ok(f"{name} ready")
        else:
            err(f"ollama pull {name} returned exit code {rc}")


# ── pip install ──────────────────────────────────────────────────────────────

def pip_install(*, yes: bool) -> None:
    if not REQS.exists():
        warn("requirements.txt missing — skipping pip install")
        return
    if not ask_yn(f"Install Python deps from {REQS.name}?", default=True, yes=yes):
        warn("Skipped pip install — run it later with: "
             f"pip install -r {REQS.name}")
        return
    say(f"  >>> {sys.executable} -m pip install -r {REQS.name}")
    rc = subprocess.call([sys.executable, "-m", "pip", "install",
                          "-r", str(REQS)])
    if rc == 0:
        ok("pip install complete")
    else:
        err(f"pip install returned exit code {rc} — check the output above")


# ── Voice download (optional) ────────────────────────────────────────────────

def download_voice(*, yes: bool) -> None:
    sys.path.insert(0, str(SCRIPTS))
    from config import TTS_VOICES_DIR   # noqa: E402

    if not ask_yn(
        f"Download Piper voice '{next(iter(VOICE_FILES))}' (~100MB) into "
        f"{TTS_VOICES_DIR.name}/?",
        default=True, yes=yes,
    ):
        warn("Voice toggles in the Arena / Research tabs will be a no-op "
             "until you download a Piper voice manually.")
        return

    TTS_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for voice_id, urls in VOICE_FILES.items():
        for suffix, url in (("onnx", urls["onnx"]),
                            ("onnx.json", urls["json"])):
            fname  = f"{voice_id}.{suffix}"
            target = TTS_VOICES_DIR / fname
            if target.exists() and target.stat().st_size > 1024:
                ok(f"{fname} already present ({target.stat().st_size // 1024} KB)")
                continue
            say(f"  Downloading {fname}…")
            try:
                urllib.request.urlretrieve(url, target)
                ok(f"{fname} → {target}")
            except urllib.error.URLError as e:
                err(f"Failed to download {fname}: {e}")


# ── Launcher (.bat + optional desktop shortcut) ──────────────────────────────

LAUNCHER_NAME = "Agent Ouija.bat"
SHORTCUT_NAME = "Agent Ouija.lnk"

LAUNCHER_BAT = r"""@echo off
title Agent Ouija
cd /d "%~dp0"
echo ============================================================
echo   Agent Ouija
echo ============================================================
echo Launching the web UI. Your browser will open automatically.
echo Closing this window will stop Agent Ouija.
echo.
python scripts\arena_ui.py
if errorlevel 1 (
    echo.
    echo Agent Ouija exited with an error.
    echo Run 'python install.py' from this folder to re-validate the setup.
    pause >nul
)
"""


# ── Git hook installer (optional, prevents committing keys) ──────────────────

PRECOMMIT_HOOK = r"""#!/usr/bin/env sh
# Agent Ouija — pre-commit hook (installed by install.py).
# Blocks a commit if any STAGED line looks like it contains an API key.
# Rewrite / disable in .git/hooks/pre-commit if you need to bypass.

set -e

# Patterns: anything ending in _KEY, _PASSWORD, _SECRET, _TOKEN with a
# non-empty value. Also blocks stray Anthropic-style sk-ant-… and OpenAI
# sk-… tokens even outside .env.
if git diff --cached --unified=0 | grep -E \
    '(API_KEY|SECRET_KEY|_PASSWORD|_TOKEN)[[:space:]]*=[[:space:]]*[^[:space:]]+' \
    | grep -Ev '=[[:space:]]*$' \
    | grep -Ev '=[[:space:]]*<[^>]+>' \
    > /dev/null; then
    echo "❌ pre-commit: refusing to commit — staged diff contains a"
    echo "   non-empty API_KEY / PASSWORD / TOKEN assignment. Double-"
    echo "   check nothing sensitive is in .env or any file you're"
    echo "   committing. To bypass (dangerous): git commit --no-verify"
    exit 1
fi

if git diff --cached | grep -E 'sk-(ant-|proj-|None-)[A-Za-z0-9_-]{16,}' > /dev/null; then
    echo "❌ pre-commit: staged diff contains what looks like an"
    echo "   Anthropic/OpenAI API key. Refusing to commit."
    exit 1
fi

exit 0
"""


def install_precommit_hook(*, yes: bool) -> None:
    """Install a pre-commit hook that blocks accidental commits of
    API keys / passwords / tokens. Hooks aren't shared through git
    (they live under .git/hooks/), so this is per-clone — a fresh
    checkout has to re-run install.py to pick it up."""
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        warn(
            f"No .git/ directory under {ROOT} — skipping pre-commit "
            "hook install. This is the correct outcome if you're "
            "running install.py on a downloaded zip; harmless."
        )
        return
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    target = hooks_dir / "pre-commit"

    # If the user already has a hook, don't clobber — offer to
    # replace instead. `yes=True` skips the prompt (Full mode).
    if target.exists() and not yes:
        keep = ask_yn(
            f"  {target.name} already exists — replace with the "
            "Agent Ouija key-guard hook?",
            default=False, yes=yes,
        )
        if not keep:
            warn("Kept existing pre-commit hook — nothing changed")
            return

    try:
        target.write_text(PRECOMMIT_HOOK, encoding="utf-8", newline="\n")
        # POSIX needs the executable bit. Windows ignores mode
        # bits; git-for-Windows still respects the hook via sh.exe.
        try:
            target.chmod(0o755)
        except OSError:
            pass
        ok(f"Installed key-guard pre-commit hook → {target}")
        say(
            "  Blocks commits whose staged diff contains a non-blank "
            "API_KEY / SECRET / PASSWORD / TOKEN assignment or an "
            "Anthropic-style sk-ant-… token. Bypass with "
            "`git commit --no-verify` if you ever need to."
        )
    except OSError as e:
        warn(f"Could not write pre-commit hook: {e}")


def create_launcher(*, yes: bool) -> None:
    """Write a double-clickable .bat launcher into the repo root."""
    target = ROOT / LAUNCHER_NAME
    target.write_text(LAUNCHER_BAT, encoding="utf-8")
    ok(f"Created {LAUNCHER_NAME} in {ROOT}")

    if not ask_yn(
        f"Also create a desktop shortcut ({SHORTCUT_NAME})?",
        default=True, yes=yes,
    ):
        return

    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        warn(f"No Desktop folder at {desktop} — skipping shortcut")
        return

    shortcut_path = desktop / SHORTCUT_NAME
    icon_path     = ROOT / "assets" / "ouija.ico"
    icon_line     = (
        f'$Shortcut.IconLocation = "{icon_path}"'
        if icon_path.exists() else ""
    )

    # PowerShell COM is the most reliable way to create a Windows .lnk
    # without pulling in pywin32. Build the script with literal paths so
    # spaces in "Agent Ouija" don't trip up the shell.
    ps = (
        f'$WshShell = New-Object -ComObject WScript.Shell; '
        f'$Shortcut = $WshShell.CreateShortcut("{shortcut_path}"); '
        f'$Shortcut.TargetPath = "{target}"; '
        f'$Shortcut.WorkingDirectory = "{ROOT}"; '
        f'{icon_line}; '
        f'$Shortcut.Save()'
    )
    try:
        rc = subprocess.call(
            ["powershell", "-NoProfile", "-Command", ps],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if rc == 0 and shortcut_path.exists():
            ok(f"Desktop shortcut → {shortcut_path}")
        else:
            warn(
                "PowerShell shortcut creation returned non-zero. "
                f"You can still launch via {target} directly."
            )
    except FileNotFoundError:
        warn(
            "PowerShell not found on PATH — skipping shortcut. "
            f"You can still launch via {target} directly."
        )


# ── Smoke test ───────────────────────────────────────────────────────────────

def smoke_test() -> None:
    """Import the heavy stack so wheel/version problems surface here
    rather than in a Gradio 500 page later. Also verify the shipped
    branding assets are present — the app falls back to text
    placeholders when they aren't, and testers have hit that with
    incomplete clones / antivirus quarantines / accidental deletes.
    Warn but don't fail on missing assets since the app boots
    either way."""
    sys.path.insert(0, str(SCRIPTS))
    checks = [
        ("gradio",                       "import gradio"),
        ("chromadb",                     "import chromadb"),
        ("llama_index.core",             "from llama_index.core import Settings"),
        ("llama_index embeddings",
            "from llama_index.embeddings.huggingface import HuggingFaceEmbedding"),
        ("llama_index chroma store",
            "from llama_index.vector_stores.chroma import ChromaVectorStore"),
        ("litellm",                      "from litellm import completion"),
        ("dotenv",                       "from dotenv import load_dotenv"),
        ("Agent Ouija config",             "from config import AVAILABLE_MODELS"),
    ]
    failed = []
    for name, stmt in checks:
        try:
            exec(stmt, {})
            ok(name)
        except Exception as e:
            err(f"{name}: {type(e).__name__}: {e}")
            failed.append(name)

    # Asset check — non-fatal but noisy so testers see it. Both
    # PNGs are tracked in git; missing ones point at an incomplete
    # clone or an over-eager antivirus quarantine.
    asset_root = ROOT / "assets"
    missing_assets = []
    for asset in ("ouija.png", "itn_logo.png"):
        p = asset_root / asset
        if p.exists() and p.stat().st_size > 0:
            ok(f"asset present: {asset} ({p.stat().st_size:,} bytes)")
        else:
            missing_assets.append(str(p))
            warn(
                f"asset missing: {p} — UI will render a text "
                "placeholder in its place. Try `git status` (was "
                "it deleted?) or `git checkout HEAD -- assets/` to "
                "restore. If it's an antivirus false-positive, "
                "whitelist the repo folder and re-checkout."
            )

    if failed:
        warn(f"Smoke test had {len(failed)} import failure(s). The "
             "web UI may not boot until these are resolved.")
    elif missing_assets:
        warn(f"Smoke test clean apart from {len(missing_assets)} "
             "missing asset(s) above. App will still boot with text "
             "placeholders.")
    else:
        ok("All imports + assets clean")


# ── Main flow ────────────────────────────────────────────────────────────────

def pick_mode() -> str:
    """Interactive mode selection at the top of the installer.
    Returns 'full' or 'custom'. --full / --custom / --add-api-key
    flags skip this prompt."""
    say("")
    say("  Install modes:")
    say("")
    say("    [1] Full install")
    say("        - Pulls every local model in config.AVAILABLE_MODELS.")
    say("        - Prompts for every API key (skip any you don't have).")
    say("        - Downloads the default Piper voice.")
    say("        - pip install -r requirements.txt.")
    say("        - Creates the launcher .bat + optional desktop shortcut.")
    say("        - No per-component Y/N — one pass, done.")
    say("")
    say("    [2] Custom install")
    say("        - Y/N per local model (skip individual models).")
    say("        - Y/N per API key (still skip individual keys).")
    say("        - Y/N for voice, pip, launcher, shortcut, smoke test.")
    say("        - Pick and choose everything.")
    say("")
    while True:
        raw = input("  Pick 1 (full) or 2 (custom): ").strip().lower()
        if raw in ("1", "full", "f"):
            return "full"
        if raw in ("2", "custom", "c"):
            return "custom"
        say("    Please answer 1 or 2.")


def run_add_api_key(args) -> int:
    """Just re-open the .env prompts on an existing install. No model
    pulls, no pip install, no smoke test — the user picked this mode
    because everything else is already set up and they want to add
    a key they've since obtained (SERPER, NEWSAPI, one of the cloud
    providers, etc.)."""
    header("Agent Ouija — Add API Key")
    say(f"  Repo root: {ROOT}")
    if not check_python():
        return 1
    header("Update .env")
    say("  Every key from .env.example will be re-prompted. Blank input")
    say("  keeps the current value; press Enter to skip a key you already")
    say("  have. The .env file is gitignored — nothing here leaves the rig.")
    ensure_env_file(yes=args.yes)
    say("")
    say("  Done. Restart Agent Ouija if it was running so the new key(s)")
    say("  pick up on the next process launch.")
    return 0


def run_install(args, mode: str) -> int:
    """Shared install driver for Full and Custom modes. Every phase is
    idempotent, so re-running is safe."""
    header(f"Agent Ouija — {mode.title()} Install")
    say(f"  Repo root: {ROOT}")
    if mode == "full":
        say("  Full install: every local model, every API-key prompt,")
        say("  voice download, launcher + shortcut.")

    header("1. Prerequisite checks")
    if not check_python():
        return 1
    if not args.no_models and not check_ollama():
        return 1

    header("2. Folder structure")
    ensure_folders()

    if not args.no_env:
        header("3. .env API keys")
        if mode == "full":
            say("  Every key from .env.example will be prompted for. Blank")
            say("  input SKIPS that key — the feature stays dormant until")
            say("  you add the key later with:  python install.py --add-api-key")
        ensure_env_file(yes=args.yes)
        # Immediately after writing plaintext .env, offer to
        # encrypt it. Skipped on --no-env.
        _maybe_encrypt_env(yes=args.yes, is_full=(mode == "full"))
    else:
        warn("Skipping .env step (--no-env)")

    if not args.no_models:
        header("4. Local model selection")
        select_and_pull_models(
            yes=args.yes,
            install_all=(mode == "full"),
        )
    else:
        warn("Skipping local model step (--no-models)")

    if not args.no_pip:
        header("5. Python dependencies")
        # Full mode: no per-component question, just install.
        pip_install(yes=args.yes or mode == "full")
    else:
        warn("Skipping pip install (--no-pip)")

    if args.voice or mode == "full":
        header("6. Piper voice")
        download_voice(yes=args.yes or mode == "full")

    header("7. Launcher")
    create_launcher(yes=args.yes or mode == "full")

    header("8. Git key-guard hook")
    install_precommit_hook(yes=args.yes or mode == "full")

    header("9. Smoke test")
    smoke_test()

    header("Done")
    say("  Two ways to start Agent Ouija:")
    say(f"    A) Double-click {LAUNCHER_NAME} (in {ROOT})")
    say(f"    B) python {SCRIPTS / 'arena_ui.py'}")
    say("")
    say("  The web UI launches at http://localhost:7860 and your default")
    say("  browser opens automatically.")
    say("")
    say("  Add an API key you didn't have earlier:")
    say("     python install.py --add-api-key")
    say("")
    say("  Remove the app + all personal data:")
    say("     python uninstall.py")
    say("")
    say("  Read USER_GUIDE.md for the per-tab walkthrough and SETUP.md for")
    say("  the abbreviated quick-start.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent Ouija installer")
    # Mode selectors — mutually exclusive, but argparse's mutually
    # exclusive groups don't play with default=None fallback well;
    # cheaper to accept all three and pick precedence in code.
    parser.add_argument("--full", action="store_true",
                        help="Full install without the mode prompt")
    parser.add_argument("--custom", action="store_true",
                        help="Custom install (per-component Y/N) without the mode prompt")
    parser.add_argument("--add-api-key", action="store_true",
                        help="Re-open .env prompts on an existing install (no other phases run)")
    parser.add_argument("--encrypt-env", action="store_true",
                        help="Encrypt an existing .env with a passphrase (or rotate the passphrase on an already-encrypted one)")
    parser.add_argument("--decrypt-env", action="store_true",
                        help="Reverse of --encrypt-env: prompt for passphrase, write plaintext back, delete .env.enc + .meta")
    # Per-phase skips — apply in Custom mode; Full mode ignores them.
    parser.add_argument("--no-pip", action="store_true",
                        help="Skip pip install -r requirements.txt")
    parser.add_argument("--no-models", action="store_true",
                        help="Skip Y/N model selection + ollama pull")
    parser.add_argument("--no-env", action="store_true",
                        help="Skip .env creation / prompting")
    parser.add_argument("--voice", action="store_true",
                        help="Custom mode: also download the default Piper voice")
    parser.add_argument("--yes", action="store_true",
                        help="Accept defaults at every prompt (CI-friendly)")
    args = parser.parse_args()

    # Short-circuit subcommands — each returns without running any
    # other install phases. Only one can be active at a time.
    if args.encrypt_env:
        return run_encrypt_env(args)
    if args.decrypt_env:
        return run_decrypt_env(args)
    if args.add_api_key:
        return run_add_api_key(args)

    # Pick a mode. Explicit flag wins; interactive prompt otherwise.
    if args.full:
        mode = "full"
    elif args.custom:
        mode = "custom"
    else:
        mode = pick_mode()

    return run_install(args, mode)


if __name__ == "__main__":
    sys.exit(main())
