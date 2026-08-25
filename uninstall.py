"""
uninstall.py — Agent Ouija uninstaller.

Removes personal data, generated files, and (optionally) the Ollama
models this app pulled down. Does NOT delete the repo folder itself —
you delete that manually as a last step once the script is done.

Three levels, picked interactively or via flag:

  DATA-ONLY    Wipes everything the app generated at runtime:
               data/, docs/, prompts/, voices/, plus the launcher .bat
               and desktop shortcut. Leaves .env intact so cloud keys
               don't have to be re-entered on a reinstall.

  DATA + KEYS  Above, PLUS deletes .env (cloud API keys). Copyright/
               folder deleted too if it exists. Nothing from the app
               remains outside the scripts folder.

  FULL         Above, PLUS runs `ollama rm <name>` for every local
               model listed in config.AVAILABLE_MODELS. Frees the
               2-8 GB per model that Ollama keeps in its own store.

Usage:
    python uninstall.py                # Interactive — pick a level
    python uninstall.py --level=data   # data-only
    python uninstall.py --level=keys   # data + .env
    python uninstall.py --level=full   # data + .env + ollama rm
    python uninstall.py --yes          # Skip the typed-YES confirmation
                                       # (dangerous — for automation only)

Always prompts for a typed "YES" confirmation before deleting anything
unless --yes is passed. Every step is logged to stdout; nothing is
silently deleted.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


# ── Console helpers ──────────────────────────────────────────────────────────

def say(msg: str = "") -> None:
    print(msg)

def header(title: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def ok(msg: str)   -> None: print(f"  ✅ {msg}")
def warn(msg: str) -> None: print(f"  ⚠️  {msg}")
def err(msg: str)  -> None: print(f"  ❌ {msg}")


# ── Removal targets ──────────────────────────────────────────────────────────

def _load_paths():
    """Import config to know exactly which folders count as user data.
    Kept lazy so uninstall.py runs even if the repo is in a mixed
    Python-version state (config only imports the stdlib).

    Falls back to a "just the repo root" dict when the config import
    fails — a broken clone shouldn't take uninstall down with it;
    the tester still gets the launcher + .env cleanup path via
    ROOT-relative removals in wipe_data / wipe_env."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        from config import (   # noqa: E402
            BACKUPS_DIR,
            CHROMA_DIR,
            CHAT_HISTORY_PATH,
            DATA_DIR,
            DOCS_DIR,
            MEDIA_DIR,
            PRIVATE_CORPUS_EPISODES_DIR,
            PROMPTS_DIR,
            SEANCES_DIR,
            SUMMARIES_DIR,
            TRAINING_DIR,
            TTS_VOICES_DIR,
        )
    except Exception as e:
        warn(
            f"Could not import config ({type(e).__name__}: {e}). "
            "Falling back to repo-root guesses — uninstall will still "
            "wipe whatever it can find but may miss data folders "
            "config would have named. Fix the config error before "
            "re-running for a complete uninstall."
        )
        DATA_DIR = ROOT / "data"
        return {
            "DATA_DIR":                    DATA_DIR,
            "CHROMA_DIR":                  DATA_DIR / "chroma",
            "CHAT_HISTORY_PATH":           DATA_DIR / "chat_history.json",
            "SEANCES_DIR":                 DATA_DIR / "seances",
            "TRAINING_DIR":                DATA_DIR / "training",
            "SUMMARIES_DIR":               DATA_DIR / "summaries",
            "BACKUPS_DIR":                 DATA_DIR / "backups",
            "DOCS_DIR":                    ROOT / "docs",
            "MEDIA_DIR":                   ROOT / "docs" / "media",
            "PROMPTS_DIR":                 ROOT / "prompts",
            "PRIVATE_CORPUS_EPISODES_DIR": ROOT / "Copyright" / "episodes",
            "TTS_VOICES_DIR":              ROOT / "voices",
        }
    return {
        "DATA_DIR":                    DATA_DIR,
        "CHROMA_DIR":                  CHROMA_DIR,
        "CHAT_HISTORY_PATH":           CHAT_HISTORY_PATH,
        "SEANCES_DIR":                 SEANCES_DIR,
        "TRAINING_DIR":                TRAINING_DIR,
        "SUMMARIES_DIR":               SUMMARIES_DIR,
        "BACKUPS_DIR":                 BACKUPS_DIR,
        "DOCS_DIR":                    DOCS_DIR,
        "MEDIA_DIR":                   MEDIA_DIR,
        "PROMPTS_DIR":                 PROMPTS_DIR,
        "PRIVATE_CORPUS_EPISODES_DIR": PRIVATE_CORPUS_EPISODES_DIR,
        "TTS_VOICES_DIR":              TTS_VOICES_DIR,
    }


def _rm_tree(p: Path) -> None:
    if not p.exists():
        say(f"     (skip) {p} — not present")
        return
    try:
        shutil.rmtree(p)
        ok(f"Removed {p}")
    except OSError as e:
        err(f"Could not remove {p}: {e}")


def _rm_file(p: Path) -> None:
    if not p.exists():
        say(f"     (skip) {p} — not present")
        return
    try:
        p.unlink()
        ok(f"Removed {p}")
    except OSError as e:
        err(f"Could not remove {p}: {e}")


def _summarize_removal(level: str, paths: dict) -> None:
    say("  This will remove:")
    say(f"    - {paths['DATA_DIR']}         (all scores, ChromaDB, seances, backups)")
    say(f"    - {paths['DOCS_DIR']}         (uploaded documents + media)")
    say(f"    - {paths['PROMPTS_DIR']}      (custom personas + system_core edits)")
    say(f"    - {paths['TTS_VOICES_DIR']}   (Piper voice files)")
    say(f"    - {ROOT / 'Copyright'}        (private-corpus source, if present)")
    say(f"    - {ROOT / 'Agent Ouija.bat'}  (launcher, if present)")
    say(f"    - Desktop shortcut 'Agent Ouija.lnk' (if present)")
    if level in ("keys", "full"):
        say(f"    - {Path.home() / '.config' / 'agentouija' / '.env'}   (cloud API keys, preferred location)")
        say(f"    - {ROOT / '.env'}                    (cloud API keys, legacy location if present)")
    if level == "full":
        say("    - Ollama models pulled by this app: `ollama rm` per model")
        say("      in config.AVAILABLE_MODELS with type='local'")
    say("")
    say("  NOT removed automatically:")
    say(f"    - {ROOT}                       (repo itself — delete manually)")
    say("    - Python packages installed via pip (they may be shared with")
    say("      other projects; `pip uninstall -r requirements.txt` if you")
    say("      want them gone).")


def _confirm(yes: bool) -> bool:
    if yes:
        return True
    say("  Type YES to confirm. Anything else cancels.")
    raw = input("  > ").strip()
    # Case-insensitive match — matches the rest of the app's typed-
    # confirmation dialogs (wipe / rollback / restore all upper()
    # before comparing).
    return raw.upper() == "YES"


# ── Actual deletion ──────────────────────────────────────────────────────────

def wipe_data(paths: dict) -> None:
    """Personal data + generated files. Idempotent — every path is
    checked before removal."""
    header("Removing data + generated files")
    for k in (
        "DATA_DIR", "DOCS_DIR", "PROMPTS_DIR", "TTS_VOICES_DIR",
    ):
        _rm_tree(paths[k])
    _rm_tree(ROOT / "Copyright")
    _rm_file(ROOT / "Agent Ouija.bat")
    _rm_file(ROOT / "Mr. Ouija.bat")           # legacy pre-rebrand name
    _rm_file(Path.home() / "Desktop" / "Agent Ouija.lnk")
    _rm_file(Path.home() / "Desktop" / "Mr. Ouija.lnk")


def wipe_env() -> None:
    """Cloud API keys. Handles the plaintext, encrypted, and
    legacy variants — every possible on-disk representation of
    the API key file is removed."""
    header("Removing .env (cloud API keys)")
    preferred = Path.home() / ".config" / "agentouija" / ".env"
    legacy    = ROOT / ".env"
    _rm_file(preferred)
    _rm_file(preferred.with_suffix(".env.enc"))
    _rm_file(preferred.with_suffix(".env.enc.meta"))
    _rm_file(legacy)
    _rm_file(legacy.with_suffix(".env.enc"))
    _rm_file(legacy.with_suffix(".env.enc.meta"))
    # Also try to remove the ~/.config/agentouija/ directory if
    # nothing else lives there — leaves no trace.
    cfg_dir = preferred.parent
    if cfg_dir.exists():
        try:
            if not any(cfg_dir.iterdir()):
                cfg_dir.rmdir()
                ok(f"Removed empty {cfg_dir}")
        except OSError as e:
            log_line = f"     (skip) {cfg_dir} — {e}"
            print(log_line)


def wipe_ollama_models() -> None:
    """`ollama rm` every local model in config.AVAILABLE_MODELS."""
    header("Removing Ollama models pulled by this app")
    sys.path.insert(0, str(SCRIPTS))
    try:
        from config import AVAILABLE_MODELS   # noqa: E402
    except Exception as e:
        err(f"Could not load config to enumerate models: {e}")
        return
    locals_ = [
        cfg["name"] for cfg in AVAILABLE_MODELS.values()
        if cfg.get("type") == "local" and cfg.get("name")
    ]
    if not locals_:
        warn("No local models declared in config — nothing to remove.")
        return
    if not shutil.which("ollama"):
        warn("Ollama binary not on PATH — skipping. Run "
             "`ollama list` + `ollama rm <model>` manually to free disk.")
        return
    for name in locals_:
        say(f"  >>> ollama rm {name}")
        rc = subprocess.call(["ollama", "rm", name])
        if rc == 0:
            ok(f"Removed Ollama model: {name}")
        else:
            warn(f"`ollama rm {name}` returned exit code {rc} "
                 "(often means the model wasn't installed)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agent Ouija uninstaller — remove personal data + "
                     "generated files + optionally Ollama models."
    )
    parser.add_argument(
        "--level", choices=["data", "keys", "full"],
        help="Skip the interactive prompt: data-only / data+env / full",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the typed-YES confirmation (dangerous — automation only)",
    )
    args = parser.parse_args()

    header("Agent Ouija — Uninstaller")
    say(f"  Repo root: {ROOT}")
    say("")

    paths = _load_paths()

    if args.level:
        level = args.level
    else:
        say("  Uninstall levels:")
        say("    [1] Data-only — wipe generated files, keep .env keys")
        say("    [2] Data + keys — also delete .env")
        say("    [3] Full — above + `ollama rm` on every local model")
        while True:
            raw = input("  Pick 1 / 2 / 3: ").strip()
            level = {"1": "data", "2": "keys", "3": "full"}.get(raw)
            if level:
                break
            say("    Please answer 1, 2 or 3.")

    say("")
    _summarize_removal(level, paths)
    say("")
    if not _confirm(args.yes):
        say("  Cancelled — nothing deleted.")
        return 0

    wipe_data(paths)
    if level in ("keys", "full"):
        wipe_env()
    if level == "full":
        wipe_ollama_models()

    header("Done")
    say("  Agent Ouija's personal data is gone.")
    say("  To finish the uninstall, delete the repo folder itself:")
    say(f"    Remove-Item -Recurse -Force {ROOT}")
    say("  (PowerShell can't delete the folder its script is running from,")
    say("   so this last step has to be manual.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
