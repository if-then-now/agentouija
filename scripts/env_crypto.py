"""env_crypto.py — Fernet-based encryption for the .env file.

Store shape:
    <.env location>          ← plaintext, unencrypted (present ONLY
                                when the user opted out of encryption)
    <.env location>.enc      ← Fernet ciphertext blob
    <.env location>.enc.meta ← JSON with the KDF salt + iterations

The passphrase is stretched to a Fernet key via PBKDF2-HMAC-SHA256
with a random 16-byte salt. Iterations pinned at 200_000 — high
enough that a laptop-loss brute-force takes real time, low enough
that startup decrypt feels instant.

Callers (install.py, arena.py, arena_ui.py) never see the derived
key. They pass the passphrase in, this module hands back plaintext
or a status dict. Errors surface as exceptions from cryptography;
callers translate them into friendly UI / CLI messages.

Companion: install.py --encrypt-env / --decrypt-env toggle between
plaintext and ciphertext modes after the initial install.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional


# Bumped only if we ever change the KDF params. Loaded from meta on
# decrypt so old files with fewer iterations still work.
_KDF_ITERATIONS = 200_000
_KDF_SALT_BYTES = 16


def _derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    """PBKDF2-HMAC-SHA256 → 32-byte key → urlsafe-base64 for Fernet."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def encrypt(plaintext: str, passphrase: str) -> tuple[bytes, dict]:
    """Encrypt env-file text with the given passphrase. Returns
    (ciphertext_bytes, meta_dict). Caller writes ciphertext to
    <path>.enc and json.dumps(meta) to <path>.enc.meta."""
    from cryptography.fernet import Fernet

    salt = os.urandom(_KDF_SALT_BYTES)
    key  = _derive_key(passphrase, salt, _KDF_ITERATIONS)
    token = Fernet(key).encrypt(plaintext.encode("utf-8"))
    meta = {
        "kdf":        "PBKDF2-HMAC-SHA256",
        "iterations": _KDF_ITERATIONS,
        "salt":       base64.b64encode(salt).decode("ascii"),
        "cipher":     "Fernet",
    }
    return token, meta


def decrypt(ciphertext: bytes, meta: dict, passphrase: str) -> str:
    """Reverse of encrypt(). Raises cryptography.fernet.InvalidToken
    on bad passphrase or tampered ciphertext — caller catches for a
    friendly retry-loop message."""
    from cryptography.fernet import Fernet

    salt = base64.b64decode(meta["salt"])
    iterations = int(meta.get("iterations", _KDF_ITERATIONS))
    key = _derive_key(passphrase, salt, iterations)
    return Fernet(key).decrypt(ciphertext).decode("utf-8")


# ── Helpers for install / runtime callers ────────────────────────────────────

def ciphertext_paths(env_path: Path) -> tuple[Path, Path]:
    """Return (ciphertext_path, meta_path) alongside a given .env
    location. Both callers (install-time write, runtime read) use
    this to stay in sync."""
    return (env_path.with_suffix(env_path.suffix + ".enc"),
            env_path.with_suffix(env_path.suffix + ".enc.meta"))


def is_encrypted_at(env_path: Path) -> bool:
    """True when a .env.enc + .env.enc.meta pair sits next to the
    given path AND the plaintext file itself is absent. Callers
    use this to decide whether to prompt for a passphrase."""
    ct_path, meta_path = ciphertext_paths(env_path)
    return (ct_path.exists() and meta_path.exists()
            and not env_path.exists())


def write_encrypted(env_path: Path, plaintext: str, passphrase: str) -> None:
    """Encrypt plaintext, write .env.enc + .env.enc.meta next to
    env_path, delete the plaintext file if it exists. Idempotent
    — safe to re-run to rotate passphrases."""
    ct_path, meta_path = ciphertext_paths(env_path)
    ct, meta = encrypt(plaintext, passphrase)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    ct_path.write_bytes(ct)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if env_path.exists():
        try:
            env_path.unlink()
        except OSError:
            pass


def read_encrypted(env_path: Path, passphrase: str) -> Optional[str]:
    """Return the decrypted plaintext, or None if the ciphertext
    pair isn't there. Passphrase failure raises — caller catches
    InvalidToken for retry loops."""
    ct_path, meta_path = ciphertext_paths(env_path)
    if not (ct_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ct   = ct_path.read_bytes()
    return decrypt(ct, meta, passphrase)


def load_into_environ(plaintext: str) -> int:
    """Parse env-format plaintext (KEY=VALUE lines, # comments
    allowed) and set os.environ entries. Returns count of vars
    set. Matches python-dotenv's behaviour closely enough for our
    use — no export= / interpolation / multi-line values, which
    .env for this project doesn't use anyway."""
    n = 0
    for raw in plaintext.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if v and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        os.environ[k] = v
        n += 1
    return n
