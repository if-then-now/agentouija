"""
tts.py — Piper TTS for the Arena tab Voice toggle.

Fully local, free, MIT-licensed. No cloud calls. Uses the same air-gap-
capable design philosophy as the rest of Agent Ouija.

One-time setup (Brainy):
    pip install piper-tts
    # Voice files go under TTS_VOICES_DIR (defaults to BASE_DIR / "voices")
    mkdir -p voices
    cd voices
    # Default voice — rich northern English male, fits Agent Ouija's tone
    curl -L -o en_GB-northern_english_male-medium.onnx \\
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx
    curl -L -o en_GB-northern_english_male-medium.onnx.json \\
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json

To swap voices later, change TTS_VOICE in config.py and drop the matching
.onnx + .onnx.json pair into the voices/ folder. Full voice catalog:
https://rhasspy.github.io/piper-samples/

Behaviour:
    - First call lazy-loads the voice model into memory (~100ms).
    - Failure to load (missing pip dep, missing voice file) is cached so we
      don't re-attempt every slide change. Restart the UI to retry after
      fixing setup.
    - Long responses are truncated to TTS_MAX_CHARS so synth latency stays
      under a few seconds even on long-form Claude answers.
"""

import logging
import re
import tempfile
import unicodedata
import wave
from pathlib import Path
from typing import Optional

from config import ACTIVE_VOICE_PATH, TTS_MAX_CHARS, TTS_VOICE, TTS_VOICES_DIR

log = logging.getLogger(__name__)

# Module-global voice singleton + cached load failure. The failure cache
# means we don't pay the import / file-check cost on every slide change
# after the first failed attempt.
_VOICE: Optional[object]      = None
_LOAD_ERROR: Optional[Exception] = None


def _active_voice_info() -> tuple[str, int]:
    """Return (voice_name, speaker_id) — the tuple that drives which
    Piper voice loads and which speaker inside it is used.

    Persona tab writes a single line to ACTIVE_VOICE_PATH in the format
    "voice_name" (single-speaker) or "voice_name|speaker_id" (multi-
    speaker, e.g. libritts). Missing/empty file OR a name that doesn't
    map to a downloaded voice both fall back to (config.TTS_VOICE, 0).
    Read at synthesis time (not import) so changes take effect on the
    next call — the singleton is invalidated by set_active_voice()."""
    if ACTIVE_VOICE_PATH.exists():
        try:
            content = ACTIVE_VOICE_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        if content:
            if "|" in content:
                name, _, sid = content.partition("|")
                name = name.strip()
                try:
                    speaker_id = int(sid.strip())
                except ValueError:
                    speaker_id = 0
            else:
                name, speaker_id = content, 0
            if name and (TTS_VOICES_DIR / f"{name}.onnx").exists():
                return name, speaker_id
    return TTS_VOICE, 0


def _active_voice_name() -> str:
    """Backwards-compat shim for callers that only want the voice name."""
    return _active_voice_info()[0]


def set_active_voice(voice_name: str, speaker_id: int = 0) -> None:
    """Persist the active voice (and optional speaker index) and drop the
    cached singleton so the next synthesize call reloads. Called from the
    Persona tab's Set-As-Default button."""
    global _VOICE, _LOAD_ERROR
    TTS_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    speaker_id = int(speaker_id or 0)
    content = (f"{voice_name.strip()}|{speaker_id}"
               if speaker_id else voice_name.strip())
    ACTIVE_VOICE_PATH.write_text(content, encoding="utf-8")
    _VOICE = None
    _LOAD_ERROR = None


def _load_voice():
    """Lazy-load the Piper voice. Caches both success and failure."""
    global _VOICE, _LOAD_ERROR
    if _VOICE is not None:
        return _VOICE
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR

    try:
        from piper import PiperVoice  # type: ignore
    except ImportError as e:
        _LOAD_ERROR = ImportError(
            f"piper-tts not installed. Run: pip install piper-tts ({e})"
        )
        raise _LOAD_ERROR

    voice_id, _ = _active_voice_info()
    voice_file  = TTS_VOICES_DIR / f"{voice_id}.onnx"
    config_file = TTS_VOICES_DIR / f"{voice_id}.onnx.json"
    if not voice_file.exists():
        _LOAD_ERROR = FileNotFoundError(
            f"Piper voice file not found: {voice_file}. "
            f"Setup: see scripts/tts.py docstring."
        )
        raise _LOAD_ERROR
    if not config_file.exists():
        # Piper can sometimes work without the .json; warn but proceed.
        log.warning(
            "Voice config not found: %s — Piper may use defaults",
            config_file,
        )

    try:
        _VOICE = PiperVoice.load(str(voice_file))
        log.info("Loaded Piper voice: %s", voice_id)
        return _VOICE
    except Exception as e:
        _LOAD_ERROR = e
        log.warning(
            "Failed to load Piper voice (%s): %s — Voice toggle will be a "
            "no-op until the UI is restarted",
            type(e).__name__, e,
        )
        raise


# ── Text sanitizer for speech ────────────────────────────────────────────────
# Model responses are markdown by default. Piper reads markdown literally —
# "asterisk asterisk bold asterisk asterisk", "hash hash header", etc. —
# which is what causes the symbol-reading + voice-fluctuation symptoms.
# Strip markdown formatting + emoji while preserving the spoken sentence.

# Order matters: code blocks → links → emphasis markers → bullets → headers.
_CODE_FENCE_RE     = re.compile(r"```.*?```",        re.DOTALL)
_INLINE_CODE_RE    = re.compile(r"`([^`]+)`")
_LINK_RE           = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BARE_URL_RE       = re.compile(r"https?://\S+")
_HEADER_RE         = re.compile(r"^#+\s*",            re.MULTILINE)
_BLOCKQUOTE_RE     = re.compile(r"^>+\s*",            re.MULTILINE)
_BULLET_RE         = re.compile(r"^\s*[-*+]\s+",      re.MULTILINE)
_NUMBERED_LIST_RE  = re.compile(r"^\s*\d+\.\s+",      re.MULTILINE)
_EMPHASIS_RE       = re.compile(r"(\*{1,3}|_{1,3}|~~)")
_BRACKETS_RE       = re.compile(r"[\[\]\{\}<>]")
_WHITESPACE_RE     = re.compile(r"\s+")


def _strip_symbols(text: str) -> str:
    """Drop Unicode symbol characters (emoji, dingbats, weird math).
    Keeps letters, numbers, punctuation, whitespace, and currency."""
    out = []
    for c in text:
        cat = unicodedata.category(c)
        # L* letters, N* numbers, P* punctuation, Z* separators, Sc currency
        if cat[0] in ("L", "N", "P", "Z") or cat == "Sc":
            out.append(c)
        elif c in " \n\t":
            out.append(c)
        # else: Symbol (So/Sm/Sk), Control (Cc), Format (Cf) — drop
    return "".join(out)


def clean_for_speech(text: str) -> str:
    """Strip markdown and noisy symbols so Piper reads natural prose.

    Heuristic — keeps sentence structure (commas, periods, question marks)
    but removes formatting markers Piper would otherwise verbalize. Code
    blocks are dropped entirely because reading source code aloud is
    actively unhelpful.

    Substitutes "Q:" → "Question:" and "A:" → "Answer:" so the reader
    doesn't say the letter aloud when a response quotes chat-style
    Q&A. Applied only at word boundaries so the middle of a word (e.g.
    "IQ:") isn't hit.
    """
    if not text:
        return ""

    text = _CODE_FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)         # [label](url) → label
    text = _BARE_URL_RE.sub(" ", text)        # bare URLs → silence
    text = _HEADER_RE.sub("", text)           # # Heading → Heading
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _BULLET_RE.sub(", ", text)         # bullet markers → soft pause
    text = _NUMBERED_LIST_RE.sub(", ", text)
    text = _EMPHASIS_RE.sub("", text)         # **bold** / *em* / ~~strike~~
    text = _BRACKETS_RE.sub(" ", text)
    # Q:/A: → Question:/Answer: — spoken form of the chat abbrev.
    text = re.sub(r"(?<!\w)Q:", "Question:", text)
    text = re.sub(r"(?<!\w)A:", "Answer:", text)
    text = _strip_symbols(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ── Synthesis config — steadier prosody ──────────────────────────────────────
# Piper's default noise_scale (~0.667) and noise_w_scale (~0.8) make the voice
# expressive but wobble across sentence boundaries — the "voice changes here
# and there and then corrects" symptom on the medium-tier northern_english_male
# voice. Lower values give a steadier delivery at the cost of some liveliness.
#
# The class name and kwargs changed across piper-tts versions:
#   1.2+   → piper.SynthesisConfig(noise_scale=..., noise_w_scale=...)
#   1.1.x  → piper.SynthesisConfig(noise_scale=..., noise_w=...)
#   older  → no SynthesisConfig at all → we fall back to Piper defaults
# Set either value to None to fully disable SynthesisConfig injection and
# let Piper use its defaults (currently the case — the high-quality voices
# don't drift the way the medium northern_english_male did, so no override
# needed). Framework kept in place so re-tuning is a one-line change.
_STEADY_NOISE_SCALE   = None
_STEADY_NOISE_W_SCALE = None

_SYN_CONFIG_CACHE   = None
_SYN_CONFIG_PROBED  = False


def _get_synthesis_config():
    """Return a cached SynthesisConfig instance or None. Probes the installed
    piper-tts once for the correct kwargs; result is cached for the process."""
    global _SYN_CONFIG_CACHE, _SYN_CONFIG_PROBED
    if _SYN_CONFIG_PROBED:
        return _SYN_CONFIG_CACHE
    _SYN_CONFIG_PROBED = True

    # Explicit opt-out: if either tuning knob is None, skip the config
    # entirely and let Piper use its natural defaults.
    if _STEADY_NOISE_SCALE is None or _STEADY_NOISE_W_SCALE is None:
        return None

    try:
        from piper import SynthesisConfig  # type: ignore
    except ImportError:
        log.debug("piper.SynthesisConfig not available — using Piper defaults")
        return None

    # 1.2+ uses noise_w_scale; 1.1.x uses noise_w. Try both.
    for kw_name in ("noise_w_scale", "noise_w"):
        try:
            _SYN_CONFIG_CACHE = SynthesisConfig(
                **{"noise_scale": _STEADY_NOISE_SCALE, kw_name: _STEADY_NOISE_W_SCALE}
            )
            log.info(
                "Piper SynthesisConfig: noise_scale=%s %s=%s",
                _STEADY_NOISE_SCALE, kw_name, _STEADY_NOISE_W_SCALE,
            )
            return _SYN_CONFIG_CACHE
        except TypeError:
            continue
        except Exception as e:
            log.debug("SynthesisConfig probe failed (%s): %s", type(e).__name__, e)
            return None
    return None


def _call_with_optional_syn_config(fn, *args):
    """Invoke fn(*args, syn_config=<cached>) if the current Piper accepts it,
    else fn(*args). On TypeError from the syn_config kwarg, invalidate the
    cache so future calls skip the probe cost and go straight to defaults."""
    global _SYN_CONFIG_CACHE
    cfg = _get_synthesis_config()
    if cfg is None:
        fn(*args)
        return
    try:
        fn(*args, syn_config=cfg)
    except TypeError:
        # Method doesn't accept syn_config kwarg — drop it permanently.
        _SYN_CONFIG_CACHE = None
        fn(*args)


def _sample_rate(voice) -> int:
    """Best-effort sample rate lookup. Piper voices ship with this in their
    .onnx.json config under various attribute names across versions.
    Falls back to 22050 (the most common Piper default)."""
    sr = getattr(voice, "sample_rate", None)
    if sr:
        return int(sr)
    config = getattr(voice, "config", None)
    if config is not None:
        sr = getattr(config, "sample_rate", None)
        if sr:
            return int(sr)
        sample_cfg = getattr(config, "audio", None)
        if sample_cfg is not None:
            sr = getattr(sample_cfg, "sample_rate", None)
            if sr:
                return int(sr)
    return 22050


# A WAV file with just headers and no audio frames is ~44 bytes. Anything
# at-or-below this means "no audio actually written". This is the key
# bug-fix sentinel — Piper's post-1.2 API accepts (text, wav_file) but
# treats the wav_file argument as a SynthesisConfig and writes nothing.
# Without this validation we'd return an empty-header WAV as "success".
_MIN_AUDIO_BYTES = 100


def _speaker_kwargs(speaker_id: int) -> dict:
    """Kwargs to try on Piper synthesize calls for multi-speaker voices.
    Empty dict when speaker_id is 0 (single-speaker) — the common path,
    avoiding a TypeError retry for every synth call."""
    return {"speaker_id": int(speaker_id)} if speaker_id else {}


def _try_synthesize_wav_method(voice, text: str, wav_path: str,
                                speaker_id: int) -> bool:
    """Piper 1.2+ convenience: voice.synthesize_wav(text, wav_file). Sets
    the WAV params and writes frames itself. The cleanest path when the
    installed Piper has it. For multi-speaker voices, speaker_id is passed
    as a kwarg; TypeError → retry without so single-speaker voices don't
    break on installations that don't accept the kwarg."""
    if not hasattr(voice, "synthesize_wav"):
        return False
    sp_kwargs = _speaker_kwargs(speaker_id)
    with wave.open(wav_path, "wb") as wav_file:
        try:
            _call_with_optional_syn_config(
                voice.synthesize_wav, text, wav_file, **sp_kwargs
            )
        except TypeError:
            _call_with_optional_syn_config(voice.synthesize_wav, text, wav_file)
    return Path(wav_path).stat().st_size > _MIN_AUDIO_BYTES


def _try_chunk_iterator(voice, text: str, wav_path: str,
                         speaker_id: int) -> bool:
    """Piper 1.2+ primary API: voice.synthesize(text) returns an iterable
    of AudioChunk objects with their own sample metadata. Collect chunks
    and write the WAV ourselves."""
    cfg = _get_synthesis_config()
    sp_kwargs = _speaker_kwargs(speaker_id)
    try:
        if cfg is not None:
            try:
                chunks = list(voice.synthesize(text, syn_config=cfg, **sp_kwargs))
            except TypeError:
                try:
                    chunks = list(voice.synthesize(text, **sp_kwargs))
                except TypeError:
                    chunks = list(voice.synthesize(text))
        else:
            try:
                chunks = list(voice.synthesize(text, **sp_kwargs))
            except TypeError:
                chunks = list(voice.synthesize(text))
    except TypeError:
        # Some legacy builds REQUIRE a second positional arg — they're
        # not chunk-iterator-style.
        return False
    if not chunks:
        return False
    # If chunks aren't AudioChunk-shaped, this isn't the right API.
    first = chunks[0]
    if not hasattr(first, "audio_int16_array"):
        return False
    channels    = int(getattr(first, "sample_channels", 1))
    sample_w    = int(getattr(first, "sample_width",   2))
    sample_rate = int(getattr(first, "sample_rate",    _sample_rate(voice)))
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_w)
        wav_file.setframerate(sample_rate)
        for chunk in chunks:
            data = getattr(chunk, "audio_int16_array", None)
            if data is None:
                continue
            wav_file.writeframes(data.tobytes())
    return Path(wav_path).stat().st_size > _MIN_AUDIO_BYTES


def _try_legacy_synthesize(voice, text: str, wav_path: str,
                            speaker_id: int) -> bool:
    """Pre-1.2 Piper API: voice.synthesize(text, wave_file). Some builds
    set the WAV params internally, others don't — pre-set them either way.
    Legacy Piper typically predates SynthesisConfig, so we don't attempt
    to pass one here. Multi-speaker voices tried with the speaker_id
    kwarg first; TypeError → retry without."""
    sr = _sample_rate(voice)
    sp_kwargs = _speaker_kwargs(speaker_id)
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        try:
            voice.synthesize(text, wav_file, **sp_kwargs)
        except TypeError:
            voice.synthesize(text, wav_file)
    return Path(wav_path).stat().st_size > _MIN_AUDIO_BYTES


def synth_to_wav(text: Optional[str]) -> Optional[Path]:
    """Synthesize text to a temporary WAV file. Returns path or None.

    Never raises — failures are logged and surfaced as None so the caller
    (the Arena slideshow handlers) can degrade gracefully without
    crashing the UI on a missing voice / pip dep.

    Truncates input to TTS_MAX_CHARS so synth latency stays bounded on
    long-form responses. The Arena slideshow already shows the full text
    in the response box; the voice is a UX enhancement, not a verbatim
    read-aloud transcript.

    Tries multiple Piper API shapes in order of preference:
        1. voice.synthesize_wav(text, wav)   — cleanest, 1.2+
        2. voice.synthesize(text) chunks      — primary 1.2+ API
        3. voice.synthesize(text, wav)        — legacy, pre-1.2
    Each attempt is validated by checking that the resulting WAV file is
    larger than just a WAV header — otherwise we'd silently "succeed"
    with an empty-audio file (the bug Brainy hit on first ship).
    """
    if not text or not text.strip():
        return None

    # Strip markdown / emoji / code BEFORE truncating so we don't waste the
    # char budget on formatting noise that wouldn't be spoken anyway.
    text = clean_for_speech(text)
    if not text:
        return None

    if len(text) > TTS_MAX_CHARS:
        text = text[:TTS_MAX_CHARS].rstrip() + "… response truncated for speech."

    try:
        voice = _load_voice()
    except Exception:
        return None

    _, speaker_id = _active_voice_info()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wav_path = tmp.name

    attempts = (
        ("synthesize_wav",      _try_synthesize_wav_method),
        ("chunk-iterator",      _try_chunk_iterator),
        ("legacy synthesize",   _try_legacy_synthesize),
    )

    errors: list[str] = []
    for name, fn in attempts:
        try:
            if fn(voice, text, wav_path, speaker_id):
                log.debug("TTS produced audio via %s API", name)
                return Path(wav_path)
            errors.append(f"{name}: returned no audio")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            log.debug("Piper %s API failed (%s)", name, e)

    log.warning("TTS synth failed — all Piper APIs unproductive: %s",
                "; ".join(errors))
    return None
