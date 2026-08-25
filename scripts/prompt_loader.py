# prompt_loader.py
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import (
    CUSTOM_ACTIVE_FLAG,
    CUSTOM_PROMPTS_DIR,
    DEFAULT_PERSONA_MODE,
    ENHANCED_ACTIVE_FLAG,
    EPISODES_DIR,
    MAX_ENHANCED_STACK_CHARS,
    PERSONA_MODES,
    PERSONA_NAMES,
    PROMPTS_DIR,
)

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Loads and assembles prompt files in configured order.
    Single Responsibility: owns all prompt loading logic.
    Open/Closed: add new prompt files via the factory below.
    """

    def __init__(self, prompt_files: list[Path]):
        self.prompt_files = prompt_files

    def load(self) -> str:
        """Load and concatenate all prompt files in order."""
        sections = []
        for prompt_file in self.prompt_files:
            try:
                content = prompt_file.read_text(encoding="utf-8")
                sections.append(content.strip())
                logger.info(f"Loaded prompt file: {prompt_file.name}")
            except FileNotFoundError:
                logger.warning(f"Prompt file not found, skipping: {prompt_file}")
            except Exception as e:
                logger.error(f"Failed to load {prompt_file.name}: {e}")
                raise
        return "\n\n---\n\n".join(sections)


def _is_custom_active() -> bool:
    """True when the user has flipped 'Use custom persona' in the Persona
    tab. The tab creates CUSTOM_ACTIVE_FLAG when the toggle turns on and
    deletes it when it turns off. Absence = default mystical Agent Ouija."""
    return CUSTOM_ACTIVE_FLAG.exists()


def _custom_files() -> list[Path]:
    """Sorted list of *.md files the user has uploaded into
    prompts/custom/. Empty when the folder doesn't exist yet or holds
    nothing — treated as 'no custom override' at the caller."""
    if not CUSTOM_PROMPTS_DIR.is_dir():
        return []
    return sorted(CUSTOM_PROMPTS_DIR.glob("*.md"))


def _base_files() -> list[Path]:
    """The 'always-on' persona layer every model gets.

    Default: system_core.md.
    Custom override active: files in prompts/custom/ fully replace
      system_core.md.
    In both cases we ALSO layer prompts/enhanced_persona_main.md on top
    when it exists — that's the one enhanced-stack file locals get too
    (small enough to fit in every prompt, defines the shared
    character floor). PromptLoader.load() silently skips it when it
    isn't uploaded.
    """
    if _is_custom_active():
        customs = _custom_files()
        if customs:
            files = list(customs)
        else:
            files = [PROMPTS_DIR / "system_core.md"]
    else:
        files = [PROMPTS_DIR / "system_core.md"]
    files.append(PROMPTS_DIR / "enhanced_persona_main.md")
    return files


def _file_size_or_zero(p: Path) -> int:
    try:
        return p.stat().st_size if p.exists() else 0
    except OSError:
        return 0


def _enhanced_files() -> list[Path]:
    """base + persona_<name>.md + episodes/<name>_*.md for each name.

    CLOUD-ONLY layer — this is where the per-persona files
    (persona_<name>.md for each name in PERSONA_NAMES) and any matching
    episode transcripts get bolted onto the base. Quantized locals tend
    to fall out of character partway through a long persona stack, so
    `get_prompt_loader(..., is_local=True)` short-circuits to
    `_base_files()` regardless of persona_mode / global flag.

    Auto-trim: if the total on-disk size of the stack exceeds
    MAX_ENHANCED_STACK_CHARS, episode files are dropped first (they're
    the biggest and least essential — reference material rather than
    character floor). Prevents 173K-token prompts blowing past provider
    context / TPM limits. Set MAX_ENHANCED_STACK_CHARS to None in
    config.py to disable.
    """
    files: list[Path] = list(_base_files())
    persona_files: list[Path] = []
    episode_files: list[Path] = []
    for name in PERSONA_NAMES:
        persona_files.append(PROMPTS_DIR / f"persona_{name}.md")
        if EPISODES_DIR.is_dir():
            episode_files.extend(sorted(EPISODES_DIR.glob(f"{name}_*.md")))

    if MAX_ENHANCED_STACK_CHARS is None:
        return files + persona_files + episode_files

    base_size    = sum(_file_size_or_zero(p) for p in files)
    persona_size = sum(_file_size_or_zero(p) for p in persona_files)
    episode_size = sum(_file_size_or_zero(p) for p in episode_files)
    total = base_size + persona_size + episode_size

    if total <= MAX_ENHANCED_STACK_CHARS:
        return files + persona_files + episode_files

    # Over budget — drop episode files first.
    logger.warning(
        "Enhanced stack %.0f KB > cap %.0f KB — dropping %d episode "
        "file(s) (%.0f KB). Move those files to Copyright/episodes/ "
        "and enable Use Private Corpus in Agent Ouija's Persona tab "
        "to keep the content available via retrieval instead of "
        "always-on system prompt.",
        total / 1024, MAX_ENHANCED_STACK_CHARS / 1024,
        len(episode_files), episode_size / 1024,
    )
    if (base_size + persona_size) <= MAX_ENHANCED_STACK_CHARS:
        return files + persona_files

    # Still over budget even without episodes — drop persona_<name>.md too.
    logger.warning(
        "Enhanced base + persona files still %.0f KB — also dropping "
        "%d persona_<name>.md file(s) (%.0f KB). Consider trimming "
        "enhanced_persona_main.md.",
        (base_size + persona_size) / 1024,
        len(persona_files), persona_size / 1024,
    )
    return files


def get_prompt_loader(persona_mode: str = DEFAULT_PERSONA_MODE,
                      is_local: bool = False,
                      force_base: bool = False) -> PromptLoader:
    """
    Return a PromptLoader for the given persona mode.

    persona_mode:
        "generic"  — just the base stack (system_core.md +
                     enhanced_persona_main.md if uploaded).
        "enhanced" — base + persona_<name>.md + episodes. Cloud only —
                     locals never receive this even when the global
                     ENHANCED_ACTIVE_FLAG file exists.

    is_local:
        True  — force base only regardless of persona_mode or the
                global flag. Every local model call passes this.
        False — cloud path. persona_mode + ENHANCED_ACTIVE_FLAG decide.

    force_base:
        True  — force base only for a CLOUD model that opts out of the
                enhanced stack (e.g. Groq, whose TPM cap makes the
                persona overlay push peer calls past a guaranteed
                429). Set by `skip_enhanced_persona: True` in the
                model's config.
        False — normal cloud-path resolution.

    Unknown mode → falls back to "generic" with a warning, so a typo in
    config.AVAILABLE_MODELS never blocks arena startup.
    """
    if persona_mode not in PERSONA_MODES:
        logger.warning(
            "Unknown persona_mode %r — falling back to %r",
            persona_mode, DEFAULT_PERSONA_MODE,
        )
        persona_mode = DEFAULT_PERSONA_MODE

    if is_local or force_base:
        # Locals never get the enhanced stack — per user design decision.
        # force_base is the per-cloud-model opt-out; same effect.
        return PromptLoader(prompt_files=_base_files())

    # Cloud path. Global enhanced flag OR the model's own persona
    # setting both mean "give this cloud model the full stack."
    if ENHANCED_ACTIVE_FLAG.exists() or persona_mode == "enhanced":
        return PromptLoader(prompt_files=_enhanced_files())

    return PromptLoader(prompt_files=_base_files())
