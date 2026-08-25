# memory.py
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from config import DATA_DIR

logger = logging.getLogger(__name__)

MEMORY_FILE = DATA_DIR / "chat_history.json"


def load_history() -> list:
    """Load conversation history from disk."""
    if not MEMORY_FILE.exists():
        logger.info("No existing chat history found. Starting fresh.")
        return []
    try:
        history = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(history)} messages from chat history.")
        return history
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
        return []


def save_history(history: list) -> None:
    """Save conversation history to disk."""
    try:
        MEMORY_FILE.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Saved {len(history)} messages to chat history.")
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")


def append_exchange(history: list, question: str, response: str) -> list:
    """Append a question/response pair to history with timestamp."""
    history.append({
        "timestamp": datetime.now().isoformat(),
        "role": "user",
        "content": question
    })
    history.append({
        "timestamp": datetime.now().isoformat(),
        "role": "assistant",
        "content": response
    })
    return history