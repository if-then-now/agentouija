# log_parser.py
import logging
import re
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Represents a single Captain's Log entry with metadata."""
    date: str
    title: str
    body: str
    tags: list[str]
    thread: str = ""
    label: str = ""


class LogParser:
    """
    Parses Captain's Log markdown/text files into structured entries.
    Single Responsibility: owns all log parsing logic.
    Each entry becomes a separate document chunk with metadata for ChromaDB.
    """

    ENTRY_PATTERN = re.compile(
        r'\[(?P<date>[^\]]+)\].*?(?P<title>[^\n]+)\n(?P<body>.*?)(?=\n---|\Z)',
        re.DOTALL
    )
    TAGS_PATTERN = re.compile(r'Tags:\s*(.+)')
    THREAD_PATTERN = re.compile(r'Thread:\s*(.+)')

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def parse(self) -> list[LogEntry]:
        """Parse log file into a list of LogEntry objects."""
        if not self.file_path.exists():
            logger.warning(f"Log file not found: {self.file_path}")
            return []

        try:
            content = self.file_path.read_text(encoding="utf-8")
            logger.info(f"Parsing log file: {self.file_path.name}")
        except Exception as e:
            logger.error(f"Failed to read log file: {e}")
            return []

        thread = self._extract_thread(content)
        entries = []

        for match in self.ENTRY_PATTERN.finditer(content):
            date = match.group("date").strip()
            title = match.group("title").strip()
            body = match.group("body").strip()

            tags = self._extract_tags(body)
            clean_body = self.TAGS_PATTERN.sub("", body).strip()

            entry = LogEntry(
                date=date,
                title=title,
                body=clean_body,
                tags=tags,
                thread=thread
            )
            entries.append(entry)
            logger.debug(f"Parsed entry: {title} | tags: {tags}")

        logger.info(f"Parsed {len(entries)} entries from {self.file_path.name}")
        return entries

    def _extract_tags(self, text: str) -> list[str]:
        """Extract tags from entry text."""
        match = self.TAGS_PATTERN.search(text)
        if match:
            return [t.strip() for t in match.group(1).split(",")]
        return []

    def _extract_thread(self, content: str) -> str:
        """Extract thread name from log header."""
        match = self.THREAD_PATTERN.search(content)
        return match.group(1).strip() if match else ""