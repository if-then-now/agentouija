# ingest.py
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from config import (
    DOCS_DIR,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBED_MODEL_NAME
)
from log_parser import LogParser

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Captain's Log filename patterns
LOG_FILE_PATTERNS = ["captain", "log", "master"]


def configure_embedding_model() -> None:
    """Configure local HuggingFace embedding model — no OpenAI, no cloud."""
    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        logger.info(f"Embedding model loaded: {EMBED_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def is_captain_log(file_path: Path) -> bool:
    """Detect Captain's Log files by filename patterns."""
    name_lower = file_path.name.lower()
    return any(pattern in name_lower for pattern in LOG_FILE_PATTERNS)


def is_tagged_markdown(file_path: Path) -> bool:
    """Detect TAGS_ prefixed markdown files for section-level tag extraction."""
    return file_path.name.startswith("TAGS_") and file_path.suffix == ".md"


def load_log_documents(file_path: Path) -> list[Document]:
    """Parse a Captain's Log file into tagged LlamaIndex Documents."""
    parser = LogParser(file_path)
    entries = parser.parse()
    documents = []

    ingest_ts = datetime.now().isoformat(timespec="seconds")
    for entry in entries:
        doc = Document(
            text=f"{entry.title}\n\n{entry.body}",
            metadata={
                "date": entry.date,
                "title": entry.title,
                "tags": ", ".join(entry.tags),
                "thread": entry.thread,
                "source": file_path.name,
                "type": "captain_log",
                # created_at is when THIS ingest ran, not entry.date —
                # the rollback control ages by "when did it enter the
                # DB" so re-ingesting old material always keeps it.
                "created_at": ingest_ts,
            }
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} tagged documents from {file_path.name}")
    return documents


def load_tagged_markdown(file_path: Path) -> list[Document]:
    """
    Parse a TAGS_ prefixed markdown file into per-section tagged Documents.
    Each ## section becomes its own chunk with extracted Tags: metadata.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read {file_path.name}: {e}")
        return []

    # Split by ## headings
    sections = re.split(r'\n(?=## )', content)
    documents = []
    tags_pattern = re.compile(r'Tags:\s*(.+)', re.IGNORECASE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract tags line
        tags_match = tags_pattern.search(section)
        tags = []
        if tags_match:
            tags = [t.strip() for t in tags_match.group(1).split(",")]
            # Remove tags line from body text
            section_text = tags_pattern.sub("", section).strip()
        else:
            section_text = section

        # Extract heading as title
        lines = section_text.split("\n")
        title = lines[0].replace("##", "").strip() if lines else "Untitled"
        body = "\n".join(lines[1:]).strip()

        doc = Document(
            text=f"{title}\n\n{body}",
            metadata={
                "title": title,
                "tags": ", ".join(tags),
                "source": file_path.name,
                "type": "tagged_markdown",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} tagged section documents from {file_path.name}")
    return documents


def load_all_documents(docs_dir: Path) -> list[Document]:
    """
    Load all documents from docs directory.
    Captain's Logs → LogParser (entry-level tagging)
    TAGS_ markdown files → section-level tag extraction
    Everything else → standard SimpleDirectoryReader
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    all_documents = []

    for file_path in docs_dir.iterdir():
        if not file_path.is_file():
            continue

        if is_captain_log(file_path):
            logger.info(f"Detected Captain's Log: {file_path.name}")
            docs = load_log_documents(file_path)
            all_documents.extend(docs)

        elif is_tagged_markdown(file_path):
            logger.info(f"Detected tagged markdown: {file_path.name}")
            docs = load_tagged_markdown(file_path)
            all_documents.extend(docs)

        else:
            logger.info(f"Loading standard document: {file_path.name}")
            try:
                docs = SimpleDirectoryReader(
                    input_files=[str(file_path)]
                ).load_data()
                # Stamp created_at so these chunks are visible to the
                # rollback-by-age control. SimpleDirectoryReader doesn't
                # set it; we backfill after load.
                stamp = datetime.now().isoformat(timespec="seconds")
                for d in docs:
                    d.metadata.setdefault("created_at", stamp)
                    d.metadata.setdefault("source", file_path.name)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Could not load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def build_index(documents: list, chroma_collection) -> VectorStoreIndex:
    """Build and persist vector index into ChromaDB."""
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    logger.info("Index built and persisted to ChromaDB.")
    return index


def run_ingestion() -> dict:
    """Run the full ingestion pipeline. Returns a result dict.

    Suitable for both CLI (via main()) and in-process callers (Gradio UI).
    Never raises — errors are logged and surfaced in result["message"].

    Result schema:
        {"status": "success"|"warning"|"error",
         "message": str,
         "documents": int}
    """
    result = {"status": "success", "message": "", "documents": 0}

    try:
        configure_embedding_model()
    except Exception as e:
        result.update(status="error", message=f"Failed to load embedding model: {e}")
        return result

    try:
        documents = load_all_documents(DOCS_DIR)
    except FileNotFoundError as e:
        result.update(status="error", message=f"Docs directory not found: {e}")
        return result
    except Exception as e:
        result.update(status="error", message=f"Failed to load documents: {e}")
        return result

    if not documents:
        result.update(status="warning",
                      message=f"No documents found in {DOCS_DIR}")
        return result
    result["documents"] = len(documents)

    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Dedup strategy: drop the collection before re-indexing. Every ingest
        # run produces a clean rebuild from docs/, so clicking the UI button
        # twice (or running this CLI twice) can never create duplicate chunks.
        try:
            existing = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
            prior_count = existing.count()
            chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info(
                "Cleared existing collection '%s' (%d chunks) before rebuild",
                CHROMA_COLLECTION_NAME, prior_count,
            )
        except Exception:
            # Collection didn't exist yet — first ever ingest, fine.
            logger.info("No prior collection '%s' to clear", CHROMA_COLLECTION_NAME)
        chroma_collection = chroma_client.create_collection(CHROMA_COLLECTION_NAME)
        logger.info(f"Connected to ChromaDB collection: {CHROMA_COLLECTION_NAME} at {CHROMA_DIR}")
    except Exception as e:
        result.update(status="error", message=f"ChromaDB connection failed: {e}")
        return result

    try:
        build_index(documents, chroma_collection)
    except Exception as e:
        result.update(status="error", message=f"Index build failed: {e}")
        return result

    result["message"] = (
        f"Ingested {len(documents)} documents into collection "
        f"'{CHROMA_COLLECTION_NAME}' at {CHROMA_DIR}"
    )
    return result


def main():
    logger.info("Starting Agent Ouija ingestion pipeline...")
    result = run_ingestion()
    if result["status"] == "error":
        logger.error(result["message"])
        sys.exit(1)
    logger.info(result["message"])
    if result["status"] == "warning":
        sys.exit(2)
    logger.info("Ingestion complete. Agent Ouija's memory is updated.")


if __name__ == "__main__":
    main()