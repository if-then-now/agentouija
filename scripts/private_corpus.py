"""
private_corpus.py — Manage the gitignored ChromaDB collection of copyrighted
material (episode transcripts, enhanced persona files, etc.).

Lives outside the main RAG collection so:
  - The source files in Copyright/ never reach the repo (gitignored)
  - The collection itself sits under data/chroma/ (also gitignored)
  - Retrieval at query time is gated by USE_PRIVATE_CORPUS + each model's
    per-model "private_corpus" flag in AVAILABLE_MODELS

Filename convention for ingest:
    Copyright/episodes/<persona>_<anything>.md
    where <persona> is one of your configured PERSONA_NAMES from config.py.
    Anything else (or anything when PERSONA_NAMES is empty) is tagged
    persona="unknown" — still ingested, still retrievable, just doesn't
    bias toward a specific persona.

Operations:
    --ingest          Scan Copyright/episodes/, chunk every .md, embed,
                      write to PRIVATE_CORPUS_COLLECTION_NAME. Idempotent:
                      drops the existing collection first so repeated runs
                      can't double-chunk.
    --status          Per-persona chunk counts in the private collection.
    --export [path]   Dump the collection (ids, documents, metadatas,
                      embeddings) to a portable JSON file. Default path:
                      data/private_corpus_<YYYYMMDD_HHMMSS>.json
    --import <file>   Read a JSON dump and replace the private collection
                      on this machine. Use this on Brainy after copying
                      the export over.
    --wipe            Drop the private collection. No source files touched.

Migration workflow (dev → Brainy):
    # On Big Ouija (dev)
    python scripts/private_corpus.py --ingest
    python scripts/private_corpus.py --export   # → data/private_corpus_*.json
    # copy the JSON to Brainy by whatever means (RDP, Tailscale, USB)
    # On Brainy
    python scripts/private_corpus.py --import data/private_corpus_*.json
    # Toggle USE_PRIVATE_CORPUS = True in config.py and (optionally) flip
    # individual models' "private_corpus" flags.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent))

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_DIR,
    DATA_DIR,
    EMBED_MODEL_NAME,
    PERSONA_NAMES,
    PRIVATE_CORPUS_COLLECTION_NAME,
    PRIVATE_CORPUS_EPISODES_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

EXPORT_FORMAT_VERSION = 1
UNKNOWN_PERSONA       = "unknown"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _persona_for_filename(name: str) -> str:
    """Map filename prefix to a persona in PERSONA_NAMES, or 'unknown'.

    e.g. if PERSONA_NAMES = ("wizard", "advisor"):
         wizard_council.md → 'wizard'
         advisor_pilot.md  → 'advisor'
         misc_lore.md      → 'unknown'
    """
    lower = name.lower()
    for persona in PERSONA_NAMES:
        if lower.startswith(f"{persona}_") or lower.startswith(f"{persona}-"):
            return persona
    return UNKNOWN_PERSONA


def _open_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _configure_embed_model() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest() -> dict:
    """Drop and rebuild the private collection from Copyright/episodes/."""
    result = {"status": "success", "files": 0, "chunks": 0,
              "per_persona": {}, "message": ""}

    if not PRIVATE_CORPUS_EPISODES_DIR.exists():
        result.update(
            status="error",
            message=f"Source dir not found: {PRIVATE_CORPUS_EPISODES_DIR}",
        )
        return result

    files = sorted(p for p in PRIVATE_CORPUS_EPISODES_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in (".md", ".txt"))
    if not files:
        result.update(
            status="warning",
            message=f"No .md / .txt files found in {PRIVATE_CORPUS_EPISODES_DIR}",
        )
        return result

    try:
        _configure_embed_model()
    except Exception as e:
        result.update(status="error", message=f"Embed model load failed: {e}")
        return result

    client = _open_client()
    # Drop-then-create so re-ingest is always a clean rebuild.
    try:
        client.delete_collection(PRIVATE_CORPUS_COLLECTION_NAME)
        log.info("Dropped existing collection '%s'", PRIVATE_CORPUS_COLLECTION_NAME)
    except Exception:
        log.info("No prior '%s' to drop — first ingest",
                 PRIVATE_CORPUS_COLLECTION_NAME)
    chroma_collection = client.create_collection(PRIVATE_CORPUS_COLLECTION_NAME)

    # Build Documents with per-file persona tagging, then let llama_index do
    # the chunking + embedding via the existing ChromaVectorStore plumbing —
    # same pipeline as the main RAG ingest, just a different collection.
    documents = []
    per_persona_files: dict[str, int] = {}
    for f in files:
        persona = _persona_for_filename(f.name)
        try:
            text = f.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("Skipping %s: %s", f.name, e)
            continue
        documents.append(Document(
            text=text,
            metadata={
                "persona": persona,
                "source":  f.name,
                "type":    "private_corpus",
            },
        ))
        per_persona_files[persona] = per_persona_files.get(persona, 0) + 1
        log.info("Loaded %s (persona=%s)", f.name, persona)

    if not documents:
        result.update(status="warning", message="No readable documents to ingest")
        return result

    try:
        vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # SentenceSplitter is the default LlamaIndex chunker. Episode
        # transcripts run long, so a larger chunk size keeps scene context
        # together while overlap preserves dialogue threading.
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    except Exception as e:
        result.update(status="error", message=f"Index build failed: {e}")
        return result

    total_chunks = chroma_collection.count()
    # Per-persona chunk counts — read back from the collection's metadata.
    per_persona = _count_chunks_by_persona(chroma_collection)

    result.update(
        files=len(documents),
        chunks=total_chunks,
        per_persona=per_persona,
        message=(
            f"Ingested {len(documents)} files → {total_chunks} chunks "
            f"into '{PRIVATE_CORPUS_COLLECTION_NAME}'"
        ),
    )
    return result


def _count_chunks_by_persona(collection) -> dict:
    data = collection.get(include=["metadatas"])
    out: dict = {}
    for md in (data.get("metadatas") or []):
        persona = (md or {}).get("persona", UNKNOWN_PERSONA)
        out[persona] = out.get(persona, 0) + 1
    return out


# ── Status ────────────────────────────────────────────────────────────────────

def status() -> dict:
    client = _open_client()
    try:
        collection = client.get_collection(PRIVATE_CORPUS_COLLECTION_NAME)
    except Exception:
        return {
            "status":   "warning",
            "exists":   False,
            "total":    0,
            "per_persona": {},
            "message":  f"Collection '{PRIVATE_CORPUS_COLLECTION_NAME}' does not exist yet.",
        }
    total = collection.count()
    per_persona = _count_chunks_by_persona(collection) if total else {}
    return {
        "status":      "success",
        "exists":      True,
        "total":       total,
        "per_persona": per_persona,
        "message":     f"'{PRIVATE_CORPUS_COLLECTION_NAME}' has {total} chunks",
    }


# ── Export / import ───────────────────────────────────────────────────────────

def export(dest: Optional[Path] = None) -> dict:
    """Dump the private collection to a portable JSON file.

    File shape:
        {
          "version":          1,
          "collection_name":  "mr_ouija_private",
          "embed_model":      "BAAI/bge-small-en-v1.5",
          "exported_at":      "2026-06-19T12:34:56",
          "ids":              [...],
          "documents":        [...],
          "metadatas":        [{persona, source, ...}, ...],
          "embeddings":       [[float, ...], ...],
        }

    Importer must use the same embed model — otherwise query embeddings
    won't match the stored ones.
    """
    client = _open_client()
    try:
        collection = client.get_collection(PRIVATE_CORPUS_COLLECTION_NAME)
    except Exception as e:
        return {"status": "error", "message": f"Collection not found: {e}"}

    try:
        data = collection.get(include=["embeddings", "documents", "metadatas"])
    except Exception as e:
        return {"status": "error", "message": f"Collection read failed: {e}"}

    # ChromaDB returns numpy arrays for embeddings — coerce to plain lists for
    # JSON serializability. metadatas can be None entries; normalize to {}.
    embeddings_raw = data.get("embeddings") or []
    embeddings = [list(map(float, e)) for e in embeddings_raw]
    metadatas  = [(m or {}) for m in (data.get("metadatas") or [])]

    payload = {
        "version":         EXPORT_FORMAT_VERSION,
        "collection_name": PRIVATE_CORPUS_COLLECTION_NAME,
        "embed_model":     EMBED_MODEL_NAME,
        "exported_at":     datetime.now().isoformat(timespec="seconds"),
        "ids":             list(data.get("ids") or []),
        "documents":       list(data.get("documents") or []),
        "metadatas":       metadatas,
        "embeddings":      embeddings,
    }

    if dest is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest  = DATA_DIR / f"private_corpus_{stamp}.json"
    dest = Path(dest)

    try:
        dest.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        return {"status": "error", "message": f"Write failed: {e}"}

    return {
        "status":  "success",
        "path":    str(dest),
        "chunks":  len(payload["ids"]),
        "size_kb": round(dest.stat().st_size / 1024, 1),
        "message": f"Exported {len(payload['ids'])} chunks → {dest.name}",
    }


def import_(src: Path) -> dict:
    """Restore a private collection from a JSON export."""
    src = Path(src)
    if not src.exists():
        return {"status": "error", "message": f"Source file not found: {src}"}

    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "error", "message": f"JSON parse failed: {e}"}

    if payload.get("version") != EXPORT_FORMAT_VERSION:
        return {
            "status": "error",
            "message": (
                f"Export format version mismatch — file has "
                f"{payload.get('version')!r}, expected {EXPORT_FORMAT_VERSION}"
            ),
        }

    if payload.get("embed_model") != EMBED_MODEL_NAME:
        log.warning(
            "Embed model mismatch — export used %r, current config is %r. "
            "Query embeddings will not match stored ones; retrieval quality "
            "will degrade.",
            payload.get("embed_model"), EMBED_MODEL_NAME,
        )

    ids        = payload.get("ids") or []
    documents  = payload.get("documents") or []
    metadatas  = payload.get("metadatas") or []
    embeddings = payload.get("embeddings") or []

    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        return {
            "status": "error",
            "message": (
                f"Length mismatch in export: ids={len(ids)} docs={len(documents)} "
                f"metadatas={len(metadatas)} embeddings={len(embeddings)}"
            ),
        }
    if not ids:
        return {"status": "warning", "message": "Export file is empty — nothing to import"}

    client = _open_client()
    try:
        client.delete_collection(PRIVATE_CORPUS_COLLECTION_NAME)
        log.info("Dropped existing '%s' before restore",
                 PRIVATE_CORPUS_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(PRIVATE_CORPUS_COLLECTION_NAME)

    try:
        # add() needs equal-length lists. ChromaDB handles batching internally
        # but very large imports may want chunked add() calls; for now the
        # whole episode corpus comfortably fits in one batch.
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=[(m or {}) for m in metadatas],
        )
    except Exception as e:
        return {"status": "error", "message": f"add() failed: {e}"}

    per_persona = _count_chunks_by_persona(collection)
    return {
        "status":      "success",
        "chunks":      len(ids),
        "per_persona": per_persona,
        "message":     f"Imported {len(ids)} chunks into '{PRIVATE_CORPUS_COLLECTION_NAME}'",
    }


# ── Wipe ──────────────────────────────────────────────────────────────────────

def wipe() -> dict:
    client = _open_client()
    try:
        prior = client.get_collection(PRIVATE_CORPUS_COLLECTION_NAME).count()
    except Exception:
        return {"status": "warning", "message": "Nothing to wipe — collection does not exist"}
    try:
        client.delete_collection(PRIVATE_CORPUS_COLLECTION_NAME)
    except Exception as e:
        return {"status": "error", "message": f"Delete failed: {e}"}
    return {
        "status":  "success",
        "chunks_deleted": prior,
        "message": f"Wiped '{PRIVATE_CORPUS_COLLECTION_NAME}' ({prior} chunks).",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_result(result: dict) -> int:
    status = result.get("status", "error")
    if status == "error":
        print(f"\n  ❌ {result.get('message', 'unknown error')}\n")
        return 1
    if status == "warning":
        print(f"\n  ⚠️  {result.get('message', '')}\n")
        return 2
    print(f"\n  ✅ {result.get('message', 'ok')}")
    if "per_persona" in result and result["per_persona"]:
        print("     Per persona:")
        for persona, count in sorted(result["per_persona"].items()):
            print(f"       {persona:<10} {count}")
    if "path" in result:
        print(f"     Path: {result['path']}")
    if "size_kb" in result:
        print(f"     Size: {result['size_kb']} KB")
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Agent Ouija private corpus manager")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", action="store_true",
                       help=f"Build collection from {PRIVATE_CORPUS_EPISODES_DIR}")
    group.add_argument("--status", action="store_true",
                       help="Show per-persona chunk counts")
    group.add_argument("--export", nargs="?", const="<auto>", metavar="PATH",
                       help="Export collection to JSON (default: data/private_corpus_<stamp>.json)")
    group.add_argument("--import", dest="import_path", metavar="PATH",
                       help="Import collection from a JSON export file")
    group.add_argument("--wipe", action="store_true",
                       help="Drop the private collection (source files untouched)")
    args = parser.parse_args()

    if args.ingest:
        sys.exit(_print_result(ingest()))
    if args.status:
        sys.exit(_print_result(status()))
    if args.export is not None:
        dest = None if args.export == "<auto>" else Path(args.export)
        sys.exit(_print_result(export(dest)))
    if args.import_path:
        sys.exit(_print_result(import_(Path(args.import_path))))
    if args.wipe:
        sys.exit(_print_result(wipe()))


if __name__ == "__main__":
    main()
