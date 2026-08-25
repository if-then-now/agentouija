"""
rag.py — ChromaDB retrieval for cloud-model context injection.

Owns the embedding model + vector store connection used to fetch chunks
relevant to a question. Cloud models receive this retrieved context as
an extra system message; locals run without it (CLAUDE.md: "Cloud vs
Local — Intentionally Unequal").

Lazy-initialized: importing this module is free. The first call to
get_rag_context() loads the embedding model and opens the ChromaDB
collection, then caches both for the lifetime of the process.

Graceful failure: if ChromaDB is empty, missing, or errors during
retrieval, get_rag_context() returns None and logs a warning. The
caller still runs the model — just without injected context.

Single Responsibility: this module is the only place that touches the
RAG retrieval path. arena.py / arena_ui.py call it, consume the string,
done.
"""

import logging
from datetime import datetime
from typing import Optional

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBED_MODEL_NAME,
    PRIVATE_CORPUS_COLLECTION_NAME,
    PRIVATE_CORPUS_TOP_K,
    PRIVATE_CORPUS_ACTIVE_FLAG,
    USE_PRIVATE_CORPUS,
)

log = logging.getLogger(__name__)

# Module-global retriever singletons. Initialized lazily on first use so
# loading rag.py stays cheap.
_RETRIEVER = None
_PRIVATE_RETRIEVER = None
_TOP_K = 5


def _init_retriever():
    """Build the retriever once. Subsequent calls reuse the cached instance."""
    global _RETRIEVER
    if _RETRIEVER is not None:
        return _RETRIEVER

    # Heavy imports are kept inside the init function so importing rag.py at
    # module load time stays cheap. The embedding model + llama-index stack
    # only load when a cloud model actually requests retrieval.
    import chromadb
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    log.info("Initializing RAG retriever (model=%s, collection=%s, dir=%s)",
             EMBED_MODEL_NAME, CHROMA_COLLECTION_NAME, CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    _RETRIEVER = index.as_retriever(similarity_top_k=_TOP_K)
    log.info("RAG retriever ready (top_k=%d)", _TOP_K)
    return _RETRIEVER


def wipe_chroma_collection() -> dict:
    """Delete the ChromaDB collection so the next Ingest Documents run can
    rebuild it from scratch. Resets the cached retriever singleton so any
    in-flight rag.get_rag_context() calls after the wipe will re-initialize
    against the new (empty) collection.

    Returns a status dict, never raises:
        {"status": "success"|"error",
         "chunks_deleted": int,
         "message": str}
    """
    global _RETRIEVER
    result = {"status": "success", "chunks_deleted": 0, "message": ""}

    # Lazy imports — keep rag.py cheap to load.
    import chromadb

    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            existing = client.get_collection(CHROMA_COLLECTION_NAME)
            prior_count = existing.count()
            client.delete_collection(CHROMA_COLLECTION_NAME)
            result.update(
                chunks_deleted=prior_count,
                message=f"Deleted {prior_count} chunks from '{CHROMA_COLLECTION_NAME}'",
            )
        except Exception:
            # Collection didn't exist — fine, treat as success.
            result.update(message=f"No prior collection '{CHROMA_COLLECTION_NAME}' to delete")
    except Exception as e:
        log.exception("wipe_chroma_collection failed")
        result.update(status="error", message=f"ChromaDB wipe failed: {e}")

    # Drop the cached retriever regardless of outcome — it would otherwise
    # point at a stale (deleted) collection on the next get_rag_context().
    _RETRIEVER = None
    return result


def reset_retriever() -> None:
    """Drop the cached main retriever so the next get_rag_context() rebuilds
    against the (now updated) collection. Called after add_to_collection so
    a freshly-curated answer becomes retrievable on the very next query
    instead of after a restart."""
    global _RETRIEVER
    _RETRIEVER = None


def count_chroma_before(cutoff_iso: str) -> dict:
    """Preview how many chunks in the main collection have
    `created_at < cutoff_iso`. Read-only — used by the rollback
    preview button so the user knows what a purge would remove
    before they commit.

    Chunks written before the created_at field was introduced (or
    docs ingested without a date) will be counted separately as
    `undated` — those don't age out on this control; only full
    wipe removes them."""
    import chromadb

    result = {"matched": 0, "undated": 0, "total": 0, "error": None}
    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        result["total"] = collection.count()
        # Chroma's where filter can only include, not exclude — so we
        # do the counting in two passes.
        got = collection.get(where={"created_at": {"$lt": cutoff_iso}},
                              include=[])
        result["matched"] = len(got.get("ids") or [])
    except Exception as e:
        log.exception("count_chroma_before failed")
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def purge_chroma_before(cutoff_iso: str) -> dict:
    """Delete every chunk in the main collection whose `created_at`
    metadata is strictly less than cutoff_iso. Returns a status dict
    with chunks_deleted + a message; never raises. Chunks without a
    created_at field are LEFT ALONE — they don't match the filter.

    Resets the cached retriever so subsequent get_rag_context() calls
    rebuild against the (now smaller) collection."""
    global _RETRIEVER
    import chromadb

    result = {"status": "success", "chunks_deleted": 0, "message": ""}
    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        got = collection.get(where={"created_at": {"$lt": cutoff_iso}},
                              include=[])
        ids = got.get("ids") or []
        if ids:
            collection.delete(ids=ids)
        result["chunks_deleted"] = len(ids)
        result["message"] = (
            f"Deleted {len(ids)} chunk(s) with created_at < {cutoff_iso}"
        )
    except Exception as e:
        log.exception("purge_chroma_before failed")
        result.update(status="error",
                        message=f"Chroma purge failed: {type(e).__name__}: {e}")

    # Drop the cached retriever whether the delete succeeded or not —
    # even a partial failure could have removed some chunks the cached
    # index still points at.
    _RETRIEVER = None
    return result


def add_to_collection(question: str,
                      response: str,
                      metadata: Optional[dict] = None) -> dict:
    """Append a Q+A pair to the main mr_ouija ChromaDB collection.

    Used by the Arena tab's "Score + Add To RAG" button so a high-scoring
    response can become reference material for future queries. Locals only
    benefit if their per-model "rag" flag is True (default False per the
    CLAUDE.md 'Cloud vs Local — Intentionally Unequal' design).

    Returns a status dict; never raises. After a successful add the
    cached retriever is reset so the new chunk is retrievable on the
    very next get_rag_context() call.
    """
    if not question or not question.strip():
        return {"status": "error", "message": "Empty question — nothing to add"}
    if not response or not response.strip():
        return {"status": "error", "message": "Empty response — nothing to add"}

    # Lazy imports — keeps loading rag.py cheap for the get-path-only callers.
    try:
        import chromadb
        from llama_index.core import (
            Document,
            Settings,
            StorageContext,
            VectorStoreIndex,
        )
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except Exception as e:
        return {"status": "error", "message": f"RAG stack import failed: {e}"}

    md: dict = dict(metadata or {})
    md.setdefault("source", "arena_curated")
    md.setdefault("type",   "scored_answer")
    # created_at drives the unified rollback control in the Stats tab
    # (Danger Zone). Every new chunk gets one so purge-by-age can find
    # it — chunks written before this field existed are un-purgeable
    # except via full wipe.
    md.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

    text = f"Question: {question.strip()}\n\nAnswer: {response.strip()}"

    try:
        # ALWAYS pin the embed model before any operation that could
        # trigger embedding. LlamaIndex's Settings.embed_model defaults to
        # a lazily-resolved OpenAI proxy (1536-dim) when not explicitly
        # set — the `is None` check we used here previously didn't catch
        # the proxy, so add_to_collection would silently embed via OpenAI
        # and Chroma would reject the dim mismatch against the existing
        # bge-small (384-dim) collection. Set unconditionally; the
        # HuggingFaceEmbedding wrapper caches the underlying model so
        # repeated sets are cheap after the first.
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

        vector_store    = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            [Document(text=text, metadata=md)],
            storage_context=storage_context,
        )

        # New chunk is on disk — drop the cached retriever so the next
        # similarity search picks it up immediately.
        reset_retriever()
        return {
            "status":  "success",
            "message": (
                f"Added curated Q+A to '{CHROMA_COLLECTION_NAME}' "
                f"(now {collection.count()} chunks)"
            ),
        }
    except Exception as e:
        log.exception("add_to_collection failed")
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


def get_rag_context(question: str) -> Optional[str]:
    """Return a formatted context block for `question`, or None.

    None signals "no usable context" — empty collection, retrieval error,
    or empty question. Callers should treat None as "run the model without
    extra context" rather than as an error.
    """
    if not question or not question.strip():
        return None
    try:
        retriever = _init_retriever()
        nodes = retriever.retrieve(question)
    except Exception as e:
        log.warning("RAG retrieval failed (%s): %s — model will run without context",
                    type(e).__name__, e)
        return None

    if not nodes:
        return None

    chunks = []
    for i, n in enumerate(nodes, 1):
        text = n.node.get_content().strip()
        if text:
            chunks.append(f"[{i}] {text}")
    return "\n\n".join(chunks) if chunks else None


# ── Private corpus retriever ──────────────────────────────────────────────────
# Separate ChromaDB collection populated by scripts/private_corpus.py.
# Gated by USE_PRIVATE_CORPUS at the call site so this path is dead code
# until the user explicitly flips the master switch.

def _init_private_retriever():
    global _PRIVATE_RETRIEVER
    if _PRIVATE_RETRIEVER is not None:
        return _PRIVATE_RETRIEVER

    import chromadb
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    log.info(
        "Initializing PRIVATE retriever (collection=%s, dir=%s, top_k=%d)",
        PRIVATE_CORPUS_COLLECTION_NAME, CHROMA_DIR, PRIVATE_CORPUS_TOP_K,
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # Pin the embed model unconditionally — see add_to_collection for the
    # rationale (the None-check doesn't catch LlamaIndex's default OpenAI
    # proxy, which would produce 1536-dim embeddings and break retrieval
    # against the 384-dim bge-small collection).
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Use get_collection (not get_or_create) — if the private collection
    # doesn't exist we want to surface that clearly rather than silently
    # create an empty one. Caller wraps this in try/except already.
    collection = client.get_collection(PRIVATE_CORPUS_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    _PRIVATE_RETRIEVER = index.as_retriever(similarity_top_k=PRIVATE_CORPUS_TOP_K)
    return _PRIVATE_RETRIEVER


def reset_private_retriever() -> None:
    """Drop the cached private retriever so the next call rebuilds it.

    Useful after running scripts/private_corpus.py --import in the same
    Python process (e.g. a long-running arena_ui session). The arena UI
    never does this today; included for symmetry with wipe_chroma_collection.
    """
    global _PRIVATE_RETRIEVER
    _PRIVATE_RETRIEVER = None


def get_private_context(question: str) -> Optional[str]:
    """Return formatted context from the private corpus, or None.

    Returns None when:
      - USE_PRIVATE_CORPUS is False (the master switch)
      - the question is empty
      - the private collection doesn't exist (first run before --ingest)
      - the retriever errors out for any reason
      - no chunks match the question
    Caller is expected to skip injection when None.
    """
    # Master switch: config constant OR the runtime flag file the Persona
    # tab manages. Either being true lights up the retrieval path — lets
    # the user flip private-corpus on/off without a UI restart.
    if not (USE_PRIVATE_CORPUS or PRIVATE_CORPUS_ACTIVE_FLAG.exists()):
        return None
    if not question or not question.strip():
        return None
    try:
        retriever = _init_private_retriever()
        nodes = retriever.retrieve(question)
    except Exception as e:
        log.warning(
            "Private-corpus retrieval failed (%s): %s — model will run "
            "without private context",
            type(e).__name__, e,
        )
        return None

    if not nodes:
        return None

    chunks = []
    for i, n in enumerate(nodes, 1):
        text = n.node.get_content().strip()
        if not text:
            continue
        persona = (n.node.metadata or {}).get("persona", "?")
        chunks.append(f"[{i}] ({persona}) {text}")
    return "\n\n".join(chunks) if chunks else None
