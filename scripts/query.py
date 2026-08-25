# query.py
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from config import (
    AVAILABLE_MODELS,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    DEFAULT_MODEL,
    DEFAULT_PERSONA_MODE,
    EMBED_MODEL_NAME,
    OLLAMA_TIMEOUT,
)
from prompt_loader import get_prompt_loader

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def configure_embedding_model() -> None:
    """Configure local HuggingFace embedding model — no cloud."""
    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        logger.info(f"Embedding model loaded: {EMBED_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def configure_llm(model_key: str = DEFAULT_MODEL) -> Ollama:
    """
    Configure LLM from AVAILABLE_MODELS.
    Add new models to config.py — no code changes needed here.
    """
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_key}' not found in AVAILABLE_MODELS. Check config.py.")

    model_config = AVAILABLE_MODELS[model_key]
    model_name = model_config["name"]

    try:
        llm = Ollama(model=model_name, request_timeout=OLLAMA_TIMEOUT)
        Settings.llm = llm
        logger.info(f"LLM configured: {model_config['display_name']} ({model_name})")
        return llm
    except Exception as e:
        logger.error(f"Failed to configure LLM {model_name}: {e}")
        raise


def load_system_prompt(model_key: str = DEFAULT_MODEL) -> str:
    """Load the system prompt for this model's configured persona mode."""
    persona_mode = AVAILABLE_MODELS.get(model_key, {}).get("persona", DEFAULT_PERSONA_MODE)
    try:
        prompt = get_prompt_loader(persona_mode).load()
        logger.info("System prompt assembled (persona_mode=%s)", persona_mode)
        return prompt
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        raise


def connect_to_chat_engine(system_prompt: str):
    """Connect to ChromaDB and return a chat engine."""
    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        logger.info(f"Connected to ChromaDB collection: {CHROMA_COLLECTION_NAME} at {CHROMA_DIR}")

        chat_engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=system_prompt,
            verbose=False
        )
        logger.info("Chat engine initialized.")
        return chat_engine
    except Exception as e:
        logger.error(f"Failed to initialize chat engine: {e}")
        raise


def run_query_loop(chat_engine, model_key: str = DEFAULT_MODEL) -> None:
    """Main interactive query loop."""
    model_display = AVAILABLE_MODELS[model_key]["display_name"]
    print(f"\nAgent Ouija is ready [{model_display}]. Ask him anything. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                logger.info("Session ended by user.")
                break
            response = chat_engine.chat(question)
            print(f"\nAgent Ouija [{model_display}]: {response}\n")
        except KeyboardInterrupt:
            logger.info("Session interrupted.")
            break
        except Exception as e:
            logger.error(f"Query failed: {e}")
            print("Something went wrong. Try again.\n")


def main(model_key: str = DEFAULT_MODEL):
    logger.info(f"Initializing Agent Ouija — Model: {model_key}")

    configure_embedding_model()
    configure_llm(model_key)

    try:
        system_prompt = load_system_prompt(model_key)
    except Exception:
        sys.exit(1)

    try:
        chat_engine = connect_to_chat_engine(system_prompt)
    except Exception:
        sys.exit(1)

    run_query_loop(chat_engine, model_key)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agent Ouija query interface")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(AVAILABLE_MODELS.keys()),
                        help="Model to use for this session")
    args = parser.parse_args()
    main(model_key=args.model)