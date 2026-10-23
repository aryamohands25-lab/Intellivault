
"""Shared settings and helpers for IntelliVault (Phase 1 + 2)."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "intellivault"

# --------------------------------------------------
# Embedding model
# --------------------------------------------------

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Prefix added to search queries for BGE embeddings.
# Documents should be embedded without this prefix.
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

# CPU optimization for your 6-logical-core machine.
CPU_THREADS = 4
EMBED_BATCH_SIZE = 32  # Increased from 16 for faster PDF embedding

# --------------------------------------------------
# Text chunking and retrieval
# --------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TOP_K = 5
MIN_SIMILARITY = 0.5

# --------------------------------------------------
# LLM provider settings
# --------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",  # confirmed valid by API
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "claude-sonnet-5-5",
)

# --------------------------------------------------
# Cached embedding model
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder():
    """Load the embedding model once and reuse it."""

    import torch
    from sentence_transformers import SentenceTransformer

    # Limit CPU threads to reduce oversubscription.
    torch.set_num_threads(CPU_THREADS)

    model = SentenceTransformer(
        EMBED_MODEL,
        device="cpu",
    )

    model.eval()
    return model


# --------------------------------------------------
# Cached persistent ChromaDB collection
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_collection():
    """Return the persistent ChromaDB collection."""

    import chromadb

    client = chromadb.PersistentClient(
        path=str(DB_DIR),
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
