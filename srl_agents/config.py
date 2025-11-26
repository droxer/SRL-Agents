"""Centralized configuration helpers for LangGraph application."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables once at import time so CLI users can rely on .env files
load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma"))


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a singleton ChatOpenAI client configured via environment variables."""
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return a singleton embedding client for semantic memory search."""
    return OpenAIEmbeddings(model=DEFAULT_EMBED_MODEL)


@lru_cache(maxsize=1)
def get_vector_client() -> PersistentClient:
    """Return a shared ChromaDB persistent client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return PersistentClient(path=str(CHROMA_DIR))
