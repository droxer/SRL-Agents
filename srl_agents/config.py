"""Centralized configuration helpers for LangGraph application."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables once at import time so CLI users can rely on .env files
load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a singleton ChatOpenAI client configured via environment variables."""
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
