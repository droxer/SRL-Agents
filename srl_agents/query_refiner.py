"""Utilities for rewriting learner queries before memory retrieval."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .logging import console

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the learner's request into a terse semantic search string. "
            "Highlight key skills, intents, and error patterns. Keep it under 40 words.",
        ),
        ("user", "Original query: {query}\nRefined search string:"),
    ]
)


class LLMQueryRefiner:
    """Callable that turns free-form learner questions into semantic search strings."""

    def __init__(self, llm: ChatOpenAI):
        self.chain = _PROMPT | llm

    def __call__(self, query: str) -> str:
        if not query.strip():
            return query
        try:
            result = self.chain.invoke({"query": query})
            refined = (result.content or "").strip()
            return refined or query
        except Exception as exc:  # pragma: no cover
            console.print(f"[yellow]Query refinement failed:[/yellow] {exc}")
            return query
