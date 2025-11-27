"""Simple web search adapter exposed as an MCP-ready tool."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

from duckduckgo_search import DDGS

from ..logging import console


@dataclass
class WebSearchResult:
    """Structured representation of a single web search hit."""

    title: str
    url: str
    snippet: str

    def to_bullet(self) -> str:
        snippet = self.snippet.strip()
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        return f"- {self.title} ({self.url}): {snippet}"


class _SearchSession(Protocol):
    def text(self, query: str, *, max_results: int) -> Iterable[dict]:
        ...


SessionFactory = Callable[[], AbstractContextManager[_SearchSession]]


class WebSearchTool:
    """Lightweight wrapper around DuckDuckGo search for external knowledge."""

    def __init__(self, *, max_results: int = 5, session_factory: SessionFactory | None = None):
        self.max_results = max_results
        self._session_factory: SessionFactory = session_factory or (lambda: DDGS())

    def search(self, query: str) -> list[WebSearchResult]:
        """Return a list of structured web search results."""
        query = query.strip()
        if not query:
            return []
        try:
            with self._session_factory() as session:
                raw_results = list(session.text(query, max_results=self.max_results) or [])
        except Exception as exc:  # pragma: no cover - network failure is best-effort
            console.print(f"[yellow]Web search failed:[/yellow] {exc}")
            return []

        results: list[WebSearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "Untitled result").strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            if not (title and url):
                continue
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        return results

    @staticmethod
    def format_results(results: Sequence[WebSearchResult], *, limit: int = 5) -> str:
        """Return a short plaintext summary suitable for prompts."""
        if not results:
            return ""
        limited = results[:limit]
        bullets = [result.to_bullet() for result in limited]
        return "\n".join(bullets)


__all__ = ["WebSearchResult", "WebSearchTool"]
