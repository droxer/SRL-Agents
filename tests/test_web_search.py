"""Tests for the DuckDuckGo-backed web search tool."""
from __future__ import annotations

from srl_agents.tools.web_search import WebSearchResult, WebSearchTool


class DummySession:
    def __init__(self, payload):
        self.payload = payload
        self.invocations: list[tuple[str, int]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query: str, *, max_results: int):
        self.invocations.append((query, max_results))
        return self.payload


def test_web_search_tool_returns_structured_results():
    payload = [
        {"title": "  Doc title  ", "href": "https://example.com/doc", "body": "Helpful summary."},
        {"title": "", "href": "", "body": "ignored"},
    ]
    session = DummySession(payload)
    tool = WebSearchTool(session_factory=lambda: session, max_results=3)

    results = tool.search("  python news  ")

    assert len(results) == 1
    assert results[0].title == "Doc title"
    assert results[0].url == "https://example.com/doc"
    assert session.invocations == [("python news", 3)]


def test_web_search_tool_handles_empty_queries():
    session = DummySession([])
    tool = WebSearchTool(session_factory=lambda: session)

    results = tool.search("   ")

    assert results == []
    assert session.invocations == []


def test_format_results_produces_bullets():
    snippets = "x" * 400
    summary = WebSearchTool.format_results(
        [
            WebSearchResult(title="Result A", url="https://a", snippet=snippets),
            WebSearchResult(title="Result B", url="https://b", snippet="Short"),
        ],
        limit=1,
    )

    assert summary.startswith("- Result A (https://a): ")
    assert summary.endswith("...")
