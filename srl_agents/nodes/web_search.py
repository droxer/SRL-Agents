"""Web search stage node."""
from __future__ import annotations

from ..logging import console
from ..state import AgentState
from ..tools.web_search import WebSearchTool


def build_web_search_node(tool: WebSearchTool):
    def web_search_node(state: AgentState):
        query = state["query"]
        console.rule("[bold cyan]2. Web Search")
        results = tool.search(query)
        summary = tool.format_results(results)
        if summary:
            console.print(summary)
        else:
            console.print("[dim]No useful external links discovered.[/dim]")
        return {"web_results": summary}

    return web_search_node
