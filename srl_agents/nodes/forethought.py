"""Forethought stage node."""
from __future__ import annotations

from ..logging import console
from ..memory import MemoryStore
from ..state import AgentState, LearningContext


def build_forethought_node(store: MemoryStore):
    def forethought_node(state: AgentState):
        query = state["query"]
        memories = store.search(query)
        learning_context: LearningContext | None = state.get("learning_context")
        needs_research = _should_research(memories, learning_context)
        console.rule("[bold cyan]1. Forethought")
        console.print(memories if memories else "No memories retrieved.")
        if not needs_research:
            console.print("[dim]Existing memories satisfy the current goal; skipping web search.[/dim]")
        return {"retrieved_memories": memories, "needs_research": needs_research}

    return forethought_node


def _should_research(memories: str, learning_context: LearningContext | None) -> bool:
    normalized = (memories or "").strip()
    has_actionable_memory = normalized.startswith("- [")
    if not has_actionable_memory:
        return True
    if not learning_context:
        return False
    criteria = learning_context.success_criteria.lower()
    freshness_keywords = ("latest", "current", "recent", "up-to-date", "news", "trend")
    return any(keyword in criteria for keyword in freshness_keywords)
