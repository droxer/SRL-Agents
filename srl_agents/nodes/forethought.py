"""Forethought stage node."""
from __future__ import annotations

from ..logging import console
from ..memory import MemoryStore
from ..state import AgentState


def build_forethought_node(store: MemoryStore):
    def forethought_node(state: AgentState):
        query = state["query"]
        memories = store.search(query)
        console.rule("[bold cyan]1. Forethought")
        console.print(memories if memories else "No memories retrieved.")
        return {"retrieved_memories": memories}

    return forethought_node
