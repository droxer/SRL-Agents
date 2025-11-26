"""Forethought stage node."""
from __future__ import annotations

from ..memory import MemoryStore
from ..state import AgentState


def build_forethought_node(store: MemoryStore):
    def forethought_node(state: AgentState):
        query = state["query"]
        memories = store.search(query)
        print(f"\n--- 1. Forethought ---\nRetrieved results: {memories}")
        return {"retrieved_memories": memories}

    return forethought_node
