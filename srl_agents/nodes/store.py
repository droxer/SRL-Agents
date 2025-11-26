"""Storage stage node."""
from __future__ import annotations

from ..memory import MemoryStore
from ..state import AgentState


def build_store_node(store: MemoryStore):
    def store_node(state: AgentState):
        reflection = state["proposed_reflection"]
        store.add(reflection)
        return {}

    return store_node
