"""Storage stage node."""
from __future__ import annotations

from ..logging import console
from ..memory import MemoryStore
from ..state import AgentState

MIN_IMPACT_SCORE = 3


def build_store_node(store: MemoryStore):
    def store_node(state: AgentState):
        reflection = state["proposed_reflection"]
        impact_score = state.get("impact_score", 0)
        if impact_score < MIN_IMPACT_SCORE:
            console.print(
                f"[yellow]Skipping storage (impact score {impact_score} < {MIN_IMPACT_SCORE}).[/yellow]"
            )
            return {}

        learning_context = state.get("learning_context")
        success_criteria = (
            learning_context.success_criteria if learning_context else None
        )
        store.add(
            reflection,
            impact_score=impact_score,
            success_criteria=success_criteria,
        )
        return {}

    return store_node
