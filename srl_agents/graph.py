"""Graph assembly helpers."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .config import get_llm
from .logging import console
from .memory import MemoryStore
from .nodes.actor import actor_node
from .nodes.critic import critic_node
from .nodes.forethought import forethought_node
from .nodes.reflector import reflector_node
from .nodes.store import build_store_node
from .state import AgentState


def _router(state: AgentState):
    decision = state.get("review_decision", "")
    retry_count = state.get("retry_count", 0)

    if decision == "APPROVE":
        return "store"
    if decision == "DISCARD":
        return END
    if decision == "REVISE":
        if retry_count >= 3:
            console.print("[red]Exceeded maximum retry count, abandoning record.[/red]")
            return END
        return "reflector"
    return END


def create_app(memory_store: MemoryStore | None = None):
    """Compile and return the LangGraph application."""
    llm = get_llm()
    store = memory_store or MemoryStore()

    workflow = StateGraph(AgentState)
    workflow.add_node("forethought", forethought_node(store))
    workflow.add_node("actor", actor_node(llm))
    workflow.add_node("reflector", reflector_node(llm))
    workflow.add_node("critic", critic_node(llm))
    workflow.add_node("store", build_store_node(store))

    workflow.add_edge(START, "forethought")
    workflow.add_edge("forethought", "actor")
    workflow.add_edge("actor", "reflector")
    workflow.add_edge("reflector", "critic")

    workflow.add_conditional_edges(
        "critic",
        _router,
        {
            "store": "store",
            "reflector": "reflector",
            END: END,
        },
    )

    workflow.add_edge("store", END)
    return workflow.compile()
