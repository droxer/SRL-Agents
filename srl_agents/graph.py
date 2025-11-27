"""Graph assembly helpers."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .config import get_embeddings, get_llm, get_vector_client
from .logging import console
from .memory import MemoryStore
from .nodes.actor import build_actor_node
from .nodes.critic import build_critic_node
from .nodes.forethought import build_forethought_node
from .nodes.reflector import build_reflector_node
from .nodes.store import build_store_node
from .nodes.web_search import build_web_search_node
from .query_refiner import LLMQueryRefiner
from .state import AgentState
from .tools.web_search import WebSearchTool


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
    store = memory_store or MemoryStore(
        embedder=get_embeddings(),
        client=get_vector_client(),
        query_refiner=LLMQueryRefiner(llm),
    )
    web_search_tool = WebSearchTool()

    workflow = StateGraph(AgentState)
    workflow.add_node("forethought", build_forethought_node(store))
    workflow.add_node("web_search", build_web_search_node(web_search_tool))
    workflow.add_node("actor", build_actor_node(llm))
    workflow.add_node("reflector", build_reflector_node(llm))
    workflow.add_node("critic", build_critic_node(llm))
    workflow.add_node("store", build_store_node(store))

    workflow.add_edge(START, "forethought")
    workflow.add_edge("forethought", "web_search")
    workflow.add_edge("web_search", "actor")
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
