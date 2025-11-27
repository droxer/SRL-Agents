"""Reflector stage node."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..logging import console
from ..state import AgentState, ReflectionOutput

_SYSTEM_MSG_INITIAL = (
    "You are a reflection assistant. Review the interaction and extract a brief, reusable technical rule."
)

_SYSTEM_MSG_REVISION = (
    "Your previous summary was rejected. "
    "Feedback: {feedback}. Revise your rule summary accordingly."
)


def build_reflector_node(llm: ChatOpenAI):
    def reflector_node(state: AgentState):
        query = state["query"]
        response = state["response"]
        actor_trace = state.get("actor_trace", [])
        retrieved_memories = state.get("retrieved_memories", "")
        web_results = state.get("web_results", "")
        feedback = state.get("critic_feedback", "")
        retry_count = state.get("retry_count", 0)

        if feedback:
            console.rule(f"[bold cyan]4. Reflector · Attempt {retry_count}")
            system_msg = _SYSTEM_MSG_REVISION.format(feedback=feedback)
        else:
            console.rule("[bold cyan]4. Reflector")
            system_msg = _SYSTEM_MSG_INITIAL

        prompt = ChatPromptTemplate.from_messages(
            _build_reflection_messages(system_msg, query, response, actor_trace, retrieved_memories, web_results)
        )

        structured_llm = llm.with_structured_output(ReflectionOutput)
        reflection = structured_llm.invoke(prompt.format())
        reflection = reflection.model_copy(update={"source_query": query})
        console.print(f"[magenta]Proposed rule:[/magenta] {reflection.insight}")
        return {"proposed_reflection": reflection, "retry_count": retry_count + 1}

    return reflector_node


def _build_reflection_messages(
    system_msg: str,
    query: str,
    response: str,
    actor_trace: list[str],
    retrieved_memories: str = "",
    web_results: str = "",
):
    """Build ReAct-style reflection messages showing Thought → Action → Observation cycle."""
    sections = []
    
    # Show actions and observations (ReAct pattern)
    if retrieved_memories and retrieved_memories.strip():
        sections.append(f"Action: Retrieved memories\nObservation: {retrieved_memories}")
    
    if web_results and web_results.strip():
        sections.append(f"Action: Web search\nObservation: {web_results}")
    
    # Show reasoning thoughts (ReAct pattern)
    if actor_trace:
        thought_lines = "\n".join(f"Thought {i+1}: {step}" for i, step in enumerate(actor_trace))
        sections.append(f"Reasoning steps:\n{thought_lines}")
    
    trace_section = ""
    if sections:
        trace_section = "\n\n" + "\n\n".join(sections)
    
    return [
        ("system", system_msg),
        ("user", f"User question: {query}\nYour response: {response}{trace_section}"),
    ]
