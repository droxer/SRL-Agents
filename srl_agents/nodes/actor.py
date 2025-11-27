"""Actor stage node."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..logging import console
from ..state import ActorOutput, AgentState

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent assistant. Think step-by-step before helping the learner. "
            "Follow this protocol:\n"
            "1. Review retrieved memories.\n"
            "2. Review recent web findings.\n"
            "3. Produce 2-4 concise reasoning thoughts referencing (Memory) or (Web) when useful.\n"
            "4. Conclude with a final helpful answer.\n\n"
            "Past experiences:\n{memories}\n\nRecent web findings:\n{web_context}",
        ),
        ("user", "{query}"),
    ]
)


def build_actor_node(llm: ChatOpenAI):
    structured_llm = llm.with_structured_output(ActorOutput)

    def actor_node(state: AgentState):
        query = state["query"]
        console.rule("[bold cyan]3. Actor")
        console.print(f"[bold]Learner query:[/bold] {query}")
        web_context = state.get("web_results") or "No useful web evidence returned."
        prompt = _PROMPT.format(
            memories=state["retrieved_memories"],
            web_context=web_context,
            query=query,
        )
        result = structured_llm.invoke(prompt)
        for idx, thought in enumerate(result.thoughts, start=1):
            console.print(f"[dim]Thought {idx}:[/dim] {thought}")
        console.print(result.answer)
        return {"response": result.answer, "actor_trace": result.thoughts}

    return actor_node
