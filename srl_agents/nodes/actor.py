"""Actor stage node."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..logging import console
from ..state import AgentState

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intelligent assistant. Use these past experiences (if any) to avoid repeating mistakes:\n{memories}",
        ),
        ("user", "{query}"),
    ]
)


def build_actor_node(llm: ChatOpenAI):
    chain = _PROMPT | llm

    def actor_node(state: AgentState):
        query = state["query"]
        console.rule("[bold cyan]2. Actor")
        console.print(f"[bold]Learner query:[/bold] {query}")
        result = chain.invoke({"memories": state["retrieved_memories"], "query": query})
        console.print(result.content)
        return {"response": result.content}

    return actor_node
