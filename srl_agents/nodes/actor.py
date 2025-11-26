"""Actor stage node."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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
        result = chain.invoke({"memories": state["retrieved_memories"], "query": state["query"]})
        print(f"\n--- 2. Actor ---\nResponse: {result.content[:100]}...")
        return {"response": result.content}

    return actor_node
