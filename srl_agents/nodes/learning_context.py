"""Learning context extractor node."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..logging import console
from ..state import AgentState, LearningContext

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a visible-learning coach. Rewrite the learner's question into structured intent metadata:\n"
            "- learning_goal: describe the target skill or understanding in the learner's own words\n"
            "- success_criteria: list observable evidence that would prove success\n"
            "- prior_knowledge: summarize what the learner already seems to know (or misconceptions)",
        ),
        ("user", "Learner question: {query}"),
    ]
)


def build_learning_context_node(llm: ChatOpenAI):
    structured_llm = llm.with_structured_output(LearningContext)

    def learning_context_node(state: AgentState):
        query = state["query"]
        console.rule("[bold cyan]0. Learning Context")
        console.print(f"[bold]Learner question:[/bold] {query}")
        context = structured_llm.invoke(_PROMPT.format(query=query))
        console.print(
            f"[green]Goal:[/green] {context.learning_goal}\n"
            f"[green]Success criteria:[/green] {context.success_criteria}\n"
            f"[green]Prior knowledge:[/green] {context.prior_knowledge}"
        )
        return {"learning_context": context}

    return learning_context_node
