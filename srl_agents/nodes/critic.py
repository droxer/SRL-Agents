"""Critic stage node."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..logging import console
from ..state import AgentState, CriticOutput

_REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict knowledge base administrator. Review this AI-generated experience rule.\n"
            "Decision criteria:\n"
            "1. APPROVE: Rule is accurate, safe, and generalizable.\n"
            "2. REVISE: Rule is ambiguous, dangerous, or inaccurate. Provide revision suggestions.\n"
            "3. DISCARD: Rule is nonsense or completely wrong.""",
        ),
        (
            "user",
            "Rule to review: [{topic}] {insight}\nReasoning: {reasoning}",
        ),
    ]
)


def critic_node(llm: ChatOpenAI):
    def critic_node(state: AgentState):
        reflection = state["proposed_reflection"]

        if not reflection.should_store:
            return {"review_decision": "DISCARD", "critic_feedback": ""}

        structured_llm = llm.with_structured_output(CriticOutput)
        result = structured_llm.invoke(
            _REVIEW_PROMPT.format(
                topic=reflection.topic, insight=reflection.insight, reasoning=reflection.reasoning
            )
        )

        console.rule("[bold cyan]4. Critic")
        console.print(f"[yellow]Decision:[/yellow] {result.decision}")
        if result.decision == "REVISE":
            console.print(f"[yellow]Feedback:[/yellow] {result.feedback}")

        update = {"review_decision": result.decision}
        if result.decision == "REVISE":
            update["critic_feedback"] = result.feedback
        else:
            update["critic_feedback"] = ""
        return update

    return critic_node
