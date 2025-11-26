"""Reflector stage node."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..logging import console
from ..state import AgentState, ReflectionOutput


def build_reflector_node(llm: ChatOpenAI):
    def reflector_node(state: AgentState):
        query = state["query"]
        response = state["response"]
        feedback = state.get("critic_feedback", "")
        retry_count = state.get("retry_count", 0)

        if feedback:
            console.rule(f"[bold cyan]3. Reflector · Attempt {retry_count}")
            system_msg = (
                "Your previous summary was rejected. "
                f"Feedback: {feedback}. Revise your rule summary accordingly."
            )
        else:
            console.rule("[bold cyan]3. Reflector")
            system_msg = (
                "You are a reflection assistant. Review the interaction and extract a brief, reusable technical rule."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("user", f"User question: {query}\nYour response: {response}"),
            ]
        )

        structured_llm = llm.with_structured_output(ReflectionOutput)
        reflection = structured_llm.invoke(prompt.format())
        reflection = reflection.model_copy(update={"source_query": query})
        console.print(f"[magenta]Proposed rule:[/magenta] {reflection.insight}")
        return {"proposed_reflection": reflection, "retry_count": retry_count + 1}

    return reflector_node
