"""Typed state and structured outputs for the SRL LangGraph."""
from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class ReflectionOutput(BaseModel):
    topic: str = Field(description="Domain where the experience applies, e.g., SQL, Python, General")
    insight: str = Field(description="Specific rule or lesson extracted")
    reasoning: str = Field(description="Why this rule is needed")
    should_store: bool = Field(description="Whether it's valuable to store in memory")
    source_query: str | None = Field(
        default=None, description="Original learner question that produced this reflection"
    )


class CriticOutput(BaseModel):
    decision: str = Field(description="Decision result: APPROVE, REVISE, or DISCARD")
    feedback: str = Field(description="If not approved, provide specific revision suggestions; if approved, leave empty")


class ActorOutput(BaseModel):
    thoughts: list[str] = Field(
        description="Ordered chain-of-thought style reasoning steps explaining how the answer was derived"
    )
    answer: str = Field(description="Final response shared with the learner")


class AgentState(TypedDict, total=False):
    query: str
    retrieved_memories: str
    web_results: str
    response: str
    actor_trace: list[str]
    proposed_reflection: ReflectionOutput
    review_decision: str
    critic_feedback: str
    retry_count: int
