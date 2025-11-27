"""Tests for reflector helpers."""
from __future__ import annotations

from srl_agents.nodes.reflector import _build_reflection_messages


def test_build_reflection_messages_includes_actor_trace():
    messages = _build_reflection_messages(
        system_msg="sys",
        query="Why is the sky blue?",
        response="Short answer.",
        actor_trace=["Consider Rayleigh scattering", "Reference (Web) article"],
    )

    assert messages[0][0] == "system"
    assert "Reasoning steps:" in messages[1][1]
    assert "Thought 1: Consider Rayleigh scattering" in messages[1][1]
    assert "Thought 2: Reference (Web) article" in messages[1][1]


def test_build_reflection_messages_includes_actions_and_observations():
    messages = _build_reflection_messages(
        system_msg="sys",
        query="Test query",
        response="Test response",
        actor_trace=["Step 1"],
        retrieved_memories="Memory content here",
        web_results="Web search results",
    )

    assert "Action: Retrieved memories" in messages[1][1]
    assert "Observation: Memory content here" in messages[1][1]
    assert "Action: Web search" in messages[1][1]
    assert "Observation: Web search results" in messages[1][1]
