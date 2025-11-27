"""Tests for the forethought helper logic."""
from __future__ import annotations

from srl_agents.nodes.forethought import _should_research
from srl_agents.state import LearningContext


def test_should_research_when_no_memories():
    context = LearningContext(
        learning_goal="Understand git cleanup",
        success_criteria="I can explain the correct command",
        prior_knowledge="Knows git basics",
    )

    assert _should_research("No relevant past experience.", context) is True


def test_should_research_when_success_criteria_requires_fresh_data():
    context = LearningContext(
        learning_goal="Track the latest numpy version",
        success_criteria="I cite the latest current release",
        prior_knowledge="Understands pip",
    )

    assert _should_research("- [Python] Some cached tip", context) is True


def test_should_not_research_when_memories_suffice():
    context = LearningContext(
        learning_goal="Handle git resets",
        success_criteria="I can run reset safely",
        prior_knowledge="Some CLI knowledge",
    )

    assert _should_research("- [Git] Use git status", context) is False
