"""In-memory stand-in for the production memory store."""
from __future__ import annotations

from typing import List

from .logging import console
from .state import ReflectionOutput


class MemoryStore:
    """Simulates vector-backed long-term memory."""

    def __init__(self) -> None:
        self.memories: List[dict] = []

    def search(self, query: str) -> str:
        """Return matching memories for a query via simple keyword search."""
        results = []
        lowered = query.lower()
        for mem in self.memories:
            if mem["topic"].lower() in lowered or "general" in mem["topic"].lower():
                results.append(f"- [{mem['topic']}] {mem['insight']}")

        if not results:
            return "No relevant past experience."
        return "\n".join(results)

    def add(self, reflection: ReflectionOutput) -> None:
        """Persist a new reflection in the simulated store."""
        console.print(
            f"[green]\\n[Database] 💾 Persisting: [{reflection.topic}] {reflection.insight}[/green]"
        )
        self.memories.append(
            {
                "topic": reflection.topic,
                "insight": reflection.insight,
                "reasoning": reflection.reasoning,
            }
        )
