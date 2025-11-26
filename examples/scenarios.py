"""Sample scenarios demonstrating the forethought-reflection loop."""
from __future__ import annotations

from rich.console import Console


def run_demo(app):  # type: ignore[override]
    console = Console()
    console.print("[bold cyan]🚀 Scenario 1 · First Encounter[/bold cyan]")
    app.invoke({"query": "What command should I use to clear all git modifications in the current directory?", "retry_count": 0})

    console.print("[bold cyan]🚀 Scenario 2 · Similar Problem[/bold cyan]")
    app.invoke({"query": "I want to delete all files that haven't been committed, how do I do that?", "retry_count": 0})
