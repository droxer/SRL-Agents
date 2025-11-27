"""Sample scenarios demonstrating the forethought-reflection loop."""
from __future__ import annotations

from rich.console import Console


def run_demo(app):  # type: ignore[override]
    console = Console()
    console.print("[bold cyan]🚀 Scenario 1 · Visible Learning · Git Hygiene[/bold cyan]")
    app.invoke(
        {
            "query": (
                "I'm leading a study group and need a clear recipe for resetting a git repo to a clean slate. "
                "How do I explain the safest way to undo local changes?"
            ),
            "retry_count": 0,
        }
    )

    # console.print("[bold cyan]🚀 Scenario 2 · SRL Evidence · AI Roadmap[/bold cyan]")
    # app.invoke(
    #     {
    #         "query": (
    #             "I'm preparing a presentation on the latest AI safety research. "
    #             "What recent developments should I highlight, and how will I know if my audience understood them?"
    #         ),
    #         "retry_count": 0,
    #     }
    # )
