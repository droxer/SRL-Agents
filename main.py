"""CLI entry point for SRL LangGraph demo."""
from __future__ import annotations

import argparse

from srl_agents import create_app
from examples.scenarios import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Reflection LangGraph demo")
    parser.add_argument(
        "--query",
        help="If provided, run the graph once with this query instead of the predefined scenarios.",
    )
    args = parser.parse_args()

    app = create_app()
    if args.query:
        app.invoke({"query": args.query, "retry_count": 0})
    else:
        run_demo(app)


if __name__ == "__main__":
    main()
