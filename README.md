# SRL-Agents

SRL-Agents is a LangGraph playground for building Self-Regulated Learning (SRL) agents. Each run walks through SRL's core loop—Forethought, Performance, and Self-Reflection—via specialized graph nodes (Forethought → Actor → Reflector → Critic → Store). Use it to prototype metacognitive coaching strategies, remediation flows, or study companions that learn from prior mistakes.

## Project Structure

- `srl_agents/config.py` – loads `.env` variables and returns a shared OpenAI client.
- `srl_agents/state.py` – typed state plus Reflection/Critic Pydantic models.
- `srl_agents/memory.py` – pluggable memory adapter (in-memory mock by default).
- `srl_agents/nodes/` – one module per SRL phase; easy to extend or swap.
- `srl_agents/logging.py` – shared Rich console instance for pretty output.
- `srl_agents/graph.py` – builds the LangGraph via `create_app()`.
- `examples/scenarios.py` – canned demos mimicking SRL tutoring sessions.
- `main.py` – CLI entry point that plays demos or accepts a custom `--query`.
- `AGENTS.md` – contributor workflow and coding guidelines.

## Prerequisites

1. Python 3.11–3.13 (LangChain’s Pydantic v1 shim is not yet compatible with 3.14+). We recommend 3.13, which matches `.python-version`.
2. `uv` (recommended) or `pip` for dependency management.
3. `OPENAI_API_KEY` exported or stored in `.env`. Optional overrides: `OPENAI_MODEL` (default `gpt-4o`), `OPENAI_TEMPERATURE` (default `0`).

## Installation

```bash
uv sync           # or: pip install -e .
cp .env.example .env  # fill in OpenAI creds
```

## Usage

```bash
python main.py                   # run demo scenarios
python main.py --query "Plan my SRL study session"

make run    # wrapper around `uv run python3 main.py`
make lint   # Ruff static analysis
make type   # Pyright type checking
make test   # Pytest (add tests under tests/)
# LangGraph CLI helper (install with `uv add langgraph-cli` if missing)
make lg-dev   # Run langgraph dev (live reload playground)
```

All CLI output uses [`rich`](https://github.com/Textualize/rich) for readable, colorized traces of each SRL phase. Demo scripts now live under `examples/`.

LangGraph CLI reads `langgraph.json`, which pins dependencies via `\"-e .\"` and maps the `srl-agents` graph id to `srl_agents.graph:create_app`.

Embed the workflow programmatically:

```python
from srl_agents import create_app
app = create_app()
app.invoke({"query": "我该如何制定复习计划？", "retry_count": 0})
```

## Extending SRL Behaviors

- Add new node variants (diagnostics, reflection rubrics) in `srl_agents/nodes/`.
- Replace `MemoryStore` with pgvector or other embeddings DB for longitudinal tracking.
- Expand `AgentState` with learner metadata (competencies, goals) to personalize prompts.
- Capture evaluation data in `scenarios.py` to benchmark interventions.

Refer to `AGENTS.md` for detailed contributor expectations, coding style, and PR steps.
