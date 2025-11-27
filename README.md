# SRL-Agents

SRL-Agents is a LangGraph playground for building Self-Regulated Learning (SRL) agents. Each run walks through SRL's core loop—Forethought, Performance, and Self-Reflection—via specialized graph nodes (Forethought → Web Search → Actor → Reflector → Critic → Store). Use it to prototype metacognitive coaching strategies, remediation flows, or study companions that learn from prior mistakes.

## Prerequisites

1. Python 3.11–3.13 (LangChain’s Pydantic v1 shim is not yet compatible with 3.14+). We recommend 3.13, which matches `.python-version`.
2. `uv` (recommended) or `pip` for dependency management.
3. `OPENAI_API_KEY` exported or stored in `.env`. Optional overrides: `OPENAI_MODEL` (default `gpt-4o`), `OPENAI_TEMPERATURE` (default `0`), `OPENAI_EMBED_MODEL` (default `text-embedding-3-small`), `CHROMA_PERSIST_DIR` (default `.chroma`).

## Installation

```bash
uv sync           # or: pip install -e .
cp .env.example .env  # fill in OpenAI creds
```

## Usage

```bash
make run    # wrapper around `uv run python3 main.py`
make lint   # Ruff static analysis
make type   # Pyright type checking
make test   # Pytest (add tests under tests/)
# LangGraph CLI helper (install with `uv add langgraph-cli` if missing)
make lg-dev   # Run langgraph dev (live reload playground)
make memory-list MEMORY_LIMIT=25  # list stored reflections via make
make memory-delete ID=<memory-id> # delete a single reflection
make memory-reset                 # wipe the memory store
```

All CLI output uses [`rich`](https://github.com/Textualize/rich) for readable, colorized traces of each SRL phase. Demo scripts now live under `examples/`.

LangGraph CLI reads `langgraph.json`, which pins dependencies via `\"-e .\"` and maps the `srl-agents` graph id to `srl_agents.graph:create_app`.

Embed the workflow programmatically:

```python
from srl_agents import create_app
app = create_app()
app.invoke({"query": "我该如何制定复习计划？", "retry_count": 0})
```

### Demo Scenarios

`examples/scenarios.py` now showcases two visible-learning runs:

1. **Git hygiene** – highlights how the Learning Context stage surfaces the learner’s question, goal, success criteria, and prior knowledge before showing a reasoning trace grounded in prior memories.
2. **AI safety roadmap** – triggers the conditional MCP web search, demonstrating how up-to-date evidence enters the ReAct thoughts and how the impact score governs storage.

Run them via `make run` (default) or import `run_demo` in your own scripts to illustrate the SRL loop to teammates.

## Workflow Overview

1. **Learning Context** rewrites the learner’s request into `learning_goal`, `success_criteria`, and `prior_knowledge` so intent is visible.
2. **Forethought** retrieves prior reflections and decides if the criteria demand fresh research (e.g., “latest”, “current”, “recent”).
3. **Web Search** (MCP tool) only runs when Forethought signals a gap, summarizing DuckDuckGo hits into bullet points for the Actor.
4. **Actor (ReAct)** reviews goal, success criteria, memories, and optional web context, emits labeled reasoning thoughts (GOAL/MEMORY/WEB), and concludes with a learner-facing answer. Thoughts are persisted on `actor_trace`.
5. **Reflector** replays the query, answer, and actor trace to distill a reusable rule; if the Critic sends feedback, it retries with that guidance.
6. **Critic** validates the reflection (APPROVE/REVISE/DISCARD) and assigns a 1–5 impact score tied to the success criteria.
7. **Store** only saves reflections meeting the minimum impact score, preserving success criteria + impact metadata for future Forethought runs.

## Extending SRL Behaviors

- Add new node variants (diagnostics, reflection rubrics) in `srl_agents/nodes/`.
- Point `CHROMA_PERSIST_DIR` to a shared volume or swap `MemoryStore` with your own pgvector/ANN implementation via `create_app(memory_store=...)` if you need external persistence.
- Expand `AgentState` with learner metadata (competencies, goals) to personalize prompts.
- Capture evaluation data in `scenarios.py` to benchmark interventions.

### External Web Search (MCP Tooling)

- `srl_agents/tools/web_search.py` implements a DuckDuckGo-backed search tool that complies with Model Context Protocol expectations, making it easy to expose via LangGraph DevTools.
- The graph conditionally runs this tool after the Forethought phase—when no high-similarity memories exist or the learner explicitly needs “latest/current” knowledge—and streams concise bullet summaries into the Actor prompt so the LLM can cite timely facts.
- Network errors are treated as soft failures—the rest of the pipeline still executes and simply omits `web_results`.
- You can swap in other providers by injecting a custom session factory when constructing `WebSearchTool` in `srl_agents/graph.py`.

### ReAct-Style Reasoning & Visibility

- The Actor node follows a ReAct-inspired template: it enumerates labeled reasoning thoughts referencing the visible learning goal, prior memories, or fresh web hits before emitting a final learner answer. These thoughts are printed alongside the answer so learners can audit the chain of evidence.
- Those reasoning traces are stored on the agent state (`actor_trace`) so the Reflector sees the whole chain of thought, yielding clearer metacognitive feedback loops.
- If you build additional MCP tools, you can extend the ReAct protocol in `srl_agents/nodes/actor.py` to let the model request them on-demand.

### Impact Tracking & Memory Governance

- `srl_agents/nodes/critic.py` now requests a 1–5 impact score from the reviewer; the Store node only persists reflections meeting the configured minimum (default 3).
- `srl_agents/memory.py` records `impact_score` and `success_criteria` metadata so Forethought surfaces both relevance and expected learning value.
- `memory_cli.py` shows the new columns so you can audit which reflections matter most.

### Testing Notes

- `tests/test_web_search.py` covers the DuckDuckGo MCP adapter to ensure formatting and empty-query handling stay stable.
- `tests/test_reflector.py` protects the helper that injects Actor reasoning into reflection prompts—update it whenever the ReAct trace format changes.
- `tests/test_memory.py` asserts that impact scores and success criteria flow into stored metadata.
- Run `make test` (or `uv run pytest`) before committing prompt changes so snapshots of reasoning traces stay trustworthy.

Refer to `AGENTS.md` for detailed contributor expectations, coding style, and PR steps.
