# Repository Guidelines

## Project Structure & Module Organization
- `srl_agents/` is the package containing configuration (`config.py`), typed state models (`state.py`), the in-memory memory adapter (`memory.py`), node implementations (`nodes/`), the LangGraph builder (`graph.py`), and logging utilities.
- `srl_agents/tools/` houses MCP-friendly adapters (e.g., DuckDuckGo web search) that nodes can call for external context.
- `examples/` houses runnable demos such as `examples/scenarios.py`; keep additional showcase scripts here instead of inside the package, and ensure they demonstrate the visible-learning surfaces (learning goals, reasoning traces, impact scores).
- `main.py` is the CLI entry point; it wires command-line arguments into the compiled graph.
- Tests live beside the code they exercise (e.g., `tests/test_graph.py`); create the directory if you add automated coverage.

## Build, Test, and Development Commands
- `make run` – executes `uv run python3 main.py`, driving either the canned scenarios or a provided `--query`.
- `make lint` – runs Ruff against the repository; fix diagnostics before committing.
- `make type` – runs Pyright for static type checking; prefer annotating new functions.
- `make test` – runs Pytest; ensure meaningful coverage for new behaviors.
- `make lg-dev` – wrapper around the LangGraph CLI dev server. Install it first (`uv add langgraph-cli`) so the command resolves.
  - The CLI reads `langgraph.json`, which installs dependencies via `"-e ."` and maps the `srl-agents` graph id directly to `srl_agents.graph:create_app`.

## Coding Style & Naming Conventions
- Python files use Black-style 4-space indentation; prefer descriptive snake_case for variables/functions and PascalCase for classes.
- Follow Ruff’s default configuration; run `make lint` before sending a PR.
- Keep module-level side effects minimal. Export reusable functions (e.g., `create_app`) from `srl_agents/__init__.py` for clarity.
- Prompts inside the SRL graph follow a ReAct-style flow: Actor outputs must include `thoughts` plus a final `answer`, and Reflector expects `actor_trace` to remain a list of concise strings. Update corresponding Pydantic models/tests if you adjust this schema.
- Visible-learning surfaces (`LearningContext`, `needs_research`, `impact_score`) are first-class: if you add new goal metadata or scoring dimensions, extend `srl_agents/state.py`, the learning-context node, Critic/store prompts, and the related tests in one change.

## Testing Guidelines
- Write Pytest tests that mirror user workflows (graph compilation, node behavior, memory interactions).
- Name test files `test_*.py` and target specific failure modes (e.g., critic rejecting unsafe reflections).
- When touching multiple components, add integration tests that invoke the LangGraph end-to-end via `create_app()`.
- Keep `tests/test_web_search.py` and `tests/test_reflector.py` current when modifying MCP tooling or ReAct prompts—they ensure reasoning traces and tool summaries stay machine-checkable.
- `tests/test_memory.py` must be updated whenever metadata stored in Chroma changes (impact scores, success criteria, etc.) so that CLI reports stay trustworthy.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (`Add reflector node retries`, `Document Makefile workflow`).
- Each PR should describe motivation, summarize major changes, list testing performed (`make lint`, `make test`), and link issues or tasks when applicable.
- Include screenshots or logs when modifying user-visible CLI output or SQL assets.

## Security & Configuration Tips
- Store `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_TEMPERATURE` in a `.env` file or your shell environment; never commit secrets.
- Use separate API keys for development and production demos, and rotate them regularly.
