# Repository Guidelines

## Project Structure & Module Organization
- `srl_agents/` is the package containing configuration (`config.py`), typed state models (`state.py`), the in-memory memory adapter (`memory.py`), node implementations (`nodes/`), the LangGraph builder (`graph.py`), and demo scripts (`scenarios.py`).
- `main.py` is the CLI entry point; it wires command-line arguments into the compiled graph.
- `sql/` houses any initialization scripts for backing stores; keep additional assets in sibling folders when needed.
- Tests live beside the code they exercise (e.g., `tests/test_graph.py`); create the directory if you add automated coverage.

## Build, Test, and Development Commands
- `make run` – executes `uv run python3 main.py`, driving either the canned scenarios or a provided `--query`.
- `make lint` – runs Ruff against the repository; fix diagnostics before committing.
- `make type` – runs Pyright for static type checking; prefer annotating new functions.
- `make test` – runs Pytest; ensure meaningful coverage for new behaviors.

## Coding Style & Naming Conventions
- Python files use Black-style 4-space indentation; prefer descriptive snake_case for variables/functions and PascalCase for classes.
- Follow Ruff’s default configuration; run `make lint` before sending a PR.
- Keep module-level side effects minimal. Export reusable functions (e.g., `create_app`) from `srl_agents/__init__.py` for clarity.

## Testing Guidelines
- Write Pytest tests that mirror user workflows (graph compilation, node behavior, memory interactions).
- Name test files `test_*.py` and target specific failure modes (e.g., critic rejecting unsafe reflections).
- When touching multiple components, add integration tests that invoke the LangGraph end-to-end via `create_app()`.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (`Add reflector node retries`, `Document Makefile workflow`).
- Each PR should describe motivation, summarize major changes, list testing performed (`make lint`, `make test`), and link issues or tasks when applicable.
- Include screenshots or logs when modifying user-visible CLI output or SQL assets.

## Security & Configuration Tips
- Store `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_TEMPERATURE` in a `.env` file or your shell environment; never commit secrets.
- Use separate API keys for development and production demos, and rotate them regularly.
