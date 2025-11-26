.PHONY: run lint type test lg-dev

run:
	uv run python3 main.py

lint:
	uv run ruff check .

type:
	uv run pyright

test:
	uv run pytest

lg-dev:
	uv run langgraph dev
