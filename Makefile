.PHONY: run lint type test

run:
	uv run python3 main.py

lint:
	uv run ruff check .

type:
	uv run pyright

test:
	uv run pytest
