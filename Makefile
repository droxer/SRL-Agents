.PHONY: run lint type test lg-dev memory-list memory-delete memory-reset

MEMORY_LIMIT ?= 20

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

memory-list:
	uv run python3 memory_cli.py list --limit $(MEMORY_LIMIT)

memory-delete:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make memory-delete ID=<memory-id>"; \
		exit 1; \
	fi
	uv run python3 memory_cli.py delete $(ID)

memory-reset:
	uv run python3 memory_cli.py reset
