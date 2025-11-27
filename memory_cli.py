"""Standalone CLI for managing SRL memory entries."""
from __future__ import annotations

import argparse

from rich.table import Table

from srl_agents.config import get_embeddings, get_vector_client
from srl_agents.logging import console
from srl_agents.memory import MemoryStore


def build_memory_store() -> MemoryStore:
    """Instantiate a MemoryStore backed by the shared Chroma client."""
    return MemoryStore(embedder=get_embeddings(), client=get_vector_client())


def list_memories(store: MemoryStore, limit: int) -> None:
    records = store.list_memories(limit=limit)
    if not records:
        console.print("[yellow]No stored memories.[/yellow]")
        return
    table = Table(title="Stored memories", show_lines=False)
    table.add_column("ID", style="bold")
    table.add_column("Topic", style="magenta")
    table.add_column("Insight", overflow="fold")
    table.add_column("Reasoning", overflow="fold")
    table.add_column("Impact", style="cyan")
    table.add_column("Success Criteria", overflow="fold")
    table.add_column("Document", overflow="fold")

    for record in records:
        table.add_row(
            record["id"],
            record.get("topic", "General") or "General",
            record.get("insight", "") or "",
            record.get("reasoning", "") or "",
            str(record.get("impact_score", "") or ""),
            record.get("success_criteria", "") or "",
            record.get("document", "") or "",
        )
    console.print(table)


def delete_memory(store: MemoryStore, memory_id: str) -> None:
    if store.delete_memory(memory_id):
        console.print(f"[green]Deleted memory {memory_id}.[/green]")
    else:
        console.print(f"[yellow]No memory found with id {memory_id}.[/yellow]")


def reset_memory(store: MemoryStore) -> None:
    deleted = store.reset_memory()
    console.print(f"[green]Cleared {deleted} stored memories.[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage SRL reflection memory.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="List stored reflections")
    list_parser.add_argument("--limit", type=int, default=20, help="How many entries to display")

    delete_parser = subparsers.add_parser("delete", help="Delete a stored reflection by id")
    delete_parser.add_argument("id", help="Memory identifier returned by the list command")

    subparsers.add_parser("reset", help="Delete every stored reflection")

    args = parser.parse_args()

    store = build_memory_store()
    action = args.action
    if action == "list":
        list_memories(store, args.limit)
    elif action == "delete":
        delete_memory(store, args.id)
    elif action == "reset":
        reset_memory(store)
    else:  # pragma: no cover
        console.print(f"[red]Unknown action: {action}[/red]")


if __name__ == "__main__":
    main()
