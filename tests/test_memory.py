"""Unit tests for the in-memory Chroma adapter."""
from __future__ import annotations

from srl_agents.memory import MemoryStore
from srl_agents.state import ReflectionOutput


class DummyEmbedder:
    def __init__(self):
        self.last_query = None

    def embed_query(self, text: str):
        self.last_query = text
        return [0.0]


class FakeCollection:
    def __init__(self, query_result=None, items=None):
        self.query_result = query_result
        self.items = items or []

    def query(self, *, query_embeddings, n_results, include):
        if self.query_result is None:
            raise AssertionError("query_result not configured")
        self.last_query = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include,
        }
        return self.query_result

    def get(self, *, ids=None, where=None, limit=None, include=None, offset=0):
        if ids is not None:
            data = [item for item in self.items if item["id"] in ids]
        else:
            data = self.items[offset:]
            if limit is not None:
                data = data[:limit]
        return {
            "ids": [item["id"] for item in data],
            "metadatas": [item.get("metadata") for item in data],
            "documents": [item.get("document") for item in data],
        }

    def delete(self, *, ids=None, where=None, where_document=None):
        if ids is None:
            raise AssertionError("ids must be provided")
        before = len(self.items)
        self.items = [item for item in self.items if item["id"] not in ids]
        return before - len(self.items)

    def add(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for idx, metadata, document in zip(ids, metadatas or [], documents or []):
            self.items.append({"id": idx, "metadata": metadata, "document": document})


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, name: str):
        return self.collection


def test_search_filters_results_with_low_similarity():
    collection = FakeCollection(
        {
            "metadatas": [
                [
                    {"topic": "SQL", "insight": "Use indexes for large joins"},
                    {"topic": "General", "insight": "Take breaks every hour"},
                ]
            ],
            "documents": [["SQL doc", "General doc"]],
            "distances": [[0.1, 0.9]],
        }
    )
    store = MemoryStore(
        embedder=DummyEmbedder(),
        client=FakeClient(collection),
        top_k=5,
        min_similarity=0.5,
    )

    result = store.search("optimize SQL performance")

    assert "- [SQL] Use indexes for large joins (score: 0.90)" in result
    assert "Take breaks every hour" not in result


def test_search_returns_fallback_when_low_similarity():
    collection = FakeCollection(
        {
            "metadatas": [[{"topic": "General", "insight": "Stay hydrated"}]],
            "documents": [["General doc"]],
            "distances": [[0.95]],
        }
    )
    store = MemoryStore(
        embedder=DummyEmbedder(),
        client=FakeClient(collection),
        min_similarity=0.8,
    )

    result = store.search("python testing strategy")

    assert result == "- [General] Stay hydrated (score: 0.05)"


def test_list_memories_returns_structured_rows():
    collection = FakeCollection(
        items=[
            {
                "id": "mem-1",
                "metadata": {
                    "topic": "Python",
                    "insight": "Prefer pathlib over os.path",
                    "reasoning": "Pathlib is cross-platform",
                },
                "document": "Python. Prefer pathlib over os.path",
            }
        ]
    )
    store = MemoryStore(embedder=None, client=FakeClient(collection))

    rows = store.list_memories()

    assert rows == [
        {
            "id": "mem-1",
            "topic": "Python",
            "insight": "Prefer pathlib over os.path",
            "reasoning": "Pathlib is cross-platform",
            "document": "Python. Prefer pathlib over os.path",
        }
    ]


def test_delete_memory_removes_entry():
    collection = FakeCollection(
        items=[
            {"id": "keep", "metadata": {"topic": "SQL", "insight": "Use explain"}, "document": "SQL doc"},
            {"id": "drop", "metadata": {"topic": "General", "insight": "Take breaks"}, "document": "General doc"},
        ]
    )
    store = MemoryStore(embedder=None, client=FakeClient(collection))

    assert store.delete_memory("drop") is True
    assert store.delete_memory("missing") is False
    remaining = store.list_memories()
    assert remaining[0]["id"] == "keep"


def test_reset_memory_clears_all_entries():
    collection = FakeCollection(
        items=[
            {"id": f"mem-{i}", "metadata": {"topic": "T", "insight": f"note-{i}"}, "document": f"doc-{i}"}
            for i in range(3)
        ]
    )
    store = MemoryStore(embedder=None, client=FakeClient(collection))

    deleted = store.reset_memory(batch_size=2)

    assert deleted == 3
    assert store.list_memories() == []


def test_search_uses_refined_query_for_embedding():
    embedder = DummyEmbedder()
    collection = FakeCollection(
        {
            "metadatas": [[{"topic": "General", "insight": "Practice estimation"}]],
            "documents": [["General doc"]],
            "distances": [[0.2]],
        }
    )
    store = MemoryStore(
        embedder=embedder,
        client=FakeClient(collection),
        query_refiner=lambda q: f"{q} refined",
    )

    store.search("initial question")

    assert embedder.last_query == "initial question refined"


def test_add_uses_same_refiner_path_as_search():
    embedder = DummyEmbedder()
    collection = FakeCollection()
    store = MemoryStore(
        embedder=embedder,
        client=FakeClient(collection),
        query_refiner=lambda text: f"{text} normalized",
    )
    reflection = ReflectionOutput(
        topic="SQL",
        insight="Join plans often need indexes",
        reasoning="Avoids nested loop scans",
        should_store=True,
        source_query="How to optimize SQL joins?",
    )

    store.add(reflection)

    assert embedder.last_query.endswith("normalized")
