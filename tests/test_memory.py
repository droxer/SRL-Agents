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
            "impact_score": None,
            "success_criteria": None,
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


def test_add_memory_stores_reflection_correctly():
    """Test that adding a memory stores all reflection components correctly."""
    embedder = DummyEmbedder()
    collection = FakeCollection()
    store = MemoryStore(
        embedder=embedder,
        client=FakeClient(collection),
    )
    reflection = ReflectionOutput(
        topic="Python",
        insight="Use list comprehensions for simple transformations",
        reasoning="More readable and often faster than loops",
        should_store=True,
        source_query="How to transform lists in Python?",
    )

    store.add(reflection)

    # Verify memory was added to collection
    assert len(collection.items) == 1
    stored_item = collection.items[0]
    assert stored_item["metadata"]["topic"] == "Python"
    assert stored_item["metadata"]["insight"] == "Use list comprehensions for simple transformations"
    assert stored_item["metadata"]["reasoning"] == "More readable and often faster than loops"
    assert stored_item["metadata"]["source_query"] == "How to transform lists in Python?"
    assert "Python" in stored_item["document"]
    assert "list comprehensions" in stored_item["document"]

    # Verify it appears in list_memories
    memories = store.list_memories()
    assert len(memories) == 1
    assert memories[0]["topic"] == "Python"
    assert memories[0]["insight"] == "Use list comprehensions for simple transformations"
    assert memories[0]["reasoning"] == "More readable and often faster than loops"
    assert memories[0]["impact_score"] is None
    assert memories[0]["success_criteria"] is None


def test_add_memory_captures_impact_and_success_criteria():
    embedder = DummyEmbedder()
    collection = FakeCollection()
    store = MemoryStore(embedder=embedder, client=FakeClient(collection))
    reflection = ReflectionOutput(
        topic="Testing",
        insight="Write assertions before refactors",
        reasoning="Keeps regressions visible",
        should_store=True,
        source_query="How to avoid regressions?",
    )

    store.add(reflection, impact_score=4, success_criteria="Green tests demonstrate success")

    metadata = collection.items[0]["metadata"]
    assert metadata["impact_score"] == 4
    assert metadata["success_criteria"] == "Green tests demonstrate success"
    memories = store.list_memories()
    assert memories[0]["impact_score"] == 4
    assert memories[0]["success_criteria"] == "Green tests demonstrate success"


def test_add_memory_with_empty_reasoning():
    """Test adding memory when reasoning is empty string."""
    embedder = DummyEmbedder()
    collection = FakeCollection()
    store = MemoryStore(
        embedder=embedder,
        client=FakeClient(collection),
    )
    reflection = ReflectionOutput(
        topic="General",
        insight="Take regular breaks",
        reasoning="",
        should_store=True,
        source_query="How to maintain productivity?",
    )

    store.add(reflection)

    memories = store.list_memories()
    assert len(memories) == 1
    assert memories[0]["topic"] == "General"
    assert memories[0]["insight"] == "Take regular breaks"
    assert memories[0]["reasoning"] == ""


def test_search_finds_relevant_added_memory():
    """Test that searching finds memories that were previously added."""
    embedder = DummyEmbedder()
    collection = FakeCollection()
    store = MemoryStore(
        embedder=embedder,
        client=FakeClient(collection),
        top_k=5,
        min_similarity=0.3,
    )

    # Add multiple memories
    reflection1 = ReflectionOutput(
        topic="SQL",
        insight="Use indexes for large table joins",
        reasoning="Speeds up query execution",
        should_store=True,
        source_query="How to optimize SQL queries?",
    )
    reflection2 = ReflectionOutput(
        topic="Python",
        insight="Use generators for large datasets",
        reasoning="Saves memory",
        should_store=True,
        source_query="How to process large files?",
    )
    reflection3 = ReflectionOutput(
        topic="General",
        insight="Review code before committing",
        reasoning="Catches bugs early",
        should_store=True,
        source_query="What's a good workflow?",
    )

    store.add(reflection1)
    store.add(reflection2)
    store.add(reflection3)

    # Verify all memories are stored
    assert len(collection.items) == 3

    # Mock search results to return relevant memory
    collection.query_result = {
        "metadatas": [
            [
                {"topic": "SQL", "insight": "Use indexes for large table joins"},
                {"topic": "Python", "insight": "Use generators for large datasets"},
            ]
        ],
        "documents": [
            [
                "SQL. Use indexes for large table joins. Speeds up query execution. How to optimize SQL queries?",
                "Python. Use generators for large datasets. Saves memory. How to process large files?",
            ]
        ],
        "distances": [[0.2, 0.4]],  # Low distance = high similarity
    }

    result = store.search("optimize database queries")

    # Should find SQL-related memory with high similarity
    assert "SQL" in result
    assert "Use indexes for large table joins" in result
    assert "0.80" in result  # similarity = 1 - 0.2 = 0.8


def test_search_returns_multiple_relevant_memories():
    """Test that search returns multiple memories when they meet similarity threshold."""
    embedder = DummyEmbedder()
    collection = FakeCollection()
    store = MemoryStore(
        embedder=embedder,
        client=FakeClient(collection),
        top_k=5,
        min_similarity=0.5,
    )

    # Mock search results with multiple high-similarity matches
    collection.query_result = {
        "metadatas": [
            [
                {"topic": "SQL", "insight": "Use indexes for joins"},
                {"topic": "SQL", "insight": "Analyze query plans"},
                {"topic": "General", "insight": "Test thoroughly"},
            ]
        ],
        "documents": [["SQL doc 1", "SQL doc 2", "General doc"]],
        "distances": [[0.1, 0.2, 0.7]],  # First two are relevant
    }

    result = store.search("SQL optimization")

    # Should return both SQL memories (similarity 0.9 and 0.8)
    assert result.count("- [SQL]") == 2
    assert "Use indexes for joins" in result
    assert "Analyze query plans" in result
    # General memory should not appear (similarity 0.3 < 0.5)
    assert "Test thoroughly" not in result


def test_add_and_search_integration():
    """End-to-end test: add memory, then search for it."""
    embedder = DummyEmbedder()
    collection = FakeCollection()
    store = MemoryStore(
        embedder=embedder,
        client=FakeClient(collection),
        top_k=3,
        min_similarity=0.4,
    )

    # Add a memory
    reflection = ReflectionOutput(
        topic="Testing",
        insight="Write tests before fixing bugs",
        reasoning="Ensures the fix works",
        should_store=True,
        source_query="How to debug effectively?",
    )
    store.add(reflection)

    # Verify it was stored
    assert len(collection.items) == 1

    # Mock search to return the stored memory
    collection.query_result = {
        "metadatas": [[{"topic": "Testing", "insight": "Write tests before fixing bugs"}]],
        "documents": [["Testing. Write tests before fixing bugs. Ensures the fix works. How to debug effectively?"]],
        "distances": [[0.15]],  # High similarity
    }

    result = store.search("debugging workflow")

    # Should find the relevant memory
    assert "Testing" in result
    assert "Write tests before fixing bugs" in result
    assert "0.85" in result  # similarity = 1 - 0.15 = 0.85
