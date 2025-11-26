"""ChromaDB-backed memory store for SRL agents."""
from __future__ import annotations

from typing import Callable, List, Optional, TypedDict
from uuid import uuid4

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings

from .logging import console
from .state import ReflectionOutput

QueryRefiner = Callable[[str], str]


class MemoryRecord(TypedDict, total=False):
    id: str
    topic: str
    insight: str
    reasoning: str | None
    document: str | None


class MemoryStore:
    """Vector database wrapper using Chroma collections."""

    def __init__(
        self,
        embedder: Optional[Embeddings],
        client: ClientAPI,
        collection_name: str = "srl-memory",
        top_k: int = 3,
        min_similarity: Optional[float] = 0.35,
        query_refiner: QueryRefiner | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection: Collection = client.get_or_create_collection(collection_name)
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.query_refiner = query_refiner

    def search(self, query: str) -> str:
        """Return top similar memories from Chroma."""
        if not self.embedder:
            return "Memory retrieval unavailable (missing embedding client)."

        normalized_query = self._normalize_text(query, context="search query")
        query_vec = self._embed_query(normalized_query)
        if query_vec is None:
            return "No relevant past experience."

        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=self.top_k,
            include=["metadatas", "documents", "distances"],
        )
        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or []
        distances = result.get("distances") or []
        if not metadatas or not any(metadatas):
            return "No relevant past experience."

        lines: list[str] = []
        fallback_lines: list[str] = []
        for meta_list, doc_list, dist_list in zip(metadatas, documents, distances):
            for meta, doc, dist in zip(meta_list or [], doc_list or [], dist_list or []):
                topic = meta.get("topic", "General")
                insight = meta.get("insight", doc)
                score = self._distance_to_similarity(dist)
                if self.min_similarity is not None and (
                    score is None or score < self.min_similarity
                ):
                    fallback_lines.append(self._format_memory_line(topic, insight, score))
                    continue
                lines.append(self._format_memory_line(topic, insight, score))

        if lines:
            return "\n".join(lines)
        if fallback_lines:
            console.print("[yellow]No high-similarity matches; showing closest memory.[/yellow]")
            return "\n".join(fallback_lines[: self.top_k])
        return "No relevant past experience."

    def add(self, reflection: ReflectionOutput) -> None:
        """Persist reflections as embedded documents."""
        if not self.embedder:
            console.print("[red]Cannot store reflection: embedding client not configured.[/red]")
            return

        components = [
            reflection.topic,
            reflection.insight,
            reflection.reasoning,
            reflection.source_query or "",
        ]
        text = ". ".join(filter(None, components))
        normalized_text = self._normalize_text(text, context="reflection")
        embedding = self._embed_query(normalized_text)
        if embedding is None:
            console.print("[red]Skipping persistence due to embedding failure.[/red]")
            return

        console.print(
            f"[green]\n[Database] 💾 Persisting: [{reflection.topic}] {reflection.insight}[/green]"
        )
        self.collection.add(
            ids=[str(uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[
                {
                    "topic": reflection.topic,
                    "insight": reflection.insight,
                    "reasoning": reflection.reasoning,
                    "source_query": reflection.source_query,
                }
            ],
        )

    def list_memories(self, limit: int = 50) -> List[MemoryRecord]:
        """Return stored memories for CLI inspection."""
        result = self.collection.get(include=["metadatas", "documents"], limit=limit)
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or [None] * len(ids)
        documents = result.get("documents") or [None] * len(ids)

        if len(metadatas) < len(ids):
            metadatas += [None] * (len(ids) - len(metadatas))
        if len(documents) < len(ids):
            documents += [None] * (len(ids) - len(documents))

        records: List[MemoryRecord] = []
        for mem_id, meta, doc in zip(ids, metadatas, documents):
            meta = meta or {}
            records.append(
                MemoryRecord(
                    id=mem_id,
                    topic=meta.get("topic", "General"),
                    insight=meta.get("insight", doc),
                    reasoning=meta.get("reasoning"),
                    document=doc,
                )
            )
        return records

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory entry by id."""
        if not memory_id:
            return False
        existing = self.collection.get(ids=[memory_id], include=[])
        if not existing.get("ids"):
            return False
        self.collection.delete(ids=[memory_id])
        return True

    def reset_memory(self, batch_size: int = 200) -> int:
        """Remove every stored memory."""
        total_deleted = 0
        while True:
            batch = self.collection.get(include=[], limit=batch_size)
            ids = batch.get("ids") or []
            if not ids:
                break
            self.collection.delete(ids=ids)
            total_deleted += len(ids)
            if len(ids) < batch_size:
                break
        return total_deleted

    def _embed_query(self, text: str):
        try:
            return self.embedder.embed_query(text) if self.embedder else None
        except Exception as exc:  # pragma: no cover
            console.print(f"[red]Embedding failed:[/red] {exc}")
            return None

    def _normalize_text(self, text: str, context: str) -> str:
        if not self.query_refiner:
            return text
        try:
            refined = self.query_refiner(text)
            if refined and refined != text:
                label = "Refined memory query" if context == "search query" else "Refined reflection embedding"
                console.print(
                    f"[dim]{label}:[/dim] "
                    f"{refined if len(refined) < 160 else refined[:157] + '...'}"
                )
            return refined or text
        except Exception as exc:  # pragma: no cover
            console.print(f"[yellow]Query refinement failed:[/yellow] {exc}")
            return text

    @staticmethod
    def _distance_to_similarity(distance):
        if isinstance(distance, (int, float)):
            return float(1 - distance)
        return None

    @staticmethod
    def _format_memory_line(topic: str, insight: str, score: float | None) -> str:
        score_txt = f" (score: {score:.2f})" if score is not None else ""
        return f"- [{topic}] {insight}{score_txt}"
