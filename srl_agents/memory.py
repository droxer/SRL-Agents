"""ChromaDB-backed memory store for SRL agents."""
from __future__ import annotations

from typing import Optional
from uuid import uuid4

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings

from .logging import console
from .state import ReflectionOutput


class MemoryStore:
    """Vector database wrapper using Chroma collections."""

    def __init__(
        self,
        embedder: Optional[Embeddings],
        client: ClientAPI,
        collection_name: str = "srl-memory",
        top_k: int = 3,
    ) -> None:
        self.embedder = embedder
        self.collection: Collection = client.get_or_create_collection(collection_name)
        self.top_k = top_k

    def search(self, query: str) -> str:
        """Return top similar memories from Chroma."""
        if not self.embedder:
            return "Memory retrieval unavailable (missing embedding client)."

        query_vec = self._embed_query(query)
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

        lines = []
        for meta_list, doc_list, dist_list in zip(metadatas, documents, distances):
            for meta, doc, dist in zip(meta_list or [], doc_list or [], dist_list or []):
                topic = meta.get("topic", "General")
                insight = meta.get("insight", doc)
                score = (1 - dist) if isinstance(dist, (int, float)) else None
                score_txt = f" (score: {score:.2f})" if isinstance(score, float) else ""
                lines.append(f"- [{topic}] {insight}{score_txt}")

        return "\n".join(lines) if lines else "No relevant past experience."

    def add(self, reflection: ReflectionOutput) -> None:
        """Persist reflections as embedded documents."""
        if not self.embedder:
            console.print("[red]Cannot store reflection: embedding client not configured.[/red]")
            return

        text = f"{reflection.topic}. {reflection.insight}"
        embedding = self._embed_query(text)
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
                }
            ],
        )

    def _embed_query(self, text: str):
        try:
            return self.embedder.embed_query(text) if self.embedder else None
        except Exception as exc:  # pragma: no cover
            console.print(f"[red]Embedding failed:[/red] {exc}")
            return None
