"""ChromaDB wrapper. Records which embedding model built the index and refuses to mix models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunker import Chunk
from .config import Settings

log = logging.getLogger(__name__)

_BATCH = 1000  # Chroma rejects very large single inserts


class EmbeddingMismatchError(RuntimeError):
    pass


@dataclass
class StoredChunk:
    id: str
    text: str
    metadata: dict
    score: float = 0.0  # cosine similarity for vector hits


class VectorStore:
    def __init__(self, settings: Settings, reset: bool = False):
        self.settings = settings
        self.identity = settings.embedding.identity
        self.name = settings.vector_store.collection
        path = settings.storage_dir / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        if reset:
            self._drop()
        self.collection = self._open()

    def _open(self):
        col = self.client.get_or_create_collection(
            name=self.name,
            # We always pass our own vectors; None stops Chroma loading its default model.
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": self.identity},
        )
        stored = (col.metadata or {}).get("embedding_model")
        if stored != self.identity:
            if col.count() == 0:  # empty index: safe to just re-create with the new model
                self._drop()
                return self._open()
            raise EmbeddingMismatchError(
                f"Index was built with '{stored}' but config uses '{self.identity}'. "
                "Vectors from different models are not comparable. Rebuild with: "
                "python -m rag ingest --rebuild"
            )
        return col

    def _drop(self) -> None:
        try:
            self.client.delete_collection(self.name)
            log.info("Deleted collection %s", self.name)
        except Exception:
            pass  # did not exist

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=embeddings[i : i + _BATCH],
            )

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def count(self) -> int:
        return self.collection.count()

    def query(self, embedding: list[float], n: int) -> list[StoredChunk]:
        n = min(n, self.count())
        if n == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding], n_results=n, include=["documents", "metadatas", "distances"]
        )
        return [
            StoredChunk(id_, doc, meta, score=1.0 - dist)  # cosine distance -> similarity
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get_all(self, where: dict | None = None) -> list[StoredChunk]:
        """All chunks (optionally filtered), paged to keep memory bounded per request."""
        out: list[StoredChunk] = []
        offset, page = 0, 5000
        while True:
            res = self.collection.get(
                where=where, include=["documents", "metadatas"], limit=page, offset=offset
            )
            out += [StoredChunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
            if len(res["ids"]) < page:
                return out
            offset += page
