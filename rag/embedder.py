"""Embedding backends behind one interface; selected by embedding.provider in config.yaml."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from .config import EmbeddingConfig

log = logging.getLogger(__name__)


def load_hf_model(cls, name: str, device: str):
    """Load a sentence-transformers model from the local cache without any network calls;
    download only if it isn't cached yet. Saves many seconds per start on slow connections."""
    try:
        return cls(name, device=device, local_files_only=True)
    except Exception:
        log.info("%s not in local cache; downloading", name)
        return cls(name, device=device)


class Embedder(ABC):
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.identity = cfg.identity  # e.g. "local:BAAI/bge-small-en-v1.5"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder(Embedder):
    """sentence-transformers model on CPU/GPU. Loaded lazily (a few seconds on first use)."""

    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s on %s", self.cfg.local.model, self.cfg.local.device)
            self._model = load_hf_model(SentenceTransformer, self.cfg.local.model, self.cfg.local.device)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.cfg.local.batch_size,
            normalize_embeddings=True,  # cosine similarity == dot product
            show_progress_bar=len(texts) > 4 * self.cfg.local.batch_size,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.cfg.local.query_prefix + text])[0]


class VoyageEmbedder(Embedder):
    """Voyage AI API (Anthropic's recommended embedding provider). Needs VOYAGE_API_KEY."""

    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        if not os.getenv("VOYAGE_API_KEY"):
            raise RuntimeError("embedding.provider is 'voyage' but VOYAGE_API_KEY is not set in .env")
        import voyageai

        self.client = voyageai.Client(max_retries=4)  # retries 429 / 5xx with backoff

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        bs = self.cfg.voyage.batch_size
        for i in range(0, len(texts), bs):
            result = self.client.embed(
                texts[i : i + bs], model=self.cfg.voyage.model, input_type=input_type
            )
            out.extend(result.embeddings)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


def get_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "local":
        return LocalEmbedder(cfg)
    if cfg.provider == "voyage":
        return VoyageEmbedder(cfg)
    raise ValueError(f"Unknown embedding provider: {cfg.provider}")
