"""Hybrid retrieval: vector search + BM25, fused with Reciprocal Rank Fusion, then cross-encoder re-rank.

Why hybrid: embeddings catch paraphrases ("how big were the chunks" ~ "chunk size"), BM25 catches
exact tokens embeddings blur (model names, "3e-5", dataset names, equation symbols).
RRF merges the two rankings without having to calibrate their very different score scales.
The cross-encoder then reads (question, chunk) pairs jointly, which is far more precise than
either first-stage score, but too slow to run on every chunk — so only on the fused top-N.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .config import Settings
from .embedder import Embedder, load_hf_model
from .vector_store import StoredChunk, VectorStore

log = logging.getLogger(__name__)

_STOPWORDS = set(
    "a an and are as at be but by for from has have how in is it its of on or that the this to was "
    "were what when where which who why will with does did do can could should would about into than "
    "then there these those they their our we you your i".split()
)
# Keep tokens like "bge-small", "3e-5", "gpt-4.1" intact.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def source(self) -> str:
        return self.metadata["source"]

    @property
    def page_label(self) -> str:
        a, b = self.metadata["page_start"], self.metadata["page_end"]
        return f"p. {a}" if a == b else f"pp. {a}-{b}"

    def scores(self) -> dict:
        return {
            "vector": _round(self.vector_score),
            "bm25": _round(self.bm25_score),
            "rrf": _round(self.rrf_score),
            "rerank": _round(self.rerank_score),
        }


def _round(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def _search_text(c: StoredChunk) -> str:
    m = c.metadata
    return f"{m.get('title', '')} {m.get('section', '')} {c.text}"


class BM25Index:
    """In-memory BM25 over all chunks. Rebuilt from Chroma when the chunk count changes (<1 s at 20k chunks)."""

    def __init__(self, store: VectorStore):
        self.store = store
        self._count = -1
        self._chunks: list[StoredChunk] = []
        self._bm25: BM25Okapi | None = None

    def _ensure(self) -> None:
        count = self.store.count()
        if count == self._count:
            return
        self._chunks = self.store.get_all()
        self._bm25 = BM25Okapi([tokenize(_search_text(c)) for c in self._chunks]) if self._chunks else None
        self._count = count
        log.info("BM25 index built over %d chunks", count)

    def invalidate(self) -> None:
        self._count = -1

    def search(self, query: str, n: int) -> list[tuple[StoredChunk, float]]:
        self._ensure()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self._chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name, self.device = model_name, device
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("Loading re-ranker %s", self.model_name)
            self._model = load_hf_model(CrossEncoder, self.model_name, self.device)
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.bm25 = BM25Index(store)
        self.reranker = Reranker(settings.retrieval.rerank.model, settings.embedding.local.device)

    def refresh(self, store: VectorStore) -> None:
        """Call after ingestion so BM25 sees new/removed chunks."""
        self.store = store
        self.bm25 = BM25Index(store)

    def retrieve(self, query: str, top_k: int | None = None, rerank: bool | None = None) -> list[RetrievedChunk]:
        cfg = self.settings.retrieval
        top_k = top_k or cfg.top_k
        rerank = cfg.rerank.enabled if rerank is None else rerank
        if self.store.count() == 0:
            return []

        vector_hits = self.store.query(self.embedder.embed_query(query), cfg.vector_candidates)
        bm25_hits = self.bm25.search(query, cfg.bm25_candidates)

        # Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank).
        fused: dict[str, RetrievedChunk] = {}
        for rank, hit in enumerate(vector_hits, start=1):
            rc = fused.setdefault(hit.id, RetrievedChunk(hit.id, hit.text, hit.metadata))
            rc.vector_score = hit.score
            rc.rrf_score += 1.0 / (cfg.rrf_k + rank)
        for rank, (hit, score) in enumerate(bm25_hits, start=1):
            rc = fused.setdefault(hit.id, RetrievedChunk(hit.id, hit.text, hit.metadata))
            rc.bm25_score = score
            rc.rrf_score += 1.0 / (cfg.rrf_k + rank)

        candidates = sorted(fused.values(), key=lambda c: c.rrf_score, reverse=True)
        if not rerank:
            return candidates[:top_k]

        candidates = candidates[: cfg.rerank.candidates]
        texts = [f"{c.metadata.get('section', '')}\n{c.text}" for c in candidates]
        for c, s in zip(candidates, self.reranker.score(query, texts)):
            c.rerank_score = s
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[:top_k]
