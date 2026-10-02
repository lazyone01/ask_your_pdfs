"""End-to-end query pipeline: rewrite -> retrieve -> generate -> log. Used by the CLI and the UI."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .config import Settings
from .embedder import get_embedder
from .generator import Answer, generate_answer, rewrite_question
from .llm import LLMError, get_llm
from .retriever import RetrievedChunk, Retriever
from .vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class QueryResult:
    question: str
    search_query: str
    chunks: list[RetrievedChunk]
    answer: Answer | None
    error: str | None = None
    timings: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = get_embedder(settings.embedding)
        self.store = VectorStore(settings)  # raises EmbeddingMismatchError on model change
        self.retriever = Retriever(settings, self.embedder, self.store)
        self.llm = get_llm(settings.generation)
        self.log_path = settings.log_dir / "queries.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def warmup(self) -> None:
        """Load the embedding model, re-ranker and BM25 index now instead of on the first question."""
        self.embedder.embed_query("warmup")
        if self.settings.retrieval.rerank.enabled:
            self.retriever.reranker.score("warmup", ["warmup"])
        self.retriever.bm25.search("warmup", 1)

    def refresh(self) -> None:
        """Re-open the index after ingestion/deletion so search sees the changes."""
        self.store = VectorStore(self.settings)
        self.retriever.refresh(self.store)

    def search(self, query: str, top_k: int | None = None, rerank: bool | None = None) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, top_k=top_k, rerank=rerank)

    def ask(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        on_token: Callable[[str], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> QueryResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        stage = on_stage or (lambda s: None)
        search_query, chunks, answer, error = question, [], None, None
        try:
            if history:
                stage("Rewriting follow-up question…")
                search_query = rewrite_question(self.llm, question, history, self.settings.generation.history_turns)
                timings["rewrite_s"] = round(time.perf_counter() - t0, 2)

            stage("Searching documents…")
            t1 = time.perf_counter()
            chunks = self.retriever.retrieve(search_query, top_k=top_k, rerank=rerank)
            timings["retrieve_s"] = round(time.perf_counter() - t1, 2)

            stage("Generating answer…")
            t2 = time.perf_counter()
            # The answer prompt gets the standalone question only — never the chat history — so the
            # model can't "answer" from earlier turns instead of from the retrieved text.
            answer = generate_answer(self.llm, search_query, chunks, self.settings.generation.max_tokens, on_token)
            timings["generate_s"] = round(time.perf_counter() - t2, 2)
        except LLMError as e:
            error = str(e)
        except Exception as e:
            log.exception("Query failed")
            error = f"Unexpected error: {e}"
        timings["total_s"] = round(time.perf_counter() - t0, 2)

        result = QueryResult(question, search_query, chunks, answer, error, timings)
        self._log(result)
        return result

    def _log(self, r: QueryResult) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": r.question,
            "search_query": r.search_query,
            "retrieved": [
                {"id": c.id, "source": c.source, "pages": c.page_label,
                 "section": c.metadata.get("section"), "scores": c.scores()}
                for c in r.chunks
            ],
            "answer": r.answer.text if r.answer else None,
            "not_found": r.answer.not_found if r.answer else None,
            "warning": r.answer.warning if r.answer else None,
            "model": r.answer.model if r.answer else None,
            "usage": r.answer.usage if r.answer else None,
            "error": r.error,
            "latency": r.timings,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("Could not write query log: %s", e)
