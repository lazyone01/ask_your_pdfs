"""Incremental ingestion: data/pdfs -> parse -> chunk -> embed -> Chroma (+ manifest)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .chunker import Chunk, chunk_document
from .config import Settings
from .embedder import Embedder, get_embedder
from .manifest import Manifest, file_sha256
from .parser import ParsedDocument, PDFParseError, parse_pdf
from .vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # indexed, but the file is gone
    removed: list[str] = field(default_factory=list)
    warnings: dict[str, list[str]] = field(default_factory=dict)


def list_pdfs(pdf_dir: Path) -> dict[str, Path]:
    """source name (path relative to pdf_dir, forward slashes) -> absolute path."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return {
        p.relative_to(pdf_dir).as_posix(): p
        for p in sorted(pdf_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() == ".pdf"
    }


def build_chunks(path: Path, source: str, sha: str, settings: Settings) -> tuple[list[Chunk], ParsedDocument]:
    parsed = parse_pdf(path, settings.parsing, source)
    chunks = chunk_document(parsed, sha, settings.chunking, settings.parsing.drop_sections)
    return chunks, parsed


def open_manifest(settings: Settings) -> Manifest:
    return Manifest(settings.storage_dir / "manifest.json")


def run_ingest(
    settings: Settings,
    rebuild: bool = False,
    prune: bool = False,
    embedder: Embedder | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> IngestReport:
    """Index new/changed PDFs. `embedder` lets the UI reuse its already-loaded model;
    `on_progress(source, i, total)` is called before each file."""
    report = IngestReport()
    store = VectorStore(settings, reset=rebuild)  # raises EmbeddingMismatchError if models differ
    manifest = open_manifest(settings)
    identity = settings.embedding.identity
    # Manifest must describe exactly what is in the store; if not (rebuild, deleted chroma
    # folder, model change on an empty index) start it over so nothing is wrongly skipped.
    if rebuild or store.count() == 0 or manifest.data.get("embedding_model") != identity:
        manifest.reset(identity)
        manifest.save()

    embedder = embedder or get_embedder(settings.embedding)
    pdfs = list_pdfs(settings.pdf_dir)
    if not pdfs:
        log.warning("No PDFs found in %s", settings.pdf_dir)

    for i, (source, path) in enumerate(pdfs.items(), start=1):
        if on_progress:
            on_progress(source, i, len(pdfs))
        try:
            sha = file_sha256(path)
            entry = manifest.get(source)
            if entry and entry["sha256"] == sha:
                report.skipped.append(source)
                continue

            t0 = time.perf_counter()
            chunks, parsed = build_chunks(path, source, sha, settings)
            if parsed.warnings:
                report.warnings[source] = parsed.warnings
            if not chunks:
                raise PDFParseError("no extractable text (scanned PDF? enable parsing.ocr)")

            texts = [c.embed_text(settings.chunking.add_context_header) for c in chunks]
            vectors = embedder.embed_documents(texts)
            # Delete old version only after the new one embedded successfully.
            store.delete_source(source)
            store.add(chunks, vectors)
            manifest.put(source, sha, parsed.num_pages, len(chunks), parsed.title, parsed.warnings)
            manifest.save()  # after every file, so an interrupted run keeps its progress
            (report.updated if entry else report.added).append(source)
            log.info(
                "Indexed %s: %d pages, %d chunks in %.1fs",
                source, parsed.num_pages, len(chunks), time.perf_counter() - t0,
            )
        except Exception as e:
            log.error("Failed %s: %s", source, e, exc_info=not isinstance(e, PDFParseError))
            report.failed.append((source, str(e)))

    for source in list(manifest.documents):
        if source not in pdfs:
            if prune:
                store.delete_source(source)
                manifest.remove(source)
                report.removed.append(source)
            else:
                report.missing.append(source)
    manifest.save()
    return report


def delete_document(settings: Settings, source: str) -> bool:
    store = VectorStore(settings)
    manifest = open_manifest(settings)
    known = manifest.get(source) is not None
    store.delete_source(source)
    manifest.remove(source)
    manifest.save()
    return known
