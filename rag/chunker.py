"""Structure-aware chunking.

Strategy
1. Group blocks into sections using detected headings. Chunks never cross a section boundary,
   so each chunk is about one topic and carries a meaningful "section" label.
2. Within a section, pack paragraphs (and tables) greedily up to chunk_size_tokens.
   A paragraph that is too long on its own is split at sentence boundaries; a table is split by
   rows with its header row repeated.
3. Consecutive chunks overlap by the last ~chunk_overlap_tokens of whole sentences, so a fact
   straddling the boundary is still retrievable from either side.
4. Sections matching parsing.drop_sections (References, Acknowledgements...) are skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .cleaner import estimate_tokens
from .config import ChunkingConfig
from .parser import ParsedDocument

# Split after . ! ? when followed by whitespace and an uppercase letter / bracket / digit.
# Lowercase after the period (e.g. "e.g. the", "et al. used") does not split.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    section: str
    page_start: int
    page_end: int
    chunk_index: int
    doc_hash: str
    kind: str = "text"  # "text" | "table" (table if the chunk is only table content)

    def embed_text(self, add_context_header: bool = True) -> str:
        """Text used for embeddings and BM25: prefixing title/section helps short chunks match."""
        if not add_context_header:
            return self.text
        return f"{self.title} > {self.section}\n\n{self.text}"

    def metadata(self) -> dict:
        # Chroma metadata values must be str/int/float/bool (no None).
        return {
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
            "doc_hash": self.doc_hash,
            "kind": self.kind,
        }


@dataclass(eq=False)  # identity comparison: overlap reuses the same piece objects
class _Piece:
    text: str
    page: int
    kind: str  # "sentence" | "table"
    tokens: int
    block_id: int  # pieces from the same paragraph are re-joined with a space


def chunk_document(
    doc: ParsedDocument, doc_hash: str, cfg: ChunkingConfig, drop_sections: list[str]
) -> list[Chunk]:
    drop_patterns = [re.compile(p, re.IGNORECASE) for p in drop_sections]
    sections: list[tuple[str, list[_Piece]]] = []
    current_heading, current, dropping = "Front matter", [], False

    for block_id, block in enumerate(doc.blocks):
        if block.kind == "heading":
            if current and not dropping:
                sections.append((current_heading, current))
            current_heading, current = block.text, []
            dropping = any(p.search(block.text.strip()) for p in drop_patterns)
            continue
        if dropping:
            continue
        current.extend(
            _split_block(block.text, block.page, block.kind, block_id, cfg.chunk_size_tokens)
        )
    if current and not dropping:
        sections.append((current_heading, current))

    chunks: list[Chunk] = []
    for heading, pieces in sections:
        for group in _pack(pieces, cfg):
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc_hash[:16]}-{idx:05d}",
                    text="\n\n".join(_merge_adjacent(group)),
                    source=doc.source,
                    title=doc.title,
                    section=heading[:200],
                    page_start=min(p.page for p in group),
                    page_end=max(p.page for p in group),
                    chunk_index=idx,
                    doc_hash=doc_hash,
                    kind="table" if all(p.kind == "table" for p in group) else "text",
                )
            )
    return chunks


def _split_block(text: str, page: int, kind: str, block_id: int, max_tokens: int) -> list[_Piece]:
    """Split a block into sentence pieces (text) or row groups (tables), each <= max_tokens."""

    def piece(t: str, k: str) -> _Piece:
        return _Piece(t, page, k, estimate_tokens(t), block_id)

    if kind == "table":
        if estimate_tokens(text) <= max_tokens:
            return [piece(text, "table")]
        rows = text.splitlines()
        header, body = rows[:2], rows[2:]  # markdown header + separator row
        pieces, buf = [], list(header)
        for row in body:
            if estimate_tokens("\n".join(buf + [row])) > max_tokens and len(buf) > len(header):
                pieces.append(piece("\n".join(buf), "table"))
                buf = list(header)
            buf.append(row)
        if len(buf) > len(header):
            pieces.append(piece("\n".join(buf), "table"))
        return pieces

    pieces = []
    for sentence in _SENTENCE_SPLIT.split(text):
        if estimate_tokens(sentence) <= max_tokens:
            pieces.append(piece(sentence, "sentence"))
            continue
        buf: list[str] = []  # pathological run-on text: hard split on words
        for w in sentence.split():
            buf.append(w)
            if estimate_tokens(" ".join(buf)) >= max_tokens:
                pieces.append(piece(" ".join(buf), "sentence"))
                buf = []
        if buf:
            pieces.append(piece(" ".join(buf), "sentence"))
    return pieces


def _pack(pieces: list[_Piece], cfg: ChunkingConfig) -> list[list[_Piece]]:
    """Greedy packing with sentence-level overlap between consecutive chunks."""
    groups: list[list[_Piece]] = []
    cur: list[_Piece] = []
    cur_tokens = 0
    fresh = 0  # pieces in `cur` that are new (not overlap carried from the previous chunk)

    for piece in pieces:
        if cur and fresh and cur_tokens + piece.tokens > cfg.chunk_size_tokens:
            groups.append(cur)
            cur = _overlap_tail(cur, cfg.chunk_overlap_tokens)
            # Drop overlap if it would not leave room for the new piece.
            if sum(p.tokens for p in cur) + piece.tokens > cfg.chunk_size_tokens:
                cur = []
            cur_tokens = sum(p.tokens for p in cur)
            fresh = 0
        cur.append(piece)
        cur_tokens += piece.tokens
        fresh += 1
    if cur and fresh:
        groups.append(cur)

    # Merge a tiny trailing chunk into its predecessor.
    if len(groups) >= 2:
        prev_ids = {id(p) for p in groups[-2]}
        last_new = [p for p in groups[-1] if id(p) not in prev_ids]
        if sum(p.tokens for p in last_new) < cfg.min_chunk_tokens:
            groups[-2] = groups[-2] + last_new
            groups.pop()
    return groups


def _overlap_tail(group: list[_Piece], overlap_tokens: int) -> list[_Piece]:
    """Trailing sentences totalling <= overlap_tokens (tables are never overlapped)."""
    tail: list[_Piece] = []
    total = 0
    for p in reversed(group):
        if p.kind == "table" or total + p.tokens > overlap_tokens:
            break
        tail.insert(0, p)
        total += p.tokens
    return tail


def _merge_adjacent(group: list[_Piece]) -> list[str]:
    """Rejoin sentences of the same source paragraph; paragraphs/tables are separated by blank lines."""
    parts: list[str] = []
    prev: _Piece | None = None
    for p in group:
        if prev is not None and p.kind == "sentence" and prev.kind == "sentence" \
                and p.block_id == prev.block_id:
            parts[-1] = f"{parts[-1]} {p.text}"
        else:
            parts.append(p.text)
        prev = p
    return parts
