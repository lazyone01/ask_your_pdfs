"""PDF -> ordered list of text/heading/table blocks with page numbers.

Design notes
- PyMuPDF "dict" output gives us font size + bold flags per line, which we use to detect headings.
- Research papers are often two-column. We order blocks column-by-column between full-width
  blocks (title, figures, wide tables) so sentences are not interleaved across columns.
- Tables are extracted with page.find_tables() and emitted as markdown; text blocks that sit
  inside a table's bbox are skipped so the table content is not duplicated.
- Running headers/footers are removed if the same (digit-masked) line appears in the top/bottom
  margin on many pages; bare page numbers in the margins are always removed.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .cleaner import clean_block_text, is_page_number, margin_key, normalize_unicode
from .config import ParsingConfig

log = logging.getLogger(__name__)

# Default dict flags minus images (we don't need them) and minus ligature preservation (ﬁ -> fi).
_TEXT_FLAGS = (
    pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_PRESERVE_LIGATURES
)
_BOLD_FLAG = 16
_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]{1,5}\.|[A-H]\.?)\s+[A-Z]")
_CAPTION = re.compile(r"^(fig(ure)?|table|tab|algorithm|listing)\.?\s*[\dIVX]", re.IGNORECASE)
_KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "preliminaries", "method",
    "methods", "methodology", "approach", "model", "experiments", "experimental setup",
    "evaluation", "results", "discussion", "analysis", "limitations", "conclusion",
    "conclusions", "future work", "references", "bibliography", "acknowledgments",
    "acknowledgements", "appendix", "supplementary material", "experimental", "experimental details",
    "results and discussion", "materials and methods", "summary", "keywords",
}
# OCR text has no reliable font size/bold, so headings are recognised by shape: "2.3. Title words".
_OCR_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?|[IVX]{1,5}\.)\s+[A-Z][^.;:!?]{2,90}$")


class PDFParseError(Exception):
    """The PDF could not be opened or has no usable content."""


@dataclass
class Block:
    text: str
    page: int  # 1-based
    kind: str  # "heading" | "text" | "table"


@dataclass
class ParsedDocument:
    source: str
    title: str
    num_pages: int
    blocks: list[Block]
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Line:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    bold: bool


@dataclass
class _RawBlock:
    bbox: tuple[float, float, float, float]
    lines: list[_Line]
    kind: str = "text"
    table_md: str = ""


@dataclass
class _Page:
    number: int
    width: float
    height: float
    blocks: list[_RawBlock]
    ocr: bool = False


def parse_pdf(path: Path, cfg: ParsingConfig, source: str | None = None) -> ParsedDocument:
    source = source or path.name
    try:
        doc = pymupdf.open(path)
    except Exception as e:  # corrupt / not a PDF
        raise PDFParseError(f"cannot open PDF: {e}") from e

    with doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFParseError("PDF is password-protected")
        if doc.page_count == 0:
            raise PDFParseError("PDF has no pages")

        warnings: list[str] = []
        pages: list[_Page] = []
        for page in doc:
            try:
                pages.append(_extract_page(page, cfg, warnings))
            except Exception as e:  # one broken page should not kill the document
                warnings.append(f"page {page.number + 1}: extraction failed ({e})")

        body_size = _body_font_size(pages)
        repeated = _repeated_margin_lines(pages, cfg)
        title = _title(doc, pages, source)
        if any(p.ocr for p in pages):
            n = sum(p.ocr for p in pages)
            warnings.append(f"{n} page(s) had no text layer and were read with OCR")

        blocks: list[Block] = []
        for p in pages:
            for rb in _reading_order(p.blocks, p.width):
                if rb.kind == "table":
                    blocks.append(Block(normalize_unicode(rb.table_md), p.number, "table"))
                    continue
                lines = [
                    ln for ln in rb.lines
                    if not (_in_margin(ln.bbox, p.height, cfg.header_footer_margin)
                            and (margin_key(ln.text) in repeated or is_page_number(ln.text)))
                ]
                if not lines:
                    continue
                text = clean_block_text("\n".join(ln.text for ln in lines))
                if not text:
                    continue
                max_size = max(ln.size for ln in lines)
                # Tiny short fragments are almost always figure labels / axis ticks.
                if body_size and max_size < 0.8 * body_size and len(text) < 60:
                    continue
                if p.ocr:
                    if _is_ocr_noise(text):
                        continue
                    kind = "heading" if _is_ocr_heading(text, lines) else "text"
                else:
                    kind = "heading" if _is_heading(text, lines, body_size, cfg.heading_size_ratio) else "text"
                blocks.append(Block(text, p.number, kind))

        return ParsedDocument(source, title, doc.page_count, blocks, warnings)


def _extract_page(page: pymupdf.Page, cfg: ParsingConfig, warnings: list[str]) -> _Page:
    pno = page.number + 1
    textpage = None
    plain = page.get_text("text").strip()
    if len(plain) < cfg.ocr.min_chars_before_ocr:
        if cfg.ocr.enabled:
            try:
                textpage = page.get_textpage_ocr(
                    language=cfg.ocr.language, dpi=cfg.ocr.dpi, full=True,
                    tessdata=_tessdata_dir(cfg.ocr.tessdata),
                )
            except Exception as e:
                warnings.append(f"page {pno}: OCR failed ({e}) - is Tesseract installed?")
        elif not plain:
            warnings.append(f"page {pno}: no text layer (scanned?); enable parsing.ocr to read it")

    tables: list[_RawBlock] = []
    if cfg.extract_tables and textpage is None:
        try:
            for t in page.find_tables().tables:
                md = (t.to_markdown() or "").strip()
                if md:
                    tables.append(_RawBlock(tuple(t.bbox), [], kind="table", table_md=md))
        except Exception as e:
            warnings.append(f"page {pno}: table extraction failed ({e})")

    data = page.get_text("dict", flags=_TEXT_FLAGS, textpage=textpage)
    blocks: list[_RawBlock] = []
    for b in data.get("blocks", []):
        if b.get("type") != 0:
            continue
        bbox = tuple(b["bbox"])
        if any(_center_inside(bbox, t.bbox) for t in tables):
            continue
        lines: list[_Line] = []
        for ln in b.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in ln["spans"]).strip()
            size = max(s["size"] for s in spans)
            bold = all((s["flags"] & _BOLD_FLAG) or "bold" in s.get("font", "").lower() for s in spans)
            lines.append(_Line(text, tuple(ln["bbox"]), size, bold))
        if lines:
            blocks.append(_RawBlock(bbox, lines))

    return _Page(pno, page.rect.width, page.rect.height, blocks + tables, ocr=textpage is not None)


def _tessdata_dir(configured: str) -> str | None:
    """Tesseract language-data folder: config, then $TESSDATA_PREFIX, then common install paths."""
    candidates = [configured, os.environ.get("TESSDATA_PREFIX", "")]
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "")):
        if base:
            candidates.append(os.path.join(base, "Tesseract-OCR", "tessdata"))
    candidates += ["/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tesseract-ocr/4.00/tessdata",
                   "/opt/homebrew/share/tessdata", "/usr/local/share/tessdata"]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


def _center_inside(inner, outer) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _in_margin(bbox, page_height: float, margin: float) -> bool:
    return bbox[3] <= page_height * margin or bbox[1] >= page_height * (1 - margin)


def _body_font_size(pages: list[_Page]) -> float:
    """Most common font size weighted by characters = body text size."""
    sizes: Counter[float] = Counter()
    for p in pages:
        for b in p.blocks:
            for ln in b.lines:
                sizes[round(ln.size * 2) / 2] += len(ln.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _repeated_margin_lines(pages: list[_Page], cfg: ParsingConfig) -> set[str]:
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for p in pages:
        keys = {
            margin_key(ln.text)
            for b in p.blocks
            for ln in b.lines
            if _in_margin(ln.bbox, p.height, cfg.header_footer_margin)
        }
        counts.update(keys)
    threshold = max(2, cfg.header_footer_min_repeat * len(pages))
    return {k for k, c in counts.items() if c >= threshold and k}


def _reading_order(blocks: list[_RawBlock], page_width: float) -> list[_RawBlock]:
    """Order blocks for 1- or 2-column layouts.

    Blocks are classified as left column, right column, or spanning. Spanning blocks act as
    separators; between two separators we emit the whole left column, then the whole right one.
    """
    mid, tol = page_width / 2, page_width * 0.03

    def column(b: _RawBlock) -> str:
        x0, _, x1, _ = b.bbox
        if x1 <= mid + tol:
            return "L"
        if x0 >= mid - tol:
            return "R"
        return "S"

    ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    if sum(column(b) == "R" for b in ordered) < 2:  # effectively single-column
        return ordered

    out: list[_RawBlock] = []
    left: list[_RawBlock] = []
    right: list[_RawBlock] = []
    for b in ordered:
        col = column(b)
        if col == "S":
            out += left + right
            left, right = [], []
            out.append(b)
        else:
            (left if col == "L" else right).append(b)
    return out + left + right


def _is_heading(text: str, lines: list[_Line], body_size: float, ratio: float) -> bool:
    if len(text) > 150 or len(lines) > 3 or not re.search(r"[A-Za-z]{2,}", text):
        return False
    if _CAPTION.match(text) or text.endswith((",", ";")):
        return False
    size = max(ln.size for ln in lines)
    if body_size and size >= body_size * ratio:
        return True
    if all(ln.bold for ln in lines) and len(text) <= 80:
        # Strip "2.1 ", "IV. ", "A. " numbering (a bare letter needs "." or a space, else "Abstract" -> "bstract").
        plain = re.sub(r"^([\d.]+\s*|[IVX]+\.\s*|[A-H](\.\s*|\s+))", "", text).lower().rstrip(".:")
        return bool(_NUMBERED_HEADING.match(text)) or plain in _KNOWN_HEADINGS
    return False


def _is_ocr_heading(text: str, lines: list[_Line]) -> bool:
    if len(lines) != 1 or len(text) > 100 or _CAPTION.match(text):
        return False
    plain = re.sub(r"^([\d.]+\s*|[IVX]+\.\s*)", "", text).lower().rstrip(".:")
    return plain in _KNOWN_HEADINGS or bool(_OCR_NUMBERED_HEADING.match(text))


def _is_ocr_noise(text: str) -> bool:
    """Figure axes / tick labels read by OCR: short and mostly digits, symbols or 1-2 letter fragments."""
    if len(text) > 80:
        return False
    letters = sum(c.isalpha() for c in text)
    words = re.findall(r"[A-Za-z]{4,}", text)
    return letters < 0.5 * len(text.replace(" ", "")) or not words


def _title(doc: pymupdf.Document, pages: list[_Page], source: str) -> str:
    meta = (doc.metadata or {}).get("title", "") or ""
    meta = meta.strip()
    bad = not meta or len(meta) < 5 or meta.lower().endswith((".pdf", ".doc", ".docx", ".dvi", ".tex")) \
        or meta.lower().startswith(("untitled", "microsoft word")) or "..." in meta or "…" in meta
    if not bad:
        return normalize_unicode(meta)
    # Fall back to the largest text on the first page.
    # (OCR font sizes are rough and big title fonts OCR badly, so don't trust them.)
    if pages and pages[0].blocks and not pages[0].ocr:
        text_lines = [ln for b in pages[0].blocks for ln in b.lines if len(ln.text) > 3]
        if text_lines:
            biggest = max(ln.size for ln in text_lines)
            parts = [ln.text for ln in text_lines if abs(ln.size - biggest) < 0.5]
            title = clean_block_text(" ".join(parts))
            if 5 <= len(title) <= 250:
                return title
    return Path(source).stem
