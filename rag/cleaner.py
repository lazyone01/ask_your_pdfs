"""Text normalisation helpers used by the parser."""

from __future__ import annotations

import re
import unicodedata

# Invisible / layout characters that PDFs are full of.
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍﻿"), None)

# "embed-\nding" -> "embedding"; only when the next line starts lowercase,
# so "state-of-the-\nArt" style proper hyphens mostly survive.
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\n([a-z])")
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)


def normalize_unicode(text: str) -> str:
    # NFKC expands ligatures (ﬁ -> fi) and full-width forms.
    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE)


def join_lines(text: str) -> str:
    """Turn a block's visual lines into one paragraph, fixing end-of-line hyphenation."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def clean_block_text(text: str) -> str:
    return join_lines(normalize_unicode(text))


def margin_key(text: str) -> str:
    """Key for comparing running headers/footers across pages (page numbers vary, so mask digits)."""
    return re.sub(r"\d+", "#", normalize_unicode(text).lower()).strip()


def is_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER.match(text.strip()))


def estimate_tokens(text: str) -> int:
    # ~4 characters per token for English; avoids loading a tokenizer during chunking.
    return max(1, len(text) // 4)
