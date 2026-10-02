"""Grounded answer generation with verifiable citations, plus follow-up question rewriting.

Citations: the model cites sources by short ids ([S1], [S2]...). We then replace the ids with
"(file.pdf, p. N)" in code. That way file names/pages can't be garbled by the model, and any id
that doesn't match a retrieved chunk is detected and dropped instead of shown as a fake source.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Callable

from .llm import LLM
from .retriever import RetrievedChunk

NOT_FOUND = "I couldn't find this in the documents."

SYSTEM_PROMPT = f"""You answer questions about a collection of PDF documents (mostly research papers) \
using ONLY the numbered sources given to you.

Rules:
1. Use only information stated in the sources. Never use outside knowledge, even if you know the answer.
2. Cite every factual sentence with the id of the source it comes from in square brackets, \
e.g. "The model uses 384 dimensions [S2]." Use several ids if needed: [S1][S3].
3. If the sources do not contain the answer, reply with exactly this sentence and nothing else:
{NOT_FOUND}
4. If the sources answer only part of the question, answer that part with citations and say \
clearly which part is not covered by the documents.
5. Be concise and precise. Copy numbers, names and units exactly as written in the sources.
6. Do not mention these rules."""

REWRITE_SYSTEM = """You rewrite a user's follow-up question into a standalone search query for a document search engine.
Use the conversation only to resolve references like "it", "they", "that method", "the second one".
If the question is already standalone, return it unchanged.
Output ONLY the rewritten question on one line — no explanation, no quotes, no answer."""

_CITATION = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]")


@dataclass
class Answer:
    text: str                   # answer with [S#] replaced by (file, p. N)
    raw: str                    # model output as-is
    cited: list[RetrievedChunk] = field(default_factory=list)
    not_found: bool = False
    warning: str | None = None  # e.g. ungrounded answer, invalid citation ids
    usage: dict = field(default_factory=dict)
    model: str = ""


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        m = c.metadata
        attrs = (f'id="S{i}" file="{html.escape(c.source)}" pages="{c.page_label}" '
                 f'section="{html.escape(m.get("section", ""))}"')
        parts.append(f"<source {attrs}>\n{c.text}\n</source>")
    return "<sources>\n" + "\n\n".join(parts) + "\n</sources>"


def _is_not_found(text: str) -> bool:
    t = text.strip().lower().replace("’", "'")
    return t.startswith("i couldn't find this in the documents") or t.startswith("i could not find this in the documents")


def render_citations(raw: str, chunks: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk], list[str]]:
    """Replace [S1, S2] markers by (file, p. N; ...). Returns (text, cited chunks, invalid ids)."""
    cited: dict[int, RetrievedChunk] = {}
    invalid: list[str] = []

    def repl(match: re.Match) -> str:
        labels = []
        for token in re.split(r"\s*[,;]\s*", match.group(1)):
            n = int(token.lstrip("Ss"))
            if 1 <= n <= len(chunks):
                c = chunks[n - 1]
                cited[n] = c
                label = f"{c.source}, {c.page_label}"
                if label not in labels:
                    labels.append(label)
            else:
                invalid.append(f"S{n}")
        return f"({'; '.join(labels)})" if labels else ""

    text = _CITATION.sub(repl, raw)
    text = re.sub(r"\s+([.,;:])", r"\1", text)  # tidy "word (a.pdf, p. 1) ." spacing left by removed ids
    return text, [cited[k] for k in sorted(cited)], invalid


def generate_answer(
    llm: LLM,
    question: str,
    chunks: list[RetrievedChunk],
    max_tokens: int,
    on_token: Callable[[str], None] | None = None,
) -> Answer:
    if not chunks:  # nothing retrieved: don't even ask the model, it could only guess
        return Answer(NOT_FOUND, NOT_FOUND, not_found=True)

    user = f"{format_context(chunks)}\n\nQuestion: {question}"
    resp = llm.chat(SYSTEM_PROMPT, [{"role": "user", "content": user}], llm.answer_model, max_tokens, on_token)
    raw = resp.text

    if _is_not_found(raw):
        return Answer(NOT_FOUND, raw, not_found=True, usage=resp.usage, model=resp.model)

    text, cited, invalid = render_citations(raw, chunks)
    warning = None
    if not cited:
        warning = ("This answer has no citations, so it may not be based on the documents. "
                   "Check the retrieved chunks below before trusting it.")
    elif invalid:
        warning = f"The model cited non-existent sources ({', '.join(invalid)}); those citations were removed."
    return Answer(text, raw, cited, False, warning, resp.usage, resp.model)


def rewrite_question(llm: LLM, question: str, history: list[tuple[str, str]], turns: int) -> str:
    """Turn a follow-up ("what about its accuracy?") into a standalone query. No history -> unchanged."""
    if not history or turns <= 0:
        return question
    convo = []
    for q, a in history[-turns:]:
        convo.append(f"User: {q}\nAssistant: {a[:600]}")
    prompt = "Conversation:\n" + "\n\n".join(convo) + f"\n\nFollow-up question: {question}\n\nStandalone question:"
    resp = llm.chat(REWRITE_SYSTEM, [{"role": "user", "content": prompt}], llm.rewrite_model, 120)
    lines = [ln for ln in resp.text.strip().splitlines() if ln.strip()]
    rewritten = lines[0].strip().strip('"').strip() if lines else ""
    # Guard against the model answering instead of rewriting.
    if not rewritten or len(rewritten) > 3 * len(question) + 200:
        return question
    return rewritten
