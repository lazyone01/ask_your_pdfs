"""Streamlit UI: upload PDFs, ask questions, see cited answers and the retrieved chunks.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st

from rag.config import load_settings
from rag.ingest import delete_document, open_manifest, run_ingest
from rag.logging_setup import setup_logging
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="Paper Q&A", page_icon="📄", layout="wide")


def export_secrets() -> None:
    """On Streamlit Cloud, keys (GROQ_API_KEY, RAG_CONFIG...) live in st.secrets; expose them as env vars."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key, value)
    except Exception:  # no secrets file (normal when running locally)
        pass


export_secrets()


@st.cache_resource(show_spinner="Loading models…")
def get_pipeline() -> RAGPipeline:
    settings = load_settings()
    setup_logging(settings)
    pipeline = RAGPipeline(settings)
    pipeline.warmup()
    return pipeline


def md_escape(text: str) -> str:
    # Streamlit markdown treats $...$ as LaTeX; papers are full of "$" that should stay literal.
    return text.replace("$", "\\$")


def safe_filename(name: str) -> str:
    name = Path(name).name  # strip any directory parts sent by the browser
    stem = re.sub(r"[^\w\-. ()]+", "_", Path(name).stem).strip() or "document"
    return f"{stem}.pdf"


try:
    pipeline = get_pipeline()
except Exception as e:  # e.g. embedding model changed -> index must be rebuilt
    st.error(f"Could not start: {e}")
    st.stop()
settings = pipeline.settings

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("uploader_key", 0)
ss.setdefault("flash", None)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 Library")

    ok, msg = pipeline.llm.status()
    (st.success if ok else st.error)(msg, icon="✅" if ok else "⚠️")

    uploaded = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, key=f"uploader_{ss.uploader_key}"
    )
    if uploaded and st.button(f"Add {len(uploaded)} file(s) to library", type="primary", use_container_width=True):
        settings.pdf_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (settings.pdf_dir / safe_filename(f.name)).write_bytes(f.getbuffer())
        with st.status("Indexing…", expanded=True) as status:
            report = run_ingest(
                settings,
                embedder=pipeline.embedder,
                on_progress=lambda src, i, n: status.write(f"({i}/{n}) {src}"),
            )
            pipeline.refresh()
            status.update(label="Indexing finished", state="complete")
        parts = []
        if report.added:
            parts.append(f"added {len(report.added)}")
        if report.updated:
            parts.append(f"updated {len(report.updated)}")
        if report.skipped:
            parts.append(f"{len(report.skipped)} unchanged")
        ss.flash = ("Indexed: " + ", ".join(parts)) if parts else "Nothing to index."
        if report.failed:
            ss.flash += "\n\nFailed: " + "; ".join(f"{n} ({r})" for n, r in report.failed)
        ss.uploader_key += 1  # clears the uploader widget
        st.rerun()

    if ss.flash:
        st.info(ss.flash)
        ss.flash = None

    docs = open_manifest(settings).documents
    st.caption(f"{len(docs)} document(s) indexed")
    for name, d in sorted(docs.items()):
        col1, col2 = st.columns([5, 1])
        col1.markdown(f"**{name}**  \n<small>{d['pages']} pages · {d['chunks']} chunks</small>",
                      unsafe_allow_html=True)
        if col2.button("🗑", key=f"del_{name}", help="Remove from index and delete the file from data/pdfs"):
            delete_document(settings, name)
            (settings.pdf_dir / name).unlink(missing_ok=True)
            pipeline.refresh()
            ss.flash = f"Removed {name}"
            st.rerun()

    st.divider()
    st.subheader("⚙️ Retrieval")
    top_k = st.slider("Chunks sent to the LLM (top-k)", 1, 12, settings.retrieval.top_k)
    rerank = st.toggle("Re-rank with cross-encoder", value=settings.retrieval.rerank.enabled)
    if st.button("Clear conversation", use_container_width=True):
        ss.messages = []
        st.rerun()

# ----------------------------------------------------------------------------- chat


def render_sources(chunks: list[dict], search_query: str, question: str, timings: dict) -> None:
    label = f"🔎 {len(chunks)} retrieved chunk(s)"
    if timings:
        label += f" · {timings.get('total_s', 0)}s"
    with st.expander(label):
        if search_query != question:
            st.caption(f"Follow-up rewritten for search as: *{search_query}*")
        if timings:
            st.caption(" · ".join(f"{k.removesuffix('_s')} {v}s" for k, v in timings.items()))
        for c in chunks:
            s = c["scores"]
            score_txt = " · ".join(f"{k} {v:.3f}" for k, v in s.items() if v is not None)
            badge = "✅ cited" if c["cited"] else ""
            st.markdown(f"**{c['label']}** · {c['source']} · {c['pages']} · *{c['section']}* {badge}")
            st.caption(score_txt)
            st.markdown(md_escape(c["text"]))
            st.divider()


def render_assistant(m: dict) -> None:
    if m.get("error"):
        st.error(m["error"])
    else:
        st.markdown(md_escape(m["content"]))
        if m.get("warning"):
            st.warning(m["warning"])
    if m.get("chunks") is not None:
        render_sources(m["chunks"], m.get("search_query", ""), m.get("question", ""), m.get("timings", {}))


st.title("📄 Ask your PDFs")
if not docs:
    st.info("Upload one or more PDFs in the sidebar to get started.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(md_escape(m["content"]))
        else:
            render_assistant(m)

if question := st.chat_input("Ask a question about your documents", disabled=not docs):
    # Previous successful Q/A pairs, used only to rewrite follow-up questions.
    history = [
        (ss.messages[i]["content"], ss.messages[i + 1]["content"])
        for i in range(0, len(ss.messages) - 1, 2)
        if ss.messages[i]["role"] == "user" and not ss.messages[i + 1].get("error")
    ]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md_escape(question))

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        placeholder = st.empty()
        streamed: list[str] = []

        def on_token(t: str) -> None:
            streamed.append(t)
            placeholder.markdown(md_escape("".join(streamed)) + " ▌")

        result = pipeline.ask(
            question, history=history, top_k=top_k, rerank=rerank,
            on_token=on_token, on_stage=lambda s: status.update(label=s),
        )
        status.update(label=f"Done in {result.timings.get('total_s', 0)}s",
                      state="error" if result.error else "complete")
        placeholder.empty()

        cited_ids = {c.id for c in result.answer.cited} if result.answer else set()
        msg = {
            "role": "assistant",
            "content": result.answer.text if result.answer else "",
            "warning": result.answer.warning if result.answer else None,
            "error": result.error,
            "question": question,
            "search_query": result.search_query,
            "timings": result.timings,
            "chunks": [
                {"label": f"S{i}", "source": c.source, "pages": c.page_label,
                 "section": c.metadata.get("section", ""), "text": c.text,
                 "scores": c.scores(), "cited": c.id in cited_ids}
                for i, c in enumerate(result.chunks, start=1)
            ],
        }
        ss.messages.append(msg)
        render_assistant(msg)
