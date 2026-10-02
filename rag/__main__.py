"""Command line interface:  python -m rag <command> ...

Stage 1 (ingestion):
  ingest [--rebuild] [--prune]   index new/changed PDFs from data/pdfs
  list                           show indexed documents
  delete <file.pdf>              remove a document from the index
  preview <file.pdf>             parse + chunk without indexing; writes storage/previews/<name>.md
  inspect <file.pdf> [-n N]      show stored chunks of an indexed document
  stats                          chunk size distribution across the index

Stage 2-3 (query):
  search "<query>" [-k N]        hybrid retrieval + re-rank, no LLM
  ask "<question>"               full answer with citations (needs Ollama running)

UI:  streamlit run app.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from .cleaner import estimate_tokens
from .config import load_settings
from .logging_setup import setup_logging


def cmd_ingest(settings, args) -> int:
    from .ingest import run_ingest

    r = run_ingest(settings, rebuild=args.rebuild, prune=args.prune)
    print(f"\nAdded: {len(r.added)}  Updated: {len(r.updated)}  Skipped (unchanged): {len(r.skipped)}  "
          f"Failed: {len(r.failed)}  Removed: {len(r.removed)}")
    for name, reason in r.failed:
        print(f"  FAILED  {name}: {reason}")
    for name, warns in r.warnings.items():
        print(f"  WARN    {name}: {len(warns)} warning(s), e.g. {warns[0]}")
    if r.missing:
        print(f"  {len(r.missing)} indexed file(s) no longer in the PDF folder "
              f"(run 'ingest --prune' to remove): {', '.join(r.missing)}")
    return 1 if r.failed else 0


def cmd_list(settings, args) -> int:
    from .ingest import open_manifest

    m = open_manifest(settings)
    docs = m.documents
    if not docs:
        print("Index is empty. Put PDFs in", settings.pdf_dir, "and run: python -m rag ingest")
        return 0
    print(f"Embedding model: {m.data.get('embedding_model')}\n")
    print(f"{'file':45} {'pages':>5} {'chunks':>6}  title")
    for name, d in sorted(docs.items()):
        print(f"{name[:45]:45} {d['pages']:>5} {d['chunks']:>6}  {d['title'][:60]}")
    return 0


def cmd_delete(settings, args) -> int:
    from .ingest import delete_document

    known = delete_document(settings, args.source)
    print(f"Deleted {args.source} from the index." if known else f"{args.source} was not in the manifest "
          "(any stray chunks were removed). Remember to also remove the file from the PDF folder, "
          "or it will be re-indexed on the next ingest.")
    return 0


def _print_chunk(c_text: str, meta: dict, full: bool) -> None:
    pages = f"p.{meta['page_start']}" + (f"-{meta['page_end']}" if meta["page_end"] != meta["page_start"] else "")
    print(f"--- #{meta['chunk_index']}  {pages}  [{meta['section'][:60]}]  "
          f"~{estimate_tokens(c_text)} tok  ({meta.get('kind', 'text')})")
    print(c_text if full or len(c_text) <= 400 else c_text[:400] + " ...")
    print()


def cmd_preview(settings, args) -> int:
    from .ingest import build_chunks, list_pdfs
    from .manifest import file_sha256

    path = Path(args.pdf)
    if not path.exists():
        path = list_pdfs(settings.pdf_dir).get(args.pdf, path)
    if not path.exists():
        print(f"Not found: {args.pdf}")
        return 1
    chunks, parsed = build_chunks(path, path.name, file_sha256(path), settings)

    headings = [b for b in parsed.blocks if b.kind == "heading"]
    tables = [b for b in parsed.blocks if b.kind == "table"]
    print(f"Title:    {parsed.title}\nPages:    {parsed.num_pages}\nBlocks:   {len(parsed.blocks)} "
          f"({len(headings)} headings, {len(tables)} tables)\nChunks:   {len(chunks)}")
    print("\nDetected headings:")
    for h in headings:
        print(f"  p.{h.page:<3} {h.text[:90]}")
    for w in parsed.warnings:
        print(f"WARN: {w}")

    out = settings.storage_dir / "previews" / f"{path.stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# {parsed.title}\n\n")
        for c in chunks:
            f.write(f"## Chunk {c.chunk_index} | pages {c.page_start}-{c.page_end} | {c.section} | "
                    f"~{estimate_tokens(c.text)} tokens\n\n{c.text}\n\n")
    print(f"\nFull chunk dump written to {out}")
    return 0


def cmd_inspect(settings, args) -> int:
    from .vector_store import VectorStore

    items = VectorStore(settings).get_all(where={"source": args.source})
    if not items:
        print(f"No chunks for '{args.source}'. Use 'python -m rag list' to see indexed names.")
        return 1
    items.sort(key=lambda c: c.metadata["chunk_index"])
    print(f"{len(items)} chunks for {args.source}\n")
    for c in items[args.start : args.start + args.n]:
        _print_chunk(c.text, c.metadata, args.full)
    return 0


def cmd_stats(settings, args) -> int:
    from .vector_store import VectorStore

    items = VectorStore(settings).get_all()
    if not items:
        print("Index is empty.")
        return 0
    toks = sorted(estimate_tokens(c.text) for c in items)
    docs = {c.metadata["source"] for c in items}
    # "inclusive" keeps percentiles within [min, max] (the default extrapolates on small samples).
    q = statistics.quantiles(toks, n=10, method="inclusive") if len(toks) > 1 else toks * 9
    print(f"Documents: {len(docs)}   Chunks: {len(items)}   Chunks/doc: {len(items) / len(docs):.0f}")
    print(f"Tokens per chunk (est.): min {toks[0]}  p10 {q[0]:.0f}  median {statistics.median(toks):.0f}  "
          f"p90 {q[-1]:.0f}  max {toks[-1]}")
    tables = sum(1 for c in items if c.metadata.get("kind") == "table")
    print(f"Table-only chunks: {tables}")
    return 0


def cmd_search(settings, args) -> int:
    from .pipeline import RAGPipeline

    pipe = RAGPipeline(settings)
    hits = pipe.search(args.query, top_k=args.k, rerank=not args.no_rerank)
    if not hits:
        print("No results (is the index empty?)")
    for i, c in enumerate(hits, start=1):
        s = "  ".join(f"{k}={v}" for k, v in c.scores().items() if v is not None)
        print(f"[{i}] {c.source} {c.page_label}  [{c.metadata.get('section', '')[:50]}]\n    {s}")
        print("    " + (c.text if args.full else c.text[:300].replace("\n", " ") + " ..."))
        print()
    return 0


def cmd_ask(settings, args) -> int:
    from .pipeline import RAGPipeline

    pipe = RAGPipeline(settings)
    ok, msg = pipe.llm.status()
    if not ok:
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    print("Answer: ", end="", flush=True)
    r = pipe.ask(args.question, on_token=lambda t: print(t, end="", flush=True))
    print("\n")
    if r.error:
        print(f"ERROR: {r.error}", file=sys.stderr)
        return 1
    print(f"Final (citations resolved):\n{r.answer.text}\n")
    if r.answer.warning:
        print(f"WARNING: {r.answer.warning}\n")
    print("Retrieved:")
    for i, c in enumerate(r.chunks, start=1):
        mark = "*" if c in r.answer.cited else " "
        print(f" {mark}S{i} {c.source} {c.page_label} [{c.metadata.get('section', '')[:50]}]")
    print(f"\nTimings: {r.timings}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not be UTF-8; never crash on printing a PDF's odd characters.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(prog="python -m rag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to config.yaml (default: $RAG_CONFIG or ./config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index new/changed PDFs")
    p.add_argument("--rebuild", action="store_true", help="drop the index and re-embed everything")
    p.add_argument("--prune", action="store_true", help="remove indexed docs whose file was deleted")
    p.set_defaults(func=cmd_ingest)

    sub.add_parser("list", help="show indexed documents").set_defaults(func=cmd_list)

    p = sub.add_parser("delete", help="remove a document from the index")
    p.add_argument("source", help="file name as shown by 'list'")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("preview", help="parse + chunk a PDF without indexing")
    p.add_argument("pdf", help="path to a PDF, or a file name inside the PDF folder")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("inspect", help="show stored chunks of a document")
    p.add_argument("source")
    p.add_argument("-n", type=int, default=10, help="number of chunks to show")
    p.add_argument("--start", type=int, default=0, help="first chunk index")
    p.add_argument("--full", action="store_true", help="don't truncate chunk text")
    p.set_defaults(func=cmd_inspect)

    sub.add_parser("stats", help="chunk size distribution").set_defaults(func=cmd_stats)

    p = sub.add_parser("search", help="hybrid retrieval only (no LLM)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=None, help="top-k (default from config)")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="retrieve + answer with citations")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    setup_logging(settings)
    try:
        return args.func(settings, args)
    except Exception as e:
        from .vector_store import EmbeddingMismatchError

        if isinstance(e, EmbeddingMismatchError):
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
