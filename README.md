# PDF RAG — question answering over research papers

Try it here: https://ask-your-pdfs01.streamlit.app/

Fully local, free and private RAG: PyMuPDF parsing → section-aware chunks → local `bge-small` embeddings →
ChromaDB + BM25 hybrid search → cross-encoder re-ranking → a local LLM via **Ollama** answers with (file, page) citations.
No PDF text leaves your machine. Claude (Anthropic API) remains available as a config switch.

## Setup (Windows, PowerShell)

Requires Python 3.11–3.13 (tested with 3.12).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # no keys needed for the default (all-local) setup
```

Install Ollama and pull the answer model (about 2.5 GB):

```powershell
winget install Ollama.Ollama   # or download from https://ollama.com
ollama pull qwen3:4b-instruct
```

Ollama runs in the background (tray icon) and listens on `http://localhost:11434`.
The first ingest downloads the embedding model (~130 MB) and the first query downloads the re-ranker (~90 MB)
from Hugging Face. After that, everything works offline.

### Choosing the local model (CPU-only laptop, 16 GB RAM)

| Model | RAM used | Speed on a 4-core CPU | Notes |
|---|---|---|---|
| `qwen3:4b-instruct` (default) | ~3.5 GB | ~15–40 s per answer | Good at following "answer only from context" |
| `qwen3:4b` | ~3.5 GB | ~80–120 s per answer | This is the *Thinking* build: it always reasons first (hidden by the app), so it's much slower |
| `qwen3:8b` | ~6 GB | ~45–90 s per answer | Noticeably better reasoning over several chunks |
| `llama3.2:3b` | ~2.5 GB | ~15–30 s per answer | Fastest; more likely to cite loosely |

To switch, run `ollama pull <model>` and set `generation.ollama.answer_model` in `config.yaml`. Small local
models are more prone to answering from their own knowledge than Claude is. To guard against that, the app
checks every citation against the retrieved chunks and flags any answer that has no valid citation.

## Stage 1 — Ingest PDFs and inspect chunks

1. Copy PDFs into `data/pdfs/` (sub-folders are fine).
2. Before indexing, preview how one PDF gets parsed:
   ```powershell
   python -m rag preview my_paper.pdf
   ```
   This prints the detected title and headings, plus any warnings. It writes every chunk to
   `storage/previews/my_paper.md` so you can read them. Check that the headings look right,
   the two-column text reads in order, running headers and page numbers are gone, and the
   References section is missing.
3. Index everything:
   ```powershell
   python -m rag ingest            # only new/changed files are processed (SHA-256 per file)
   python -m rag list              # what is indexed
   python -m rag stats             # chunk size distribution
   python -m rag inspect my_paper.pdf -n 5 --full
   ```
4. Remove things:
   ```powershell
   python -m rag delete my_paper.pdf   # remove from index (also delete the file, or it comes back on the next ingest)
   python -m rag ingest --prune        # remove index entries whose files were deleted from data/pdfs
   python -m rag ingest --rebuild      # wipe and re-embed everything
   ```

**Incremental rules:** an unchanged file (same hash) is skipped. A changed file has its old chunks
replaced, but only after the new version has been embedded successfully. A corrupt or
password-protected PDF is reported, and the other files continue.

## Web app — upload PDFs and ask questions

```powershell
streamlit run app.py
```

Open **http://localhost:8501**. The app only listens on this computer (`.streamlit/config.toml`).

### Keep it always running (installed on this PC)

A Windows Scheduled Task named **"PDF QA app"** starts the app hidden every time you log in. It runs
`run_app.bat`, which restarts Streamlit within about 5 s if it ever crashes. Ollama starts with Windows on its own.

| What | How |
|---|---|
| Stop the app | double-click `stop_app.bat` |
| Start it again | double-click `run_app.bat` (does nothing if it's already running), or `Start-ScheduledTask "PDF QA app"` |
| Logs | `logs\streamlit.log` |
| Turn off auto-start | `Unregister-ScheduledTask "PDF QA app"` (PowerShell), or delete it in Task Scheduler |

After you change `config.yaml` or the code, run `stop_app.bat` and then `run_app.bat` to restart.

- **Sidebar:** upload PDFs, then click *Add to library*. This runs the same incremental ingest as the CLI, and
  unchanged files are skipped. 🗑 removes a document from the index and deletes its copy in `data/pdfs/`.
  The status box tells you if Ollama isn't running or the model hasn't been downloaded.
- **Chat:** the answer streams in as it's generated. Citations appear as `(file.pdf, p. 3)`. Expand
  *retrieved chunks* to see every chunk the model was given, with its vector, BM25, fusion and re-rank scores,
  and a ✅ on the ones it cited.
- **Follow-ups** ("what about its accuracy?") are rewritten into a standalone question before searching. The
  rewritten version is shown under the retrieved chunks. The chat history is used *only* for this rewrite,
  never as answer material.
- If the answer isn't in the documents, you get *"I couldn't find this in the documents."* An answer
  without any valid citation shows a warning.

## Deploy online for free (Streamlit Community Cloud + Gemini)

Free hosts don't have the RAM to run Ollama, so the online copy gets its answers from **Google Gemini's free API**
(`gemini-flash-latest`, through the `openai_compat` provider). Parsing, OCR, embeddings, search and re-ranking
still run inside the app. Your laptop version keeps using Ollama. (Groq was tried first, but it blocks
Streamlit Cloud's servers with HTTP 403.)

1. Get a free key at https://aistudio.google.com/apikey (no card).
2. Push this repo to a **private** GitHub repo (`.env`, PDFs and the index are git-ignored).
3. At https://share.streamlit.io: *Create app* → pick the repo, branch `main`, file `app.py`.
   Under *Advanced settings*, choose **Python 3.12** (the CPU torch wheel is for 3.12) and paste these secrets:
   ```toml
   GEMINI_API_KEY = "AIza..."
   RAG_CONFIG = "config.cloud.yaml"
   ```
4. To share it, use *Share* in the app's top-right to invite people by email (private repo = private app).

`packages.txt` installs Tesseract for OCR. Limits: question text and retrieved chunks are sent to Google
(on the free tier Google may use them to improve its products), uploaded PDFs are erased when the app restarts
or sleeps (after ~12 h with no visitors), and the free tier has per-minute and per-day request limits.
Every `git push` redeploys automatically.

**Another provider:** `openai_compat` works with any OpenAI-compatible API. Change `base_url`, `api_key_env`
and the model names in `config.cloud.yaml` (e.g. Cerebras: `https://api.cerebras.ai/v1`, `CEREBRAS_API_KEY`).

## Query from the command line

```powershell
python -m rag search "what learning rate was used?"   # retrieval only: check the right chunks come back
python -m rag ask "what learning rate was used?"      # full answer with citations
```

Every query (CLI or UI) is appended to `logs/queries.jsonl`: the question, the rewritten search query,
retrieved chunk ids/pages/scores, the answer, token counts and a timing breakdown. Application logs go to `logs/rag.log`.

## Settings (`config.yaml`)

| Setting | Default | Notes |
|---|---|---|
| `chunking.chunk_size_tokens` | 350 | bge-small reads up to 512 tokens. At 350, one chunk usually holds one claim plus its context. |
| `chunking.chunk_overlap_tokens` | 50 | Whole sentences carried into the next chunk. |
| `parsing.drop_sections` | References, Acknowledgements | Regexes matched against headings. |
| `parsing.ocr.enabled` | true | Reads pages with no text layer (scans, "Microsoft Print to PDF" files) with [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (`winget install UB-Mannheim.TesseractOCR`). Only runs on pages with almost no text; about 5–6 s per page. |
| `embedding.provider` | local | `local` (bge-small, free) or `voyage` (needs `VOYAGE_API_KEY`). |
| `retrieval.top_k` | 6 | Chunks sent to the LLM (also a slider in the UI). More chunks = more context but slower on CPU. |
| `retrieval.rerank.enabled` | true | Cross-encoder re-ranking of the top 30 fused hits (~1 s on CPU). |
| `generation.provider` | ollama | `ollama` (local) or `anthropic` (needs `ANTHROPIC_API_KEY`). |
| `generation.ollama.num_ctx` | 8192 | Context window. Ollama's default is small and it silently cuts off the prompt, so keep this ≥ 8192 with `top_k: 6`. |

Changing chunking or parsing settings only affects files that are re-ingested. Run `ingest --rebuild` to apply them to everything.

**Embedding model lock:** the index records the embedding model that built it. If
`config.yaml` names a different model, every command stops with an error and asks you to run
`python -m rag ingest --rebuild`. Vectors from two different models can't be compared, so mixing them
would silently produce garbage results.

## When to move off Chroma

Chroma (embedded, single process) is right for this scale: under 100 papers is roughly 5–20k chunks.
Move to **Qdrant** (you want a server, payload filtering, or more than about 1M vectors) or to **pgvector**
(you already run Postgres, or need transactional joins with other data) when several users or processes
need to write at the same time, or when the app is deployed as a shared service.
