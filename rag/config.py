"""Typed settings loaded from config.yaml (+ secrets from .env)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    pdf_dir: str = "data/pdfs"
    storage_dir: str = "storage"
    log_dir: str = "logs"


class OCRConfig(BaseModel):
    enabled: bool = False
    min_chars_before_ocr: int = 50
    language: str = "eng"
    dpi: int = 300
    # Folder with *.traineddata; empty = $TESSDATA_PREFIX or the default Windows install.
    tessdata: str = ""


class ParsingConfig(BaseModel):
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    extract_tables: bool = True
    header_footer_margin: float = 0.08
    header_footer_min_repeat: float = 0.5
    heading_size_ratio: float = 1.15
    drop_sections: list[str] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50
    min_chunk_tokens: int = 40
    add_context_header: bool = True


class LocalEmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    query_prefix: str = ""


class VoyageEmbeddingConfig(BaseModel):
    model: str = "voyage-3.5"
    batch_size: int = 64


class EmbeddingConfig(BaseModel):
    provider: Literal["local", "voyage"] = "local"
    local: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)
    voyage: VoyageEmbeddingConfig = Field(default_factory=VoyageEmbeddingConfig)

    @property
    def identity(self) -> str:
        """Stable id of the embedding model; stored in the index and checked at query time."""
        model = self.local.model if self.provider == "local" else self.voyage.model
        return f"{self.provider}:{model}"


class VectorStoreConfig(BaseModel):
    collection: str = "papers"


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 30


class RetrievalConfig(BaseModel):
    vector_candidates: int = 30
    bm25_candidates: int = 30
    rrf_k: int = 60
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    top_k: int = 6


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    answer_model: str = "qwen3:4b-instruct"
    rewrite_model: str = "qwen3:4b-instruct"
    num_ctx: int = 8192
    temperature: float = 0.0
    timeout_seconds: float = 300


class AnthropicConfig(BaseModel):
    answer_model: str = "claude-sonnet-5"
    rewrite_model: str = "claude-haiku-4-5-20251001"
    max_retries: int = 4
    timeout_seconds: float = 60


class GroqConfig(BaseModel):
    answer_model: str = "llama-3.3-70b-versatile"
    rewrite_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_retries: int = 4
    timeout_seconds: float = 60


class OpenAICompatConfig(BaseModel):
    """Any OpenAI-compatible chat API. Defaults: Google Gemini's free tier."""
    name: str = "Gemini"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_env: str = "GEMINI_API_KEY"
    answer_model: str = "gemini-flash-latest"
    rewrite_model: str = "gemini-flash-lite-latest"
    temperature: float = 0.0
    reasoning_effort: str = ""  # e.g. "low" for thinking models; empty = provider default
    max_retries: int = 4
    timeout_seconds: float = 90


class GenerationConfig(BaseModel):
    provider: Literal["ollama", "groq", "openai_compat", "anthropic"] = "ollama"
    history_turns: int = 3
    max_tokens: int = 1000
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @staticmethod
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def pdf_dir(self) -> Path:
        return self._resolve(self.paths.pdf_dir)

    @property
    def storage_dir(self) -> Path:
        return self._resolve(self.paths.storage_dir)

    @property
    def log_dir(self) -> Path:
        return self._resolve(self.paths.log_dir)


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings. Precedence: explicit path > $RAG_CONFIG > ./config.yaml."""
    load_dotenv(PROJECT_ROOT / ".env")
    # Windows without Developer Mode can't symlink; HF falls back to copies fine, so silence the warning.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cfg_path = Path(path or os.getenv("RAG_CONFIG") or PROJECT_ROOT / "config.yaml")
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    data = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return Settings.model_validate(data)
