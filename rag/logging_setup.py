"""Application logging: console + rotating file in logs/rag.log."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import Settings

_configured = False


def setup_logging(settings: Settings) -> None:
    global _configured
    if _configured:
        return
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(
        settings.log_dir / "rag.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(settings.logging.level.upper())
    root.addHandler(file_handler)
    root.addHandler(console)
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # "unauthenticated requests to the HF Hub" is harmless for public models.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    _configured = True
