"""storage/manifest.json: which PDFs are indexed, with their content hash, for incremental ingest."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"embedding_model": None, "documents": {}}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)

    @property
    def documents(self) -> dict[str, dict]:
        return self.data["documents"]

    def get(self, source: str) -> dict | None:
        return self.documents.get(source)

    def put(self, source: str, sha256: str, pages: int, chunks: int, title: str, warnings: list[str]) -> None:
        self.documents[source] = {
            "sha256": sha256,
            "title": title,
            "pages": pages,
            "chunks": chunks,
            "warnings": warnings,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def remove(self, source: str) -> None:
        self.documents.pop(source, None)

    def reset(self, embedding_model: str) -> None:
        self.data = {"embedding_model": embedding_model, "documents": {}}

    def save(self) -> None:
        # Write to a temp file then rename, so a crash can't leave a half-written manifest.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)
