"""Embedder implementations + a factory. Default is BGE-M3 (dense+sparse)."""
from __future__ import annotations

import os

from src.embedders.base_embedder import BaseEmbedder
from src.logging_config import get_logger

logger = get_logger(__name__)


def build_embedder(name: str | None = None) -> BaseEmbedder:
    """Construct an embedder by name. Defaults to ``bge`` (env ``EMBEDDER``)."""
    name = (name or os.getenv("EMBEDDER") or "bge").strip().lower()
    if name in ("bge", "bge-m3", "bge_m3"):
        from src.embedders.bge_embedder import BGEEmbedder

        return BGEEmbedder()
    raise ValueError(f"unknown embedder {name!r} (expected 'bge')")


__all__ = ["BaseEmbedder", "build_embedder"]
