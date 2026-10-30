"""Reranker implementations + a factory.

The service contract is uniform across implementations:
``rerank(query: str, documents: list[str]) -> list[float]`` returning one score
per document, aligned to input order. Jina is the default.
"""
from __future__ import annotations

import os

from src.rerankers.base_reranker import BaseReranker
from src.logging_config import get_logger

logger = get_logger(__name__)


def build_reranker(name: str | None = None) -> BaseReranker:
    """Construct a reranker by name. Defaults to ``jina`` (env ``RERANKER``)."""
    name = (name or os.getenv("RERANKER") or "jina").strip().lower()
    if name in ("jina", "jina-reranker-v3", "v3"):
        from src.rerankers.jina_reranker import JinaReranker

        return JinaReranker()
    if name in ("bge", "bge-reranker-v2-m3", "m3"):
        from src.rerankers.bge_reranker import BGEReranker

        return BGEReranker()
    raise ValueError(f"unknown reranker {name!r} (expected 'jina' or 'bge')")


__all__ = ["BaseReranker", "build_reranker"]
