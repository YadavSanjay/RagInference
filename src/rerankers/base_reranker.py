"""Minimal reranker interface for the service.

Unlike XbrlRag's in-repo ``BaseReranker`` (which also carried subquery fusion,
recency/table boosts and Qdrant ``ScoredPoint`` coupling), the service only
needs the raw scoring contract; the score-combination orchestration stays on
the client side (XbrlRag) where the chunk metadata lives.
"""
from __future__ import annotations

from typing import List


class BaseReranker:
    """Score each document against a query; higher = more relevant."""

    #: human-readable model id, set by subclasses for /healthz + responses
    model_name: str = "base"

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        raise NotImplementedError("subclasses must implement rerank()")
