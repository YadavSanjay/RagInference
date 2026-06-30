"""Minimal embedder interface for the service.

The service returns plain JSON-able structures; clients rebuild whatever
dataclasses they use. One result dict per input text, aligned to input order:

    {"dense": [float, ...] | None,
     "sparse": {"indices": [int, ...], "values": [float, ...]} | None}
"""
from __future__ import annotations

from typing import Any, Dict, List


class BaseEmbedder:
    """Encode texts to dense and/or sparse vectors."""

    model_name: str = "base"

    def embed(self, texts: List[str], dense: bool = True, sparse: bool = True) -> List[Dict[str, Any]]:
        raise NotImplementedError("subclasses must implement embed()")
