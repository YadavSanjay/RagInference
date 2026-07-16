"""RagInference — a standalone inference service for shared RAG models.

Extracted from ARag so the heavy models (``BAAI/bge-m3`` embedder, dense+sparse,
and ``jinaai/jina-reranker-v3`` reranker) are loaded **once** in a single process
and shared by every client (ARag, MacroRAG, ...) over HTTP — instead of each
client subprocess loading its own multi-GB copy and exhausting RAM.

Endpoints: ``POST /embed`` (dense+sparse), ``POST /rerank``, ``GET /healthz``.
"""
try:
    from dotenv import load_dotenv

    load_dotenv()  # load .env if present (optional) — before settings/logging read env
except Exception:  # pragma: no cover - dotenv optional
    pass

from src.logging_config import configure_logging

configure_logging()  # configure logging once at package import (settings-aware)

# Defaults for the factored implementation.
DEFAULT_RERANKER = "jina"
DEFAULT_EMBEDDER = "bge"
