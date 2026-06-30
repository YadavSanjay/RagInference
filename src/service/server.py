"""Entrypoint for the RagInference HTTP service.

Run:
    conda run -n py312 python -m src.service.server
    # host/port/models via env:
    RAGINFER_HOST=0.0.0.0 RAGINFER_PORT=8799 RERANKER=jina EMBEDDER=bge \
        conda run -n py312 python -m src.service.server

Loads the default embedder + reranker once at startup (warm-up), then serves
``/embed`` and ``/rerank``. Clients (ARag, MacroRAG) point ``RAGINFER_URL`` at
``http://<host>:<port>`` and call the ``/embed`` / ``/rerank`` paths.
"""
from __future__ import annotations

import os

# transformers tokenizers must not fork inside the server's worker threads.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import uvicorn

from src.service.app import app, get_embedder, get_reranker
from src.logging_config import get_logger

logger = get_logger(__name__)


def _warmup() -> None:
    """Load the default embedder + reranker once and run a tiny inference each so
    the first real request is on the warm path. Best-effort: a load failure is
    logged, not fatal, so /healthz still reports per-model state."""
    import time

    t0 = time.time()
    logger.info("RagInference warm-up: loading default embedder %r + reranker %r ...",
                os.getenv("EMBEDDER", "bge"), os.getenv("RERANKER", "jina"))
    try:
        emb = get_embedder()
        emb.embed(["warmup"], dense=True, sparse=True)
        logger.info("RagInference warm-up: embedder %s ready (%.1fs)", emb.model_name, time.time() - t0)
    except Exception:
        logger.exception("RagInference warm-up: embedder failed (non-fatal)")
    try:
        rr = get_reranker()
        rr.rerank("warmup", ["warmup document text"])
        logger.info("RagInference warm-up: reranker %s ready", rr.model_name)
    except Exception:
        logger.exception("RagInference warm-up: reranker failed (non-fatal)")
    logger.info("RagInference warm-up: complete in %.1fs", time.time() - t0)


def main() -> None:
    host = os.getenv("RAGINFER_HOST", "127.0.0.1")
    port = int(os.getenv("RAGINFER_PORT", "8799"))
    if os.getenv("RAGINFER_WARMUP", "1").strip().lower() not in ("0", "false", "no", "off"):
        _warmup()
    logger.info("starting RagInference service on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":
    main()
