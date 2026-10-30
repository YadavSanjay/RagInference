"""FastAPI app exposing the shared inference models (embed + rerank).

A single process loads the (default BGE-M3 embedder + Jina reranker) once and
serves ``POST /embed`` and ``POST /rerank`` to every client, so XbrlRag / MacroRAG
no longer each load their own multi-GB copies.

Each model is built lazily under a lock on first use and reused thereafter; the
server entrypoint (``server.py``) warms them at startup so the first real call
is already on the warm path.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.config.settings import get_settings
from src.embedders import BaseEmbedder, build_embedder
from src.rerankers import BaseReranker, build_reranker
from src.service.schemas import (
    EmbeddingItem,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    RerankRequest,
    RerankResponse,
    SparseVectorModel,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

_rerank_lock = threading.Lock()
_embed_lock = threading.Lock()
# Inference locks — distinct from the build locks above. The sync endpoints run
# on FastAPI's threadpool, so without these, overlapping requests would run
# concurrent forward passes on one model: each uses every core (torch
# intra-op threads), so they oversubscribe the CPU and multiply activation
# memory. Serializing per model keeps one forward in flight; extra requests
# queue here (FIFO-ish) and the wait is logged. Embed and rerank stay separate
# so a sub-second query embed doesn't wait behind a multi-minute rerank.
_rerank_infer_lock = threading.Lock()
_embed_infer_lock = threading.Lock()
_rerankers: dict[str, BaseReranker] = {}
_embedders: dict[str, BaseEmbedder] = {}


def get_reranker(name: Optional[str] = None) -> BaseReranker:
    """Return a cached reranker by name (default from env), building once."""
    key = (name or "").strip().lower() or "default"
    rr = _rerankers.get(key)
    if rr is None:
        with _rerank_lock:
            rr = _rerankers.get(key)
            if rr is None:
                rr = build_reranker(None if key == "default" else key)
                _rerankers[key] = rr
                _rerankers.setdefault(rr.model_name.split("/")[-1].lower(), rr)
    return rr


def get_embedder(name: Optional[str] = None) -> BaseEmbedder:
    """Return a cached embedder by name (default from env), building once."""
    key = (name or "").strip().lower() or "default"
    emb = _embedders.get(key)
    if emb is None:
        with _embed_lock:
            emb = _embedders.get(key)
            if emb is None:
                emb = build_embedder(None if key == "default" else key)
                _embedders[key] = emb
                _embedders.setdefault(emb.model_name.split("/")[-1].lower(), emb)
    return emb


def _enforce_limits(texts: list[str]) -> None:
    """Reject a payload over the configured bounds before it reaches a model.

    Args:
        texts: Every text in the request, including a rerank query.

    Raises:
        HTTPException: 413 when the count or any single text exceeds its limit.
    """
    settings = get_settings()
    if len(texts) > settings.max_items:
        raise HTTPException(
            status_code=413,
            detail=f"{len(texts)} items exceeds RAGINFER_MAX_ITEMS={settings.max_items}",
        )
    longest = max((len(t) for t in texts), default=0)
    if longest > settings.max_text_chars:
        raise HTTPException(
            status_code=413,
            detail=f"text of {longest} chars exceeds "
            f"RAGINFER_MAX_TEXT_CHARS={settings.max_text_chars}",
        )


app = FastAPI(title="RagInference", version="0.1.0")


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    rr = _rerankers.get("default")
    emb = _embedders.get("default")
    return HealthResponse(
        status="ok",
        reranker=rr.model_name if rr else "(not loaded)",
        embedder=emb.model_name if emb else "(not loaded)",
        reranker_loaded=rr is not None,
        embedder_loaded=emb is not None,
    )


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness probe for load balancers / ECS target-group health checks.

    Unlike ``/healthz`` (always 200 while the process is up = liveness), this
    returns **503 until both default models are loaded**, so a balancer won't
    route traffic to a task that is still warming up or syncing weights.
    """
    rr = _rerankers.get("default")
    emb = _embedders.get("default")
    ready = rr is not None and emb is not None
    body = {
        "status": "ready" if ready else "loading",
        "reranker": rr.model_name if rr else "(not loaded)",
        "embedder": emb.model_name if emb else "(not loaded)",
        "reranker_loaded": rr is not None,
        "embedder_loaded": emb is not None,
    }
    return JSONResponse(body, status_code=200 if ready else 503)


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    _enforce_limits([req.query, *req.documents])
    try:
        rr = get_reranker(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not req.documents:
        return RerankResponse(scores=[], model=rr.model_name, count=0)
    t_wait = time.time()
    with _rerank_infer_lock:
        t0 = time.time()
        scores = rr.rerank(req.query, req.documents)
    logger.info("rerank: model=%s docs=%d waited=%.1fs took=%.1fs query=%r",
                rr.model_name, len(req.documents), t0 - t_wait, time.time() - t0,
                req.query[:80])
    return RerankResponse(scores=scores, model=rr.model_name, count=len(scores))


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    _enforce_limits(req.texts)
    try:
        emb = get_embedder(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not req.texts:
        return EmbedResponse(embeddings=[], model=emb.model_name, count=0)
    t_wait = time.time()
    with _embed_infer_lock:
        t0 = time.time()
        raw = emb.embed(req.texts, dense=req.dense, sparse=req.sparse)
    waited = t0 - t_wait
    items = [
        EmbeddingItem(
            dense=r.get("dense"),
            sparse=SparseVectorModel(**r["sparse"]) if r.get("sparse") is not None else None,
        )
        for r in raw
    ]
    logger.info("embed: model=%s texts=%d dense=%s sparse=%s waited=%.2fs took=%.2fs",
                emb.model_name, len(req.texts), req.dense, req.sparse, waited,
                time.time() - t0)
    return EmbedResponse(embeddings=items, model=emb.model_name, count=len(items))
