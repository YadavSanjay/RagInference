"""FastAPI app exposing the shared inference models (embed + rerank).

A single process loads the (default BGE-M3 embedder + Jina reranker) once and
serves ``POST /embed`` and ``POST /rerank`` to every client, so ARag / MacroRAG
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
    try:
        rr = get_reranker(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not req.documents:
        return RerankResponse(scores=[], model=rr.model_name, count=0)
    t0 = time.time()
    scores = rr.rerank(req.query, req.documents)
    logger.info("rerank: model=%s docs=%d took=%.1fs query=%r",
                rr.model_name, len(req.documents), time.time() - t0, req.query[:80])
    return RerankResponse(scores=scores, model=rr.model_name, count=len(scores))


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    try:
        emb = get_embedder(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not req.texts:
        return EmbedResponse(embeddings=[], model=emb.model_name, count=0)
    t0 = time.time()
    raw = emb.embed(req.texts, dense=req.dense, sparse=req.sparse)
    items = [
        EmbeddingItem(
            dense=r.get("dense"),
            sparse=SparseVectorModel(**r["sparse"]) if r.get("sparse") is not None else None,
        )
        for r in raw
    ]
    logger.info("embed: model=%s texts=%d dense=%s sparse=%s took=%.2fs",
                emb.model_name, len(req.texts), req.dense, req.sparse, time.time() - t0)
    return EmbedResponse(embeddings=items, model=emb.model_name, count=len(items))
