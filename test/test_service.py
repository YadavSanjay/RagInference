"""Unit tests for the RagInference service — no models are loaded (stub embedder
+ reranker are injected), so these are fast and CI-friendly.

    conda run -n py312 python -m pytest test/test_service.py -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.embedders.base_embedder import BaseEmbedder
from src.rerankers.base_reranker import BaseReranker
from src.service import app as app_module


class _StubReranker(BaseReranker):
    model_name = "stub/reranker"

    def rerank(self, query, documents):
        # deterministic, length-based score so we can assert ordering/alignment
        return [float(len(d)) for d in documents]


class _StubEmbedder(BaseEmbedder):
    model_name = "stub/embedder"

    def embed(self, texts, dense=True, sparse=True):
        out = []
        for i, t in enumerate(texts):
            item = {}
            if dense:
                item["dense"] = [float(len(t)), float(i)]
            if sparse:
                item["sparse"] = {"indices": [i], "values": [float(len(t))]}
            out.append(item)
        return out


@pytest.fixture()
def client(monkeypatch):
    # inject stubs as the cached defaults so no real models are built
    monkeypatch.setattr(app_module, "_rerankers", {"default": _StubReranker()})
    monkeypatch.setattr(app_module, "_embedders", {"default": _StubEmbedder()})
    return TestClient(app_module.app)


def test_healthz_reports_both_models(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["reranker"] == "stub/reranker" and body["reranker_loaded"] is True
    assert body["embedder"] == "stub/embedder" and body["embedder_loaded"] is True


def test_rerank_returns_aligned_scores(client):
    r = client.post("/rerank", json={"query": "q", "documents": ["aa", "bbbb", "c"]})
    assert r.status_code == 200
    body = r.json()
    assert body["scores"] == [2.0, 4.0, 1.0]
    assert body["count"] == 3
    assert body["model"] == "stub/reranker"


def test_rerank_empty_documents(client):
    r = client.post("/rerank", json={"query": "q", "documents": []})
    assert r.status_code == 200
    assert r.json()["scores"] == []


def test_embed_returns_dense_and_sparse(client):
    r = client.post("/embed", json={"texts": ["abc", "de"], "dense": True, "sparse": True})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and body["model"] == "stub/embedder"
    assert body["embeddings"][0]["dense"] == [3.0, 0.0]
    assert body["embeddings"][0]["sparse"] == {"indices": [0], "values": [3.0]}
    assert body["embeddings"][1]["dense"] == [2.0, 1.0]


def test_embed_dense_only_omits_sparse(client):
    r = client.post("/embed", json={"texts": ["abc"], "dense": True, "sparse": False})
    assert r.status_code == 200
    item = r.json()["embeddings"][0]
    assert item["dense"] == [3.0, 0.0]
    assert item["sparse"] is None


def test_embed_empty_texts(client):
    r = client.post("/embed", json={"texts": []})
    assert r.status_code == 200
    assert r.json()["embeddings"] == []


def test_rerank_unknown_model_is_422(client, monkeypatch):
    def boom(name=None):
        raise ValueError(f"unknown reranker {name!r}")

    monkeypatch.setattr(app_module, "get_reranker", boom)
    r = client.post("/rerank", json={"query": "q", "documents": ["x"], "model": "nope"})
    assert r.status_code == 422
    assert "unknown reranker" in r.json()["detail"]
