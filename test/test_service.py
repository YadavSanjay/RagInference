"""Unit tests for the RagInference service — no models are loaded (stub embedder
+ reranker are injected), so these are fast and CI-friendly.

    conda run -n py312 python -m pytest test/test_service.py -q
"""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from src.config.settings import Settings
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


def test_readyz_200_when_models_loaded(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["reranker_loaded"] is True and body["embedder_loaded"] is True


def test_readyz_503_before_models_loaded(monkeypatch):
    # no stubs injected -> caches empty -> not ready
    monkeypatch.setattr(app_module, "_rerankers", {})
    monkeypatch.setattr(app_module, "_embedders", {})
    c = TestClient(app_module.app)
    r = c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "loading"
    assert body["reranker_loaded"] is False and body["embedder_loaded"] is False


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


@pytest.fixture()
def tight_limits(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_settings",
        lambda: Settings(_env_file=None, max_items=2, max_text_chars=5),
    )


def test_defaults_clear_what_clients_send():
    # MacroRAG reranks at most 24 candidates; XbrlRag's vector load sends 400+ texts at once.
    assert Settings(_env_file=None).max_items >= 1000
    assert Settings(_env_file=None).max_text_chars >= 6000


def test_rerank_at_the_limits_is_accepted(client, tight_limits):
    r = client.post("/rerank", json={"query": "q", "documents": ["abcde"]})
    assert r.status_code == 200


def test_rerank_over_item_limit_is_413_counting_the_query(client, tight_limits):
    r = client.post("/rerank", json={"query": "q", "documents": ["a", "b"]})
    assert r.status_code == 413
    assert "RAGINFER_MAX_ITEMS" in r.json()["detail"]


def test_rerank_over_length_query_is_413(client, tight_limits):
    r = client.post("/rerank", json={"query": "abcdef", "documents": ["a"]})
    assert r.status_code == 413
    assert "RAGINFER_MAX_TEXT_CHARS" in r.json()["detail"]


def test_embed_over_item_limit_is_413(client, tight_limits):
    r = client.post("/embed", json={"texts": ["a", "b", "c"]})
    assert r.status_code == 413


def test_embed_over_length_text_is_413(client, tight_limits):
    r = client.post("/embed", json={"texts": ["abcdef"]})
    assert r.status_code == 413


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


class _ConcurrencyProbe:
    """Records the peak number of calls in flight at once."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def __enter__(self):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def __exit__(self, *exc):
        time.sleep(0.05)  # hold the "forward pass" open so overlaps would show
        with self._lock:
            self.active -= 1


def _fire_concurrently(fn, n=4):
    threads = [threading.Thread(target=fn) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_reranks_are_serialized(monkeypatch):
    probe = _ConcurrencyProbe()

    class _SlowReranker(_StubReranker):
        def rerank(self, query, documents):
            with probe:
                return super().rerank(query, documents)

    monkeypatch.setattr(app_module, "_rerankers", {"default": _SlowReranker()})
    c = TestClient(app_module.app)
    statuses = []
    _fire_concurrently(lambda: statuses.append(
        c.post("/rerank", json={"query": "q", "documents": ["aa"]}).status_code))
    assert statuses == [200] * 4
    assert probe.peak == 1


def test_concurrent_embeds_are_serialized(monkeypatch):
    probe = _ConcurrencyProbe()

    class _SlowEmbedder(_StubEmbedder):
        def embed(self, texts, dense=True, sparse=True):
            with probe:
                return super().embed(texts, dense, sparse)

    monkeypatch.setattr(app_module, "_embedders", {"default": _SlowEmbedder()})
    c = TestClient(app_module.app)
    statuses = []
    _fire_concurrently(lambda: statuses.append(
        c.post("/embed", json={"texts": ["abc"]}).status_code))
    assert statuses == [200] * 4
    assert probe.peak == 1


def test_rerank_unknown_model_is_422(client, monkeypatch):
    def boom(name=None):
        raise ValueError(f"unknown reranker {name!r}")

    monkeypatch.setattr(app_module, "get_reranker", boom)
    r = client.post("/rerank", json={"query": "q", "documents": ["x"], "model": "nope"})
    assert r.status_code == 422
    assert "unknown reranker" in r.json()["detail"]
