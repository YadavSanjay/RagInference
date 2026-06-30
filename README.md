# RagInference

A standalone **inference service** for shared RAG models, extracted from ARag so
the heavy models load **once** in a single process and are shared by every client
(ARag, MacroRAG, …) over HTTP — instead of each client subprocess loading its own
multi-GB copy and exhausting RAM.

Models (defaults):
- **Embedder** — `BAAI/bge-m3` (dense + sparse / lexical).
- **Reranker** — `jinaai/jina-reranker-v3` (local `transformers` model — no Jina
  API / `JINA_API_KEY`). `RERANKER=bge` selects the lighter `bge-reranker-v2-m3`.

## Why

Two MCP subprocesses (ARag + MacroRAG) each loaded their own copy of BGE-M3
(~2.5 GB) + jina-reranker-v3 (~0.9 GB), and concurrent cold loads spiked RAM into
swap — inflating a ~92s rerank to >300s. One shared service removes the duplicate
copies and serializes calls so the two servers stop contending.

## Layout (mirrors ARag)

```
src/
  __init__.py            # configure_logging + load_dotenv; DEFAULT_RERANKER/EMBEDDER
  logging_config.py      # stderr-only logging
  embedders/
    base_embedder.py     # embed(texts, dense, sparse) -> list[{dense, sparse}]
    bge_embedder.py      # BAAI/bge-m3 dense+sparse (default)
    __init__.py          # build_embedder(name)
  rerankers/
    base_reranker.py     # rerank(query, list[str]) -> list[float]
    jina_reranker.py     # jina-reranker-v3 (default; bounded-blocking memory fix)
    bge_reranker.py      # bge-reranker-v2-m3 cross-encoder (RERANKER=bge)
    __init__.py          # build_reranker(name)
  service/
    schemas.py           # Embed/Rerank/Health wire models
    app.py               # FastAPI: POST /embed, POST /rerank, GET /healthz
    server.py            # uvicorn entrypoint + startup warm-up (both models)
configs/logging.yaml
test/
```

## Run

```bash
conda run -n py312 python -m src.service.server
# or configure:
RAGINFER_HOST=0.0.0.0 RAGINFER_PORT=8799 RERANKER=jina EMBEDDER=bge \
    conda run -n py312 python -m src.service.server
```

Env:
- `RAGINFER_HOST` (default `127.0.0.1`) / `RAGINFER_PORT` (default `8799`)
- `RERANKER` — `jina` (default) | `bge`
- `EMBEDDER` — `bge` (default)
- `RAGINFER_WARMUP` — `1` (default) loads + warms both models at startup

## API

`POST /embed`
```json
{ "texts": ["export control restrictions"], "dense": true, "sparse": true }
```
→
```json
{ "embeddings": [ { "dense": [0.01, "..."], "sparse": { "indices": [4, 91], "values": [0.7, 0.3] } } ],
  "model": "BAAI/bge-m3", "count": 1 }
```

`POST /rerank`
```json
{ "query": "export control restrictions", "documents": ["text a", "text b"] }
```
→ `{ "scores": [0.91, 0.12], "model": "jinaai/jina-reranker-v3", "count": 2 }`

`GET /healthz` → `{ "status": "ok", "embedder": "...", "reranker": "...", "embedder_loaded": true, "reranker_loaded": true }`

## Clients

Point each client's `RAGINFER_URL` at `http://<host>:8799`. ARag ships
`RemoteReranker` (`ARag/src/retriever/remote_reranker.py`) and `RemoteEmbedder`
(`ARag/src/embeddings/remote_embedder.py`); the XBRL narrative retriever uses
both automatically when `RAGINFER_URL` is set, falling back to local models only
when it is not. MacroRAG needs equivalent shims.
