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
  __init__.py            # load_dotenv + configure_logging (settings-aware); DEFAULT_RERANKER/EMBEDDER
  logging_config.py      # local: stderr + rotating file; aws: JSON to stdout (CloudWatch)
  config/
    settings.py          # typed Settings + DEPLOY_MODE (local|aws); one switch, legacy env preserved
  storage/
    __init__.py          # Storage seam: LocalStorage | S3Storage (build_storage); future blob IO
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
    app.py               # FastAPI: POST /embed, POST /rerank, GET /healthz, GET /readyz
    server.py            # uvicorn entrypoint + startup warm-up (both models)
configs/logging.yaml
infra/                   # containerization + AWS (infra/aws/) — see infra/README.md
docs/aws-deployment-plan.md  # local + AWS deployment analysis / plan / change list
test/                    # test_service.py, test_settings.py (stub models, no weights)
```

## Run

```bash
conda run -n py312 python -m src.service.server
# or configure:
RAGINFER_HOST=0.0.0.0 RAGINFER_PORT=8799 RERANKER=jina EMBEDDER=bge \
    conda run -n py312 python -m src.service.server
```

Env:
- `DEPLOY_MODE` — `local` (default) | `aws`. In `aws` it derives cloud-friendly
  defaults (bind `0.0.0.0`, JSON logs to stdout, no log file, S3 storage backend);
  any explicit var below still overrides. Local behavior is unchanged when unset.
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
(always 200 while the process is up = **liveness**.)

`GET /readyz` → same body, but **503 until both default models are loaded**, then 200
(= **readiness**; used as the load-balancer / ECS target-group health check so traffic
only lands on a warm task).

## Deployment (local & AWS)

The same image runs unchanged locally and on AWS; `DEPLOY_MODE` flips the defaults.
The chosen cloud target is **ECS Fargate (CPU, ARM64/Graviton — ~20% cheaper), VPC-internal**,
with optional **scheduled scale-to-zero** off-hours (load is batch-tolerant).

- **Copy-paste runbook** — local amd64 build/run + AWS arm64 build & push to ECR:
  [`docs/runbook/build-and-push-ecr.md`](docs/runbook/build-and-push-ecr.md).
- **Full deployment guide** — build → push to ECR → pull on a host → run on EC2/Fargate:
  [`docs/deployment-guide.md`](docs/deployment-guide.md).
- Full analysis, change list, and the **AWS services the infra project must provision**:
  [`docs/aws-deployment-plan.md`](docs/aws-deployment-plan.md).
- Container build (fat vs slim, ARM64 buildx recipe): [`infra/README.md`](infra/README.md).
- Fargate contract, task-def template, scheduled scale-to-zero: [`infra/aws/`](infra/aws/).

## Clients

Point each client's `RAGINFER_URL` at `http://<host>:8799`. ARag ships
`RemoteReranker` (`ARag/src/retriever/remote_reranker.py`) and `RemoteEmbedder`
(`ARag/src/embeddings/remote_embedder.py`); the XBRL narrative retriever uses
both automatically when `RAGINFER_URL` is set, falling back to local models only
when it is not. MacroRAG needs equivalent shims.

See [`docs/user-guide.md`](docs/user-guide.md) for a client-facing guide:
connecting via `RAGINFER_URL`, the full API reference, and common troubleshooting.
