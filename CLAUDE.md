# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone HTTP **inference service** that loads heavy RAG models (a BGE-M3
embedder + a reranker) **once** in a single process and serves `/embed` and
`/rerank` to multiple clients (ARag, MacroRAG). The whole point is to avoid each
client subprocess loading its own multi-GB copy and exhausting RAM — so
correctness here means *one shared copy, memory-safe, calls serialized*.

## Commands

Dev happens in the **`py312` conda env** (models need Python 3.12 + torch).

```bash
# Run the service (127.0.0.1:8799 by default)
conda run -n py312 python -m src.service.server
# Configure host/port/models via env:
RAGINFER_HOST=0.0.0.0 RAGINFER_PORT=8799 RERANKER=jina EMBEDDER=bge \
    conda run -n py312 python -m src.service.server

# Tests (fast — stub models injected, no weights loaded)
conda run -n py312 python -m pytest test/ -q                            # service + settings
conda run -n py312 python -m pytest test/test_service.py -q -k rerank   # a single test
conda run -n py312 python -m pytest test/test_settings.py -q            # DEPLOY_MODE / storage seam
```

Env knobs: `DEPLOY_MODE` (`local` default | `aws`), `RAGINFER_HOST` / `RAGINFER_PORT`,
`RERANKER` (`jina` default | `bge`), `EMBEDDER` (`bge`), `RAGINFER_WARMUP` (`1` default —
loads + warms both models at startup so the first real request is on the warm path).

Container builds live in `infra/` — see [`infra/README.md`](infra/README.md). Two
image shapes from one `Dockerfile` via `--build-arg BAKE_MODELS`: fat (weights
baked in) vs slim (weights referenced from S3/volume at runtime). The seam is
`HF_HOME` — the service always reads weights from the HF cache, so build-time
bake and runtime sync are interchangeable. The `Dockerfile` is **multi-arch**
(`TARGETARCH`): x86_64 pins the CPU torch index, **arm64/Graviton** uses the PyPI
aarch64 CPU wheel. Measured sizes: slim ~2.4 GB, fat ~6.5 GB. AWS deployment
(ECS Fargate, ARM64, VPC-internal, optional scheduled scale-to-zero) is analyzed in
[`docs/aws-deployment-plan.md`](docs/aws-deployment-plan.md); the Fargate contract +
task-def template live in [`infra/aws/`](infra/aws/); the build → ECR → EC2/Fargate
runbook is [`docs/deployment-guide.md`](docs/deployment-guide.md).

The container **fails fast** if `MODEL_SOURCE=baked` but `HF_HOME` is empty (slim
image with no weights), and its `HEALTHCHECK` is bound to **`/readyz`**, not
`/healthz` — an always-200 liveness probe would report a model-less container as
healthy. Don't regress either.

## Architecture

The request path is thin; the weight is in the model wrappers.

- **Uniform contracts.** Every embedder implements
  `embed(texts, dense, sparse) -> list[{dense, sparse}]`; every reranker
  implements `rerank(query, documents) -> list[float]` **aligned to input order**.
  The FastAPI layer ([src/service/app.py](src/service/app.py)) only marshals JSON
  to/from these. Preserve alignment and the `{indices, values}` sparse shape —
  clients rebuild vectors from it.

- **Factories select implementations by name.**
  [build_embedder](src/embedders/__init__.py) / [build_reranker](src/rerankers/__init__.py)
  map an alias (or the `EMBEDDER`/`RERANKER` env var) to a concrete class and
  import it lazily. Adding a model = new wrapper subclass + a branch in the
  factory; nothing else changes.

- **Lazy singleton caching under locks.** `get_embedder` / `get_reranker` in
  [app.py](src/service/app.py) build each model once on first use behind a
  `threading.Lock` (double-checked) and cache it in a module dict keyed by
  `"default"` and by model name. This is what makes it *shared* — do not
  introduce per-request construction or multiple uvicorn workers; scale with
  replicas, not workers (each worker = another full copy of the weights).

- **Startup warm-up.** [server.py](src/service/server.py) loads and runs a tiny
  inference on both models before serving (best-effort; a load failure is logged,
  not fatal, and `/healthz` reports per-model `*_loaded` state). `/healthz` is
  always-200 **liveness**; `/readyz` is **readiness** — 503 until both models load
  — and is the intended load-balancer / ECS target-group health check.

- **One config seam for local ↔ AWS.** [src/config/settings.py](src/config/settings.py)
  is a typed `Settings` driven by `DEPLOY_MODE` (`local` default | `aws`).
  `DEPLOY_MODE` only supplies *defaults* — every legacy env var
  (`RAGINFER_HOST`, `RERANKER`, `EMBEDDER`, `MODEL_SOURCE`, `RAGINFER_WARMUP`,
  `LOG_LEVEL`, …) still wins when set, so local behavior is unchanged. In `aws`:
  bind `0.0.0.0`, **JSON logs to stdout** (no rotating file, for CloudWatch — see
  [logging_config.py](src/logging_config.py)), and the storage backend defaults to
  S3. Read config through `get_settings()`, not ad-hoc `os.getenv`.

- **Storage seam (forward-looking).** [src/storage/__init__.py](src/storage/__init__.py)
  (`LocalStorage` | `S3Storage` via `build_storage`) is the local-FS↔S3 adapter for
  *future* blob IO. The `/embed` `/rerank` path does no blob IO today, and **model
  weights deliberately do NOT go through it** — they load from `HF_HOME` via
  `MODEL_SOURCE` (baked/hf/s3), because `from_pretrained` needs a real directory.

### Memory-safety invariants (the reason this service exists — don't regress them)

These are CPU forward passes; the defaults below are deliberately conservative to
avoid segfaults/OOM, not arbitrary:

- **BGE-M3** ([bge_embedder.py](src/embedders/bge_embedder.py)) caps
  `batch_size=12` / `max_length=8192`. FlagEmbedding's own default
  `batch_size=256` builds one huge forward pass and segfaults on CPU.

- **jina-reranker-v3** ([jina_reranker.py](src/rerankers/jina_reranker.py)) is
  *listwise* (O(total_tokens²) memory per forward). Its shipped
  `model_max_length` sentinel (131072) disables the model's internal token-budget
  blocking, so everything packs into one giant forward and OOMs on long SEC-style
  chunks. The wrapper forces `_tokenizer.model_max_length = 4096` (block budget)
  and truncates docs to 512 tokens. Both bounds are load-bearing.

## Clients

Clients point `RAGINFER_URL` at `http://<host>:8799`. ARag ships `RemoteReranker`
/ `RemoteEmbedder` shims that call this service when `RAGINFER_URL` is set and
fall back to local models otherwise; MacroRAG needs equivalent shims. See
[docs/user-guide.md](docs/user-guide.md).
