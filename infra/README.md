# RagInference — Containerization (`infra/`)

Container build and runtime for the RagInference service, designed around one
constraint: the model weights total **~4 GB** (`BAAI/bge-m3` ~2.3 GB +
`jinaai/jina-reranker-v3` ~0.9 GB, plus the optional `bge-reranker-v2-m3`
~1.1 GB). Everything here is CPU-only — the service does CPU forward passes and
never touches CUDA.

Two weight-delivery strategies, both produced from a **single parameterized
[`Dockerfile`](Dockerfile)** (toggled by the `BAKE_MODELS` build arg):

| | **Case A — fat / self-contained** | **Case B — slim / referenced** |
|---|---|---|
| Weights | Baked into the image at build | In S3 / shared volume, synced at runtime |
| Image size | ~4–6 GB | **~1–1.5 GB** |
| Cold start | Fastest (no fetch) | First run pays a sync/download |
| Runtime deps | None (air-gap OK) | Object store + credentials |
| Weight update | Rebuild + repush whole image | Update S3; restart, no rebuild |
| Best for | Reproducible/offline deploys, CI pins | Many replicas, frequent weight churn |

Both rely on the same seam: the service reads weights from **`HF_HOME`**. We just
change what's behind it.

## The size levers (why the image isn't 8 GB)

1. **CPU-only torch** — installed from `download.pytorch.org/whl/cpu`, *before*
   the other requirements, so no CUDA/`nvidia-*` wheels are ever pulled. ~200 MB
   instead of ~2.5 GB. Single biggest win, applies to **both** cases.
2. **Multi-stage build** — `build-essential` and the download tooling live in
   the builder stage; the runtime image is `python:3.12-slim` + venv + code.
3. **Trimmed snapshots** — [`download_models.py`](download_models.py) ignores
   onnx / TF / Flax / gguf weights the service never loads.
4. **Context hygiene** — [`Dockerfile.dockerignore`](Dockerfile.dockerignore)
   keeps `.git`, local caches, tests, and docs out of the build context.

---

## Case A — fat image (weights baked in)

```bash
# from repo root
docker build -f infra/Dockerfile --build-arg BAKE_MODELS="bge jina" -t raginfer:fat .
docker run --rm -p 8799:8799 raginfer:fat
# add the bge reranker too:  --build-arg BAKE_MODELS="bge jina bge-reranker"
```

At runtime `MODEL_SOURCE=baked` (the default) sets `HF_HUB_OFFLINE=1` — the
container never reaches out to HuggingFace.

Compose equivalent:

```bash
docker compose -f infra/docker-compose.yml --profile fat up --build
```

## Case B — slim image (weights referenced from S3)

The image carries no weights. At runtime the weights are made available under
`HF_HOME` (`/models`) by one of:

- **Init container (recommended, in the compose file):** an `aws-cli` container
  `s3 sync`s the weights into a shared volume once, then the slim app container
  mounts it read-only and runs offline. Keeps the app image free of S3 tooling.
- **In-container sync:** run with `MODEL_SOURCE=s3` + `MODEL_S3_URI=...` and an
  S3 client (`s5cmd`/`aws`) on PATH; [`entrypoint.sh`](entrypoint.sh) syncs then
  runs offline.
- **Persistent HF cache:** `MODEL_SOURCE=hf` with `HF_HOME` on a named volume —
  first run downloads from the Hub, later runs/replicas reuse it. No S3 needed.

One-time: put the weights in your bucket in HF-cache layout.

```bash
# 1. materialize weights locally (into ./models)
HF_HOME=./models python infra/download_models.py bge jina
# 2. upload to S3 (or let the compose 'minio-setup' service seed local MinIO)
aws s3 sync ./models s3://raginfer-models
```

Run the full local demo (slim image + MinIO standing in for S3):

```bash
cp infra/.env.example infra/.env        # tweak if desired
HF_HOME=./models python infra/download_models.py   # seed ./models for MinIO
docker compose -f infra/docker-compose.yml --profile slim up --build
```

For **real S3**: drop the `minio*` services, unset `AWS_ENDPOINT_URL`, set
`MODEL_S3_URI=s3://your-bucket/prefix`, and supply credentials via an IAM role
(preferred) or `AWS_*` env vars.

---

## Verify

```bash
curl localhost:8799/healthz
curl -s localhost:8799/embed  -H 'content-type: application/json' \
     -d '{"texts":["export control restrictions"],"dense":true,"sparse":true}'
curl -s localhost:8799/rerank -H 'content-type: application/json' \
     -d '{"query":"q","documents":["a","b"]}'
```

`/healthz` reports `embedder_loaded` / `reranker_loaded` — both are `true` once
the startup warm-up finishes (give it up to a few minutes on first cold start;
the `HEALTHCHECK` has a 180 s `start-period` for exactly this).

## Files

| File | Purpose |
|---|---|
| [`Dockerfile`](Dockerfile) | Multi-stage build; `BAKE_MODELS` selects fat vs slim |
| [`Dockerfile.dockerignore`](Dockerfile.dockerignore) | Build-context ignore rules |
| [`download_models.py`](download_models.py) | Snapshot weights into `HF_HOME` (build-time bake or S3 prep) |
| [`entrypoint.sh`](entrypoint.sh) | Resolves `MODEL_SOURCE` (baked/hf/s3), then starts the service |
| [`docker-compose.yml`](docker-compose.yml) | `fat` and `slim` profiles; MinIO demo for Case B |
| [`.env.example`](.env.example) | Template env for the compose Case-B setup |

## Notes & tradeoffs

- **`RAGINFER_HOST` is forced to `0.0.0.0`** in the image (the app defaults to
  `127.0.0.1`, which is unreachable from outside a container).
- **Warm-up + first cold start.** With `MODEL_SOURCE=hf` or `s3`, the very first
  container also pays the download/sync before it can serve — plan health checks
  and orchestration timeouts accordingly (hence the generous `start-period`).
- **Concurrency.** The service serializes model calls with locks and one uvicorn
  worker by design (shared single copy). Scale by running more replicas behind a
  load balancer, not more workers per container — in Case B they can all share
  one pre-synced weights volume.
- **Security.** Runs as a non-root user (`raginfer`, uid 10001). For Case B on
  AWS, prefer an IAM role over baking `AWS_*` keys into the image or env.
- **`tini` as PID 1** ensures a clean `SIGTERM` → uvicorn shutdown.
