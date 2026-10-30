<h1 align="center">RagInference</h1>

<p align="center">
  <b>One copy of your RAG models, shared by every client.</b><br>
  A small HTTP service that loads a <b>BGE-M3 embedder and a reranker once</b> and serves
  <code>/embed</code> and <code>/rerank</code> to every RAG application on the machine or network,
  on plain CPUs, with the memory limits that keep long documents from crashing the process.
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="CPU only" src="https://img.shields.io/badge/inference-CPU%20only-0f766e">
  <img alt="Docker amd64 and arm64" src="https://img.shields.io/badge/Docker-amd64%20%2B%20arm64-2496ED?logo=docker&logoColor=white">
  <img alt="AWS Fargate ready" src="https://img.shields.io/badge/AWS-Fargate%20ready-b45309">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

> **The code is MIT. The default reranker, `jina-reranker-v3`, is CC BY-NC 4.0 (non-commercial).**
> For commercial use, set `RERANKER=bge`. See [License and support](#license-and-support).

---

## Why RagInference

A RAG application usually loads its embedder and reranker into its own process. That is fine for
one application. It stops being fine when you run several, or when one application starts several
subprocesses, because each one holds its own multi-GB copy of the same weights.

This project came out of exactly that failure. Two MCP servers each loaded BGE-M3 (~2.5 GB) and
jina-reranker-v3 (~0.9 GB). Concurrent cold starts pushed RAM into swap, and a ~92 s rerank took
more than 300 s.

Moving the models behind one service fixed it:

- **One copy, however many clients.** Models load once in a single process. In the original
  migration, client memory fell from **4.02 GB to 0.54 GB**, and remote embeddings and rerank
  scores matched the local ones exactly.
- **Safe on long documents.** Library defaults that work on short text can kill a CPU process on
  long chunks such as SEC filing tables, so the wrappers set explicit limits
  ([details](#the-limits-that-keep-it-alive)).
- **Calls don't fight over the CPU.** Each model runs one forward pass at a time, because a single
  pass already uses every core. Embed and rerank queue separately, so a sub-second query embedding
  never waits behind a minutes-long rerank.
- **Traffic only lands on a warm process.** Models load and warm at startup, and `/readyz` stays
  503 until both are ready. Containers fail fast when their weights are missing, instead of coming
  up healthy and useless.
- **The same code runs locally and in the cloud.** `DEPLOY_MODE` switches bind address, logging
  and storage defaults. Multi-arch images run on x86 or on Graviton.

## What it serves

| Endpoint | Purpose | Returns |
|---|---|---|
| `POST /embed` | Dense and sparse vectors from BGE-M3 | One `{dense, sparse}` per text, in input order |
| `POST /rerank` | Relevance scores for candidate documents | One score per document, in input order |
| `GET /healthz` | Liveness | Always 200 while the process is up, with each model's load state |
| `GET /readyz` | Readiness | 503 until both default models are loaded, then 200 |

```bash
curl -s localhost:8799/embed -H 'content-type: application/json' \
  -d '{"texts": ["export control restrictions"], "dense": true, "sparse": true}'
```

```json
{
  "embeddings": [
    { "dense": [0.01, "..."], "sparse": { "indices": [4, 91], "values": [0.7, 0.3] } }
  ],
  "model": "BAAI/bge-m3",
  "count": 1
}
```

```bash
curl -s localhost:8799/rerank -H 'content-type: application/json' \
  -d '{"query": "export control restrictions", "documents": ["text a", "text b"]}'
```

```json
{ "scores": [0.91, 0.12], "model": "jinaai/jina-reranker-v3", "count": 2 }
```

Scores stay aligned to the order you sent, so a client can sort them without tracking ids through
the service. There is no SDK: it is plain HTTP and JSON. The
[user guide](docs/user-guide.md) has the full API, a Python client and troubleshooting.

---

## Architecture at a glance

The request path is deliberately thin, and the weight is in the model wrappers:

<p align="center">
  <a href="docs/images/architecture.svg">
    <img src="docs/images/architecture.svg" width="100%"
         alt="RagInference architecture: RAG clients call one service process over HTTP using RAGINFER_URL. A FastAPI layer enforces request bounds and exposes /embed, /rerank, /healthz and /readyz. Embed and rerank each take their own inference lock and run on one shared model: BGE-M3 for dense and sparse vectors, and jina-reranker-v3 or bge-reranker-v2-m3 for scores. Models are built once under a lock, warmed at startup and cached. Weights come from the Hugging Face cache at HF_HOME, filled by a fat image, the Hub or an S3 sync. DEPLOY_MODE switches local and AWS defaults.">
  </a>
</p>

- **Uniform contracts.** Every embedder implements `embed(texts, dense, sparse)`, and every
  reranker implements `rerank(query, documents)` returning scores in input order. FastAPI only
  marshals JSON to and from them.
- **Models are chosen by name.** Adding a model means a new wrapper subclass and a branch in its
  factory. Nothing else changes.
- **Scale with replicas, not workers.** Every uvicorn worker would load another full copy of the
  weights, which is the problem this service exists to remove.
- **Weights arrive through `HF_HOME`.** The service always reads the Hugging Face cache, so baking
  weights into an image, downloading from the Hub and syncing from S3 are interchangeable.

### The limits that keep it alive

These are CPU forward passes, and the limits are what make them safe. Each one comes from a real
failure:

| Model | Default behaviour | What goes wrong | What this service does |
|---|---|---|---|
| BGE-M3 | `batch_size=256` | One huge forward pass segfaults on CPU | Caps `batch_size=12`, `max_length=8192` |
| jina-reranker-v3 | Packs every candidate into one forward pass | It is listwise, so memory grows with the square of total tokens; SEC-style chunks went past 6 GB and crashed | Forces a 4,096-token block budget and truncates documents to 512 tokens |
| jina-reranker-v3 | bf16 weights | Emulated on most x86 CPUs: a 24-document rerank took 169 s against 92 s in fp32, with negligible score change | Loads fp32 by default (`RERANKER_DTYPE`) |
| Any | Unbounded request bodies | One bad payload ties up the CPU or memory | Returns 413 over `RAGINFER_MAX_ITEMS` or `RAGINFER_MAX_TEXT_CHARS` |

Timings are from one x86 desktop CPU and will vary with your hardware.

---

## Quick start

You need Python 3.12, 16 GB of RAM (8 GB is tight with both models warm) and about 4 GB of disk for
the weights. No GPU and no API key are required.

```bash
git clone <this-repo> && cd RagInference
python -m venv .venv && source .venv/bin/activate

# CPU-only torch first: on x86 the default PyPI wheel pulls in multi-GB CUDA libraries
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

python -m src.service.server        # binds 127.0.0.1:8799
```

The first start downloads the weights from Hugging Face, then warms both models. Wait for readiness:

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:8799/readyz    # 503 while loading, then 200
```

For commercial use, switch reranker with `RERANKER=bge`.

### With Docker

```bash
# Slim image (~2.4 GB): weights download on first start, into a volume
docker build -f infra/Dockerfile -t raginfer:slim .
docker run --rm -p 8799:8799 -e MODEL_SOURCE=hf -v raginfer-models:/models raginfer:slim
```

`infra/README.md` covers the fat image, which has the weights baked in, and the S3-backed route.

### Connecting a client

Point the client at the service with `RAGINFER_URL=http://<host>:8799`. Any HTTP client works, and
the [user guide](docs/user-guide.md#connecting-from-your-own-code) has a minimal Python example.

RagInference was built for two companion projects, which are separate from this repository:
**XbrlRag** (SEC filings) and **MacroRAG** (macro research). Both ship `RemoteEmbedder` and
`RemoteReranker` shims that use the service when `RAGINFER_URL` is set and load local models when
it is not.

## Configuration

Everything is set through environment variables. Unset means the default.

| Variable | Default | Meaning |
|---|---|---|
| `DEPLOY_MODE` | `local` | `aws` sets bind `0.0.0.0`, JSON logs to stdout and the S3 storage backend; explicit variables still win |
| `RAGINFER_HOST` / `RAGINFER_PORT` | `127.0.0.1` / `8799` | Bind address |
| `RERANKER` | `jina` | `jina` or `bge` |
| `EMBEDDER` | `bge` | `bge` |
| `RERANKER_DTYPE` | `float32` | `float32`, `bfloat16` or `auto` |
| `RAGINFER_WARMUP` | `1` | Load and warm both models at startup |
| `RAGINFER_MAX_ITEMS` | `4096` | Most texts or documents in one request |
| `RAGINFER_MAX_TEXT_CHARS` | `200000` | Longest single text in a request |
| `MODEL_SOURCE` | `baked` | Weight delivery in containers: `baked`, `hf` or `s3` |

## Deployment

The service has **no authentication and no TLS**. Keep it on loopback or a private network, and
see [SECURITY.md](SECURITY.md).

The AWS target is ECS Fargate on ARM64/Graviton, inside a VPC, with optional scheduled
scale-to-zero for batch-tolerant loads. CPU forward passes are enough at this load, so a GPU would
mostly sit idle.

| Guide | Covers |
|---|---|
| [Build and push runbook](docs/runbook/build-and-push-ecr.md) | Copy-paste local amd64 and AWS arm64 builds, ECR push, arch verification |
| [Deployment guide](docs/deployment-guide.md) | ECR pull setup, EC2 and Fargate runs, VPC endpoints, troubleshooting |
| [AWS deployment plan](docs/aws-deployment-plan.md) | Cost analysis, CPU against GPU, options considered, what to provision |
| [Containers](infra/README.md) | Fat and slim images, multi-arch builds, MinIO demo |
| [Fargate contract](infra/aws/README.md) | Task definition template, environment, scheduled scaling |

## Explore further

```text
src/
  service/      FastAPI app, wire schemas, uvicorn entrypoint with warm-up
  embedders/    BGE-M3 wrapper and factory
  rerankers/    jina-reranker-v3 and bge-reranker-v2-m3 wrappers and factory
  config/       Typed Settings driven by DEPLOY_MODE
  storage/      Local and S3 adapter for future blob IO (weights do not use it)
infra/          Dockerfile, compose file, entrypoint, AWS templates
docs/           User guide, deployment guide, runbook, AWS plan
test/           Service, settings and repo checks, with stub models and no weights
```

### Development

Tests inject stub models, so they need no weights, GPU or network, and they do not install torch:

```bash
pip install -r requirements-dev.txt
python -m pytest test/ -q
ruff check .
mypy
python scripts/check_public_safe.py    # no machine paths, account ids or keys in tracked files
```

CI runs the same four checks on every pull request.

## License and support

MIT — see [`LICENSE`](LICENSE).

The models are not part of this repository. They are downloaded from Hugging Face and carry their
own licenses, so check them before you deploy:

| Model | Used for | License |
|---|---|---|
| [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | Dense and sparse embeddings (default) | MIT |
| [`jinaai/jina-reranker-v3`](https://huggingface.co/jinaai/jina-reranker-v3) | Reranking (default) | **CC BY-NC 4.0 — non-commercial use only** |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Reranking (`RERANKER=bge`) | Apache-2.0 |

A fat image built with `BAKE_MODELS="bge jina"` contains the Jina weights, so its license terms
apply to anyone you share that image with. The Jina model also runs Python from its model
repository (`trust_remote_code`), so this project pins it to one commit: an upstream change can't
alter what runs here until the pin moves.

This is a personal project, shared as-is: no support, no stability guarantees, and no promise to
review issues or pull requests. To report a security problem, see [SECURITY.md](SECURITY.md).
