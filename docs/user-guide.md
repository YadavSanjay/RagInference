# RagInference — User Guide

This guide is for **clients of the RagInference service**: anyone (ARag, MacroRAG,
a script, a notebook) that wants to embed or rerank text by calling the shared
inference server over HTTP, instead of loading the heavy models into their own
process.

If you are running or operating the server itself, the [README](../README.md) is
the place to start; this guide focuses on **connecting to** a running service and
**troubleshooting** the connection.

---

## 1. What this service does (and why you'd use it)

RagInference loads two large RAG models **once** in a single process and serves
them over HTTP:

| Capability | Default model | Endpoint |
|------------|---------------|----------|
| Embedding (dense + sparse/lexical) | `BAAI/bge-m3` | `POST /embed` |
| Reranking (relevance scoring) | `jinaai/jina-reranker-v3` | `POST /rerank` |
| Health / readiness | — | `GET /healthz` |

**Why call the service instead of loading the models yourself?** Each copy of
BGE-M3 (~2.5 GB) plus jina-reranker-v3 (~0.9 GB) is multi-GB of RAM. When several
clients each loaded their own copy, concurrent cold loads spiked memory into swap
and inflated a ~92 s rerank to >300 s. One shared service removes the duplicate
copies and serializes calls so clients stop contending for RAM.

You get the same vectors and scores you'd get locally — the embedder returns
JSON-able dense lists and sparse `{indices, values}` that your client rebuilds
into its own structures.

---

## 2. Connecting to the service

### The one setting that matters: `RAGINFER_URL`

Point your client at the running server by setting **`RAGINFER_URL`** to its base
URL (scheme + host + port, no trailing path):

```bash
export RAGINFER_URL=http://127.0.0.1:8799     # local, same machine
# or a shared host:
export RAGINFER_URL=http://rag-host.internal:8799
```

The default port is **8799**. The default host the server binds to is
`127.0.0.1` (loopback only) — if you need to reach it from another machine, the
operator must start it with `RAGINFER_HOST=0.0.0.0` (see
[§7 Operator notes](#7-operator-notes-for-people-running-the-server)).

### How known clients use it

- **XbrlRag** and **MacroRAG** each ship `RemoteReranker`
  (`src/retriever/remote_reranker.py`) and `RemoteEmbedder`
  (`src/embeddings/remote_embedder.py`). Both use them **automatically when
  `RAGINFER_URL` is set**, and load local models only when it is **not** set. So
  connecting is usually just: set `RAGINFER_URL`, restart.

### Connecting from your own code

There is no SDK required — it's plain HTTP + JSON. Minimal Python client:

```python
import os, httpx

BASE = os.environ.get("RAGINFER_URL", "http://127.0.0.1:8799")

def embed(texts, dense=True, sparse=True):
    r = httpx.post(f"{BASE}/embed",
                   json={"texts": texts, "dense": dense, "sparse": sparse},
                   timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]

def rerank(query, documents):
    r = httpx.post(f"{BASE}/rerank",
                   json={"query": query, "documents": documents},
                   timeout=300)
    r.raise_for_status()
    return r.json()["scores"]
```

Use a **generous timeout** (rerank on a large candidate set can take tens of
seconds to minutes on CPU) and reuse a single client/connection pool if you make
many calls.

### Verify your connection in one command

```bash
curl -s "$RAGINFER_URL/healthz"
```

A healthy server replies:

```json
{ "status": "ok", "embedder": "BAAI/bge-m3", "reranker": "jinaai/jina-reranker-v3",
  "embedder_loaded": true, "reranker_loaded": true }
```

If `*_loaded` is `false`, the model hasn't been built yet (warm-up still running
or disabled) — it will build lazily on the first real request. See
[§5 Troubleshooting](#5-troubleshooting).

---

## 3. API reference

All endpoints accept and return JSON. Results are **aligned to input order** in
every case.

### `POST /embed`

Request:

```json
{ "texts": ["export control restrictions"], "dense": true, "sparse": true }
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `texts` | `string[]` | required | Texts to embed. |
| `dense` | `bool` | `true` | Return dense vectors. |
| `sparse` | `bool` | `true` | Return sparse (lexical) vectors. |
| `model` | `string?` | server default | Override embedder (`"bge"`). |

Response:

```json
{ "embeddings": [
    { "dense": [0.01, "..."],
      "sparse": { "indices": [4, 91], "values": [0.7, 0.3] } } ],
  "model": "BAAI/bge-m3", "count": 1 }
```

- `dense` is `null` when `dense: false`; `sparse` is `null` when `sparse: false`.
- Sparse `indices` are BGE-M3 token-ids, `values` their weights (aligned).
- Empty `texts` → `{ "embeddings": [], "count": 0 }` (no error).

### `POST /rerank`

Request:

```json
{ "query": "export control restrictions", "documents": ["text a", "text b"] }
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | `string` | required | Query to score documents against. |
| `documents` | `string[]` | required | Candidate texts, scored in place. |
| `model` | `string?` | server default | Override reranker (`"jina"` or `"bge"`). |

Response:

```json
{ "scores": [0.91, 0.12], "model": "jinaai/jina-reranker-v3", "count": 2 }
```

- One score per input document, aligned to order. Higher = more relevant. You sort
  on the client side.
- Empty `documents` → `{ "scores": [], "count": 0 }` (no error).

### `GET /healthz`

Returns `status`, the `embedder` / `reranker` model names, and
`embedder_loaded` / `reranker_loaded` booleans. Always returns `200` while the
process is up, even before models load — this is a **liveness** check. Read the
`*_loaded` flags (or use `/readyz`) to know readiness.

### `GET /readyz`

Same body as `/healthz`, but a **readiness** check: returns **`503` until both
default models are loaded**, then `200`. This is what a load balancer or AWS ECS
target group should poll, so traffic only reaches a task once it is warm (weights
synced + warm-up done). As a client you normally just use `/healthz` + the
`*_loaded` flags; `/readyz` mainly matters to operators wiring health checks.

### Selecting a non-default model

The optional `model` field lets you request an alternate model **without
restarting the server** — the first such call builds and caches it (one-time load
cost), and it's reused after that:

- Embedder: `"bge"` / `"bge-m3"` (only BGE is implemented today).
- Reranker: `"jina"` / `"jina-reranker-v3"` / `"v3"`, or the lighter
  `"bge"` / `"bge-reranker-v2-m3"` / `"m3"`.

An unknown model name returns **HTTP 422** with a message like
`unknown reranker 'nope' (expected 'jina' or 'bge')`.

---

## 4. What to expect (performance & behavior)

- **First call is slow, later calls are fast.** The operator's server normally
  *warms up* both default models at startup, so the first real request is already
  on the warm path. If warm-up is off, the very first `/embed` and `/rerank` each
  pay a one-time model-load cost (several seconds to minutes) while the model
  builds under a lock.
- **Calls are CPU forward passes and effectively serialized.** This is by design —
  it's what stops clients from contending for RAM. Don't expect high concurrency;
  expect predictable, non-swapping latency. Send reasonably sized batches rather
  than thousands of tiny concurrent calls.
- **Long documents are bounded automatically.** The reranker truncates each
  document (per-doc cap ~512 tokens) and blocks the forward pass (~4096-token
  budget) so very long candidates (e.g. SEC tables) don't blow up memory. You may
  see slightly different scores than an unbounded local run — this is the
  intentional memory-safety trade-off, and it's top-k neutral in practice.
- **The embedder caps its batch size** (12) and `max_length` (8192) for CPU
  memory safety. Very large `texts` arrays are processed in internal batches; you
  don't need to pre-batch, but extremely large requests will take proportionally
  longer.

---

## 5. Troubleshooting

Work from the top — most issues are "URL wrong" or "server not up / not warm yet."

### Connection refused / timeout on connect

```
httpx.ConnectError: [Errno 111] Connection refused
```

- **Is the server running?** Ask the operator, or check the host. The process is
  `python -m src.service.server`.
- **Right host/port?** Default is `127.0.0.1:8799`. Confirm `RAGINFER_URL` matches
  what the server actually bound to.
- **Reaching it from another machine?** A server started with the default
  `RAGINFER_HOST=127.0.0.1` is **loopback-only** and is unreachable from other
  hosts even though it's "running." It must be started with
  `RAGINFER_HOST=0.0.0.0`, and any firewall must allow the port.
- **Quick probe:** `curl -v "$RAGINFER_URL/healthz"`.

### `RAGINFER_URL` not taking effect / client still loads local models

- Confirm the variable is **exported in the same environment** the client runs in
  (a different shell, service manager, or container won't see your `export`).
- For XbrlRag and MacroRAG: if `RAGINFER_URL` is unset, the client silently **falls
  back to local models** — that's the expected behavior, not a bug. Set the variable
  and restart to switch to the remote service.

### `/healthz` works but `embedder_loaded` / `reranker_loaded` is `false`

- The model is **built lazily** and hasn't been used yet, or startup **warm-up was
  disabled** (`RAGINFER_WARMUP=0`). It will build on the first real request.
- If warm-up was on but a load **failed**, health still reports `false` (load
  failures are non-fatal by design). Check the server log for a warm-up exception
  (see [§6 Logs](#6-logs)).

### First request hangs for a long time, then succeeds

- That's the **cold model load** (no warm-up, or first use of a non-default
  `model`). Use a long client timeout (≥120 s for embed, ≥300 s for a large
  rerank) and let it finish — subsequent calls are warm.

### `HTTP 422 Unprocessable Entity`

Two common causes:

- **Unknown `model`** — e.g. `{"detail": "unknown reranker 'nope' ..."}`. Use a
  supported name (see [§3](#selecting-a-non-default-model)).
- **Malformed body** — missing required field (`texts`, or `query`/`documents`),
  or wrong types. The `detail` field names the offending field.

### Rerank is extremely slow or the server process dies on huge inputs

- Huge candidate sets or very long documents are the usual cause. The server
  bounds per-doc and per-block tokens to avoid OOM, but you can still help by
  **pre-filtering candidates** (e.g. send your top ~100 by a cheap retriever, not
  thousands) before reranking.
- If the process actually crashed (you get connection-refused **after** it was
  working), it's likely a memory event — tell the operator; the server log and
  system memory pressure are the things to check.

### Scores or vectors look different from a previous local run

- Expected and benign for rerank: the service applies the bounded-blocking
  memory fix (per-doc truncation + token-budget blocking). Ranking order for the
  top results is preserved; absolute scores can shift slightly.

### How to report a problem

Include: the exact `RAGINFER_URL`, the output of `curl "$RAGINFER_URL/healthz"`,
the request body that failed, the HTTP status + `detail` you got back, and the
relevant server log lines (next section).

---

## 6. Logs

The server logs to **stderr** and to a rotating file at
`logs/raginference.log` (10 MB × 3 backups), relative to where the server was
started. Useful lines:

- Startup / warm-up: `RagInference warm-up: ... ready` (or `... failed (non-fatal)`).
- Per request: `embed: model=... texts=N ... took=Xs` and
  `rerank: model=... docs=N took=Xs query=...`.

Set `LOG_LEVEL=DEBUG` (server-side) for more detail. Note logs go to **stderr,
never stdout**, so the service can be embedded behind a stdio transport without
corrupting it.

---

## 7. Operator notes (for people running the server)

These are the knobs that affect what clients can reach. Full details in the
[README](../README.md).

Start the service:

```bash
conda run -n py312 python -m src.service.server
# with overrides:
RAGINFER_HOST=0.0.0.0 RAGINFER_PORT=8799 RERANKER=jina EMBEDDER=bge \
    conda run -n py312 python -m src.service.server
```

Environment variables:

| Variable | Default | Effect |
|----------|---------|--------|
| `DEPLOY_MODE` | `local` | `aws` derives cloud defaults: bind `0.0.0.0`, JSON logs to stdout (CloudWatch), no log file, S3 storage backend. Any explicit var below still overrides. |
| `RAGINFER_HOST` | `127.0.0.1` | Bind address. **Use `0.0.0.0` to allow remote clients** (auto in `aws` mode). |
| `RAGINFER_PORT` | `8799` | Bind port (clients' `RAGINFER_URL` must match). |
| `RERANKER` | `jina` | Default reranker (`jina` \| `bge`). |
| `EMBEDDER` | `bge` | Default embedder (`bge`). |
| `RAGINFER_WARMUP` | `1` | `1` loads + warms both models at startup; `0`/`false` defers to first request. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

Running on AWS (ECS Fargate, VPC-internal) is covered in
[`../docs/aws-deployment-plan.md`](aws-deployment-plan.md) and [`../infra/aws/`](../infra/aws/);
in `aws` mode logs go to CloudWatch (stdout) rather than `logs/raginference.log`.

Operator checklist when a client can't connect:

1. Process up? (`python -m src.service.server` running)
2. Bound where clients expect? (`RAGINFER_HOST`/`RAGINFER_PORT` vs their `RAGINFER_URL`)
3. Remote clients need `RAGINFER_HOST=0.0.0.0` + open firewall port.
4. `curl http://127.0.0.1:8799/healthz` on the server box confirms the app itself
   is healthy independent of networking.

---

## 8. Quick reference

```bash
# Point a client at the service
export RAGINFER_URL=http://127.0.0.1:8799

# Health check
curl -s "$RAGINFER_URL/healthz"

# Embed
curl -s "$RAGINFER_URL/embed" -H 'content-type: application/json' \
  -d '{"texts":["export control restrictions"],"dense":true,"sparse":true}'

# Rerank
curl -s "$RAGINFER_URL/rerank" -H 'content-type: application/json' \
  -d '{"query":"export control restrictions","documents":["text a","text b"]}'
```

| Symptom | First thing to check |
|---------|----------------------|
| Connection refused | Server running? `RAGINFER_URL` host/port correct? |
| Works locally, not remotely | Server started with `RAGINFER_HOST=0.0.0.0`? Firewall? |
| Client still loads local models | `RAGINFER_URL` exported in the client's env? |
| `*_loaded: false` | Warm-up off or load failed — check `logs/raginference.log`. |
| HTTP 422 | Unknown `model`, or malformed/missing request fields. |
| First call very slow | Cold model load — increase client timeout, then it's warm. |
