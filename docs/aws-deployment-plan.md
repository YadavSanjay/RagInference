# RagInference — AWS Deployment Analysis & Plan

**Status:** proposal · **Date:** 2026-07-16 · **Audience:** RagInference maintainers + the separate infra/provisioning team

This document analyzes what it takes to run RagInference in **two modes from one
codebase** — unchanged local Windows dev, and AWS (VPC-internal, consumed by
sibling services). It covers the data-interaction surface, a `DEPLOY_MODE`
config seam, AWS hosting options with a **cost focus on GPU vs CPU**, the fat-vs-slim
image tradeoff on AWS, and a concrete phased change list.

> **Decisions baked in (from product owner):**
> - **Latency is batch-tolerant** (tens of seconds acceptable; cold starts OK) → *scale-to-zero is viable*.
> - **AWS resources are provisioned by a separate infra project** → this doc *names the services to provision* rather than shipping IaC.
> - **Chosen target: ECS Fargate (CPU) with optional scheduled scale-to-zero off-hours.**

> **Implementation status (landed in this repo — the code/packaging parts that need no live AWS):**
> - ✅ `DEPLOY_MODE` typed settings — [`src/config/settings.py`](../src/config/settings.py) (local defaults unchanged; aws flips host/logging/storage).
> - ✅ Cloud logging — [`src/logging_config.py`](../src/logging_config.py) emits JSON to stdout in aws mode, file+stderr locally.
> - ✅ `GET /readyz` readiness probe (503 until warm) — [`src/service/app.py`](../src/service/app.py).
> - ✅ Local↔S3 storage seam — [`src/storage/__init__.py`](../src/storage/__init__.py) (forward-looking; not in request path yet).
> - ✅ `s5cmd` baked into the image (arch-aware) for the slim/S3 sync, **plus arch-aware CPU torch** (x86 → cpu index; **arm64/Graviton → PyPI aarch64 wheel**) — [`infra/Dockerfile`](../infra/Dockerfile). ARM is the committed Fargate target (~20% cheaper, par performance).
> - ✅ Fargate contract + scheduled scale-to-zero reference + task-def template — [`infra/aws/`](../infra/aws/).
> - ✅ Tests — [`test/test_settings.py`](../test/test_settings.py) + `/readyz` cases in `test/test_service.py` (17 pass, no weights loaded).
> - ✅ **Image built & run-verified** (slim, amd64, ~2.4 GB): non-root, `/healthz` 200, `/readyz` 503→gated, JSON logs to stdout under `DEPLOY_MODE=aws`. Two defects found & fixed — container `HEALTHCHECK` now uses `/readyz` (it reported *healthy* with zero models loaded), and the entrypoint **fails fast** when `MODEL_SOURCE=baked` but `HF_HOME` is empty.
> - ✅ **Packaging/deployment runbook** — [`docs/deployment-guide.md`](deployment-guide.md): local build, ECR repo/login/tag/push, **ECR pull setup** (IAM instance profile, credential helper, private-subnet VPC endpoints), EC2 run + systemd, env reference, troubleshooting.
> - ⬜ **Needs live AWS (separate infra project):** ECR multi-arch push, ECS/ALB/S3/IAM provisioning, the scheduled-action registration, CloudWatch dashboards.

---

## 1. TL;DR — recommendation

1. **Do not use GPU.** The models are moderate (~4 GB total, ~0.6 B params each), the
   load is mild, and CPU forward passes already work. A 24/7 GPU costs **$384–538/mo**
   to sit ~99% idle. GPU is only justified at sustained utilization we don't have.
2. **Keep the FastAPI HTTP service exactly as-is** and run it as a **container**, so the
   client contract (`RAGINFER_URL` → `POST /embed` `/rerank`) is identical local and cloud.
3. **Primary target: ECS Fargate (CPU, ARM/Graviton), 1 task, VPC-internal**, behind
   an internal NLB/ALB or Cloud Map. ~**$85–140/mo** always-on; optionally scheduled
   **scale-to-zero** off-hours since load is batch-tolerant.
4. **Cheapest scale-to-zero alternative: Lambda container** (FastAPI via Mangum) — true
   pay-per-request, but pays a cold-start model load and sits near the 10 GB memory
   ceiling; viable given batch tolerance, documented with caveats below.
5. **Weights: slim image + S3** on AWS (native fit, cheap ECR, easy weight churn),
   **fat image** stays the default for local/offline/CI. Both already exist in `infra/`.
6. **Code changes are small** — the service is stateless HTTP with a tiny data surface.
   Add a typed `Settings`/`DEPLOY_MODE` layer, cloud-friendly logging, and a `/readyz`
   readiness probe. No change to the model wrappers or the request path.

---

## 2. Current-state analysis — every data interaction

The reason migration is low-risk: **this service is stateless request/response.** It
has no database, no queue, no user-data persistence, and does not fetch documents —
clients send text inline in the HTTP body. The entire external surface:

| # | Interaction | Direction | Today (local) | What changes on AWS |
|---|---|---|---|---|
| 1 | Client `/embed` `/rerank` `/healthz` | Inbound HTTP | `127.0.0.1:8799` loopback | VPC-internal, `0.0.0.0`, SG-restricted to client SGs |
| 2 | **Model weights** (the only real "data source") | Read at load | HF cache dir `HF_HOME` (local/conda) | Baked in image, **or S3 → `HF_HOME`** (already supported: `MODEL_SOURCE=baked\|hf\|s3`) |
| 3 | Logs | Write | stderr **+** rotating file `logs/raginference.log` | **stdout → CloudWatch**; drop the file handler in cloud |
| 4 | HuggingFace Hub | Outbound (only if `MODEL_SOURCE=hf`) | Internet at first load | Avoid in VPC; prefer S3 (VPC gateway endpoint) or baked |
| 5 | Credentials | — | none | IAM **task role** for S3 read (no keys in image/env) |

**Implication:** the "local FS vs S3 via a deploy-mode parameter" the request calls for
is, in practice, exactly interactions **#2 (weights)** and **#3 (logs)**. Weights already
have a local↔S3 seam at the shell/entrypoint level; this plan lifts that decision into a
**typed Python config** so both concerns are driven by one `DEPLOY_MODE` switch, and adds a
forward-looking storage adapter for any future artifact IO (none exists in the request path today).

---

## 3. The `DEPLOY_MODE` config seam (local ↔ aws)

### 3.1 Design

Introduce one typed settings object (`pydantic-settings`, already have pydantic v2) that
reads the **same env vars we use today** plus a top-level `DEPLOY_MODE`, and derives sane
per-mode defaults. Nothing about local dev changes if the var is unset (`DEPLOY_MODE=local`
is the default).

```python
# src/config/settings.py  (new)
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    deploy_mode: Literal["local", "aws"] = "local"

    # network
    host: str = "127.0.0.1"        # image/aws override -> 0.0.0.0
    port: int = 8799

    # models
    embedder: str = "bge"
    reranker: str = "jina"
    warmup: bool = True

    # weights delivery (mirrors entrypoint.sh)
    model_source: Literal["baked", "hf", "s3"] = "baked"
    model_s3_uri: str | None = None
    hf_home: str | None = None

    # storage abstraction (future artifact IO; weights use model_source above)
    #   local -> LocalStorage(root=...)   aws -> S3Storage(bucket=...)
    artifact_backend: Literal["local", "s3"] | None = None  # default derived from deploy_mode

    # logging
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"   # aws default -> json
    log_to_file: bool = True                        # aws default -> False (stdout only)

    # aws
    aws_region: str | None = None

    class Config:
        env_prefix = "RAGINFER_"   # keeps RAGINFER_HOST/PORT working; add aliases for legacy names

    def resolved(self) -> "Settings":
        """Apply per-mode defaults without overriding anything set explicitly."""
        if self.deploy_mode == "aws":
            self.host = "0.0.0.0" if "RAGINFER_HOST" not in _env else self.host
            self.log_format = "json"  if "RAGINFER_LOG_FORMAT" not in _env else self.log_format
            self.log_to_file = False  if "RAGINFER_LOG_TO_FILE" not in _env else self.log_to_file
            self.artifact_backend = self.artifact_backend or "s3"
        else:
            self.artifact_backend = self.artifact_backend or "local"
        return self
```

**Backward-compat rule:** `DEPLOY_MODE` only supplies *defaults*. Any legacy env var
(`RAGINFER_HOST`, `RERANKER`, `EMBEDDER`, `MODEL_SOURCE`, …) still wins when set, so today's
local and container invocations behave identically. We keep reading `RERANKER`/`EMBEDDER`/
`MODEL_SOURCE` as aliases so nothing in `infra/` or the conda workflow breaks.

### 3.2 Storage adapter (forward-looking seam)

The request path needs **no** blob IO today. But since the ask explicitly names "local FS
vs S3," add a thin, optional adapter so *future* features (cached embeddings, corpus
loading, warm dumps) have one seam instead of scattered `open()`/`boto3` calls:

```python
# src/storage/__init__.py  (new, optional to use)
class Storage(Protocol):
    def read_bytes(self, path: str) -> bytes: ...
    def write_bytes(self, path: str, data: bytes) -> None: ...
    def exists(self, path: str) -> bool: ...

def build_storage(settings) -> Storage:      # local -> LocalStorage, aws -> S3Storage(boto3)
    ...
```

Weights intentionally stay on the **`MODEL_SOURCE` path** (proven, works at the HF-cache
layer). Don't reroute weight loading through this adapter — `from_pretrained` expects a real
directory, and the entrypoint sync already delivers that.

### 3.3 Logging change (the one real cloud gap)

`configs/logging.yaml` hard-codes a `RotatingFileHandler` at `logs/raginference.log` and a
stderr handler. In a container/Fargate/Lambda world you want **structured logs to stdout**
so CloudWatch captures them; a file handler on an ephemeral FS is dead weight and can even
fail if the dir isn't writable.

- `DEPLOY_MODE=aws` → JSON formatter, **stdout**, no file handler.
- `DEPLOY_MODE=local` → unchanged (stderr + rotating file).
- Note the existing "stderr never stdout, for stdio JSON-RPC" constraint is a *local MCP*
  concern; in AWS there is no stdio transport, so stdout is correct there. Gate on mode.

---

## 4. AWS hosting options — the core evaluation

All options keep the **same container image**. They differ in *who runs it* and *what
idle costs.* Prices are **approximate us-east-1 on-demand, 730 hr/mo — verify at deploy
time**; Savings Plans/Reserved cut always-on ~30–50%.

### 4.1 Option matrix

| Option | Idle cost | Client change | Cold start | Ops burden | Fit for us |
|---|---|---|---|---|---|
| **A. ECS Fargate CPU, 1 task always-on** | ~$85–170/mo | **none** | none | low | ✅ **Recommended default** |
| A′. Fargate CPU + scheduled scale-to-zero | ~$40–90/mo | none | task boot + warm (~1–3 min) | low-med | ✅ good, batch-tolerant |
| **B. Lambda container (FastAPI+Mangum)** | ~$0 | none (HTTPS URL) | **model load per cold start (~30–90 s)** | low | ✅ cheapest; caveats §4.4 |
| C. SageMaker Async Inference, min=0 (CPU) | ~$0 idle | **large** (S3-async / InvokeEndpoint) | instance boot (min) | med | ⚠️ managed scale-to-zero, but rewrites clients |
| D. SageMaker Real-time, scale-to-zero (CPU) | ~$0 idle | **medium** (SigV4 InvokeEndpoint) | scale-from-zero (min) | med | ⚠️ managed, non-HTTP contract |
| E. EC2 CPU always-on (self-managed) | ~$120–260/mo | none | none | high | ➖ more ops than Fargate |
| F. **Any GPU always-on** (g4dn/g5, EC2 or SM) | **$384–738/mo** | varies | none | med-high | ❌ wasteful at mild load |

### 4.2 CPU sizing note

Both models loaded ≈ **4 GB weights + torch/activation overhead → plan 8–16 GB RAM.**
8 GB is tight with both warm; **16 GB is the safe default.** vCPU drives rerank latency
(jina is listwise, CPU-heavy) — 4 vCPU is a reasonable start, more only if rerank latency
bites. Prefer **ARM/Graviton** (Fargate ARM, `c7g`/`c8g`) for ~20% better price/perf; the
CPU-only torch wheel supports ARM.

### 4.3 GPU cost reality check (the headline question)

| Instance | GPU | ~$/hr | ~$/mo 24/7 | Note |
|---|---|---|---|---|
| Fargate 4 vCPU/16 GB | — | ~$0.23 | **~$170** | CPU, our baseline |
| Fargate 4 vCPU/16 GB **ARM** | — | ~$0.19 | **~$136** | ~20% cheaper |
| g4dn.xlarge | T4 16 GB | ~$0.526 | **~$384** | cheapest real GPU |
| g5.xlarge | A10G 24 GB | ~$1.006 | **~$734** | overkill here |
| SageMaker RT ml.g4dn.xlarge | T4 | ~$0.736 | **~$538** | +~40% managed premium |
| inf2.xlarge | Inferentia2 | ~$0.76 | ~$555 | needs Neuron compile; not worth it at this scale |

**Utilization math:** at "mild" load — say a few hundred calls/day, a few seconds
each — real GPU compute is on the order of **minutes per day**. A 24/7 GPU therefore runs
at **~1–3% utilization**, i.e. you pay $384+ for ~$5 of useful compute. The only way a GPU
pays off is **scale-to-zero** (SageMaker Async/RT), and even then a CPU box handles this
load without the client-contract disruption. **Verdict: CPU. Revisit GPU only if sustained
utilization climbs past ~30–40%,** and if so, do it via SageMaker Async scale-to-zero, not
an always-on instance.

### 4.4 Lambda caveats (option B)

Attractive (true pay-per-use, keeps an HTTPS URL via a Function URL/ALB + Mangum), but:
- **Cold start** reloads ~4 GB of weights per scale-up (~30–90 s). Batch tolerance makes
  this OK; interactive would not.
- **10 GB memory ceiling.** Both models + torch runtime sit close to it. If it's tight,
  **split into two functions** (embed / rerank) — each fits comfortably, and Lambda's
  per-function model suits that cleanly. This is the one place the "one shared process"
  premise doesn't transfer — but in cloud the premise was about *client subprocesses*
  duplicating weights, which a single deployment already solves.
- **15-min timeout / mem-scaled vCPU** — fine for our request sizes.
- No warm sharing across concurrent invocations, but concurrency is low by assumption.

---

## 5. Image strategy on AWS — fat vs slim

`infra/` already builds **both** shapes from one Dockerfile (`BAKE_MODELS`). Mapping the
existing tradeoff table onto AWS:

| | Fat (baked) | Slim (S3-referenced) |
|---|---|---|
| ECR image | ~6.5 GB (est. = slim + ~4 GB weights) | **~2.4 GB** (measured) |
| ECR storage cost | ~$0.65/mo | ~$0.24/mo (trivial either way) |
| Cold start | image pull only | image pull **+ ~4 GB S3 sync** |
| Weight update | rebuild + repush | update S3, restart — **no rebuild** |
| Runtime deps | none | S3 access (VPC endpoint + IAM) |
| Best on AWS for | Lambda (baked ≤10 GB image), simplest Fargate | Fargate/SageMaker multi-replica, frequent weight churn |

**Recommendation:**
- **Fargate / SageMaker → slim + S3.** Native fit; weights live in S3 (SageMaker's model
  artifact model *is* this). Use the **S3 VPC gateway endpoint (free)** so the 4 GB sync
  never touches a NAT gateway. For multi-replica Fargate, seed weights **once to EFS** and
  mount read-only to skip per-task downloads.
- **Lambda → fat (baked into the ≤10 GB image).** Avoids per-cold-start S3 sync latency.
- **Local / offline / CI → fat**, exactly as today.
- For Fargate slim in-container sync, add **`s5cmd`** to the runtime stage (entrypoint
  already prefers it); or use an init/sidecar container as the compose file demonstrates.

---

## 6. VPC & networking (consumed by sibling services)

- Deploy tasks in **private subnets**; **no public exposure.**
- Front with an **internal ALB** (or **NLB**, or **ECS Service Connect / Cloud Map**
  service discovery). Sibling services set `RAGINFER_URL=http://raginfer.internal:8799`.
- **Security group:** inbound `8799` **only from client service SGs** — SG-to-SG, not CIDR.
- **S3 access via VPC gateway endpoint** (free, keeps weight sync off NAT). If HF Hub is
  ever needed (`MODEL_SOURCE=hf`), that alone needs NAT/internet — prefer S3 or baked to
  avoid it.
- **Health checks:** target group hits a readiness endpoint with a **generous start
  period** (weights sync + warm-up can take minutes) — see the `/readyz` change below.
- **Auth:** network isolation (SG) is the primary control. Optionally add a shared-secret
  header check for defense-in-depth; not required if SG-restricted.

---

## 7. AWS services to provision — infra hand-off

**Scope: the chosen path (ECS Fargate CPU/ARM64, VPC-internal, slim image + S3
weights, optional scheduled scale-to-zero).** This is the checklist for the infra
project. "Required" = needed for the service to run; "Optional" = enable when the
noted condition applies. Mirrored, with the app-side contract + task-def template +
scale-to-zero CLI, in [`infra/aws/`](../infra/aws/).

| # | AWS service | Purpose | Required? |
|---|---|---|---|
| 1 | **Amazon ECR** | Store the container image (slim, `arm64` tag; multi-arch optional) | ✅ Required |
| 2 | **Amazon S3** — weights bucket | HF-cache-layout weights synced to `HF_HOME` at start (`MODEL_SOURCE=s3`) | ✅ Required |
| 3 | **S3 VPC Gateway Endpoint** | Free, private S3 access; keeps the ~4 GB weight sync off NAT | ✅ Required |
| 4 | **IAM roles** — task + execution | Task role: `s3:GetObject`/`ListBucket` on the bucket. Execution role: ECR pull + CloudWatch Logs | ✅ Required |
| 5 | **CloudWatch Logs** — log group `/ecs/raginfer` | Capture JSON stdout logs (`DEPLOY_MODE=aws`), set retention | ✅ Required |
| 6 | **ECS cluster + Fargate service + task def** | Run the container (4 vCPU / 16 GB, `ARM64`); service holds subnets/SG/desiredCount | ✅ Required |
| 7 | **Internal ALB** (or NLB / Cloud Map) | VPC-internal endpoint `raginfer.internal:8799`; target-group health check → `/readyz` | ✅ Required |
| 8 | **Security groups** | Service SG inbound `8799` **only from client service SGs** (SG-to-SG) | ✅ Required |
| 9 | **Private subnets** (≥2 AZ) | Task placement, no public IP | ✅ Required |
| 10 | **SSM Parameter Store / Secrets Manager** | Inject `DEPLOY_MODE`, `MODEL_S3_URI`, `AWS_REGION`, etc. into the task | ✅ Required |
| 11 | **CloudWatch Alarms / dashboard** | CPU/mem/5xx/latency alarms; cost-tag all resources | ⭐ Recommended |
| 12 | **Application Auto Scaling** — scheduled actions | Scale service `DesiredCount` 1↔0 off-hours (register scalable target min 0) | ⬜ Optional — scale-to-zero |
| 13 | **Amazon EFS + access point** | Pre-seeded read-only `/models` shared across tasks (skip per-task S3 sync) | ⬜ Optional — multi-replica |
| 14 | **VPC interface endpoints** (ECR api/dkr, Logs, STS, SSM) | Private control-plane access when the VPC has **no NAT** at all | ⬜ Optional — fully-private VPC |

Not for the chosen path (documented only, would need a client-contract change):
Lambda + Function URL (Option B), SageMaker async/real-time (Options C/D). See §4.

**Weight-bucket seeding (one-time, infra or CI):** materialize weights and upload —
`HF_HOME=./models python infra/download_models.py bge jina` then
`aws s3 sync ./models s3://<weights-bucket>` (see [`infra/README.md`](../infra/README.md)).

---

## 8. Change list — status

Legend: **✅ done** (landed + tested in this repo, no live AWS needed) ·
**🔶 partial** · **⬜ pending** (needs the live-AWS infra project or a later decision).
Phases 0–1 are backward-compatible and unblock local + Fargate.

### Phase 0 — config & logging seam (code, no behavior change locally)
1. ✅ **`requirements.txt`**: added `pydantic-settings`, `python-json-logger`, `boto3`.
2. ✅ **`src/config/settings.py`**: typed `Settings` + `DEPLOY_MODE` with per-mode defaults
   via a `model_validator`; legacy env names read as aliases. (§3.1)
3. ✅ **`src/logging_config.py`**: `aws` → JSON to stdout, no file handler; local unchanged. (§3.3)
4. ✅ **`src/service/server.py`**: builds `get_settings()` once; uses `settings.host/port/warmup`.
5. ✅ **Tests**: `test/test_settings.py` — local defaults unchanged, `aws` flips
   host/log/backends, legacy env wins, storage selection. (8 tests)

### Phase 1 — cloud readiness (code)
6. ✅ **`src/service/app.py`**: added **`GET /readyz`** (503 until both models loaded);
   `/healthz` stays always-200 liveness. (+2 tests)
7. ✅ **`src/storage/`**: `Storage` protocol + `LocalStorage`/`S3Storage` via `build_storage`,
   selected by `artifact_backend`. Forward-looking seam, not in the request path. (§3.2)
8. ✅ **Graceful shutdown**: `tini` PID 1 + uvicorn SIGTERM already clean; startup log now
   reports `deploy_mode`/`model_source`.

### Phase 2 — packaging for AWS
9.  ✅ **`infra/Dockerfile`**: `s5cmd` baked into the runtime stage (arch-aware), **plus
    arch-aware CPU torch** (x86 → cpu index; arm64/Graviton → PyPI aarch64 wheel). Both the
    torch aarch64 wheel and the `s5cmd` amd64/arm64 assets were verified to resolve.
10. 🔶 **Multi-arch build → ECR**: Dockerfile is multi-arch-ready and the ARM `buildx` recipe
    is documented ([`infra/README.md`](../infra/README.md)); the **actual build+push to ECR**
    is the infra/CI project (prefer a native ARM builder over QEMU).
11. ✅ **`infra/` docs**: `DEPLOY_MODE`/AWS section in `.env.example`; AWS mapping in
    `infra/README.md`; Fargate contract + `task-def.template.json` + scheduled scale-to-zero
    in [`infra/aws/`](../infra/aws/).
12. ⬜ **CI**: GitHub Actions for test + image build/push to ECR. (not yet added)

### Phase 3 — Fargate provisioning (infra project — needs live AWS)
- ⬜ ECS cluster + Fargate service/task def (4 vCPU / 16 GB, **ARM64**), internal ALB with
  target-group health check → `/readyz`, SG-to-SG, S3 VPC gateway endpoint, IAM task/exec roles.
- ⬜ **Scheduled scale-to-zero**: register the scalable target (min 0) + up/down scheduled
  actions — CLI ready in [`infra/aws/README.md`](../infra/aws/README.md).
- ⬜ Weights bucket seeded (one-time `download_models.py` → `s3 sync`).
- ⬜ *(optional)* EFS-mounted `/models` for multi-replica.
- *(Not chosen: Lambda/SageMaker — documented in §4 only; would change the client contract.)*

### Phase 4 — observability & cost guardrails (infra project)
13. ⬜ CloudWatch dashboards/alarms (CPU, mem, 5xx, latency); log-group retention on `/ecs/raginfer`.
14. ⬜ Cost tags on all resources; if scale-to-zero, an alarm on unexpected always-on.

**Summary:** all repo-side code + packaging is **done and green** (17 tests, no weights
loaded); everything remaining is **live-AWS provisioning owned by the infra project** (§7),
plus the optional CI workflow.

---

## 9. Backward-compatibility guarantee

- `DEPLOY_MODE` defaults to `local`; **unset = today's behavior, byte-for-byte.**
- All legacy env vars keep working and **override** mode defaults.
- Model wrappers, the `embed`/`rerank` contracts, sparse `{indices, values}` shape, and the
  fat-image path are **untouched**. Local `conda run -n py312 python -m src.service.server`
  and `docker compose --profile fat` are unaffected.

## 10. Open questions / risks

1. **Client integration ceiling:** Options A/A′/B keep the plain-HTTP `RAGINFER_URL`
   contract (zero client change). C/D (SageMaker) do **not** — they'd force XbrlRag/MacroRAG
   shim rewrites. Given "used by other services in a VPC," staying on plain HTTP (Fargate
   or Lambda) is the low-friction path. Confirm we want to protect that contract.
2. **Lambda memory headroom:** needs a real load test of both models at 10 GB before
   committing; the split-function fallback is the mitigation.
3. **Cold-start budget:** scheduled scale-to-zero (A′) and Lambda (B) both trade a
   multi-minute (Fargate) / ~minute (Lambda) first-call latency for near-zero idle cost —
   confirmed acceptable by the batch-tolerant decision, but worth a documented SLO.
4. **Weight-sync path in fully-private VPC:** requires the S3 gateway endpoint (and ECR/Logs
   interface endpoints if no NAT at all). Flag for the infra team.
5. **Concurrency model:** the service serializes calls (locks, single worker). Under mild
   load that's fine; if concurrency ever rises, scale **replicas** behind the LB — never
   uvicorn workers (each worker = another full weight copy).
```
