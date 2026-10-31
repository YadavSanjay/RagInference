# Runbook — build & push the RagInference image

Concrete, copy-paste commands for **this** project. For the conceptual reference
(ECR pull setup, EC2/Fargate run, env vars, troubleshooting) see
[`../deployment-guide.md`](../deployment-guide.md); for image internals see
[`../../infra/README.md`](../../infra/README.md).

## Architecture policy

| Target | Arch | How it's built |
|---|---|---|
| **Local** (dev/test on an x86 workstation) | **amd64** | plain `docker build` (host arch) |
| **AWS** (ECS Fargate / EC2 on Graviton) | **arm64** | `docker buildx --platform linux/arm64 --push` |

> ⚠️ The two are **not interchangeable**. An amd64 image fails with `exec format
> error` on a Graviton task; an arm64 image won't run on the local x86 Docker.
> Always verify the arch before you push (step 4).

> 🔒 Real values (account id, profile) live in **`infra/aws/deploy.env`**, which is
> **gitignored** — never committed. Copy the template once and fill it in:
> `cp infra/aws/deploy.env.example infra/aws/deploy.env`. The commands below source it.

---

## Local — amd64 (dev/test)

```powershell
# from the repo root

# slim (no weights) — supply weights at run time (MODEL_SOURCE=hf pulls from the Hub)
docker build -f infra/Dockerfile -t raginfer:slim .
docker run --rm -p 8799:8799 -e MODEL_SOURCE=hf -v raginfer-models:/models raginfer:slim

# or fat (weights baked in — needs nothing at run time, ~6.5 GB)
docker build -f infra/Dockerfile --build-arg BAKE_MODELS="bge jina" -t raginfer:fat .
docker run --rm -p 8799:8799 raginfer:fat
```

Verify: `curl -s localhost:8799/readyz` → 503 while loading, 200 when both models
are ready. (`/healthz` is liveness-only and 200s before models load.)

---

## AWS — arm64 (build & push to ECR)

```powershell
# from the repo root

# --- 0. Load deploy vars from the gitignored env file, then derive the rest ---
#     (one-time: cp infra/aws/deploy.env.example infra/aws/deploy.env  and fill in)
Get-Content infra\aws\deploy.env | ForEach-Object {
  if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { Set-Item "Env:$($matches[1])" $matches[2] }
}
$REGISTRY = "$($env:ACCT).dkr.ecr.$($env:REGION).amazonaws.com"
$REPO     = "$REGISTRY/$($env:ECR_NAMESPACE)"
$SHA      = (git rev-parse --short HEAD)     # tag = exact commit; requires a clean tree
$TAG      = "${REPO}:slim-arm64-$SHA"
# $env:AWS_PROFILE is now set from deploy.env (profile whose role has ecr:*)

# --- 1. Log docker in to ECR (as $env:AWS_PROFILE) ---
aws ecr get-login-password --region $env:REGION | docker login --username AWS --password-stdin $REGISTRY

# --- 2. A buildx container builder (needed for cross-arch build + --push) ---
#     First time: creates and selects it. If it already exists, run instead:
#         docker buildx use raginfer
docker buildx create --name raginfer --driver docker-container --use --bootstrap

# --- 3. Build arm64 (slim) and push, from the repo root ---
#     --provenance=false keeps a clean single-arch image manifest (no attestation
#     index), which ECS/ECR expect for a single-platform deploy.
docker buildx build -f infra/Dockerfile --platform linux/arm64 --provenance=false `
  -t $TAG --push .

# --- 4. Verify the pushed image is actually arm64 (do this every push) ---
docker buildx imagetools inspect $TAG | Select-String -Pattern "Platform"
#   Platform:  linux/arm64        <-- expected
```

### Notes on the flags

- **`docker buildx create --driver docker-container`** — the default `docker`
  driver can't do cross-arch build-and-push cleanly; the container driver can.
  The command is idempotent-ish: if the builder already exists it errors — then
  just `docker buildx use raginfer`.
- **`--platform linux/arm64`** — forces the Graviton target regardless of host
  arch. On an x86 host the build runs under **QEMU emulation** (slow — minutes;
  the tax is emulating the torch/deps install, not the weights). On a native ARM
  builder (Graviton CodeBuild/CI runner, Apple silicon) it's fast.
- **`--provenance=false`** — suppresses the SBOM/provenance attestation manifest
  buildx adds by default, which would otherwise make the pushed tag a multi-manifest
  index. ECS/ECR want a single-platform image manifest here.
- **`$SHA` from `git rev-parse --short HEAD`** — tags the image with the exact
  commit; only meaningful with a clean tree (`git status` clean). Pin a literal SHA
  if you're building an older commit.

### Weights (slim image)

The slim image carries no weights. On AWS, provide them at run time via
`MODEL_SOURCE=s3` + `MODEL_S3_URI=s3://<bucket>` (synced into `HF_HOME` at start;
task role needs `s3:GetObject`/`ListBucket`). One-time bucket seed:

```powershell
$env:HF_HOME = ".\models"
conda run -n py312 python infra\download_models.py bge jina
aws s3 sync .\models s3://<weights-bucket>
```

See [`../deployment-guide.md`](../deployment-guide.md) §5–§7 for the pull-side IAM,
VPC endpoints, and the EC2/Fargate run.
