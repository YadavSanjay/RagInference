# RagInference — Packaging & Deployment Guide

End-to-end runbook: **build the image locally → push to Amazon ECR → let a host
pull from ECR → run it on EC2 or ECS Fargate.**

For *why* the architecture is what it is (CPU vs GPU cost, hosting options), see
[`aws-deployment-plan.md`](aws-deployment-plan.md). For the image internals and the
fat/slim tradeoff, see [`../infra/README.md`](../infra/README.md). For the Fargate
service contract specifically, see [`../infra/aws/README.md`](../infra/aws/README.md).

> **Just want the exact commands we run?** The copy-paste build/push runbook
> (local amd64 + AWS arm64 → ECR) is [`runbook/build-and-push-ecr.md`](runbook/build-and-push-ecr.md).
> This guide is the fuller reference behind it.

---

## 0. What you are deploying

One `Dockerfile` ([`infra/Dockerfile`](../infra/Dockerfile)), two shapes:

| Shape | `BAKE_MODELS` | Size | Weights come from |
|---|---|---|---|
| **slim** | *(omitted)* | **~2.4 GB** (measured) | S3 / mounted volume at runtime |
| **fat** | `"bge jina"` | ~6.5 GB (est. = slim + ~4 GB weights) | baked into the image |

Both run identically otherwise: non-root user `raginfer` (uid 10001), `tini` as
PID 1, listens on **8799**, `s5cmd` on `PATH` for S3 sync, and a container
`HEALTHCHECK` bound to **`/readyz`** (503 until both models load).

> **Which shape?** For AWS use **slim + S3 weights** (cheap ECR, update weights
> without a rebuild). Use **fat** for local/offline/air-gapped runs and for AWS
> Lambda. See the tradeoff table in [`../infra/README.md`](../infra/README.md).

---

## 1. Prerequisites

- Docker with BuildKit (Docker Desktop, or `docker buildx` on Linux).
- AWS CLI v2, authenticated with permission to create/push to ECR.
- The account id and region you are deploying into.

Set these once — every command below reuses them:

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=<acct>                      # 12-digit account id
export ECR_REPO=raginfer
export REGISTRY=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
export IMAGE=${REGISTRY}/${ECR_REPO}
```

---

## 2. Build locally

```bash
# from the repo root (build context = repo root)
# slim (recommended for AWS)
docker build -f infra/Dockerfile -t raginfer:slim .

# fat (weights baked in — adds ~4 GB and a long first build)
docker build -f infra/Dockerfile --build-arg BAKE_MODELS="bge jina" -t raginfer:fat .
```

### Building for ARM64 / Graviton (recommended for AWS — ~20% cheaper)

The Dockerfile is multi-arch (`TARGETARCH`): x86_64 pins the CPU torch index,
arm64 uses the PyPI aarch64 CPU wheel. Build with `buildx`:

```bash
docker buildx build -f infra/Dockerfile --platform linux/arm64 -t raginfer:slim-arm64 --load .
```

> ⚠️ **The image architecture must match the host.** An `arm64` image will not run
> on an x86 EC2 instance and vice-versa. arm64 → Graviton (`t4g`/`m7g`/`c7g`);
> amd64 → `t3`/`m7i`/`c7i`. Building arm64 on an x86 machine uses QEMU emulation
> and is **slow** — prefer a native ARM builder (Graviton CI runner / CodeBuild ARM).

---

## 3. Run locally to verify

The slim image has **no weights**, so give it some. Fastest local check — let it
pull from HuggingFace into a named volume:

```bash
docker run --rm -p 8799:8799 \
  -e MODEL_SOURCE=hf \
  -v raginfer-models:/models \
  raginfer:slim
```

Or run the fat image, which needs nothing:

```bash
docker run --rm -p 8799:8799 raginfer:fat
```

Verify (first start downloads/loads ~4 GB — allow several minutes):

```bash
curl -s localhost:8799/healthz          # liveness: 200 as soon as the process is up
curl -s localhost:8799/readyz           # readiness: 503 until BOTH models load, then 200
curl -s localhost:8799/embed  -H 'content-type: application/json' \
     -d '{"texts":["export control restrictions"],"dense":true,"sparse":true}'
curl -s localhost:8799/rerank -H 'content-type: application/json' \
     -d '{"query":"q","documents":["a","b"]}'
```

> **If the container exits immediately** with `MODEL_SOURCE=baked but /models is
> EMPTY`, you ran the **slim** image without supplying weights. That is the
> guard working as intended — pick one of the options it prints.

The full local Case-A/Case-B compose demo (including MinIO standing in for S3) is
in [`../infra/README.md`](../infra/README.md).

---

## 4. Push to ECR

### 4.1 Create the repository (once)

```bash
aws ecr create-repository \
  --repository-name "${ECR_REPO}" \
  --region "${AWS_REGION}" \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256
```

Optional but recommended — a lifecycle policy so old images don't accumulate:

```bash
aws ecr put-lifecycle-policy --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
  --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"keep last 10","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}'
```

### 4.2 Authenticate Docker to ECR

```bash
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"
```

> The ECR auth token is valid for **12 hours** — re-run this when pushing later.

### 4.3 Tag and push

> 🎯 **The AWS image must be `linux/arm64`** to get the Graviton cost/performance
> benefit. A plain `docker build` on an x86 machine produces an **amd64** image — if
> you push that and run it on a Graviton task it fails with `exec format error`, and
> if it runs on an x86 Fargate task you silently lose the ~20% saving. **Always build
> the AWS image with `--platform linux/arm64` and verify the arch before pushing.**

**Build-and-push the ARM64 image** (git SHA + moving tag, so deployments are traceable):

```bash
GIT_SHA=$(git rev-parse --short HEAD)

docker buildx build -f infra/Dockerfile --platform linux/arm64 \
  -t "${IMAGE}:slim-arm64-${GIT_SHA}" -t "${IMAGE}:slim-arm64" --push .
```

**Confirm the pushed image is actually arm64** (do this every time — it's the check
that catches an accidental amd64 push):

```bash
# inspect the manifest in ECR — must report arm64
docker buildx imagetools inspect "${IMAGE}:slim-arm64" | grep -i platform
#   Platform:  linux/arm64        <-- expected

# or from a local image built with --load:
docker image inspect "${IMAGE}:slim-arm64" --format '{{.Architecture}}'   # -> arm64
```

**Multi-arch single tag** (one manifest serving both arches — a Graviton *or* x86 host
each pulls the right variant automatically; safest default if hosts vary):

```bash
docker buildx build -f infra/Dockerfile --platform linux/amd64,linux/arm64 \
  -t "${IMAGE}:slim-${GIT_SHA}" -t "${IMAGE}:slim" --push .
```

**amd64 only** (x86 hosts — `t3`/`m7i`/`c7i`; *not* the recommended AWS path):

```bash
docker tag raginfer:slim "${IMAGE}:slim-amd64-${GIT_SHA}"
docker push "${IMAGE}:slim-amd64-${GIT_SHA}"
```

List what's in the repo:

```bash
aws ecr list-images --repository-name "${ECR_REPO}" --region "${AWS_REGION}"
```

---

## 5. Allow hosts to **pull** from ECR

This is what lets an EC2 instance (or ECS task, or your laptop) download the image.

### 5.1 IAM permissions for the pulling host

The puller needs these four actions. The AWS managed policy
**`AmazonEC2ContainerRegistryReadOnly`** covers them:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage"
    ],
    "Resource": "*"
  }]
}
```

`ecr:GetAuthorizationToken` must be on `Resource: "*"`; the other three can be
scoped to the repository ARN
(`arn:aws:ecr:<region>:<acct>:repository/raginfer`) if you want least privilege.

**Attach it via an instance profile** (EC2) or the **task execution role** (ECS) —
never long-lived access keys on the host:

```bash
aws iam create-role --role-name raginfer-ec2-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy --role-name raginfer-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# weights bucket read access (slim image, MODEL_SOURCE=s3)
aws iam put-role-policy --role-name raginfer-ec2-role --policy-name raginfer-weights-read \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],"Resource":["arn:aws:s3:::<weights-bucket>","arn:aws:s3:::<weights-bucket>/*"]}]}'

aws iam create-instance-profile --instance-profile-name raginfer-ec2-profile
aws iam add-role-to-instance-profile --instance-profile-name raginfer-ec2-profile \
  --role-name raginfer-ec2-role
```

### 5.2 Log Docker in on the host

With the instance profile attached, the host authenticates with no stored secrets:

```bash
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"
docker pull "${IMAGE}:slim"
```

**For long-running hosts, the 12-hour token expiry will break unattended pulls.**
Install the ECR credential helper so Docker refreshes credentials automatically:

```bash
sudo yum install -y amazon-ecr-credential-helper     # Amazon Linux
mkdir -p ~/.docker
echo '{"credHelpers":{"'"${REGISTRY}"'":"ecr-login"}}' > ~/.docker/config.json
docker pull "${IMAGE}:slim"                          # no docker login needed
```

### 5.3 Pulling from a **private subnet** (no NAT gateway)

ECR pulls need three endpoints — a commonly-missed one is S3, because **ECR stores
image layers in S3**:

| VPC endpoint | Type | Why |
|---|---|---|
| `com.amazonaws.<region>.ecr.api` | Interface | ECR control plane (auth, manifests) |
| `com.amazonaws.<region>.ecr.dkr` | Interface | Docker registry protocol |
| `com.amazonaws.<region>.s3` | **Gateway** | **Image layer download** + your weights bucket |
| `com.amazonaws.<region>.logs` | Interface | CloudWatch Logs (if shipping logs) |

Without the S3 **gateway** endpoint, `docker pull` authenticates and then hangs or
fails on layer download. The same gateway endpoint also keeps the ~4 GB weight
sync off NAT (and free).

---

## 6. Deploy on EC2

### 6.1 Instance sizing

| | Recommendation |
|---|---|
| **Memory** | **16 GB** — both models ≈ 4 GB of weights plus torch/activation overhead. 8 GB is tight. |
| **vCPU** | 4 to start; rerank (listwise Jina) is the CPU-bound path. |
| **Arch** | Match the image. arm64 → `m7g.xlarge` / `c7g.xlarge` (~20% cheaper); amd64 → `m7i.xlarge` / `t3.xlarge`. |
| **Disk** | ≥ 30 GB (slim image ~2.4 GB + ~4 GB weights + room). Fat image: ≥ 40 GB. |

Attach the **`raginfer-ec2-profile`** instance profile from §5.1 and a security
group allowing inbound **8799 from client security groups only** (not `0.0.0.0/0`).

### 6.2 Install Docker and pull

```bash
sudo dnf install -y docker            # Amazon Linux 2023 (or: yum/apt)
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user      # re-login for this to take effect

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"
docker pull "${IMAGE}:slim"
```

### 6.3 Run it

```bash
docker run -d --name raginfer \
  --restart unless-stopped \
  -p 8799:8799 \
  -e DEPLOY_MODE=aws \
  -e MODEL_SOURCE=s3 \
  -e MODEL_S3_URI=s3://<weights-bucket> \
  -e AWS_REGION="${AWS_REGION}" \
  -e RERANKER=jina -e EMBEDDER=bge \
  -v raginfer-models:/models \
  "${IMAGE}:slim"
```

- `DEPLOY_MODE=aws` → binds `0.0.0.0`, **JSON logs to stdout** (so `docker logs` /
  the CloudWatch agent capture them), no rotating log file.
- `-v raginfer-models:/models` persists the synced weights across restarts, so only
  the first start pays the ~4 GB S3 sync.
- The instance profile supplies credentials — **pass no `AWS_ACCESS_KEY_ID`**.

Watch it come up (first start syncs weights, then warms both models):

```bash
docker logs -f raginfer
curl -s localhost:8799/readyz          # 503 while loading -> 200 when ready
docker inspect --format '{{.State.Health.Status}}' raginfer
```

### 6.4 Survive reboots (systemd)

`--restart unless-stopped` handles daemon restarts. For a managed unit instead:

```ini
# /etc/systemd/system/raginfer.service
[Unit]
Description=RagInference
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=10
ExecStartPre=-/usr/bin/docker rm -f raginfer
ExecStart=/usr/bin/docker run --rm --name raginfer -p 8799:8799 \
  -e DEPLOY_MODE=aws -e MODEL_SOURCE=s3 -e MODEL_S3_URI=s3://<weights-bucket> \
  -e AWS_REGION=us-east-1 -v raginfer-models:/models \
  <acct>.dkr.ecr.us-east-1.amazonaws.com/raginfer:slim
ExecStop=/usr/bin/docker stop raginfer

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now raginfer
```

### 6.5 Point clients at it

```bash
export RAGINFER_URL=http://<ec2-private-ip-or-dns>:8799
```

Clients use the private VPC address — see [`user-guide.md`](user-guide.md).

---

## 7. Deploy on ECS Fargate (the recommended target)

EC2 above is the simplest path; **Fargate is the recommended production target**
(no host to patch, and it supports scheduled scale-to-zero). The task definition
template, health-check wiring, networking and the scale-to-zero CLI live in
[`../infra/aws/README.md`](../infra/aws/README.md). Provisioning itself is owned by
the separate infra project — the services it must create are listed in
[`aws-deployment-plan.md` §7](aws-deployment-plan.md).

---

## 8. Environment variable reference

Everything the container reads. `DEPLOY_MODE` only sets **defaults** — any explicit
variable still wins.

| Variable | Default | Purpose |
|---|---|---|
| `DEPLOY_MODE` | `local` | `aws` → bind `0.0.0.0`, JSON logs to stdout, no log file, S3 artifact backend. **Set this in every container deployment.** |
| `MODEL_SOURCE` | `baked` | `baked` (weights in image/volume) \| `hf` (download from HF Hub) \| `s3` (sync from `MODEL_S3_URI`) |
| `MODEL_S3_URI` | — | **Required when `MODEL_SOURCE=s3`**, e.g. `s3://raginfer-models` |
| `HF_HOME` | `/models` | Where weights live in the container; mount a volume here to persist |
| `AWS_REGION` | — | Region for S3 weight sync / storage |
| `AWS_ENDPOINT_URL` | — | S3-compatible override (MinIO in the local compose demo) |
| `RAGINFER_HOST` | `0.0.0.0` (in image) | Bind address |
| `RAGINFER_PORT` | `8799` | Bind port |
| `RERANKER` | `jina` | `jina` \| `bge` |
| `EMBEDDER` | `bge` | Embedder selection |
| `RAGINFER_WARMUP` | `1` | Load + warm both models at startup |
| `LOG_LEVEL` | `INFO` | Verbosity |
| `RAGINFER_LOG_FORMAT` | `text` (`json` in aws) | Log formatter |
| `RAGINFER_LOG_TO_FILE` | `true` (`false` in aws) | Rotating file vs stdout only |
| `RAGINFER_ARTIFACT_BUCKET` / `_PREFIX` / `_ROOT` | — | Storage seam (future blob IO; unused by `/embed` `/rerank` today) |

**No credentials belong in env or the image** — use an EC2 instance profile or an
ECS task role.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Container exits: `MODEL_SOURCE=baked but /models is EMPTY` | Slim image with no weights. Set `MODEL_SOURCE=s3`+`MODEL_S3_URI`, or `hf`, or mount weights, or use the fat image. |
| `docker pull` hangs on layer download in a private subnet | Missing **S3 gateway** VPC endpoint (ECR layers live in S3). See §5.3. |
| `no basic auth credentials` on pull | ECR token expired (12 h). Re-run `docker login`, or install the credential helper (§5.2). |
| `exec format error` | Image/host architecture mismatch (arm64 image on x86, or vice-versa). See §2. |
| Health stays `starting`, `/readyz` 503 | Still syncing weights / warming up (can take minutes). Check `docker logs`. If it never flips, the model load failed — look for a warm-up exception. |
| `/healthz` 200 but `/rerank` fails | `/healthz` is liveness only. Always gate traffic on **`/readyz`**. |
| Container killed / OOM | Under 16 GB RAM with both models. Resize, or run only one model. |
| Logs go to a file instead of stdout | `DEPLOY_MODE` not set to `aws`. |
