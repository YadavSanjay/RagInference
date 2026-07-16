# RagInference on AWS — ECS Fargate reference (for the infra project)

Reference configuration for running RagInference on **ECS Fargate (CPU),
VPC-internal**, with **optional scheduled scale-to-zero off-hours**. Actual
provisioning lives in the separate infra project — this folder gives the app-side
contract (env, ports, health checks) and copy-paste-able Application Auto Scaling
config so the two stay in sync.

See the full analysis in [`docs/aws-deployment-plan.md`](../../docs/aws-deployment-plan.md).

## Shape

```
client services (same VPC)
        │  RAGINFER_URL=http://raginfer.internal:8799
        ▼
  internal ALB / Cloud Map  ──health check──►  GET /readyz (503 until warm)
        ▼
  ECS Fargate service  (desiredCount 0..1)
     └─ task: raginfer container  (CPU, 4 vCPU / 16 GB, arm64)
          ├─ MODEL_SOURCE=s3  ──sync──►  s3://raginfer-models  (via S3 VPC gateway endpoint)
          └─ stdout JSON logs ─────────►  CloudWatch Logs
```

## Container contract

| Item | Value |
|---|---|
| Image | slim (`BAKE_MODELS` empty); weights synced from S3 at start |
| Port | `8799` (tcp) |
| Liveness | `GET /healthz` → always 200 while process up |
| **Readiness** | `GET /readyz` → **503 until both models loaded**, then 200 |
| Health check start period | **≥ 300 s** (weight sync + warm-up on cold start) |
| CPU / memory | 4 vCPU / 16 GB (both models ≈ 4 GB weights + torch overhead) |
| Architecture | `ARM64` preferred (~20% cheaper); `X86_64` also fine |
| Stop timeout | ≥ 30 s (clean SIGTERM via tini → uvicorn) |

### Task environment (from SSM Parameter Store / Secrets Manager)

```
DEPLOY_MODE=aws                       # → host 0.0.0.0, JSON stdout logs, no file, s3 storage
MODEL_SOURCE=s3
MODEL_S3_URI=s3://raginfer-models     # HF-cache-layout weights (see infra/download_models.py)
HF_HOME=/models
RERANKER=jina
EMBEDDER=bge
AWS_REGION=us-east-1
# optional artifact seam (only if a future feature uses src/storage):
# RAGINFER_ARTIFACT_BUCKET=raginfer-artifacts
```

Credentials: **IAM task role** with `s3:GetObject`/`s3:ListBucket` on the weights
bucket — never bake `AWS_*` keys. Reach S3 through the **S3 VPC gateway endpoint**
(free) so the ~4 GB sync never traverses a NAT gateway.

> **Multi-replica note:** to avoid every task downloading 4 GB, seed the weights
> once into **EFS** and mount it read-only at `/models` (set `MODEL_SOURCE=baked`
> so the app runs offline against the pre-populated mount). For a single task the
> in-container `s5cmd` S3 sync (baked into the image) is simpler.

## Networking

- Tasks in **private subnets**, no public IP.
- Service exposed via **internal ALB** (or Cloud Map / ECS Service Connect):
  clients set `RAGINFER_URL=http://raginfer.internal:8799`.
- **Security group:** inbound `8799` **only from client service SGs** (SG-to-SG).
- VPC endpoints for a fully-private VPC: **S3 (gateway)**, plus **ECR api/dkr**,
  **CloudWatch Logs**, **STS** (interface) if there is no NAT.

## Optional: scheduled scale-to-zero off-hours

Load is batch-tolerant, so the service can drop to **0 tasks** off-hours and back
to 1 during working hours. First request after a scale-up pays a cold start
(task boot + weight sync + warm-up, ~1–3 min) — acceptable per the batch-tolerant
SLO. Use **Application Auto Scaling scheduled actions** on the ECS service's
`DesiredCount` (`ecs:service:DesiredCount`).

Register the scalable target (min 0 so scheduled/again scaling can reach zero):

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/<cluster>/raginfer \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 0 --max-capacity 1
```

Scale **up** at 08:00 and **down** at 20:00 on weekdays (UTC — adjust `Timezone`):

```bash
# Up weekday mornings
aws application-autoscaling put-scheduled-action \
  --service-namespace ecs \
  --resource-id service/<cluster>/raginfer \
  --scalable-dimension ecs:service:DesiredCount \
  --scheduled-action-name raginfer-up \
  --schedule "cron(0 8 ? * MON-FRI *)" --timezone "UTC" \
  --scalable-target-action MinCapacity=1,MaxCapacity=1

# Down weekday evenings (→ 0 tasks, ~$0 compute overnight/weekends)
aws application-autoscaling put-scheduled-action \
  --service-namespace ecs \
  --resource-id service/<cluster>/raginfer \
  --scalable-dimension ecs:service:DesiredCount \
  --scheduled-action-name raginfer-down \
  --schedule "cron(0 20 ? * MON-FRI *)" --timezone "UTC" \
  --scalable-target-action MinCapacity=0,MaxCapacity=0
```

Weekends stay at 0 because the last weekday action leaves capacity at 0 until the
next Monday `up`. Keep the ALB deregistration delay short and the target-group
health check pointed at `/readyz` so traffic only lands on a warm task.

> **Not request-driven:** an ALB cannot itself wake a 0-task service on an
> incoming request. If you need wake-on-request instead of a fixed schedule, that
> is the SageMaker/Lambda path in the deployment plan — a bigger change that also
> touches the client contract.

## AWS services this deployment uses

ECR · S3 (weights, + optional artifacts) · **S3 VPC gateway endpoint** · IAM
(task + execution roles) · CloudWatch Logs + Alarms · SSM Parameter Store /
Secrets Manager · ECS cluster + Fargate service/task def · internal ALB (or Cloud
Map) · Application Auto Scaling · *(optional)* EFS · *(fully-private VPC)* ECR/
Logs/STS interface endpoints.

## `task-def.template.json`

A skeleton container definition is in [`task-def.template.json`](task-def.template.json)
— fill the `<PLACEHOLDERS>` (account, region, image tag, role ARNs, subnets/SGs
are set on the *service*, not the task def) in the infra project.
