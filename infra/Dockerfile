# syntax=docker/dockerfile:1.7
#
# RagInference — CPU-only inference service image.
#
# ONE Dockerfile, TWO shapes, selected by the BAKE_MODELS build arg:
#
#   Case A ("fat" / self-contained): weights baked into the image
#       docker build -f infra/Dockerfile --build-arg BAKE_MODELS="bge jina" -t raginfer:fat .
#
#   Case B ("slim" / referenced): no weights in the image; they are synced
#   from S3 / a shared volume into HF_HOME at runtime (see infra/entrypoint.sh)
#       docker build -f infra/Dockerfile -t raginfer:slim .
#
# Build context is the REPO ROOT (so we can COPY src/, requirements.txt).
# BuildKit honours infra/Dockerfile.dockerignore for the context.

ARG PYTHON_VERSION=3.12

##############################################################################
# Stage 1: builder — compile/install all Python deps into an isolated venv.
# The CPU-only torch wheel is the single biggest size win (~2.3 GB vs CUDA).
##############################################################################
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is only needed to build any sdist wheels; it stays in this
# stage and never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 1) CPU-only torch FIRST so the requirements install below sees it satisfied
#    and never pulls a multi-GB CUDA build as a transitive dep. The source is
#    arch-dependent:
#      * x86_64 (amd64): the dedicated CPU wheel index — the default PyPI wheel
#        is CUDA-enabled (~2.5 GB), so we must pin the cpu index.
#      * arm64 (Graviton): the default PyPI wheel is *already* CPU-only (no CUDA
#        exists for aarch64) and carries the Arm Compute Library optimizations,
#        so we install it straight from PyPI. Verified wheel:
#        torch-*-cp312-manylinux_2_28_aarch64.whl.
#    TARGETARCH is provided automatically by buildx for multi-arch builds.
ARG TARGETARCH
RUN case "${TARGETARCH:-amd64}" in \
      arm64) pip install torch ;; \
      *)     pip install --index-url https://download.pytorch.org/whl/cpu torch ;; \
    esac

# 2) The rest of the service requirements + the hub client used to fetch models
#    (hf_transfer gives much faster parallel snapshot downloads).
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install "huggingface_hub[hf_transfer]"

##############################################################################
# Stage 2: models — optionally bake weights into an HF cache at /models.
# When BAKE_MODELS is empty (Case B) this produces just an empty /models dir,
# so the runtime COPY below is a no-op and the image stays slim.
##############################################################################
FROM builder AS models

ARG BAKE_MODELS=""
ENV HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1

COPY infra/download_models.py /tmp/download_models.py
RUN mkdir -p /models \
    && if [ -n "$BAKE_MODELS" ]; then \
         echo "Baking models: $BAKE_MODELS"; \
         python /tmp/download_models.py $BAKE_MODELS; \
       else \
         echo "BAKE_MODELS empty -> slim image (weights referenced at runtime)"; \
       fi

##############################################################################
# Stage 3: runtime — slim base + venv + models + app code. No build toolchain.
##############################################################################
FROM python:${PYTHON_VERSION}-slim AS runtime

# tini gives us proper PID 1 signal handling (clean SIGTERM -> uvicorn shutdown).
# s5cmd is baked in for the slim/S3 weight-sync path (entrypoint.sh prefers it
# over the aws CLI). It's a ~15 MB static binary; unused/harmless in the fat
# (baked, offline) image. ca-certificates stays for runtime TLS to S3.
# TARGETARCH is provided automatically by buildx for multi-arch builds.
ARG TARGETARCH
ARG S5CMD_VERSION=2.2.2
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini ca-certificates curl \
    && case "${TARGETARCH:-amd64}" in \
         arm64) S5_ARCH="Linux-arm64" ;; \
         *)     S5_ARCH="Linux-64bit" ;; \
       esac \
    && curl -sSL "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_${S5_ARCH}.tar.gz" \
       | tar -xz -C /usr/local/bin s5cmd \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models \
    # In-container the service must listen on all interfaces.
    RAGINFER_HOST=0.0.0.0 \
    RAGINFER_PORT=8799 \
    # baked | hf | s3  (see infra/entrypoint.sh)
    MODEL_SOURCE=baked

# venv (all deps, incl. CPU torch) and the model cache (empty in the slim case).
COPY --from=builder /opt/venv /opt/venv
COPY --from=models  /models   /models

WORKDIR /app
COPY src/     /app/src/
COPY configs/ /app/configs/
COPY infra/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run as an unprivileged user; give it the model cache so runtime sync can write.
RUN useradd --create-home --uid 10001 raginfer \
    && chown -R raginfer:raginfer /models /app
USER raginfer

EXPOSE 8799

# Container health = READINESS (/readyz), not liveness (/healthz).
# /healthz is always-200 while the process is up, so using it here would report
# "healthy" for a container whose models failed to load — exactly the silent
# failure we must not ship. /readyz is 503 until both models are loaded.
# start-period covers weight sync + warm-up on a cold start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/readyz'%os.getenv('RAGINFER_PORT','8799'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status==200 else 1)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
