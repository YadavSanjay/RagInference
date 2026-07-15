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

# 1) CPU-only torch FIRST, from the dedicated CPU wheel index. Installing it
#    up front means the requirements install below sees torch already
#    satisfied and never pulls the multi-GB CUDA build as a transitive dep.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

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
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
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

# Container-level liveness: hit the service's own /healthz.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/healthz'%os.getenv('RAGINFER_PORT','8799'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status==200 else 1)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
