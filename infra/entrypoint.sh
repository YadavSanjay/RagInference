#!/bin/sh
# RagInference container entrypoint.
#
# Resolves where the model weights come from, then execs the service. The
# service code is unchanged: it just reads weights from $HF_HOME.
#
#   MODEL_SOURCE=baked  (default, Case A)
#       Weights are already in the image at $HF_HOME. Run fully offline.
#
#   MODEL_SOURCE=hf     (Case B, no object store)
#       Let HuggingFace Hub download on first use into $HF_HOME. Point
#       $HF_HOME at a persistent named volume so replicas/restarts reuse it.
#
#   MODEL_SOURCE=s3     (Case B, shared object storage)
#       Sync $MODEL_S3_URI into $HF_HOME once at startup, then run offline.
#       Needs s5cmd (preferred) or the aws CLI on PATH, OR the volume to be
#       pre-populated by an init container (see infra/docker-compose.yml),
#       in which case this just verifies and runs offline.
set -eu

: "${HF_HOME:=/models}"
: "${MODEL_SOURCE:=baked}"
export HF_HOME

log() { echo "[entrypoint] $*" >&2; }

case "$MODEL_SOURCE" in
  baked)
    export HF_HUB_OFFLINE=1
    log "MODEL_SOURCE=baked -> using weights in image at $HF_HOME (offline)"
    ;;

  hf)
    log "MODEL_SOURCE=hf -> weights fetched from HuggingFace Hub into $HF_HOME on demand"
    ;;

  s3)
    : "${MODEL_S3_URI:?MODEL_SOURCE=s3 requires MODEL_S3_URI (e.g. s3://my-bucket/raginfer-models)}"
    src="${MODEL_S3_URI%/}"
    mkdir -p "$HF_HOME"
    if command -v s5cmd >/dev/null 2>&1; then
      log "syncing $src -> $HF_HOME via s5cmd"
      s5cmd ${AWS_ENDPOINT_URL:+--endpoint-url "$AWS_ENDPOINT_URL"} sync "$src/*" "$HF_HOME/"
    elif command -v aws >/dev/null 2>&1; then
      log "syncing $src -> $HF_HOME via aws s3 sync"
      aws ${AWS_ENDPOINT_URL:+--endpoint-url "$AWS_ENDPOINT_URL"} s3 sync "$src" "$HF_HOME"
    elif [ -n "$(ls -A "$HF_HOME" 2>/dev/null)" ]; then
      log "no s5cmd/aws on PATH, but $HF_HOME is already populated (pre-mounted volume) -> using it"
    else
      log "ERROR: MODEL_SOURCE=s3 but no s5cmd/aws CLI and $HF_HOME is empty."
      log "       Either bake an S3 client into the image, or pre-populate the"
      log "       volume with an init container (see infra/docker-compose.yml)."
      exit 1
    fi
    export HF_HUB_OFFLINE=1
    log "s3 sync complete -> running offline from $HF_HOME"
    ;;

  *)
    log "ERROR: unknown MODEL_SOURCE='$MODEL_SOURCE' (expected baked|hf|s3)"
    exit 1
    ;;
esac

exec python -m src.service.server
