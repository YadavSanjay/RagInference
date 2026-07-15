#!/usr/bin/env python
"""Download model snapshots into HF_HOME.

Used in two places:
  * Case A (build time) — bake weights into the image (`BAKE_MODELS` arg).
  * Case B (prep time)  — populate an S3 bucket / shared volume once, offline.

Downloads land in the standard HF cache layout under $HF_HOME, which is exactly
where `from_pretrained` / `BGEM3FlagModel(...)` look them up at runtime — so a
baked image or a synced volume is a drop-in for an online HF fetch.

Usage:
    HF_HOME=/models python download_models.py [name ...]

Names are short aliases (see REPOS) OR raw HF repo ids. With no args, the
service defaults (bge + jina) are downloaded.
"""
from __future__ import annotations

import os
import sys

from huggingface_hub import snapshot_download

# Short alias -> HF repo id. Mirrors the service's build_embedder/build_reranker.
REPOS = {
    "bge": "BAAI/bge-m3",                     # default embedder (dense + sparse)
    "jina": "jinaai/jina-reranker-v3",        # default reranker (local transformers)
    "bge-reranker": "BAAI/bge-reranker-v2-m3",  # RERANKER=bge alternative
}
DEFAULTS = ["bge", "jina"]

# Skip weights for frameworks this service never uses (torch only). Trims the
# snapshot substantially without touching what transformers/FlagEmbedding load.
IGNORE_PATTERNS = [
    "*.onnx", "onnx/*", "openvino/*",   # ONNX / OpenVINO exports
    "*.h5", "tf_model*",                 # TensorFlow
    "*.msgpack", "flax_model*",          # Flax
    "*.gguf",                            # llama.cpp quant
]


def main(argv: list[str]) -> int:
    names = argv or DEFAULTS
    hf_home = os.environ.get("HF_HOME", "(default ~/.cache/huggingface)")
    for name in names:
        repo = REPOS.get(name, name)  # allow raw "org/model" ids too
        print(f"[download_models] {repo} -> {hf_home}", flush=True)
        snapshot_download(
            repo_id=repo,
            ignore_patterns=IGNORE_PATTERNS,
            # trust_remote_code models (jina-v3) ship .py in the snapshot; those
            # are matched by default patterns and kept.
        )
    print("[download_models] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
