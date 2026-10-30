"""The code-execution pin agrees everywhere it is written — no models loaded.

The service loads Jina's remote code at one commit and the download script fetches
weights at another; if they drift, a baked or offline deployment can't find the
snapshot it needs. Both are read as text because the modules import torch and
huggingface_hub, which the CI environment doesn't have.

    conda run -n py312 python -m pytest test/test_model_pins.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = re.compile(r'"([0-9a-f]{40})"')


def test_jina_revision_matches_between_service_and_download_script():
    service = COMMIT.findall((REPO_ROOT / "src/rerankers/jina_reranker.py").read_text("utf-8"))
    script = COMMIT.findall((REPO_ROOT / "infra/download_models.py").read_text("utf-8"))
    assert len(service) == 1
    assert service == script


def test_jina_loads_at_the_pinned_revision():
    source = (REPO_ROOT / "src/rerankers/jina_reranker.py").read_text("utf-8")
    assert "revision=_MODEL_REVISION" in source
