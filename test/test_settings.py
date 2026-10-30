"""Settings / DEPLOY_MODE tests — no models loaded.

    conda run -n py312 python -m pytest test/test_settings.py -q

Verifies (1) local defaults are unchanged, (2) aws mode flips host/logging/storage
defaults, (3) any explicit legacy env var still wins, and (4) the storage factory
selects the right backend.
"""
from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.storage import LocalStorage, S3Storage, build_storage

# Every env var Settings reads — cleared before each test so host machine env
# (AWS_REGION, HF_HOME, …) can't leak into default assertions.
_VARS = [
    "DEPLOY_MODE", "RAGINFER_DEPLOY_MODE",
    "RAGINFER_HOST", "RAGINFER_PORT",
    "EMBEDDER", "RAGINFER_EMBEDDER", "RERANKER", "RAGINFER_RERANKER", "RAGINFER_WARMUP",
    "RERANKER_DTYPE", "RAGINFER_RERANKER_DTYPE",
    "RAGINFER_MAX_ITEMS", "RAGINFER_MAX_TEXT_CHARS",
    "MODEL_SOURCE", "MODEL_S3_URI", "HF_HOME",
    "RAGINFER_ARTIFACT_BACKEND", "RAGINFER_ARTIFACT_ROOT", "RAGINFER_ARTIFACT_BUCKET",
    "RAGINFER_ARTIFACT_PREFIX",
    "LOG_LEVEL", "RAGINFER_LOG_LEVEL", "RAGINFER_LOG_FORMAT", "RAGINFER_LOG_TO_FILE",
    "AWS_REGION", "AWS_DEFAULT_REGION", "RAGINFER_AWS_REGION",
]


@pytest.fixture()
def clean_env(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _settings() -> Settings:
    # _env_file=None so a stray repo .env can't influence the test.
    return Settings(_env_file=None)


def test_local_defaults_unchanged(clean_env):
    s = _settings()
    assert s.deploy_mode == "local"
    assert s.host == "127.0.0.1"
    assert s.port == 8799
    assert s.embedder == "bge" and s.reranker == "jina"
    assert s.warmup is True
    assert s.model_source == "baked"
    assert s.log_format == "text" and s.log_to_file is True
    assert s.artifact_backend == "local"


def test_aws_mode_flips_defaults(clean_env):
    clean_env.setenv("DEPLOY_MODE", "aws")
    s = _settings()
    assert s.deploy_mode == "aws"
    assert s.host == "0.0.0.0"
    assert s.log_format == "json"
    assert s.log_to_file is False
    assert s.artifact_backend == "s3"


def test_explicit_env_wins_over_mode_defaults(clean_env):
    clean_env.setenv("DEPLOY_MODE", "aws")
    clean_env.setenv("RAGINFER_HOST", "127.0.0.1")   # override aws default 0.0.0.0
    clean_env.setenv("RAGINFER_LOG_FORMAT", "text")  # override aws default json
    s = _settings()
    assert s.host == "127.0.0.1"
    assert s.log_format == "text"


def test_legacy_env_names_still_read(clean_env):
    clean_env.setenv("RERANKER", "bge")
    clean_env.setenv("EMBEDDER", "bge")
    clean_env.setenv("MODEL_SOURCE", "s3")
    clean_env.setenv("MODEL_S3_URI", "s3://raginfer-models")
    clean_env.setenv("RAGINFER_WARMUP", "0")
    s = _settings()
    assert s.reranker == "bge" and s.embedder == "bge"
    assert s.model_source == "s3" and s.model_s3_uri == "s3://raginfer-models"
    assert s.warmup is False


def test_reranker_dtype_defaults_to_float32_in_both_modes(clean_env):
    # fp32 by default: bf16 is emulated (slower) on CPUs without native bf16.
    assert _settings().reranker_dtype == "float32"
    clean_env.setenv("DEPLOY_MODE", "aws")
    assert _settings().reranker_dtype == "float32"


def test_reranker_dtype_override_and_validation(clean_env):
    clean_env.setenv("RERANKER_DTYPE", "auto")
    assert _settings().reranker_dtype == "auto"
    clean_env.setenv("RERANKER_DTYPE", "fp16")  # not an allowed value -> raises
    with pytest.raises(ValueError):
        _settings()


def test_build_storage_local(clean_env):
    s = _settings()
    st = build_storage(s)
    assert isinstance(st, LocalStorage)


def test_build_storage_s3(clean_env):
    clean_env.setenv("DEPLOY_MODE", "aws")
    clean_env.setenv("RAGINFER_ARTIFACT_BUCKET", "my-bucket")
    clean_env.setenv("RAGINFER_ARTIFACT_PREFIX", "raginfer")
    st = build_storage(_settings())
    assert isinstance(st, S3Storage)
    assert st.bucket == "my-bucket" and st.prefix == "raginfer"


def test_build_storage_s3_requires_bucket(clean_env):
    clean_env.setenv("DEPLOY_MODE", "aws")  # backend=s3 but no bucket
    with pytest.raises(ValueError, match="RAGINFER_ARTIFACT_BUCKET"):
        build_storage(_settings())


def test_local_storage_roundtrip(tmp_path):
    st = LocalStorage(root=str(tmp_path))
    assert st.exists("a/b.bin") is False
    st.write_bytes("a/b.bin", b"hello")
    assert st.exists("a/b.bin") is True
    assert st.read_bytes("a/b.bin") == b"hello"
