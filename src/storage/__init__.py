"""Artifact storage seam — local filesystem ↔ S3, selected by ``DEPLOY_MODE``.

This is the "local FS vs S3" adapter the deployment plan calls for. It is a
**forward-looking seam**: the ``/embed`` and ``/rerank`` request path does not do
any blob IO today, and **model weights deliberately do NOT go through here** —
they load from ``HF_HOME`` via the ``MODEL_SOURCE`` (baked/hf/s3) path, because
``from_pretrained`` needs a real directory. Use this for any *future* artifact IO
(cached embeddings, corpus dumps, warm caches) so there is one seam instead of
scattered ``open()`` / ``boto3`` calls.

    storage = build_storage()            # local or S3 per settings
    storage.write_bytes("cache/x.bin", data)
    if storage.exists("cache/x.bin"):
        data = storage.read_bytes("cache/x.bin")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from src.logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Storage(Protocol):
    """Minimal blob interface. Paths are backend-relative POSIX-style keys."""

    def read_bytes(self, path: str) -> bytes: ...
    def write_bytes(self, path: str, data: bytes) -> None: ...
    def exists(self, path: str) -> bool: ...


class LocalStorage:
    """Filesystem-backed storage rooted at ``root`` (default ``./data``)."""

    def __init__(self, root: str = "data") -> None:
        self.root = Path(root)

    def _p(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def read_bytes(self, path: str) -> bytes:
        return self._p(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def exists(self, path: str) -> bool:
        return self._p(path).exists()


class S3Storage:
    """S3-backed storage. The boto3 client is created lazily on first use, so
    constructing this (and ``build_storage`` in aws mode) needs no credentials."""

    def __init__(self, bucket: str, prefix: str = "", region: Optional[str] = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3  # lazy: only needed when S3 is actually used

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def _key(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self.prefix}/{path}" if self.prefix else path

    def read_bytes(self, path: str) -> bytes:
        resp = self.client.get_object(Bucket=self.bucket, Key=self._key(path))
        return resp["Body"].read()

    def write_bytes(self, path: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(path), Body=data)

    def exists(self, path: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(path))
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise


def build_storage(settings=None) -> Storage:
    """Construct the storage backend for the current deploy mode.

    ``local`` -> :class:`LocalStorage` under ``RAGINFER_ARTIFACT_ROOT``.
    ``s3``    -> :class:`S3Storage` on ``RAGINFER_ARTIFACT_BUCKET`` (required).
    """
    if settings is None:
        from src.config.settings import get_settings

        settings = get_settings()

    backend = settings.artifact_backend or ("s3" if settings.deploy_mode == "aws" else "local")
    if backend == "s3":
        if not settings.artifact_bucket:
            raise ValueError(
                "artifact_backend=s3 requires RAGINFER_ARTIFACT_BUCKET to be set"
            )
        logger.info("storage backend=s3 bucket=%s prefix=%r", settings.artifact_bucket, settings.artifact_prefix)
        return S3Storage(settings.artifact_bucket, settings.artifact_prefix, settings.aws_region)

    logger.info("storage backend=local root=%s", settings.artifact_root)
    return LocalStorage(settings.artifact_root)


__all__ = ["Storage", "LocalStorage", "S3Storage", "build_storage"]
