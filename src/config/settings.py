"""Typed settings for RagInference — one switch for local ↔ AWS deployment.

The service is stateless HTTP; its only real data surface is *model weights* and
*logs*. This module centralizes the environment into a typed :class:`Settings`
object driven by a single ``DEPLOY_MODE`` (``local`` default | ``aws``):

* ``DEPLOY_MODE`` only supplies **defaults**. Any explicit env var still wins, so
  today's local invocation and the container invocation behave identically when
  the mode is unset. See :meth:`Settings._apply_mode_defaults`.
* All the legacy env names keep working (``RAGINFER_HOST``, ``RERANKER``,
  ``EMBEDDER``, ``MODEL_SOURCE``, ``RAGINFER_WARMUP``, ``LOG_LEVEL`` …) via
  per-field aliases — nothing in ``infra/`` or the conda workflow breaks.

Per-mode defaults applied when *not* set explicitly:

============  =================  =================
field         local (default)    aws
============  =================  =================
host          ``127.0.0.1``      ``0.0.0.0``
log_format    ``text``           ``json``
log_to_file   ``True``           ``False`` (stdout only, for CloudWatch)
artifact_backend ``local``       ``s3``
============  =================  =================
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DeployMode = Literal["local", "aws"]


class Settings(BaseSettings):
    """Resolved service configuration. Construct via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # allow constructing by field name in tests, e.g. Settings(host="x")
        populate_by_name=True,
    )

    # -- top-level switch ---------------------------------------------------- #
    deploy_mode: DeployMode = Field(
        "local",
        validation_alias=AliasChoices("DEPLOY_MODE", "RAGINFER_DEPLOY_MODE"),
    )

    # -- network ------------------------------------------------------------- #
    host: str = Field("127.0.0.1", validation_alias=AliasChoices("RAGINFER_HOST"))
    port: int = Field(8799, validation_alias=AliasChoices("RAGINFER_PORT"))

    # -- models -------------------------------------------------------------- #
    embedder: str = Field("bge", validation_alias=AliasChoices("EMBEDDER", "RAGINFER_EMBEDDER"))
    reranker: str = Field("jina", validation_alias=AliasChoices("RERANKER", "RAGINFER_RERANKER"))
    warmup: bool = Field(True, validation_alias=AliasChoices("RAGINFER_WARMUP"))

    # -- weight delivery (mirrors infra/entrypoint.sh) ----------------------- #
    model_source: Literal["baked", "hf", "s3"] = Field(
        "baked", validation_alias=AliasChoices("MODEL_SOURCE")
    )
    model_s3_uri: Optional[str] = Field(None, validation_alias=AliasChoices("MODEL_S3_URI"))
    hf_home: Optional[str] = Field(None, validation_alias=AliasChoices("HF_HOME"))

    # -- artifact storage seam (future blob IO; weights use model_source) ---- #
    # None here means "derive from deploy_mode" in _apply_mode_defaults.
    artifact_backend: Optional[Literal["local", "s3"]] = Field(
        None, validation_alias=AliasChoices("RAGINFER_ARTIFACT_BACKEND")
    )
    artifact_root: str = Field("data", validation_alias=AliasChoices("RAGINFER_ARTIFACT_ROOT"))
    artifact_bucket: Optional[str] = Field(None, validation_alias=AliasChoices("RAGINFER_ARTIFACT_BUCKET"))
    artifact_prefix: str = Field("", validation_alias=AliasChoices("RAGINFER_ARTIFACT_PREFIX"))

    # -- logging ------------------------------------------------------------- #
    log_level: str = Field("INFO", validation_alias=AliasChoices("LOG_LEVEL", "RAGINFER_LOG_LEVEL"))
    log_format: Literal["text", "json"] = Field(
        "text", validation_alias=AliasChoices("RAGINFER_LOG_FORMAT")
    )
    log_to_file: bool = Field(True, validation_alias=AliasChoices("RAGINFER_LOG_TO_FILE"))

    # -- aws ----------------------------------------------------------------- #
    aws_region: Optional[str] = Field(
        None, validation_alias=AliasChoices("AWS_REGION", "AWS_DEFAULT_REGION", "RAGINFER_AWS_REGION")
    )

    @model_validator(mode="after")
    def _apply_mode_defaults(self) -> "Settings":
        """Fill per-mode defaults, but never override an explicitly-set value.

        ``model_fields_set`` holds the fields that came from the environment (or
        an explicit kwarg), so a value the operator set always survives.
        """
        explicit = self.model_fields_set
        if self.deploy_mode == "aws":
            if "host" not in explicit:
                self.host = "0.0.0.0"
            if "log_format" not in explicit:
                self.log_format = "json"
            if "log_to_file" not in explicit:
                self.log_to_file = False
            if self.artifact_backend is None:
                self.artifact_backend = "s3"
        else:  # local
            if self.artifact_backend is None:
                self.artifact_backend = "local"
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (built once from the env)."""
    return Settings()
