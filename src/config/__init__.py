"""Typed configuration for RagInference.

One :class:`Settings` object reads the service's environment (the same env vars
used today, plus a top-level ``DEPLOY_MODE``) and derives per-mode defaults so the
identical codebase runs unchanged locally and on AWS. See ``settings.py``.
"""
from src.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
