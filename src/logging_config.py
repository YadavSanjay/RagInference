"""Minimal logging setup for the ARagReranker service.

Loads ``configs/logging.yaml`` if present; otherwise falls back to a basic
stderr config. Logs go to **stderr** (never stdout) so the same module can be
embedded behind a stdio transport without corrupting a JSON-RPC stream — the
lesson learned in ARag's MCP server.
"""
from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "logging.yaml"


def configure_logging(config_path: str | Path | None = None, default_level: int = logging.INFO) -> None:
    config_path = Path(config_path) if config_path else _default_config_path()
    env_level = os.getenv("LOG_LEVEL")

    if config_path.exists() and yaml is not None:
        with open(config_path, "rt", encoding="utf8") as f:
            cfg = yaml.safe_load(f)
        # Ensure parent dirs for any file handler exist (RotatingFileHandler
        # won't create them and dictConfig would otherwise fail at import).
        for handler in (cfg.get("handlers") or {}).values():
            filename = handler.get("filename")
            if filename:
                Path(filename).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            stream=__import__("sys").stderr,
        )

    if env_level:
        try:
            logging.getLogger().setLevel(getattr(logging, env_level.upper()))
        except Exception:
            pass


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "")
