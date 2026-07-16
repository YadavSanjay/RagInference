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


def _coerce_level(value: str | None, default_level: int) -> int:
    if not value:
        return default_level
    return getattr(logging, str(value).upper(), default_level)


def _configure_cloud_logging(level: int, as_json: bool) -> None:
    """Cloud sink: a single stream handler to **stdout** (CloudWatch captures it),
    no rotating file. JSON when requested, else plain text. Used in aws mode."""
    import sys

    if as_json:
        try:  # python-json-logger 3.x
            from pythonjsonlogger.json import JsonFormatter  # type: ignore
        except Exception:
            try:  # python-json-logger <3
                from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore
            except Exception:
                JsonFormatter = None  # type: ignore
        if JsonFormatter is not None:
            formatter: logging.Formatter = JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        else:  # graceful fallback if the lib is missing
            formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    # 'src.*' loggers must reach the root handler and not carry a stale file handler.
    src_logger = logging.getLogger("src")
    for h in list(src_logger.handlers):
        src_logger.removeHandler(h)
    src_logger.propagate = True


def configure_logging(
    config_path: str | Path | None = None,
    default_level: int = logging.INFO,
    settings=None,
) -> None:
    if settings is None:
        try:
            from src.config.settings import get_settings

            settings = get_settings()
        except Exception:  # pragma: no cover - config optional/at-worst legacy path
            settings = None

    env_level = os.getenv("LOG_LEVEL") or (getattr(settings, "log_level", None) if settings else None)

    # Cloud path: JSON (or plain) to stdout, no file handler. Triggered by aws
    # mode's derived defaults (log_to_file=False / log_format=json) or explicit env.
    if settings is not None and (not settings.log_to_file or settings.log_format == "json"):
        _configure_cloud_logging(_coerce_level(env_level, default_level), as_json=(settings.log_format == "json"))
        return

    # --- Local path (unchanged): configs/logging.yaml -> stderr + rotating file.
    config_path = Path(config_path) if config_path else _default_config_path()

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
