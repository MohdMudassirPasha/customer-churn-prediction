"""Logging configuration."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _coerce_level(level: int | str | None) -> int:
    """Convert a logging level name or value into a logging module constant."""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    normalized = level.strip().upper()
    return getattr(logging, normalized, logging.INFO)


def setup_logging(
    level: int | str | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure root logging for command-line, API, and training processes."""
    root_logger = logging.getLogger()
    root_logger.setLevel(_coerce_level(level))

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
