"""Reusable structured project logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import ensure_directory, safe_path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a named project logger.

    A console handler is added once per logger. When ``log_file`` is supplied,
    one file handler is added for that normalized destination. Repeated calls do
    not duplicate existing handlers.

    Args:
        name: Logger name, usually the importing module's ``__name__``.
        level: Logging level accepted by ``logging.Logger.setLevel``.
        log_file: Optional file to receive the same formatted log records.

    Returns:
        A configured logger with console output and optional file output.

    Raises:
        ValueError: If ``name`` is empty or ``level`` is invalid.
    """
    if not name:
        raise ValueError("Logger name must be a non-empty string.")

    normalized_level = _normalize_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(normalized_level)
    logger.propagate = False

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
    _add_console_handler(logger, normalized_level, formatter)
    if log_file is not None:
        _add_file_handler(logger, normalized_level, formatter, log_file)
    return logger


def _normalize_level(level: int | str) -> int:
    if isinstance(level, str):
        normalized_level = logging.getLevelName(level.upper())
    else:
        normalized_level = level
    if not isinstance(normalized_level, int):
        raise ValueError(f"Invalid logging level: {level!r}.")
    return normalized_level


def _add_console_handler(
    logger: logging.Logger,
    level: int,
    formatter: logging.Formatter,
) -> None:
    for handler in logger.handlers:
        if getattr(handler, "_project_console_handler", False):
            handler.setLevel(level)
            handler.setFormatter(formatter)
            return

    handler = logging.StreamHandler()
    handler._project_console_handler = True
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _add_file_handler(
    logger: logging.Logger,
    level: int,
    formatter: logging.Formatter,
    log_file: str | Path,
) -> None:
    log_path = safe_path(log_file)
    ensure_directory(log_path.parent)
    for handler in logger.handlers:
        if getattr(handler, "_project_log_path", None) == log_path:
            handler.setLevel(level)
            handler.setFormatter(formatter)
            return

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler._project_log_path = log_path
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
