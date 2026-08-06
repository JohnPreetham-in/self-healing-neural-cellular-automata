"""Filesystem and YAML serialization helpers for experiment infrastructure."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def safe_path(path: str | Path, base_directory: str | Path | None = None) -> Path:
    """Return a normalized path, optionally constrained to a base directory.

    Args:
        path: Path to normalize.
        base_directory: Optional directory that must contain the returned path.

    Returns:
        A normalized absolute path.

    Raises:
        ValueError: If ``path`` is outside ``base_directory``.
    """
    resolved_path = Path(path).expanduser().resolve(strict=False)
    if base_directory is None:
        return resolved_path

    resolved_base = Path(base_directory).expanduser().resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_base)
    except ValueError as error:
        raise ValueError(
            f"Path '{resolved_path}' must be inside '{resolved_base}'."
        ) from error
    return resolved_path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and its parents when needed.

    Args:
        path: Directory path to create.

    Returns:
        The normalized directory path.

    Raises:
        NotADirectoryError: If the path exists and is not a directory.
    """
    directory = safe_path(path)
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory path, got '{directory}'.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_directory(path: str | Path) -> Path:
    """Create a directory and return its normalized path.

    This alias keeps call sites expressive while centralizing directory behavior.

    Args:
        path: Directory path to create.

    Returns:
        The normalized directory path.
    """
    return ensure_directory(path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk using PyYAML's safe loader.

    Args:
        path: YAML file to read.

    Returns:
        Parsed YAML mapping. Empty files produce an empty mapping.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        IsADirectoryError: If the path identifies a directory.
        ValueError: If the YAML content is invalid or is not a mapping.
    """
    yaml_path = safe_path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file does not exist: '{yaml_path}'.")
    if yaml_path.is_dir():
        raise IsADirectoryError(f"Expected a YAML file, got directory '{yaml_path}'.")

    try:
        with yaml_path.open("r", encoding="utf-8") as file_handle:
            content = yaml.safe_load(file_handle)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in '{yaml_path}': {error}") from error

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"YAML root in '{yaml_path}' must be a mapping.")
    return content


def save_yaml(data: Mapping[str, Any], path: str | Path) -> Path:
    """Safely serialize a mapping to a YAML file.

    Args:
        data: Mapping to serialize.
        path: Destination YAML file.

    Returns:
        The normalized destination path.

    Raises:
        ValueError: If the destination does not use a YAML extension.
    """
    yaml_path = safe_path(path)
    if yaml_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a .yaml or .yml path, got '{yaml_path}'.")

    ensure_directory(yaml_path.parent)
    with yaml_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(
            dict(data),
            file_handle,
            default_flow_style=False,
            sort_keys=False,
        )
    return yaml_path
