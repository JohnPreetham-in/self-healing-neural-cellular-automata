"""Checkpoint serialization for training state and reproducibility metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.training.scheduler import Scheduler
from src.utils.io import ensure_directory, safe_path


@dataclass(frozen=True)
class CheckpointData:
    """Metadata restored from a saved training checkpoint."""

    epoch: int
    metadata: Mapping[str, Any]
    config: Mapping[str, Any] | None


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Scheduler | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Save model and optional training state to a checkpoint file.

    Args:
        path: Destination checkpoint path.
        model: Model whose parameters are saved.
        epoch: Completed training epoch.
        optimizer: Optional optimizer whose state is saved.
        scheduler: Optional scheduler whose state is saved.
        metadata: Optional run metadata such as loss or source revision.
        config: Optional resolved experiment configuration snapshot.

    Returns:
        Normalized path of the saved checkpoint.

    Raises:
        ValueError: If epoch is negative.
    """
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer.")

    checkpoint_path = safe_path(path)
    ensure_directory(checkpoint_path.parent)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metadata": _to_serializable_mapping(metadata),
        "config": _to_serializable_mapping(config) if config is not None else None,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Scheduler | None = None,
    map_location: torch.device | str | None = None,
) -> CheckpointData:
    """Load model and optional training state from a checkpoint file.

    Args:
        path: Checkpoint path to restore.
        model: Model that receives saved parameters.
        optimizer: Optional optimizer that receives saved state when present.
        scheduler: Optional scheduler that receives saved state when present.
        map_location: Optional device mapping passed to torch.load.

    Returns:
        Restored epoch, metadata, and configuration snapshot.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        ValueError: If the checkpoint structure is invalid or incompatible.
    """
    checkpoint_path = safe_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: '{checkpoint_path}'.")
    if checkpoint_path.is_dir():
        raise IsADirectoryError(f"Expected a checkpoint file, got '{checkpoint_path}'.")

    payload = _load_checkpoint_payload(checkpoint_path, map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")
    _require_checkpoint_fields(payload, ("model_state", "epoch", "metadata", "config"))

    model.load_state_dict(payload["model_state"])
    _load_optional_state(optimizer, payload, "optimizer_state")
    _load_optional_state(scheduler, payload, "scheduler_state")

    epoch = payload["epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("Checkpoint epoch must be a non-negative integer.")
    metadata = payload["metadata"]
    config = payload["config"]
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping.")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("Checkpoint config must be a mapping or None.")

    return CheckpointData(epoch=epoch, metadata=dict(metadata), config=config)


def _load_optional_state(
    component: Any,
    payload: Mapping[str, Any],
    state_key: str,
) -> None:
    if component is not None and state_key in payload:
        component.load_state_dict(payload[state_key])


def _load_checkpoint_payload(
    path: Path,
    map_location: torch.device | str | None,
) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _require_checkpoint_fields(
    payload: Mapping[str, Any],
    fields: tuple[str, ...],
) -> None:
    missing_fields = [field for field in fields if field not in payload]
    if missing_fields:
        raise ValueError(
            "Checkpoint is missing field(s): " + ", ".join(missing_fields) + "."
        )


def _to_serializable_mapping(
    mapping: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if mapping is None:
        return {}
    return {key: _to_serializable(value) for key, value in mapping.items()}


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value
