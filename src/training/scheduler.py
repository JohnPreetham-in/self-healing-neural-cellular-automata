"""Optional learning-rate scheduler construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

Scheduler = torch.optim.lr_scheduler.LRScheduler


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any] | None,
) -> Scheduler | None:
    """Construct an optional scheduler from YAML-derived configuration.

    Args:
        optimizer: Optimizer managed by the scheduler.
        config: Scheduler configuration. None or enabled false disables
            scheduling. A step scheduler requires name, step_size, and gamma.

    Returns:
        A configured scheduler, or None when scheduling is disabled.

    Raises:
        KeyError: If an enabled scheduler omits required settings.
        ValueError: If scheduler settings are invalid or unsupported.
    """
    if config is None or not bool(config.get("enabled", False)):
        return None

    required_fields = ("name", "step_size", "gamma")
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise KeyError(
            "Scheduler configuration is missing field(s): "
            f"{', '.join(missing_fields)}."
        )

    scheduler_name = str(config["name"]).lower()
    scheduler_builders = {"step": torch.optim.lr_scheduler.StepLR}
    try:
        scheduler_class = scheduler_builders[scheduler_name]
    except KeyError as error:
        supported_schedulers = ", ".join(sorted(scheduler_builders))
        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. Supported schedulers: "
            f"{supported_schedulers}."
        ) from error

    step_size = _validate_positive_integer(config["step_size"], "step_size")
    gamma = _validate_positive_number(config["gamma"], "gamma")
    return scheduler_class(optimizer, step_size=step_size, gamma=gamma)


def _validate_positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_positive_number(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return float(value)
