"""Optimizer construction for NCA training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch


def create_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Construct an optimizer from YAML-derived hyperparameters.

    Args:
        parameters: Trainable parameters to optimize.
        config: Optimizer configuration with name, learning_rate, and
            weight_decay fields.

    Returns:
        A configured PyTorch optimizer.

    Raises:
        KeyError: If a required optimizer field is absent.
        ValueError: If the optimizer name or hyperparameters are invalid.
    """
    required_fields = ("name", "learning_rate", "weight_decay")
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise KeyError(
            "Optimizer configuration is missing field(s): "
            f"{', '.join(missing_fields)}."
        )

    optimizer_name = str(config["name"]).lower()
    learning_rate = _validate_non_negative_number(
        config["learning_rate"],
        "learning_rate",
        allow_zero=False,
    )
    weight_decay = _validate_non_negative_number(
        config["weight_decay"],
        "weight_decay",
    )
    optimizer_builders = {"adam": torch.optim.Adam}
    try:
        optimizer_class = optimizer_builders[optimizer_name]
    except KeyError as error:
        supported_optimizers = ", ".join(sorted(optimizer_builders))
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Supported optimizers: "
            f"{supported_optimizers}."
        ) from error

    return optimizer_class(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _validate_non_negative_number(
    value: Any,
    name: str,
    *,
    allow_zero: bool = True,
) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    if value < 0.0 or (not allow_zero and value == 0.0):
        comparison = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be {comparison}.")
    return float(value)
