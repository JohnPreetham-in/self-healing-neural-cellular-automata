"""Composable training-loss construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class MeanSquaredErrorLoss(nn.Module):
    """Mean squared reconstruction loss for visible NCA state channels."""

    def __init__(self, reduction: str = "mean") -> None:
        """Initialize mean squared error loss.

        Args:
            reduction: Reduction mode accepted by torch.nn.MSELoss.
        """
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            prediction: Predicted visible-state tensor.
            target: Target tensor with the same shape as prediction.

        Returns:
            Scalar or unreduced MSE tensor according to the configured reduction.
        """
        return self.loss(prediction, target)


def create_loss(config: Mapping[str, Any]) -> nn.Module:
    """Construct a loss module from a YAML-derived configuration mapping.

    Args:
        config: Loss configuration containing name and optional reduction.

    Returns:
        A configured PyTorch loss module.

    Raises:
        KeyError: If name is absent.
        ValueError: If the requested loss is unsupported.
    """
    if "name" not in config:
        raise KeyError("Loss configuration requires a 'name' field.")

    loss_name = str(config["name"]).lower()
    loss_builders = {"mse": MeanSquaredErrorLoss}
    try:
        loss_class = loss_builders[loss_name]
    except KeyError as error:
        supported_losses = ", ".join(sorted(loss_builders))
        raise ValueError(
            f"Unsupported loss '{loss_name}'. Supported losses: {supported_losses}."
        ) from error

    reduction = str(config.get("reduction", "mean"))
    return loss_class(reduction=reduction)
