"""Stochastic asynchronous state-update masking."""

from __future__ import annotations

import torch
from torch import nn

from src.data.preprocess import validate_tensor


class StochasticUpdate(nn.Module):
    """Apply a shared per-cell Bernoulli mask to proposed state deltas."""

    def __init__(self, update_rate: float) -> None:
        """Initialize asynchronous update masking.

        Args:
            update_rate: Probability that an individual cell applies its delta.

        Raises:
            ValueError: If ``update_rate`` is outside ``[0, 1]``.
        """
        super().__init__()
        if (
            not isinstance(update_rate, int | float)
            or isinstance(update_rate, bool)
            or not 0.0 <= update_rate <= 1.0
        ):
            raise ValueError("update_rate must be a number between 0 and 1.")
        self.update_rate = float(update_rate)

    def forward(
        self,
        state_delta: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Mask proposed deltas so a subset of cells updates asynchronously.

        Args:
            state_delta: Float32 delta tensor in ``(B, C, H, W)`` layout.
            generator: Optional generator compatible with the delta device.

        Returns:
            A delta tensor with the same shape and dtype as ``state_delta``.
        """
        validate_tensor(state_delta, name="state delta")
        if self.update_rate == 0.0:
            return torch.zeros_like(state_delta)
        if self.update_rate == 1.0:
            return state_delta

        mask_shape = (
            state_delta.shape[0],
            1,
            state_delta.shape[2],
            state_delta.shape[3],
        )
        update_mask = torch.rand(
            mask_shape,
            dtype=state_delta.dtype,
            device=state_delta.device,
            generator=generator,
        ) <= self.update_rate
        return state_delta * update_mask.to(dtype=state_delta.dtype)
