"""Alive-cell masking for stable neural cellular automata evolution."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional

from src.model.state import extract_visible_channels, validate_state_tensor


class AliveMask(nn.Module):
    """Keep cells alive when visible activity exists in their local neighborhood."""

    def __init__(
        self,
        channels: int,
        threshold: float,
        visible_channels: Sequence[int] = (0, 1, 2),
        neighborhood_size: int = 3,
    ) -> None:
        """Initialize alive-mask behavior.

        Args:
            channels: Number of channels in the automata state.
            threshold: Minimum visible activity required for a live cell.
            visible_channels: Channels used to measure tissue activity.
            neighborhood_size: Odd square neighborhood used to propagate life.

        Raises:
            ValueError: If parameters are incompatible with local masking.
        """
        super().__init__()
        _validate_positive_integer(channels, "channels")
        if not isinstance(threshold, int | float) or isinstance(threshold, bool):
            raise TypeError("threshold must be numeric.")
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative.")
        _validate_neighborhood_size(neighborhood_size)

        visible_indices = tuple(visible_channels)
        _validate_visible_channels(visible_indices, channels)
        self.channels = channels
        self.threshold = float(threshold)
        self.visible_channels = visible_indices
        self.neighborhood_size = neighborhood_size

    def compute_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Compute a boolean alive mask with shape ``(B, 1, H, W)``.

        Args:
            state: Float32 state tensor in ``(B, C, H, W)`` layout.

        Returns:
            A boolean local-neighborhood alive mask.
        """
        validate_state_tensor(state, channels=self.channels)
        visible_state = extract_visible_channels(state, self.visible_channels)
        activity = visible_state.abs().amax(dim=1, keepdim=True)
        neighborhood_activity = functional.max_pool2d(
            activity,
            kernel_size=self.neighborhood_size,
            stride=1,
            padding=self.neighborhood_size // 2,
        )
        return neighborhood_activity > self.threshold

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the local alive mask without mutating the input state.

        Args:
            state: Float32 state tensor in ``(B, C, H, W)`` layout.

        Returns:
            State with inactive locations set to zero.
        """
        return state * self.compute_mask(state).to(dtype=state.dtype)


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_neighborhood_size(neighborhood_size: int) -> None:
    _validate_positive_integer(neighborhood_size, "neighborhood_size")
    if neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size must be odd.")


def _validate_visible_channels(
    visible_channels: tuple[int, ...],
    channels: int,
) -> None:
    if not visible_channels:
        raise ValueError("At least one visible channel is required.")
    if len(set(visible_channels)) != len(visible_channels):
        raise ValueError("visible_channels must be unique.")
    if any(
        not isinstance(channel, int)
        or isinstance(channel, bool)
        or channel < 0
        or channel >= channels
        for channel in visible_channels
    ):
        raise ValueError("visible_channels must contain valid state-channel indices.")
