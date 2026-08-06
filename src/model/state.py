"""State validation and channel access helpers for NCA computation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.data.preprocess import validate_tensor

DEFAULT_VISIBLE_CHANNELS = (0, 1, 2)


def validate_state_tensor(
    state: torch.Tensor,
    *,
    channels: int,
    grid_size: int | None = None,
    batch_size: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> None:
    """Validate an automata state in canonical ``(B, C, H, W)`` layout.

    Args:
        state: State tensor to validate.
        channels: Expected state channel count.
        grid_size: Optional expected square spatial size.
        batch_size: Optional expected batch size.
        dtype: Expected state dtype.
        device: Optional expected device.

    Raises:
        TypeError: If the tensor or dtype is invalid.
        ValueError: If state shape or device requirements are not satisfied.
    """
    validate_tensor(
        state,
        name="automata state",
        batch_size=batch_size,
        channels=channels,
        height=grid_size,
        width=grid_size,
        dtype=dtype,
        device=device,
    )


def extract_visible_channels(
    state: torch.Tensor,
    visible_channels: Sequence[int] = DEFAULT_VISIBLE_CHANNELS,
) -> torch.Tensor:
    """Return the configured visible channels from a state tensor.

    Args:
        state: State tensor in ``(B, C, H, W)`` layout.
        visible_channels: State-channel indices considered visible.

    Returns:
        A tensor containing the selected visible channels.

    Raises:
        ValueError: If channel indices are empty, duplicated, or out of range.
    """
    validate_tensor(state, name="automata state")
    channel_indices = _validate_channel_indices(visible_channels, state.shape[1])
    return state[:, channel_indices, :, :]


def extract_hidden_channels(
    state: torch.Tensor,
    visible_channels: Sequence[int] = DEFAULT_VISIBLE_CHANNELS,
) -> torch.Tensor:
    """Return the non-visible channels from a state tensor.

    Args:
        state: State tensor in ``(B, C, H, W)`` layout.
        visible_channels: State-channel indices considered visible.

    Returns:
        A tensor containing the remaining hidden channels.

    Raises:
        ValueError: If channel indices are empty, duplicated, or out of range.
    """
    validate_tensor(state, name="automata state")
    visible_indices = set(_validate_channel_indices(visible_channels, state.shape[1]))
    hidden_indices = [
        channel_index
        for channel_index in range(state.shape[1])
        if channel_index not in visible_indices
    ]
    return state[:, hidden_indices, :, :]


def clone_state(state: torch.Tensor) -> torch.Tensor:
    """Return a differentiable copy of an automata state.

    Args:
        state: State tensor to copy.

    Returns:
        A cloned state tensor that remains connected to the autograd graph.
    """
    validate_tensor(state, name="automata state")
    return state.clone()


def _validate_channel_indices(
    channel_indices: Sequence[int],
    channel_count: int,
) -> tuple[int, ...]:
    indices = tuple(channel_indices)
    if not indices:
        raise ValueError("At least one visible channel index is required.")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise TypeError("Channel indices must be integers.")
    if len(set(indices)) != len(indices):
        raise ValueError("Channel indices must be unique.")
    if any(index < 0 or index >= channel_count for index in indices):
        raise ValueError(
            f"Channel indices must be within the state range [0, {channel_count})."
        )
    return indices
