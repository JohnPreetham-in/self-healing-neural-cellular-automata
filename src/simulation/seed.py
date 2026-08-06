"""Canonical automata state initialization and seed generation utilities."""

from __future__ import annotations

from typing import Literal

import torch

from src.data.preprocess import validate_tensor

DEFAULT_GRID_SIZE = 32
DEFAULT_STATE_CHANNELS = 16
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED_VALUE = 1.0

SeedKind = Literal["center", "bottom", "custom", "random"]


def initialize_state(
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    channels: int = DEFAULT_STATE_CHANNELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a zero-valued batched automata state.

    Args:
        grid_size: Square grid height and width.
        channels: Number of state channels.
        batch_size: Number of independent states.
        dtype: Floating-point dtype for the state.
        device: Optional device for the state tensor.

    Returns:
        A zero-valued state tensor with shape ``(B, C, H, W)``.

    Raises:
        ValueError: If a dimension is not a positive integer.
        TypeError: If ``dtype`` is not floating point.
    """
    _validate_positive_integer(grid_size, "grid_size")
    _validate_positive_integer(channels, "channels")
    _validate_positive_integer(batch_size, "batch_size")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("State dtype must be a floating-point torch dtype.")

    return torch.zeros(
        (batch_size, channels, grid_size, grid_size),
        dtype=dtype,
        device=device,
    )


def create_seed_state(
    *,
    seed_kind: SeedKind = "center",
    coordinate: tuple[int, int] | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    channels: int = DEFAULT_STATE_CHANNELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed_value: float = DEFAULT_SEED_VALUE,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    random_seed: int | None = None,
) -> torch.Tensor:
    """Create a zero state with one identically initialized seed cell per batch.

    The selected seed cell receives ``seed_value`` in every state channel,
    including visible and hidden channels. All other cells remain zero.

    Args:
        seed_kind: Seed location policy: ``center``, ``bottom``, ``custom``, or
            ``random``.
        coordinate: Required ``(row, column)`` for a custom seed.
        grid_size: Square grid height and width.
        channels: Number of state channels.
        batch_size: Number of independent states.
        seed_value: Value assigned to every channel of the seed cell.
        dtype: Floating-point dtype for the state.
        device: Optional device for the state tensor.
        random_seed: Optional local seed used only for ``random`` placement.

    Returns:
        A seeded state tensor with shape ``(B, C, H, W)``.

    Raises:
        ValueError: If seed arguments or coordinates are invalid.
        TypeError: If ``seed_value`` is not numeric.
    """
    state = initialize_state(
        grid_size=grid_size,
        channels=channels,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    row, column = _resolve_seed_coordinate(
        seed_kind,
        coordinate,
        grid_size,
        random_seed,
    )
    if not isinstance(seed_value, int | float) or isinstance(seed_value, bool):
        raise TypeError("seed_value must be a numeric value.")

    state[:, :, row, column] = seed_value
    return state


def validate_state_tensor(
    state: torch.Tensor,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    channels: int = DEFAULT_STATE_CHANNELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> None:
    """Validate a canonical automata state tensor.

    Args:
        state: State tensor in ``(B, C, H, W)`` layout.
        grid_size: Expected square grid size.
        channels: Expected state channel count.
        batch_size: Expected batch size.
        dtype: Expected floating-point dtype.
        device: Optional expected device.

    Raises:
        TypeError: If ``state`` has an invalid dtype.
        ValueError: If rank, shape, or device requirements fail.
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


def _resolve_seed_coordinate(
    seed_kind: SeedKind,
    coordinate: tuple[int, int] | None,
    grid_size: int,
    random_seed: int | None,
) -> tuple[int, int]:
    if seed_kind == "center":
        return grid_size // 2, grid_size // 2
    if seed_kind == "bottom":
        return grid_size - 1, grid_size // 2
    if seed_kind == "custom":
        if coordinate is None:
            raise ValueError("coordinate is required when seed_kind is 'custom'.")
        return _validate_coordinate(coordinate, grid_size)
    if seed_kind == "random":
        return _random_coordinate(grid_size, random_seed)
    raise ValueError(
        "seed_kind must be one of: 'center', 'bottom', 'custom', or 'random'."
    )


def _random_coordinate(grid_size: int, random_seed: int | None) -> tuple[int, int]:
    if random_seed is not None:
        _validate_positive_integer(random_seed, "random_seed", allow_zero=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
    else:
        generator = None

    coordinate = torch.randint(grid_size, (2,), generator=generator)
    return int(coordinate[0].item()), int(coordinate[1].item())


def _validate_coordinate(
    coordinate: tuple[int, int],
    grid_size: int,
) -> tuple[int, int]:
    if not isinstance(coordinate, tuple | list):
        raise TypeError("coordinate must be a (row, column) pair.")
    if len(coordinate) != 2:
        raise ValueError("coordinate must contain exactly (row, column).")
    row, column = coordinate
    if not isinstance(row, int) or isinstance(row, bool):
        raise TypeError("coordinate row must be an integer.")
    if not isinstance(column, int) or isinstance(column, bool):
        raise TypeError("coordinate column must be an integer.")
    if not 0 <= row < grid_size or not 0 <= column < grid_size:
        raise ValueError(
            f"coordinate {(row, column)} must be within a {grid_size}x{grid_size} grid."
        )
    return row, column


def _validate_positive_integer(
    value: int,
    name: str,
    *,
    allow_zero: bool = False,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparison = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be {comparison}.")
