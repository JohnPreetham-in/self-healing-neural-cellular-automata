"""Composable neural cellular automata model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.model.alive_mask import AliveMask
from src.model.perception import Perception
from src.model.state import clone_state, validate_state_tensor
from src.model.stochastic import StochasticUpdate
from src.model.update_rule import UpdateRule


class NeuralCellularAutomaton(nn.Module):
    """Evolve a batched cellular state through shared local neural updates."""

    def __init__(
        self,
        *,
        channels: int,
        grid_size: int,
        update_rate: float,
        alive_threshold: float,
        visible_channels: Sequence[int] = (0, 1, 2),
        update_hidden_channels: int | None = None,
        alive_neighborhood_size: int = 3,
    ) -> None:
        """Initialize an NCA model from experiment configuration values.

        Args:
            channels: Number of state channels.
            grid_size: Expected square state-grid size.
            update_rate: Per-cell probability of applying an update.
            alive_threshold: Visible-activity threshold for alive masking.
            visible_channels: State channels with visible tissue semantics.
            update_hidden_channels: Width of the pointwise update network.
            alive_neighborhood_size: Odd neighborhood size for alive masking.
        """
        super().__init__()
        _validate_positive_integer(channels, "channels")
        _validate_positive_integer(grid_size, "grid_size")

        self.channels = channels
        self.grid_size = grid_size
        self.perception = Perception(channels)
        self.update_rule = UpdateRule(
            perception_channels=self.perception.output_channels,
            state_channels=channels,
            hidden_channels=update_hidden_channels,
        )
        self.stochastic_update = StochasticUpdate(update_rate)
        self.alive_mask = AliveMask(
            channels=channels,
            threshold=alive_threshold,
            visible_channels=visible_channels,
            neighborhood_size=alive_neighborhood_size,
        )

    def forward(
        self,
        state: torch.Tensor,
        steps: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Evolve an automata state for one or more local-update steps.

        Args:
            state: Float32 state tensor in ``(B, C, H, W)`` layout.
            steps: Number of NCA evolution steps to apply.
            generator: Optional random generator for asynchronous update masks.

        Returns:
            The evolved state tensor with the same shape as ``state``.

        Raises:
            ValueError: If ``steps`` is not a positive integer or state is invalid.
        """
        validate_state_tensor(state, channels=self.channels, grid_size=self.grid_size)
        _validate_positive_integer(steps, "steps")

        evolved_state = clone_state(state)
        for _ in range(steps):
            evolved_state = self._step(evolved_state, generator)
        return evolved_state

    def _step(
        self,
        state: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        perception = self.perception(state)
        state_delta = self.update_rule(perception)
        state_delta = self.stochastic_update(state_delta, generator)
        updated_state = state + state_delta
        return self.alive_mask(updated_state)


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
