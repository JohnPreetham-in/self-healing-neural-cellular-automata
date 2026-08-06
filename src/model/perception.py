"""Fixed local perception filters for neural cellular automata."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from src.model.state import validate_state_tensor

_PERCEPTION_KERNEL_SIZE = 3


class Perception(nn.Module):
    """Apply fixed identity and Sobel filters independently per state channel."""

    def __init__(self, channels: int) -> None:
        """Initialize the fixed local perception module.

        Args:
            channels: Number of channels in the automata state.

        Raises:
            ValueError: If ``channels`` is not positive.
        """
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer.")

        self.channels = channels
        self.register_buffer("kernels", _create_perception_kernels(), persistent=True)

    @property
    def output_channels(self) -> int:
        """Return the number of features produced by the perception filters."""
        return self.channels * self.kernels.shape[0]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute local perception features for every state channel.

        Args:
            state: Float32 state tensor with shape ``(B, C, H, W)``.

        Returns:
            Perception tensor with shape ``(B, 3 * C, H, W)``.
        """
        validate_state_tensor(state, channels=self.channels)
        kernels = self.kernels.repeat(self.channels, 1, 1, 1)
        return functional.conv2d(
            state,
            kernels,
            padding=_PERCEPTION_KERNEL_SIZE // 2,
            groups=self.channels,
        )


def _create_perception_kernels() -> torch.Tensor:
    identity = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
    sobel_x = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    sobel_y = ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))
    return torch.tensor((identity, sobel_x, sobel_y), dtype=torch.float32).unsqueeze(1)
