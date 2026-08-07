"""Reproducibility helpers for Python, NumPy, and PyTorch."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed supported random number generators for an experiment.

    Args:
        seed: Non-negative seed shared by Python, NumPy, and PyTorch.
        deterministic: Whether to enable PyTorch deterministic algorithms where
            supported. This can reduce performance and may still vary by platform.

    Raises:
        ValueError: If ``seed`` is negative or is not an integer.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
