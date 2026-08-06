"""Device selection utilities for CPU, CUDA, and Apple MPS execution."""

from __future__ import annotations

import torch


def is_cuda_available() -> bool:
    """Return whether a CUDA-capable PyTorch device is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Return whether the Apple Metal Performance Shaders backend is available."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def get_device() -> torch.device:
    """Select the preferred available PyTorch device.

    CUDA is preferred when available, followed by Apple MPS and then CPU.

    Returns:
        The selected PyTorch device.
    """
    if is_cuda_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")
