"""RGB target image loading for neural cellular automata experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.data.preprocess import (
    image_to_tensor,
    move_to_device,
    resize_image,
    validate_target_tensor,
)

_SUPPORTED_TARGET_SUFFIXES = frozenset({".png"})


def load_target(
    path: str | Path,
    *,
    grid_size: int = 32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load a PNG target image as a normalized batched RGB tensor.

    The returned tensor has canonical target layout ``(1, 3, H, W)`` and
    float32 values in ``[0, 1]``.

    Args:
        path: PNG target image path.
        grid_size: Target height and width after resizing.
        device: Optional destination device for the returned tensor.

    Returns:
        A normalized target tensor with shape ``(1, 3, grid_size, grid_size)``.

    Raises:
        FileNotFoundError: If the target image does not exist.
        IsADirectoryError: If ``path`` identifies a directory.
        ValueError: If the format, dimensions, or image content is invalid.
    """
    target_path = _validate_target_path(path)
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer.")

    image = _load_rgb_image(target_path)
    resized_image = resize_image(image, grid_size)
    target_tensor = image_to_tensor(resized_image).unsqueeze(0)
    target_tensor = move_to_device(target_tensor, device)
    validate_target_tensor(target_tensor, grid_size=grid_size, device=device)
    return target_tensor


def _validate_target_path(path: str | Path) -> Path:
    target_path = Path(path).expanduser().resolve(strict=False)
    if not target_path.exists():
        raise FileNotFoundError(f"Target image does not exist: '{target_path}'.")
    if target_path.is_dir():
        raise IsADirectoryError(f"Expected a PNG image, got directory '{target_path}'.")
    if target_path.suffix.lower() not in _SUPPORTED_TARGET_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_TARGET_SUFFIXES))
        raise ValueError(
            f"Unsupported target image format '{target_path.suffix}'. "
            f"Supported format(s): {supported}."
        )
    return target_path


def _load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source_image:
            source_image.load()
            if source_image.format != "PNG":
                raise ValueError(
                    f"Target image '{path}' must contain PNG data, "
                    f"received '{source_image.format}'."
                )
            return source_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read target image '{path}': {error}") from error
