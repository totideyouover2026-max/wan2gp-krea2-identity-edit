"""UI-safe helpers for describing uploaded image dimensions."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from PIL import Image


def _positive_dimensions(width: Any, height: Any) -> tuple[int, int] | None:
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def image_dimensions(value: Any) -> tuple[int, int] | None:
    """Return ``(width, height)`` for common Gradio image value shapes."""

    if value is None:
        return None

    if isinstance(value, Image.Image):
        return _positive_dimensions(*value.size)

    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if not path.is_file():
            return None
        try:
            with Image.open(path) as image:
                return _positive_dimensions(*image.size)
        except (OSError, ValueError):
            return None

    # Gradio image arrays use HWC. Tensors reaching this UI are normally CHW.
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            shape = tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            shape = ()
        if len(shape) == 2:
            return _positive_dimensions(shape[1], shape[0])
        if len(shape) >= 3:
            if shape[-1] in (1, 3, 4):
                return _positive_dimensions(shape[-2], shape[-3])
            if shape[-3] in (1, 3, 4):
                return _positive_dimensions(shape[-1], shape[-2])

    if isinstance(value, dict):
        for key in (
            "path",
            "name",
            "image",
            "background",
            "composite",
            "value",
        ):
            if key in value:
                dimensions = image_dimensions(value[key])
                if dimensions is not None:
                    return dimensions
        return None

    path = getattr(value, "path", None)
    if path is not None:
        dimensions = image_dimensions(path)
        if dimensions is not None:
            return dimensions

    image = getattr(value, "image", None)
    if image is not None:
        dimensions = image_dimensions(image)
        if dimensions is not None:
            return dimensions

    # A Gradio Gallery entry is normally ``(image, caption)``.
    if isinstance(value, (list, tuple)) and value:
        return image_dimensions(value[0])

    return None


def gallery_dimensions(value: Any) -> list[tuple[int, int]]:
    """Return dimensions for every valid image in a Gradio gallery value."""

    if value is None:
        return []

    root = getattr(value, "root", None)
    if root is not None:
        value = root
    elif isinstance(value, dict) and "root" in value:
        value = value["root"]

    if not isinstance(value, (list, tuple)):
        value = [value]

    dimensions = []
    for entry in value:
        size = image_dimensions(entry)
        if size is not None:
            dimensions.append(size)
    return dimensions


def aspect_ratio(width: int, height: int) -> str:
    """Return an exact, reduced whole-number aspect ratio."""

    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def dimensions_text(width: int, height: int) -> str:
    return f"{width}x{height}, {aspect_ratio(width, height)}"


def image_input_label(base_label: str, value: Any) -> str:
    """Append dimensions to a single-image component label when available."""

    size = image_dimensions(value)
    if size is None:
        return base_label
    return f"{base_label} ({dimensions_text(*size)})"


def gallery_input_label(base_label: str, value: Any) -> str:
    """Append compact per-image dimensions to a gallery component label."""

    sizes = gallery_dimensions(value)
    if not sizes:
        return base_label
    if len(sizes) == 1:
        return f"{base_label} ({dimensions_text(*sizes[0])})"
    entries = [
        f"{index}: {dimensions_text(*size)}"
        for index, size in enumerate(sizes, start=1)
    ]
    return f"{base_label} ({'; '.join(entries)})"
