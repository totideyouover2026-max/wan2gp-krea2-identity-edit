"""Geometry and protected-source compositing for Krea 2 Registered Outpaint.

Adapted for WanGP from yijunwang2/krea2-outpaint ``outpaint.py`` (Apache-2.0).
The WanGP integration supplies the canvas rectangle; no model weights are bundled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


SOURCE_MAX_EDGE = 384
DEFAULT_SEAM_PX = 32


@dataclass(frozen=True)
class RegisteredSource:
    condition: Image.Image
    placed_source: Image.Image
    canvas_size: tuple[int, int]
    bbox: tuple[int, int, int, int]
    seam_px: int = DEFAULT_SEAM_PX

    @property
    def bbox_normalized(self) -> list[float]:
        width, height = self.canvas_size
        x0, y0, x1, y1 = self.bbox
        return [x0 / width, y0 / height, x1 / width, y1 / height]


def _resize_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    if max(image.size) <= max_edge:
        return image.copy()
    scale = max_edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def align_down(value: int, alignment: int = 16) -> int:
    return max(alignment, int(value) // alignment * alignment)


def canvas_bbox(
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
    outpainting_dims,
) -> tuple[int, int, int, int]:
    """Resolve WanGP percentage margins into an aspect-preserving source box."""
    source_width, source_height = map(int, source_size)
    canvas_width, canvas_height = map(int, canvas_size)
    if outpainting_dims is None or len(outpainting_dims) != 4:
        raise ValueError("Registered Outpaint requires spatial outpainting margins")
    top, bottom, left, right = [max(0.0, float(v)) for v in outpainting_dims]
    scale = min(
        canvas_width / (source_width * (1 + (left + right) / 100)),
        canvas_height / (source_height * (1 + (top + bottom) / 100)),
    )
    box_width = align_down(round(source_width * scale))
    box_height = align_down(round(source_height * scale))
    x0 = round(canvas_width * left / (100 + left + right)) if left + right else 0
    y0 = round(canvas_height * top / (100 + top + bottom)) if top + bottom else 0
    x0 = max(0, min(canvas_width - box_width, x0))
    y0 = max(0, min(canvas_height - box_height, y0))
    return x0, y0, x0 + box_width, y0 + box_height


def prepare_source(
    source: Image.Image,
    canvas_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    source_max_edge: int = SOURCE_MAX_EDGE,
    seam_px: int = DEFAULT_SEAM_PX,
) -> RegisteredSource:
    width, height = canvas_size
    x0, y0, x1, y1 = bbox
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Outpaint canvas dimensions must be multiples of 16")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Source bbox is outside the canvas: {bbox}")
    source = source.convert("RGB")
    placed = source.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    return RegisteredSource(
        condition=_resize_max_edge(placed, source_max_edge),
        placed_source=placed,
        canvas_size=canvas_size,
        bbox=bbox,
        seam_px=max(0, int(seam_px)),
    )


def composite(generated: Image.Image, prepared: RegisteredSource) -> Image.Image:
    """Restore source pixels, feathering only the inward edge of its rectangle."""
    generated = generated.convert("RGB")
    if generated.size != prepared.canvas_size:
        raise ValueError("Generated image size does not match the outpaint canvas")
    width, height = prepared.placed_source.size
    if prepared.seam_px <= 0:
        result = generated.copy()
        result.paste(prepared.placed_source, prepared.bbox[:2])
        return result
    yy, xx = np.mgrid[:height, :width]
    edge_distance = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy))
    alpha = np.clip(edge_distance / prepared.seam_px, 0.0, 1.0)
    alpha_image = Image.fromarray((alpha * 255).astype(np.uint8))
    result = generated.copy()
    result.paste(prepared.placed_source, prepared.bbox[:2], alpha_image)
    return result
