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


def _split_padding(
    total_padding: int,
    before_weight: float,
    after_weight: float,
) -> tuple[int, int]:
    """Split required blank pixels using WanGP's directional slider weights."""
    total_padding = max(0, int(total_padding))
    before_weight = max(0.0, float(before_weight))
    after_weight = max(0.0, float(after_weight))
    if total_padding == 0:
        return 0, 0
    if before_weight == after_weight:
        before = total_padding // 2
    elif before_weight == 0:
        before = 0
    elif after_weight == 0:
        before = total_padding
    else:
        before = round(
            total_padding * before_weight / (before_weight + after_weight)
        )
    before = max(0, min(total_padding, int(before)))
    return before, total_padding - before


def canvas_bbox(
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
    outpainting_dims,
    *,
    ratio_mode: bool = False,
) -> tuple[int, int, int, int]:
    """Resolve WanGP directions into an aspect-preserving one-pass source box.

    The final canvas already determines the amount of required expansion. In
    manual mode, slider values may select only one axis because the released
    adapter requires a two-pass plan for an interior source rectangle. In
    aspect-ratio mode WanGP treats all four values as placement weights and
    ignores the pair on the axis that does not need padding.
    """
    source_width, source_height = map(int, source_size)
    canvas_width, canvas_height = map(int, canvas_size)
    if min(source_width, source_height, canvas_width, canvas_height) <= 0:
        raise ValueError("Outpaint source and canvas dimensions must be positive")
    if outpainting_dims is None or len(outpainting_dims) != 4:
        raise ValueError("Registered Outpaint requires spatial outpainting margins")
    top, bottom, left, right = [max(0.0, float(v)) for v in outpainting_dims]
    horizontal_requested = left > 0 or right > 0
    vertical_requested = top > 0 or bottom > 0
    if horizontal_requested and vertical_requested and not ratio_mode:
        raise ValueError(
            "Registered Outpaint currently supports one expansion axis per "
            "pass. Select left/right or top/bottom; mixed horizontal and "
            "vertical margins require the pending two-pass implementation."
        )

    source_ratio = source_width / source_height
    canvas_ratio = canvas_width / canvas_height
    ratio_tolerance = 1.0 / max(canvas_width, canvas_height)
    if canvas_ratio > source_ratio + ratio_tolerance:
        if vertical_requested and not horizontal_requested and not ratio_mode:
            raise ValueError(
                "The selected canvas requires horizontal outpainting, but only "
                "top/bottom margins were selected."
            )
        box_height = canvas_height
        box_width = max(1, min(canvas_width, round(box_height * source_ratio)))
        x0, _right_padding = _split_padding(
            canvas_width - box_width, left, right
        )
        y0 = 0
    elif canvas_ratio < source_ratio - ratio_tolerance:
        if horizontal_requested and not vertical_requested and not ratio_mode:
            raise ValueError(
                "The selected canvas requires vertical outpainting, but only "
                "left/right margins were selected."
            )
        box_width = canvas_width
        box_height = max(1, min(canvas_height, round(box_width / source_ratio)))
        y0, _bottom_padding = _split_padding(
            canvas_height - box_height, top, bottom
        )
        x0 = 0
    else:
        raise ValueError(
            "Registered Outpaint requires a canvas aspect ratio different from "
            "the source for the current one-pass implementation."
        )

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
    source_ratio = source.width / source.height
    box_ratio = (x1 - x0) / (y1 - y0)
    tolerance = max(0.025, 2.0 / min(x1 - x0, y1 - y0))
    if abs(box_ratio / source_ratio - 1.0) > tolerance:
        raise ValueError("Source bbox must preserve the source image aspect ratio")
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
