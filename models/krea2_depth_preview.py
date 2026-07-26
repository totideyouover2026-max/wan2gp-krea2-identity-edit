"""Preview helpers for Krea 2 custom depth masks.

These functions intentionally keep preview work separate from inference.  The
alignment preview never changes the Control Image, and effective depth is
estimated from the complete Control Image before the uploaded mask is applied.
"""

from __future__ import annotations

import gc
import os

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .krea2_custom_depth_mask import load_custom_depth_mask


_PREVIEW_MAX_SIDE = 1024
_MINIMUM_DEPTH_VRAM_BYTES = 3 * 1024**3


def _load_image(source, label: str) -> Image.Image:
    if source is None:
        raise ValueError(f"{label} was not provided")
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source.copy()).convert("RGB")
    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert("RGB")
    if isinstance(source, dict):
        for key in ("composite", "background", "image"):
            if source.get(key) is not None:
                return _load_image(source[key], label)
        raise ValueError(f"{label} does not contain an image")
    if not isinstance(source, (str, os.PathLike)) and hasattr(source, "name"):
        source = source.name
    try:
        path = os.fspath(source)
    except TypeError as exc:
        raise TypeError(f"{label} must be an image") from exc
    try:
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a readable image") from exc


def _validate_mask_ratio(mask: Image.Image, control: Image.Image) -> None:
    mask_ratio = mask.width / mask.height
    control_ratio = control.width / control.height
    if abs(mask_ratio / control_ratio - 1.0) > 0.02:
        raise ValueError(
            "Custom depth mask aspect ratio must match the Control Image "
            f"(mask {mask.width}x{mask.height}, control "
            f"{control.width}x{control.height})"
        )


def prepare_preview_mask(mask_source, control_size, *, invert=False, feather_px=0):
    """Return hard and feathered L masks aligned to a control image."""
    control = Image.new("RGB", tuple(control_size))
    mask, channel = load_custom_depth_mask(mask_source, invert=bool(invert))
    _validate_mask_ratio(mask, control)
    mask = mask.resize(control.size, Image.Resampling.BILINEAR)
    hard = mask.point(lambda value: 255 if value >= 128 else 0, mode="1").convert("L")
    radius = max(0, min(64, int(round(float(feather_px or 0)))))
    feathered = hard.filter(ImageFilter.BoxBlur(radius)) if radius else hard.copy()
    return hard, feathered, channel


def build_masked_control_preview(
    control_source,
    mask_source,
    *,
    invert=False,
    feather_px=0,
):
    """Show which control pixels will be allowed to contribute depth."""
    control = _load_image(control_source, "Control Image")
    original_max_side = max(control.size)
    _hard, feathered, channel = prepare_preview_mask(
        mask_source,
        control.size,
        invert=invert,
        feather_px=feather_px,
    )
    black = Image.new("RGB", control.size, "black")
    preview = Image.composite(control, black, feathered)
    return preview, channel


def _depth_image(value) -> Image.Image:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Depth Anything returned no preview")
        value = value[0]
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, np.ndarray):
        array = value
        if np.issubdtype(array.dtype, np.floating):
            maximum = float(np.nanmax(array)) if array.size else 0.0
            if maximum <= 1.0:
                array = array * 255.0
            array = np.nan_to_num(array).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    else:
        raise TypeError("Depth Anything returned an unsupported preview type")
    return image.convert("L")


def apply_mask_to_depth_preview(
    depth_source,
    mask_source,
    *,
    invert=False,
    feather_px=0,
):
    """Render depth influence as grayscale inside the mask and black outside."""
    depth = _depth_image(depth_source)
    _hard, feathered, channel = prepare_preview_mask(
        mask_source,
        depth.size,
        invert=invert,
        feather_px=feather_px,
    )
    black = Image.new("L", depth.size, 0)
    return Image.composite(depth, black, feathered).convert("RGB"), channel


def _resize_depth_preview_input(
    control: Image.Image, max_side: int = _PREVIEW_MAX_SIDE
) -> Image.Image:
    """Bound the disposable UI preview without changing its aspect ratio."""
    if max(control.size) <= max_side:
        return control
    preview = control.copy()
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return preview


def _make_depth_vram_available(release_model=None) -> None:
    """Release the active WanGP model only when Depth Anything needs room."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return

    gc.collect()
    torch.cuda.empty_cache()
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, AttributeError):
        return
    if free_bytes < _MINIMUM_DEPTH_VRAM_BYTES and callable(release_model):
        release_model()
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    if free_bytes < _MINIMUM_DEPTH_VRAM_BYTES:
        free_gib = free_bytes / 1024**3
        raise RuntimeError(
            "Depth Anything needs about 3 GiB of free VRAM for this preview "
            f"({free_gib:.2f} GiB currently free). Use WanGP's Unload Models "
            "control, then try again."
        )


def generate_effective_depth_preview(
    control_source,
    mask_source,
    *,
    get_preprocessor,
    release_model=None,
    invert=False,
    feather_px=0,
):
    """Run WanGP's configured depth estimator, then apply the custom mask."""
    if not callable(get_preprocessor):
        raise RuntimeError("WanGP did not expose its depth preprocessor")
    control = _load_image(control_source, "Control Image")
    # Validate before the comparatively expensive model load.
    prepare_preview_mask(
        mask_source,
        control.size,
        invert=invert,
        feather_px=feather_px,
    )

    original_max_side = max(control.size)
    control = _resize_depth_preview_input(control)
    preview_scale = max(control.size) / original_max_side
    preview_feather_px = float(feather_px or 0) * preview_scale
    _make_depth_vram_available(release_model)

    preprocessor = None
    try:
        preprocessor = get_preprocessor("depth", None)
        # WanGP exposes a video annotator here, even for a one-frame image.
        # Passing a bare HWC array makes it interpret every image row as a
        # separate frame and can cause an enormous bogus CUDA allocation.
        depth = preprocessor([np.asarray(control, dtype=np.uint8)])
        return apply_mask_to_depth_preview(
            depth,
            mask_source,
            invert=invert,
            feather_px=preview_feather_px,
        )
    finally:
        preprocessor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
