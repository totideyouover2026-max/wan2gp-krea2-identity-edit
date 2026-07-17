"""Pure validation and sizing helpers for Krea 2 Identity Edit."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence


MIN_GROUNDING_PX = 384
MAX_GROUNDING_PX = 1536
DEFAULT_GROUNDING_PX = 768
MAX_OUTPUT_PIXELS = 2_000_000
TWO_REFERENCE_RECOMMENDED_PIXELS = 1_500_000


def resolve_wangp_checkpoint(path, locate_file) -> str:
    """Resolve absolute, checkpoint-relative, and ckpts-prefixed host paths."""
    if path is None:
        raise FileNotFoundError("WanGP did not provide a checkpoint path")
    path = os.fspath(path)
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    candidates = [path]
    normalized = path.replace("\\", "/").lstrip("./")
    parts = normalized.split("/")
    if parts and parts[0].lower() in {"ckpts", "checkpoints"} and len(parts) > 1:
        candidates.append(os.path.join(*parts[1:]))
    candidates.append(os.path.basename(path))
    for candidate in dict.fromkeys(candidates):
        resolved = locate_file(candidate)
        if resolved is not None:
            return os.path.abspath(resolved)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Unable to locate WanGP checkpoint {path!r}; tried {candidates}"
    )


def validate_grounding_px(value) -> int:
    """Return a bounded integer grounding resolution."""
    if value is None:
        return DEFAULT_GROUNDING_PX
    if isinstance(value, bool):
        raise ValueError("grounding_px must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("grounding_px must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("grounding_px must be an integer") from exc
    if parsed < MIN_GROUNDING_PX or parsed > MAX_GROUNDING_PX:
        raise ValueError(
            f"grounding_px must be between {MIN_GROUNDING_PX} and {MAX_GROUNDING_PX}"
        )
    return parsed


def validate_reference_images(images) -> list:
    """Validate the public one/two-reference contract without reordering inputs."""
    if images is None:
        images = []
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        images = [images]
    result = [image for image in images if image is not None]
    if not 1 <= len(result) <= 2:
        raise ValueError("Krea 2 Identity Edit requires one or two reference images")
    for image in result:
        if not hasattr(image, "convert") or not hasattr(image, "size"):
            raise TypeError("Each Krea 2 Identity Edit reference must be a PIL image")
    return result


def match_reference_dimensions(
    requested_width: int,
    requested_height: int,
    reference_size: tuple[int, int],
    *,
    align: int = 16,
    max_pixels: int = MAX_OUTPUT_PIXELS,
) -> tuple[int, int]:
    """Match the first reference aspect ratio while respecting area and alignment."""
    ref_width, ref_height = map(int, reference_size)
    if ref_width <= 0 or ref_height <= 0:
        raise ValueError("The primary reference image has invalid dimensions")
    if max(ref_width, ref_height) / min(ref_width, ref_height) > 200:
        raise ValueError("The primary reference aspect ratio must not exceed 200:1")
    requested_area = max(align * align, int(requested_width) * int(requested_height))
    target_area = min(requested_area, max_pixels)
    aspect = ref_width / ref_height
    width = math.sqrt(target_area * aspect)
    height = width / aspect
    width = max(align, int(round(width / align)) * align)
    height = max(align, int(round(height / align)) * align)
    while width * height > max_pixels:
        if width >= height and width > align:
            width -= align
        elif height > align:
            height -= align
        else:
            break
    return width, height


def identity_lora_url(variant: str) -> str:
    variants = {
        "full_v1.2": "krea2_identity_edit_v1_2.safetensors",
        "full_v1.1": "krea2_identity_edit_v1_1.safetensors",
        "r128": "krea2_identity_edit_v1_1_r128.safetensors",
        "r64": "krea2_identity_edit_v1_1_r64.safetensors",
    }
    try:
        filename = variants[str(variant)]
    except KeyError as exc:
        raise ValueError("identity_lora_variant must be one of: full v1.2, full v1.1, r128, r64") from exc
    return f"https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/{filename}"


def preprocess_identity_lora_state_dict(state_dict: dict) -> dict:
    """Map ai-toolkit/Comfy diffusion-model keys to WanGP's Krea 2 transformer."""
    prefix = "diffusion_model."
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }
