"""Load a separately uploaded depth-area mask for the WanGP plugin.

The host exposes this file through its generic ``custom_guide`` slot.  Ordinary
grayscale/RGB masks use luminance; an image with a non-uniform alpha channel
uses alpha instead.  White/opaque pixels are the selected area.
"""

from __future__ import annotations

import os

from PIL import Image, ImageOps


def load_custom_depth_mask(source, *, invert: bool = False):
    """Return one 8-bit mask image and the channel used to construct it."""
    if source is None:
        raise ValueError("A custom depth mask file was not provided")

    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        if not isinstance(source, (str, os.PathLike)) and hasattr(source, "name"):
            source = source.name
        try:
            source = os.fspath(source)
        except TypeError as exc:
            raise TypeError("Custom depth mask must be an image file") from exc
        try:
            with Image.open(source) as opened:
                image = opened.copy()
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Custom depth mask is not a readable image: {source}"
            ) from exc

    image = ImageOps.exif_transpose(image)
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    channel = "luminance"
    if has_alpha:
        alpha = image.convert("RGBA").getchannel("A")
        alpha_min, alpha_max = alpha.getextrema()
        if alpha_min != alpha_max:
            mask = alpha
            channel = "alpha"
        else:
            mask = image.convert("L")
    else:
        mask = image.convert("L")

    if invert:
        mask = ImageOps.invert(mask)
    return mask, channel
