"""Optional YuNet head crop for Krea 2 ReID references.

Adapted from ``face_crop.py`` in yijunwang2/krea2-reid (MIT), copyright
the respective authors.  The detector weights are downloaded separately and
are never stored in this repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DETECTION_SIZE = 320
DEFAULT_MIN_CONFIDENCE = 0.35


@dataclass(frozen=True)
class SquareCropTransform:
    x: int
    y: int
    side: int

    def map_bbox(self, bbox):
        x, y, width, height = bbox
        scale = self.side / DETECTION_SIZE
        return (
            self.x + math.floor(x * scale),
            self.y + math.floor(y * scale),
            self.x + math.ceil((x + width) * scale),
            self.y + math.ceil((y + height) * scale),
        )


@dataclass(frozen=True)
class FaceCropResult:
    image: Image.Image
    metadata: dict[str, Any]


def detection_square(image: Image.Image):
    """Return the official top-centred 320px detector view and its transform."""
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    x = (width - side) // 2 if width > height else 0
    y = 0
    square = image.crop((x, y, x + side, y + side))
    if square.size != (DETECTION_SIZE, DETECTION_SIZE):
        square = square.resize(
            (DETECTION_SIZE, DETECTION_SIZE), Image.Resampling.LANCZOS
        )
    return square, SquareCropTransform(x=x, y=y, side=side)


def expanded_head_crop(bbox, image_size):
    """Expand a detected face to the reference pipeline's square head crop."""
    left, _top, right, bottom = bbox
    image_width, image_height = image_size
    face_width = max(1, right - left)
    side = face_width * 2
    crop_left = round(left - face_width / 2)
    crop_right = crop_left + side
    crop_top = bottom - side
    return (
        max(0, crop_left),
        max(0, crop_top),
        min(image_width, crop_right),
        min(image_height, bottom),
    )


class YuNetFaceCropper:
    """Detect the strongest face and return the tested ReID head crop."""

    def __init__(self, model_path, *, min_confidence=DEFAULT_MIN_CONFIDENCE):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - host dependency
            raise RuntimeError(
                "Automatic ReID face cropping requires OpenCV. Update WanGP or "
                "select Keep full reference."
            ) from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing YuNet face detector: {model_path}")
        self._cv2 = cv2
        self.min_confidence = float(min_confidence)
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (DETECTION_SIZE, DETECTION_SIZE),
            self.min_confidence,
            0.3,
            5000,
        )

    def crop(self, image: Image.Image) -> FaceCropResult:
        original = image.convert("RGB")
        square, transform = detection_square(original)
        detector_input = self._cv2.cvtColor(
            np.asarray(square), self._cv2.COLOR_RGB2BGR
        )
        self.detector.setInputSize((DETECTION_SIZE, DETECTION_SIZE))
        _, faces = self.detector.detect(detector_input)
        candidate_count = 0 if faces is None else len(faces)
        metadata = {
            "applied": False,
            "min_confidence": self.min_confidence,
            "confidence": None,
            "candidate_count": candidate_count,
            "face_bbox_original": None,
            "crop_bbox_original": None,
            "original_size": original.size,
            "output_size": original.size,
        }
        if faces is None:
            return FaceCropResult(original, metadata)

        face = max(faces, key=lambda candidate: float(candidate[14]))
        confidence = float(face[14])
        metadata["confidence"] = round(confidence, 6)
        if confidence < self.min_confidence:
            return FaceCropResult(original, metadata)

        left, top, right, bottom = transform.map_bbox(
            tuple(float(value) for value in face[:4])
        )
        left = max(0, min(original.width - 1, left))
        top = max(0, min(original.height - 1, top))
        right = max(left + 1, min(original.width, right))
        bottom = max(top + 1, min(original.height, bottom))
        face_bbox = (left, top, right, bottom)
        crop_bbox = expanded_head_crop(face_bbox, original.size)
        cropped = original.crop(crop_bbox)
        metadata.update(
            {
                "applied": True,
                "face_bbox_original": list(face_bbox),
                "crop_bbox_original": list(crop_bbox),
                "output_size": cropped.size,
            }
        )
        return FaceCropResult(cropped, metadata)
