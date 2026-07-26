from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from models.krea2_depth_preview import (
    apply_mask_to_depth_preview,
    build_masked_control_preview,
    generate_effective_depth_preview,
)


class DepthPreviewTests(unittest.TestCase):
    def setUp(self):
        self.control = Image.new("RGB", (4, 2), (120, 80, 40))
        self.mask = Image.new("L", (4, 2), 0)
        for y in range(2):
            for x in range(2):
                self.mask.putpixel((x, y), 255)

    def test_alignment_preview_applies_hard_mask_at_zero_feather(self):
        preview, channel = build_masked_control_preview(
            self.control, self.mask, feather_px=0
        )
        self.assertEqual(channel, "luminance")
        self.assertEqual(preview.getpixel((0, 0)), (120, 80, 40))
        self.assertEqual(preview.getpixel((3, 0)), (0, 0, 0))

    def test_effective_depth_preview_is_black_outside_mask(self):
        depth = np.full((2, 4, 3), 192, dtype=np.uint8)
        preview, _channel = apply_mask_to_depth_preview(
            depth, self.mask, feather_px=0
        )
        self.assertEqual(preview.getpixel((0, 0)), (192, 192, 192))
        self.assertEqual(preview.getpixel((3, 0)), (0, 0, 0))

    def test_black_area_mode_inverts_selection(self):
        preview, _channel = build_masked_control_preview(
            self.control, self.mask, invert=True, feather_px=0
        )
        self.assertEqual(preview.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(preview.getpixel((3, 0)), (120, 80, 40))

    def test_depth_estimation_receives_unmasked_control(self):
        observed = {}

        def factory(process_type, inpaint_color):
            self.assertEqual(process_type, "depth")
            self.assertIsNone(inpaint_color)

            def preprocess(frames):
                self.assertEqual(len(frames), 1)
                observed["image"] = frames[0].copy()
                return [np.full(frames[0].shape, 160, dtype=np.uint8)]

            return preprocess

        with patch(
            "models.krea2_depth_preview._make_depth_vram_available"
        ) as make_room:
            preview, _channel = generate_effective_depth_preview(
                self.control,
                self.mask,
                get_preprocessor=factory,
                feather_px=0,
            )
        make_room.assert_called_once_with(None)
        self.assertEqual(tuple(observed["image"][0, 3]), (120, 80, 40))
        self.assertEqual(preview.getpixel((3, 0)), (0, 0, 0))

    def test_depth_preview_caps_large_input_and_preserves_ratio(self):
        observed = {}

        def factory(_process_type, _inpaint_color):
            def preprocess(frames):
                observed["size"] = frames[0].shape[:2]
                return [np.full(frames[0].shape, 128, dtype=np.uint8)]

            return preprocess

        control = Image.new("RGB", (2048, 1024), "white")
        mask = Image.new("L", control.size, 255)
        with patch("models.krea2_depth_preview._make_depth_vram_available"):
            preview, _channel = generate_effective_depth_preview(
                control,
                mask,
                get_preprocessor=factory,
            )
        self.assertEqual(observed["size"], (512, 1024))
        self.assertEqual(preview.size, (1024, 512))

    def test_mask_ratio_mismatch_is_rejected_before_depth_load(self):
        loaded = False

        def factory(_process_type, _inpaint_color):
            nonlocal loaded
            loaded = True
            return lambda image: image

        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            generate_effective_depth_preview(
                self.control,
                Image.new("L", (2, 4), 255),
                get_preprocessor=factory,
            )
        self.assertFalse(loaded)


if __name__ == "__main__":
    unittest.main()
