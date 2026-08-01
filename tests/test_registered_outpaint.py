from __future__ import annotations

import unittest

from PIL import Image

from models.krea2_registered_outpaint import canvas_bbox, composite, prepare_source


class RegisteredOutpaintTests(unittest.TestCase):
    def test_margin_geometry_preserves_source_aspect(self):
        bbox = canvas_bbox((640, 480), (960, 480), [0, 0, 25, 25])
        x0, y0, x1, y1 = bbox
        self.assertEqual(bbox, (160, 0, 800, 480))
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 4 / 3, delta=0.05)
        self.assertGreater(x0, 0)

    def test_each_single_direction_places_source_on_the_opposite_edge(self):
        self.assertEqual(
            canvas_bbox((640, 480), (800, 480), [0, 0, 100, 0]),
            (160, 0, 800, 480),
        )
        self.assertEqual(
            canvas_bbox((640, 480), (800, 480), [0, 0, 0, 100]),
            (0, 0, 640, 480),
        )
        self.assertEqual(
            canvas_bbox((640, 480), (640, 640), [100, 0, 0, 0]),
            (0, 160, 640, 640),
        )
        self.assertEqual(
            canvas_bbox((640, 480), (640, 640), [0, 100, 0, 0]),
            (0, 0, 640, 480),
        )

    def test_ratio_weights_do_not_create_padding_on_the_wrong_axis(self):
        bbox = canvas_bbox(
            (944, 976),
            (1280, 720),
            [0, 0, 0, 100],
            ratio_mode=True,
        )
        self.assertEqual(bbox, (0, 0, 696, 720))
        self.assertEqual(720 - bbox[3], 0)
        self.assertEqual(1280 - bbox[2], 584)

    def test_ratio_mode_ignores_weights_on_the_axis_that_needs_no_padding(self):
        bbox = canvas_bbox(
            (944, 976),
            (1280, 720),
            [100, 0, 0, 100],
            ratio_mode=True,
        )
        self.assertEqual(bbox, (0, 0, 696, 720))

    def test_manual_mixed_axis_expansion_requires_two_passes(self):
        with self.assertRaisesRegex(ValueError, "two-pass"):
            canvas_bbox((640, 480), (800, 640), [25, 0, 0, 25])

    def test_manual_direction_must_match_the_canvas_expansion_axis(self):
        with self.assertRaisesRegex(ValueError, "requires horizontal"):
            canvas_bbox((640, 480), (800, 480), [25, 0, 0, 0])

    def test_composite_restores_protected_source_interior(self):
        source = Image.new("RGB", (64, 64), "red")
        prepared = prepare_source(source, (96, 64), (16, 0, 80, 64), seam_px=8)
        result = composite(Image.new("RGB", (96, 64), "blue"), prepared)
        self.assertEqual(result.getpixel((48, 32)), (255, 0, 0))
        self.assertEqual(result.getpixel((0, 32)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
