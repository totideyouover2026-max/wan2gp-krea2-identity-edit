from __future__ import annotations

import unittest

from PIL import Image

from models.krea2_registered_outpaint import canvas_bbox, composite, prepare_source


class RegisteredOutpaintTests(unittest.TestCase):
    def test_margin_geometry_preserves_source_aspect(self):
        bbox = canvas_bbox((640, 480), (960, 480), [0, 0, 25, 25])
        x0, y0, x1, y1 = bbox
        self.assertEqual((x1 - x0) % 16, 0)
        self.assertEqual((y1 - y0) % 16, 0)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 4 / 3, delta=0.05)
        self.assertGreater(x0, 0)

    def test_composite_restores_protected_source_interior(self):
        source = Image.new("RGB", (64, 64), "red")
        prepared = prepare_source(source, (96, 64), (16, 0, 80, 64), seam_px=8)
        result = composite(Image.new("RGB", (96, 64), "blue"), prepared)
        self.assertEqual(result.getpixel((48, 32)), (255, 0, 0))
        self.assertEqual(result.getpixel((0, 32)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
