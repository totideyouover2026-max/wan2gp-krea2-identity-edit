from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from models.krea2_custom_depth_mask import load_custom_depth_mask


class CustomDepthMaskTests(unittest.TestCase):
    def test_grayscale_file_uses_luminance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            image = Image.new("L", (2, 1))
            image.putdata([0, 255])
            image.save(path)

            mask, channel = load_custom_depth_mask(path)

        self.assertEqual(channel, "luminance")
        self.assertEqual(mask.mode, "L")
        self.assertEqual(list(mask.getdata()), [0, 255])

    def test_nonuniform_alpha_is_the_mask(self):
        image = Image.new("RGBA", (2, 1), (255, 255, 255, 255))
        image.putdata([(255, 255, 255, 0), (0, 0, 0, 255)])

        mask, channel = load_custom_depth_mask(image)

        self.assertEqual(channel, "alpha")
        self.assertEqual(list(mask.getdata()), [0, 255])

    def test_uniform_alpha_falls_back_to_luminance(self):
        image = Image.new("RGBA", (2, 1))
        image.putdata([(0, 0, 0, 255), (255, 255, 255, 255)])

        mask, channel = load_custom_depth_mask(image)

        self.assertEqual(channel, "luminance")
        self.assertEqual(list(mask.getdata()), [0, 255])

    def test_outside_mode_can_invert_the_mask(self):
        image = Image.new("L", (2, 1))
        image.putdata([0, 255])

        mask, channel = load_custom_depth_mask(image, invert=True)

        self.assertEqual(channel, "luminance")
        self.assertEqual(list(mask.getdata()), [255, 0])

    def test_unreadable_file_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-an-image.png"
            path.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a readable image"):
                load_custom_depth_mask(path)


if __name__ == "__main__":
    unittest.main()
