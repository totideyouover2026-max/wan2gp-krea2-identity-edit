from __future__ import annotations

import unittest

from PIL import Image

from models.krea2_reid_face_crop import detection_square, expanded_head_crop


class ReIDFaceCropTests(unittest.TestCase):
    def test_detection_square_is_top_centred_without_modifying_original(self):
        image = Image.new("RGB", (1280, 720), "white")
        square, transform = detection_square(image)
        self.assertEqual(square.size, (320, 320))
        self.assertEqual((transform.x, transform.y, transform.side), (280, 0, 720))
        self.assertEqual(image.size, (1280, 720))

    def test_expanded_crop_is_twice_face_width_and_clamped(self):
        self.assertEqual(
            expanded_head_crop((500, 100, 600, 240), (1280, 720)),
            (450, 40, 650, 240),
        )
        self.assertEqual(
            expanded_head_crop((10, 0, 110, 140), (1280, 720)),
            (0, 0, 160, 140),
        )


if __name__ == "__main__":
    unittest.main()
