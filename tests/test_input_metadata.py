import tempfile
import unittest
from pathlib import Path

from PIL import Image

from models.krea2_input_metadata import (
    aspect_ratio,
    gallery_input_label,
    image_dimensions,
    image_input_label,
)


class InputMetadataTests(unittest.TestCase):
    def test_exact_aspect_ratio_is_reduced(self):
        self.assertEqual(aspect_ratio(1280, 720), "16:9")
        self.assertEqual(aspect_ratio(1280, 704), "20:11")

    def test_pil_image_label_contains_dimensions_and_ratio(self):
        image = Image.new("RGB", (1280, 720))
        self.assertEqual(
            image_input_label("Control Image", image),
            "Control Image (1280x720, 16:9)",
        )

    def test_filepath_and_gradio_file_shapes_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.new("L", (780, 800)).save(path)
            self.assertEqual(image_dimensions(path), (780, 800))
            self.assertEqual(image_dimensions({"path": str(path)}), (780, 800))

    def test_empty_input_restores_base_label(self):
        self.assertEqual(image_input_label("Control Image", None), "Control Image")
        self.assertEqual(
            gallery_input_label("Reference Images", []), "Reference Images"
        )

    def test_gallery_label_describes_each_reference(self):
        references = [
            (Image.new("RGB", (1280, 720)), None),
            (Image.new("RGB", (780, 800)), "subject"),
        ]
        self.assertEqual(
            gallery_input_label("Reference Images", references),
            (
                "Reference Images "
                "(1: 1280x720, 16:9; 2: 780x800, 39:40)"
            ),
        )


if __name__ == "__main__":
    unittest.main()
