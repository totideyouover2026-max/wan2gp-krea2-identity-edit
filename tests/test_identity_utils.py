from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from models.krea2_identity_utils import (
    identity_lora_url,
    match_reference_dimensions,
    preprocess_identity_lora_state_dict,
    resolve_wangp_checkpoint,
    validate_grounding_px,
    validate_reference_images,
)


class FakeImage:
    def __init__(self, size=(1200, 800)):
        self.size = size

    def convert(self, mode):
        return self


class IdentityUtilsTests(unittest.TestCase):
    def test_grounding_px_is_bounded(self):
        self.assertEqual(validate_grounding_px(None), 768)
        self.assertEqual(validate_grounding_px("1024"), 1024)
        for invalid in (383, 1537, 768.5, True, "bad"):
            with self.assertRaises(ValueError):
                validate_grounding_px(invalid)

    def test_reference_count_and_order(self):
        scene, subject = FakeImage(), FakeImage()
        self.assertEqual(validate_reference_images([scene, subject]), [scene, subject])
        for invalid in ([], [scene, subject, FakeImage()]):
            with self.assertRaises(ValueError):
                validate_reference_images(invalid)

    def test_reference_type_is_checked(self):
        with self.assertRaises(TypeError):
            validate_reference_images([object()])

    def test_output_matches_reference_aspect_and_cap(self):
        width, height = match_reference_dimensions(2048, 2048, (1600, 900))
        self.assertLessEqual(width * height, 2_000_000)
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertAlmostEqual(width / height, 16 / 9, delta=0.03)
        with self.assertRaises(ValueError):
            match_reference_dimensions(1024, 1024, (1000, 1))

    def test_lora_variants_are_exact_published_files(self):
        self.assertTrue(identity_lora_url("full_v1.2").endswith("v1_2.safetensors"))
        self.assertTrue(identity_lora_url("full").endswith("v1_1.safetensors"))
        self.assertTrue(identity_lora_url("full_v1.1").endswith("v1_1.safetensors"))
        self.assertTrue(identity_lora_url("r128").endswith("v1_1_r128.safetensors"))
        self.assertTrue(identity_lora_url("r64").endswith("v1_1_r64.safetensors"))
        with self.assertRaises(ValueError):
            identity_lora_url("v1")

    def test_lora_key_conversion_is_minimal(self):
        marker = object()
        converted = preprocess_identity_lora_state_dict(
            {"diffusion_model.blocks.0.attn.wq.lora_A.weight": marker}
        )
        self.assertEqual(
            converted, {"blocks.0.attn.wq.lora_A.weight": marker}
        )

    def test_ckpts_prefixed_paths_resolve_through_checkpoint_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "Qwen3-VL-4B-Instruct" / "encoder.safetensors"
            target.parent.mkdir()
            target.touch()

            def locate(candidate):
                path = root / candidate
                return str(path) if path.is_file() else None

            resolved = resolve_wangp_checkpoint(
                r"ckpts\Qwen3-VL-4B-Instruct\encoder.safetensors", locate
            )
            self.assertEqual(Path(resolved), target)


if __name__ == "__main__":
    unittest.main()
