from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "models" / "krea2_identity_main.py").read_text(
            encoding="utf-8"
        )
        cls.tree = ast.parse(cls.source)

    def test_runtime_reads_generic_wangp_inputs(self):
        self.assertIn("original_input_ref_images", self.source)
        self.assertIn("custom_settings", self.source)
        self.assertIn("grounding_px", self.source)

    def test_wangp_checkpoint_paths_are_normalized(self):
        self.assertIn("def _resolve_wangp_checkpoint", self.source)
        self.assertIn("transformer_filename = _resolve_wangp_checkpoint", self.source)
        self.assertIn("text_encoder_filename = _resolve_wangp_checkpoint", self.source)

    def test_stream_order_is_text_sources_target(self):
        self.assertIn("torch.cat([context, *source_imgs, img]", self.source)
        self.assertIn("source_pos[..., 0] = frame", self.source)

    def test_target_only_slice_is_returned(self):
        self.assertIn("start = txtlen + srclen", self.source)
        self.assertIn("start : start + target_len", self.source)

    def test_cfg_and_interrupt_paths_exist(self):
        functions = {
            node.name for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("_identity_forward_cfg", functions)
        self.assertIn("_interrupt", functions)

    def test_visual_checkpoint_is_public_and_explicit(self):
        handler = (ROOT / "models" / "krea2_identity_handler.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"Comfy-Org/Krea-2"', handler)
        self.assertIn('"qwen3vl_4b_fp8_scaled.safetensors"', handler)
        self.assertIn('os.path.join("text_encoders", VISION_FILENAME)', self.source)

    def test_qwen_processor_supplies_video_processor(self):
        self.assertIn("from transformers.video_processing_utils import BaseVideoProcessor", self.source)
        self.assertIn("video_processor=BaseVideoProcessor()", self.source)

    def test_nonpersistent_qwen_rotary_buffers_are_materialized_before_loading(self):
        language_reset = self.source.index(
            "qwen.language_model.rotary_emb.reset_inv_freq()"
        )
        visual_reset = self.source.index("qwen.visual.rotary_pos_emb.reset_inv_freq()")
        first_load = self.source.index("offload.load_model_data(", language_reset)
        self.assertLess(language_reset, first_load)
        self.assertLess(visual_reset, first_load)


if __name__ == "__main__":
    unittest.main()
