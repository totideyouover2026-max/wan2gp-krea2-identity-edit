from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class FakeKrea2Handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {"image_outputs": True, "base_type_seen": base_model_type}

    @staticmethod
    def query_model_files(compute_list, base_model_type, model_def=None):
        return [{"repoId": "base", "sourceFolderList": [""], "fileList": [[]]}]

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults["base_type_seen"] = base_model_type

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        return None


def load_handler_module():
    package = types.ModuleType("models.krea2")
    package.__path__ = []
    upstream = types.ModuleType("models.krea2.krea2_handler")
    upstream.family_handler = FakeKrea2Handler
    name = "models.krea2_identity_handler_contract_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "models" / "krea2_identity_handler.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"models.krea2": package, "models.krea2.krea2_handler": upstream},
    ):
        spec.loader.exec_module(module)
    return module


class HandlerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_handler_module()
        cls.handler = cls.module.family_handler

    def test_identity_types_map_to_wangp_base_types(self):
        raw = self.handler.query_model_def("krea2_identity_raw", {})
        turbo = self.handler.query_model_def("krea2_identity_turbo", {})
        self.assertEqual(raw["base_type_seen"], "krea2_raw")
        self.assertEqual(turbo["base_type_seen"], "krea2_turbo")
        self.assertFalse(raw["inpaint_support"])
        self.assertTrue(raw["at_least_one_image_ref_needed"])

    def test_visual_file_is_added_to_base_downloads(self):
        files = self.handler.query_model_files([], "krea2_identity_turbo", {})
        self.assertEqual(files[0]["repoId"], "base")
        self.assertEqual(files[-1]["repoId"], "Comfy-Org/Krea-2")
        self.assertEqual(
            files[-1]["fileList"], [["qwen3vl_4b_fp8_scaled.safetensors"]]
        )

    def test_defaults_encode_effective_cfg_mapping(self):
        raw, turbo = {}, {}
        self.handler.update_default_settings("krea2_identity_raw", {}, raw)
        self.handler.update_default_settings("krea2_identity_turbo", {}, turbo)
        self.assertEqual((raw["num_inference_steps"], raw["guidance_scale"]), (20, 2.0))
        self.assertEqual((turbo["num_inference_steps"], turbo["guidance_scale"]), (10, 0))

    def test_lora_control_exposes_v12_and_versioned_v11_variants(self):
        model_def = self.handler.query_model_def("krea2_identity_turbo", {})
        lora_setting = next(
            setting
            for setting in model_def["custom_settings"]
            if setting["id"] == "identity_lora_variant"
        )
        values = [value for _label, value in lora_setting["choices"]]
        self.assertEqual(
            values,
            ["full_v1.2", "r64", "r128", "full_v1.1"],
        )
        self.assertEqual(lora_setting["default"], "r64")

    def test_loader_maps_wangp_arguments_into_factory_keywords(self):
        calls = []

        class FakeProcessor:
            transformer = object()
            text_encoder = object()
            vae = object()

        factory_module = types.ModuleType("models.krea2_identity_main")

        def fake_factory(**kwargs):
            calls.append(kwargs)
            return FakeProcessor()

        factory_module.model_factory = fake_factory
        qwen_module = types.ModuleType("models.ideogram4.qwen3_vl_transformers")
        qwen_module.Qwen3VLVisionModel = object
        krea_main = types.ModuleType("models.krea2.krea2_main")
        krea_main.Krea2Pipeline = object
        with patch.dict(
            sys.modules,
            {
                factory_module.__name__: factory_module,
                "models.ideogram4.qwen3_vl_transformers": qwen_module,
                "models.krea2.krea2_main": krea_main,
            },
        ):
            self.handler.load_model(
                "ckpts/model.safetensors",
                model_type="selected_model",
                base_model_type="krea2_identity_turbo",
                model_def={"name": "test"},
                text_encoder_filename="ckpts/text.safetensors",
                dtype="bf16",
                VAE_dtype="fp32",
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["checkpoint_dir"], "ckpts")
        self.assertEqual(calls[0]["model_filename"], "ckpts/model.safetensors")
        self.assertEqual(calls[0]["base_model_type"], "krea2_identity_turbo")
        self.assertEqual(calls[0]["text_encoder_filename"], "ckpts/text.safetensors")


if __name__ == "__main__":
    unittest.main()
