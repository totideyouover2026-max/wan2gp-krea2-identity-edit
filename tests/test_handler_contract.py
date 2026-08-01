from __future__ import annotations

import importlib.util
import json
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

    def test_model_definition_exposes_supported_identity_routes(self):
        model_def = self.handler.query_model_def("krea2_identity_turbo", {})
        settings = {item["id"]: item for item in model_def["custom_settings"]}
        self.assertEqual(
            [value for _label, value in settings["identity_method"]["choices"]],
            [
                "identity_edit",
                "identity_edit_ref2",
                "identity_edit_ref4",
                "identity_edit_ref8",
                "identity_edit_ref4_scene2",
                "identity_edit_ref8_scene2",
                "depth_prompt",
            ],
        )
        self.assertEqual(
            model_def["guide_preprocessing"]["labels"]["VG"],
            "Direct Image → Identity Edit",
        )

    def test_identity_types_map_to_wangp_base_types(self):
        raw = self.handler.query_model_def("krea2_identity_raw", {})
        turbo = self.handler.query_model_def("krea2_identity_turbo", {})
        self.assertEqual(raw["base_type_seen"], "krea2_raw")
        self.assertEqual(turbo["base_type_seen"], "krea2_turbo")
        self.assertFalse(raw["inpaint_support"])
        self.assertFalse(raw["at_least_one_image_ref_needed"])
        self.assertEqual(raw["video_guide_outpainting"], [1])

    def test_unified_full_bf16_encoder_is_the_only_grounding_stack(self):
        definition = self.handler.query_model_def("krea2_identity_turbo", {})
        self.assertEqual(len(definition["text_encoder_URLs"]), 1)
        self.assertTrue(
            definition["text_encoder_URLs"][0].endswith(
                "Qwen3-VL-4B-Instruct_bf16.safetensors"
            )
        )
        files = self.handler.query_model_files([], "krea2_identity_turbo", {})
        self.assertEqual(files[0]["repoId"], "base")
        self.assertFalse(
            any(item["repoId"] == "Comfy-Org/Krea-2" for item in files)
        )
        sam3 = next(item for item in files if item["repoId"] == "DeepBeepMeep/Wan2.1")
        self.assertEqual(sam3["sourceFolderList"], ["sam3"])
        self.assertEqual(
            sam3["fileList"],
            [["sam3.1_multiplex_bf16.safetensors", "bpe_simple_vocab_16e6.txt.gz"]],
        )
        self.assertFalse(
            any(
                item["repoId"] == "depth-anything/Depth-Anything-V2-Large-hf"
                for item in files
            )
        )

    def test_defaults_encode_effective_cfg_mapping(self):
        raw, turbo = {}, {}
        self.handler.update_default_settings("krea2_identity_raw", {}, raw)
        self.handler.update_default_settings("krea2_identity_turbo", {}, turbo)
        self.assertEqual((raw["num_inference_steps"], raw["guidance_scale"]), (20, 2.0))
        self.assertEqual((turbo["num_inference_steps"], turbo["guidance_scale"]), (8, 0))
        self.assertEqual(raw["custom_settings"]["identity_lora_variant"], "full_v1.2")
        self.assertEqual(turbo["custom_settings"]["identity_lora_variant"], "full_v1.2")
        self.assertEqual(raw["custom_settings"]["identity_method"], "identity_edit")
        self.assertEqual(turbo["custom_settings"]["identity_method"], "identity_edit")
        self.assertEqual(raw["denoising_strength"], 0.25)
        self.assertEqual(raw["custom_settings"]["depth_control_strength"], 1.0)
        self.assertEqual(raw["custom_settings"]["depth_mask_feather_px"], 16)
        self.assertEqual(raw["custom_settings"]["depth_user_lora_timing"], "depth_first")
        self.assertEqual(raw["custom_settings"]["depth_user_lora_ramp_early"], 0.0)
        self.assertEqual(raw["custom_settings"]["depth_user_lora_ramp_middle"], 0.25)
        self.assertEqual(raw["custom_settings"]["depth_user_lora_ramp_final"], 1.0)
        self.assertEqual(raw["custom_settings"]["subject_attention_timing"], "constant")
        self.assertEqual(raw["custom_settings"]["subject_attention_ramp_early"], 1.0)
        self.assertEqual(raw["custom_settings"]["subject_attention_ramp_middle"], 2.0)
        self.assertEqual(raw["custom_settings"]["subject_attention_ramp_final"], 8.0)
        self.assertEqual(raw["custom_settings"]["generation_process"], "standard")
        self.assertIsNone(raw["custom_guide"])
        self.assertEqual(raw["video_prompt_type"], "KI")
        self.assertEqual(raw["remove_background_images_ref"], 0)
        self.assertEqual(raw["custom_settings"]["subject_background_removal"], "off")
        self.assertEqual(raw["custom_settings"]["subject_segmentation_prompt"], "person")
        self.assertNotIn("outpaint_mode", raw["custom_settings"])

    def test_registered_outpaint_is_an_advanced_optional_task(self):
        model_def = self.handler.query_model_def("krea2_identity_turbo", {})
        settings = {item["id"]: item for item in model_def["custom_settings"]}
        advanced = json.loads(settings["advanced_settings"]["default"])
        self.assertEqual(advanced["generation_process"], "standard")
        self.assertIn("outpaint_prompt", advanced)
        self.assertEqual(len(model_def["custom_settings"]), 5)
        self.assertEqual(settings["advanced_settings"]["video_prompt_type"], "~")

    def test_registered_outpaint_does_not_mutate_the_generation_prompt(self):
        prompt = "A portrait of the existing person in the existing room."
        processed = self.handler.custom_prompt_preprocess(
            prompt,
            video_guide_outpainting="0 0 0 100",
            video_guide_outpainting_ratio="16:9",
            model_mode=0,
        )
        self.assertEqual(processed, prompt)
        self.assertNotIn("red paddings", processed)

    def test_validation_preserves_ratio_mode_for_outpaint_geometry(self):
        inputs = {
            "video_prompt_type": "KI",
            "image_refs": [object()],
            "video_guide_outpainting": "0 0 0 100",
            "video_guide_outpainting_ratio": "16:9",
            "custom_settings": {
                "identity_method": "identity_edit",
                "generation_process": "identity_then_outpaint",
            },
        }
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, inputs
            )
        )
        self.assertEqual(
            inputs["custom_settings"]["_registered_outpaint_ratio"],
            "16:9",
        )

    def test_validation_records_conditional_output_metadata_context(self):
        inputs = {
            "video_prompt_type": "KIDV",
            "image_guide": object(),
            "image_refs": [object()],
            "plugin_data": {"Unrelated Plugin Data": "Preserved"},
            "custom_settings": {
                "identity_method": "identity_edit",
                "depth_control_strength": 1.0,
                "builtin_adapter_timing": "simultaneous",
            },
        }
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, inputs
            )
        )
        self.assertEqual(
            inputs["plugin_data"]["Unrelated Plugin Data"], "Preserved"
        )
        self.assertEqual(
            inputs["plugin_data"]["krea2_identity_output_metadata"],
            {
                "identity_method": "identity_edit",
                "depth_active": True,
                "depth_mask_active": False,
                "user_lora_count": 0,
            },
        )

    def test_identity_reference_boost_profiles_keep_standard_and_reject_nag(self):
        standard = {
            "custom_settings": {"identity_method": "identity_edit"},
            "NAG_scale": 2.0,
        }
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, standard
            )
        )
        boosted = {
            "custom_settings": {"identity_method": "identity_edit_ref4"},
            "NAG_scale": 2.0,
        }
        self.assertEqual(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, boosted
            ),
            "Identity Edit reference-fidelity boosts currently require NAG "
            "scale 1.0 because both features use the transformer attention mask.",
        )

    def test_lora_control_defaults_to_v12_full(self):
        model_def = self.handler.query_model_def("krea2_identity_turbo", {})
        packed = next(
            setting["default"]
            for setting in model_def["custom_settings"]
            if setting["id"] == "advanced_settings"
        )
        self.assertEqual(json.loads(packed)["identity_lora_variant"], "full_v1.2")

    def test_prompt_guide_and_generic_template_are_exposed(self):
        model_def = self.handler.query_model_def("krea2_identity_turbo", {})
        title, markdown = model_def["prompt_infos"]
        self.assertEqual(title, "Krea 2 Identity Edit Prompt Guide")
        self.assertIn("## One reference", markdown)
        self.assertIn("## Two references", markdown)
        self.assertIn("## Depth control", markdown)
        self.assertNotIn("balcony", markdown.lower())
        self.assertNotIn("railing", markdown.lower())

        defaults = {}
        self.handler.update_default_settings(
            "krea2_identity_turbo", model_def, defaults
        )
        self.assertIn("[desired appearance or action]", defaults["prompt"])
        self.assertIn("[desired environment]", defaults["prompt"])

    def test_depth_uses_wangp_control_image_dropdown_and_upload_contract(self):
        model_def = self.handler.query_model_def("krea2_identity_turbo", {})
        settings = {
            setting["id"]: setting for setting in model_def["custom_settings"]
        }
        self.assertEqual(
            model_def["guide_preprocessing"],
            {
                "selection": ["", "VG", "DV"],
                "labels": {
                    "": "No Control Image",
                    "VG": "Direct Image → Identity Edit",
                    "DV": "Transfer Depth",
                },
                "default": "",
                "label": "Control Image Process",
            },
        )
        self.assertNotIn("depth_control_enabled", settings)
        self.assertNotIn("depth_control_source", settings)
        self.assertNotIn("skip_video_guide_preprocess", model_def)
        self.assertIsNone(model_def["model_modes"])
        self.assertNotIn("custom_denoising_strength", model_def)
        self.assertNotIn("denoising_strength", model_def)
        self.assertEqual(
            settings["depth_control_strength"],
            {
                "id": "depth_control_strength",
                "name": "Depth ControlNet-LoRA strength",
                "label": "Depth ControlNet-LoRA strength",
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "inc": 0.01,
                "video_prompt_type": "D",
                "info": (
                    "Multiplier applied directly to the experimental "
                    "Krea 2 depth adapter."
                ),
            },
        )
        self.assertEqual(
            model_def["mask_preprocessing"],
            {
                "selection": ["", "A", "NA", "Y", "NY"],
                "labels": {
                    "": "Whole Frame",
                    "A": "Painted Mask — White Area",
                    "NA": "Painted Mask — Black Area",
                    "Y": "Custom Mask — White Area",
                    "NY": "Custom Mask — Black Area",
                },
                "default": "",
                "label": "Control Area Processed",
            },
        )
        self.assertEqual(
            model_def["custom_guide"],
            {
                "label": (
                    "Optional Custom Depth Mask — use a Custom Mask option "
                    "under Control Area Processed"
                ),
                "file_types": [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".bmp",
                    ".tif",
                    ".tiff",
                ],
                "required": False,
            },
        )
        advanced = json.loads(settings["advanced_settings"]["default"])
        self.assertEqual(advanced["depth_mask_feather_px"], 16)
        self.assertEqual(advanced["depth_user_lora_timing"], "depth_first")
        self.assertEqual(advanced["depth_user_lora_ramp_early"], 0.0)
        self.assertEqual(advanced["depth_user_lora_ramp_middle"], 0.25)
        self.assertEqual(advanced["depth_user_lora_ramp_final"], 1.0)
        self.assertEqual(advanced["subject_attention_timing"], "constant")
        self.assertEqual(advanced["subject_attention_ramp_early"], 1.0)
        self.assertEqual(advanced["subject_attention_ramp_middle"], 2.0)
        self.assertEqual(advanced["subject_attention_ramp_final"], 8.0)

    def test_direct_image_identity_edit_uses_raw_control_and_native_denoising(self):
        valid = {
            "video_prompt_type": "KIVG",
            "image_guide": object(),
            "denoising_strength": 0.25,
            "custom_settings": {
                "identity_method": "identity_edit",
                "generation_process": "standard",
            },
        }
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, dict(valid)
            )
        )
        wrong_method = dict(valid)
        wrong_method["custom_settings"] = {"identity_method": "depth_prompt"}
        self.assertEqual(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, wrong_method
            ),
            "Direct Image → Identity Edit requires an Identity Edit method.",
        )
        masked = dict(valid)
        masked["video_prompt_type"] = "KIVGA"
        self.assertEqual(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, masked
            ),
            "Direct Image → Identity Edit currently supports Whole Frame only.",
        )
        missing = dict(valid)
        missing.pop("image_guide")
        self.assertEqual(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, missing
            ),
            "Direct Image → Identity Edit requires a Control Image.",
        )

    def test_hidden_krea_lanpaint_mode_cannot_reset_depth_strength(self):
        inputs = {
            "model_mode": 2,
            "denoising_strength": 1.0,
            "custom_settings": {"depth_control_strength": 0.0},
            "video_prompt_type": "KIDV",
        }
        self.handler.validate_generative_prompt(
            "krea2_identity_turbo", {}, inputs, "test prompt"
        )
        self.assertEqual(inputs["custom_settings"]["depth_control_strength"], 0.0)

    def test_reference_background_removal_preserves_scene_reference(self):
        model_def = self.handler.query_model_def("krea2_identity_turbo", {})
        self.assertTrue(model_def["no_background_removal"])
        self.assertTrue(callable(model_def["custom_image_ref_postprocessor"]))
        settings = {item["id"]: item for item in model_def["custom_settings"]}
        removal = settings["subject_background_removal"]
        self.assertEqual(removal["default"], "off")
        self.assertEqual(
            [value for _label, value in removal["choices"]],
            ["off", "sam3", "stable"],
        )
        self.assertEqual(settings["subject_segmentation_prompt"]["default"], "person")
        self.assertEqual(settings["subject_segmentation_prompt"]["type"], "text")
        self.assertEqual(
            model_def["image_ref_choices"],
            {
                "choices": [
                    (
                        "Identity Edit — scene, then optional subject",
                        "KI",
                    ),
                    ("Depth + prompt only — no reference", ""),
                ],
                "letters_filter": "KI",
                "default": "KI",
                "label": "Identity Edit References",
            },
        )

    def test_native_background_removal_preset_migrates_to_stable_plugin_path(self):
        defaults = {
            "remove_background_images_ref": 1,
            "custom_settings": {},
        }
        self.handler.fix_settings("krea2_identity_turbo", 0, {}, defaults)
        self.assertEqual(defaults["remove_background_images_ref"], 0)
        self.assertEqual(
            defaults["custom_settings"]["subject_background_removal"], "stable"
        )

    def test_identity_edit_reference_keeps_scene_first_mode(self):
        inputs = {
            "video_prompt_type": "IDV",
            "image_guide": object(),
            "custom_settings": {"identity_method": "identity_edit"},
        }
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, inputs
            )
        )
        self.assertEqual(inputs["video_prompt_type"], "KIDV")

    def test_removed_native_role_is_migrated_back_to_scene_first_mode(self):
        inputs = {
            "video_prompt_type": "KIDV",
            "image_guide": object(),
            "custom_settings": {
                "identity_method": "identity_edit",
                "single_reference_role": "native_subject",
            },
        }
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, inputs
            )
        )
        self.assertEqual(inputs["video_prompt_type"], "KIDV")

        defaults = {
            "video_prompt_type": "KIDV",
            "custom_settings": {
                "identity_method": "identity_edit",
                "single_reference_role": "native_subject",
            },
        }
        self.handler.fix_settings("krea2_identity_turbo", 0, {}, defaults)
        self.assertEqual(defaults["video_prompt_type"], "KIDV")
        self.assertNotIn(
            "single_reference_role", defaults["custom_settings"]
        )

    def test_one_reference_identity_edit_does_not_treat_scene_as_subject(self):
        class FakeImage:
            def convert(self, _mode):
                return self

        reference, cutout = FakeImage(), FakeImage()
        rembg = types.ModuleType("rembg")
        rembg.remove = lambda image, **_kwargs: cutout
        shared_utils = types.ModuleType("shared.utils.utils")
        shared_utils.new_rembg_session = lambda: object()
        with patch.dict(
            sys.modules,
            {"rembg": rembg, "shared.utils.utils": shared_utils},
        ):
            output, _masks = self.module._postprocess_identity_references(
                [reference], [None], 0, 0, None, "", None, "I",
                None, {},
                {
                    "identity_method": "identity_edit",
                    "single_reference_role": "native_subject",
                    "subject_background_removal": "stable",
                },
            )
        self.assertIs(output[0], reference)

    def test_depth_prompt_is_reference_free_and_requires_active_depth(self):
        model_def = self.handler.query_model_def("krea2_identity_turbo", {})
        self.assertFalse(model_def["at_least_one_image_ref_needed"])
        inputs = {
            "video_prompt_type": "KIDV",
            "image_guide": object(),
            "custom_settings": {
                "identity_method": "depth_prompt",
                "depth_control_strength": 0.98,
            },
        }
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, inputs
            )
        )
        self.assertEqual(inputs["video_prompt_type"], "DV")
        defaults = {
            "video_prompt_type": "KIDV",
            "custom_settings": {"identity_method": "depth_prompt"},
        }
        self.handler.fix_settings(
            "krea2_identity_turbo", 0, {}, defaults
        )
        self.assertEqual(defaults["video_prompt_type"], "DV")
        self.assertEqual(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo",
                {},
                {
                    "video_prompt_type": "KI",
                    "custom_settings": {"identity_method": "depth_prompt"},
                },
            ),
            "Depth + prompt only requires Transfer Depth and a Control Image.",
        )
        self.assertEqual(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo",
                {},
                {
                    "video_prompt_type": "DV",
                    "image_guide": object(),
                    "custom_settings": {
                        "identity_method": "depth_prompt",
                        "depth_control_strength": 0,
                    },
                },
            ),
            "Depth + prompt only requires depth strength greater than 0.",
        )

    def test_stable_subject_removal_skips_scene_and_alpha_matting(self):
        calls = []

        class FakeImage:
            def __init__(self, name):
                self.name = name

            def convert(self, _mode):
                return self

        scene, subject, cutout = FakeImage("scene"), FakeImage("subject"), FakeImage("cutout")
        rembg = types.ModuleType("rembg")

        def remove(image, **kwargs):
            calls.append((image, kwargs))
            return cutout

        rembg.remove = remove
        shared_utils = types.ModuleType("shared.utils.utils")
        session = object()
        shared_utils.new_rembg_session = lambda: session
        with patch.dict(
            sys.modules,
            {"rembg": rembg, "shared.utils.utils": shared_utils},
        ):
            output, masks = self.module._postprocess_identity_references(
                [scene, subject], [None, None], 0, 0, None, "", None, "KI",
                None, {}, {"subject_background_removal": "stable"},
            )
        self.assertIs(output[0], scene)
        self.assertIs(output[1], cutout)
        self.assertEqual(masks, [None, None])
        self.assertIs(calls[0][0], subject)
        self.assertFalse(calls[0][1]["alpha_matting"])
        self.assertIs(calls[0][1]["session"], session)

    def test_sam3_subject_removal_uses_semantic_phrase_and_preserves_scene(self):
        calls = []

        class FakeImage:
            def __init__(self, name):
                self.name = name

            def convert(self, _mode):
                return self

        scene = FakeImage("scene")
        subject = FakeImage("subject")
        mask = FakeImage("mask")
        cutout = FakeImage("cutout")
        magic_mask = types.ModuleType("shared.magic_mask")

        def generate_image_mask(image, phrase):
            calls.append((image, phrase))
            return image, mask, [phrase]

        magic_mask.generate_image_mask = generate_image_mask
        with patch.dict(sys.modules, {"shared.magic_mask": magic_mask}), patch.object(
            self.module, "_composite_subject_on_white", return_value=cutout
        ) as composite:
            output, masks = self.module._postprocess_identity_references(
                [scene, subject], [None, None], 0, 0, None, "", None, "KI",
                None, {},
                {
                    "subject_background_removal": "sam3",
                    "subject_segmentation_prompt": "man in the centre",
                },
            )
        self.assertIs(output[0], scene)
        self.assertIs(output[1], cutout)
        self.assertEqual(masks, [None, None])
        self.assertEqual(calls, [(subject, "man in the centre")])
        composite.assert_called_once_with(subject, mask)

    def test_sam3_composite_protects_subject_and_rejects_empty_mask(self):
        from PIL import Image

        subject = Image.new("RGB", (9, 9), (180, 20, 10))
        mask = Image.new("L", subject.size, 0)
        mask.putpixel((4, 4), 255)
        result = self.module._composite_subject_on_white(subject, mask)
        self.assertEqual(result.getpixel((4, 4)), (180, 20, 10))
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        with self.assertRaisesRegex(ValueError, "did not find the subject"):
            self.module._composite_subject_on_white(
                subject, Image.new("L", subject.size, 0)
            )

    def test_depth_dropdown_requires_its_control_image(self):
        self.assertEqual(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, {"video_prompt_type": "IDV"}
            ),
            "Transfer Depth requires a Control Image.",
        )

    def test_old_reference_depth_toggle_is_not_migrated_without_an_upload(self):
        defaults = {
            "custom_settings": {
                "depth_control_enabled": True,
                "depth_control_source": "subject",
            }
        }
        self.handler.fix_settings(
            "krea2_identity_turbo", 0, {}, defaults
        )
        self.assertNotIn("depth_control_enabled", defaults["custom_settings"])
        self.assertNotIn("depth_control_source", defaults["custom_settings"])
        self.assertEqual(defaults["custom_settings"]["depth_control_strength"], 1.0)
        self.assertEqual(defaults["denoising_strength"], 1.0)
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo",
                {},
                {"video_prompt_type": "IDV", "image_guide": object()},
            )
        )

    def test_masked_control_editor_counts_as_a_control_image(self):
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo",
                {},
                {"video_prompt_type": "KIDVA", "image_mask_guide": object()},
            )
        )

    def test_custom_mask_mode_requires_upload_and_rejects_painted_route(self):
        common = {
            "image_guide": object(),
            "custom_settings": {"identity_method": "identity_edit"},
        }
        missing = dict(common, video_prompt_type="IDVY")
        self.assertEqual(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, missing
            ),
            "The selected Custom Mask area requires a Custom Depth Mask upload.",
        )
        valid = dict(common, video_prompt_type="IDVY", custom_guide="mask.png")
        self.assertIsNone(
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, valid
            )
        )
        painted = dict(common, video_prompt_type="IDVA", custom_guide="mask.png")
        self.assertIn(
            "must use Custom Mask",
            self.handler.validate_generative_settings(
                "krea2_identity_turbo", {}, painted
            ),
        )

    def test_early_custom_mask_settings_migrate_off_painted_preprocessing(self):
        defaults = {
            "video_prompt_type": "IDVNA",
            "custom_guide": "mask.png",
            "custom_settings": {"identity_method": "identity_edit"},
        }
        self.handler.fix_settings("krea2_identity_turbo", 0, {}, defaults)
        self.assertEqual(defaults["video_prompt_type"], "KIDVNY")

    def test_dedicated_depth_strength_is_preserved_and_native_guide_flag_removed(self):
        defaults = {
            "video_prompt_type": "KIDV",
            "custom_settings": {
                "depth_control_strength": 0.98,
            }
        }
        self.handler.fix_settings("krea2_identity_turbo", 0, {}, defaults)
        self.assertEqual(defaults["video_prompt_type"], "KIDV")
        self.assertEqual(defaults["denoising_strength"], 1.0)
        self.assertEqual(defaults["custom_settings"]["depth_control_strength"], 0.98)

    def test_legacy_identity_reference_mode_is_migrated_to_scene_safe_mode(self):
        defaults = {"video_prompt_type": "I", "custom_settings": {}}
        self.handler.fix_settings("krea2_identity_turbo", 0, {}, defaults)
        self.assertEqual(defaults["video_prompt_type"], "KI")
        self.assertEqual(defaults["remove_background_images_ref"], 0)
        self.assertEqual(defaults["custom_settings"]["depth_mask_feather_px"], 16)

    def test_legacy_fixed_lora_is_cleared_before_wangp_merges_model_def(self):
        model_def = {
            "loras": [
                "https://huggingface.co/conradlocke/krea2-identity-edit/"
                "resolve/main/krea2_identity_edit_v1_1.safetensors"
            ],
            "loras_multipliers": [1.0],
        }
        defaults = self.handler.query_model_def("krea2_identity_turbo", model_def)

        # WanGP applies the supplied model definition after the handler defaults.
        defaults.update(model_def)

        self.assertEqual(defaults["loras"], [])
        self.assertEqual(defaults["loras_multipliers"], [])

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
        krea_main.Krea2Qwen3VLProcessor = object
        with patch.dict(
            sys.modules,
            {
                factory_module.__name__: factory_module,
                "models.ideogram4.qwen3_vl_transformers": qwen_module,
                "models.krea2.krea2_main": krea_main,
            },
        ):
            _processor, profile = self.handler.load_model(
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
        self.assertNotIn("depth_estimator", profile)


if __name__ == "__main__":
    unittest.main()
