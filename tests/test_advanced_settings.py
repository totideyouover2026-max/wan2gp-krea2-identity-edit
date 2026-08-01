from __future__ import annotations

import json
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from models.krea2_advanced_settings import (
    ADVANCED_SETTINGS_DEFAULTS,
    DEFAULT_OUTPAINT_PROMPT,
    advanced_settings_metadata,
    encode_advanced_settings,
    expand_advanced_settings,
    normalize_advanced_settings,
)


ROOT = Path(__file__).resolve().parents[1]


class AdvancedSettingsTests(unittest.TestCase):
    def test_round_trip_is_canonical_and_validated(self):
        packed = encode_advanced_settings(
            {
                "generation_process": "standard",
                "outpaint_prompt": "Continue the stone wall into the right side.",
                "grounding_px": 1024,
                "output_resolution_limit": "unlimited",
                "identity_lora_variant": "r64",
                "identity_lora_variant_schema": "v1.2",
                "subject_attention_timing": "ramp",
                "subject_attention_ramp_early": 1.0,
                "subject_attention_ramp_middle": 3.0,
                "subject_attention_ramp_final": 7.0,
                "builtin_adapter_timing": "depth_then_identity",
                "builtin_depth_ramp_early": 1.0,
                "builtin_depth_ramp_middle": 0.5,
                "builtin_depth_ramp_final": 0.0,
                "builtin_identity_ramp_early": 0.25,
                "builtin_identity_ramp_middle": 0.75,
                "builtin_identity_ramp_final": 1.0,
                "depth_mask_feather_px": 0,
                "depth_user_lora_timing": "all_steps",
                "depth_user_lora_ramp_early": 0.1,
                "depth_user_lora_ramp_middle": 0.5,
                "depth_user_lora_ramp_final": 0.9,
            }
        )
        self.assertEqual(packed, json.dumps(json.loads(packed), separators=(",", ":"), sort_keys=True))
        self.assertEqual(
            normalize_advanced_settings(packed),
            {
                "generation_process": "standard",
                "outpaint_prompt": "Continue the stone wall into the right side.",
                "grounding_px": 1024,
                "output_resolution_limit": "unlimited",
                "identity_lora_variant": "r64",
                "identity_lora_variant_schema": "v1.2",
                "subject_attention_timing": "ramp",
                "subject_attention_ramp_early": 1.0,
                "subject_attention_ramp_middle": 3.0,
                "subject_attention_ramp_final": 7.0,
                "builtin_adapter_timing": "depth_then_identity",
                "builtin_depth_ramp_early": 1.0,
                "builtin_depth_ramp_middle": 0.5,
                "builtin_depth_ramp_final": 0.0,
                "builtin_identity_ramp_early": 0.25,
                "builtin_identity_ramp_middle": 0.75,
                "builtin_identity_ramp_final": 1.0,
                "depth_mask_feather_px": 0,
                "depth_user_lora_timing": "all_steps",
                "depth_user_lora_ramp_early": 0.1,
                "depth_user_lora_ramp_middle": 0.5,
                "depth_user_lora_ramp_final": 0.9,
            },
        )

    def test_legacy_flat_settings_are_preserved_and_packed(self):
        expanded = expand_advanced_settings(
            {
                "identity_method": "identity_edit",
                "grounding_px": 896,
                "identity_lora_variant": "r128",
                "depth_mask_feather_px": 4,
                "depth_user_lora_timing": "all_steps",
                "generation_process": "standard",
            }
        )
        self.assertEqual(expanded["identity_method"], "identity_edit")
        self.assertEqual(expanded["generation_process"], "standard")
        self.assertEqual(expanded["outpaint_prompt"], DEFAULT_OUTPAINT_PROMPT)
        self.assertEqual(expanded["grounding_px"], 896)
        self.assertEqual(expanded["output_resolution_limit"], "safe_2mp")
        self.assertEqual(expanded["identity_lora_variant"], "r128")
        self.assertEqual(expanded["subject_attention_timing"], "constant")
        self.assertEqual(expanded["subject_attention_ramp_early"], 1.0)
        self.assertEqual(expanded["subject_attention_ramp_middle"], 2.0)
        self.assertEqual(expanded["subject_attention_ramp_final"], 8.0)
        self.assertEqual(expanded["builtin_adapter_timing"], "simultaneous")
        self.assertEqual(expanded["builtin_depth_ramp_early"], 1.0)
        self.assertEqual(expanded["builtin_depth_ramp_middle"], 0.5)
        self.assertEqual(expanded["builtin_depth_ramp_final"], 0.0)
        self.assertEqual(expanded["builtin_identity_ramp_early"], 0.25)
        self.assertEqual(expanded["builtin_identity_ramp_middle"], 0.75)
        self.assertEqual(expanded["builtin_identity_ramp_final"], 1.0)
        self.assertEqual(expanded["depth_mask_feather_px"], 4)
        self.assertEqual(expanded["depth_user_lora_timing"], "all_steps")
        self.assertEqual(expanded["depth_user_lora_ramp_early"], 0.0)
        self.assertEqual(expanded["depth_user_lora_ramp_middle"], 0.25)
        self.assertEqual(expanded["depth_user_lora_ramp_final"], 1.0)
        self.assertEqual(
            json.loads(expanded["advanced_settings"])["generation_process"],
            "standard",
        )

    def test_packed_values_override_legacy_flat_values(self):
        packed = encode_advanced_settings(
            {
                **ADVANCED_SETTINGS_DEFAULTS,
                "grounding_px": 1152,
                "identity_lora_variant": "r64",
            }
        )
        expanded = expand_advanced_settings(
            {
                "advanced_settings": packed,
                "grounding_px": 512,
                "identity_lora_variant": "full_v1.1",
            }
        )
        self.assertEqual(expanded["grounding_px"], 1152)
        self.assertEqual(expanded["identity_lora_variant"], "r64")

    def test_unmarked_legacy_rank_64_payload_migrates_to_full_v12(self):
        legacy_payload = json.dumps(
            {"identity_lora_variant": "r64"}, separators=(",", ":")
        )
        migrated = normalize_advanced_settings(legacy_payload)
        self.assertEqual(migrated["identity_lora_variant"], "full_v1.2")
        self.assertEqual(migrated["identity_lora_variant_schema"], "v1.2")

    def test_marked_rank_64_payload_remains_an_explicit_choice(self):
        packed = encode_advanced_settings({"identity_lora_variant": "r64"})
        self.assertEqual(
            normalize_advanced_settings(packed)["identity_lora_variant"], "r64"
        )

    def test_invalid_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            normalize_advanced_settings("not-json")
        with self.assertRaisesRegex(ValueError, "grounding_px"):
            normalize_advanced_settings({"grounding_px": 100})
        with self.assertRaisesRegex(ValueError, "output_resolution_limit"):
            normalize_advanced_settings({"output_resolution_limit": "adaptive"})
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            normalize_advanced_settings(
                {
                    "depth_user_lora_ramp_early": 0.0,
                    "depth_user_lora_ramp_middle": 1.0,
                    "depth_user_lora_ramp_final": 0.5,
                }
            )
        with self.assertRaisesRegex(ValueError, "subject_attention_timing"):
            normalize_advanced_settings({"subject_attention_timing": "late"})
        with self.assertRaisesRegex(ValueError, "subject attention ramp"):
            normalize_advanced_settings(
                {
                    "subject_attention_ramp_early": 1.0,
                    "subject_attention_ramp_middle": 8.0,
                    "subject_attention_ramp_final": 4.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "outpaint_prompt"):
            normalize_advanced_settings({"outpaint_prompt": "x" * 2001})

    def test_empty_outpaint_prompt_uses_conservative_default(self):
        normalized = normalize_advanced_settings({"outpaint_prompt": "  "})
        self.assertEqual(normalized["outpaint_prompt"], DEFAULT_OUTPAINT_PROMPT)

    def test_removed_single_reference_role_is_ignored_during_migration(self):
        normalized = normalize_advanced_settings(
            {"single_reference_role": "native_subject"}
        )
        self.assertNotIn("single_reference_role", normalized)

    def test_removed_secondary_geometry_is_ignored_during_migration(self):
        normalized = normalize_advanced_settings(
            {"secondary_reference_geometry": "stretch"}
        )
        self.assertNotIn("secondary_reference_geometry", normalized)

    def test_removed_v11_full_selection_migrates_to_v12_full_default(self):
        self.assertEqual(
            normalize_advanced_settings(
                {"identity_lora_variant": "full_v1.1"}
            )["identity_lora_variant"],
            "full_v1.2",
        )
        self.assertEqual(
            ADVANCED_SETTINGS_DEFAULTS["identity_lora_variant"],
            "full_v1.2",
        )

    def test_output_metadata_hides_inactive_builtin_ramps(self):
        rows = advanced_settings_metadata(
            {"builtin_adapter_timing": "simultaneous"},
            identity_method="identity_edit",
            depth_active=True,
        )
        self.assertEqual(rows["Builtin Adapter Timing"], "Simultaneous")
        self.assertFalse(
            any(label.startswith("Builtin Depth Ramp") for label in rows)
        )
        self.assertFalse(
            any(label.startswith("Builtin Identity Edit Ramp") for label in rows)
        )

    def test_output_metadata_records_only_active_outpaint_prompt(self):
        inactive = advanced_settings_metadata(
            {"outpaint_prompt": "Continue the tiled floor."},
            identity_method="identity_edit",
        )
        self.assertNotIn("Registered Outpaint Prompt", inactive)
        active = advanced_settings_metadata(
            {
                "generation_process": "identity_then_outpaint",
                "outpaint_prompt": "Continue the tiled floor.",
            },
            identity_method="identity_edit",
        )
        self.assertEqual(
            active["Registered Outpaint Prompt"],
            "Continue the tiled floor.",
        )

    def test_output_metadata_shows_only_active_dependent_settings(self):
        rows = advanced_settings_metadata(
            {
                "builtin_adapter_timing": "depth_then_identity",
                "subject_attention_timing": "ramp",
                "depth_user_lora_timing": "all_steps",
            },
            identity_method="identity_edit",
            depth_active=True,
            depth_mask_active=True,
            user_lora_count=1,
        )
        self.assertEqual(
            rows["Builtin Adapter Timing"],
            "Depth Layout → Identity Refinement",
        )
        self.assertIn("Builtin Depth Ramp — Early", rows)
        self.assertIn("Builtin Identity Edit Ramp — Final", rows)
        self.assertIn("Subject Attention Ramp — Middle", rows)
        self.assertEqual(rows["Depth Mask Feather"], "16 px")
        self.assertEqual(
            rows["Additional LoRA Timing With Depth"], "All Steps"
        )
        self.assertFalse(
            any(label.startswith("Additional LoRA Ramp") for label in rows)
        )

    def test_output_metadata_omits_settings_for_inactive_features(self):
        rows = advanced_settings_metadata(
            {},
            identity_method="depth_prompt",
            depth_active=True,
            user_lora_count=0,
        )
        self.assertEqual(rows, {"Generation Process": "Standard Single Pass"})

    def test_plugin_owns_launcher_modal_and_adaptive_visibility(self):
        source = (ROOT / "plugin.py").read_text(encoding="utf-8")
        self.assertIn('label="Show Advanced Settings"', source)
        self.assertIn('label="Registered Outpaint prompt"', source)
        self.assertIn(
            "initial_launcher_visible = _plugin_active(self.state.value)", source
        )
        self.assertIn(
            'label="Built-in Identity Edit and Depth timing"', source
        )
        self.assertIn(
            "Depth layout → Identity refinement (experimental)", source
        )
        self.assertIn(
            'label="Reference-mode output resolution limit"', source
        )
        self.assertIn('choices=[("No", "no"), ("Yes", "yes")]', source)
        self.assertIn("krea2-advanced-floating", source)
        self.assertIn('elem_id="krea2-advanced-floating-modal"', source)
        self.assertIn("def refresh_launcher(state):", source)
        self.assertIn("active = _plugin_active(state)", source)
        self.assertIn('elem_id="krea2-advanced-drag-handle"', source)
        self.assertIn('document.addEventListener("pointerdown"', source)
        self.assertIn("self.add_custom_js(_SCRIPT)", source)
        self.assertIn('self.insert_after("custom_settings_visibility_trigger"', source)
        self.assertIn('self.insert_after("custom_guide"', source)
        self.assertIn('label="Optional Custom Depth Mask"', source)
        self.assertNotIn('label="Control + Custom Mask Alignment (preview only)"', source)
        self.assertIn('"Generate Effective Depth Preview"', source)
        self.assertIn('initial_mode == "identity_edit"', source)
        self.assertIn('choices=_ALL_IDENTITY_METHOD_CHOICES', source)
        self.assertIn(
            '("Depth + prompt only (selected by reference mode)", "depth_prompt")',
            source,
        )
        self.assertIn('reference_visible = mode == "identity_edit"', source)
        self.assertIn('preparation == "sam3"', source)
        self.assertIn('"outpaint_only"', source)
        self.assertIn('"identity_then_outpaint"', source)
        self.assertIn('maximum=64', source)
        self.assertIn('label="Early third"', source)
        self.assertNotIn("Single-reference Identity Edit role", source)
        self.assertNotIn("Native I — identity subject", source)
        self.assertIn('label="Middle third"', source)
        self.assertIn('label="Final third"', source)
        self.assertIn('label="Identity Edit subject-attention timing"', source)
        self.assertIn('label="Subject early third"', source)
        self.assertIn('label="Subject middle third"', source)
        self.assertIn('label="Subject final third"', source)
        self.assertNotIn("secondary_reference_geometry", source)
        self.assertNotIn("Stretch to output geometry", source)
        self.assertIn('("v1.2 Full — 1.83 GB (recommended)", "full_v1.2")', source)
        self.assertIn('("v1.2 Rank 128 — 0.91 GB", "r128")', source)
        self.assertIn('("v1.2 Rank 64 — 0.46 GB", "r64")', source)
        self.assertNotIn('"v1.1 Rank', source)
        self.assertNotIn('"v1.1 Full', source)

    def test_model_type_falls_back_during_add_edit_form_transitions(self):
        package_name = "krea2_identity_model_type_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]
        shared = types.ModuleType("shared")
        shared_utils = types.ModuleType("shared.utils")
        shared_plugins = types.ModuleType("shared.utils.plugins")
        shared_plugins.WAN2GPPlugin = object
        module_name = f"{package_name}.plugin"
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "plugin.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                package_name: package,
                "shared": shared,
                "shared.utils": shared_utils,
                "shared.utils.plugins": shared_plugins,
            },
        ):
            spec.loader.exec_module(module)

        self.assertTrue(module._plugin_active({"model_type": "krea2_identity_turbo"}))
        self.assertTrue(
            module._plugin_active(
                {"active_form": "edit", "model_type": "krea2_identity_raw"}
            )
        )
        self.assertFalse(module._plugin_active({"model_type": "wan2.1_t2v"}))
        self.assertFalse(module._plugin_active({"active_form": "edit"}))

    def test_plugin_ui_builds_with_wangp_public_extension_contract(self):
        import gradio as gr

        class FakePluginBase:
            def __init__(self):
                self._component_requests = []
                self._insert_after_requests = []
                self._global_requests = []
                self._data_hooks = {}

            def request_component(self, component_id):
                self._component_requests.append(component_id)

            def request_global(self, global_name):
                self._global_requests.append(global_name)

            def add_custom_js(self, js_code):
                self._custom_js = js_code

            def register_data_hook(self, hook_name, callback):
                self._data_hooks.setdefault(hook_name, []).append(callback)

            def insert_after(self, target, constructor):
                self._insert_after_requests.append((target, constructor))

        package_name = "krea2_identity_ui_contract_test"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]
        shared = types.ModuleType("shared")
        shared_utils = types.ModuleType("shared.utils")
        shared_plugins = types.ModuleType("shared.utils.plugins")
        shared_plugins.WAN2GPPlugin = FakePluginBase
        module_name = f"{package_name}.plugin"
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "plugin.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                package_name: package,
                "shared": shared,
                "shared.utils": shared_utils,
                "shared.utils.plugins": shared_plugins,
            },
        ):
            spec.loader.exec_module(module)

        identity_choice_values = {
            value for _label, value in module._ALL_IDENTITY_METHOD_CHOICES
        }
        synchronized_values = {
            module._synchronized_identity_method("identity_edit_ref4", prompt_type)
            for prompt_type in ("KI", "I", "DV")
        }
        self.assertEqual(
            module._synchronized_identity_method("identity_edit_ref4", "I"),
            "identity_edit_ref4",
        )
        self.assertTrue(synchronized_values <= identity_choice_values)

        inactive_state = {"active_form": "add", "model_type": "wan2.1_t2v"}
        unchanged = {"__type__": "update"}
        self.assertEqual(
            module._refresh_shared_image_label(
                inactive_state, None, "Other Control"
            ),
            unchanged,
        )
        self.assertEqual(
            module._refresh_shared_gallery_label(
                inactive_state, None, "Other References"
            ),
            unchanged,
        )
        self.assertEqual(
            module._sync_custom_mask_upload(inactive_state, "Y", "mask.png"),
            unchanged,
        )
        self.assertEqual(
            module._refresh_custom_guide_visibility(inactive_state),
            unchanged,
        )
        self.assertEqual(
            module._refresh_context_controls(
                inactive_state, "other_method", "other_preparation", "I"
            ),
            (unchanged, unchanged, unchanged),
        )
        self.assertEqual(
            module._refresh_segmentation_phrase(
                inactive_state, "sam3", "KI"
            ),
            unchanged,
        )

        extension = module.Krea2IdentityAdvancedUI()
        extension.setup_ui()
        self.assertIn("krea2-advanced-drag-handle", extension._custom_js)
        self.assertIn("before_metadata_save", extension._data_hooks)
        cleaned_metadata = extension._data_hooks["before_metadata_save"][0](
            {
                "video_prompt_type": "KIDV",
                "custom_settings": {
                    "identity_method": "identity_edit",
                    "depth_control_strength": 1.0,
                    "_registered_outpaint_ratio": "16:9",
                    "advanced_settings": encode_advanced_settings(
                        {"builtin_adapter_timing": "simultaneous"}
                    ),
                }
            },
            plugin_data={
                "krea2_identity_output_metadata": {
                    "identity_method": "identity_edit",
                    "depth_active": True,
                    "depth_mask_active": False,
                    "reference_count": 1,
                    "user_lora_count": 0,
                }
            },
            model_type="krea2_identity_turbo",
        )
        self.assertNotIn(
            "advanced_settings", cleaned_metadata["custom_settings"]
        )
        self.assertNotIn(
            "_registered_outpaint_ratio", cleaned_metadata["custom_settings"]
        )
        self.assertEqual(
            cleaned_metadata["custom_settings"]["builtin_adapter_timing"],
            "simultaneous",
        )
        self.assertEqual(
            cleaned_metadata["extra_info"]["Builtin Adapter Timing"],
            "Simultaneous",
        )
        self.assertNotIn(
            "Builtin Depth Ramp — Early", cleaned_metadata["extra_info"]
        )
        with gr.Blocks():
            extension.state = gr.State(
                {"active_form": "add", "model_type": "krea2_identity_turbo"}
            )
            extension.refresh_form_trigger = gr.Textbox(visible=False)
            extension.video_prompt_type = gr.Textbox(value="KI", visible=False)
            extension.image_guide = gr.Image(type="pil")
            extension.image_refs = gr.Gallery()
            extension.custom_guide = gr.File(type="filepath")
            extension.get_preprocessor = lambda process_type, color: (
                lambda image: image
            )
            extension.custom_settings_visibility_trigger = gr.Textbox(visible=False)
            extension.custom_setting_text_inputs = [
                gr.Textbox(value="") for _index in range(4)
            ] + [gr.Textbox(value=encode_advanced_settings(), visible=False)]
            extension.custom_setting_dropdown_inputs = [
                gr.Dropdown(
                    choices=[("Identity Edit", "identity_edit")],
                    value="identity_edit",
                ),
                gr.Dropdown(),
                gr.Dropdown(
                    choices=[("Keep background", "off"), ("SAM3", "sam3")],
                    value="off",
                ),
                gr.Dropdown(),
                gr.Dropdown(),
            ]
            extension.post_ui_setup({})
            identity_choice_values = {
                choice[1]
                for choice in extension.custom_setting_dropdown_inputs[0].choices
            }
            self.assertIn("identity_edit_ref2", identity_choice_values)
            self.assertIn("identity_edit_ref8_scene2", identity_choice_values)
            self.assertIn("depth_prompt", identity_choice_values)
            self.assertEqual(len(extension._insert_after_requests), 2)
            constructors = dict(extension._insert_after_requests)
            self.assertEqual(
                extension._global_requests,
                ["get_preprocessor", "release_model"],
            )
            self.assertIn("custom_guide", constructors)
            self.assertIn("custom_settings_visibility_trigger", constructors)
            for constructor in constructors.values():
                panel = constructor()
                self.assertIsInstance(panel, gr.Column)

        inactive_extension = module.Krea2IdentityAdvancedUI()
        inactive_extension.setup_ui()
        with gr.Blocks():
            inactive_extension.state = gr.State(inactive_state)
            inactive_extension.refresh_form_trigger = gr.Textbox(visible=False)
            inactive_extension.video_prompt_type = gr.Textbox(value="I", visible=False)
            inactive_extension.image_guide = gr.Image(
                type="pil", label="Other Control"
            )
            inactive_extension.image_refs = gr.Gallery(label="Other References")
            inactive_extension.custom_guide = gr.File(
                type="filepath", label="Other Custom Guide", visible=False
            )
            inactive_extension.get_preprocessor = lambda process_type, color: (
                lambda image: image
            )
            inactive_extension.custom_settings_visibility_trigger = gr.Textbox(
                visible=False
            )
            inactive_payload = gr.Textbox(value="other payload", visible=True)
            inactive_segmentation = gr.Textbox(
                value="other phrase", visible=True
            )
            inactive_extension.custom_setting_text_inputs = [
                gr.Textbox(value="") for _index in range(3)
            ] + [inactive_segmentation, inactive_payload]
            inactive_identity = gr.Dropdown(
                choices=[("Other Method", "other_method")],
                value="other_method",
                visible=True,
            )
            inactive_preparation = gr.Dropdown(
                choices=[("Other Preparation", "other_preparation")],
                value="other_preparation",
                visible=True,
            )
            inactive_extension.custom_setting_dropdown_inputs = [
                inactive_identity,
                gr.Dropdown(),
                inactive_preparation,
                gr.Dropdown(),
                gr.Dropdown(),
            ]
            inactive_extension.post_ui_setup({})

            self.assertTrue(inactive_payload.visible)
            self.assertFalse(inactive_extension.custom_guide.visible)
            self.assertTrue(inactive_identity.visible)
            self.assertEqual(inactive_identity.value, "other_method")
            self.assertEqual(
                inactive_identity.choices, [("Other Method", "other_method")]
            )
            self.assertTrue(inactive_preparation.visible)
            self.assertTrue(inactive_segmentation.visible)
            self.assertEqual(
                inactive_extension.image_guide.label, "Other Control"
            )
            self.assertEqual(
                inactive_extension.image_refs.label, "Other References"
            )


if __name__ == "__main__":
    unittest.main()
