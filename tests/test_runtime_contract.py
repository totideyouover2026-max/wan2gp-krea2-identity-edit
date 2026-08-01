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
        self.assertIn("denoising_strength", self.source)
        self.assertIn("video_prompt_type", self.source)
        self.assertIn("input_frames", self.source)
        self.assertIn("input_masks", self.source)
        self.assertIn("input_custom", self.source)
        self.assertIn("input_ref_images", self.source)

    def test_wangp_checkpoint_paths_are_normalized(self):
        self.assertIn("def _resolve_wangp_checkpoint", self.source)
        self.assertIn("transformer_filename = _resolve_wangp_checkpoint", self.source)
        self.assertIn("text_encoder_filename = _resolve_wangp_checkpoint", self.source)

    def test_stream_order_is_text_sources_target(self):
        self.assertIn("torch.cat([context, *source_imgs, img]", self.source)
        self.assertIn("source_pos[..., 0] = frame", self.source)

    def test_identity_reference_boost_biases_target_to_reference_attention(self):
        self.assertIn("def _identity_reference_attention_mask", self.source)
        self.assertIn("bias = combined.new_zeros((1, 1, total_len, total_len))", self.source)
        self.assertIn("[:, :, source_end:target_end", self.source)
        self.assertIn("math.log(max(boost, 1e-4))", self.source)
        self.assertIn("padding_bias.masked_fill_(~base_mask", self.source)
        self.assertIn("reference_subject_boost=reference_subject_boost", self.source)
        self.assertIn("reference_scene_boost=reference_scene_boost", self.source)
        self.assertIn("reference attention boost active", self.source)

    def test_subject_attention_can_ramp_once_per_denoising_timestep(self):
        self.assertIn("def _identity_advance_subject_attention", self.source)
        self.assertIn("subject_attention_boost_for_step", self.source)
        self.assertIn("_identity_last_timestep", self.source)
        self.assertIn("subject_attention_timing=subject_attention_timing", self.source)
        self.assertIn("subject-attention ramp active", self.source)

    def test_secondary_reference_geometry_is_always_preserved_and_centered(self):
        self.assertIn("fit_identity_reference_geometry", self.source)
        self.assertIn("fit_secondary_reference and reference_index >= 1", self.source)
        self.assertIn("fit_secondary_reference=True", self.source)
        self.assertNotIn("secondary_reference_geometry", self.source)
        self.assertIn("self.transformer._identity_source_grids = source_grids", self.source)
        self.assertIn("offset_h = (target_grid_h - grid_h) // 2", self.source)
        self.assertIn("offset_w = (target_grid_w - grid_w) // 2", self.source)

    def test_target_only_slice_is_returned(self):
        self.assertIn("start = txtlen + srclen", self.source)
        self.assertIn("start : start + target_len", self.source)

    def test_depth_projects_target_but_not_identity_sources(self):
        self.assertIn("def _project_identity_inputs", self.source)
        self.assertIn("contribution = F.linear", self.source)
        self.assertIn("target = target + contribution", self.source)
        self.assertIn("model.first(_match_token_batch(source", self.source)
        self.assertNotIn("torch.cat([source, depth]", self.source)

    def test_depth_runtime_state_is_always_cleared(self):
        self.assertIn("self.transformer._depth_control_tokens = None", self.source)
        self.assertIn(
            "self.transformer._depth_control_runtime_weight = None", self.source
        )
        self.assertIn(
            "self.transformer._depth_control_mask_tokens = None", self.source
        )

    def test_depth_adapter_is_stacked_with_its_own_strength(self):
        self.assertIn("loras.append(depth_control_lora_url())", self.source)
        self.assertIn("multipliers.append(depth_strength)", self.source)
        self.assertIn("preprocess_krea2_adapter_state_dict", self.source)
        self.assertIn("depth_control_selected(video_prompt_type)", self.source)

    def test_depth_layout_identity_refinement_scales_both_builtin_paths(self):
        self.assertIn("schedule_builtin_identity_depth_adapters", self.source)
        self.assertIn('builtin_adapter_timing == "depth_then_identity"', self.source)
        self.assertIn("identity_ramp=builtin_identity_ramp", self.source)
        self.assertIn("depth_ramp=builtin_depth_ramp", self.source)
        self.assertIn("depth_control_strength=depth_strength", self.source)
        self.assertIn("F.linear(depth.to(target.dtype), weight) * depth_scale", self.source)

    def test_depth_can_delay_user_loras_until_geometry_is_established(self):
        self.assertIn("validate_depth_user_lora_timing", self.source)
        self.assertIn("validate_depth_user_lora_ramp", self.source)
        self.assertIn("delay_user_loras_for_depth", self.source)
        self.assertIn("phase_scales=depth_user_lora_ramp", self.source)
        self.assertIn("early={early:.2f}", self.source)
        self.assertIn("middle={middle:.2f}", self.source)
        self.assertIn("final={final:.2f}", self.source)

    def test_direct_image_identity_edit_reuses_source_restart_without_depth(self):
        self.assertIn("direct_image_control_selected", self.source)
        self.assertIn("def _wangp_control_to_pil", self.source)
        self.assertIn("direct_image_control = input_frames if direct_image_active else None", self.source)
        self.assertIn("[Krea2 Identity][Direct Image] Identity Edit refinement", self.source)
        self.assertIn("images = self.pipeline.generate_identity", self.source)
        self.assertIn("reference_images=identity_pass_references", self.source)
        self.assertIn("source_image=direct_source", self.source)
        self.assertIn("denoising_strength=direct_image_denoising", self.source)
        self.assertIn("image_mask=full_frame_mask", self.source)
        self.assertIn("depth_control=None", self.source)
        self.assertIn("if direct_image_active", self.source)
        self.assertIn("else original_references[0].size", self.source)

    def test_plugin_phase_switches_target_leading_functional_loras(self):
        self.assertIn("def _with_plugin_lora_weights", self.source)
        self.assertIn("values[: len(weights)] = list(weights)", self.source)
        self.assertNotIn("values[-len(weights) :]", self.source)

    def test_depth_uses_wangp_processed_control_tensor(self):
        self.assertIn("depth_control = input_frames if depth_active else None", self.source)
        self.assertIn("WanGP control contract: channels, frames, height, width", self.source)
        self.assertNotIn("AutoModelForDepthEstimation", self.source)
        self.assertNotIn("depth_reference_index", self.source)

    def test_input_fingerprints_are_available_only_in_debug_mode(self):
        self.assertIn("def _content_fingerprint", self.source)
        self.assertIn("KREA2_IDENTITY_DEBUG", self.source)
        self.assertIn("if _debug_diagnostics_enabled():", self.source)
        self.assertIn("[Krea2 Identity][Inputs]", self.source)
        self.assertIn("processed_depth={_content_fingerprint(depth_control)}", self.source)
        self.assertIn("mask={_content_fingerprint(depth_mask)}", self.source)

    def test_debug_mode_can_log_grounded_conditioning_fingerprint(self):
        self.assertIn("def _sampled_conditioning_diagnostics", self.source)
        self.assertIn(
            "[Krea2 Identity][Debug][Encoder] grounded conditioning:", self.source
        )
        self.assertIn("mask_source={mask_source}", self.source)

    def test_custom_depth_mask_uses_a_separate_full_depth_route(self):
        self.assertIn("load_custom_depth_mask", self.source)
        self.assertIn("def _load_custom_depth_mask_tensor", self.source)
        self.assertIn("custom_depth_mask = None", self.source)
        self.assertIn("depth_mask = custom_depth_mask", self.source)
        self.assertIn('mask_source = "custom-upload"', self.source)
        self.assertIn("custom_mask_mode", self.source)
        self.assertIn("painted_mask_mode", self.source)
        self.assertIn("host_depth=full-frame", self.source)
        self.assertIn("must use Custom Mask", self.source)
        self.assertIn("ratio_error > 0.02", self.source)

    def test_depth_mask_is_feathered_and_gates_target_projection(self):
        self.assertIn("def _prepare_depth_mask", self.source)
        self.assertIn("F.avg_pool2d", self.source)
        self.assertIn(
            "depth_pixels = depth_pixels * hard_mask - (1 - hard_mask)",
            self.source,
        )
        self.assertIn("contribution = contribution * depth_mask", self.source)
        self.assertIn("mode=\"area\"", self.source)
        self.assertIn("depth_control_mask_selected(video_prompt_type)", self.source)

    def test_depth_is_diagnosed_and_canonicalized_to_grayscale(self):
        self.assertIn("rgb_spread = depth_pixels.amax", self.source)
        self.assertIn("active_rgb_spread", self.source)
        self.assertIn(
            "depth_pixels.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)",
            self.source,
        )
        self.assertIn("channels=canonical-grayscale", self.source)

    def test_processed_references_feed_conditioning_but_original_sets_aspect(self):
        self.assertIn(
            "input_ref_images\n                if input_ref_images is not None",
            self.source,
        )
        self.assertIn("original_references[0].size", self.source)
        self.assertIn("identity_pass_references = references", self.source)

    def test_identity_then_outpaint_uses_original_upload_in_identity_phase(self):
        self.assertIn("identity_pass_references = references", self.source)
        self.assertIn(
            "identity_pass_references = original_references",
            self.source,
        )
        self.assertIn(
            "reference_images=identity_pass_references",
            self.source,
        )
        self.assertIn("identity phase reference source=", self.source)

    def test_outpaint_uses_an_isolated_prompt_and_negative_prompt(self):
        self.assertIn(
            "outpaint_prompts = [outpaint_prompt] * int(batch_size)",
            self.source,
        )
        self.assertIn(
            'outpaint_prompt = str(settings.get("outpaint_prompt", "") or "").strip()',
            self.source,
        )
        self.assertIn(
            'outpaint_common["negative_prompts"] = [',
            self.source,
        )
        self.assertIn(
            "generate_registered_outpaint(\n                outpaint_prompts,",
            self.source,
        )
        self.assertIn(
            "outpainting_ratio=registered_outpaint_ratio",
            self.source,
        )

    def test_cfg_and_interrupt_paths_exist(self):
        functions = {
            node.name for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("_identity_forward_cfg", functions)
        self.assertIn("_outpaint_forward_cfg", functions)
        self.assertIn("_interrupt", functions)

    def test_depth_prompt_bypasses_reference_mode_output_cap(self):
        self.assertIn(
            'if identity_method == "depth_prompt":\n'
            "                # There are no reference tokens competing for memory",
            self.source,
        )
        self.assertIn('{"max_pixels": None}', self.source)
        self.assertIn('"output_resolution_limit"', self.source)
        self.assertIn("[Krea2 Identity][Resolution]", self.source)

    def test_removed_native_i_ab_mode_is_absent(self):
        self.assertNotIn("single_reference_role", self.source)
        self.assertNotIn("native_single_reference", self.source)

    def test_depth_prompt_uses_no_reference_stream_or_identity_adapter(self):
        self.assertIn("def _depth_forward(", self.source)
        self.assertIn("def _depth_forward_cfg(", self.source)
        self.assertIn("def generate_depth_prompt(", self.source)
        self.assertIn("self.encoder.allow_text_only = True", self.source)
        self.assertIn(
            'self.transformer._conditioning_mode = "depth_prompt"', self.source
        )
        self.assertIn(
            'if identity_method == "depth_prompt":\n            original_references = []',
            self.source,
        )
        self.assertIn("images = self.pipeline.generate_depth_prompt", self.source)
        self.assertIn(
            "return [depth_control_lora_url()], [depth_strength]", self.source
        )

    def test_only_unified_bf16_encoder_stack_remains(self):
        handler = (ROOT / "models" / "krea2_identity_handler.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('Qwen3-VL-4B-Instruct_bf16.safetensors', handler)
        self.assertNotIn('Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors', handler)
        self.assertNotIn('Comfy-Org/Krea-2', handler)
        self.assertNotIn("def _uses_legacy_encoder_stack", self.source)
        self.assertIn('stack=unified-bf16', self.source)

    def test_qwen_processor_uses_upstream_preprocessor_configuration(self):
        self.assertIn("Qwen2VLImageProcessorFast.from_pretrained", self.source)
        self.assertIn("Krea2Qwen3VLProcessor,", self.source)
        self.assertIn("processor = Krea2Qwen3VLProcessor", self.source)

    def test_v1234_image_keyword_is_accepted_by_prompt_override(self):
        self.assertIn(
            "def _encode_prompts(self, prompts, device, dtype, images=None)",
            self.source,
        )

    def test_v1234_target_length_keyword_is_accepted_by_all_forward_modes(self):
        functions = {
            node.name: node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "_identity_forward",
            "_identity_forward_cfg",
            "_depth_forward",
            "_depth_forward_cfg",
            "_outpaint_forward_cfg",
        ):
            parameters = [argument.arg for argument in functions[name].args.args]
            self.assertIn("target_len", parameters, name)

    def test_multimodal_tokenizer_does_not_leak_max_length_into_processor(self):
        tokenizer_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "AutoTokenizer"
        ]
        self.assertEqual(len(tokenizer_calls), 1)
        keyword_names = {keyword.arg for keyword in tokenizer_calls[0].keywords}
        self.assertNotIn("max_length", keyword_names)

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
