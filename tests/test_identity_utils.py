from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from models.krea2_identity_utils import (
    delay_user_loras_for_depth,
    schedule_builtin_identity_depth_adapters,
    depth_control_mask_selected,
    depth_control_selected,
    depth_control_lora_url,
    direct_image_control_selected,
    fit_identity_reference_geometry,
    fit_reference_pixel_budget,
    identity_lora_url,
    match_reference_dimensions,
    migrate_generation_process,
    preprocess_identity_lora_state_dict,
    preprocess_krea2_adapter_state_dict,
    resolve_generation_process,
    resolve_identity_reference_boosts,
    subject_attention_boost_for_step,
    resolve_wangp_checkpoint,
    outpaint_lora_url,
    reid_lora_url,
    validate_outpaint_mode,
    validate_outpaint_seam_px,
    validate_phase2_denoising_strength,
    validate_phase2_depth_mode,
    validate_depth_control_strength,
    validate_depth_user_lora_ramp,
    validate_depth_user_lora_timing,
    validate_direct_image_denoising_strength,
    validate_depth_mask_feather_px,
    validate_grounding_px,
    validate_generation_process,
    validate_identity_method,
    validate_reference_images,
    validate_reid_reference_images,
    validate_reid_lora_strength,
    validate_subject_attention_ramp,
    validate_subject_attention_timing,
    validate_two_phase_mode,
)


class FakeImage:
    def __init__(self, size=(1200, 800)):
        self.size = size

    def convert(self, mode):
        return self


class FakeExpandedWeight:
    shape = (6144, 128)

    def __init__(self):
        self.last_slice = None

    def __getitem__(self, item):
        self.last_slice = item
        return self

    def contiguous(self):
        return self


class IdentityUtilsTests(unittest.TestCase):
    def test_reid_lora_strength_is_bounded(self):
        self.assertEqual(validate_reid_lora_strength(None), 1.0)
        self.assertEqual(validate_reid_lora_strength("0.5"), 0.5)
        self.assertEqual(validate_reid_lora_strength(2), 2.0)
        for invalid in (-0.1, 2.1, True, "bad", float("inf")):
            with self.assertRaises(ValueError):
                validate_reid_lora_strength(invalid)

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

    def test_output_cap_can_be_disabled_without_exceeding_selected_area(self):
        width, height = match_reference_dimensions(
            2560,
            1440,
            (1600, 900),
            max_pixels=None,
        )
        self.assertEqual((width, height), (2560, 1440))
        self.assertEqual(width * height, 2560 * 1440)

    def test_secondary_reference_fit_preserves_aspect_and_alignment(self):
        size, crop = fit_identity_reference_geometry(
            (720, 1280),
            (1280, 720),
        )
        self.assertEqual(size, (400, 720))
        self.assertIsNone(crop)
        self.assertAlmostEqual(size[0] / size[1], 720 / 1280, delta=0.01)
        self.assertEqual((size[0] % 16, size[1] % 16), (0, 0))

    def test_near_matching_secondary_reference_is_centre_cropped(self):
        size, crop = fit_identity_reference_geometry(
            (1600, 950),
            (1280, 720),
        )
        self.assertEqual(size, (1280, 720))
        self.assertIsNotNone(crop)
        left, top, right, bottom = crop
        self.assertEqual(left, 0)
        self.assertEqual(right, 1600)
        self.assertGreater(top, 0)
        self.assertLess(bottom, 950)

    def test_lora_variants_are_exact_published_files(self):
        self.assertTrue(identity_lora_url("full_v1.2").endswith("v1_2.safetensors"))
        self.assertTrue(identity_lora_url("r128").endswith("v1_2_r128.safetensors"))
        self.assertTrue(identity_lora_url("r64").endswith("v1_2_r64.safetensors"))
        self.assertTrue(identity_lora_url("full").endswith("v1_2.safetensors"))
        self.assertTrue(identity_lora_url("full_v1.1").endswith("v1_2.safetensors"))
        with self.assertRaises(ValueError):
            identity_lora_url("v1")

    def test_reid_contract_uses_published_reference_budget(self):
        reference = FakeImage((1600, 900))
        self.assertEqual(validate_identity_method(None), "identity_edit")
        self.assertEqual(validate_identity_method("reid"), "reid")
        self.assertEqual(validate_identity_method("depth_prompt"), "depth_prompt")
        with self.assertRaises(ValueError):
            validate_identity_method("stacked")
        self.assertEqual(validate_reid_reference_images([reference]), [reference])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_reid_reference_images([reference, FakeImage()])
        width, height = fit_reference_pixel_budget(reference.size)
        self.assertLessEqual(width * height, 384 * 384)
        self.assertEqual((width % 16, height % 16), (0, 0))
        self.assertAlmostEqual(width / height, 16 / 9, delta=0.08)
        self.assertTrue(reid_lora_url().endswith("krea2_reid_rank32.safetensors"))

    def test_identity_edit_reference_fidelity_profiles_are_opt_in(self):
        self.assertEqual(validate_identity_method("identity_edit_ref4"), "identity_edit")
        self.assertEqual(
            resolve_identity_reference_boosts("identity_edit"), (1.0, 1.0)
        )
        self.assertEqual(
            resolve_identity_reference_boosts("identity_edit_ref2"), (2.0, 1.0)
        )
        self.assertEqual(
            resolve_identity_reference_boosts("identity_edit_ref4"), (4.0, 1.0)
        )
        self.assertEqual(
            resolve_identity_reference_boosts("identity_edit_ref8"), (8.0, 1.0)
        )
        self.assertEqual(
            resolve_identity_reference_boosts("identity_edit_ref4_scene2"),
            (4.0, 2.0),
        )
        self.assertEqual(
            resolve_identity_reference_boosts("identity_edit_ref8_scene2"),
            (8.0, 2.0),
        )
        self.assertEqual(resolve_identity_reference_boosts("reid"), (1.0, 1.0))
        with self.assertRaises(ValueError):
            resolve_identity_reference_boosts("identity_edit_ref1000")

    def test_subject_attention_ramp_uses_denoising_thirds(self):
        self.assertEqual(validate_subject_attention_timing(None), "constant")
        self.assertEqual(validate_subject_attention_timing("ramp"), "ramp")
        with self.assertRaises(ValueError):
            validate_subject_attention_timing("late")
        self.assertEqual(
            validate_subject_attention_ramp(None, None, None),
            (1.0, 2.0, 8.0),
        )
        self.assertEqual(
            validate_subject_attention_ramp("1.5", 3, 6.5),
            (1.5, 3.0, 6.5),
        )
        for invalid in ((0.5, 2, 8), (1, 2, 8.5), (1, True, 8)):
            with self.assertRaises(ValueError):
                validate_subject_attention_ramp(*invalid)
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            validate_subject_attention_ramp(1, 6, 4)

        constant = [
            subject_attention_boost_for_step(4, "constant", (1, 2, 8), step, 8)
            for step in range(8)
        ]
        self.assertEqual(constant, [(4.0, 0)] * 8)
        ramped = [
            subject_attention_boost_for_step(4, "ramp", (1, 2, 8), step, 8)
            for step in range(8)
        ]
        self.assertEqual(
            ramped,
            [(1.0, 0)] * 3 + [(2.0, 1)] * 3 + [(8.0, 2)] * 2,
        )

    def test_two_phase_settings_are_opt_in_and_low_noise(self):
        self.assertEqual(validate_two_phase_mode(None), "off")
        self.assertEqual(validate_two_phase_mode("depth_then_reid"), "depth_then_reid")
        self.assertEqual(validate_phase2_denoising_strength(None), 0.25)
        self.assertEqual(validate_phase2_denoising_strength("0.15"), 0.15)
        self.assertEqual(validate_phase2_depth_mode(None), "keep")
        self.assertEqual(validate_phase2_depth_mode("off"), "off")
        for invalid in (0.0, 0.51, float("inf"), True, "bad"):
            with self.assertRaises(ValueError):
                validate_phase2_denoising_strength(invalid)
        with self.assertRaises(ValueError):
            validate_two_phase_mode("replace_standard")
        with self.assertRaises(ValueError):
            validate_phase2_depth_mode("sometimes")

    def test_visible_generation_profiles_restore_hidden_controls(self):
        self.assertEqual(validate_generation_process(None), "standard")
        self.assertEqual(
            resolve_generation_process({"generation_process": "two_phase_015_keep"}),
            ("depth_then_reid", 0.15, "keep", "off"),
        )
        self.assertEqual(
            resolve_generation_process({"generation_process": "two_phase_025_off"}),
            ("depth_then_reid", 0.25, "off", "off"),
        )
        self.assertEqual(
            resolve_generation_process({"generation_process": "outpaint_only"}),
            ("off", 0.25, "keep", "outpaint_only"),
        )
        self.assertEqual(
            migrate_generation_process(
                {
                    "two_phase_mode": "depth_then_reid",
                    "phase2_denoising_strength": 0.25,
                    "phase2_depth_mode": "keep",
                }
            ),
            "two_phase_025_keep",
        )
        with self.assertRaises(ValueError):
            validate_generation_process("custom")

    def test_registered_outpaint_settings_and_asset(self):
        self.assertTrue(outpaint_lora_url().endswith("krea2_outpaint_rank32.safetensors"))
        self.assertEqual(validate_outpaint_mode(None), "off")
        self.assertEqual(validate_outpaint_mode("outpaint_only"), "outpaint_only")
        self.assertEqual(validate_outpaint_seam_px(None), 32)
        self.assertEqual(validate_outpaint_seam_px("64"), 64)
        for invalid in ("other", 1):
            with self.assertRaises(ValueError):
                validate_outpaint_mode(invalid)
        for invalid in (-1, 129, 2.5, True):
            with self.assertRaises(ValueError):
                validate_outpaint_seam_px(invalid)

    def test_lora_key_conversion_is_minimal(self):
        marker = object()
        converted = preprocess_identity_lora_state_dict(
            {"diffusion_model.blocks.0.attn.wq.lora_A.weight": marker}
        )
        self.assertEqual(
            converted, {"blocks.0.attn.wq.lora_A.weight": marker}
        )

    def test_depth_control_settings_are_bounded_and_dropdown_driven(self):
        self.assertFalse(depth_control_selected(None))
        self.assertFalse(depth_control_selected("I"))
        self.assertFalse(depth_control_selected("IVG"))
        self.assertTrue(depth_control_selected("IDV"))
        self.assertTrue(direct_image_control_selected("IVG"))
        self.assertFalse(direct_image_control_selected("IDV"))
        self.assertEqual(validate_direct_image_denoising_strength(None), 0.25)
        self.assertEqual(validate_direct_image_denoising_strength("0.35"), 0.35)
        for invalid in (-0.1, 1.1, float("inf"), True, "bad"):
            with self.assertRaises(ValueError):
                validate_direct_image_denoising_strength(invalid)
        self.assertEqual(validate_depth_control_strength(None), 1.0)
        self.assertEqual(validate_depth_control_strength("0.65"), 0.65)
        for invalid in (-0.1, 2.1, float("inf"), True, "bad"):
            with self.assertRaises(ValueError):
                validate_depth_control_strength(invalid)
        self.assertFalse(depth_control_mask_selected("KIDV"))
        self.assertTrue(depth_control_mask_selected("KIDVA"))
        self.assertTrue(depth_control_mask_selected("KIDVNA"))
        self.assertTrue(depth_control_mask_selected("KIDVY"))
        self.assertTrue(depth_control_mask_selected("KIDVNY"))
        self.assertEqual(validate_depth_mask_feather_px(None), 16)
        self.assertEqual(validate_depth_mask_feather_px("24"), 24)
        for invalid in (-1, 65, 2.5, True, "bad"):
            with self.assertRaises(ValueError):
                validate_depth_mask_feather_px(invalid)
        self.assertEqual(validate_depth_user_lora_timing(None), "depth_first")
        self.assertEqual(validate_depth_user_lora_timing("all_steps"), "all_steps")
        with self.assertRaises(ValueError):
            validate_depth_user_lora_timing("late_only")
        self.assertEqual(
            validate_depth_user_lora_ramp(None, None, None),
            (0.0, 0.25, 1.0),
        )
        self.assertEqual(
            validate_depth_user_lora_ramp("0.1", 0.5, 0.9),
            (0.1, 0.5, 0.9),
        )
        for invalid in ((-0.1, 0.5, 1.0), (0.0, 1.1, 1.0), (0.0, 0.5, True)):
            with self.assertRaises(ValueError):
                validate_depth_user_lora_ramp(*invalid)
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            validate_depth_user_lora_ramp(0.0, 0.75, 0.5)

    def test_depth_first_schedule_delays_only_user_loras(self):
        schedules = {
            "transformer": {
                "phase1": [1.0, 1.0, 0.01, 0.4],
                "phase2": [1.0, 1.0, 0.01, 0.4],
                "phase3": [1.0, 1.0, 0.01, 0.4],
                "shared": [True, True, False, False],
            }
        }
        delayed, count = delay_user_loras_for_depth(schedules, 2, 6)
        self.assertEqual(count, 2)
        unlocker = [0.0, 0.0, 0.0025, 0.0025, 0.01, 0.01]
        style = [0.0, 0.0, 0.1, 0.1, 0.4, 0.4]
        for phase in ("phase1", "phase2", "phase3"):
            self.assertEqual(
                delayed["transformer"][phase],
                [1.0, 1.0, unlocker, style],
            )
        self.assertEqual(delayed["transformer"]["shared"], [True, True, True, True])
        self.assertEqual(schedules["transformer"]["phase1"], [1.0, 1.0, 0.01, 0.4])

        custom, count = delay_user_loras_for_depth(
            schedules, 2, 6, phase_scales=(0.1, 0.5, 0.9)
        )
        self.assertEqual(count, 2)
        self.assertEqual(
            custom["transformer"]["phase1"][2],
            [0.001, 0.001, 0.005, 0.005, 0.009000000000000001, 0.009000000000000001],
        )

    def test_builtin_identity_depth_schedule_ramps_the_leading_adapters(self):
        schedules = {
            "transformer": {
                "phase1": [1.0, 0.8, 0.4],
                "phase2": [1.0, 0.8, 0.4],
                "phase3": [1.0, 0.8, 0.4],
                "shared": [False, False, False],
            }
        }
        ramped = schedule_builtin_identity_depth_adapters(schedules, 6)
        expected_identity = [0.25, 0.25, 0.75, 0.75, 1.0, 1.0]
        expected_depth = [0.8, 0.8, 0.4, 0.4, 0.0, 0.0]
        for phase in ("phase1", "phase2", "phase3"):
            self.assertEqual(ramped["transformer"][phase][0], expected_identity)
            self.assertEqual(ramped["transformer"][phase][1], expected_depth)
            self.assertEqual(ramped["transformer"][phase][2], 0.4)
        self.assertEqual(ramped["transformer"]["shared"], [True, True, False])
        self.assertEqual(schedules["transformer"]["phase1"], [1.0, 0.8, 0.4])

    def test_depth_adapter_keys_and_projection_are_split_for_mmgp(self):
        expanded = FakeExpandedWeight()
        marker_a, marker_b = object(), object()
        converted, control_weight = preprocess_krea2_adapter_state_dict(
            {
                "first.weight": expanded,
                "first.bias": object(),
                "blocks.0.attn.wq.A": marker_a,
                "blocks.0.attn.wq.B": marker_b,
            }
        )
        self.assertIs(control_weight, expanded)
        self.assertEqual(expanded.last_slice, (slice(None), slice(64, None)))
        self.assertEqual(
            converted,
            {
                "blocks.0.attn.wq.lora_A.weight": marker_a,
                "blocks.0.attn.wq.lora_B.weight": marker_b,
            },
        )
        self.assertTrue(
            depth_control_lora_url().endswith("depth-control-lora.safetensors")
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
