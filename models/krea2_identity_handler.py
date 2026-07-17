"""WanGP family-handler scaffold for Krea 2 Identity Edit.

The handler is intentionally discoverable but its model definitions remain hidden
until the dual-conditioning pipeline is complete and validated.
"""

from __future__ import annotations

import os

from models.krea2.krea2_handler import family_handler as _Krea2Handler


RAW_MODEL_TYPE = "krea2_identity_raw"
TURBO_MODEL_TYPE = "krea2_identity_turbo"
_BASE_TYPES = {
    RAW_MODEL_TYPE: "krea2_raw",
    TURBO_MODEL_TYPE: "krea2_turbo",
}
_PROFILE_DIR = "krea2_identity"
_VISION_REPO = "Comfy-Org/Krea-2"
_VISION_FOLDER = "text_encoders"
_VISION_FILENAME = "qwen3vl_4b_fp8_scaled.safetensors"
MINIMUM_WANGP = "WanGP v12.3 (public API audited at commit 5582327dc25e45fec6cda0f27144d4dcf7ed104b)"


def _base_type(model_type: str) -> str:
    try:
        return _BASE_TYPES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported Krea 2 Identity Edit model type: {model_type}") from exc


class family_handler(_Krea2Handler):
    """Register Identity Edit as a separate Krea 2-derived model family."""

    @staticmethod
    def query_model_def(base_model_type, model_def):
        result = dict(_Krea2Handler.query_model_def(_base_type(base_model_type), model_def))
        result.update(
            {
                "profiles_dir": [_PROFILE_DIR],
                "preset_profiles_dir": [],
                "image_ref_choices": {
                    "choices": [
                        ("One or two references (scene, then subject)", "I"),
                    ],
                    "letters_filter": "I",
                    "default": "I",
                    "label": "Identity Edit References",
                },
                "at_least_one_image_ref_needed": True,
                "one_image_ref_only": False,
                "inpaint_support": False,
                "no_background_removal": True,
                "resolutions_categories": ["<=2k"],
                "custom_settings": [
                    {
                        "id": "grounding_px",
                        "name": "Grounding resolution",
                        "label": "Qwen3-VL grounding resolution (px)",
                        "type": "int",
                        "default": 768,
                        "min": 384,
                        "max": 1536,
                        "inc": 64,
                    },
                    {
                        "id": "identity_lora_variant",
                        "name": "Identity Edit LoRA variant",
                        "label": "Identity Edit LoRA",
                        "type": "dropdown",
                        "default": "r64",
                        "choices": [
                            ("v1.2 Full — 1.83 GB (recommended)", "full_v1.2"),
                            ("v1.1 Rank 64 — 0.46 GB", "r64"),
                            ("v1.1 Rank 128 — 0.91 GB", "r128"),
                            ("v1.1 Full — 1.83 GB", "full_v1.1"),
                        ],
                    },
                ],
            }
        )
        return result

    @staticmethod
    def query_supported_types():
        return [RAW_MODEL_TYPE, TURBO_MODEL_TYPE]

    @staticmethod
    def query_family_maps():
        compatibility = {
            RAW_MODEL_TYPE: [RAW_MODEL_TYPE, TURBO_MODEL_TYPE],
            TURBO_MODEL_TYPE: [RAW_MODEL_TYPE, TURBO_MODEL_TYPE],
        }
        return {}, compatibility

    @staticmethod
    def query_model_family():
        return "krea2_identity"

    @staticmethod
    def query_family_infos():
        return {"krea2_identity": (1151, "Krea 2 Identity Edit")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-krea2-identity",
            type=str,
            default=None,
            help=(
                "Path containing Krea 2 Identity Edit LoRAs "
                f"(default: {os.path.join(lora_root, 'krea2_identity')})"
            ),
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_krea2_identity", None) or os.path.join(
            lora_root, "krea2_identity"
        )

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        base_files = list(_Krea2Handler.query_model_files(
            computeList, _base_type(base_model_type), model_def=model_def
        ))
        base_files.append(
            {
                "repoId": _VISION_REPO,
                "sourceFolderList": [_VISION_FOLDER],
                "fileList": [[_VISION_FILENAME]],
            }
        )
        return base_files

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=None,
        VAE_dtype=None,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        try:
            from models.krea2.krea2_main import Krea2Pipeline  # noqa: F401
            from models.ideogram4.qwen3_vl_transformers import Qwen3VLVisionModel  # noqa: F401
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "This Krea 2 Identity Edit plugin requires " + MINIMUM_WANGP + ". "
                "Update WanGP, restart it, and try again."
            ) from exc
        from .krea2_identity_main import model_factory

        if dtype is None or VAE_dtype is None:
            import torch

            dtype = torch.bfloat16 if dtype is None else dtype
            VAE_dtype = torch.float32 if VAE_dtype is None else VAE_dtype
        processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            save_quantized=save_quantized,
        )
        return processor, {
            "transformer": processor.transformer,
            "text_encoder": processor.text_encoder,
            "vae": processor.vae,
        }

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        _Krea2Handler.update_default_settings(
            _base_type(base_model_type), model_def, ui_defaults
        )
        ui_defaults.update(
            {
                "video_prompt_type": "I",
                "custom_settings": {
                    "grounding_px": 768,
                    "identity_lora_variant": "r64",
                },
            }
        )
        if base_model_type == TURBO_MODEL_TYPE:
            ui_defaults["num_inference_steps"] = 10
            ui_defaults["guidance_scale"] = 0
        else:
            ui_defaults["num_inference_steps"] = 20
            # TODO: confirm the WanGP UI-to-effective-CFG mapping with golden tests.
            ui_defaults["guidance_scale"] = 2.0

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        _Krea2Handler.fix_settings(
            _base_type(base_model_type), settings_version, model_def, ui_defaults
        )
        ui_defaults.setdefault("video_prompt_type", "I")
        custom_settings = ui_defaults.setdefault("custom_settings", {})
        if isinstance(custom_settings, dict):
            custom_settings.setdefault("grounding_px", 768)
            custom_settings.setdefault("identity_lora_variant", "r64")
