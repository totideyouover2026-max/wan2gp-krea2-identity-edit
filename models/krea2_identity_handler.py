"""WanGP family handler for the experimental Krea 2 Identity Edit preview."""

from __future__ import annotations

import os

from models.krea2.krea2_handler import (  # pyright: ignore[reportMissingImports]
    family_handler as _Krea2Handler,
)

from .krea2_advanced_settings import (
    ADVANCED_SETTINGS_ID,
    OUTPUT_METADATA_CONTEXT_KEY,
    encode_advanced_settings,
    expand_advanced_settings,
)
from .krea2_identity_prompt import (
    DEFAULT_IDENTITY_PROMPT,
    prompt_infos,
)
from .krea2_identity_utils import (
    depth_control_selected,
    direct_image_control_selected,
    resolve_generation_process,
    resolve_identity_reference_boosts,
    validate_depth_control_strength,
    validate_depth_user_lora_ramp,
    validate_depth_user_lora_timing,
    validate_direct_image_denoising_strength,
    validate_identity_method,
    validate_subject_attention_ramp,
    validate_subject_attention_timing,
)


RAW_MODEL_TYPE = "krea2_identity_raw"
TURBO_MODEL_TYPE = "krea2_identity_turbo"
_BASE_TYPES = {
    RAW_MODEL_TYPE: "krea2_raw",
    TURBO_MODEL_TYPE: "krea2_turbo",
}
_PROFILE_DIR = "krea2_identity"
_BF16_TEXT_ENCODER_URL = (
    "https://huggingface.co/DeepBeepMeep/krea-2/resolve/main/"
    "Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_bf16.safetensors"
)
_SAM3_REPO = "DeepBeepMeep/Wan2.1"
_SAM3_FOLDER = "sam3"
_SAM3_FILES = [
    "sam3.1_multiplex_bf16.safetensors",
    "bpe_simple_vocab_16e6.txt.gz",
]
MINIMUM_WANGP = "WanGP v12.34 (public API audited at commit 6b92c54f92bde24d6d309d6f61249353b0ec783d)"


def _item_count(value) -> int:
    if value is None:
        return 0
    root = getattr(value, "root", None)
    if root is not None:
        value = root
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        return int(bool(value.strip()))
    return 1


def _composite_subject_on_white(image, mask):
    """Protect a binary semantic silhouette while gently softening its edge."""
    from PIL import Image, ImageChops, ImageFilter

    source = image.convert("RGB")
    alpha = mask.convert("L")
    if alpha.size != source.size:
        alpha = alpha.resize(source.size, resample=Image.Resampling.NEAREST)
    if alpha.getbbox() is None:
        raise ValueError(
            "SAM3 did not find the subject. Change the Subject segmentation phrase "
            "to a short visual description such as 'person' or 'blue car'."
        )
    # SAM3 is intentionally binary. A small outward expansion protects narrow
    # limbs, hair and clothing that would otherwise be clipped; the following
    # one-pixel feather avoids a visibly jagged white boundary.
    protected_core = alpha
    feathered_edge = alpha.filter(ImageFilter.MaxFilter(3))
    feathered_edge = feathered_edge.filter(ImageFilter.GaussianBlur(radius=1.0))
    alpha = ImageChops.lighter(protected_core, feathered_edge)
    result = Image.new("RGB", source.size, (255, 255, 255))
    result.paste(source, (0, 0), alpha)
    return result


def _postprocess_identity_references(
    images,
    masks,
    _width,
    _height,
    _image_start,
    _image_prompt_type,
    _image_end,
    _video_prompt_type,
    send_cmd,
    _model_def,
    custom_settings,
    **_kwargs,
):
    """Isolate the active subject reference without PyMatting Cholesky."""
    settings = custom_settings if isinstance(custom_settings, dict) else {}
    try:
        settings = expand_advanced_settings(settings)
    except ValueError:
        # Generation validation reports malformed packed settings. Preserve the
        # visible reference-preparation values here so preprocessing itself
        # does not silently change behavior before that error is surfaced.
        settings = dict(settings)
    removal_mode = settings.get("subject_background_removal", "off")
    if removal_mode not in {"stable", "sam3"}:
        return images, masks
    identity_method = validate_identity_method(settings.get("identity_method"))
    if identity_method == "depth_prompt":
        return images, masks
    subject_index = 1
    if images is None or len(images) <= subject_index:
        return images, masks

    output = list(images)
    if removal_mode == "sam3":
        if callable(send_cmd):
            send_cmd("progress", [0, "Segmenting Subject Reference with SAM3"])
        phrase = str(settings.get("subject_segmentation_prompt", "person") or "").strip()
        if not phrase:
            phrase = "person"
        try:
            from shared.magic_mask import generate_image_mask

            subject, semantic_mask, _keywords = generate_image_mask(
                output[subject_index].convert("RGB"), phrase
            )
        except (FileNotFoundError, ImportError) as exc:
            raise RuntimeError(
                "SAM3 subject isolation requires WanGP's Magic Mask assets and API. "
                "Update WanGP and allow the declared SAM3 assets to download."
            ) from exc
        output[subject_index] = _composite_subject_on_white(subject, semantic_mask)
    else:
        if callable(send_cmd):
            send_cmd("progress", [0, "Removing Subject Reference Background"])
        from rembg import remove
        from shared.utils.utils import new_rembg_session

        session = new_rembg_session()
        # Identity Edit isolates reference 2. Use direct predicted alpha rather
        # than closed-form alpha matting.
        output[subject_index] = remove(
            output[subject_index].convert("RGB"),
            session=session,
            alpha_matting=False,
            bgcolor=(255, 255, 255, 255),
        ).convert("RGB")
    return output, masks


def _base_type(model_type: str) -> str:
    try:
        return _BASE_TYPES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported Krea 2 Identity Edit model type: {model_type}") from exc


class family_handler(_Krea2Handler):
    """Register Identity Edit as a separate Krea 2-derived model family."""

    @staticmethod
    def query_model_def(base_model_type, model_def):
        # WanGP appends model-definition LoRAs to those returned dynamically by
        # get_loras_transformer. Clear legacy fixed entries in-place because
        # WanGP merges this input over the handler defaults after this call.
        model_def["loras"] = []
        model_def["loras_multipliers"] = []
        image_ref_choices = [
            ("Identity Edit — scene, then optional subject", "KI"),
            ("Depth + prompt only — no reference", ""),
        ]
        identity_method_choices = [
            ("Identity Edit — standard fidelity (1x / 1x)", "identity_edit"),
            ("Identity Edit — subject fidelity 2x", "identity_edit_ref2"),
            ("Identity Edit — subject fidelity 4x (experimental)", "identity_edit_ref4"),
            ("Identity Edit — subject fidelity 8x (experimental)", "identity_edit_ref8"),
            ("Identity Edit — subject 4x + scene 2x (two refs)", "identity_edit_ref4_scene2"),
            ("Identity Edit — subject 8x + scene 2x (two refs)", "identity_edit_ref8_scene2"),
        ]
        identity_method_choices.append(
            ("Depth + prompt only — no identity reference", "depth_prompt")
        )
        guide_preprocessing = {
            "selection": ["", "VG", "DV"],
            "labels": {
                "": "No Control Image",
                "VG": "Direct Image → Identity Edit",
                "DV": "Transfer Depth",
            },
            "default": "",
            "label": "Control Image Process",
        }
        subject_preparation_choices = [
            ("Keep background", "off"),
            ("SAM3 semantic isolation (recommended)", "sam3"),
            ("rembg fast isolation (legacy)", "stable"),
        ]
        result = dict(_Krea2Handler.query_model_def(_base_type(base_model_type), model_def))
        result.update(
            {
                "profiles_dir": [_PROFILE_DIR],
                "preset_profiles_dir": [],
                "text_encoder_URLs": [_BF16_TEXT_ENCODER_URL],
                "prompt_infos": prompt_infos(),
                "image_ref_choices": {
                    "choices": image_ref_choices,
                    "letters_filter": "KI",
                    "default": "KI",
                    "label": "Identity Edit References",
                },
                # Depth + prompt is a real zero-reference mode. Identity Edit
                # retains its stricter runtime reference validation.
                "at_least_one_image_ref_needed": False,
                "one_image_ref_only": False,
                "inpaint_support": False,
                # Identity Edit does not use Krea's LanPaint modes.
                "model_modes": None,
                # Reuse WanGP's native margin/aspect-ratio controls. The
                # registered adapter consumes the resolved rectangle itself.
                "video_guide_outpainting": [1],
                "outpainting_quantize_margins": 16,
                # WanGP's native path forces PyMatting alpha matting, whose
                # incomplete-Cholesky preconditioner is unstable for some
                # reference masks. Keep it disabled and isolate reference 2 in
                # the plugin postprocessor with semantic SAM3 or direct rembg.
                "no_background_removal": True,
                "custom_image_ref_postprocessor": _postprocess_identity_references,
                # WanGP exposes this generic, queue-persisted file slot to
                # plugins as `input_custom`. The runtime treats it as an
                # optional replacement for the mask painted in ImageEditor.
                "custom_guide": {
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
                "resolutions_categories": ["<=2k"],
                "guide_preprocessing": guide_preprocessing,
                "mask_preprocessing": {
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
                # WanGP v12.34 surfaces/persists only five model-defined custom
                # controls. The fifth is an intentionally hidden JSON carrier
                # used by plugin.py's Advanced Settings modal.
                "custom_settings": [
                    {
                        "id": "identity_method",
                        "name": "Identity method / reference fidelity",
                        "label": "Identity method / reference fidelity",
                        "type": "dropdown",
                        "default": "identity_edit",
                        "choices": identity_method_choices,
                        "info": (
                            "Identity Edit fidelity profiles reproduce the upstream "
                            "target-to-reference attention boost. The subject is the "
                            "last reference; any earlier reference is treated as the "
                            "scene. Standard 1x / 1x preserves existing behavior. "
                            "Depth + prompt loads only the depth adapter and requires "
                            "Transfer Depth to be selected."
                        ),
                        # The top Identity Edit References selector owns the
                        # Identity Edit/Depth-only family. plugin.py keeps
                        # this dropdown visible only for Identity Edit fidelity.
                        "video_prompt_type": "K",
                    },
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
                    {
                        "id": "subject_background_removal",
                        "name": "Subject reference preparation",
                        "label": "Subject reference preparation",
                        "type": "dropdown",
                        "default": "off",
                        "choices": subject_preparation_choices,
                        "info": (
                            "SAM3/rembg isolate the second reference while preserving "
                            "the first scene reference."
                        ),
                        "video_prompt_type": "IK",
                    },
                    {
                        "id": "subject_segmentation_prompt",
                        "name": "Subject segmentation phrase",
                        "label": "Subject segmentation phrase (SAM3)",
                        "type": "text",
                        "default": "person",
                        "info": (
                            "Short visual phrase identifying the subject reference, for example "
                            "'person', 'woman on the left', or 'blue car'."
                        ),
                        "video_prompt_type": "IK",
                    },
                    {
                        "id": ADVANCED_SETTINGS_ID,
                        "name": "Krea 2 advanced settings data",
                        "label": "Krea 2 advanced settings data",
                        "type": "text",
                        "default": encode_advanced_settings(),
                        # Keep the serialized carrier out of the form. plugin.py
                        # provides the visible Yes/No launcher and modal.
                        "video_prompt_type": "~",
                    },
                ],
            }
        )
        # WanGP's native denoising_strength control has a host-wide hard-coded
        # 0..1 range. Depth uses the dedicated custom setting above instead.
        result.pop("custom_denoising_strength", None)
        result.pop("denoising_strength", None)
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
                "repoId": _SAM3_REPO,
                "sourceFolderList": [_SAM3_FOLDER],
                "fileList": [list(_SAM3_FILES)],
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
            from models.krea2.krea2_main import (  # pyright: ignore[reportMissingImports]
                Krea2Qwen3VLProcessor,  # noqa: F401
                Krea2Pipeline,  # noqa: F401
            )
            from models.ideogram4.qwen3_vl_transformers import (  # pyright: ignore[reportMissingImports]
                Qwen3VLVisionModel,  # noqa: F401
            )
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
                "video_prompt_type": "KI",
                "prompt": DEFAULT_IDENTITY_PROMPT,
                "remove_background_images_ref": 0,
                # Hidden unless WanGP's raw VG Control Image route is selected.
                # Depth and ordinary reference generation are forced to full
                # denoising by the host because their prompt type has no G.
                "denoising_strength": 0.25,
                "custom_guide": None,
                "custom_settings": {
                    "identity_method": "identity_edit",
                    "depth_control_strength": 1.0,
                    "grounding_px": 768,
                    "identity_lora_variant": "full_v1.2",
                    "subject_attention_timing": "constant",
                    "subject_attention_ramp_early": 1.0,
                    "subject_attention_ramp_middle": 2.0,
                    "subject_attention_ramp_final": 8.0,
                    "depth_mask_feather_px": 16,
                    "depth_user_lora_timing": "depth_first",
                    "depth_user_lora_ramp_early": 0.0,
                    "depth_user_lora_ramp_middle": 0.25,
                    "depth_user_lora_ramp_final": 1.0,
                    "subject_background_removal": "off",
                    "subject_segmentation_prompt": "person",
                    "generation_process": "standard",
                    ADVANCED_SETTINGS_ID: encode_advanced_settings(),
                },
            }
        )
        if base_model_type == TURBO_MODEL_TYPE:
            ui_defaults["num_inference_steps"] = 8
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
        existing_custom_settings = ui_defaults.get("custom_settings", {})
        existing_custom_settings = (
            existing_custom_settings
            if isinstance(existing_custom_settings, dict)
            else {}
        )
        try:
            identity_method = validate_identity_method(
                existing_custom_settings.get("identity_method")
            )
        except ValueError:
            identity_method = "identity_edit"
            existing_custom_settings["identity_method"] = "identity_edit"
        video_prompt_type = str(ui_defaults.get("video_prompt_type", "") or "")
        if identity_method == "depth_prompt":
            video_prompt_type = video_prompt_type.replace("K", "").replace("I", "")
        elif "I" in video_prompt_type and "K" not in video_prompt_type:
            video_prompt_type = "K" + video_prompt_type
        if "D" in video_prompt_type and "V" in video_prompt_type:
            # G displays WanGP's shared 0..1 denoising slider. This plugin uses
            # its own 0..2 depth-adapter multiplier instead.
            video_prompt_type = video_prompt_type.replace("G", "")
        if (
            ui_defaults.get("custom_guide") is not None
            and "D" in video_prompt_type
            and "V" in video_prompt_type
            and "A" in video_prompt_type
        ):
            # Early custom-mask builds reused the painted A/NA routes. WanGP
            # composites those before model invocation, replacing unpainted
            # depth with source RGB. Move saved tasks to the full-depth Y/NY
            # routes while preserving white/black polarity.
            video_prompt_type = video_prompt_type.replace("A", "Y")
        ui_defaults["video_prompt_type"] = (
            video_prompt_type
            if identity_method == "depth_prompt"
            else video_prompt_type or "KI"
        )
        legacy_remove_background = int(
            ui_defaults.get("remove_background_images_ref", 0) or 0
        )
        # Disable WanGP/PyMatting processing. Migrate an enabled native toggle
        # to this plugin's stable subject-only remover.
        ui_defaults["remove_background_images_ref"] = 0
        ui_defaults.setdefault("custom_guide", None)
        ui_defaults.setdefault("denoising_strength", 1.0)
        custom_settings = ui_defaults.setdefault("custom_settings", {})
        if isinstance(custom_settings, dict):
            custom_settings.setdefault("identity_method", "identity_edit")
            native_depth_strength = ui_defaults.get("denoising_strength", 1.0)
            try:
                custom_settings.setdefault(
                    "depth_control_strength",
                    validate_depth_control_strength(native_depth_strength),
                )
                custom_settings["depth_control_strength"] = (
                    validate_depth_control_strength(
                        custom_settings["depth_control_strength"]
                    )
                )
            except ValueError:
                custom_settings["depth_control_strength"] = 1.0
            custom_settings.setdefault(
                "subject_background_removal",
                "stable" if legacy_remove_background else "off",
            )
            custom_settings.setdefault("subject_segmentation_prompt", "person")
            try:
                custom_settings.update(expand_advanced_settings(custom_settings))
            except ValueError:
                custom_settings.update(expand_advanced_settings({}))
            for legacy_process_setting in (
                "outpaint_mode",
            ):
                custom_settings.pop(legacy_process_setting, None)
            # The initial preview derived depth from an identity reference.
            # Do not migrate that toggle to the new independent control-image
            # path because an old preset cannot contain the required upload.
            custom_settings.pop("depth_control_enabled", None)
            custom_settings.pop("depth_control_source", None)
            custom_settings.pop("single_reference_role", None)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        settings = inputs.get("custom_settings", {})
        settings = settings if isinstance(settings, dict) else {}
        try:
            settings = expand_advanced_settings(settings)
        except ValueError as exc:
            return str(exc)
        inputs["custom_settings"] = settings
        identity_profile = settings.get("identity_method")
        identity_method = validate_identity_method(identity_profile)
        subject_boost, scene_boost = resolve_identity_reference_boosts(
            identity_profile
        )
        subject_attention_timing = validate_subject_attention_timing(
            settings.get("subject_attention_timing")
        )
        subject_attention_ramp = validate_subject_attention_ramp(
            settings.get("subject_attention_ramp_early"),
            settings.get("subject_attention_ramp_middle"),
            settings.get("subject_attention_ramp_final"),
        )
        attention_boost_active = (
            subject_boost != 1.0
            or scene_boost != 1.0
            or (
                subject_attention_timing == "ramp"
                and any(boost != 1.0 for boost in subject_attention_ramp)
            )
        )
        if attention_boost_active and float(
            inputs.get("NAG_scale", 1.0) or 1.0
        ) != 1.0:
            return (
                "Identity Edit reference-fidelity boosts currently require NAG "
                "scale 1.0 because both features use the transformer attention mask."
            )
        video_prompt_type = str(inputs.get("video_prompt_type", "") or "")
        if identity_method == "depth_prompt":
            video_prompt_type = video_prompt_type.replace("K", "").replace("I", "")
        elif "I" in video_prompt_type and "K" not in video_prompt_type:
            video_prompt_type = "K" + video_prompt_type
        inputs["video_prompt_type"] = video_prompt_type
        depth_selected = depth_control_selected(video_prompt_type)
        direct_image_selected = direct_image_control_selected(video_prompt_type)
        settings.pop("_registered_outpaint_ratio", None)
        validate_depth_user_lora_timing(settings.get("depth_user_lora_timing"))
        validate_depth_user_lora_ramp(
            settings.get("depth_user_lora_ramp_early"),
            settings.get("depth_user_lora_ramp_middle"),
            settings.get("depth_user_lora_ramp_final"),
        )
        _, _, _, outpaint_mode = resolve_generation_process(settings)
        if identity_method == "depth_prompt" and outpaint_mode != "off":
            return "Depth + prompt only cannot be combined with Registered Outpaint."
        if direct_image_selected:
            if identity_method != "identity_edit":
                return "Direct Image → Identity Edit requires an Identity Edit method."
            if "A" in video_prompt_type or "Y" in video_prompt_type:
                return "Direct Image → Identity Edit currently supports Whole Frame only."
            if outpaint_mode != "off":
                return "Direct Image → Identity Edit cannot be combined with Registered Outpaint."
            try:
                validate_direct_image_denoising_strength(
                    inputs.get("denoising_strength")
                )
            except ValueError as exc:
                return str(exc)
        if identity_method == "depth_prompt" and not depth_selected:
            return "Depth + prompt only requires Transfer Depth and a Control Image."
        if identity_method == "depth_prompt":
            try:
                depth_strength = validate_depth_control_strength(
                    settings.get("depth_control_strength")
                )
            except ValueError as exc:
                return str(exc)
            if depth_strength <= 0:
                return "Depth + prompt only requires depth strength greater than 0."
        custom_mask = inputs.get("custom_guide")
        if depth_selected and "A" in video_prompt_type and custom_mask is not None:
            return (
                "An uploaded Custom Depth Mask must use Custom Mask — White "
                "Area or Custom Mask — Black Area, not a Painted Mask option."
            )
        if depth_selected and "Y" in video_prompt_type and custom_mask is None:
            return "The selected Custom Mask area requires a Custom Depth Mask upload."
        if outpaint_mode != "off":
            margins = str(inputs.get("video_guide_outpainting", "") or "").strip()
            ratio = str(inputs.get("video_guide_outpainting_ratio", "") or "").strip()
            if not margins and not ratio:
                return "Registered Outpaint requires Spatial Outpainting margins or a target aspect ratio."
            # WanGP resolves aspect-ratio slider weights only against its
            # Control Image input. Registered Outpaint instead uses an image
            # reference, so retain the selected ratio for the plugin runtime
            # to resolve against the actual source of each pass.
            settings["_registered_outpaint_ratio"] = ratio
            if depth_selected:
                return "Depth control cannot be combined with Registered Outpaint in the same pass."
        if "V" in video_prompt_type:
            control_inputs = (
                inputs.get("image_guide"),
                inputs.get("video_guide"),
                inputs.get("image_mask_guide"),
            )
            if all(value is None for value in control_inputs):
                if direct_image_selected:
                    return "Direct Image → Identity Edit requires a Control Image."
                if depth_selected:
                    return "Transfer Depth requires a Control Image."
                return "The selected Control Image process requires a Control Image."

        try:
            depth_strength = validate_depth_control_strength(
                settings.get("depth_control_strength")
            )
        except ValueError as exc:
            return str(exc)
        depth_active = depth_selected and depth_strength > 0
        depth_mask_active = depth_active and any(
            flag in video_prompt_type for flag in ("A", "Y")
        )
        plugin_data = inputs.get("plugin_data")
        plugin_data = dict(plugin_data) if isinstance(plugin_data, dict) else {}
        plugin_data[OUTPUT_METADATA_CONTEXT_KEY] = {
            "identity_method": identity_method,
            "depth_active": depth_active,
            "depth_mask_active": depth_mask_active,
            "user_lora_count": _item_count(
                inputs.get("loras_choices", inputs.get("activated_loras"))
            ),
        }
        inputs["plugin_data"] = plugin_data
        return None

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, prompt):
        """Avoid Krea LanPaint normalization for this non-inpainting plugin."""
        return None

    @staticmethod
    def custom_prompt_preprocess(prompt, **_kwargs):
        """Keep WanGP's native red-padding instruction out of plugin prompts."""
        return prompt
