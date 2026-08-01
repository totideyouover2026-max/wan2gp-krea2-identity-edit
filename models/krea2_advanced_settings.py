"""Packing helpers for Krea 2 settings kept behind the WanGP UI modal.

WanGP v12.34 serializes at most five model-defined custom controls.  The UI
extension stores the less frequently changed controls as one canonical JSON
value in the fifth slot; the handler expands it before validation/inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from .krea2_identity_utils import (
    migrate_generation_process,
    validate_builtin_adapter_timing,
    validate_builtin_depth_ramp,
    validate_builtin_identity_ramp,
    validate_depth_mask_feather_px,
    validate_depth_user_lora_ramp,
    validate_depth_user_lora_timing,
    validate_grounding_px,
    validate_identity_lora_variant,
    validate_subject_attention_ramp,
    validate_subject_attention_timing,
)


ADVANCED_SETTINGS_ID = "advanced_settings"
OUTPUT_METADATA_CONTEXT_KEY = "krea2_identity_output_metadata"
DEFAULT_OUTPAINT_PROMPT = (
    "Seamlessly continue the existing image into the expanded canvas. Preserve "
    "the same scene, subject count, composition, perspective, lighting, color "
    "palette, textures, and visual style. Do not add new subjects, objects, "
    "text, logos, or unrelated details."
)
_IDENTITY_LORA_VARIANT_SCHEMA_KEY = "identity_lora_variant_schema"
_IDENTITY_LORA_VARIANT_SCHEMA_VERSION = "v1.2"
ADVANCED_SETTING_KEYS = (
    "generation_process",
    "outpaint_prompt",
    "grounding_px",
    "output_resolution_limit",
    "identity_lora_variant",
    _IDENTITY_LORA_VARIANT_SCHEMA_KEY,
    "subject_attention_timing",
    "subject_attention_ramp_early",
    "subject_attention_ramp_middle",
    "subject_attention_ramp_final",
    "depth_mask_feather_px",
    "depth_user_lora_timing",
    "depth_user_lora_ramp_early",
    "depth_user_lora_ramp_middle",
    "depth_user_lora_ramp_final",
    "builtin_adapter_timing",
    "builtin_depth_ramp_early",
    "builtin_depth_ramp_middle",
    "builtin_depth_ramp_final",
    "builtin_identity_ramp_early",
    "builtin_identity_ramp_middle",
    "builtin_identity_ramp_final",
)
ADVANCED_SETTINGS_DEFAULTS = {
    "generation_process": "standard",
    "outpaint_prompt": DEFAULT_OUTPAINT_PROMPT,
    "grounding_px": 768,
    "output_resolution_limit": "safe_2mp",
    "identity_lora_variant": "full_v1.2",
    _IDENTITY_LORA_VARIANT_SCHEMA_KEY: _IDENTITY_LORA_VARIANT_SCHEMA_VERSION,
    "subject_attention_timing": "constant",
    "subject_attention_ramp_early": 1.0,
    "subject_attention_ramp_middle": 2.0,
    "subject_attention_ramp_final": 8.0,
    "depth_mask_feather_px": 16,
    "depth_user_lora_timing": "depth_first",
    "depth_user_lora_ramp_early": 0.0,
    "depth_user_lora_ramp_middle": 0.25,
    "depth_user_lora_ramp_final": 1.0,
    "builtin_adapter_timing": "simultaneous",
    "builtin_depth_ramp_early": 1.0,
    "builtin_depth_ramp_middle": 0.5,
    "builtin_depth_ramp_final": 0.0,
    "builtin_identity_ramp_early": 0.25,
    "builtin_identity_ramp_middle": 0.75,
    "builtin_identity_ramp_final": 1.0,
}
ADVANCED_METADATA_LABELS = frozenset(
    {
        "Generation Process",
        "Registered Outpaint Prompt",
        "Identity Edit LoRA Variant",
        "Reference Grounding Budget",
        "Output Resolution Limit",
        "Subject Attention Timing",
        "Subject Attention Ramp — Early",
        "Subject Attention Ramp — Middle",
        "Subject Attention Ramp — Final",
        "Builtin Adapter Timing",
        "Builtin Depth Ramp — Early",
        "Builtin Depth Ramp — Middle",
        "Builtin Depth Ramp — Final",
        "Builtin Identity Edit Ramp — Early",
        "Builtin Identity Edit Ramp — Middle",
        "Builtin Identity Edit Ramp — Final",
        "Depth Mask Feather",
        "Additional LoRA Timing With Depth",
        "Additional LoRA Ramp — Early",
        "Additional LoRA Ramp — Middle",
        "Additional LoRA Ramp — Final",
    }
)


def _decode_payload(raw_value) -> dict:
    if raw_value in (None, ""):
        return {}
    if isinstance(raw_value, Mapping):
        return dict(raw_value)
    if not isinstance(raw_value, str):
        raise ValueError("advanced_settings must be a JSON object")
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("advanced_settings contains invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("advanced_settings must be a JSON object")
    return decoded


def normalize_advanced_settings(raw_value=None, legacy_settings=None) -> dict:
    """Return validated advanced values, migrating legacy flat settings.

    Explicit values in ``raw_value`` win.  Flat values are accepted as a
    fallback so settings created before the modal remain lossless.
    """

    legacy = dict(legacy_settings) if isinstance(legacy_settings, Mapping) else {}
    payload = _decode_payload(raw_value)
    values = dict(ADVANCED_SETTINGS_DEFAULTS)
    for key in ADVANCED_SETTING_KEYS:
        if key in legacy:
            values[key] = legacy[key]
    values.update(
        {
            key: value
            for key, value in payload.items()
            if key in ADVANCED_SETTING_KEYS
        }
    )

    # Preserve migration from older individual process/outpaint fields.
    if "generation_process" in payload:
        process_source = {"generation_process": payload["generation_process"]}
    elif "generation_process" in legacy:
        process_source = {"generation_process": legacy["generation_process"]}
    else:
        process_source = legacy
    values["generation_process"] = migrate_generation_process(process_source)
    values["outpaint_prompt"] = str(values["outpaint_prompt"] or "").strip()
    if not values["outpaint_prompt"]:
        values["outpaint_prompt"] = DEFAULT_OUTPAINT_PROMPT
    if len(values["outpaint_prompt"]) > 2000:
        raise ValueError("outpaint_prompt must be 2000 characters or fewer")
    values["grounding_px"] = validate_grounding_px(values["grounding_px"])
    if values["output_resolution_limit"] not in {"safe_2mp", "unlimited"}:
        raise ValueError(
            "output_resolution_limit must be 'safe_2mp' or 'unlimited'"
        )
    values["identity_lora_variant"] = validate_identity_lora_variant(
        values["identity_lora_variant"]
    )
    if (
        values["identity_lora_variant"] == "r64"
        and payload.get(_IDENTITY_LORA_VARIANT_SCHEMA_KEY)
        != _IDENTITY_LORA_VARIANT_SCHEMA_VERSION
        and legacy.get(_IDENTITY_LORA_VARIANT_SCHEMA_KEY)
        != _IDENTITY_LORA_VARIANT_SCHEMA_VERSION
    ):
        # Rank 64 was the earlier default. Existing saved payloads have no
        # marker, while values saved after this migration retain the marker.
        values["identity_lora_variant"] = "full_v1.2"
    values[_IDENTITY_LORA_VARIANT_SCHEMA_KEY] = (
        _IDENTITY_LORA_VARIANT_SCHEMA_VERSION
    )
    values["subject_attention_timing"] = validate_subject_attention_timing(
        values["subject_attention_timing"]
    )
    (
        values["subject_attention_ramp_early"],
        values["subject_attention_ramp_middle"],
        values["subject_attention_ramp_final"],
    ) = validate_subject_attention_ramp(
        values["subject_attention_ramp_early"],
        values["subject_attention_ramp_middle"],
        values["subject_attention_ramp_final"],
    )
    values["depth_mask_feather_px"] = validate_depth_mask_feather_px(
        values["depth_mask_feather_px"]
    )
    values["depth_user_lora_timing"] = validate_depth_user_lora_timing(
        values["depth_user_lora_timing"]
    )
    (
        values["depth_user_lora_ramp_early"],
        values["depth_user_lora_ramp_middle"],
        values["depth_user_lora_ramp_final"],
    ) = validate_depth_user_lora_ramp(
        values["depth_user_lora_ramp_early"],
        values["depth_user_lora_ramp_middle"],
        values["depth_user_lora_ramp_final"],
    )
    values["builtin_adapter_timing"] = validate_builtin_adapter_timing(
        values["builtin_adapter_timing"]
    )
    (
        values["builtin_depth_ramp_early"],
        values["builtin_depth_ramp_middle"],
        values["builtin_depth_ramp_final"],
    ) = validate_builtin_depth_ramp(
        values["builtin_depth_ramp_early"],
        values["builtin_depth_ramp_middle"],
        values["builtin_depth_ramp_final"],
    )
    (
        values["builtin_identity_ramp_early"],
        values["builtin_identity_ramp_middle"],
        values["builtin_identity_ramp_final"],
    ) = validate_builtin_identity_ramp(
        values["builtin_identity_ramp_early"],
        values["builtin_identity_ramp_middle"],
        values["builtin_identity_ramp_final"],
    )
    return values


def encode_advanced_settings(values=None, legacy_settings=None) -> str:
    if isinstance(values, Mapping):
        values = dict(values)
        values.setdefault(
            _IDENTITY_LORA_VARIANT_SCHEMA_KEY,
            _IDENTITY_LORA_VARIANT_SCHEMA_VERSION,
        )
    normalized = normalize_advanced_settings(values, legacy_settings)
    return json.dumps(normalized, separators=(",", ":"), sort_keys=True)


def expand_advanced_settings(settings) -> dict:
    """Merge the packed modal value into a normal runtime settings mapping."""

    expanded = dict(settings) if isinstance(settings, Mapping) else {}
    normalized = normalize_advanced_settings(
        expanded.get(ADVANCED_SETTINGS_ID), expanded
    )
    expanded.update(normalized)
    expanded[ADVANCED_SETTINGS_ID] = encode_advanced_settings(normalized)
    return expanded


def _display_number(value) -> str:
    number = float(value)
    return f"{number:.2f}".rstrip("0").rstrip(".")


def advanced_settings_metadata(
    settings,
    *,
    identity_method: str,
    depth_active: bool = False,
    depth_mask_active: bool = False,
    user_lora_count: int = 0,
) -> dict[str, str]:
    """Return output-info rows only for advanced settings used by this run."""

    packed = (
        settings.get(ADVANCED_SETTINGS_ID)
        if isinstance(settings, Mapping)
        else settings
    )
    values = normalize_advanced_settings(packed, settings)
    process = values["generation_process"]
    identity_edit_active = (
        identity_method == "identity_edit" and process != "outpaint_only"
    )
    process_labels = {
        "standard": "Standard Single Pass",
        "outpaint_only": "Registered Outpaint Only",
        "identity_then_outpaint": "Identity Edit, Then Registered Outpaint",
    }
    rows = {
        "Generation Process": process_labels.get(process, process),
    }
    if process in {"outpaint_only", "identity_then_outpaint"}:
        rows["Registered Outpaint Prompt"] = values["outpaint_prompt"]

    if identity_edit_active:
        identity_lora_labels = {
            "full_v1.2": "v1.2 Full",
            "r128": "v1.2 Rank 128",
            "r64": "v1.2 Rank 64",
        }
        rows["Identity Edit LoRA Variant"] = identity_lora_labels[
            values["identity_lora_variant"]
        ]
        rows["Reference Grounding Budget"] = f"{values['grounding_px']} px"
        rows["Output Resolution Limit"] = (
            "Safe 2 MP"
            if values["output_resolution_limit"] == "safe_2mp"
            else "Full Selected Resolution"
        )
        subject_timing = values["subject_attention_timing"]
        rows["Subject Attention Timing"] = (
            "Constant Selected Fidelity"
            if subject_timing == "constant"
            else "Ramp Over Denoising Thirds"
        )
        if subject_timing == "ramp":
            for stage in ("early", "middle", "final"):
                rows[f"Subject Attention Ramp — {stage.title()}"] = (
                    f"{_display_number(values[f'subject_attention_ramp_{stage}'])}x"
                )

    standard_depth_pass = depth_active and process == "standard"
    if identity_edit_active and standard_depth_pass:
        builtin_timing = values["builtin_adapter_timing"]
        rows["Builtin Adapter Timing"] = (
            "Simultaneous"
            if builtin_timing == "simultaneous"
            else "Depth Layout → Identity Refinement"
        )
        if builtin_timing == "depth_then_identity":
            for adapter in ("depth", "identity"):
                label = "Depth" if adapter == "depth" else "Identity Edit"
                for stage in ("early", "middle", "final"):
                    rows[f"Builtin {label} Ramp — {stage.title()}"] = (
                        _display_number(
                            values[f"builtin_{adapter}_ramp_{stage}"]
                        )
                    )

    if depth_mask_active:
        rows["Depth Mask Feather"] = f"{values['depth_mask_feather_px']} px"

    if standard_depth_pass and user_lora_count > 0:
        user_timing = values["depth_user_lora_timing"]
        rows["Additional LoRA Timing With Depth"] = (
            "Depth First" if user_timing == "depth_first" else "All Steps"
        )
        if user_timing == "depth_first":
            for stage in ("early", "middle", "final"):
                rows[f"Additional LoRA Ramp — {stage.title()}"] = (
                    _display_number(values[f"depth_user_lora_ramp_{stage}"])
                )

    return rows
