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
    validate_reid_lora_strength,
    validate_subject_attention_ramp,
    validate_subject_attention_timing,
)


ADVANCED_SETTINGS_ID = "advanced_settings"
_IDENTITY_LORA_VARIANT_SCHEMA_KEY = "identity_lora_variant_schema"
_IDENTITY_LORA_VARIANT_SCHEMA_VERSION = "v1.2"
ADVANCED_SETTING_KEYS = (
    "generation_process",
    "grounding_px",
    "output_resolution_limit",
    "identity_lora_variant",
    _IDENTITY_LORA_VARIANT_SCHEMA_KEY,
    "reid_lora_strength",
    "reid_reference_method",
    "secondary_reference_geometry",
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
    "grounding_px": 768,
    "output_resolution_limit": "safe_2mp",
    "identity_lora_variant": "full_v1.2",
    _IDENTITY_LORA_VARIANT_SCHEMA_KEY: _IDENTITY_LORA_VARIANT_SCHEMA_VERSION,
    "reid_lora_strength": 1.0,
    "reid_reference_method": "isolated_cache",
    "secondary_reference_geometry": "fit",
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

    # Preserve migration from the older individual two-phase/outpaint fields.
    if "generation_process" in payload:
        process_source = {"generation_process": payload["generation_process"]}
    elif "generation_process" in legacy:
        process_source = {"generation_process": legacy["generation_process"]}
    else:
        process_source = legacy
    values["generation_process"] = migrate_generation_process(process_source)
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
    values["reid_lora_strength"] = validate_reid_lora_strength(
        values["reid_lora_strength"]
    )
    if values["reid_reference_method"] not in {
        "joint_timestep_zero",
        "isolated_cache",
    }:
        raise ValueError(
            "reid_reference_method must be 'joint_timestep_zero' or "
            "'isolated_cache'"
        )
    if values["secondary_reference_geometry"] not in {"fit", "stretch"}:
        raise ValueError(
            "secondary_reference_geometry must be 'fit' or 'stretch'"
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
