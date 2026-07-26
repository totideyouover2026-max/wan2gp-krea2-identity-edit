#!/usr/bin/env python3
"""Import the plugin against a designated WanGP environment without loading weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wangp_root", type=Path)
    args = parser.parse_args()
    wangp_root = args.wangp_root.resolve()
    plugin_root = Path(__file__).resolve().parents[1]
    if not (wangp_root / "wgp.py").is_file():
        raise SystemExit(f"ERROR: not a WanGP root: {wangp_root}")
    sys.path.insert(0, str(wangp_root))
    package_name = "wangp_krea2_identity_edit_smoke"
    spec = importlib.util.spec_from_file_location(
        package_name,
        plugin_root / "__init__.py",
        submodule_search_locations=[str(plugin_root)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    module = importlib.import_module(f"{package_name}.models.krea2_identity_handler")
    metadata = importlib.import_module("models.model_metadata")
    field_help = importlib.import_module("shared.gradio.field_help")
    handler = module.family_handler
    supported = handler.query_supported_types()
    expected = {"krea2_identity_raw", "krea2_identity_turbo"}
    if set(supported) != expected:
        raise SystemExit(f"ERROR: unexpected supported types: {supported}")
    for model_type in supported:
        definition = handler.query_model_def(model_type, {})
        if not definition.get("image_outputs"):
            raise SystemExit(f"ERROR: {model_type} is not declared as image output")
        guide = definition.get("guide_preprocessing", {})
        if guide.get("selection") != ["", "DV"]:
            raise SystemExit(
                f"ERROR: {model_type} omits the native depth control-image dropdown"
            )
        if guide.get("labels", {}).get("DV") != "Transfer Depth":
            raise SystemExit(f"ERROR: {model_type} has the wrong depth control label")
        mask = definition.get("mask_preprocessing", {})
        if mask.get("selection") != ["", "A", "NA"]:
            raise SystemExit(
                f"ERROR: {model_type} omits inside/outside depth masking"
            )
        references = definition.get("image_ref_choices", {})
        if references.get("default") != "KI":
            raise SystemExit(
                f"ERROR: {model_type} does not protect its first scene reference"
            )
        custom_settings = {
            item.get("id"): item
            for item in definition.get("custom_settings", [])
            if isinstance(item, dict)
        }
        strength = custom_settings.get("depth_control_strength", {})
        if (
            strength.get("type") != "float"
            or strength.get("min") != 0.0
            or strength.get("max") != 2.0
            or strength.get("inc") != 0.01
            or strength.get("video_prompt_type") != "V"
        ):
            raise SystemExit(
                f"ERROR: {model_type} has the wrong dedicated depth strength setting"
            )
        stable_removal = custom_settings.get("subject_background_removal", {})
        if (
            not definition.get("no_background_removal", False)
            or not callable(definition.get("custom_image_ref_postprocessor"))
            or "stable" not in [
                value for _label, value in stable_removal.get("choices", [])
            ]
        ):
            raise SystemExit(
                f"ERROR: {model_type} omits stable subject isolation"
            )
        prompt_help = field_help.get_model_prompt_help(definition)
        if not isinstance(prompt_help, (list, tuple)) or len(prompt_help) < 2:
            raise SystemExit(
                f"ERROR: {model_type} omits the host-rendered prompt guide"
            )
        if prompt_help[0] != "Krea 2 Identity Edit Prompt Guide":
            raise SystemExit(f"ERROR: {model_type} has the wrong prompt-guide title")
        prompt_tools = field_help.render_model_prompt_tools(
            "Prompt",
            "krea2-identity-prompt-smoke",
            model_type,
            definition,
            "advanced",
        )
        if "Krea 2 Identity Edit Prompt Guide" not in prompt_tools:
            raise SystemExit(
                f"ERROR: {model_type} prompt guide does not render through WanGP"
            )
        media_inputs = metadata.infer_media_inputs(definition)
        if not media_inputs["image"]["control"]:
            raise SystemExit(
                f"ERROR: {model_type} does not expose WanGP's Control Image upload"
            )
        if not media_inputs["image"]["mask"]:
            raise SystemExit(
                f"ERROR: {model_type} does not expose WanGP's Control Image mask"
            )
        text_encoder_urls = definition.get("text_encoder_URLs", [])
        if len(text_encoder_urls) != 1 or not text_encoder_urls[0].endswith(
            "Qwen3-VL-4B-Instruct_bf16.safetensors"
        ):
            raise SystemExit(
                f"ERROR: {model_type} does not force the full BF16 Qwen3-VL encoder"
            )
        files = handler.query_model_files([], model_type, {})
        if any(item.get("repoId") == "Comfy-Org/Krea-2" for item in files):
            raise SystemExit(
                f"ERROR: {model_type} still requests the obsolete split visual checkpoint"
            )
    print("OK: plugin handler imports and resolves both experimental architectures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
