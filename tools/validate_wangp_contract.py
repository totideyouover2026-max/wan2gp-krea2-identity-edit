#!/usr/bin/env python3
"""Statically validate the public WanGP interfaces used by this plugin."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


REQUIRED = {
    "models/krea2/krea2_main.py": {
        "Krea2Pipeline",
        "_load_transformer",
        "_load_vae",
        "_pack_image_latents",
    },
    "models/krea2/krea2_mmdit.py": {"key_padding_mask"},
    "models/ideogram4/qwen3_vl_transformers.py": {
        "Qwen3VLModel",
        "Qwen3VLPreTrainedModel",
        "Qwen3VLTextModel",
        "Qwen3VLVisionModel",
    },
    "models/ideogram4/qwen3_vl_configuration.py": {
        "Qwen3VLConfig",
        "register_qwen3_vl_config",
    },
}


def public_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def validate(root: Path) -> list[str]:
    errors = []
    for relative, required_names in REQUIRED.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        missing = sorted(required_names - public_names(path))
        if missing:
            errors.append(f"{relative} lacks: {', '.join(missing)}")
    wgp = root / "wgp.py"
    if not wgp.is_file():
        errors.append("missing wgp.py")
    else:
        source = wgp.read_text(encoding="utf-8")
        for required_text in (
            "original_input_ref_images",
            "custom_settings=custom_settings_for_model",
            "get_loras_transformer",
            "load_loras_into_model",
        ):
            if required_text not in source:
                errors.append(f"wgp.py lacks runtime contract: {required_text}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wangp_root", type=Path, help="Path to a clean public WanGP checkout"
    )
    args = parser.parse_args()
    errors = validate(args.wangp_root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "OK: required WanGP Krea 2, Qwen3-VL, LoRA and generic-input "
        "contracts are present"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
