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
    handler = module.family_handler
    supported = handler.query_supported_types()
    expected = {"krea2_identity_raw", "krea2_identity_turbo"}
    if set(supported) != expected:
        raise SystemExit(f"ERROR: unexpected supported types: {supported}")
    for model_type in supported:
        definition = handler.query_model_def(model_type, {})
        if not definition.get("image_outputs"):
            raise SystemExit(f"ERROR: {model_type} is not declared as image output")
        files = handler.query_model_files([], model_type, {})
        if not any(item.get("repoId") == "Comfy-Org/Krea-2" for item in files):
            raise SystemExit(f"ERROR: {model_type} omits the public visual checkpoint")
    print("OK: plugin handler imports and resolves both hidden architectures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
