#!/usr/bin/env python3
"""Validate the repository without importing WanGP or loading model weights."""

from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path.relative_to(ROOT)}: {exc}")


def main() -> int:
    required = [
        "AGENTS.md",
        "README.md",
        "SPEC.md",
        "ARCHITECTURE.md",
        "IMPLEMENTATION_PLAN.md",
        "THIRD_PARTY_NOTICES.md",
        "LICENSE",
        "NOTICE",
        "__init__.py",
        "plugin_info.json",
        "requirements.txt",
        "models/__init__.py",
        "models/krea2_identity_handler.py",
        "models/krea2_identity_main.py",
        "models/krea2_identity_utils.py",
        "tools/validate_wangp_contract.py",
        "tools/validate_remote_assets.py",
        "tools/smoke_test_wangp_import.py",
        "defaults/krea2_identity_raw.json",
        "defaults/krea2_identity_turbo.json",
        "profiles/krea2_identity/README.md",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    if missing:
        fail(f"missing required files: {', '.join(missing)}")

    manifest = load_json(ROOT / "plugin_info.json")
    if manifest.get("type") != "model":
        fail("plugin_info.json must declare type=model")
    for key in ("model_handlers", "defaults", "profiles"):
        if not manifest.get(key):
            fail(f"plugin_info.json is missing {key}")

    architectures = set()
    for path in sorted((ROOT / "defaults").glob("*.json")):
        definition = load_json(path)
        model = definition.get("model", {})
        architecture = model.get("architecture")
        if not architecture or not architecture.startswith("krea2_identity_"):
            fail(f"invalid architecture in {path.relative_to(ROOT)}")
        if architecture in architectures:
            fail(f"duplicate architecture: {architecture}")
        if model.get("visible") is not True:
            fail(f"experimental preview model must remain visible: {path.relative_to(ROOT)}")
        architectures.add(architecture)

    for path in sorted(ROOT.rglob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            fail(f"Python syntax error in {path.relative_to(ROOT)}: {exc}")

    print(f"OK: validated {len(required)} required paths and {len(architectures)} model definitions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
