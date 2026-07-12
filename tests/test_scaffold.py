from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScaffoldTests(unittest.TestCase):
    def test_manifest_targets_existing_paths(self):
        manifest = json.loads((ROOT / "plugin_info.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["type"], "model")
        self.assertTrue((ROOT / manifest["defaults"]).is_dir())
        self.assertTrue((ROOT / manifest["profiles"]).is_dir())
        self.assertEqual(manifest["model_handlers"], [".models.krea2_identity_handler"])

    def test_model_architectures_are_unique_and_visible_for_preview(self):
        architectures = set()
        for path in sorted((ROOT / "defaults").glob("*.json")):
            definition = json.loads(path.read_text(encoding="utf-8"))
            model = definition["model"]
            architecture = model["architecture"]
            self.assertNotIn(architecture, architectures)
            self.assertTrue(architecture.startswith("krea2_identity_"))
            self.assertTrue(model["visible"])
            architectures.add(architecture)
        self.assertEqual(
            architectures,
            {"krea2_identity_raw", "krea2_identity_turbo"},
        )

    def test_no_model_weights_are_committed(self):
        forbidden = {".safetensors", ".ckpt", ".pt", ".pth"}
        weights = [path for path in ROOT.rglob("*") if path.suffix.lower() in forbidden]
        self.assertEqual(weights, [])

    def test_models_do_not_preload_a_fixed_identity_lora(self):
        for path in sorted((ROOT / "defaults").glob("*.json")):
            model = json.loads(path.read_text(encoding="utf-8"))["model"]
            self.assertEqual(model.get("loras"), [])
            self.assertEqual(model.get("loras_multipliers"), [])

    def test_release_profiles_remain_gated(self):
        profiles = list((ROOT / "profiles" / "krea2_identity").glob("*.json"))
        self.assertEqual(profiles, [])


if __name__ == "__main__":
    unittest.main()
