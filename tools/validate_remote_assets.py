#!/usr/bin/env python3
"""Validate public safetensors URLs and key conventions without downloading weights."""

from __future__ import annotations

import json
import re
import struct
import urllib.request


ASSETS = {
    "vision": (
        "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/"
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
    ),
    "v1.2_full": (
        "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/"
        "krea2_identity_edit_v1_2.safetensors"
    ),
    "v1.1_full": (
        "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/"
        "krea2_identity_edit_v1_1.safetensors"
    ),
    "r128": (
        "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/"
        "krea2_identity_edit_v1_1_r128.safetensors"
    ),
    "r64": (
        "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/"
        "krea2_identity_edit_v1_1_r64.safetensors"
    ),
}
EXPECTED_RANKS = {"v1.2_full": 256, "v1.1_full": 256, "r128": 128, "r64": 64}
LORA_MODULE = re.compile(
    r"^diffusion_model\."
    r"(?:blocks\.\d+|txtfusion\.(?:layerwise_blocks|refiner_blocks)\.\d+)\."
    r"(?:attn\.(?:gate|wk|wo|wq|wv)|mlp\.(?:down|gate|up))\."
    r"lora_[AB]\.weight$"
)


def read_range(url: str, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{end}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(end + 1)


def safetensors_header(url: str) -> dict:
    prefix = read_range(url, 7)
    if len(prefix) != 8:
        raise ValueError("Unable to read the safetensors header length")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 0 or header_length > 16 * 1024 * 1024:
        raise ValueError(f"Invalid safetensors header length: {header_length}")
    data = read_range(url, header_length + 7)
    return json.loads(data[8 : 8 + header_length])


def main() -> int:
    for name, url in ASSETS.items():
        header = safetensors_header(url)
        keys = [key for key in header if key != "__metadata__"]
        if name == "vision":
            visual = [key for key in keys if key.startswith("model.visual.")]
            if not visual:
                raise SystemExit("ERROR: visual checkpoint has no model.visual weights")
            if any(header[key].get("dtype") != "BF16" for key in visual):
                raise SystemExit("ERROR: expected the Qwen3-VL visual prefix to be BF16")
            print(f"OK: vision exposes {len(visual)} BF16 model.visual tensors")
        else:
            if not keys or any(not key.startswith("diffusion_model.") for key in keys):
                raise SystemExit(f"ERROR: {name} LoRA has unexpected key prefixes")
            if not any(".lora_A.weight" in key for key in keys):
                raise SystemExit(f"ERROR: {name} LoRA has no LoRA A tensors")
            if any(LORA_MODULE.fullmatch(key) is None for key in keys):
                raise SystemExit(f"ERROR: {name} LoRA targets unknown Krea 2 modules")
            ranks = {
                value["shape"][0]
                for key, value in header.items()
                if ".lora_A.weight" in key
            }
            if ranks != {EXPECTED_RANKS[name]}:
                raise SystemExit(f"ERROR: {name} LoRA has unexpected ranks: {ranks}")
            print(
                f"OK: {name} LoRA exposes {len(keys)} Krea 2 tensors "
                f"at rank {EXPECTED_RANKS[name]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
