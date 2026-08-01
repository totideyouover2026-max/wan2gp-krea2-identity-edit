#!/usr/bin/env python3
"""Validate public safetensors URLs and key conventions without downloading weights."""

from __future__ import annotations

import json
import re
import struct
import urllib.request


ASSETS = {
    "qwen3_vl_bf16": (
        "https://huggingface.co/DeepBeepMeep/krea-2/resolve/main/"
        "Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_bf16.safetensors"
    ),
    "v1.2_full": (
        "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/"
        "krea2_identity_edit_v1_2.safetensors"
    ),
    "v1.2_r128": (
        "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/"
        "krea2_identity_edit_v1_2_r128.safetensors"
    ),
    "v1.2_r64": (
        "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/"
        "krea2_identity_edit_v1_2_r64.safetensors"
    ),
    "depth_control": (
        "https://huggingface.co/Patil/Krea-2-depth-controlnet/resolve/main/"
        "depth-control-lora.safetensors"
    ),
}
EXPECTED_RANKS = {
    "v1.2_full": 256,
    "v1.2_r128": 128,
    "v1.2_r64": 64,
}
LORA_MODULE = re.compile(
    r"^diffusion_model\."
    r"(?:blocks\.\d+|txtfusion\.(?:layerwise_blocks|refiner_blocks)\.\d+)\."
    r"(?:attn\.(?:gate|wk|wo|wq|wv)|mlp\.(?:down|gate|up))\."
    r"lora_[AB]\.weight$"
)
DEPTH_LORA_MODULE = re.compile(
    r"^blocks\.\d+\."
    r"(?:attn\.(?:gate|wk|wo|wq|wv)|mlp\.(?:down|gate|up))\."
    r"[AB]$"
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
        if name == "qwen3_vl_bf16":
            visual = [key for key in keys if key.startswith("visual.")]
            language = [key for key in keys if key.startswith("language_model.")]
            if not visual or not language:
                raise SystemExit(
                    "ERROR: full Qwen3-VL checkpoint lacks visual or language weights"
                )
            selected = visual + language
            if any(header[key].get("dtype") != "BF16" for key in selected):
                raise SystemExit("ERROR: expected full Qwen3-VL weights to be BF16")
            print(
                "OK: full Qwen3-VL checkpoint exposes "
                f"{len(visual)} visual and {len(language)} language BF16 tensors"
            )
        elif name == "depth_control":
            adapter_keys = [key for key in keys if key not in {"first.weight", "first.bias"}]
            if len(adapter_keys) != 448:
                raise SystemExit(
                    f"ERROR: depth control exposes {len(adapter_keys)} block tensors"
                )
            if any(DEPTH_LORA_MODULE.fullmatch(key) is None for key in adapter_keys):
                raise SystemExit("ERROR: depth control targets unknown Krea 2 modules")
            ranks = {
                header[key]["shape"][0]
                for key in adapter_keys
                if key.endswith(".A")
            }
            if ranks != {64}:
                raise SystemExit(
                    f"ERROR: depth control has unexpected adapter ranks: {ranks}"
                )
            if header.get("first.weight", {}).get("shape") != [6144, 128]:
                raise SystemExit("ERROR: depth control lacks the 6144x128 input projection")
            if header.get("first.bias", {}).get("shape") != [6144]:
                raise SystemExit("ERROR: depth control lacks the 6144 input bias")
            print(
                "OK: depth control exposes 448 rank-64 block tensors and a "
                "6144x128 input projection"
            )
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
