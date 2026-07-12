# Architecture

## Plugin boundary

This project is a WanGP **model plugin**, not a UI application plugin and not a Pinokio launcher. `plugin_info.json` points WanGP at one `family_handler`, a defaults root and a profiles root. No `plugin.py` is needed unless a future feature cannot be represented through the model-handler contract.

The plugin uses unique architecture identifiers because WanGP rejects multiple handlers that claim the same model type.

## Runtime flow

```text
reference image(s)
   ├─ Qwen image VAE ─ clean latent tokens ───────────────┐
   └─ Qwen3-VL vision tower + instruction ─ text states ─┤
                                                         v
noise target ─ target latent tokens ──> [text | refs | target]
                                        Krea 2 MMDiT
                                              |
                                      target tokens only
                                              |
                                            VAE decode
```

The source/reference block is not ordinary masked inpainting. The transformer sequence must include clean source tokens at distinct RoPE frame indices. The model returns only the target-token portion.

## Reuse strategy

Prefer importing stable components from the host WanGP Krea 2 implementation instead of copying large files:

- transformer definitions and checkpoint conversion;
- Qwen image VAE loading;
- timestep schedule helpers;
- preview decoding and callback conventions;
- MMGP/offload registration.

Keep identity-specific logic in this repository:

- full multimodal Qwen3-VL conditioner;
- image-aware prompt templates;
- reference preprocessing and grounding resolution;
- source-token stream construction;
- model definitions, LoRA choices and validation.

## Implemented host boundary

The plugin currently targets WanGP v12.3 and imports the host's
`Krea2Pipeline`, transformer/VAE loaders, Krea 2 MMDiT modules and Qwen3-VL
model classes. It does not vendor WanGP source.

`IdentityKrea2Pipeline` reuses the host denoising loop unchanged. Before that
loop it VAE-encodes the ordered references and installs their packed clean
tokens on the transformer for the duration of one call. The identity forward
methods build the expanded stream, preserve the host block/callback/LoRA
execution, and clear the temporary source state in a `finally` block.

The host's normal Krea 2 language checkpoint is reused so WanGP can retain its
BF16/int8 choice. The missing vision tower is loaded from the public
`Comfy-Org/Krea-2` checkpoint. Only its 315 BF16 `model.visual` tensors are
materialized and passed to MMGP; its unrelated language tensors are ignored.

Identity Edit LoRAs are selected dynamically through `get_loras_transformer`,
which participates in WanGP's normal LoRA download, loading, scheduling and
unloading flow. The ai-toolkit `diffusion_model.` prefix is removed by a small
preprocessor before MMGP matches transformer modules.

If WanGP does not expose a stable reusable boundary, copy only the smallest necessary Apache-compatible/reference sections and record their origin in `THIRD_PARTY_NOTICES.md`.

## Host inputs already available

WanGP passes model factories several useful generic inputs, including:

- `original_input_ref_images` for unprocessed reference PIL images;
- `custom_settings` for `grounding_px` and LoRA-variant controls;
- normal generation settings and callbacks;
- model LoRA lists and schedules.

The implementation should use these generic inputs rather than introducing a bespoke Gradio tab.

## Major technical risk

WanGP's existing Krea 2 text conditioner is language-only. Identity Edit requires the Qwen3-VL visual encoder and image processor as well. The implementation must package/download compatible vision weights, register them with MMGP, and avoid keeping unnecessary language/vision modules resident in VRAM.

Reference latent tokens also increase transformer sequence length. Two references can materially increase attention time and memory, so 1–2 MP limits and offload testing are release requirements.
