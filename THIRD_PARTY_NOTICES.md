# Third-party notices and release licensing

This file records runtime dependencies and adapted implementation sources. It is
not legal advice.

## Krea 2 and Identity Edit weights

- Base model: <https://huggingface.co/krea/Krea-2-Raw>
- Adapter: <https://huggingface.co/conradlocke/krea2-identity-edit>
- Supported adapter files: v1.2 full rank, r128 and r64.
- Governing terms: Krea 2 Community License and associated acceptable-use terms.
- Do not commit or redistribute the weights through this Git repository.
- The plugin should download weights from their authoritative repositories and clearly surface applicable terms.

## Reference node implementation

- Project: <https://github.com/lbouaraba/comfyui-krea2edit>
- Stated license: Apache-2.0.
- The sequence construction, RoPE frame convention, grounded prompt template,
  reference preprocessing and optional reference-attention bias in
  `models/krea2_identity_main.py` are an independent WanGP adaptation of the
  project's Apache-2.0 implementation.
- Attribution is preserved in `NOTICE`. No ComfyUI wrapper code is bundled.

## WanGP

- Project: <https://github.com/deepbeepmeep/Wan2GP>
- Audited public API revision:
  `6b92c54f92bde24d6d309d6f61249353b0ec783d` (2026-07-19).
- Current upstream license at that revision: WanGP Community License 2.0.
- WanGP is not bundled. The plugin imports its installed Krea 2, Qwen3-VL,
  scheduler, VAE and MMGP interfaces at runtime.
- Optional depth-mask editing, mask expansion and subject-reference background
  removal use preprocessing facilities supplied by the host WanGP installation;
  the plugin does not bundle their models or UI code.
- The prompt guide is original plugin documentation rendered through WanGP's
  public `prompt_infos`/field-help interface; no LTX prompt text is copied.
- The adaptive form and movable Advanced Settings panel are original plugin code built
  against WanGP's public `WAN2GPPlugin` component-request and `insert_after`
  interfaces, rechecked at upstream revision
  `6af948127cd71ff96de0e1444ba1a1f8ed798fa1`. No WanGP UI source is copied or
  patched.

## SAM3 subject segmentation

- Runtime API and asset declaration: WanGP's shared `shared.magic_mask` module.
- Model assets: `DeepBeepMeep/Wan2.1`, folder `sam3`, including the SAM3.1
  multiplex checkpoint and BPE vocabulary.
- Runtime purpose: keyword-guided semantic isolation of reference 2 before
  Identity Edit conditioning.
- The plugin calls the installed host API and does not copy SAM3 or WanGP model
  implementation code. Meta's applicable SAM3 model and software terms apply.

## Qwen3-VL language and visual weights

- Source: `DeepBeepMeep/krea-2`, file
  `Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_bf16.safetensors`.
- The plugin defaults to the complete BF16 language and visual modules through
  MMGP, matching WanGP v12.34's native Krea 2 Identity Edit precision contract.
- A temporary regression selector can instead load the Quanto language model
  from `DeepBeepMeep/krea-2`, file
  `Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors`,
  together with the scaled-FP8 visual tower from `Comfy-Org/Krea-2`, file
  `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`.
- The legacy processor and split language/vision loading route in
  `models/krea2_identity_main.py` restore this plugin's earlier adaptation of
  the Apache-2.0 `comfyui-krea2edit` implementation for controlled A/B testing.
- Model terms published by each source repository apply. Neither checkpoint is
  committed to this repository.
- Qwen3-VL terms and the source repository's model card apply.

## Experimental Krea 2 depth control

- Adapter: <https://huggingface.co/Patil/Krea-2-depth-controlnet>, file
  `depth-control-lora.safetensors`.
- Adapter terms: Krea 2 Community License, as stated by its model card.
- Documented inference reference:
  <https://github.com/Tanmaypatil123/Krea-2-controlnet>, inspected at
  `909682ae0bdd9eb87c8258894c0003224db00d0b`.
- ComfyUI integration reference:
  <https://github.com/facok/comfyui-krea2-controlnet>, inspected at
  `79ebfd3bd80d2180b334dd7ce57f3c9ddaa0848f`.
- Neither reference repository declared a code license in the inspected tree.
  No source code from them is bundled; this plugin independently implements the
  documented channel-concatenated target projection and adapter conversion.

## Depth Anything V2 Large

- Model: <https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf>.
- Runtime purpose: generate the inverse-depth representation from the
  separately uploaded control image through WanGP's shared `DV` preprocessor.
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (`CC-BY-NC-4.0`), as stated by the model card.
- The model is supplied and executed by the host WanGP installation rather than
  downloaded or loaded separately by this plugin.
- This non-commercial restriction is additional to the Krea model terms. Do
  not enable automatic V2 depth estimation for commercial use without replacing
  it with a suitably licensed depth source/model.

## Krea 2 Registered Outpaint

- Adapter: <https://huggingface.co/yijunwang2/krea2-outpaint>, file
  `krea2_outpaint_rank32.safetensors`.
- Adapter terms: Krea 2 Community License Agreement supplied by its repository.
- Pipeline/helper source: Apache License 2.0, copyright 2026 Ostris, LLC.
- `models/krea2_registered_outpaint.py` adapts the published placement and
  protected-source compositing helper. The runtime independently integrates the
  published destination-relative rotary and isolated reference K/V contract
  with WanGP's native Krea 2 implementation.
- Model weights are downloaded separately and are not committed here.
- This is unofficial and is not endorsed by Krea or Ostris.

## Krea 2 ReID

- Adapter: <https://huggingface.co/yijunwang2/krea2-reid>, file
  `krea2_reid_rank32.safetensors`.
- Adapter terms: Krea 2 Community License Agreement supplied by its repository.
- Pinned inference pipeline: `yijunwang2/krea2-reid` revision
  `121fb0183944f1befeb712d92e9ca07d0e282088`, Apache License 2.0,
  copyright 2026 Ostris, LLC; the repository states its `pipeline.py` was
  pinned from `ostris/Krea2OstrisEdit`.
- The runtime independently adapts the published single-reference,
  aspect-preserving 384²-area view shared by Qwen and the clean-reference VAE,
  frame indexing, and isolated per-block post-RoPE K/V captured at timestep
  zero and injected during target denoising through WanGP's native transformer
  and MMGP scheduling interfaces. The plugin also retains an experimental joint
  timestep-zero target/reference stream as a diagnostic path. No external pipeline file or model
  weight is bundled.
- Optional automatic face/head cropping adapts the repository's MIT-licensed
  `face_crop.py` helper. Its YuNet INT8 ONNX detector is declared as a separate
  runtime download from `yijunwang2/krea2-reid/models`; detector weights are not
  committed here. The plugin also retains a Keep full reference option.
- This is unofficial and is not endorsed by Krea or Ostris.

## Repository code license

- Original plugin code: Apache License 2.0; see `LICENSE`.
- Required attribution: `NOTICE`.

## Model-use obligations before deployment

- Surface the Krea 2 Community License to users before weights are downloaded.
- Implement reasonable content moderation for any deployment exposed to other
  users, as required by the model terms.
- Follow applicable AI-output disclosure requirements.
- Do not imply affiliation with or endorsement by Krea.

## Remaining before public release

- Complete clean-host GPU acceptance and record it in `GPU_ACCEPTANCE.md`.
- Verify the displayed download/license flow in the target WanGP revision.
