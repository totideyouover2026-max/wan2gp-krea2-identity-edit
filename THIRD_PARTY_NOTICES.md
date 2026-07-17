# Third-party notices and release licensing

This file records runtime dependencies and adapted implementation sources. It is
not legal advice.

## Krea 2 and Identity Edit weights

- Base model: <https://huggingface.co/krea/Krea-2-Raw>
- Adapter: <https://huggingface.co/conradlocke/krea2-identity-edit>
- Supported adapter files: v1.2 full rank and v1.1 full/r128/r64.
- Governing terms: Krea 2 Community License and associated acceptable-use terms.
- Do not commit or redistribute the weights through this Git repository.
- The plugin should download weights from their authoritative repositories and clearly surface applicable terms.

## Reference node implementation

- Project: <https://github.com/lbouaraba/comfyui-krea2edit>
- Stated license: Apache-2.0.
- The sequence construction, RoPE frame convention, grounded prompt template
  and reference preprocessing in `models/krea2_identity_main.py` are an
  independent WanGP adaptation of the project's Apache-2.0 implementation.
- Attribution is preserved in `NOTICE`. No ComfyUI wrapper code is bundled.

## WanGP

- Project: <https://github.com/deepbeepmeep/Wan2GP>
- Audited public API revision:
  `5582327dc25e45fec6cda0f27144d4dcf7ed104b` (2026-07-11).
- Current upstream license at that revision: WanGP Community License 2.0.
- WanGP is not bundled. The plugin imports its installed Krea 2, Qwen3-VL,
  scheduler, VAE and MMGP interfaces at runtime.

## Qwen3-VL visual weights

- Source: `Comfy-Org/Krea-2`, file
  `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`.
- The plugin loads only the `model.visual` prefix; those tensors are BF16 even
  though the combined checkpoint name describes its quantized language portion.
- The file is opened through safetensors and filtered before MMGP loading so the
  unrelated language tensors are not materialized in RAM.
- Qwen3-VL terms and the source repository's model card apply.

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
