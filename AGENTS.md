# Workspace instructions

## Project identity

This repository is a standalone WanGP **model plugin** for Krea 2 Identity Edit. It is not a Pinokio launcher, a standalone application, or a patch to WanGP core.

Before implementation work, read these files in order:

1. `SPEC.md`
2. `ARCHITECTURE.md`
3. `IMPLEMENTATION_PLAN.md`
4. `THIRD_PARTY_NOTICES.md`

## Non-negotiable constraints

- Keep the repository directly installable as one folder beneath WanGP's `plugins/` directory.
- Do not require users to edit WanGP core files.
- Do not register `krea2_raw` or `krea2_turbo`; those names belong to WanGP. Keep the plugin's unique `krea2_identity_*` architecture names.
- Do not make either default model definition visible until dual conditioning works and GPU acceptance tests pass.
- Do not fall back to applying the LoRA to WanGP's text-only Krea 2 path. Identity Edit requires both clean source-latent tokens and image-grounded Qwen3-VL encoding.
- Do not commit model weights, generated outputs, user images, environments or caches.
- Preserve interruption, preview callbacks, model switching and MMGP offload behavior.
- Keep scene-first, subject-second ordering for two-reference editing.
- Record the source and license of copied or adapted code.

## Validation

Run before finishing any change:

```sh
python tools/validate_scaffold.py
python -m unittest discover -s tests -v
```

For inference changes, also document the WanGP revision, GPU, profile, resolution, model variant, LoRA variant, peak VRAM and observed result.

