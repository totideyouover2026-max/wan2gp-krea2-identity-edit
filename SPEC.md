# Product specification

## Objective

Provide Krea 2 Identity Edit as a separately installable WanGP model plugin without modifying WanGP core files.

## Upstream inputs

- WanGP Krea 2 Raw/Turbo implementation: `models/krea2/` in the host WanGP checkout.
- Identity Edit weights and recommended settings: <https://huggingface.co/conradlocke/krea2-identity-edit>
- Reference dual-conditioning implementation: <https://github.com/lbouaraba/comfyui-krea2edit>
- Krea 2 base checkpoints and license: <https://huggingface.co/krea/Krea-2-Raw>

## Functional requirements

1. Register unique WanGP model architectures:
   - `krea2_identity_turbo`
   - `krea2_identity_raw`
2. Reuse WanGP's Krea 2 transformer, Qwen image VAE and compatible scheduler behavior where possible.
3. Load the complete Qwen3-VL conditioning path required for image grounding, including the visual encoder.
4. Accept one or two ordered reference images.
5. VAE-encode each reference and prepend its clean latent tokens to the noisy target sequence.
6. Assign RoPE frame indices in training order:
   - target: frame 0
   - first reference/scene: frame 1
   - second reference/subject: frame 2
7. Encode the instruction with the same source images through Qwen3-VL.
8. Ground the negative/empty prompt with the same images whenever guidance requires an unconditional pass.
9. Expose `grounding_px` with a 768 default and a sensible bounded range.
10. Download and apply Identity Edit at strength 1.0, with v1.2 full and v1.1 full/r128/r64 choices.
11. Preserve WanGP interruption, preview callback, LoRA scheduling and MMGP offload behavior.
12. Return normal WanGP image output tensors and metadata.

## Recommended presets

### Turbo

- 8–12 steps; default 10.
- Effective CFG 1.0/no separate unconditional pass.
- LoRA strength 1.0.
- Best for normal edits, recoloring, additions, restyling and re-staging.

### Raw

- Approximately 20 steps.
- Effective CFG approximately 3.0, mapped carefully to WanGP's guidance convention.
- Grounded empty negative conditioning.
- Best for removals and large deletions.

## Input constraints

- Output aspect ratio should match the primary source image.
- Keep output at or below 2 MP.
- Prefer roughly 1–1.5 MP for two-person editing.
- For two-reference composition, image 1 is the scene and image 2 is the subject.

## Non-goals for the first release

- Training or fine-tuning the LoRA.
- Video editing.
- More than two simultaneous references.
- A standalone web server or Pinokio launcher.
- Bundling model weights inside the Git repository.

## Release acceptance criteria

- Clean install from a public GitHub URL through WanGP's plugin manager.
- No modifications required in the host WanGP checkout.
- Turbo and Raw single-reference golden tests pass.
- Two-reference ordering test passes.
- The v1.2 full and v1.1 full/r128/r64 LoRAs load successfully.
- Cancellation and model switching release memory correctly.
- At least one documented low-VRAM profile is validated.
- License and moderation obligations are documented.

