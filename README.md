# WanGP Krea 2 Identity Edit

> [!IMPORTANT]
> This is a roughly put together temporary plug-in for Wan2GP users to be able to test out Krea2 Identity Edit. It has been quickly created with minimal testing, and serves only to be a temporary solution while DeepBeepMeep is away. Hopefully once DeepBeepMeep returns, they will be able to assess whether the Krea2 Identity Edit can be added smoother as a pipeline implementation.

Description:
Standalone WanGP model-plugin project for instruction-based, identity-preserving image editing with the community [Krea 2 Identity Edit](https://huggingface.co/conradlocke/krea2-identity-edit) LoRA.

> [!IMPORTANT]
> The dual-conditioning runtime is available as an experimental public preview and the model definitions are visible for testing. Do not treat it as release-ready until the acceptance record in `GPU_ACCEPTANCE.md` is complete.

## Intended capabilities

- Krea 2 Turbo and Raw editing modes.
- One source/reference image for normal edits.
- Ordered two-image editing: scene first, subject second.
- VAE source latents injected as clean in-context transformer tokens.
- Image-grounded Qwen3-VL instruction conditioning.
- Configurable `grounding_px`, defaulting to 768.
- Automatic Identity Edit v1.1 LoRA download at strength 1.0.
- Full, rank-128 and rank-64 LoRA variants.
- WanGP/MMGP model offloading and low-VRAM compatibility where practical.

## Repository status

Implemented:

- public WanGP model-plugin handler with collision-free architecture names;
- full Qwen3-VL visual grounding using the BF16 visual tower contained in
  `Comfy-Org/Krea-2/text_encoders/qwen3vl_4b_fp8_scaled.safetensors`;
- clean VAE source tokens ordered as `[text | scene | subject | target]` with
  reference RoPE frames 1/2 and target frame 0;
- grounded positive, CFG-negative and NAG conditioning;
- dynamic Identity Edit v1.1 full/r128/r64 LoRA selection at strength 1.0;
- one/two-reference validation, first-reference aspect matching and a 2 MP cap;
- WanGP callbacks, interruption, LoRA scheduling and standard image tensors.

Still pending before a stable release:

- actual model/LoRA downloads and end-to-end Turbo/Raw generations;
- golden images and two-reference ordering acceptance;
- cancellation, switching and peak VRAM measurements;
- release-facing profiles, screenshots, a fresh GitHub-URL installation and the `v0.1.0` tag.

## Experimental preview

The Turbo and Raw entries are intentionally visible so WanGP users can test
the current implementation before the full GPU acceptance matrix is complete.
Expect model downloads, VRAM usage and image quality to vary by WanGP version,
GPU and memory profile. Do not upload model weights, user images or generated
outputs to this repository. The first image is the scene reference; when using
two images, the second is the subject reference.

Start with:

1. `SPEC.md`
2. `ARCHITECTURE.md`
3. `IMPLEMENTATION_PLAN.md`

## Local development

This repository must be installed as one directory beneath WanGP's `plugins/` directory. During development, use a directory junction or symbolic link rather than maintaining a second copy.

Windows example, run from the WanGP `app` directory:

```powershell
New-Item -ItemType Junction `
  -Path "plugins\wan2gp-krea2-identity-edit" `
  -Target "D:\path\to\wan2gp-krea2-identity-edit"
```

Linux/macOS example:

```sh
ln -s /path/to/wan2gp-krea2-identity-edit plugins/wan2gp-krea2-identity-edit
```

Enable the plugin in WanGP's Plugins tab and restart WanGP after manifest or handler changes.

## Validation

The structural checks use only the Python standard library:

```sh
python tools/validate_scaffold.py
python -m unittest discover -s tests -v
```

Against a clean public WanGP source checkout, also run:

```sh
python tools/validate_wangp_contract.py /path/to/Wan2GP
```

To verify the remote safetensors headers without downloading the weights:

```sh
python tools/validate_remote_assets.py
```

From the Python environment belonging to a clean designated WanGP checkout:

```sh
python tools/smoke_test_wangp_import.py /path/to/Wan2GP
```

This imports and queries the handler without downloading or loading weights.

These checks do not prove that inference works. GPU integration tests must be added during implementation.

For compatibility and GPU acceptance procedures, see `GPU_ACCEPTANCE.md`.

## Installation for testing

Users can install the GitHub repository URL through WanGP's **Plugins → Install New Plugin** interface, enable it, and restart WanGP. This is an experimental preview, so keep the acceptance limitations above in mind.

## Compatibility

The current compatibility floor is WanGP v12.3. The handler performs a feature
guard before loading and reports an actionable update error when the required
Krea 2/Qwen3-VL interfaces are absent. The public API was audited at WanGP
commit `5582327dc25e45fec6cda0f27144d4dcf7ed104b`.

## Licensing

Original plugin code is Apache-2.0. The Krea 2 base model and Identity Edit
weights are governed separately by the Krea 2 Community License. The reference
ComfyUI node implementation is Apache-2.0. See `LICENSE`, `NOTICE` and
`THIRD_PARTY_NOTICES.md`.
