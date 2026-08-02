# WanGP Krea 2 Identity Edit (2.0.0-Release)

Description:
Standalone WanGP model-plugin project for instruction-based, identity-preserving image editing with the community [Krea 2 Identity Edit](https://huggingface.co/conradlocke/krea2-identity-edit) LoRA.

## Intended capabilities

- Krea 2 Turbo and Raw editing modes.
- One source/reference image for normal edits.
- Ordered two-image editing: scene first, subject second.
- VAE source latents injected as clean in-context transformer tokens.
- Image-grounded Qwen3-VL instruction conditioning.
- Configurable `grounding_px`, defaulting to 768.
- Automatic Identity Edit LoRA download at strength 1.0.
- v1.2 full-rank, rank-128 and rank-64 LoRA choices, with full rank as default.
- Optional experimental depth control from a separate Control Image upload.
- Reference-free **Depth + Prompt** generation using only the prompt and depth
- Adjustable depth block-adapter strength from 0.0 to 2.0.
- Whole-frame, inside-mask and outside-mask depth control with edge feathering.
- Optional automatic background isolation for the second/subject reference.
- Optional Registered Spatial Outpaint from one uploaded image or as a
  separately scheduled second generation after Identity Edit.
- A generic editable default-prompt template and WanGP prompt-guidelines modal.
- An adaptive main form plus a **Show Advanced Settings** Yes/No launcher for
  less frequently changed Krea 2 options.
- WanGP/MMGP model offloading and low-VRAM compatibility where practical.

## Repository status

Implemented:

- public WanGP model-plugin handler with collision-free architecture names;
- full Qwen3-VL visual grounding using WanGP's complete
  `Qwen3-VL-4B-Instruct_bf16.safetensors` language-and-vision checkpoint by
  default, with no obsolete split vision-checkpoint dependency;
- clean VAE source tokens ordered as `[text | scene | subject | target]` with
  reference RoPE frames 1/2 and target frame 0;
- grounded positive, CFG-negative and NAG conditioning;
- dynamic Identity Edit v1.2 full/r128/r64 LoRA selection at strength 1.0;
- optional Identity Edit subject-attention scheduling across denoising thirds,
  with constant attention retained as the compatibility default;
- target-only depth conditioning using WanGP's native Depth Anything V2 Large
  preprocessor, the Qwen image VAE and `Patil/Krea-2-depth-controlnet`;
- reference-free Depth + Prompt conditioning with no identity adapter or
  grounded reference stream;
- additive MMGP stacking of the Identity Edit and depth block adapters while
  clean identity-reference tokens retain Krea 2's native input projection;
- scene-safe WanGP reference preprocessing: reference 1 remains the complete
  scene while automatic background removal may isolate reference 2;
- spatial depth masking before VAE encoding and again at the target-token
  projection, with configurable feathering;
- one/two-reference validation, first-reference output aspect matching,
  aspect-preserving secondary-reference FIT geometry and a default 2 MP
  reference-mode cap, with an Advanced Settings override for full selected
  resolution; reference-free Depth + Prompt uses the selected resolution
  directly;
- WanGP callbacks, interruption, LoRA scheduling and standard image tensors.

Still pending before a stable release:

- actual model/LoRA downloads and end-to-end Turbo/Raw generations;
- golden images and two-reference ordering acceptance;
- cancellation, switching and peak VRAM measurements;
- release-facing profiles, screenshots, a fresh GitHub-URL installation and the
  final `v2.0.0` release tag.

## Review status

The Turbo and Raw definitions remain hidden while the GPU acceptance matrix is
incomplete. The implementation is ready for source review, not release or merge
acceptance. Do not upload model weights, user images or generated outputs to
this repository. The first image is the scene reference; when using two images,
the second is the subject reference.

The upstream-recommended v1.2 full-rank weight is available from the **Identity
Edit LoRA** dropdown and is the current default. This plugin matches the output
aspect ratio to the first reference and preserves the second reference's aspect
ratio through centred FIT latent geometry. The upstream node pack's optional
`ref_boost` dial is not implemented.

### Depth + prompt only

Choose **Depth + prompt only — no identity reference** under **Identity
method**, then select **Transfer Depth** and upload a Control Image. Do not
upload a Reference Image. This mode loads the depth adapter without Identity
Edit, uses the selected output resolution, and conditions generation only with
the prompt and processed depth map. Depth strength must be greater than `0`.

This is useful as a normal reference-free workflow and diagnostic baseline.
Whole-frame and masked depth controls remain available; Registered Outpaint is
not available in the same task.

The default prompt is an editable template. Replace every bracketed placeholder
before generating. Use the prompt-help icon beside WanGP's prompt field for the
full Krea 2 Identity Edit guide, including one-reference, two-reference, depth,
masking, Raw and Turbo guidance. The guide is supplied through WanGP's native
`prompt_infos` modal rather than a plugin-specific UI.

### Adaptive and advanced settings

The main form now keeps only currently relevant controls visible. The top
**Identity Edit References** selector is authoritative: Identity Edit exposes
its fidelity selector, and Depth + Prompt hides identity controls entirely.
Subject reference preparation is hidden for **Depth + Prompt only**, and the
SAM3 segmentation phrase appears only after **SAM3 semantic isolation** is
selected. Depth strength remains directly below the control image and appears
only when Transfer Depth is active.

Set **Show Advanced Settings** to **Yes** to open the plugin-owned floating
panel. Drag its title bar to move it while keeping the main form visible. It
contains Generation Process, reference-grounding budget, Identity Edit LoRA
variant, the reference-mode output-resolution limit, subject-attention timing,
depth-mask feathering, built-in Identity Edit/Depth timing
and additional-LoRA timing. **Apply Settings**
saves them to the queued task; **Cancel** or **Close** leaves the previous values
unchanged. The selector returns to No after the panel closes. Older presets
with those values stored as separate fields are migrated automatically.

The **Reference-mode output resolution limit** defaults to the safe 2 MP cap.
Choose **Use full selected resolution** when reference-based generation has
enough memory. Depth + Prompt has no reference-token memory overhead and
therefore always uses the selected output dimensions; this selector does not
affect that mode.

For Identity Edit, **Subject attention: constant** preserves the fidelity
profile selected on the main form for every denoising step. The experimental
**Ramp over denoising thirds** option instead uses absolute early, middle and
final subject multipliers; its defaults are `1x`, `2x` and `8x`. Scene attention
keeps the selected profile's scene multiplier. This allows composition and head
placement to form before stronger subject fidelity is introduced. Ramp values
must be non-decreasing. Depth + Prompt ignores these controls.

### Subject-reference isolation

With two references, reference 1 is always treated as the protected scene and
reference 2 as the optional subject. Choose **SAM3 semantic isolation
(recommended)** under **Subject reference preparation**, then enter a short
visual identifier such as `person`, `woman on the left`, or `blue car` in
**Subject segmentation phrase**. WanGP's shared SAM3.1 Magic Mask identifies
the requested subject before the plugin protects narrow edge details, feathers
the silhouette and composites it onto white. The resulting reference is sent
through both Qwen3-VL and the clean-reference VAE path.

The setting is off by default. In Identity Edit it applies only to reference 2
and intentionally does nothing to a single scene reference. **rembg fast
isolation (legacy)** remains available for a quicker cut-out and avoids
PyMatting's unstable incomplete-Cholesky alpha refinement, but it can retain
unrelated objects or remove subject parts in difficult scenes. SAM3 is binary
rather than a true alpha-matting model, so transparent clothing and extremely
fine hair can still require a manually prepared cut-out.

### Experimental depth control

Choose **Transfer Depth** in **Control Image Process** to reveal WanGP's separate
**Control Image** uploader. Upload the image whose composition and 3D structure
should guide the generated target. Identity references stay in the independent
**Reference Images** section and keep their scene-first, subject-second order.

Start at depth strength `1.0`. Lower values relax structural adherence; `0`
bypasses adapter injection, although WanGP still prepares the selected control
image. The slider uses `0.01` increments so values such as `0.98` and `0.99` are
available. It appears below the Control Image only when **Transfer Depth** is
selected. Selecting **No Control Image** hides it and skips depth preprocessing
entirely.

When combining depth with an additional Unlocker or style LoRA, use **Depth
first, additional LoRAs late (recommended)**. Identity Edit and depth remain
active throughout the generation, while user-added LoRAs use an editable
early/middle/final ramp. The defaults are 0% of the chosen strength during the
first third, 25% during the middle third and 100% during the final third. The
three ramp sliders appear in Advanced Settings when **Depth first** timing is
selected and must remain non-decreasing. This lets early steps establish pose,
subject scale and placement before the additional LoRA concentrates on identity,
texture or style. Select **Additional LoRAs active for all steps** to restore
WanGP's ordinary scheduling behavior.

For Identity Edit with Transfer Depth, **Built-in Identity Edit and Depth
timing** offers **Depth layout -> Identity refinement (experimental)**. The
default remains simultaneous behavior for A/B comparison. The experimental
preset applies independent three-stage ramps to the built-in adapters: Depth
defaults to `1.00`, `0.50`, `0.00`; Identity Edit defaults to `0.25`, `0.75`,
`1.00`. Depth's direct target-token projection follows both the selected Depth
Strength and the same Depth ramp. Keep subject-attention fidelity constant for
the initial comparison so that the adapter schedule is the only variable.

Use **Control Area Processed** to choose:

- **Whole Frame** for the complete depth map;
- **Inside Mask** to keep depth influence only in the painted area;
- **Outside Mask** to exclude the painted area and control everything around it.

Inside/Outside Mask reveals WanGP's combined Control Image and Mask editor.
The host's **Mask Expand** control adjusts the hard selection before depth
processing; **Depth mask feather** then softens its edge from `0` to `64`
pixels. The plugin reports RGB channel disagreement in the processed control,
canonicalizes the active depth area to three identical grayscale channels,
neutralizes excluded RGB before VAE encoding, and multiplies the trained depth
contribution by the downsampled mask at Krea's target-token grid. Excluded areas
therefore receive the native target projection rather than a black depth value.
For a person in front of railings, paint the person and use
**Inside Mask** so rail geometry outside the silhouette does not condition the
target. Geometry that visibly crosses the person may still need manual mask
refinement or removal from the control image.

For a mask made in Photoshop or another external editor, choose **Custom Mask —
White Area** or **Custom Mask — Black Area**, upload the normal Control Image,
and upload the separate file through **Optional Custom Depth Mask**. These
dedicated choices keep WanGP's full-frame depth preprocessing intact before the
plugin applies the external mask. A custom upload is resized to the processed
control and receives the same **Depth mask feather**.
Its aspect ratio must match the Control Image/output aspect ratio, although its
pixel dimensions may differ. **Custom Mask — White Area** selects white pixels;
**Custom Mask — Black Area** reverses the selection. Do not use an uploaded mask
with either Painted Mask option: WanGP composites those controls before the
plugin receives them. WanGP's native **Mask Expand** applies only to a painted
editor mask, so any expansion or contraction required by a custom mask must be
authored into that file.

When a Custom Mask mode is selected, the plugin shows **Control + Custom Mask
Alignment** directly below the mask upload. It is a preview only: the original
Control Image remains intact so depth can still be estimated correctly.
**Generate Effective Depth Preview** runs WanGP's configured Depth Anything
preprocessor on the complete Control Image and then applies the uploaded mask,
its White/Black selection and the current **Depth mask feather**. The result is
black outside the effective area; setting feather to `0` produces a hard edge.
Changing the control, mask, mode or feather clears that generated preview.

The plugin reuses WanGP's native depth preprocessor and does not download or
load a duplicate estimator. This adapter targets **Depth Anything V2 Large**;
leave WanGP's **Depth Anything Preprocessor** setting on `vitl` (V2 Large).
The DA3 choices are not validated for this adapter. When depth is enabled it
downloads the approximately 862 MB depth adapter through the normal LoRA cache.
The processed depth image is shown through WanGP's normal queue-thumbnail path.
Combined Identity Edit + depth inference has not yet passed the GPU acceptance
matrix.

These masks limit conditioning; they are not output/inpainting masks and do not
guarantee that unselected output pixels remain unchanged.
Identity Edit also keeps clean reference tokens in the live denoising stream.
Its 2x/4x/8x subject-fidelity profiles can therefore compete with masked depth
for pose and placement; compare the standard 1x/1x profile when depth geometry
is the priority.

### Experimental registered spatial outpaint

> [!IMPORTANT]
> Registered Outpaint supports one expansion axis per pass: left/right or
> top/bottom. Selecting both axes in manual margin mode is rejected until the
> planned two-pass interior-placement implementation is GPU-validated.

In Advanced settings, choose **Outpaint uploaded image only** to expand a single
uploaded reference without Identity Edit, or **Identity Edit, then outpaint
result** to use the first generation as the protected source of a second pass.
Configure WanGP's native **Spatial Outpainting** margins or target aspect ratio.
The source pixels are restored after generation with a 32-pixel inward seam
feather.

This is a real additional generation rather than a pixel-only postprocess.
Identity and depth adapters are inactive during the outpaint pass, and depth is
rejected while Registered Outpaint is selected. Set **Registered Outpaint
prompt** in Advanced settings to describe only the complete, context-consistent
continuation. The outpaint pass receives that dedicated prompt and a clean
default negative prompt; it does not reuse the main generation prompt.

In target-aspect-ratio mode, the direction sliders act as placement weights. The
plugin derives how many pixels the final canvas actually needs, then applies
only the left/right or top/bottom pair for that expansion axis. Manual
outpainting supports one axis per pass. A mixed horizontal-and-vertical
selection is rejected with an actionable error until registered two-pass
interior placement is implemented and GPU-validated.

Start with:

1. `SPEC.md`
2. `ARCHITECTURE.md`
3. `IMPLEMENTATION_PLAN.md`

## Local development

This repository must be installed as one directory beneath WanGP's `plugins/` directory. During development, use a directory junction or symbolic link rather than maintaining a second copy.

Normal inference logs omit image/tensor hashes and sampled conditioning
statistics. For a focused diagnostic run, set `KREA2_IDENTITY_DEBUG=1` before
starting WanGP; this enables those heavier diagnostics without changing the
saved generation settings.

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

The current compatibility floor is WanGP v12.34. The handler performs a feature
guard before loading and reports an actionable update error when the required
Krea 2/Qwen3-VL interfaces are absent. The public API was audited at WanGP
commit `6b92c54f92bde24d6d309d6f61249353b0ec783d`.

## Licensing

Original plugin code is Apache-2.0. The Krea 2 base model and Identity Edit
weights are governed separately by the Krea 2 Community License. The reference
ComfyUI node implementation is Apache-2.0. See `LICENSE`, `NOTICE` and
`THIRD_PARTY_NOTICES.md`.

The host-provided Depth Anything V2 Large estimator is CC-BY-NC-4.0 and therefore
adds a non-commercial restriction when automatic depth estimation is used.
