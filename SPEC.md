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
3. Use the complete BF16 Qwen3-VL conditioning path required for image
   grounding, including the visual encoder, from WanGP's unified Identity Edit
   checkpoint.
4. Accept one or two ordered reference images.
5. VAE-encode each reference and prepend its clean latent tokens to the noisy target sequence.
6. Assign RoPE frame indices in training order:
   - target: frame 0
   - first reference/scene: frame 1
   - second reference/subject: frame 2
7. Encode the instruction with the same source images through Qwen3-VL.
8. Ground the negative/empty prompt with the same images whenever guidance requires an unconditional pass.
9. Expose `grounding_px` with a 768 default and a sensible bounded range.
10. Download and apply Identity Edit at strength 1.0, with v1.2 full (default),
    r128 and r64 choices.
11. Preserve WanGP interruption, preview callback, LoRA scheduling and MMGP offload behavior.
12. Return normal WanGP image output tensors and metadata.
12a. Offer opt-in Identity Edit reference-fidelity profiles that add a
    target-query-to-reference-key attention-logit bias. Keep 1x subject/1x
    scene as the unchanged default; treat the last reference as the subject
    and every earlier reference as scene conditioning.
12aa. Offer an optional Identity Edit subject-attention ramp over denoising
    thirds. Constant mode must remain the default. Ramp mode uses validated,
    non-decreasing absolute subject multipliers in the 1x..8x range while the
    selected fidelity profile continues to control scene attention.
13. Accept a dedicated depth control image separately from the ordered identity
    references, reuse WanGP's native Depth Anything V2 Large preprocessing and
    queue preview, VAE-encode the processed tensor, and apply the public Krea 2
    depth ControlNet-LoRA to target tokens only.
14. Expose WanGP's native Control Image Process dropdown and upload area plus a
    bounded depth-strength setting, with No Control Image selected by default.
15. Support whole-frame, inside-mask and outside-mask depth conditioning through
    WanGP's native mask editor. Neutralize excluded control pixels before VAE
    encoding and gate the depth contribution again at the target-token grid.
16. Expose bounded depth-mask feathering and preserve WanGP's mask expansion.
16a. Accept an optional, separately uploaded grayscale or alpha depth mask
    through WanGP's generic custom-guide attachment. White selects the inside
    area and black-area mode reverses it. Keep uploaded and painted-mask routes
    mutually exclusive so WanGP produces a complete depth map before plugin
    gating, without requiring WanGP core changes.
16b. Show the uploaded custom mask as an image and provide an on-demand
    effective-depth preview that estimates complete depth before applying mask
    polarity and feathering. Keep preview inference bounded and release the
    active WanGP generation model first when GPU headroom is insufficient.
17. Allow optional semantic SAM3 subject isolation or fast rembg background
    removal for reference 2 while protecting reference 1/scene and keeping
    removal off by default.
18. Provide a generic editable default-prompt template and expose model-specific
    prompting guidance through WanGP's native prompt-help modal contract.
19. Provide registered spatial outpainting as either an outpaint-only task for
    one uploaded image or an isolated second generation after Identity Edit.
    Give that pass a dedicated editable prompt with a conservative continuation
    default; do not reuse the main generation prompt or its negative prompt.
20. Reuse WanGP's spatial controls, register source coordinates to the
    destination rectangle, cache isolated source K/V, and restore protected
    source pixels with an inward seam feather. Resolve ratio-mode direction
    values as placement weights on the axis actually expanded by the final
    canvas. Until interior placement is implemented as two passes, reject
    manual margins that mix horizontal and vertical expansion.
21. When depth is active, optionally apply an editable three-stage per-step
    ramp to user-added LoRAs during denoising while leaving the plugin's
    selected identity method and depth adapters active throughout, so geometry
    is established before style/detail adapters peak. Default to multipliers
    0.00, 0.25 and 1.00 across the early, middle and final thirds.
22. Expose a reference-free Depth + Prompt method that loads only the depth
    adapter, uses text-only Qwen3-VL conditioning, preserves the requested
    output aspect, and requires an active depth control image with nonzero
    strength.
23. Expose a mutually exclusive raw Control Image process for Identity Edit. It
    must bypass depth preprocessing, use the uploaded image as WanGP's native
    low-denoise source, preserve the selected canvas aspect, and leave the
    ordinary Identity Edit reference stream as the identity and visual-content
    authority.
24. Keep the main WanGP form adaptive: show reference preparation only for
    reference-backed methods, show its SAM3 phrase only when SAM3 is selected,
    and keep depth strength conditional on depth control. Provide a Yes/No
    Advanced Settings launcher that opens a plugin-owned modal for the less
    frequently changed process, grounding, Identity Edit LoRA, subject-attention,
    mask-feather and user-LoRA timing controls. Persist those values through one canonical JSON
    custom setting and migrate older flat settings without WanGP core changes.

## Recommended presets

### Turbo

- 8–12 steps; default 8, matching WanGP's native Identity Edit setup.
- Effective CFG 1.0/no separate unconditional pass.
- LoRA strength 1.0.
- Best for normal edits, recoloring, additions, restyling and re-staging.

### Raw

- Approximately 20 steps.
- Effective CFG approximately 3.0, mapped carefully to WanGP's guidance convention.
- Grounded empty negative conditioning.
- Best for removals and large deletions.

## Input constraints

- Identity Edit output aspect ratio should match the primary source image by
  default.
- Keep output at or below 2 MP.
- Prefer roughly 1–1.5 MP for two-person editing.
- For two-reference composition, image 1 is the scene and image 2 is the subject.
- A lone Identity Edit reference keeps the established `KI` main-image/scene
  role.
- Identity Edit reference-fidelity boosts follow that same order: the last
  reference receives the subject multiplier and earlier references receive the
  scene multiplier. They do not apply to Depth + Prompt.
- In subject-attention ramp mode, the last-reference multiplier changes once per
  distinct denoising timestep; repeated CFG/NAG transformer calls at the same
  timestep must not advance the schedule.
- Preserve reference 2's aspect ratio when producing its clean VAE tokens;
  centre its aligned latent grid within the target instead of stretching it.
- Experimental depth control uses a separate Control Image and does not add or
  reorder an identity reference.
- Depth + Prompt accepts no identity references and uses the selected output
  resolution rather than deriving its aspect from an image reference.
- Direct Image to Identity Edit accepts the normal one or two ordered Identity
  Edit references plus one separate Control Image, uses WanGP's native 0..1
  denoising strength, and initially supports whole-frame refinement only. It
  is unavailable for Depth + Prompt and Registered Outpaint.
- Depth masks limit conditioning only; they are not output/inpainting masks.
- Automatic reference isolation applies to Identity Edit reference 2.
- Prompt documentation must remain generic and must not encode private test
  images or conversation-specific scenarios.
- Registered Outpaint requires one source per pass. Depth is unavailable in an
  outpaint pass; Identity + Outpaint uses two separately scheduled passes. Its
  dedicated prompt describes only context-consistent canvas continuation and is
  isolated from the main generation prompt.

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
- The v1.2 full/r128/r64 LoRAs load successfully.
- The depth adapter loads alongside Identity Edit and affects only target-token
  projection while clean reference-token projection remains native.
- Inside/outside depth masks produce zero depth-adapter contribution outside
  the selected, feathered area.
- Subject background removal never alters the first scene reference.
- Cancellation and model switching release memory correctly.
- At least one documented low-VRAM profile is validated.
- License and moderation obligations are documented.

