# Architecture

## Plugin boundary

This project is a WanGP **model plugin**, not a standalone UI application and
not a Pinokio launcher. `plugin_info.json` points WanGP at one `family_handler`,
a defaults root and a profiles root. A small `plugin.py` extension uses WanGP's
public component-request/insertion API solely to make the model's own settings
adaptive and to render its Advanced Settings modal. It does not patch WanGP
core or add a separate top-level application tab.

The plugin uses unique architecture identifiers because WanGP rejects multiple handlers that claim the same model type.

## Runtime flow

```text
reference image(s)
   ├─ Qwen image VAE ─ clean latent tokens ───────────────┐
   └─ Qwen3-VL vision tower + instruction ─ text states ─┤
depth control image (optional)
   └─ WanGP Depth Anything V2 + optional mask + queue preview
      └─ neutralize excluded pixels ─ Qwen image VAE
         └─ feathered token gate ─ depth tokens ──────────┤
                                                         v
noise target ─ target latent tokens ──> [text | refs | depth-target]
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

The plugin currently targets WanGP v12.34 and imports the host's
`Krea2Pipeline`, transformer/VAE loaders, Krea 2 MMDiT modules and Qwen3-VL
model classes. It does not vendor WanGP source.

`IdentityKrea2Pipeline` reuses the host denoising loop unchanged. Before that
loop it VAE-encodes the ordered references and installs their packed clean
tokens on the transformer for the duration of one call. The identity forward
methods build the expanded stream, preserve the host block/callback/LoRA
execution, and clear the temporary source state in a `finally` block.

Experimental depth control receives its own image from WanGP's native Control
Image upload. WanGP's shared `DV` path derives and displays inverse depth with
Depth Anything V2 Large, then passes the same `C x T x H x W` tensor in
`input_frames`. The plugin VAE-encodes and patch-packs its first frame at the
target resolution. The public depth checkpoint's
6144x128 input projection is split so clean references continue through Krea
2's native 64-feature `first` projection while the target receives the trained
64-feature control contribution. Its rank-64 block adapters are normalized to
MMGP's key convention and stacked with the selected Identity Edit adapter.
The adapter compatibility target is WanGP's `vitl` V2 Large depth setting; DA3
output is intentionally not claimed as compatible.

Depth-first additional-LoRA timing installs an explicit per-denoising-step ramp
in WanGP's LoRA schedule. Plugin-owned identity-method and depth adapters remain
at their selected strengths; only user LoRAs appended after that plugin-owned
head are scaled across the early, middle and final thirds. The editable ramp is
bounded to non-decreasing 0..1 multipliers and defaults to 0%, 25% and 100%.
This remains effective for Krea Turbo even though it normally exposes only one
host guidance phase. It allows depth to establish pose, scale and composition
before an Unlocker or style adapter reaches full user strength. The original
all-steps behavior remains selectable.

WanGP's `mask_preprocessing` contract supplies whole-frame, inside-mask and
outside-mask modes plus its mask-expansion control. In masked modes the host
passes the processed control in `input_frames` and a `0..1` spatial selection in
`input_masks`. The plugin replaces excluded raw-control pixels with normalized
black/far depth (`-1`) before VAE encoding, downsamples the feathered mask to the packed Krea
token grid, and multiplies only the depth projection by that mask. The native
target projection remains present everywhere, so an excluded area is not
misrepresented as black/far depth. Mask tensors are cleared with the other
temporary conditioning state in the generation `finally` block.

The model definition also exposes WanGP's generic `custom_guide` file slot as
an optional external depth mask. WanGP passes that attachment to the model
factory as `input_custom`; the plugin reads meaningful alpha when present or
otherwise converts luminance, validates its aspect against the processed
control, reverses it for the Black Area option, and routes it through the same
feathered token gate. The custom White/Black choices use WanGP's mask-selection
letters without `A`, preventing its painted-mask compositor from replacing
unpainted regions of the processed depth with source RGB. Uploaded and painted
mask routes are therefore mutually exclusive. Mask expansion remains a host
editor operation, so it is intentionally not applied to the separate file.
The plugin UI mirrors WanGP's filename-only custom-guide slot through an image
uploader while retaining the hidden host component for queue persistence. The
right-hand output contains only the optional effective-depth preview. It calls
WanGP's configured depth preprocessor with a single-frame video list on a
bounded-resolution copy of the complete Control Image, and only then applies
the uploaded mask and configured feather. When GPU headroom is insufficient,
the helper releases the active generation model before loading Depth Anything;
it then releases the preprocessor and CUDA cache. Neither generation input is
replaced.

Reference 1 uses WanGP's `K` main-scene semantic and is never eligible for
automatic background removal. Reference 2 may be isolated by the plugin's
reference postprocessor using WanGP's shared SAM3.1 Magic Mask API and an
editable semantic phrase. A small outward mask expansion protects narrow limbs
and fine details before a one-pixel edge feather and white compositing. The
faster legacy option uses rembg's direct predicted mask without PyMatting alpha
refinement, avoiding the latter's incomplete-Cholesky instability. The plugin
consumes the processed PIL reference for both grounded Qwen3-VL encoding and
clean VAE tokens, while the untouched first upload remains the authority for
output aspect ratio.

Identity Edit defaults to WanGP's complete
`Qwen3-VL-4B-Instruct_bf16.safetensors` checkpoint and loads its language and
visual modules together through MMGP. Its image processor is built from
WanGP's published `preprocessor_config.json`, matching the native Krea 2 Edit
conditioning contract. A temporary load-time encoder A/B also exposes the
former stack: the host-selected Quanto checkpoint supplies the language model,
`Comfy-Org/Krea-2` supplies the scaled-FP8 visual tower, and the plugin restores
the former manually configured Qwen2-VL processor. The built-in WanGP text
encoder selector is used because this choice must be known before model
construction; the unified BF16 checkpoint remains first and default.

Identity Edit v1.2 full/r128/r64 LoRAs are selected dynamically through `get_loras_transformer`,
which participates in WanGP's normal LoRA download, loading, scheduling and
unloading flow. The ai-toolkit `diffusion_model.` prefix is removed by a small
preprocessor before MMGP matches transformer modules.
The handler clears any legacy fixed Identity Edit LoRA in the installed model
definition so WanGP does not append it to the variant selected in the dropdown.

The Identity Edit path keeps its first-reference aspect-ratio matching behavior
for each supported variant. Reference 1 remains canvas-sized. Reference 2 uses
the upstream FIT behavior: near-matching ratios are centre-cropped, while larger
ratio differences are aspect-preservingly contained at an aligned size and
assigned a centred latent/RoPE grid. A temporary Advanced Settings diagnostic
can restore the former direct stretch to the output latent/RoPE geometry for
Picture 2 only. FIT remains the default; the switch does not alter Picture 1,
ReID, Depth + Prompt or the Qwen3-VL encoder stack. The v1.2 node pack's `ref_boost` attention
control is adapted as opt-in fidelity profiles within the existing identity
method selector, avoiding an additional WanGP custom-control slot. The default
profile remains 1x subject/1x scene and therefore produces the pre-boost
attention path. Boosted profiles add `log(multiplier)` only to target-query to
reference-key attention logits: the final reference is the subject and earlier
references are scenes. An optional subject-attention schedule replaces the
constant subject multiplier with absolute early/middle/final values over the
denoising thirds. The transformer tracks distinct descending timesteps, rather
than forward-call count, so CFG/NAG duplicate calls cannot advance the ramp
twice. Constant timing remains the default and scene attention remains fixed at
the selected profile value.
references are scenes. WanGP's boolean key-padding mask is converted to an
additive mask first. The plugin builds the `[batch, head, query, key]` layout
required across WanGP's complete Krea attention path: Krea transposes the
middle axes before dispatch, then the shared masked SageAttention/SDPA wrapper
transposes them back for its kernel. This matters only for query-dependent
masks; the standard singleton padding mask looks identical after either
transpose. Boost profiles are currently mutually exclusive with non-default
NAG because NAG constructs a different per-query attention mask.

The removed KI/native-I experiment is migrated away rather than retained as a
second variable. A lone Identity Edit reference always keeps the established
WanGP `KI` main-image role, and two references always remain scene first and
subject second. This isolates encoder precision and processor behavior as the
only intended variable in the temporary A/B.

ReID is a separate conditioning mode rather than another adapter layered onto
Identity Edit. It is restricted to the Turbo checkpoint and one identity
reference. Its reference mode deliberately omits WanGP's `K` main-scene flag,
so the host preserves the identity image's aspect ratio instead of resizing it
to the output canvas. Qwen3-VL receives the reference through the adapter's
trained `Picture 1: <vision>` message contract. One aspect-preserving,
16-pixel-aligned reference within a 384×384 area budget feeds both Qwen3-VL and
the clean-reference VAE, matching the published Comfy graph. The default path
keeps `[text | noisy target | clean reference]` as one live self-attention
stream once through every block at timestep zero and retain each block's
post-RoPE K/V. Target denoising appends those isolated cached keys and values at
every step. RoPE frame 1 identifies the reference. Only the target is decoded.
The experimental joint target/reference timestep-zero stream remains selectable
as an Advanced Settings diagnostic.
ReID output geometry follows the selected output resolution rather than the
reference aspect. It forces 8 Turbo steps and effective CFG 1, supports the
same target-only depth projection, and rejects Registered Outpaint.

ReID reference preparation optionally follows the release's YuNet helper. It
detects on a 320-square top-centred view, maps the strongest face back to the
original pixels, and takes an expanded head crop before both Qwen and clean-VAE
conditioning. The unchanged full reference remains available when clothing,
pose or scene context is intentionally part of the reference signal.

Depth + Prompt is the reference-free diagnostic and generation mode. It clears
the grounded-image list, enables the conditioner's existing text-only path and
runs the native `[text | noisy target]` stream with the trained depth projection
added only to target tokens. Its LoRA list begins with the depth adapter rather
than Identity Edit or ReID, so it provides a clean baseline for determining
whether pose behavior comes from depth itself or competition with identity
conditioning. It requires Transfer Depth with strength above zero, follows the
selected output resolution, supports the same whole-frame/inside/outside mask
paths, and cannot be combined with Registered Outpaint.

The optional **Depth then ReID** generation process composes those existing
paths instead of replacing either one. Phase 1 calls Depth + Prompt with the
ReID adapter and all user-added LoRAs set to zero. Its decoded RGB result is the
source image for phase 2, which uses WanGP's existing source-image low-denoise
restart with a full-frame mask. Phase 2 restores ReID and user-added LoRAs at
their selected strengths and can either keep or disable depth. The handoff is a
VAE decode/re-encode boundary, not an in-memory latent continuation; this keeps
the experiment within WanGP's public pipeline contract and preserves normal
callbacks, interruption and MMGP LoRA scheduling. Standard single-pass routing
is unchanged when the option is off.

The optional **Direct Image → ReID Edit** Control Image process uses WanGP's
native `VG` contract. `V` sends the uploaded image through the host's unchanged
raw preprocessing path and `G` exposes its standard 0..1 denoising slider. The
plugin converts the processed first frame back to RGB, supplies a full-frame
edit mask, and calls the same source-image restart used by phase two, but in a
single ReID pass. The separate ReID reference still supplies identity through
Qwen3-VL and the selected official isolated-cache or diagnostic joint timestep-zero
reference path. Depth preprocessing and the depth adapter are not loaded for
this route. Initial support deliberately rejects painted masks, two-phase
generation and Registered Outpaint.

WanGP v12.34 exposes and persists only the first five model-defined custom
controls. The handler uses four slots for the context-sensitive main form:
identity method, depth strength, subject-reference preparation and the SAM3
phrase. The fifth slot is an always-hidden canonical JSON carrier. `plugin.py`
inserts a visible **Show Advanced Settings** Yes/No launcher after the host's
custom-setting rows and edits that carrier through a fixed overlay modal. The
modal owns Generation Process, grounding budget, Identity Edit LoRA variant,
Identity Edit subject-attention timing, depth-mask feather and
additional-LoRA timing. Legacy flat settings and the
former independent `two_phase_mode`, phase-2 and `outpaint_mode` fields are
expanded and migrated before validation or inference.

The same UI extension treats WanGP's top reference selector as authoritative:
it synchronizes the hidden ReID/Depth-only method values, exposes the fidelity
selector only for Identity Edit, hides subject-reference preparation in Depth +
Prompt mode, and hides the semantic phrase unless SAM3 is selected. WanGP's
native visibility flag continues to control the dedicated depth-strength slider.

If WanGP does not expose a stable reusable boundary, copy only the smallest necessary Apache-compatible/reference sections and record their origin in `THIRD_PARTY_NOTICES.md`.

## Host inputs already available

WanGP passes model factories several useful generic inputs, including:

- `original_input_ref_images` for unprocessed reference PIL images;
- `input_ref_images` for scene-safe processed references and optional subject
  background isolation;
- `video_prompt_type` and `input_frames` for the native `DV` depth dropdown or
  `VG` unchanged-image dropdown; WanGP retains the uploaded `video_guide` and
  refreshes the queue thumbnail;
- `input_masks` for the native depth-area mask;
- `input_custom` for an optional separately uploaded depth-area mask;
- `custom_settings` for `grounding_px` and LoRA-variant controls;
- a dedicated `custom_settings["depth_control_strength"]` multiplier with a
  0..2 range; WanGP's native `denoising_strength` field cannot be reused because
  its host-wide UI range is fixed to 0..1;
- `custom_settings` for the depth-mask feather;
- normal generation settings and callbacks;
- model LoRA lists and schedules.

The implementation uses these generic inputs and WanGP's public insertion API;
it does not introduce a bespoke top-level Gradio tab or modify host files.

Prompt guidance follows the same model-handler boundary. The handler supplies a
plugin-owned `(title, Markdown)` value in `prompt_infos`; WanGP's shared
`field_help` renderer adds its normal hover modal beside the prompt field. The
plugin does not modify WanGP's prompt UI code. Raw and Turbo share one generic,
editable default template from `models/krea2_identity_prompt.py`. The separate
advanced-settings overlay is plugin-owned and does not replace this native
prompt-help modal.

Registered Outpaint is an optional advanced generation task. WanGP resolves its
native spatial margins and target aspect ratio; the plugin gives the unpadded
source destination-relative rotary coordinates and computes isolated per-block
K/V once at flow time zero. Only target/text tokens remain in the denoising
loop. After decode, protected source pixels are restored with an inward feather.
Identity + Outpaint reverses the two functional-adapter weights between passes;
outpaint-only skips identity and Qwen image grounding.

## Major technical risk

WanGP's ordinary Krea 2 text conditioner is language-only. Identity Edit
requires the Qwen3-VL visual encoder and image processor as well. Both A/B
stacks therefore construct a multimodal conditioner and register its language
and vision modules with MMGP; only the selected stack is loaded into the active
pipeline.

Reference latent tokens also increase transformer sequence length. Two references can materially increase attention time and memory, so 1–2 MP limits and offload testing are release requirements.
