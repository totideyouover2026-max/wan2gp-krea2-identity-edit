# Implementation plan

## Phase 1 — establish the plugin contract

- [x] Create standalone repository layout.
- [x] Add model-plugin manifest.
- [x] Register collision-free architecture names.
- [x] Add hidden Raw and Turbo model definitions.
- [x] Add standard-library structural validation.
- [x] Confirm the minimum supported WanGP version/commit.
- [x] Add a compatibility guard with an actionable error message.

## Phase 2 — model and asset loading

- [x] Reuse host Krea 2 transformer and Qwen image VAE loaders.
- [x] Reuse WanGP's complete BF16 Qwen3-VL 4B checkpoint containing both language and visual weights as the default.
- [x] Reuse WanGP tokenizer/preprocessor assets and construct the image processor from the published config.
- [x] Implement MMGP/offload-aware full BF16 language and visual model loading.
- [x] Add a temporary load-time encoder-stack A/B: unified BF16 by default, or
  the former Quanto-language plus Comfy scaled-FP8-vision stack.
- [x] Implement Identity Edit v1.2 LoRA selection: full (default), r128 and r64.
- [x] Add and unit-test LoRA key conversion for WanGP's Krea 2 transformer names; real loading remains an acceptance item.

## Phase 3 — dual conditioning

- [x] Implement image resize/capping for `grounding_px`.
- [x] Implement one- and two-image Qwen3-VL grounded prompt encoding.
- [x] Preserve the 12 selected Krea 2 text-encoder hidden layers.
- [x] VAE-encode clean source/reference latents.
- [x] Preserve secondary-reference aspect ratio and centre its aligned latent grid.
- [x] Add a temporary FIT-versus-legacy-stretch selector for Picture 2 geometry
  regression testing without changing the default FIT path.
- [x] Pack source and target latent patches consistently.
- [x] Build `[text | source(s) | target]` sequence positions.
- [x] Assign reference RoPE frames 1..N and target frame 0.
- [x] Slice transformer output back to target tokens only.
- [x] Ground the unconditional/negative pass when CFG is active.
- [x] Adapt upstream reference-attention fidelity boosts as opt-in subject and
  scene profiles while preserving the standard 1x/1x path.

## Phase 4 — WanGP integration

- [x] Read `original_input_ref_images` in `model_factory.generate`.
- [x] Read and validate `custom_settings["grounding_px"]`.
- [x] Expose one/two reference selection through standard WanGP controls.
- [x] Match output aspect ratio to the first reference by default.
- [x] Add <=2 MP capping and two-reference warnings.
- [x] Map documented CFG values to WanGP's internal guidance convention.
- [x] Preserve interruption, callbacks, previews and LoRA schedules in the reused pipeline.
- [x] Return standard WanGP image tensors.
- [x] Add a generic default prompt and native WanGP prompt-guidelines modal.
- [x] Add an adaptive four-control main layout plus a Yes/No Advanced Settings
  launcher backed by a plugin-owned modal and a hidden canonical JSON carrier.
- [x] Migrate legacy flat advanced settings before validation and inference.
- [x] Add optional Identity Edit subject-attention timing with validated
  early/middle/final denoising-third boosts and constant compatibility default.
- [x] Remove the temporary single-reference KI/native-I role selector and
  migrate old values back to the stable scene-first KI contract.

## Phase 5 — profiles and tests

- [x] Add an experimental target-only depth-control projection.
- [x] Convert and stack the public rank-64 depth block adapter through MMGP.
- [x] Reuse WanGP's native Depth Anything V2 preprocessing and queue preview for
  a separate control image; consume its processed `input_frames` tensor.
- [x] Expose WanGP's native control-image dropdown/upload and a depth-strength
  setting, with No Control Image selected by default.
- [x] Reuse WanGP's mask editor for whole-frame, inside-mask and outside-mask
  depth control.
- [x] Expose WanGP's generic custom-guide upload as an optional external
  grayscale/alpha depth mask, with separate White/Black-area modes that retain
  full-frame host depth preprocessing.
- [x] Show the uploaded custom mask as an image and add an on-demand,
  bounded-resolution effective-depth preview that runs WanGP Depth Anything
  with its single-frame video contract before masking.
- [x] Neutralize excluded control pixels and apply a feathered target-token
  gate to the depth projection.
- [x] Protect reference 1 as the main scene and expose optional SAM3 semantic
  isolation plus rembg fallback for reference 2/subject.
- [x] Add depth-first per-step timing for user-added LoRAs while preserving
  full-pass Identity Edit and depth conditioning.
- [x] Expose validated early/middle/final depth-first ramp multipliers in the
  plugin-owned Advanced Settings modal.
- [x] Add a mutually exclusive Turbo ReID rank-32 method with one reference.
- [x] Add ReID's shared aspect-preserving 384²-area Qwen/clean-VAE reference,
  frame-1 positions, released isolated timestep-zero K/V cache, fixed 8-step
  sampling and optional target-only depth control.
- [x] Retain the experimental joint timestep-zero stream behind an Advanced
  Settings selector for same-seed reference-injection A/B tests.
- [x] Add sampled isolated-cache fingerprints and RMS diagnostics for adapter
  strength A/B tests.
- [x] Add a diagnostic 0..2 ReID LoRA multiplier so a same-seed 0-versus-1
  run can isolate adapter contribution without changing reference conditioning.
- [x] Add the ReID release's optional YuNet face/head crop with separately
  downloaded detector weights and a full-reference fallback.
- [x] Add reference-free Depth + Prompt generation with only the depth adapter
  and text conditioning.
- [x] Add an opt-in Turbo/ReID two-phase experiment: Depth + Prompt first, then
  a low-denoise source-image ReID refinement with independently scheduled user
  LoRAs and optional phase-2 depth.
- [x] Add an opt-in raw Control Image to ReID route using WanGP's native `VG`
  preprocessing and low-denoise source-image restart.
- [ ] Run Direct Image to ReID GPU comparisons at denoising strengths
  0.15/0.25/0.35 and confirm the uploaded source is not depth-preprocessed.
- [ ] Run two-phase GPU comparisons across phase-2 denoise 0.15/0.25/0.35,
  phase-2 depth on/off, and Unlocker strengths 0.01/0.5/1.0.
- [ ] Run ReID identity, pose-angle and ReID + depth GPU golden tests.
- [ ] Run masked-depth and isolated-subject GPU golden tests.
- [ ] Run combined Identity Edit + depth golden tests on Raw and Turbo.
- [ ] Run a same-seed simultaneous versus Depth layout -> Identity refinement
  comparison; record the selected built-in ramps, per-phase projection scale,
  identity retention and structural artifacts.
- [ ] Run Identity Edit fidelity comparisons at subject 1x/2x/4x/8x and the
  two-reference subject 4x + scene 2x profile on Raw and Turbo.
- [ ] Run same-seed constant 8x versus ramped 1x/2x/8x Identity Edit head-swap
  comparisons, checking composition, pose, lighting and final likeness.
- [ ] Measure WanGP preprocessing, the depth adapter and combined sequence VRAM overhead.

- [ ] Make model definitions visible only after inference tests pass.
- [ ] Add optional Turbo 10 and 12 step profiles; keep the native 8-step default.
- [ ] Add Raw removal/CFG profile.
- [ ] Add single-reference golden test fixtures.
- [ ] Add two-reference scene/subject ordering fixture.
- [ ] Test aspect-ratio mismatch warning.
- [ ] Test all four LoRA variants.
- [ ] Test model cancellation and model switching.
- [ ] Measure peak RAM/VRAM and runtime at 1 MP, 1.5 MP and 2 MP.

## Phase 6 — release

- [x] Add registered spatial outpaint geometry and protected-source compositing.
- [x] Add isolated registered-reference K/V conditioning.
- [x] Add outpaint-only and separately scheduled Identity + Outpaint tasks.
- [ ] Run Turbo/Raw Registered Outpaint GPU golden tests and record peak VRAM.
- [ ] Validate one-pass edge placements before adding interior two-pass placement.

- [x] Select a license for original plugin code.
- [x] Complete third-party notices and preserve adapted-code attribution.
- [x] Document Krea 2 license, acceptable-use and moderation requirements.
- [ ] Add screenshots produced from permitted test images.
- [ ] Validate clean install from a GitHub URL.
- [ ] Promote `2.0.0-beta.1` to `v2.0.0` only after end-to-end GPU validation.

## Definition of done

The plugin is done when a user with a compatible WanGP installation can install the repository URL, enable it, restart, select a Krea 2 Identity Edit model, provide one or two references, generate a correctly conditioned image, and switch away without leaked model state or required core patches.
