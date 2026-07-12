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
- [x] Identify a redistributable/downloadable full Qwen3-VL 4B checkpoint containing the visual tower.
- [x] Add required visual assets to `query_model_files`; reuse WanGP tokenizer assets and construct the image processor from the published config.
- [x] Implement MMGP/offload-aware visual and language model loading.
- [x] Make WanGP's language quantization path and the BF16 visual-prefix path explicit.
- [x] Implement Identity Edit LoRA selection: v1.1 full, r128 and r64.
- [x] Add and unit-test LoRA key conversion for WanGP's Krea 2 transformer names; real loading remains an acceptance item.

## Phase 3 — dual conditioning

- [x] Implement image resize/capping for `grounding_px`.
- [x] Implement one- and two-image Qwen3-VL grounded prompt encoding.
- [x] Preserve the 12 selected Krea 2 text-encoder hidden layers.
- [x] VAE-encode clean source/reference latents.
- [x] Pack source and target latent patches consistently.
- [x] Build `[text | source(s) | target]` sequence positions.
- [x] Assign reference RoPE frames 1..N and target frame 0.
- [x] Slice transformer output back to target tokens only.
- [x] Ground the unconditional/negative pass when CFG is active.

## Phase 4 — WanGP integration

- [x] Read `original_input_ref_images` in `model_factory.generate`.
- [x] Read and validate `custom_settings["grounding_px"]`.
- [x] Expose one/two reference selection through standard WanGP controls.
- [x] Match output aspect ratio to the first reference by default.
- [x] Add <=2 MP capping and two-reference warnings.
- [x] Map documented CFG values to WanGP's internal guidance convention.
- [x] Preserve interruption, callbacks, previews and LoRA schedules in the reused pipeline.
- [x] Return standard WanGP image tensors.

## Phase 5 — profiles and tests

- [ ] Make model definitions visible only after inference tests pass.
- [ ] Add Turbo 8, 10 and 12 step profiles.
- [ ] Add Raw removal/CFG profile.
- [ ] Add single-reference golden test fixtures.
- [ ] Add two-reference scene/subject ordering fixture.
- [ ] Test aspect-ratio mismatch warning.
- [ ] Test all three LoRA variants.
- [ ] Test model cancellation and model switching.
- [ ] Measure peak RAM/VRAM and runtime at 1 MP, 1.5 MP and 2 MP.

## Phase 6 — release

- [x] Select a license for original plugin code.
- [x] Complete third-party notices and preserve adapted-code attribution.
- [x] Document Krea 2 license, acceptable-use and moderation requirements.
- [ ] Add screenshots produced from permitted test images.
- [ ] Validate clean install from a GitHub URL.
- [ ] Tag `v0.1.0` only after end-to-end GPU validation.

## Definition of done

The plugin is done when a user with a compatible WanGP installation can install the repository URL, enable it, restart, select a Krea 2 Identity Edit model, provide one or two references, generate a correctly conditioned image, and switch away without leaked model state or required core patches.
