# GPU acceptance record

The `2.0.0-beta.1` public beta exposes the model definitions so testers can
exercise the implementation. A stable `2.0.0` release still requires all rows
below to pass in a clean, explicitly designated WanGP installation. Do not use
a personal or production WanGP instance for these tests.

## Compatibility target

- Minimum: WanGP v12.34.
- Public API audit: commit `6b92c54f92bde24d6d309d6f61249353b0ec783d`.
- Plugin install method: clean GitHub URL through WanGP's plugin manager.

Before downloading weights, run the static contract validator and the import
smoke test documented in `README.md` using that clean installation's Python
environment.

## Required environment record

For every inference run record:

- WanGP revision;
- GPU and driver;
- WanGP memory profile and attention mode;
- output resolution and reference count;
- model variant (Raw/Turbo);
- Identity Edit LoRA variant (v1.2 full/r128/r64);
- Qwen3-VL encoder stack (unified full BF16 by default, or the temporary legacy
  Quanto-language/scaled-FP8-vision regression option);
- identity method and, for Identity Edit, the subject/scene reference-fidelity
  profile; for ReID, rank-32 adapter/reference crop details;
- for Direct Image to ReID, record the raw Control Image hash, denoising
  strength, processed source dimensions and effective ReID step count;
- for Depth + Prompt, confirm zero references and that only the depth adapter
  was loaded;
- depth dropdown/area selection, control-image and mask fixture/hash, mask
  expand/feather and depth strength;
- subject-reference background-removal selection;
- step count and effective CFG;
- peak VRAM and wall time;
- observed result and output path/hash.

Do not commit user images, model weights, generated outputs, caches or the test
environment. Golden fixtures must be small, redistributable source images or
their approved hashes/URLs.

## Acceptance matrix

| Test | Status | Required observation |
| --- | --- | --- |
| Turbo, one reference, r64, 8 steps | Pending | Identity and unedited content preserved |
| Turbo, one reference, r128, native-default 8 steps | Pending | Correct grounded instruction edit |
| Turbo, one reference, v1.2 full, 12 steps | Pending | v1.2 full LoRA loads at strength 1.0 |
| Raw, one reference, 20 steps, effective CFG 3 | Pending | Grounded empty negative pass; removal succeeds |
| Turbo, scene then subject, two references | Pending | Scene remains frame 1; subject remains frame 2 |
| Turbo, landscape scene plus portrait subject | Pending | Subject aspect is preserved in a centred FIT latent grid |
| Turbo, one reference, fidelity 1x vs 2x vs 4x vs 8x | Pending | Increasing subject boost changes identity/reference adherence without stale state |
| Turbo, scene then subject, subject 4x + scene 2x | Pending | Subject and scene receive their independently logged multipliers in reference order |
| Raw, one reference, fidelity 1x vs 4x | Pending | CFG positive/negative streams use the same reference bias and preserve output validity |
| Aspect-ratio mismatch request | Pending | Output follows primary source aspect ratio |
| Output request above 2 MP | Pending | Output is capped at or below 2 MP |
| Cancellation during Qwen/denoising | Pending | Prompt interruption and no retained source tensors |
| Switch Raw to Turbo and away | Pending | MMGP releases/reuses components without leaked state |
| Low-VRAM profile | Pending | Successful documented run with measured peak VRAM |
| Turbo, one reference, depth strength 1.0 | Pending | V2 depth thumbnail appears; identity and target depth structure both hold |
| Turbo, Identity Edit + depth, simultaneous vs Depth layout -> Identity refinement | Pending | Same seed; scheduled run logs `layout-to-identity adapter timing active`, Depth projection scale falls by phase, and compare pose retention/identity against simultaneous |
| Turbo, two references plus depth control image | Pending | References remain frames 1/2 and only target projection receives depth |
| Raw, one reference, depth strength 1.0 | Pending | CFG pass shares identical depth control |
| Depth cancellation and model switch | Pending | Estimator/control tensors and adapters release cleanly |
| Turbo, masked person depth with background railing | Pending | Person pose holds; railing outside mask is not transferred |
| Turbo, outside-mask depth | Pending | Painted subject is free while surrounding depth structure holds |
| Depth mask feather 0 vs 16 vs 64 | Pending | Increasing feather softens the control boundary without changing mask polarity |
| Turbo, scene plus isolated subject reference | Pending | Scene remains intact; only reference 2 has its background removed |
| Turbo ReID, one close identity reference, 8 steps | Pending | ReID adapter loads alone; identity and requested output aspect hold |
| Turbo ReID, full identity reference, 8 steps | Pending | 384² Qwen and clean-VAE budgets plus isolated K/V cache complete without retained state |
| Turbo ReID LoRA strength 0 vs 1, same seed | Pending | Reference/cache fingerprints match while output changes only if the registered adapter contributes |
| Turbo ReID plus depth control | Pending | Face identity and target pose both respond; depth affects target only |
| ReID cancellation and switch away | Pending | Cached reference K/V and depth state are cleared |
| Turbo ReID plus Direct Image, denoise 0.15 | Pending | Source composition is closely preserved and no depth adapter loads |
| Turbo ReID plus Direct Image, denoise 0.25 | Pending | Identity changes while source pose, framing and background remain recognizable |
| Turbo ReID plus Direct Image, denoise 0.35 | Pending | Stronger edit remains source-guided without stale control state |
| Direct Image cancellation and switch away | Pending | Source tensor and ReID cache are released cleanly |
| Turbo, Depth + Prompt only, whole frame | Pending | Zero references; only depth adapter loads; prompt and pose both respond |
| Raw, Depth + Prompt only, CFG | Pending | Text-only positive/negative passes share the same depth control |
| Depth + Prompt masked control and cancellation | Pending | Mask gates depth and all temporary depth/text-only state is restored |

## Direct Image implementation record

- WanGP revision: v12.34 / audited commit
  `6b92c54f92bde24d6d309d6f61249353b0ec783d`.
- GPU and driver: pending tester run.
- Memory profile and attention mode: pending tester run.
- Resolution: begin with `1280x720`; pending tester run.
- Model variant: Krea 2 Identity Turbo.
- Functional adapter: ReID rank 32; Identity Edit LoRA variant not applicable.
- Text encoder: full BF16 Qwen3-VL.
- Peak VRAM and wall time: pending tester run.
- Observed result: static handler/runtime contracts and 77 unit tests pass; GPU
  image behavior is pending. The terminal must show
  `[Krea2 Identity][Direct Image] ReID refinement` and must not report loading
  `depth-control-lora.safetensors`.

## Reference-fidelity implementation record

- WanGP revision: v12.34 / audited commit
  `6b92c54f92bde24d6d309d6f61249353b0ec783d`.
- GPU and driver: pending tester run.
- Memory profile and attention mode: pending tester run.
- Initial resolution: `1280x720`; pending tester run.
- Model variants: Krea 2 Identity Turbo first, then Raw CFG validation.
- Identity Edit LoRA variant: v1.2 full first; v1.2 rank variants pending.
- Text encoder: full BF16 Qwen3-VL.
- Reference profiles: standard 1x/1x, subject 2x, subject 4x, subject 8x,
  and two-reference subject 4x + scene 2x.
- Peak VRAM and wall time: pending tester run. The additive attention bias is
  dense, so compare memory against the 1x/1x path and begin at 1280x720.
- Observed result: scaffold validation and 92 unit tests pass; GPU image
  behavior is pending. A boosted run must log
  `[Krea2 Identity][Identity Edit] reference attention boost active` with the
  selected subject and scene multipliers. NAG scale must remain 1.0.

## ReID reference-budget diagnostic record

- WanGP revision: v12.34 / audited commit
  `6b92c54f92bde24d6d309d6f61249353b0ec783d`.
- GPU: 12 GB VRAM; exact model and driver pending tester record.
- Memory profile and attention mode: pending tester record.
- Resolution and model: `1280x720`, Krea 2 Identity Turbo.
- Functional adapter: ReID rank 32; unified BF16 Qwen3-VL encoder.
- Peak VRAM: not captured. The 1024² diagnostic clean-reference cache used
  more memory and time than the published 384² path.
- Pre-correction observation: with the same seed and prompt, face-crop on and
  off produced 572/576 cached identity tokens and converged on a similar
  generic identity. The local runtime had incorrectly reused Qwen's 384²
  budget for the clean VAE reference.
- Diagnostic result: the aligned `1296x704` reference logged 3,564 clean VAE
  tokens instead of 576, but still converged on essentially the same generic
  identity. Reference-token resolution is therefore not the primary fault.
- Restored contract: both Qwen and clean VAE reference preparation use the
  author's tested 384² budgets. The next diagnostic is a same-seed ReID LoRA
  strength A/B at 0.0 versus 1.0; all reference encoding and cache inputs must
  remain unchanged between those runs.

## Resolution sweep

Run at approximately 1 MP, 1.5 MP and 2 MP. The two-reference run should target
1-1.5 MP first. Record failures as failures; do not enable visibility based only
on a lower-resolution smoke test.

## Release gate

Only after the matrix passes:

1. add validated profile JSON files beneath `profiles/krea2_identity/`;
2. add permitted screenshots;
3. confirm both model definitions remain `visible: true`;
4. validate a fresh GitHub-URL installation;
5. change the beta version to `2.0.0` and tag `v2.0.0`.
