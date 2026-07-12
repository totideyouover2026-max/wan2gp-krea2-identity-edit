# GPU acceptance record

The current experimental preview exposes the model definitions so testers can
exercise the implementation. A stable release still requires all rows below
to pass in a clean, explicitly designated WanGP installation. Do not use a
personal or production WanGP instance for these tests.

## Compatibility target

- Minimum: WanGP v12.3.
- Public API audit: commit `5582327dc25e45fec6cda0f27144d4dcf7ed104b`.
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
- Identity Edit LoRA variant (full/r128/r64);
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
| Turbo, one reference, r128, 10 steps | Pending | Correct grounded instruction edit |
| Turbo, one reference, full, 12 steps | Pending | Full LoRA loads at strength 1.0 |
| Raw, one reference, 20 steps, effective CFG 3 | Pending | Grounded empty negative pass; removal succeeds |
| Turbo, scene then subject, two references | Pending | Scene remains frame 1; subject remains frame 2 |
| Aspect-ratio mismatch request | Pending | Output follows primary source aspect ratio |
| Output request above 2 MP | Pending | Output is capped at or below 2 MP |
| Cancellation during Qwen/denoising | Pending | Prompt interruption and no retained source tensors |
| Switch Raw to Turbo and away | Pending | MMGP releases/reuses components without leaked state |
| Low-VRAM profile | Pending | Successful documented run with measured peak VRAM |

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
5. change the development version and tag `v0.1.0`.
