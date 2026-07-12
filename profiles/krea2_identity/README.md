# Krea 2 Identity Edit profiles

Release-facing preset JSON files are intentionally absent. Add them only after
the corresponding rows in `GPU_ACCEPTANCE.md` pass:

- Turbo 8 steps, effective CFG 1;
- Turbo 10 steps, effective CFG 1;
- Turbo 12 steps, effective CFG 1;
- Raw 20 steps, effective CFG 3 with grounded empty negative conditioning.

Each profile must state its LoRA variant and preserve scene-first,
subject-second ordering.
