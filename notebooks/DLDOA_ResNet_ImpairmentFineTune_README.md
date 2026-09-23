# ResNet Impairment-Aware Fine-Tuning — Freeze-Ratio Sweep

`DLDOA_ResNet_ImpairmentFineTune.ipynb` adapts Meneses-Albalá et al.'s U-Net fine-tuning idea
(freeze part of the network, fine-tune the rest, on phase-impaired data) to this project's
**ResNet** (Teacher, 64 blocks; Student, also 64 blocks, pruned width r=8).

## Why not "freeze the encoder" like the paper

The paper's U-Net has a genuine encoder (5 downsampling blocks) and decoder (5 upsampling
blocks) — architecturally different parts. This project's ResNet
(`DL_DOA/src/tvt_models.py`) does not: one upsampling `Conv2DTranspose` (64×64→128×128), then
**64 structurally identical residual blocks at a fixed 128×128 resolution** (no downsampling
anywhere), then one final `Conv2DTranspose` (128×128→256×256). There is no encoder/decoder
split to freeze.

So this notebook uses the more general, standard deep-CNN transfer-learning analogue: **freeze
the first K of 64 blocks (+ the input stem), fine-tune the last (64−K) blocks (+ the output
layer)**. Same underlying idea (protect generic early features, adapt the later/output-facing
ones), reframed for an architecture without a literal decoder.

## What it runs

**Sweep:** freeze ratio K/64 ∈ {0%, 20%, 50%, 80%, 90%} (the paper's own 5 points) × 2 base
models (Teacher, Student) = **10 fine-tuning runs**, identical code path for both.

Each run: 200 epochs (paper's own hyperparameters — Adam, lr 1e-4, batch 32, reduce LR 25%
after 20 epochs without improvement down to a 1e-6 floor, early stop after 80 stagnant epochs),
training on data with **fresh per-antenna phase error every sample** (reusing
`beamforming_vector_generation_P/Q`'s existing `error_deg` parameter — no new physics code
needed). `delta_max = 5°`, the one condition where the paper found a real, measurable effect
(mild impairment: the base model is already robust, `[ASSUMPTION]`, changeable in the config
cell).

After every run, the fine-tuned model (best checkpoint) is compared against the **unmodified
base model** on:
- the same impaired condition (does fine-tuning help under impairment?)
- a **clean** condition (does freezing actually prevent forgetting the un-impaired case?)

Progress is printed every epoch (training loss) and every 20 epochs (a small live accuracy
check), so you can watch each run without waiting for it to finish.

## Verified before handover (CPU, this machine — no GPU here)

The generated `.ipynb` was executed end-to-end via `nbclient` in smoke mode (tiny epoch/sample
counts): all 10 fine-tune runs + base/fine-tuned comparisons completed with no errors before
being handed off. **No real-scale run has happened yet on any GPU** — run this notebook the
same way as `IABR_Net_TestSuite`: `SMOKE_TEST = True` first (few minutes), confirm
`outputs_resnet_ft_smoke/RESULTS.md` shows all runs `done`, then `SMOKE_TEST = False` for the
real sweep. Results land in `outputs_resnet_ft/RESULTS.md`, with per-run checkpoints (resumable
on interruption) and loss-curve/sweep figures.

## Assumptions made (paper doesn't specify these)

- `delta_max = 5°` only (not a sweep over {1°,2°,5°}) — keeps total runs to 10, not 30.
- `steps_per_epoch = 40` (paper gives batch size but not steps/epoch) — fine-tuning from an
  already-converged model needs far fewer updates than training from scratch.
- LR-plateau and early-stopping use **training loss**, not a separate validation loss — a real
  validation pass needs a full blob-detection evaluation, too slow to run every epoch.
