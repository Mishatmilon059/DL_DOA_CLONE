# DL-DOA Project — Status & Findings Log

Living document. Updated alongside every notebook push — see [`github-always-sync`] policy: nothing goes to Kaggle without also landing here first.

Baseline paper: Lloria et al., *"Deep-Learning-Based AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems,"* IEEE TVT 2026 (`papers/Deep_Learning_Based_AoA_and_AoD_Estimation_in_Analog_Millimeter (1).pdf`). Conference precursor: PIMRC 2024 (`papers/A_ResNet_Approach_for_AoA_and_AoD_Estimation...pdf`).

---

## Baseline reproduction — CONFIRMED VALID

Pretrained `inf_model_007_256_resnet.h5` (469,393 params) re-evaluated with the **original** repo evaluator (`DL_DOA/src/TVT_Blob_Inference.py`), on a frozen 8000-sample bank — matches paper Table II to within 0.4%:

| SNR | Our Pd | Paper Pd |
|---|---|---|
| 0 dB | 0.639 | 0.643 |
| 20 dB | 0.925 | 0.925 |

**⚠️ Metric definition gotcha**: the original evaluator scores each angle *component* (ψ or φ) independently against the 1° threshold and *drops* samples with too-few detected peaks entirely from the denominator. Several earlier exploratory notebooks in this repo use a stricter per-source (Hungarian, both-components-must-hit, no-sample-dropping) metric instead. **Numbers from those notebooks are NOT directly comparable to paper Table II or to the compression-project notebooks** — only `DLDOA_Compression_0X_*.ipynb` use the original, paper-faithful metric.

---

## Key findings (in the order discovered)

1. **Pd ceiling ≈ 0.972** for the 256×256 heatmap+blob-detection representation, even with a *perfect* model (ground-truth heatmap fed straight to the blob detector). Paper's ResNet already sits at 96.5% of this ceiling — architecture tweaks alone have very little headroom left. Root cause: a Jacobian singularity (`d(π cos ψ)/dψ = -π sin ψ`) near end-fire angles (ψ≈0°/180°) amplifies fixed pixel-quantization error into large angle error. See `Ceiling_And_Diffusion_GoNoGo.ipynb`.

2. **Diffusion-based sharpening: ruled out.** Hypothesis was "MSE loss → posterior-mean blur → low-SNR misses." Tested directly: artificially sharpening (`p⁴`) the model's raw output made Pd *worse*, not better, and an oracle peak-finder (no blob-shape constraint) didn't beat the existing blob detector either. Conclusion: misses are from peaks landing in the *wrong location* (the Jacobian effect above), not blur — diffusion would not help. See `Ceiling_And_Diffusion_GoNoGo.ipynb`.

3. **Classical baselines validated**: 2D-DFT+SIC on `ifft2(Y)` reproduces paper's DFT-CEA numbers closely (e.g. 20dB: our Pd=0.884 vs paper 0.891). Confirms the whole physics/generation pipeline is correct. See `Classical_Baseline_SanityTest.ipynb`.

4. **Two-stage channel-estimation approach** (Y → small CNN → Ĥ → classical angle extraction) was built and is architecturally sound (H has the *same* 2D-sinusoid structure as `ifft2(Y)`, verified to <0.05° reconstruction error), but lands at roughly the same ceiling as the direct approach — not pursued further as the headline compression story. See `ChannelEst_LowParam_TwoStage.ipynb`.

5. **PIA-Net has an unresolved physics-consistency issue**: the learned ISTA state is real/ReLU-constrained but a physical path has arbitrary complex phase — flagged, not yet fixed. See `BUJHO_SHOHOJ_VABE.md`, `DLDOA_PIANet_Standalone_v4.ipynb`.

---

## Active work — structured compression (1-week plan, `NOVELTY_RESEARCH_AND_ONE_WEEK_PLAN.md`)

Three-notebook pipeline, each hard-capped at **2.5h wall-clock** (Kaggle T4). Frozen banks (`eval_bank.npz` 8000 samples / `calibration_bank.npz` 600 samples) generated once locally (`scripts/generate_frozen_banks.py`) using the **original** `validation_data_generator`, uploaded to Kaggle as a dataset — never regenerated on Kaggle, so every run compares apples-to-apples.

| Notebook | Status | What it does |
|---|---|---|
| `DLDOA_Compression_01_Foundation.ipynb` | ✅ All checks passed | Frozen-bank teacher spot-check, weight-transfer equivalence test (r=12 identity, matches to 2.4e-07), per-step timing |
| `DLDOA_Compression_02_PruneAndFinetune.ipynb` | ✅ 2 runs complete — **this is now the headline result** | Channel pruning (12→8 width, 33% fewer params). Two selection methods compared: **magnitude-based** vs **SNR-stratified Taylor-importance** (`snr.ipynb` = the executed snr_aware run) |
| `DLDOA_Compression_03_LowRank.ipynb` | ❌ **Dropped from the plan** (2026-09-21) | Notebook exists and its math was verified locally (SVD reconstruction, see Gotchas), but the low-rank axis was dropped to keep scope to a single, well-validated compression story within the time budget. Kept in the repo for reference / a possible future follow-up, not part of the current deliverable. |

### Compression results so far (r=8, 33% param reduction, 1200-sample subsampled eval)

| Method | Mean ΔPd (all SNR) | Mean ΔPd (low SNR ≤0dB) |
|---|---|---|
| Magnitude-based pruning | −0.0221 | −0.0127 |
| SNR-aware pruning | **−0.0190** | **−0.0082** |

SNR-aware wins on average and specifically in the low-SNR band it targets, though not uniformly at every SNR point (some individual-point differences are within ~2% sampling noise at n=150/SNR — full 1000/SNR re-eval would firm this up, but student weights from these two runs were **lost** — Kaggle `/kaggle/working/` is ephemeral unless "Save Version → Save & Run All" is clicked; do this on every future run).

**Unexpected finding**: compression hurts *mid-to-high* SNR (10-25dB) more than low SNR — opposite of the initial hypothesis. Likely because low-SNR performance is already noise-limited for both models (little room to get worse), while high-SNR performance is close to the ceiling and needs the full parameter budget for precision.

### Updated plan (low-rank dropped, 2026-09-21)
```
1. Re-run Notebook 2 with SELECTION_METHOD='snr_aware' (the winning method so far)
   -- this time click "Save Version -> Save & Run All (Commit)" at the end,
   so the student weights actually persist (lost twice already, see Gotchas)
2. Build Notebook 4 -- nested/scattering-path robustness test (below), using
   that saved snr_aware student vs the teacher
3. Final report: pruning comparison (magnitude vs snr_aware) + robustness test
```

### Notebook 4 — nested/scattering-path robustness test — 🔲 built, not yet run
`DLDOA_Compression_04_NestedPathRobustness.ipynb`. Supervisor's idea: fix 3 principal paths per scene (**200 independent scenes**, not one -- a single fixed scene would let the model memorize positions and gives no statistical spread), sweep a 4th nuisance path's power (−20/−10/0dB) within each scene while holding the 3 principal paths + noise constant, at 2 SNR levels (0dB, 15dB). Checks whether the r8-magnitude student's recovery of the 3 principal paths degrades faster than the teacher's as interference increases (separately from whether the nuisance path itself gets detected).

**Evaluation only — no training, runs in minutes.** Uses the teacher directly + the r8-magnitude student's weights, which were recovered from a committed Kaggle Version's Output Data panel (a "Save Version" from an earlier run had actually persisted, discovered after the session initially thought both students' weights were lost) rather than needing a costly re-run. The snr_aware student's weights are still not recovered/saved -- this test currently only covers the magnitude-pruned student.

---

## Proposed next-stage direction (for a second Q1 paper, not scoped into the 1-week plan)

Replace the heatmap+blob-detection output head with a **differentiable set-prediction** architecture (DETR-style query slots, Hungarian-matching loss) — removes the pixel-quantization step that causes the Pd ceiling, and naturally handles unknown path count L (neither paper does — both assume L is given at evaluation). See `My_Proposed_Architecture.md` for the original 3-stage sketch this connects to.

---

## Gotchas / lessons learned (read before re-running anything)

- **Kaggle `/kaggle/working/` is ephemeral** — always "Save Version → Save & Run All (Commit)" or manually download before the session ends. Lost both Notebook-2 student weight files this way.
- **OOM on tiny tensors after a crash** = leftover memory fragmentation, not insufficient GPU memory — restart the kernel, don't just retry with a smaller batch in the same session.
- **`Conv2D` kernel shape is `(kh, kw, in_ch, out_ch)`** — slicing/reshaping the wrong axis is an easy, silent bug (hit twice: once slicing axis=1 instead of axis=2 for Conv2's input-channel slice in `copy_weights_to_pruned`, once an erroneous `.T` before reshape in the low-rank SVD decomposition). **Always verify numerically** (identity/full-rank reconstruction test) before trusting a new weight-manipulation function.
- **Dataset generation needs no Kaggle upload for training** — `training_data_generator` / `validation_data_generator` run on-the-fly from pure numpy/scipy, no TF needed even locally.
