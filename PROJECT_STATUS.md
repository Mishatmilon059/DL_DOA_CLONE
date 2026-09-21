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

### Notebook 4 — nested/scattering-path robustness test — ✅ RUN COMPLETE (2026-09-22)
`DLDOA_Compression_04_NestedPathRobustness.ipynb`. Supervisor's idea: fix 3 principal paths per scene (**200 independent scenes**, not one -- a single fixed scene would let the model memorize positions and gives no statistical spread), sweep a 4th "nuisance" (interfering) path's power (−20/−10/0 dB relative to the 3 principal paths' total power) within each scene while holding the 3 principal paths + noise constant, at 2 SNR levels (0dB hard, 15dB moderate). CPU-only, no training, 200 scenes × 2 SNR × 3 power × 2 models finished in **17.3 minutes**.

Ran with the **r8-magnitude** student (recovered from a committed Kaggle Version's Output Data panel, avoiding a costly re-run — the snr_aware student's weights are still not recovered).

**Full results (Pd_p = recovery of the 3 known/principal paths; Pd_n = detection of the new nuisance path itself):**

| SNR | Nuisance | Teacher Pd_p | Student Pd_p | ΔPd_p | Teacher Pd_n | Student Pd_n | ΔPd_n |
|---|---|---|---|---|---|---|---|
| 0 dB | −20 dB | 0.5909 | 0.5765 | −0.0145 | 0.0591 | 0.0550 | −0.0040 |
| 0 dB | −10 dB | 0.6406 | 0.6233 | −0.0173 | 0.4777 | 0.4467 | −0.0309 |
| 0 dB |   0 dB | 0.6065 | **0.6174** | **+0.0109** | 0.8389 | 0.7955 | −0.0434 |
| 15 dB | −20 dB | 0.8939 | 0.8873 | −0.0066 | 0.6117 | 0.6000 | −0.0117 |
| 15 dB | −10 dB | 0.8953 | 0.8824 | −0.0129 | 0.8796 | 0.8778 | −0.0018 |
| 15 dB |   0 dB | 0.8754 | 0.8595 | −0.0159 | 0.8918 | 0.8639 | −0.0279 |

**Gap trend as nuisance power increases (weakest −20dB → strongest 0dB):**
```
SNR=0dB:  gap goes from -0.0145 to +0.0109  ->  NARROWS by +0.0254 (student overtakes teacher at the hardest point)
SNR=15dB: gap goes from -0.0066 to -0.0159  ->  ~flat, by -0.0093 (no strong trend)
```

**Discussion.**
- Pd_n rises steeply with nuisance power at both SNRs (e.g. 0.06→0.48→0.84 at SNR=0dB) — expected physics, a weaker interferer is intrinsically harder to detect for *either* model; this is not a finding about compression, just a sanity check that the protocol behaves correctly.
- Pd_p (the actual question) stays relatively stable across nuisance levels for both models — adding an interferer does not catastrophically break principal-path recovery for either the teacher or the student.
- **The standout result: at SNR=0dB (the harder, noisier condition), the compressed student's principal-path Pd *exceeds* the teacher's once the nuisance path is as strong as the principal paths (0dB) — the only outperforming data point across the whole compression project so far.** At SNR=15dB the student stays a bit behind the teacher throughout, with only a mild (not statistically striking) widening trend.
- Working hypothesis for *why*: pruning + distillation may push the student toward a flatter/less-overfit solution than the teacher's, which shows up as reduced fragility specifically under stress (strong interference, high noise) — plausible, not proven; a single low-SNR crossover point is suggestive, not conclusive evidence on its own.
- Framing for the report: **not** "the student is uniformly better" — rather, "compression does not make the model *more* fragile to unexpected interference, and in the hardest tested condition (low SNR + strong interferer) it performs at least as well as, arguably slightly better than, the uncompressed teacher." This is a legitimately new finding — neither paper tests this at all.
- Not yet done: repeating this with the snr_aware student (once its weights are recovered/re-saved) to see whether the effect is method-specific or general to compression at r=8.

---

## Architecture screening — alternatives to the ResNet block (🔲 built, not yet run, 2026-09-24)

`DLDOA_Architecture_Screening_4Way.ipynb`. Separate track from the compression project — asks whether a fundamentally different core block (not just a smaller/factorized ResNet) suits this specific 2D-frequency-estimation task better. Deep-research-grounded (web search, not guessed) before building; full citations captured in this session's research turn.

**4 candidates, all wrapped in the identical input/output shell** (`build_model_with_body(body_fn)` — same pattern as `build_pruned_resnet`/`build_lowrank_resnet`, only the core block differs):

| Candidate | Core idea | Why it fits *this* task specifically |
|---|---|---|
| SIREN-body | sin() activations throughout | Our signal literally *is* a sum of sinusoids; SIREN is designed to overcome spectral bias, connecting to the "compression hurts high-SNR precision" finding above |
| FNO-body | Spectral (Fourier-domain) convolution | Input Y is already a frequency-domain representation; FNO gives near-free global receptive field instead of 64 stacked local-receptive-field blocks |
| Window-Attention-body | Local-neighborhood self-attention | Adapted stand-in for message-passing/GNN DOA work (true antenna-graph would need a non-image input, out of scope here) |
| Gridless-Unfold-body | Learned complex soft-threshold refinement | Directly targets our own **Pd ceiling ≈0.972** finding (gridless = no pixel quantization); explicitly fixes PIA-Net's known bug (magnitude-soft-threshold + exact phase preservation, not ReLU-clamped real values) |

**Explicit scope compromise (stated in the notebook itself):** 8 blocks each (not 64 — none of these have pretrained weights to warm-start from, so shallower depth keeps the 4-way comparison fair), ~20 min hard-capped training per architecture, simplified/adapted versions of each literature family rather than literal reproductions. This is a **screening study** — answers "which family shows enough promise to invest in further," not "which is best at paper scale."

**Verified locally (pure numpy) before writing any TF code**, given this project's history of axis/reshape bugs (Conv2 slicing, SVD transpose):
1. FNO spectral-conv round-trip (rfft2 → mode-truncated complex-weight multiply → irfft2) — finite, correctly-shaped, sane-scale output
2. Complex soft-threshold — phase preserved exactly for a purely-imaginary input (the exact case that broke PIA-Net), magnitude correctly reduced by the threshold
3. Even/odd channel-pair interleave-back (stack+reshape) — reproduces original channel order exactly

**Bug caught and fixed before pushing:** the batch-size probe originally ran gradient steps on the *real* model being trained (garbage updates from random noise before real training started) — fixed to probe with a disposable throwaway model, matching the Notebook 2/3 pattern.

---

## Proposed next-stage direction (for a second Q1 paper, not scoped into the 1-week plan)

Replace the heatmap+blob-detection output head with a **differentiable set-prediction** architecture (DETR-style query slots, Hungarian-matching loss) — removes the pixel-quantization step that causes the Pd ceiling, and naturally handles unknown path count L (neither paper does — both assume L is given at evaluation). See `My_Proposed_Architecture.md` for the original 3-stage sketch this connects to.

---

## Gotchas / lessons learned (read before re-running anything)

- **Kaggle `/kaggle/working/` is ephemeral** — always "Save Version → Save & Run All (Commit)" or manually download before the session ends. Lost both Notebook-2 student weight files this way.
- **OOM on tiny tensors after a crash** = leftover memory fragmentation, not insufficient GPU memory — restart the kernel, don't just retry with a smaller batch in the same session.
- **`Conv2D` kernel shape is `(kh, kw, in_ch, out_ch)`** — slicing/reshaping the wrong axis is an easy, silent bug (hit twice: once slicing axis=1 instead of axis=2 for Conv2's input-channel slice in `copy_weights_to_pruned`, once an erroneous `.T` before reshape in the low-rank SVD decomposition). **Always verify numerically** (identity/full-rank reconstruction test) before trusting a new weight-manipulation function.
- **Dataset generation needs no Kaggle upload for training** — `training_data_generator` / `validation_data_generator` run on-the-fly from pure numpy/scipy, no TF needed even locally.
