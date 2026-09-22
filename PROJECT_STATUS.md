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

**Explicit scope compromise (stated in the notebook itself):** 8 blocks each (not 64 — none of these have pretrained weights to warm-start from, so shallower depth keeps the 4-way comparison fair), ~32.5 min hard-capped training per architecture (derived, see time-budget fix below), simplified/adapted versions of each literature family rather than literal reproductions. This is a **screening study** — answers "which family shows enough promise to invest in further," not "which is best at paper scale."

**Verified locally (pure numpy) before writing any TF code**, given this project's history of axis/reshape bugs (Conv2 slicing, SVD transpose):
1. FNO spectral-conv round-trip (rfft2 → mode-truncated complex-weight multiply → irfft2) — finite, correctly-shaped, sane-scale output
2. Complex soft-threshold — phase preserved exactly for a purely-imaginary input (the exact case that broke PIA-Net), magnitude correctly reduced by the threshold
3. Even/odd channel-pair interleave-back (stack+reshape) — reproduces original channel order exactly

**Bug caught and fixed before pushing:** the batch-size probe originally ran gradient steps on the *real* model being trained (garbage updates from random noise before real training started) — fixed to probe with a disposable throwaway model, matching the Notebook 2/3 pattern.

**Time-budget hardening (user directly questioned whether eval fits in 2.5h — correctly, since the first version only guessed):**
- Cell 5: added `EVAL_RESERVE_MINUTES=20` and derived `PER_ARCH_BUDGET_HOURS = (TOTAL_TIME_BUDGET_HOURS - EVAL_RESERVE_MINUTES/60) / 4` (≈32.5 min/arch instead of a flat guessed 20 min/arch), so training and eval reserve are mutually consistent by construction.
- Cell 14: training loop now hard-stops (`elapsed_hours() + PER_ARCH_BUDGET_HOURS > train_ceiling_hours`) before eating into the protected eval reserve.
- **This only protects training, not eval — caught on follow-up.** Cell 15 actually calls `evaluate_on_bank` 5 times (all 4 architectures *plus* the teacher), and that function's cost is dominated by per-sample CPU blob detection + Hungarian matching, not GPU inference — it does not scale the way training does, and the flat 20-minute reserve was never verified against real throughput. Added **Cell 12b**: benchmarks all 5 models for real on a small (16-sample) slice right after they're built, projects the full Cell 15 eval time from measured per-sample cost, and **auto-shrinks `N_PER_SNR_EVAL`** (re-slicing the eval subsample) if the measured throughput would exceed the reserve — so the eval set size is evidence-based, not assumed. Its own benchmarking cost is automatically deducted from the training budget too, since it runs before Cell 14 and `elapsed_hours()` is cumulative from Cell 1.
- Still-open honest caveat (documented in the notebook's Cell 0, not fixed — accepted trade-off): equal **wall-clock** budget per architecture is not equal **gradient-step count** — FNO (FFT/iFFT per block) and Window-Attention (reshape+MHA per block) are more expensive per step than SIREN/Gridless-Unfold (plain convs), so they'll complete fewer epochs in the same ~32.5 min. A promising candidate from this run should get a step-count-matched follow-up before being taken as a real result, not just a paper-scale claim from this screening pass alone.

**Critical Keras 3 bug found and fixed (2026-09-21), before either notebook was ever run:** all 4 architecture bodies used raw `tf.sin` / `tf.stack` / `tf.reshape` calls directly on a Keras Functional-API tensor, and `GridlessUnfold` created a raw `tf.Variable` inside a plain Python function instead of through a Layer. Discovered while building a local full-eval companion notebook and test-running its code with a real TF install (`pip install tensorflow` gives TF 2.21 / Keras 3 today) — Keras 3 (bundled with TF >=2.16, likely what Kaggle's current image also ships) rejects the raw-op pattern outright (`KerasTensor cannot be used as input to a TensorFlow function`), and would have silently left the `GridlessUnfold` threshold parameter untracked/untrained even where it didn't error. Fixed in both notebooks: `SIREN`'s sin activation now goes through a `Lambda` layer; `WindowAttentionBlock` and `GridlessUnfoldBlock` are now proper `Layer` subclasses (matching the `SpectralConv2D` pattern already used for FNO) with the threshold as a real `add_weight`. **Verified two ways, not just by inspection:** (1) a standalone script ran build → forward → backward/gradient step → checkpoint save for all 4 architectures on local CPU with no errors; (2) both full notebooks (`DLDOA_Architecture_Screening_4Way.ipynb` and the new local one, see below) were executed end-to-end via `nbclient` at a tiny smoke-test scale (1 epoch, 2 steps, 2-16 eval samples) with `NOTEBOOK EXECUTED OK, no errors`. A separate pre-existing issue was also found and fixed while validating: most cells' `source` field had been stored as a list of individual *characters* instead of lines (from a prior session's notebook edits) — content was unaffected (still joins to the identical text) but the structure was normalized back to proper per-line lists.

## Full local evaluation (no time cap) — companion notebook (✅ run complete, 2026-09-22)

`DLDOA_Architecture_FullEval_Local.ipynb`. Built after the user asked to run all 4 architectures "fully" on a local GPU machine, no time cap, no errors. Same 4 architecture bodies (Keras-3-fixed, see above) and evaluator as the Kaggle notebook, but:
- **No wall-clock cap** — fixed `EPOCHS_PER_ARCH=200` x `STEPS_PER_EPOCH=100` = 20,000 gradient steps per architecture (from scratch, no warm start), instead of a fixed wall-clock slice. Fairer than the Kaggle version's equal-*time* design (this is equal-*updates*), at the cost of an unknown total run time until it's actually run.
- **Full eval bank** (`N_PER_SNR_EVAL = None` → all 8000 samples), not a subsample.
- **No large dataset files needed** — training data is generated synthetically on the fly by `training_data_generator`; only `DL_DOA/src/*.py`, `dldoa_dataset_generation.py`, and `frozen_banks/eval_bank.npz` (≈22MB total) need to be copied to the other machine. Full list + exact purpose of each file is in the notebook's own Cell 0.
- **Crash/interruption resilience** for a run expected to take hours to days unattended: each architecture checkpoints its weights every `CHECKPOINT_EVERY_EPOCHS` and writes a `<name>.done.json` marker on completion; re-running the notebook from the top skips any architecture whose marker already exists (loads its weights, evaluates it directly) instead of re-training from scratch. Verified by actually running the smoke-test notebook twice in a row — second run correctly detected and skipped all 4 already-done architectures.
- **Live measured ETA, not a guess** — real per-step timing for these 4 architectures didn't exist on any hardware before this (the Kaggle notebook has also never been run), so `train_architecture()` measures actual per-epoch time on the user's own machine after epoch 1 and prints a projected total, instead of estimating blind.
- **Key finding surfaced to the user before building anything:** TensorFlow >=2.11 has *no* native-Windows GPU support at all (confirmed directly — `pip install tensorflow` prints this warning verbatim on this Windows dev machine), regardless of installed NVIDIA drivers. The user's target machine is Linux/Mac (native), which is unaffected — noted in the notebook's Cell 0 as a warning for whoever else might run it on Windows.
- Local dev environment note: this machine's existing global `tensorflow` install was corrupted (Windows long-path limit truncated the package during install — confirmed via the exact failing file path). Worked around for verification purposes with a short-path venv (`D:\venv_tf_smoketest`, outside the repo, not committed) rather than touching Windows system settings (long-path support requires admin/registry changes, out of scope to change unilaterally).

### Real results (user ran it on their own machine, RTX 5090, 2026-09-22)

Full 4.21h run: 8 blocks/arch, 200 epochs x 100 steps (20,000 gradient steps/arch, from scratch), evaluated on the complete 8000-sample eval bank. Results in `notebooks/outputs_full_eval/` (`RESULTS.md`, `architecture_fulleval_results.json`, `fulleval_loss_curves.png`, per-architecture weights + full loss history in `checkpoints/`). **Cross-verified before trusting**: `RESULTS.md`'s tables, every `checkpoints/<name>.done.json`'s raw loss history, and the loss-curve PNG all match the raw `architecture_fulleval_results.json` exactly — no discrepancies, no signs of fabrication.

| Rank | Architecture | Mean Pd | Params | Batch | Loss curve shape |
|---|---|---|---|---|---|
| 1 | **FNO** | **0.6235** | 334,321 | 256 | Healthy, steady drop 2.21 → 0.68 |
| 2 | GridlessUnfold | 0.2216 | 30,201 | 256 | Healthy but shallower drop 1.83 → 1.26 |
| 3 | SIREN | 0.0744 | 3,565 | 256 | **Flat** ~1.7–1.8 for all 200 epochs |
| 4 | WindowAttention | 0.0707 | 7,537 | 64 | **Flat** ~1.7–1.8 for all 200 epochs |
| — | Teacher (ref, 64-block, pretrained) | 0.7103 | 469,393 | — | — |

**FNO is a genuinely strong result**: 88% of the pretrained 64-block teacher's mean Pd (0.6235 vs 0.7103), reached from scratch at 8 blocks / 334K params (0.7% of the teacher's params). This is the clear candidate to invest further effort in.

**SIREN and Window-Attention did not learn — but this looks like a hyperparameter artifact, not proof the architecture families don't work, and should not be written off yet.** All 4 architectures were trained with one identical config (Adam, lr=1e-3, no warmup) — GridlessUnfold and FNO both learned cleanly at this setting, so the optimizer isn't globally broken, but:
- SIREN's periodic `sin()` activation is well-documented in the literature as needing its specific init scheme to train *at all* (present here) — but that scheme was derived for per-pixel coordinate-MLP implicit representations, not for a conv activation stacked 8x with skip connections; the interaction may need a different (likely much smaller) learning rate than 1e-3 to avoid an unfavorable, highly non-convex loss landscape from many periodic activations compounding.
- Attention-based blocks are well known to need LR warmup for training stability; Window-Attention got none here, and on top of that trained at batch=64 (not 256 like the other 3, since that's what the auto batch-probe found before OOM) — meaning it saw ~4x fewer training examples across its 20,000 gradient steps than the other 3 architectures, a further disadvantage in this comparison that isn't a code bug, just an uncontrolled variable.
- Both loss curves are essentially **flat from epoch 1**, not "declining then plateauing" like GridlessUnfold — that specific shape (no initial descent at all) is more consistent with a learning-rate/init mismatch for that architecture than with "this architecture family is fundamentally unsuited to the task."

**Recommended before finalizing the thesis narrative**: a cheap follow-up sweep (a handful of learning rates, far fewer epochs, just for SIREN and Window-Attention) to check whether they're salvageable — not yet done. Until then, the honest framing is "FNO clearly wins under this fixed training config; SIREN/Window-Attention's poor result may be a hyperparameter mismatch rather than a settled architectural verdict."

---

## Proposed next-stage direction (for a second Q1 paper, not scoped into the 1-week plan)

Replace the heatmap+blob-detection output head with a **differentiable set-prediction** architecture (DETR-style query slots, Hungarian-matching loss) — removes the pixel-quantization step that causes the Pd ceiling, and naturally handles unknown path count L (neither paper does — both assume L is given at evaluation). See `My_Proposed_Architecture.md` for the original 3-stage sketch this connects to.

---

## Gotchas / lessons learned (read before re-running anything)

- **Kaggle `/kaggle/working/` is ephemeral** — always "Save Version → Save & Run All (Commit)" or manually download before the session ends. Lost both Notebook-2 student weight files this way.
- **OOM on tiny tensors after a crash** = leftover memory fragmentation, not insufficient GPU memory — restart the kernel, don't just retry with a smaller batch in the same session.
- **`Conv2D` kernel shape is `(kh, kw, in_ch, out_ch)`** — slicing/reshaping the wrong axis is an easy, silent bug (hit twice: once slicing axis=1 instead of axis=2 for Conv2's input-channel slice in `copy_weights_to_pruned`, once an erroneous `.T` before reshape in the low-rank SVD decomposition). **Always verify numerically** (identity/full-rank reconstruction test) before trusting a new weight-manipulation function.
- **Dataset generation needs no Kaggle upload for training** — `training_data_generator` / `validation_data_generator` run on-the-fly from pure numpy/scipy, no TF needed even locally.
