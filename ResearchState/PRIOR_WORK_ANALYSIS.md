# Previous-Result Analysis — DL-DOA Project (`D:\ai_ml_project`)

Sources read in full: `PROJECT_STATUS.md`, `My_Proposed_Architecture.md`, `baseline_models/README.md`, `notebooks/outputs_full_eval/RESULTS.md`, `BUJHO_SHOHOJ_VABE.md` (PIA-Net deep-dive, opened because PROJECT_STATUS.md names it with a specific unresolved claim), plus `git log` on the key files to check dating, and a directory listing of `notebooks/` (25 notebooks) to know what exists without opening all of it.

---

## 1. What has already been tried

**Baseline validation**
- Re-evaluated the paper's pretrained 64-block ResNet (`inf_model_007_256_resnet.h5`, 469,393 params) with the *original* repo evaluator on a frozen 8000-sample bank — reproduces paper Table II to within 0.4% (0 dB: 0.639 vs 0.643; 20 dB: 0.925 vs 0.925).
- Classical 2D-DFT+SIC baseline reproduced paper's DFT-CEA numbers (20 dB: 0.884 vs paper 0.891) — validates the whole physics/data-generation pipeline independent of any learned model.

**Diagnosis of the "ceiling"**
- Found a hard Pd ceiling ≈0.972 for the 256×256 heatmap+blob-detection output representation, even feeding the blob detector a *perfect* ground-truth heatmap. Root cause identified as a Jacobian singularity near end-fire angles (`d(π cos ψ)/dψ = -π sin ψ`) amplifying pixel-quantization error.
- Tested "diffusion-style sharpening" (p⁴ sharpening of raw output, oracle peak-finder) as a fix for this — **ruled out**: made Pd worse, confirming misses come from peaks landing in the wrong pixel location, not from blur.

**Two-stage channel estimation** (Y → CNN → Ĥ → classical angle extraction): built, architecturally validated (<0.05° reconstruction error vs `ifft2(Y)`), but lands at roughly the same ceiling as the direct approach. Not pursued further.

**PIA-Net** (physics-informed deep-unfolded sparse recovery + learned CNN branch + attention, v1→v4): four real implementation bugs were found and fixed over the iterations (`BUJHO_SHOHOJ_VABE.md`):
1. Training data starvation (only 428 of ~500k samples actually used)
2. Steering-vector dictionary sign flipped (peaks off by ~78°)
3. ISTA step size 725× too large, previously masked by clipping rather than fixed
4. Coordinate-system mismatch between the physics branch's answer and the image it was plotted into (119 px / ~57° error → fixed to 7.8 px)

**Model compression** (headline 1-week deliverable): channel pruning of the teacher ResNet (width 12→8, 33% fewer params), two selection methods compared — magnitude-based vs SNR-stratified Taylor-importance ("snr_aware") — plus a low-rank factorization axis that was scoped out.

**Robustness testing**: a novel (not in either paper) 4th-path nuisance-interference test — 200 independent scenes, 3 principal paths + 1 sweep-power nuisance path, 2 SNR levels — comparing teacher vs pruned student.

**Architecture screening**: 4 alternative core blocks (SIREN, FNO/spectral-conv, local Window-Attention, Gridless-Unfold with learned complex soft-threshold) swapped into an identical ResNet-shell shape, each at 8 blocks (no warm start), trained from scratch and compared to the 64-block pretrained teacher — first on a time-capped Kaggle run (never executed), then on an uncapped local RTX 5090 run (executed, 4.21h, full 8000-sample eval bank).

**Baseline registry**: Teacher, magnitude-pruned Student, and FNO all re-evaluated on the *identical* full 8000-sample bank for a rigorous apples-to-apples 3-way comparison.

Not yet built: the 3-stage `My_Proposed_Architecture.md` design (physics-informed deep-unfolding → subspace attention/transformer → differentiable DETR-style set prediction with Hungarian matching) — this is a sketch, not an implementation.

---

## 2. What worked (with numbers)

| Result | Number | Source |
|---|---|---|
| Baseline reproduction | Pd 0.639/0.925 (0/20 dB) vs paper 0.643/0.925 | PROJECT_STATUS.md §"Baseline reproduction" |
| Classical DFT-SIC sanity check | Pd 0.884 vs paper 0.891 @20dB | PROJECT_STATUS.md #3 |
| Magnitude-pruned Student, full 8000-sample bank | Mean Pd 0.6956 vs Teacher 0.7103 (Δ=−0.0147), 314,513 params (67% of teacher) | `baseline_models/README.md` |
| FNO (8-block, trained from scratch, full bank) | Mean Pd 0.6235 (88% of teacher's), 334,321 params (~71% of teacher, i.e. ~29% smaller, **not** ~99% as an earlier draft wrongly claimed — self-corrected in the doc) | `RESULTS.md`, `baseline_models/README.md` |
| GridlessUnfold (8-block, from scratch) | Mean Pd 0.2216, only 30,201 params | `RESULTS.md` |
| Robustness test standout | At SNR=0dB + strongest (0dB) nuisance, pruned student's principal-path Pd (0.6174) **exceeds** teacher's (0.6065), Δ=+0.0109 — the only point in the whole project where a compressed model beats the teacher | PROJECT_STATUS.md §"Notebook 4" |
| SNR-aware vs magnitude pruning (1200-sample subsample, r=8) | SNR-aware mean ΔPd −0.0190 (low-SNR ≤0dB: −0.0082) beats magnitude's −0.0221 (−0.0127) | PROJECT_STATUS.md §"Compression results" |
| Evaluator determinism check | Teacher's Pd reproduced to 13+ significant figures (0.7103495885388549) across independent GPU and CPU runs | `baseline_models/README.md` |

---

## 3. What failed (with numbers) and why

| Failure | Number | Stated/inferable reason |
|---|---|---|
| Diffusion sharpening of model output | Made Pd *worse* | Misses are location errors (Jacobian effect), not blur — wrong fix for the actual failure mode |
| SIREN core block | Mean Pd 0.0744, loss flat ~1.7–1.8 for all 200 epochs (no initial descent at all) | SIREN's init scheme is designed for per-pixel coordinate-MLP implicit representations, not an 8× stacked conv activation with skip connections at lr=1e-3 — likely an LR/init mismatch, not a fundamental architecture failure. **Flagged as unresolved** — a cheap LR sweep was recommended but not yet run. |
| Window-Attention core block | Mean Pd 0.0707, also flat loss from epoch 1 | No LR warmup (attention blocks are known to need it) + forced to batch=64 vs 256 for the other 3 (OOM-driven auto-probe), i.e. ~4× fewer examples seen over the same 20,000 steps — a genuine confound, not a controlled architecture comparison. Same "may be salvageable" caveat as SIREN. |
| GridlessUnfold vs teacher | Mean Pd 0.2216 vs teacher 0.7103 | Learned cleanly (healthy loss curve) but far behind at only 8 blocks/30K params — under-capacity relative to the task, not a training-dynamics failure like SIREN/Window-Attention |
| FNO at high SNR | Pd drops from ~0.80 (10dB) to 0.70 (25dB) while teacher climbs to 0.94 | Spectral representation may lose fine-grained precision once noise stops being the bottleneck — flagged as worth investigating, not resolved |
| PIA-Net (pre-bug-4-fix numbers) | RMSE flat ≈0.579 across SNR at one point (≈ the 0.5774 "pure chance" floor) before earlier bugs were found; later, best reported Pd only 0.60 @25dB vs ResNet's 0.94 | 4 sequential implementation bugs (see §1); **critically, per `BUJHO_SHOHOJ_VABE.md` §8, the model was never retrained after the 4th/biggest fix (coordinate mismatch), so even the 0.60 number is stale** — current true performance is unknown |
| Kaggle student weights (2 runs) | Lost entirely, twice | `/kaggle/working/` is ephemeral unless "Save Version → Save & Run All (Commit)" is clicked — a workflow gotcha, not a modeling failure, but it has real consequences (see §6) |
| Low-rank factorization compression axis | Dropped from the plan (2026-09-21) | Not a result failure — math was verified locally (SVD reconstruction) but the axis was cut to keep scope to one well-validated compression story within the time budget |

---

## 4. Flagged inconsistencies (not silently resolved)

- **Stale status tag in PROJECT_STATUS.md itself**: the "Architecture screening" section header reads `(🔲 built, not yet run, 2026-09-24)`, but the very same section, a few lines below, documents a completed run with real results dated 2026-09-22, and `git log` confirms the notebook was logged 2026-09-21 and results added 2026-09-22. 2026-09-24 is also *after* today (2026-09-23 per session context) — the tag is simply stale and was never updated after the run completed. Treat the "not yet run" framing as wrong; the run happened.
- **The "SNR-aware pruning wins" claim is not the model that ended up in the rigorous full-bank comparison.** The compression section reports SNR-aware beating magnitude-based pruning, but only on a 1200-sample subsample (n≈150/SNR, explicitly flagged as having ~2% sampling noise). The SNR-aware student's weights were then lost twice on Kaggle and were **never recovered** — so `baseline_models/README.md`'s full-8000-sample 3-way comparison (Teacher vs Student vs FNO) uses the **magnitude**-pruned student, not the nominally "winning" SNR-aware one. Similarly, the Notebook-4 robustness test also used the magnitude-based student "avoiding a costly re-run." So the project's headline full-bank numbers are for the *runner-up* compression method by the project's own earlier comparison — the supposedly-better method's real-scale performance is simply unknown, not confirmed-then-superseded.
- **Two different "PIA-Net physics issue" claims that don't obviously map onto each other.** PROJECT_STATUS.md item 5 says PIA-Net "has an unresolved physics-consistency issue: the learned ISTA state is real/ReLU-constrained but a physical path has arbitrary complex phase — flagged, not yet fixed." `BUJHO_SHOHOJ_VABE.md` documents 4 *different* named bugs (data starvation, dictionary sign, ISTA step-size, coordinate mismatch), the last of which is described as already fixed (just not retrained). Neither document cross-references the other's specific claim, so it is unclear whether the "real/ReLU vs complex phase" issue is a 5th distinct unresolved design flaw or a restatement of one of the 4. Do not assume they're the same issue without checking the v4 notebook directly if this matters to the new work.
- **Self-corrected (worth noting, not a live inconsistency)**: an earlier draft in PROJECT_STATUS.md claimed FNO was "~99% smaller" than the teacher; the doc itself catches and corrects this to ~29% smaller (334K vs 469K params) — a good sign of rigor, flagged here only so nobody reproduces the wrong 99% figure from an older commit or cached copy.

---

## 5. What should NOT be repeated

- Re-deriving that the baseline/paper reproduction is correct — already confirmed to within 0.4% of Table II, and the classical DFT-SIC sanity check independently confirms the physics/data pipeline.
- Diffusion-based output sharpening as a fix for low Pd — tested directly, made things worse, root cause (Jacobian singularity, not blur) is understood.
- Low-rank SVD factorization as a compression axis — validated then explicitly dropped for scope reasons; revisit only if compression is being revisited as its own topic, not by re-deriving the math.
- Treating the *stricter* per-source Hungarian/no-sample-dropping metric used in several early exploratory notebooks as comparable to paper Table II or to the compression-project notebooks — it is a different metric definition. Any new work must use `evaluate_on_bank` from `DL_DOA/src/TVT_Blob_Inference.py` on `frozen_banks/eval_bank.npz` to stay comparable to everything in `baseline_models/`.
- Retraining/rerunning Kaggle notebooks without clicking "Save Version → Save & Run All (Commit)" — this has already destroyed two full training runs' worth of weights.
- Assuming SIREN/Window-Attention "don't work" as an architecture family — the data supports "didn't work under this one untuned config," not a settled negative result; PROJECT_STATUS.md itself says not to write this off yet pending an LR/warmup sweep that hasn't been run.

---

## 6. Promising components/ideas to build on rather than reinvent

- **FNO / spectral-convolution core block** — clear current best among the 4 screened alternatives (0.6235 mean Pd from scratch, 8 blocks, 334K params, 88% of teacher's Pd) and a strong architectural fit rationale (input Y is already frequency-domain; near-free global receptive field vs 64 stacked local blocks). Its high-SNR degradation is a known, scoped-out weak spot worth targeted follow-up rather than wholesale replacement.
- **Magnitude-pruned Student (r8, 64-block)** — currently the strongest single compressed candidate on the only rigorous full-bank comparison available (Δ=−0.0147 vs teacher, smaller than FNO both in params and gap-to-teacher), and it only needed fine-tuning from teacher weights rather than training from scratch. It is also the model already wired into `baseline_models/` reload code, ready to extend.
- **Robustness-test protocol from Notebook 4** — a genuinely novel (neither paper does this) nested/nuisance-path methodology, already implemented, fast (17.3 min for 200 scenes × 2 SNR × 3 power × 2 models on CPU), and produced the project's one clear "compression doesn't make things worse, and may help under stress" data point. Reusable directly for testing the SNR-aware student (once recovered) or any future architecture.
- **Gridless-Unfold's complex soft-threshold layer** — directly targets the diagnosed Pd-ceiling root cause (pixel quantization) and already has a working, verified `Layer` implementation (phase-preserving complex soft-threshold, unit-tested against PIA-Net's exact known bug). Under-performed only because of shallow depth (8 blocks) in the screening run, not because the mechanism failed — a plausible candidate to combine with FNO or scale up rather than discard.
- **`My_Proposed_Architecture.md`'s Stage 3 (differentiable set prediction / DETR-style queries + Hungarian matching)** — directly removes the non-differentiable blob-detection step, which is the exact mechanism identified as the source of the measured Pd ceiling (≈0.972) and the SNR/end-fire Jacobian problem. This is the most evidence-backed unbuilt idea in the repo: it targets a *root cause* that multiple independent experiments (ceiling test, diffusion-sharpening ablation) converged on, rather than a hypothesis.
- **Physics + learned dual-branch fusion idea from PIA-Net** — the core "physics branch keeps the model interpretable and prevents catastrophic failure, learned branch gives fast convergence" concept is validated as sound reasoning even though the current implementation is bug-affected/stale; the fusion *idea* itself, not the current buggy weights, is worth carrying into any new physics-informed design.

---

## 7. Unexplored directions (grounded in documented gaps)

1. **LR/warmup sweep for SIREN and Window-Attention** — explicitly recommended in PROJECT_STATUS.md, not yet run. Cheap (few configs, few epochs) and could reverse the current "essentially failed" verdict for either family before it's written off in the thesis.
2. **Step-count-matched (not just wall-clock-matched) re-run of the 4-way screening** — PROJECT_STATUS.md's own caveat: FNO/Window-Attention do fewer gradient steps than SIREN/Gridless-Unfold per unit wall-clock because FFT/attention ops are more expensive per step, so the current ranking conflates architecture quality with step count for the time-capped Kaggle variant (though the *executed* comparison was actually step-matched at 20,000 steps each — this gap applies mainly if the Kaggle time-capped notebook is ever run as originally planned).
3. **Recovering/re-training the SNR-aware ("snr_aware") pruned student and running it through both the full 8000-sample bank and the Notebook-4 robustness protocol** — this is the single most concrete open loop in the project: the project's own earlier evidence suggested SNR-aware pruning is the better method, but no full-scale or robustness number exists for it, only for the weaker magnitude method. This directly resolves the inconsistency flagged in §4.
4. **PIA-Net retraining after the bug-4 (coordinate mismatch) fix, plus resolving whether the "real/ReLU vs complex phase" issue in PROJECT_STATUS.md is the same as or separate from the 4 bugs in BUJHO_SHOHOJ_VABE.md** — currently all reported PIA-Net numbers are known-stale by the project's own documentation.
5. **FNO's high-SNR degradation** — flagged twice (RESULTS.md and baseline_models/README.md) as worth investigating but not yet studied; a natural target given FNO is the current best alternative architecture.
6. **Building `My_Proposed_Architecture.md`'s full 3-stage design (or at minimum Stage 3, the differentiable set-prediction head)** — a proposed architecture sketch, never implemented, that targets the one quantitatively-confirmed structural bottleneck (blob-detection pixel quantization / Jacobian singularity) found across three independent experiments.
7. **Combining a promising core block (FNO or Gridless-Unfold) with pruning/distillation**, rather than treating "architecture screening" and "compression" as separate tracks — no experiment yet compresses one of the 4 screened alternatives; the two tracks have run fully independently so far.

---

## 8. Scope note — notebooks not opened

Per instructions, I did not exhaustively open all ~25 notebooks in `notebooks/`. I opened the 4 primary sources plus `BUJHO_SHOHOJ_VABE.md` (opened because PROJECT_STATUS.md names PIA-Net and `DLDOA_PIANet_Standalone_v4.ipynb` with a specific unresolved claim). I did **not** open, and PROJECT_STATUS.md does **not** reference by name with a specific claim requiring verification: `DLDOA_SetRegNet_Standalone.ipynb`, all 3 `DLDOA_SE_ResNet_Lite*.ipynb` variants, `proposed pin architecture.ipynb`, `DLDOA_PIANet_Standalone.ipynb`/`_v2`/`_v3` (only v4 is cited), `DLDOA_NoClone_StepByStep_attempt_2.ipynb`, `DLDOA_ResNet_Reproduction.ipynb`, `DLDOA_Classical_OMP.ipynb`, `DL_DOA_Dataset_Explorer.ipynb`, `LowSNR_Visual_Investigation.ipynb`, `DLDOA_Visualization.ipynb`, `SE_ResNet_Full64_Kaggle.ipynb`, and `csida baseline vs tcnpooled multi seed.ipynb`. These exist in the repo and may contain relevant exploratory results, but nothing in the primary sources makes a specific numeric claim about them that needed cross-checking, so they were left unopened per the scoping instruction. A subsequent agent should open these only if a specific claim about one of them surfaces that needs verifying.