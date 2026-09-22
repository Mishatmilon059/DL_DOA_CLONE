# MOTHER RESEARCH AGENT — FINAL CONVERGENCE REPORT
## Lightweight Joint AoA/AoD Estimation — Multi-Agent Research Loop, Cycle 1

**Project:** `D:\ai_ml_project` | **Date:** 2026-09-23 | **Authority:** Mother Research Agent, final arbiter over all lower-level agent outputs per protocol §21-22

---

## 1. Executive Summary

This report converges 6 Tier-1 papers + 8 Tier-2 papers of independently-verified literature evidence, a full audit of this project's own prior experimental work, 8 specialist research agents (architecture, compression, robustness, signal-processing, novelty, experiment-design), and 3 independently-assembled candidate architectures that each survived two rounds of adversarial red-team review, into a single recommended research direction.

**Selected final candidate: IABR-Net** (Impairment-Adaptive Beamspace Residual-correction Network, Candidate B, Revision 2) — a fixed 2D-DFT beamspace front end, a novel per-element shared-weight impairment-correction module (IABC-v2), a shallow 10-block ResNet trunk operating entirely in beamspace (no resolution up/downsampling), and a coarse-heatmap + fixed-size top-K crop-refinement dual head that preserves the joint AoA/AoD pairing property. Estimated **≈193K parameters (≈41% of the 469,393-param teacher)** and **≈95.3M FLOPs** — by a wide margin the most compute-efficient of the three candidates after all arithmetic was independently re-verified twice.

This is **not** a confident, closed decision. It is a best-available trade-off under real, stated uncertainty: IABR-Net's central robustness claim (closing Meneses-Albalá 2026's own named gain-error gap) is currently **unbuildable** — gain-error injection code does not exist anywhere in this project's codebase (confirmed by direct code audit in two independent adversarial reviews) — and must be built as a **blocking Phase-0 prerequisite** before any robustness claim is made. Its efficiency numbers are estimates pending real `ptflops`/`thop` instrumentation. Its capacity-vs-impairment-adaptivity attribution is unconfirmed pending a controlled ablation.

A second architectural direction — Candidate A's differentiable, DETR-style set-prediction head, which removes the discrete-heatmap/non-differentiable-peak-search mechanism diagnosed (by this project's own ceiling test + diffusion-sharpening ablation) as the root cause of a hard Pd≈0.972 ceiling — is **not selected as the primary candidate** (its FLOPs, params, and implementation risk are all higher, and its own screening evidence was shown under adversarial review to validate a different configuration than what it deploys), but is retained as the **highest-value Phase-2 research direction**, to be fused onto IABR-Net's efficient trunk once IABR-Net's own open items are resolved.

Candidate C (SSC-Net) is **rejected**: after two rounds of correction its own numbers show it to be the *least* efficient of the three (≈8.72 GFLOPs, roughly 8–90× the other two), it does not remove the diagnosed non-differentiable bottleneck, and its most distinctive design element is explicitly self-gated behind a diagnostic it may fail.

---

## 2. User's Research Objective

Implement a multi-agent, loop-based research protocol (per `MULTI-AGENT RESEARCH LOOP ENGINEERING PROTOCOL.md`) to converge this project's prior lightweight-neural-DOA work with the joint AoA/AoD estimation literature, and produce a rigorous, trade-off-weighted, non-fabricated final research report — a candidate lightweight architecture, its justification, its risks, and a concrete experimental roadmap — suitable to hand to a thesis supervisor as an honest snapshot of where the research stands, not a finished result.

---

## 3. Literature Landscape

14 papers were reviewed: 6 Tier-1 (independently specialist-extracted **and** independently verified — every load-bearing numeric claim survived verification except one flagged fabrication, see §11/Contradiction 7) and 8 Tier-2 (grouped extraction, lighter verification). The lineage centers on **Lloria et al.** (TVT 2026 base paper + PIMRC 2024 prototype), extended by **Meneses-Albalá et al.** (J.Supercomputing 2026, edge/energy variant), running in parallel to a **complex-valued MLP** lineage (Naoumi et al., JSTSP 2024 + GCWkshps 2023, bistatic ISAC) and a **non-DL Bayesian/classical** lineage (Gupta et al., TCOMM 2025). Eight Tier-2 papers (Koh & Lee 2026, Lim et al. 2021, BeamSeek, Sub6GHz CNN/UNet, SubspaceNet, Robust DoA DAE-DNN, MIMO-Radar V2X, Tensor PARAFAC) sit adjacent — mostly single-axis DOA, not joint 2D — but contribute reusable *patterns* (SubspaceNet's learned-surrogate-feeds-classical-backend design; Koh & Lee's sim-vs-real domain-shift benchmark methodology).

A supplementary novelty-focused search (Agent-R7) found the 14-paper corpus is **not exhaustive**: at least four additional on-topic papers (a 2023 joint AoD/AoA-under-hardware-impairment paper, a 2024 AoD/AoA-map paper, the 2025 SABER symbolic-regression estimator, and a cluster of attention/transformer 2D-DOA papers) exist outside it and were not read in full. This materially narrows confidence in any "the field has never tried X" claim — see §11 and §28.

---

## 4. Complete Paper Comparison (condensed master matrix)

| Paper | Joint AoA/AoD? | Family | Key metric | Params | FLOPs/latency | Robustness axes | HW tested |
|---|---|---|---|---|---|---|---|
| Lloria TVT2026 (base) | YES, genuine | ResNet+U-Net | PD/RMSE @20dB: ResNet .925/.253°, U-Net .945/.230° | NOT REPORTED | asymptotic O(PQ) only | SNR, L∈1–6, phase δ∈{1,2,5}° | NOT REPORTED (sim only) |
| Lloria PIMRC2024 | YES | ResNet+PH search | figure-only | NOT REPORTED | asymptotic only | SNR, L∈1–6 | NOT REPORTED |
| Naoumi JSTSP2024 | YES (bistatic) | Complex MLP | MSE rad² vs SNR; ~9dB gap to CRB | NOT REPORTED (formulas only) | 6.5×/10.3× fewer mults vs 2D-algo | SNR, #targets/peak | NOT REPORTED |
| Naoumi GCWkshps2023 | YES | Complex MLP | MSE≈10⁻⁶ @SNR≥9dB | **8,584** | 4.321 MMACs fwd | SNR only, narrow [20°,40°] window | NOT REPORTED |
| Gupta TCOMM2025 | **NO** (deliberately not joint) | SBL+MUSIC (classical) | NMSE (RCS) vs SNR | N/A | O(Gτ³Gν³Gθ³) | SNR, scale | NOT REPORTED |
| Meneses-Albalá 2026 | YES (behavior, not word) | U-Net reuse + freeze-FT | RMSE 0.28→0.27, PD 0.93→0.94 @γ=5° | NOT REPORTED | training energy 0.28–0.41 J/sample | γ∈{1,2,5}°, SNR, L, codebook | **YES — real Jetson Orin Nano** |
| SubspaceNet | NO (1D DOA) | CNN-DCNN AE→MUSIC | RMSPE 12.48°→0.20° | **41,761** | NOT REPORTED | coherent, broadband, few-snapshot, miscalibration | NOT REPORTED |
| BeamSeek | NO (1D azimuth) | SwiGLU-MLP | DOA err 3.86–21.45° | NOT REPORTED | NOT REPORTED (latency claim fabricated upstream — excluded) | SNR, beam-count | real 60GHz testbed |
| Koh&Lee 2026 | NO (BLE) | 11-model survey | CNN MAE 2.98°, GRU<1.3° real | 19.8k (best CNN) | 39k FLOPs | sim-vs-real domain shift | real BLE |

**Comparability rule applied throughout:** Lloria↔Meneses-Albalá = HIGH (identical generator/eval); Lloria↔Naoumi and any joint↔non-joint pairing = LOW/NOT COMPARABLE (units, task framing differ). No cross-paper number in this report is quoted without this qualifier.

---

## 5. Dataset Comparison

| Paper family | Data | Array | SNR range | Notes |
|---|---|---|---|---|
| Lloria-family (TVT2026/PIMRC2024/Meneses2026) | Synthetic, single-user mmWave MIMO | ULA 16/32 | Text: −10…25dB; **code: −15…24dB** (Contradiction, §11) | This project's generator inherits the **code** range |
| Naoumi-family | Synthetic bistatic ISAC | Nt=8, Nr=10 | Train 5–40dB / eval −5…30dB (JSTSP); −9…30dB (GCWkshps) | LOW comparability to Lloria (different antenna counts/framing) |
| Gupta 2025 | Synthetic, 2 configs | — | −5…25dB | NOT COMPARABLE (non-joint, non-DL) |
| This project (`eval_bank.npz`) | Synthetic, inherits Lloria code generator | nt=nr=16, P=Q=16 | −10…25dB, 8 pts, σ=0.07 | **Fixed L=3, zero impairment** in the frozen bank — every other condition needs a new bank generation call, not new physics code |

---

## 6. Architecture Comparison

Zero of the 6 Tier-1 papers use Transformer/Attention (a corpus-scoped gap, not a field-wide one per §3/§28). Component families seen: ResNet, U-Net, complex-valued MLP, classical subspace (MUSIC/SBL/EM), CNN-DCNN autoencoder, LSTM+EKF. This project's own from-scratch screening (identical ResNet-shell, 8 blocks each) ranked: **FNO** (Pd 0.6235, 334,321 params) > **magnitude-pruned Student** (Pd 0.6956, 314,513 params, fine-tuned not scratch) > **GridlessUnfold** (Pd 0.2216, 30,201 params) > **SIREN/WindowAttention** (near-zero Pd, confounded training runs, not a settled negative — §12).

---

## 7. Metric Comparison

| Metric family | Who reports it | This project |
|---|---|---|
| PD (paper-style, drops missed-detection trials from both num/denom) | Lloria TVT2026, Meneses2026 | Inherited **as-is** in `evaluate_on_bank` — a non-standard, PD-inflating convention (Contradiction 5, Gap 4) |
| PD (strict, missed=0) | **Nobody** | Not yet implemented — a ~3-line, low-risk addition, recommended as standard on every future table |
| RMSE (combined ψ+φ, pooled) | Lloria-family | Teacher reproduces paper Table II to within 0.4% (0.639/0.925 vs 0.643/0.925 @0/20dB) |
| MSE (rad²) | Naoumi-family | NOT COMPARABLE (different units/task) |
| AoA-only / AoD-only RMSE | **Nobody** | Not yet split out — cheap, same underlying arrays |
| P95 tail error | **Nobody** | Not measured; directly relevant to the diagnosed end-fire ceiling |

---

## 8. Robustness Comparison

No paper in the reviewed set covers more than 3 of 7 robustness axes (SNR, multipath, phase/gain error, array perturbation, domain shift, mobility, hardware). SubspaceNet is the single most comprehensive (4 axes, but 1D-only). Genuine, literature-wide gaps: gain/amplitude-mismatch robustness (explicitly named as unfinished future work by Meneses-Albalá 2026 itself), phase error beyond 5°, array-element perturbation for the joint-2D task, and genuine train/test distribution shift (zero Tier-1 papers test this). This project's own Notebook 4 nuisance-interference test is the only beyond-the-paper's-own-protocol robustness result in the whole evidence base, and it surfaced the project's single most interesting robustness finding: **at SNR=0dB + strongest nuisance, the pruned student's principal-path Pd (0.6174) exceeds the teacher's (0.6065), Δ=+0.0109** — compression did not cost robustness here, and may have helped.

---

## 9. Complexity Comparison

| Model | Params | FLOPs |
|---|---|---|
| Teacher (64-block ResNet, pretrained) | 469,393 | NOT MEASURED |
| Magnitude-pruned Student (r=8, fine-tuned) | 314,513 | NOT MEASURED |
| FNO (8-block, scratch) | 334,321 | NOT MEASURED |
| GridlessUnfold (8-block, scratch) | 30,201 | NOT MEASURED |
| Naoumi GCWkshps2023 (different task) | 8,584 | 4.321 MMACs fwd |
| SubspaceNet (different task, 1D) | 41,761 | NOT MEASURED |
| **IABR-Net (proposed, estimate)** | **≈193,104 (≈41.1% of teacher)** | **≈95.3M FLOPs (estimate, unvalidated — no in-repo FLOP tool exists)** |
| Candidate A / SpectraSet (rejected primary, estimate) | ≈255,671 (≈54.5% of teacher) | ≈0.78 GFLOPs (estimate) |
| Candidate C / SSC-Net (rejected, estimate) | ≈199K–224K (central) | ≈8.72 GFLOPs (estimate — FFT-dominated) |

**No tool for FLOP counting exists anywhere in this project's codebase today** (confirmed by Agent-R8's code audit). All FLOP figures above are hand-derived estimates, explicitly flagged as such by their own authors, and none should be treated as measured until `ptflops`/`thop`/manual instrumentation is built and run.

---

## 10. Hardware/Latency Comparison

Only **Meneses-Albalá 2026** used real edge hardware (Jetson Orin Nano) in the entire 14-paper corpus, and it reports training/fine-tuning **energy** (0.28–0.41 J/sample), not standalone inference **latency**. BeamSeek and Koh&Lee 2026 used real hardware (60GHz testbed, BLE) but for 1D-only tasks with no comparable latency figure (BeamSeek's latency numbers were **fabricated upstream in this pipeline and are explicitly excluded**, Contradiction 7). **No paper in the entire reviewed set reports standalone inference latency for a joint 2D AoA/AoD estimator.** This project has a working `benchmark_latency()` function (`notebooks/proposed pin architecture.ipynb`, wall-clock, TF `.count_params()`, 20-run mean±std, single-sample batch) that exists today but has never been run against the current baseline registry (Teacher/Student/FNO) or any proposed candidate — this is the single cheapest, highest-value open experiment identified anywhere in this evidence chain (Gap 8).

---

## 11. Research Gaps

Ten evidence-tagged gaps were identified (Paper Head synthesis §5); the four most load-bearing for the architecture decision:

- **Gap 1** — no paper reports params+FLOPs+latency+memory together for a joint AoA/AoD model; this project already has the params half of this story and can close it cheaply.
- **Gap 2** — the dense/flatten bottleneck is the literature's own self-flagged worst component (Lloria TVT2026 Table IV's only quadratic-in-(P'Q') term); this project's FNO result is direct evidence a non-dense alternative is viable.
- **Gap 5** — the diagnosed Jacobian-singularity/pixel-quantization Pd≈0.972 ceiling (confirmed by this project's own ceiling test + diffusion-sharpening ablation, which ruled out blur) is absent from every Lloria-family paper's own limitations section despite sharing the identical output representation. This is the single most evidence-convergent unbuilt idea in the whole evidence base.
- **Gap 3** (robustness/OOD) — zero Tier-1 papers test genuine distribution shift; gain-error and >5° phase-error are named, open, literature-licensed gaps this project could close first.

**13 contradictions were flagged and preserved as open uncertainty** rather than silently resolved (full list in the literature synthesis, §6). Most consequential for architecture work: (Contradiction 2/3) the released Lloria U-Net weight file's actual kernel size and whether it uses the dense bottleneck at all is unverified from the paper text alone — any future U-Net baseline citation needs the `.h5` file inspected first, not assumed.

---

## 12. Previous Experiment Analysis (this project's own prior work)

**What worked:** baseline reproduction (Pd 0.639/0.925 vs paper 0.643/0.925), classical DFT-SIC sanity check (Pd 0.884 vs paper 0.891 @20dB — 95.6% of the 469K-param teacher's accuracy with **zero learned parameters**, the single strongest piece of evidence motivating IABR-Net's classical front end), magnitude-pruned student full-bank result (Pd 0.6956 vs teacher 0.7103, 314,513 params), FNO screening (Pd 0.6235, 334,321 params, 88% of a depth/pretraining-**mismatched** teacher reference — caveat now carried through this whole report), the Notebook 4 robustness standout described in §8, and 13+-significant-figure evaluator determinism.

**What failed, and why:** diffusion-style output sharpening (made Pd worse — confirmed misses are *location* errors, not blur — this is the key diagnostic fact behind Gap 5); SIREN and WindowAttention core blocks (near-zero Pd, but **both runs are confounded** — untuned LR/init for SIREN, no warmup + 4× smaller batch for WindowAttention — not a settled negative, an unrun LR/warmup sweep is still pending); GridlessUnfold under-performed at only 8 blocks (Pd 0.2216, but whether this is capacity-limited or mechanism-limited is itself **unconfirmed**, per Critique #1 on Candidate C); PIA-Net (four sequential real bugs found and partially fixed, but never retrained after the last fix — all current PIA-Net numbers are **known-stale**).

**Flagged inconsistencies not silently resolved:** the SNR-aware pruned student's weights were lost twice on Kaggle and never recovered, so the project's rigorous full-bank comparisons use the (weaker, by the project's own earlier partial evidence) magnitude-pruned student, not the nominally-better SNR-aware one — recovering this is a concrete, still-open item (§21).

---

## 13. Candidate Architectures (all 3, briefly)

- **Candidate A — SpectraSet** (FNO trunk → depth-extended GridlessUnfold precision stage → shared-query DETR-style set-prediction head, Hungarian/greedy matching). Directly removes the discrete-grid mechanism diagnosed as Gap 5's root cause "by construction." Revised estimate ≈255,671 params (≈54.5% of teacher), ≈0.78 GFLOPs. Highest implementation risk and highest novelty ceiling.
- **Candidate B — IABR-Net** (fixed DFT beamspace front end → per-element shared-weight impairment-correction module IABC-v2 → shallow 10-block ResNet trunk, held at 16×16 throughout → coarse heatmap + fixed top-K=10 crop-refinement dual head). Revised estimate ≈193,104 params (≈41.1% of teacher), ≈95.3M FLOPs. Robustness-first, lowest compute cost, has a blocking unbuilt-code dependency.
- **Candidate C — SSC-Net** (grouped-channel FNO trunk at 64×64 → conditionally depth-extended GridlessUnfold head at 256×256 → fusion gate → post-hoc pruning). Revised estimate ≈199K–224K params (central), ≈8.72 GFLOPs. Most confounded (5 simultaneously-varying novel components), most walked-back after critique.

All three underwent **two independent adversarial reviews** that found real, load-bearing arithmetic errors (channel-width miscounts, inconsistent complex/real parameter accounting, omitted FFT cost) in every one of the three original drafts. Every number quoted above is the **post-correction, Revision 2** figure.

---

## 14. Rejected Architectures and Why

**Candidate C (SSC-Net) — rejected.** After correction it is the least compute-efficient of the three (≈8.72 GFLOPs, dominated by an FFT term the original draft omitted entirely and a 256×256-resolution precision head), it retains the non-differentiable blob-detection step (does not address Gap 5 architecturally, only via an unvalidated loss-reweighting term whose causal claim both critics independently rejected as mechanically unable to fix a representational problem), its second-most-distinctive component (GU depth extension) is explicitly gated behind a diagnostic experiment it may fail, its pruning-ratio and grouped-conv-reduction assumptions were both shown to be unjustified transplants from structurally different architectures, and its backbone-pattern novelty was downgraded to KNOWN COMBINATION even at the pattern level once compared against the broader (not just DOA-specific) computer-vision literature.

**Candidate A (SpectraSet) — not selected as primary, retained as Phase-2 direction.** Its central architectural claim (no discrete grid exists at inference, so the diagnosed failure mode cannot occur "by construction") is the strongest, most root-cause-targeted claim of the three and was the one claim neither critic challenged. But: (1) its FLOPs (0.78 GFLOPs) and params (255,671, 54.5% of teacher) are both meaningfully higher than IABR-Net's; (2) its "reuses validated components" argument was shown to be weak — the FNO/GridlessUnfold screening results it cites validated a different depth, head, and loss than SpectraSet actually deploys; (3) it carries the highest implementation risk (Hungarian matching is CPU-bound and non-batched, DETR-style heads are documented outside this literature to converge slowly, and N_q=8 cannot even represent the project's own OOD L∈{7,8,10} robustness sweep by construction — a hard capacity ceiling); (4) no latency number exists or is even estimated, only flagged qualitatively; (5) its narrowest, most defensible novelty claim (the DETR/set-prediction pattern) remains explicitly UNCERTAIN per Agent-R7, pending a dedicated literature search that has not yet been run.

---

## 15. Final Proposed Architecture

**IABR-Net (Impairment-Adaptive Beamspace Residual-correction Network)**, Candidate B, Revision 2, selected for this report's next research phase. Selection rationale, weighing all axes together (not any single metric):

| Axis | Assessment |
|---|---|
| Efficiency (params) | Best of the three: ≈193K (≈41.1% of teacher), below FNO (334K) and the pruned student (314K) |
| Efficiency (FLOPs) | Best of the three by a wide margin: ≈95.3M vs A's ≈0.78B and C's ≈8.72B (estimates, unmeasured) |
| Accuracy/Detection | Untested — no run yet exists; inherits some risk from the retained discrete coarse grid |
| Robustness | Strongest explicit design intent — targets two literature-named, currently-untested gaps (gain error, phase>5°) — but currently **unbuildable pending Phase-0 code** |
| Generalization | Untested; capacity-vs-adaptivity attribution unconfirmed pending a controlled ablation |
| Novelty | PARTIALLY NEW at best, UNCERTAIN pending a dedicated literature search (unchanged verdict from Agent-R7) |
| Implementation risk | Lowest of the three — no Hungarian matching, no FFT-based spectral layers, no depth-gated conditional component |
| Root-cause fix for Gap 5 | **Not solved** — this is IABR-Net's single largest scientific weakness relative to Candidate A |

IABR-Net wins on the combination of efficiency, robustness-relevance, and buildability; it explicitly does **not** win on directly resolving this project's own most evidence-convergent open question (Gap 5), which is why Candidate A's set-prediction head is retained as the named Phase-2 fusion target (§31).

---

## 16. Detailed Architecture Diagram (data flow)

```
Input Y (raw complex antenna-domain signal, Nr x Nt = 16x16)
   │
   ├─► [FIXED, 0 params] 2D-DFT beamspace projection  F_codebook^H · Y  →  Y_bs (16x16, complex)
   │
   ├─► [LEARNED, IABC-v2 — shared-weight per-element correction]
   │        Global context: pool(mean|mag|, mean∠, std|mag|, std∠) over 32 elements → FC(4→8)
   │        Per-element (x32, SHARED weights): own(mag,phase)[2] ⊕ context[8] → FC(10→16)→ReLU→FC(16→2)
   │              → (δ̂ phase, γ̂ gain) per element, applied multiplicatively to Y_bs
   │        →  Y_bs_corrected (16x16, 2-channel real/imag)
   │
   ├─► [SHARED TRUNK] 10-block shallow ResNet (C=32, 3x3 conv + SE channel-attention/block)
   │        held at 16x16 spatial size throughout (no down/upsampling)
   │        →  F_shared (16x16x32)
   │
   ├─► [FUSION] Path-token 1x1 conv (32→32) on F_shared → F_token (standard shared-bottleneck pattern)
   │
   ├──────────────┬───────────────────────────────┬─────────────────────────────┐
   ▼               ▼                                ▼                             ▼
 Coarse joint    AoA crop-refine head          AoD crop-refine head        SE→SNR diagnostic
 2D heatmap      fixed top-K=10 batched         fixed top-K=10 batched      head (auxiliary,
 (1x1 conv,      7x7 gather around each         7x7 gather around each      ablatable, tests
 blob-detector-  coarse peak → conv→FC          coarse peak → conv→FC       A3's hypothesis)
 compatible)     → Δψ per slot                  → Δφ per slot
   │                   │                                │
   ▼                   ▼                                ▼
 Coarse (ψ,φ)     ψ_final = ψ_coarse+Δψ          φ_final = φ_coarse+Δφ
   │                   └──────────── paired by shared coarse-peak index ────────┘
   └──────────────────────────────────► Final (AoA, AoD) output pairs
```

Every stage from the DFT front end through the path-token fusion is shared; branching occurs only at the three parallel heads, which are paired by coarse-peak index (a standard shared-bottleneck pattern — this document does not claim it as a novel fusion mechanism).

---

## 17. Layer-by-Layer Specification

| # | Layer | Shape in → out | Params (est.) |
|---|---|---|---|
| 0 | DFT beamspace projection | 16×16 complex → 16×16 complex | 0 (fixed) |
| 1 | IABC-v2 context FC | 4 → 8 | 40 |
| 2 | IABC-v2 per-element MLP (shared, ×32 applications) | 10 → 16 → 2 | 210 |
| 3 | Trunk stem conv | 3×3, Cin=2 → Cout=32 | 608 |
| 4–13 | 10× residual block (C=32) | 3×3 conv ×2 + BN | 185,600 total |
| 4b–13b | 10× SE channel-attention (r=8) | 32→4→32 | 2,920 total |
| 14 | Path-token 1×1 conv | 32 → 32 | 1,056 |
| 15 | Coarse heatmap head, 1×1 conv | 32 → 1 | 33 |
| 16 | AoA crop-refine head (7×7×32 crop) | conv 32→16, FC 784→1 | 1,313 |
| 17 | AoD crop-refine head | identical shape | 1,313 |
| 18 | SE→SNR diagnostic head (auxiliary) | 10 → 1 | 11 |
| | **Total (estimate)** | | **≈193,104** |

Estimate only — component-level param arithmetic verified twice by adversarial review; the total has **not** been measured against an actual model instantiation (`model.count_params()`), which is a required first step before any downstream claim (§30 Reproducibility Checklist item 1).

---

## 18. Mathematical Formulation

**DFT beamspace projection:** `Y_bs = F_codebook^H · Y`, where `F_codebook` is the fixed P×Q angular-grid DFT codebook shared with Lloria TVT2026's own output grid convention.

**IABC-v2 per-element correction:** for element `i ∈ {1..32}`, `context = FC(pool(mag, phase))` (dim 8, shared across elements); `(δ̂_i, γ̂_i) = MLP(mag_i, phase_i, context)`; correction applied multiplicatively to that element's contribution before beamforming.

**Circular angle error** (reused unmodified from the project's own evaluator, `get_ang_difference`): `H = angle(exp(i·ψ_gt)·exp(-i·ψ̂))`, avoiding wraparound artifacts — this convention is kept for comparability to `baseline_models/README.md`.

**Peak-conditioned refinement:** for each of the top-K=10 coarse-heatmap scores, a 7×7 crop around the peak location feeds a per-angle offset regression: `ψ_final = ψ_coarse + Δψ`, masked/excluded from loss for slots beyond the true path count `L`.

---

## 19. Loss Function

```
L = λ1 · L_heatmap        (BCE/MSE, joint 2D coarse heatmap vs Gaussian-blurred GT peak map — Lloria-style)
  + λ2 · L_offset          (Smooth-L1 on Δψ, Δφ at matched, unmasked top-K slots; nearest-GT
                             assignment valid at L≤10, well-separated sources — no Hungarian matcher needed)
  + λ3 · L_pair_consist    (MSE between IABC-v2-corrected beamspace map and the SAME channel realization's
                             clean, δ=γ=0 beamspace map — synthetic paired supervision, training-time only;
                             this is ordinary regression against a simulator-paired target, not
                             self-supervised learning in the standard sense)
  + λ4 · L_SE_snr           (small-weight auxiliary regression: pooled SE-gate → true per-sample SNR label;
                             ablatable, does not affect angle outputs if removed)
```

Recommended starting weights (unvalidated, requires the sensitivity sweep in §24/§25): `λ1=1.0, λ2=1.0, λ3=0.5, λ4=0.1`.

---

## 20. Training Protocol

- **Data:** `frozen_banks/eval_bank.npz` for in-distribution evaluation (must stay untouched for comparability); new banks generated via the existing, unmodified `validation_data_generator` for multipath (L∈{1..6}) and phase-error (δ∈{0,1,2,5}°) sweeps; **new physics code required** for gain-error injection (Phase-0 blocking prerequisite, §21).
- **Step budget:** must be step-count-matched against the project's existing screening baselines (20,000 steps, per the executed FNO/GridlessUnfold/SIREN/WindowAttention comparison) to avoid the step-count confound flagged in Prior-Work §7 item 2.
- **Optimizer/schedule:** [TO MEASURE / TO DECIDE] — not specified in any source document; needs an explicit decision before training begins (flag for the Mother Agent / implementation team).
- **Phase-0 gate:** gain-error injection code must exist and be exercised in training data before any gain-robustness claim is made (§21).
- **Fine-tune step (new, added in Revision 2):** none required for IABR-Net itself (unlike Candidate C, which requires a mandatory post-pruning fine-tune) — IABR-Net is trained end-to-end in one pass.
- **Checkpointing:** if run on Kaggle, "Save Version → Save & Run All (Commit)" is mandatory — this project has already lost two full training runs' worth of weights (the SNR-aware pruned student) to this exact workflow gotcha.

---

## 21. Baselines

| Baseline | Source | Role |
|---|---|---|
| Teacher (64-block ResNet, pretrained) | `baseline_models/` | Upper-bound reference (depth/pretraining-mismatched — always cite this caveat) |
| Magnitude-pruned Student | `baseline_models/` | Compression-track comparison point |
| FNO (8-block, scratch) | `RESULTS.md` | Architecture-screening comparison point |
| GridlessUnfold (8-block, scratch) | `RESULTS.md` | Precision-mechanism comparison point |
| Classical DFT-SIC (zero learned params) | Prior-work §2 | **Critical baseline** — IABR-Net's whole motivation rests on beating this 0-param, Pd-0.884 reference by more than its ≈193K learned parameters are "worth" |
| Capacity-matched control (new, required) | §24 | Same-parameter-count module, no impairment-specific loss — isolates whether gains come from impairment-adaptivity or added capacity |
| Front-end-swap control (new, required) | §24 | Same trunk, raw-dense (262,656-param) front end instead of DFT+IABC-v2 — empirically tests the parameter-efficiency claim rather than leaving it as arithmetic |
| SNR-aware pruned student (recovery, not yet done) | Prior-Work §4 | Still-missing full-bank number; should be recovered before claiming magnitude pruning is this project's best compression method |

---

## 22. Complete Experiment Matrix

| Experiment | Baseline | Proposed | Metric | Expected outcome | Purpose |
|---|---|---|---|---|---|
| E0a — Phase-0 code build | — | Gain-error injection in `tvt_data_generation_v3.py` | code exists, unit-tested | Runnable | Unblocks all gain-robustness claims |
| E0b — Model instantiation | — | IABR-Net | `model.count_params()` | ≈193,104 [TO MEASURE] | Replace estimate with a measured number |
| E1 — Capacity-matched control | Same-param, no-impairment-loss module | IABR-Net | Pd, RMSE | [HYPOTHESIS] IABR-Net > control at high impairment | Attribute gains to adaptivity, not capacity |
| E2 — Front-end-swap control | Raw-dense (262,656p) front end + same trunk | DFT+IABC-v2 front end + same trunk | Params, FLOPs, Pd | [HYPOTHESIS] comparable Pd at far fewer front-end params | Validate the parameter-efficiency claim empirically |
| E3 — In-distribution SNR sweep | Teacher, Student, FNO | IABR-Net | PD (paper-style + strict), RMSE | [TO MEASURE] | Direct, HIGH-comparability match to Meneses-Albalá's 8-point grid |
| E4 — Phase-error sweep δ∈{0,1,2,5}° | Teacher | IABR-Net | PD, RMSE | [TO MEASURE] | HIGH comparability to Lloria/Meneses |
| E5 — Gain-error sweep {0,0.5,1,2,4}dB (post-E0a) | none exists | IABR-Net | PD, RMSE | [HYPOTHESIS] | Closes Meneses-Albalá's own named gap — first in the field |
| E6 — OOD phase tail {10°,15°} | none exists | IABR-Net | PD, RMSE | [HYPOTHESIS: degrades] | Genuine distribution-shift test, `[GENUINE GAP]` |
| E7 — Path-count sweep L∈{1..6}, OOD {7,8,10} | Teacher | IABR-Net | PD, pairing-error rate | [HYPOTHESIS] | Tests path-token pairing at/near capacity |
| E8 — Angular separation Δθ down to 1° | — | IABR-Net | PD, RMSE | [HYPOTHESIS: collapses <5–10°] | Licensed by Naoumi's own named gap; resolution-limit caveat applies |
| E9 — Latency, NN-only vs end-to-end | Teacher, Student, FNO (never benchmarked before) | IABR-Net | ms, p50/p99 | [TO MEASURE] | Closes Gap 8, first for this task family |
| E10 — Notebook-4-style nuisance-path stress test | Teacher, magnitude-pruned Student | IABR-Net | Principal-path Pd under nuisance | [HYPOTHESIS] | Reuses the project's one existing beat-the-teacher robustness result |
| E11 — Multi-seed variance | Teacher | IABR-Net | Pd std across seeds | [TO MEASURE] | Closes Gap 10, unmeasured anywhere in the corpus |

---

## 23. Ablation Study (plan)

1. **− IABC-v2** (revert to fixed, uncorrected DFT front end), evaluated across the Tier-1 phase/gain grid — isolates IABC-v2's real contribution.
2. **− crop-refinement heads** (coarse heatmap + blob detector only) vs. full model, with **P95 tail error reported specifically near end-fire angles** — this is the direct test of whether IABR-Net's local refinement escapes any part of the diagnosed Gap-5 ceiling, or fully inherits it.
3. **− SE / `L_SE_snr` auxiliary loss** — tests A3's "attention activations correlate with SNR" hypothesis on real data rather than leaving it a hypothesis.
4. **Capacity-matched control** (§21/§22 E1) — the single most important ablation; without it, "impairment-adaptivity helps" and "adding 193K learned parameters to a 0-param baseline helps" are indistinguishable claims.
5. **Front-end-swap control** (§21/§22 E2) — empirically grounds the parameter-efficiency claim currently resting on arithmetic alone.
6. **λ1–λ4 loss-weighting sensitivity sweep** — minimum grid `λ3∈{0,0.1,0.5,1.0}`, `λ4∈{0,0.1,0.3}` — currently "to be tuned" with zero characterization, flagged explicitly as a gap by adversarial review.

---

## 24. Robustness Study (plan)

Following Agent-R3's tiered plan, adapted to IABR-Net specifically:

**Tier 1 (must run, cheap, literature-anchored):** SNR sweep [−15,24]dB (5dB steps, HIGH comparability to Meneses-Albalá); phase error δ∈{0,1,2,5}° (HIGH comparability to Lloria/Meneses); path count L∈{1..6} (HIGH comparability to Lloria/PIMRC2024).

**Tier 2 (moderate cost, project-novel):** angular separation Δθ∈{30°…1°} (licensed by Naoumi's own named gap; below ~5–10° treat as resolution-limit-diagnostic, not a defect); unseen-SNR extrapolation tails {−25,−20,30,35}dB.

**Tier 3 (genuine literature gaps, new physics code required):** gain mismatch {0,0.5,1,2,4}dB — **the literature-first, blocking-on-Phase-0 experiment this entire candidate is designed around**; unseen path count L∈{7,8,10} and unseen phase severity {10°,15°}.

**Tier 4 (stretch/out of scope):** RF-chain/calibration error, coherent/correlated sources, mobility/Doppler — no literature anchor or system-model support today.

**Reporting requirement (Gap 4):** both paper-style and strict per-trial PD reported at every condition, every cross-paper comparison carrying its comparability level, and a catastrophic-degradation threshold (>50% relative Pd drop, `[ASSUMPTION — needs explicit sign-off, no paper defines one]`) flagged rather than silently applied.

---

## 25. Efficiency Study (plan)

1. Real `ptflops`/`thop`/manual-instrumentation FLOP measurement for IABR-Net and for the **entire existing baseline registry** (Teacher, Student, FNO, GridlessUnfold) — none of these have ever been FLOP-measured; this is a zero-new-modeling-work, high-literature-value addition (closes Gap 1).
2. Model-size-on-disk (MB) for all models — trivial, not currently tabulated anywhere.
3. Peak memory (RAM/VRAM) instrumentation — not measured anywhere in the project; needed to satisfy the research objective's explicit "memory" axis.
4. Loss-weighting-driven training-time cost (Phase-0 gain-error code, once built, adds training-data-generation time — should be profiled, not assumed free).

---

## 26. Hardware/Latency Study (plan)

1. **NN-only forward latency**, generalizing the existing `benchmark_latency()` from single-model (ResNet/U-Net/PIA-Net) use to the full baseline registry + IABR-Net, with batch-size sweep (currently fixed at 1) and p50/p99 reporting (currently mean±std only).
2. **End-to-end pipeline latency** — NN forward pass **plus** the CPU-side, OpenCV-based blob-detection/peak-search step, measured **separately and never merged** with (1). This distinction is universally ignored in the reviewed literature and is one of this project's cleanest, cheapest differentiators.
3. **Real edge hardware** (a Jetson-class device, matching Meneses-Albalá 2026's own methodology) — currently entirely absent from this project; even a first-pass measurement on the already-trained Teacher/Student/FNO checkpoints would be a literature-first for this specific architecture family.
4. Throughput (samples/sec) under a batch-size sweep — not measured anywhere today.

---

## 27. Expected Results (hypotheses)

All items below are **[HYPOTHESIS]**, not results. None has been run.

- **[HYPOTHESIS]** IABR-Net's PD/RMSE at matched SNR will fall between the classical DFT-SIC baseline (Pd 0.884 @20dB, 0 params) and the full teacher (Pd 0.925 @20dB, 469,393 params), given its intermediate ≈193K-param capacity — direction plausible, magnitude unknown.
- **[HYPOTHESIS]** IABC-v2's per-element correction will measurably reduce PD degradation under phase error beyond 5° relative to the uncorrected classical front end, but will **not** generalize to gain error until Phase-0 code exists and is used in training.
- **[HYPOTHESIS]** The capacity-matched control (E1) will show that *some*, not all, of IABR-Net's advantage over the 0-param classical baseline is attributable to impairment-adaptivity specifically, rather than added capacity alone — this is the single most likely place for the headline claim to be partially, not fully, vindicated.
- **[HYPOTHESIS]** IABR-Net will **still exhibit a meaningful fraction of the diagnosed Gap-5 end-fire ceiling**, because it retains a discrete P×Q=16×16 coarse grid; the crop-refinement heads may reduce but are unlikely to eliminate this effect, motivating the Phase-2 fusion with Candidate A's set-prediction head (§31).
- **[HYPOTHESIS]** NN-only latency will be substantially lower than the teacher's (proportional to its ~2.4× smaller parameter count and much lower FLOPs), but end-to-end latency (including CPU-bound blob search) may narrow that gap considerably — untested.

---

## 28. Novelty Claim (honest self-classification)

**PARTIALLY NEW, UNCERTAIN on the most load-bearing component.**

At the **pattern level**, "a small learned network corrects/regularizes a classical processing stage's inputs" is an established paradigm (SubspaceNet is a verified instance of it, in the opposite direction — DL then classical, vs. IABR-Net's classical then DL). This is **KNOWN COMBINATION** at the pattern level, per Agent-R7's own explicit test.

At the **specific-application level** — a fixed DFT beamspace reduction, combined with a synthetic-paired-consistency-trained, identifiability-aware per-element correction module specifically targeting Meneses-Albalá's own named gain-error gap, combined with a path-token-coupled coarse+local-refine dual head preserving joint AoA/AoD pairing, applied to the joint 2D task — is **not present in the 14-paper corpus**. Agent-R7 rates this direction **UNCERTAIN**, not confirmed novel, pending a dedicated search specifically for "classical front-end, learned impairment correction, joint AoA/AoD" — this search has **not** been run, and this report does not claim it has.

The measurement-rigor bundle this project can add regardless of architecture — real inference latency, strict per-trial PD, multi-seed variance, all reported together for the joint 2D task — remains the **single cheapest, most defensible, near-zero-risk novelty contribution** available (Agent-R7's own bottom line), independent of which architecture is ultimately selected.

**This report does not claim NEW anywhere.**

---

## 29. Threats to Validity

1. **Internal validity — capacity confound.** Without the capacity-matched control (§21 E1), any observed IABR-Net gain over the 0-param classical baseline cannot be attributed to impairment-adaptivity vs. added capacity.
2. **Internal validity — Gap-5 ceiling may still be present.** IABR-Net retains a discrete coarse grid; results showing high PD could still mask a hidden end-fire tail failure unless P95/end-fire-specific reporting (§23 item 2) is included.
3. **Construct validity — PD convention.** The inherited paper-style PD metric silently drops missed-detection trials, inflating results relative to a strict per-trial convention; every table must report both (Gap 4).
4. **External validity — synthetic-only.** Every number in this evidence chain, this project's own included, comes from simulation; zero of the Tier-1 papers and none of this project's own experiments have been validated on real hardware/real channel data for the joint 2D task.
5. **External validity — corpus incompleteness.** At least four on-topic papers outside the 14-paper corpus were identified but not read (§3/§11); novelty and "the field hasn't tried X" claims are correspondingly weaker than they appear.
6. **Reproducibility — code/paper discrepancies in the base paper itself.** Lloria TVT2026's own released code cannot reconstruct its plotted ResNet-512 variant, and the U-Net's actual kernel size/bottleneck usage is unverified without inspecting the `.h5` weight file directly (Contradictions 2, 3, 13) — any number sourced from that paper as a "reproducible baseline" carries this caveat.
7. **Statistical validity.** No paper in the corpus, and no experiment in this project to date, reports cross-seed training variance; single-run numbers throughout this report (including all of IABR-Net's own future numbers) should be read as point estimates pending Experiment E11.

---

## 30. Reproducibility Checklist

- [ ] Build gain-error injection code in `tvt_data_generation_v3.py` (Phase-0, blocking) and unit-test against a known perturbation.
- [ ] Instantiate IABR-Net and replace every estimated param figure in this report with `model.count_params()`.
- [ ] Add `ptflops`/`thop` (or manual layer-wise accounting with an explicit hook for any FFT ops, if later fusing in Candidate A's FNO stage) — none exists in-repo today.
- [ ] Add a `strict: bool` parameter to `evaluate_on_bank` rather than mutating its existing behavior, preserving every number in `baseline_models/README.md` as reproducible unchanged.
- [ ] All new robustness banks generated via the existing, unmodified `validation_data_generator` — confirm angular-separation control is actually exposed by the generator signature before scheduling that experiment as "cheap."
- [ ] Any Kaggle run: "Save Version → Save & Run All (Commit)" before ending the session — this exact omission has already destroyed two full training runs' worth of weights.
- [ ] Recover or re-run the SNR-aware pruned student at full 8000-sample scale before citing it as this project's best compression method.
- [ ] Fix random seeds and log them; report multi-seed variance (E11) before any single-seed number is presented as final.
- [ ] Every cross-paper number quoted must carry its comparability tag (HIGH/MEDIUM/LOW/NOT COMPARABLE) from §4/§9.

---

## 31. Final Research Roadmap

1. **Phase 0 (blocking):** build gain-error injection code; instantiate IABR-Net and measure its real param/FLOP/memory footprint; run the capacity-matched and front-end-swap controls (§22 E1–E2).
2. **Phase 1 (Tier-1 robustness + baseline registry FLOP/latency backfill):** run SNR/phase/L sweeps on IABR-Net alongside a first-ever FLOP and NN-only/end-to-end latency measurement for the **entire existing baseline registry** (Teacher, Student, FNO, GridlessUnfold) — this alone closes Gaps 1 and 8 regardless of how IABR-Net itself performs.
3. **Phase 2 (Gap-5 fusion, contingent on Phase 1 results):** if Phase 1's end-fire P95 analysis (§23 item 2) confirms IABR-Net still substantially inherits the diagnosed ceiling, fuse Candidate A's shared-query set-prediction head onto IABR-Net's efficient DFT+IABC-v2+trunk front end, in place of its own coarse-heatmap+crop-refine head — combining B's efficiency/robustness story with A's root-cause fix for the single most evidence-convergent open question in this evidence base. Run the full DETR-vs-crop-refine ablation before committing.
4. **Phase 3 (genuine-gap robustness, post-Phase-0):** gain-error sweep, phase>5° extrapolation, L∈{7,8,10} OOD, angular separation to the array's resolution limit.
5. **Phase 4 (rigor bundle, can run in parallel with any phase above):** strict PD reporting, multi-seed variance, real-hardware latency — the zero-architectural-risk, near-zero-cost novelty contribution identified independently by Agent-R7.
6. **Housekeeping, do in parallel:** recover/re-run the SNR-aware pruned student at full scale; run the still-pending SIREN/WindowAttention LR+warmup sweep (this could reopen the attention-viability question closed prematurely by confounded runs); resolve whether PROJECT_STATUS.md's and BUJHO_SHOHOJ_VABE.md's differing PIA-Net physics-issue descriptions are the same bug before any PIA-Net work resumes.

---

### Experiment Matrix (repeated per protocol format)

| Experiment | Baseline | Proposed | Metric | Expected outcome | Purpose |
|---|---|---|---|---|---|
| SNR sweep | Teacher/Student/FNO | IABR-Net | PD, RMSE | [HYPOTHESIS] intermediate between DFT-SIC and Teacher | Core accuracy characterization |
| Phase-error sweep | Teacher | IABR-Net | PD, RMSE | [HYPOTHESIS] IABC-v2 reduces degradation vs. uncorrected DFT | Tests IABC-v2's designed function |
| Gain-error sweep (post-Phase-0) | none in corpus | IABR-Net | PD, RMSE | [HYPOTHESIS] | Closes Meneses-Albalá's own named gap |
| Capacity-matched control | Same-size, no-adaptivity module | IABR-Net | ΔPd attributable | [HYPOTHESIS] partial attribution to adaptivity | Resolves central attribution confound |
| End-fire density probe | Blob-detector-only variant | Full IABR-Net | P95 tail error | [HYPOTHESIS] partial ceiling relief | Direct Gap-5 test |
| Latency (NN-only, e2e) | Never measured for any model | IABR-Net + full registry | ms, p50/p99 | [TO MEASURE] | Closes Gap 8, literature-first |

### Condition / accuracy table

| Condition | Baseline (DFT-SIC) | Baseline (Teacher) | Proposed (IABR-Net) | AoA RMSE | AoD RMSE | PD |
|---|---|---|---|---|---|---|
| SNR=20dB, in-distribution, no impairment | Pd 0.884 (measured) | Pd 0.925 (measured) | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] |
| SNR=0dB, in-distribution | Pd n/a (not separately reported) | Pd 0.639 (measured, TVT2026-analog point) | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] |
| δ=5° phase error | not tested | tested by Lloria (figure-only) | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] |
| γ=4dB gain error (post-Phase-0) | not tested anywhere | not tested anywhere | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] |
| L=8 (OOD path count) | not tested | not tested (L≤6 only) | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] |

### Model comparison table

| Model | Params | FLOPs | Memory | Latency | RMSE | PD |
|---|---|---|---|---|---|---|
| Teacher (64-block ResNet) | 469,393 (measured) | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | ~0.253° @20dB (measured) | 0.925 @20dB (measured) |
| Magnitude-pruned Student | 314,513 (measured) | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | 0.6956 mean (measured) |
| FNO (8-block, scratch) | 334,321 (measured) | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | 0.6235 mean (measured) |
| Classical DFT-SIC | 0 | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | 0.884 @20dB (measured) |
| **IABR-Net (proposed)** | ≈193,104 (estimate) | ≈95.3M (estimate) | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] | [TO MEASURE] |

---

## FINAL RESEARCH QUESTION

**Can a beamspace-domain, per-element impairment-correction front end paired with a shallow residual trunk and a joint-pairing-preserving dual-head estimator match a substantial fraction of a 64-block ResNet teacher's joint AoA/AoD accuracy at roughly 41% of its parameters and orders of magnitude fewer FLOPs, while extending robust operation into two hardware-impairment regimes — amplitude/gain mismatch and phase error beyond 5° — that no paper in the reviewed joint-AoA/AoD literature has tested, and does closing this project's own diagnosed pixel-quantization ceiling ultimately require replacing its retained discrete coarse grid with a differentiable set-prediction head, as this project's own ceiling-test and diffusion-sharpening evidence suggests?**

---

## ADDENDUM — Orchestrator verification after the workflow finished (2026-09-23)

Added after the 34-agent run completed, from direct checks against this repo's code. Per protocol §21, the agents' text above is left unchanged and the corrections are recorded here.

### A1. The input representation of IABR-Net is described wrongly, but the design survives once corrected [EVIDENCE: CODE]

§16 and §18 say the input is a "raw complex antenna-domain signal Y" and that stage 0 applies a forward DFT, `F_codebook^H · Y`. This project's generator does not work that way (`dldoa_dataset_generation.py`):
- The observation is `Y = Wᴴ H F + Z`, where `F` and `W` are **DFT codebooks** (lines 137, 165–168, 201–204). So `Y` is already beamspace. Raw per-antenna signals are never observed, which is the whole point of the analog architecture.
- The 64×64×2 network input is this 16×16 `Y`, upsampled ×4 by nearest neighbour (lines 481–485). Strided sampling undoes that exactly, so the real 16×16 `Y` is recoverable at no cost.
- Hardware phase error is **per antenna and shared across all beams** (`phase_error` has size `nt` / `nr`, lines 161 and 197). That gives `F = D_t F_ideal` and `W = D_r W_ideal`, so `Y = W_idealᴴ (D_rᴴ H D_t) F_ideal + Z`.

**Correction.** With the eval bank's square codebooks (P=nt=16, Q=nr=16), stage 0 should be the fixed **inverse** codebook transform `H̃ = W_ideal^{-H} Y F_ideal^{-1}`. It is still 0 params. It maps the observation back to the antenna domain, where the impairment is exactly the two diagonal phase matrices `D_r`, `D_t`: 16 Rx + 16 Tx = the 32 per-element parameters IABC-v2 already targets. Per-element correction is then physically identifiable, up to a common phase. After correcting, re-apply the forward codebook and hand the trunk a 16×16 beamspace map. For training samples with P=32 > nt, use a pseudo-inverse instead of an inverse. Gain mismatch, once the Phase-0 code exists, has the same diagonal structure.

**Why this matters.** None of the 6 adversarial reviews raised this. As written, the IABC-v2 inputs ("per-element mag/phase of Y") do not exist in the data. With the correction, the module is better founded than the report states. This must be fixed in the implementation before E0b.

### A2. Arithmetic check of IABR-Net's efficiency estimates [INFERENCE — hand-verified, not measured]
- Trunk: 20 convs, each 3×3, 32→32, at 16×16 → 20 × 16·16·9·32·32 = 47.2M MACs ≈ **94.4M FLOPs**. Matches the report's ≈95.3M.
- Trunk params: 10 × [2 × (9·32·32 + 32) + BN] ≈ 187.5K. Consistent with the ≈193K total.
- Teacher (not measured anywhere in this project): input 64×64 → stride-2 transposed conv → 128×128 → 64 blocks × 2 convs × 128·128·25·12·12 MACs ≈ 7.55 GMACs ≈ **15.1 GFLOPs**. That makes IABR-Net ≈160× cheaper, so the final research question's "orders of magnitude fewer FLOPs" is plausible (about 2 orders). It is still an estimate until E0b and §25 item 1 are run.

### A3. A trade-off the report does not tie together [INFERENCE]
Most of IABR-Net's FLOP saving comes from dropping the teacher's 256×256 output grid and working on a 16×16 coarse grid. Gap 5 (the Pd≈0.972 ceiling) is a pixel-quantization / discrete-grid problem, so a coarser grid is the same design choice pushing the wrong way on it. The efficiency gain and the ceiling risk are **coupled, not independent**. Ablation §23 item 2 (end-fire P95 with and without crop refinement) is therefore the deciding experiment for IABR-Net, not a secondary check.
