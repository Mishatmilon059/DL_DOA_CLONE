# MASTER SYNTHESIS REPORT — Paper Head Agent
## Multi-Agent Research Loop: Joint AoA/AoD Estimation Literature Review

**Scope:** 6 Tier-1 papers (independently specialist-extracted + independently verified), 8 Tier-2 papers (grouped extraction), and this project's own prior-experiment analysis (`D:\ai_ml_project`). All Tier-1 verification agents returned VERIFIED or PARTIALLY_VERIFIED on every load-bearing numeric claim they spot-checked; no fabricated Tier-1 numbers survived verification except one flagged Tier-2 item (BeamSeek latency figures — see §6).

---

## 1. MASTER LITERATURE MATRIX

| # | Paper | Joint AoA/AoD? | Architecture family | Dataset | SNR range | Key accuracy metric + value | PD (if reported) | Params | FLOPs / latency | Robustness tested | Hardware tested | Public code? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Lloria et al., TVT 2026** (base paper) | **YES**, genuine, single 2D output (Eq. 9–11) | ResNet (64 RB) **and** U-Net (Wave-U-Net) | Synthetic, single-user mmWave MIMO, ULA 16/32 | Text: −10…25 dB; **code: −15…24 dB** (discrepancy) | PD/RMSE @20dB, δ=0°: ResNet .925/.253, U-Net .945/.230 | Yes (Table II, exact) | NOT REPORTED (paper); project's own reproduction measured 469,393 (ResNet) | NOT REPORTED numerically — only asymptotic O(PQ) | SNR sweep, L∈{1..6}, phase error δmax∈{1,2,5}° | NOT REPORTED (software sim only) | YES (GitHub; training code absent, ResNet-512 not reconstructible — see §6) |
| 2 | **Lloria et al., PIMRC 2024** (prototype) | YES, genuine | ResNet (64 RB) + PH peak search (+ optional LM refine) | Synthetic, same family, ULA 16/32 | −10…25 dB | PD/RMSE vs SNR, figure-only (no table) | Figure-only, no exact values | NOT REPORTED | Asymptotic O(P²logP)/O(PQ log PQ), no concrete numbers | SNR sweep, L∈{1..6} (no phase-error test) | NOT REPORTED | Not mentioned |
| 3 | **Naoumi et al., JSTSP 2024** (bistatic ISAC) | YES, genuine (θ at radar, φ at BS, jointly output) | Complex-valued MLP (3 hidden layers) | Synthetic bistatic ISAC, Nt=8, Nr=10 | Train {5,10,...,40}dB; test −5…30dB | MSE (rad²) vs SNR; ~9dB gap to CRB at 10⁻⁶ MSE target; underperforms 2D-algo beyond 13dB | Not reported (no PD-style metric) | NOT REPORTED numerically (formulas only) | Closed-form; **6.5×/10.3× fewer mults** than 2D algo at Nr=8/16 (headline) | SNR, #targets-per-peak (1,2,5) | NOT REPORTED | Claimed (github.com/salmane-s9/Bistatic_ISAC), not verified |
| 4 | **Naoumi et al., GCWkshps 2023** (earlier/shorter) | YES, genuine | Complex-valued MLP (3 hidden layers) | Synthetic, Nt=8/Nr=10, narrow angle window [20°,40°] | Train −5…30dB; eval −9…30dB | MSE (rad²) vs SNR; reaches ~10⁻⁶ at SNR≥9dB (exact text quote) | Not reported | **8,584** (explicit, exact) | 4.321 MMACs fwd pass; 1.875 TMACs total training (exact) | SNR only; well-separated-targets assumption untested at failure | NOT REPORTED | Not mentioned |
| 5 | **Gupta et al., TCOMM 2025** (Bayesian ISAC) | **NO** — explicitly avoids joint estimation (monostatic AoA=AoD collapse; AoD at BS deliberately eliminated via UE-side ZF beamformer) | Classical: Sparse Bayesian Learning (EM) + MUSIC + Alternating Minimization — **not DL** | Synthetic, 2 system configs (Table II) | −5…25 dB | NMSE (RCS) vs SNR; 5dB/15dB gain vs OMP/FOCUSS (exact text quotes) | N/A (thresholded imaging, no PD/ROC curve) | N/A (no NN) | Big-O only (cubic in grid sizes: O(Gτ³Gν³Gθ³)); no concrete FLOPs | SNR, system scale, beamforming trade-off δ; **no hardware-impairment/distribution-shift test** | NOT REPORTED (pure simulation) | Not mentioned |
| 6 | **Meneses-Albalá et al., J.Supercomputing 2026** (edge/energy) | YES (reuses Lloria's U-Net verbatim, single 2D map) | U-Net (reused from [7]) + partial-encoder-freeze fine-tuning | Synthetic, inherits Lloria's system, implied 16×16 codebook | −10,−5,0,5,10,15,20,25 dB (8 pts) | RMSE/PD Table 2; γmax=5°,15dB: RMSE 0.28→0.27 (3.6% rel. gain, exact quote); γmax=5,25dB: PD 0.93→0.94 (+1pp) | Yes (Table 2, exact, 0.22–0.95 range) | NOT REPORTED | NOT REPORTED for inference; training energy 0.28–0.41 J/sample on real hardware | Phase error γmax∈{1,2,5}° (PRIMARY focus) + SNR + L + codebook | **YES — real Jetson Orin Nano** (only Tier-1 paper with real edge hardware) | Base weights on GitHub (DL_DOA repo); this paper's own code not mentioned |
| 7 | Koh & Lee 2026 (benchmark survey) | NO (AoA/DoA only, BLE indoor) | Survey of 11 models: DNN/CNN/A-CRNN/GRU/DAE+MUSIC | Ray-traced + real BLE IQ (TI BOOSTXL-AoA) | Not centrally reported (BLE) | Best CNN MAE=2.98° (39k FLOPs/19.8k params); GRU <1.3° MAE real-world | N/A | 19.8k (best CNN) | 39k FLOPs (best CNN) | Cross-model, sim-vs-real domain shift (major finding) | YES, real BLE hardware | Not stated |
| 8 | Lim et al. 2021 (LSTM beam tracking) | Partial (tracks existing angles over time, not joint estimation from raw signal) | LSTM + sequential Bayesian (EKF/UT) | Synthetic mobility, IMU-assisted, 3M examples | Not primary axis | BER gain vs EKF: 1dB→3dB→10dB as mobility aavg 0.1π→0.4π increases | N/A | Hidden size 32 | NOT REPORTED | **Mobility/Doppler** (unique) | NOT REPORTED | Not stated |
| 9 | BeamSeek (arXiv 2025) | NO (1D azimuth DOA only) | MLP (SwiGLU-gated FFN, hidden 384) | Real 60GHz, NSF COSMOS testbed | Low/Mid-Low/Mid-High bins | DOA error 3.86–21.45° (SNR/beam-count dependent); up to 8° gain over correlation baseline | N/A | NOT REPORTED exactly | **NOT REPORTED** — see §6 discrepancy flag | SNR, #scan beams; single-RF-chain hardware constraint | YES, real 60GHz testbed | Not stated |
| 10 | Sub6GHz CNN/UNet (arXiv 2025) | NO (AoA/AoD are classical pre-estimated inputs, not DL output) | CNN (9-layer) & UNet (2enc/2dec) | Synthetic, 8×8 MIMO, 500k/500k | [−20,10] dB | SE gain 3.35%/4.03% vs MRC; 159%/161% vs in-band-only | N/A | NOT REPORTED | NOT REPORTED | SNR, K-factor, AoA/AoD range | NOT REPORTED | Not stated |
| 11 | SubspaceNet (Shmuel et al. 2025 TVT) | NO (1D DOA, single ULA) | CNN-DCNN autoencoder → classical subspace (hybrid model-based DL) | Synthetic; coherent/broadband/few-snapshot | Low-SNR emphasis | RMSPE ~60× better than classical Root-MUSIC (12.48°→0.20°, M=2 coherent) | N/A | **41,761** (vs >21M black-box CNN) | NOT REPORTED explicitly | Coherent sources, broadband, few snapshots, low SNR, **array miscalibration** — most comprehensive in set | NOT REPORTED | Not stated |
| 12 | Robust DoA DAE-DNN (Chen et al. 2022) | NO (1D single-array) | Denoising autoencoder + DNN classifier cascade | Synthetic, mutual-coupling perturbed | >0dB emphasis | Own-paper RMSE<0.5°; **independent re-impl. (Koh&Lee 2026): MAE 30°(sim)/>70°(real)** | N/A | NOT REPORTED | O(ΣnL-1·nL), lower than MUSIC | Mutual coupling (ρ≤0.8), low SNR, unknown source count | NOT REPORTED | Not stated |
| 13 | MIMO-Radar-Aided V2X (Huang et al. 2021) | Partial/classical (angle+Doppler estimated once, reused) | Classical MIMO-radar signal processing — no DL | Not extracted | Not extracted | Not extracted (pilot-overhead reduction claim) | N/A | N/A | N/A | Mobility/time-varying (V2X) | NOT REPORTED | Not stated |
| 14 | Tensor PARAFAC (Du et al. 2021 TSP) | Partial/classical (joint channel+symbol via tensor decomp) | Tensor/PARAFAC — no DL | Not extracted | Not extracted | Not extracted | N/A | N/A | N/A | Time-varying massive MIMO | NOT REPORTED | Not stated |

### Cross-paper comparability notes (per "no false comparison" rule)

| Comparison | Level | Why |
|---|---|---|
| Lloria TVT2026 U-Net vs Meneses-Albalá 2026 base model (RMSE/PD @ SNR=20dB) | **HIGH** | Same U-Net, same data generator, same eval protocol, same array config — numbers align closely (.945/.230 vs .95/.23 at δ/γ=1°) |
| Lloria TVT2026 vs Lloria PIMRC2024 (PD/RMSE trend) | **MEDIUM** | Same authors/system model, but PIMRC gives figure-read values only (no table), and peak-detector differs (PH vs blob) — trend-comparable, value-comparable only approximately |
| Naoumi JSTSP2024 vs GCWkshps2023 (MSE rad² vs SNR) | **HIGH** | Identical system config (Nt=8,Nr=10), same architecture family, journal extension of same lineage |
| Naoumi (either) vs Lloria (either) | **LOW** | Different metric units (MSE rad² vs RMSE degrees+PD), different antenna counts, different system framing (bistatic ISAC vs single-user analog BF), different SNR-range definitions |
| Gupta 2025 vs any Lloria/Naoumi paper | **NOT COMPARABLE** | Different task (NMSE of sparse RCS vector vs angle RMSE/PD), no joint AoA/AoD output, non-DL |
| SubspaceNet vs any joint-AoA/AoD paper | **NOT COMPARABLE** | 1D DOA vs 2D joint estimation; RMSPE metric not equivalent to RMSE+PD |
| BeamSeek vs any full-array paper | **NOT COMPARABLE** | Single-RF-chain beam-switching hardware constraint changes the estimation problem fundamentally; 1D only |
| Koh&Lee 2026 vs mmWave papers | **NOT COMPARABLE** | BLE narrowband indoor positioning, different frequency regime and application |
| Robust DoA DAE-DNN own-paper vs Koh&Lee's re-implementation numbers | **LOW / flagged contradiction** | Same architecture family, but the ~60–140× discrepancy (0.5° vs 30–70°) reflects real dataset/domain mismatch, not a transcription error — see §6 |

---

## 2. ARCHITECTURE COMPONENT MATRIX

| Paper | CNN | ResNet | U-Net | MLP | Transformer/Attention | Complex-valued NN | TDA/topological | Tensor/PARAFAC | Classical subspace | Other |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| Lloria TVT2026 | — | ✓ | ✓ | — | — | — | — | — | — | Blob detection (non-learned) |
| Lloria PIMRC2024 | — | ✓ | — | — | — | — | ✓ | — | — | LM Gaussian-fit refinement |
| Naoumi JSTSP2024 | (mentioned, undetailed) | — | — | ✓ | — | ✓ | — | — | — | 2D parametric algorithm (classical) |
| Naoumi GCWkshps2023 | — | — | — | ✓ | — | ✓ | — | — | — | — |
| Gupta TCOMM2025 | — | — | — | — | — | — | — | — | ✓ (MUSIC) | Sparse Bayesian Learning (EM), Alternating Min |
| Meneses-Albalá 2026 | (ResNet mentioned qualitatively) | — | ✓ (reused) | — | — | — | — | — | — | Encoder-freeze fine-tuning |
| Koh & Lee 2026 | ✓ | — | — | ✓ (DNN) | ✓ (A-CRNN) | — | — | — | ✓ (DAE+MUSIC hybrid) | GRU (RNN) |
| Lim et al. 2021 | — | — | — | — | — | — | — | — | — | LSTM + EKF/UT |
| BeamSeek | — | — | — | ✓ (SwiGLU-FFN) | — | — | — | — | — | — |
| Sub6GHz CNN/UNet | ✓ | — | ✓ | — | — | — | — | — | — | — |
| SubspaceNet | ✓ (CNN-DCNN AE) | — | — | — | — | — | — | — | ✓ (MUSIC/ESPRIT/Root-MUSIC/MVDR) | — |
| Robust DoA DAE-DNN | — | — | — | ✓ (DNN classifier) | — | — | — | — | — | Denoising autoencoder |
| MIMO-Radar V2X | — | — | — | — | — | — | — | — | ✓ (implied) | — |
| Tensor PARAFAC | — | — | — | — | — | — | — | ✓ | — | — |

**Notable pattern:** zero of the 6 Tier-1 papers use Transformer/Attention. This is a literature-wide gap (see §5, item 9).

---

## 3. ROBUSTNESS MATRIX

| Paper | Low SNR | Multipath | Phase/gain error | Array perturbation | Domain shift | Mobility | Hardware/edge |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Lloria TVT2026 | ✓ | ✓ (L=1–6) | ✓ (phase only, δmax=1/2/5°) | ✗ | ✗ (in-distribution only) | ✗ | ✗ (sim only) |
| Lloria PIMRC2024 | ✓ | ✓ (L=1–6) | ✗ | ✗ | ✗ | ✗ | ✗ |
| Naoumi JSTSP2024 | ✓ | ✓ (#targets/peak) | ✗ | ✗ | Partial (multi-SNR training) | ✗ (static targets) | ✗ |
| Naoumi GCWkshps2023 | ✓ | Partial (well-separated only) | ✗ | ✗ | ✗ (narrow [20°,40°] window) | ✗ | ✗ |
| Gupta TCOMM2025 | ✓ | ✓ | ✗ (explicitly untested) | ✗ | ✗ | ✓ (Doppler modeled) | ✗ |
| Meneses-Albalá 2026 | ✓ | ✓ (L sweep) | ✓ (γmax, PRIMARY focus) | ✗ (gain error flagged future work) | Partial (fine-tune across severities) | ✗ | **✓ real Jetson Orin Nano** |
| Koh & Lee 2026 | Partial | Partial (indoor NLOS) | ✗ | ✗ | ✓ (sim vs real, major finding) | ✗ | ✓ real BLE |
| Lim et al. 2021 | Partial | ✗ | ✗ | ✗ | ✗ | ✓ (PRIMARY focus) | ✗ |
| BeamSeek | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ real 60GHz testbed |
| Sub6GHz CNN/UNet | ✓ | ✗ | ✗ | ✗ | Partial (K-factor sweep) | ✗ | ✗ |
| SubspaceNet | ✓ | ✓ (coherent sources) | Partial (via miscalibration) | ✓ (explicit) | ✗ | ✗ | ✗ |
| Robust DoA DAE-DNN | ✓ | ✗ | ✗ | ✓ (mutual coupling) | ✓ (but **failed** under independent re-test) | ✗ | ✗ |
| MIMO-Radar V2X | ✗ | Partial | ✗ | ✗ | ✗ | ✓ (V2X) | ✗ |
| Tensor PARAFAC | ✗ | Partial | ✗ | ✗ | ✗ | ✓ | ✗ |

**Gap column observation:** no single paper covers more than 3 of the 7 robustness axes. SubspaceNet is the single most robustness-comprehensive paper (4 axes) but only for 1D DOA, not joint 2D estimation.

---

## 4. GAP MATRIX

| Paper | Strength | Weakness | Missing experiment | Potential improvement |
|---|---|---|---|---|
| **Lloria TVT2026** | Rigorous, verified joint 2D formulation; largest, most complete robustness grid of any Tier-1 paper; open code | No params/FLOPs/latency numbers anywhere; U-Net has 2 unresolved code-vs-paper conflicts (kernel size, dense bottleneck — see §6); P_D definition silently drops under-detected trials | No OOD robustness (unseen SNR/L/angle-sep); no gain-mismatch/array-perturbation test; no real hardware | Replace dense/flatten bottleneck (paper's own Table IV quadratic-cost term) with conv/attention bottleneck; report real params/FLOPs/latency |
| **Lloria PIMRC2024** | Established the joint problem framing + PH-based peak search; complexity formulas exact-verified | Superseded by TVT2026 on almost every axis; PH is grid-resolution-limited (authors' own admission); no exact numeric tables, figure-read only | No hardware-impairment test at all (added only in TVT2026); no params/FLOPs | N/A — mainly historical/lineage value |
| **Naoumi JSTSP2024** | Best-verified complexity-reduction headline claim (6.5×/10.3× fewer mults) in the whole set; clean complex-MLP + coarse-timing-IFFT input-reduction idea | Explicit high-SNR performance saturation (authors' own admitted open weakness); no hardware-impairment/domain-shift test; assumes fixed q per peak | No tracking/temporal-continuity (explicit future work); no real hardware validation | Fix high-SNR saturation architecturally (not "just deeper," which authors flag as complexity-costly); add impairment/domain-shift robustness |
| **Naoumi GCWkshps2023** | Exact, fully-reported param count (8,584) and MACs — best-documented complexity of any paper reviewed | Narrow [20°,40°] angle window both train/test; well-separated-targets assumption breaks under limited temporal resolution (authors' own admission) | No closely-spaced-target stress test; no hardware impairment; no wide angular range test | Learned (not purely peak-based) target-separation front end for closely-spaced targets |
| **Gupta TCOMM2025** | Rigorous Bayesian/CRB-grounded theoretical framework; exact NMSE/SE numbers, fully verified | **Not genuine joint AoA/AoD** — deliberately eliminates AoD at BS; not DL, offers no transferable architecture; cubic complexity in grid sizes | No hardware impairment, no distribution shift, no detection-probability/ROC curve despite threshold-based detection | A DL joint 2-angle estimator that beats 3D-BL's O(Gτ³Gν³Gθ³) per-iteration cost with a single forward pass is a direct, quotable novelty target |
| **Meneses-Albalá 2026** | Only paper with real edge-hardware energy/latency numbers; reproducible impairment-injection methodology; partial-layer-freeze adaptation idea | U-Net backbone explicitly conceded "larger than ResNet" by the authors themselves; gains from fine-tuning are numerically small (3.6% RMSE, 1pp PD) and materialize mainly at the most severe impairment level; no standalone inference latency, only training/fine-tuning energy | No amplitude-error robustness (explicit future work); no ray-tracing/deterministic environment; no params/FLOPs | Reuse the Jetson profiling *methodology* (not the U-Net) for a genuinely lightweight model's standalone inference latency/energy |

**Tier-2 condensed gap notes:**
- **SubspaceNet**: strength = extreme parameter efficiency (41.7k) + most comprehensive robustness set; weakness = 1D-only; missing = joint 2D extension; improvement = combine its "learn covariance surrogate, keep classical back-end" pattern with the project's Gridless-Unfold complex soft-threshold layer.
- **BeamSeek**: strength = only hardware-realistic single-RF-chain solution with real 60GHz validation; weakness = 1D azimuth only, no latency numbers actually reported (see §6); missing = joint AoA/AoD extension under the same single-RF-chain constraint.
- **Koh & Lee 2026**: strength = only paper directly demonstrating sim-to-real generalization failure at scale (methodologically valuable); weakness = BLE, not mmWave; missing = a similarly rigorous sim-vs-real benchmark for mmWave joint AoA/AoD.

---

## 5. RESEARCH GAPS FOR A LIGHTWEIGHT JOINT AoA/AoD ARCHITECTURE (evidence-tagged, cross-referenced against this project's prior experiments)

**Gap 1 — No paper reports params + FLOPs + latency + memory together for a joint AoA/AoD model.**
[EVIDENCE: Lloria TVT2026 Sec K — asymptotic only; PIMRC2024 — same; Meneses2026 — training energy only, no inference latency]. This project's own baseline registry *already has* exact, cross-model-comparable numbers (Teacher 469,393 params; magnitude-pruned Student 314,513 params; FNO 334,321 params, all evaluated on the identical 8,000-sample bank) [EVIDENCE: PROJECT `baseline_models/README.md`]. This is the single most ready-made, zero-new-work novelty claim in the whole evidence base — simply reporting these numbers already out-argues every reviewed paper's Table K/III/IV on quantitative grounds.

**Gap 2 — The dense/flatten U-Net bottleneck is the literature's own self-flagged worst component, and the project has already-executed evidence for a fix.**
TVT2026's own Table IV identifies the Dense Block as the only quadratic-in-(P'Q') term in its complexity table [EVIDENCE: Lloria TVT2026 Sec K/N]. The project's FNO screening result (spectral/conv bottleneck instead of dense, Pd 0.6235 at 334K params = 88% of teacher's Pd, from scratch, 8 blocks) is direct, already-executed evidence that a non-dense bottleneck is viable [EVIDENCE: PROJECT `RESULTS.md`]. This connects a literature-identified gap directly to an existing project result rather than treating it as separate.

**Gap 3 — Zero papers in the Tier-1 set test genuine out-of-distribution robustness (unseen SNR/L/angle-separation ranges).**
[EVIDENCE: Lloria TVT2026 Sec J — test distribution = train distribution exactly; Naoumi GCWkshps2023 — narrow fixed [20°,40°] window both train/test]. Meneses2026's impairment fine-tuning is the closest approach to a domain-shift test, and even it shows the base model is "already robust" to mild impairments (small gains only at the most severe level) [EVIDENCE: Meneses2026 Table 2]. The project's own novel 4th-path nuisance-interference robustness test (Notebook 4: 200 scenes, 2 SNR levels, 3 nuisance-power levels) is the *only* documented beyond-the-paper's-own-protocol robustness experiment in this entire evidence base, and it already surfaced a non-trivial, non-obvious finding (pruned student beating teacher under interference stress) [EVIDENCE: PROJECT `PROJECT_STATUS.md`, Notebook 4]. This is strong existing proof that a project-designed OOD/robustness protocol is both novel relative to the literature and already tractable.

**Gap 4 — The base-paper family's P_D metric is non-standard (drops under-detected trials rather than penalizing them), and no other paper even attempts a comparable detection-probability metric.**
[EVIDENCE: Lloria TVT2026 code audit item 9 — `continue`d trials excluded from both numerator and denominator, likely inflating reported PD]. This project's own evaluator inherits this exact convention [EVIDENCE: PROJECT §5]. A lightweight-architecture paper reporting *both* the paper-style and a strict per-trial PD (missed detections counted as 0) would be a low-cost, high-credibility differentiator that no paper in the set currently provides.

**Gap 5 — The diagnosed Jacobian-singularity/pixel-quantization Pd ceiling (~0.972) is a project-only finding absent from every reviewed paper's own limitations sections, despite all 3 Lloria-family papers sharing the identical heatmap+peak-search paradigm.**
[EVIDENCE: PROJECT ceiling test + diffusion-sharpening ablation — the latter directly ruled out blur as the cause, confirming pixel-quantization/location error near end-fire angles]. None of TVT2026/PIMRC2024/Meneses2026 diagnose this specific failure mode, even though they share the identical output representation. `My_Proposed_Architecture.md`'s Stage 3 (differentiable DETR-style set prediction + Hungarian matching) directly removes the non-differentiable peak-search step that causes this, and is the most evidence-backed unbuilt idea in the project — three independent project experiments (ceiling test, sharpening ablation, and the shared literature pattern) converge on the same root cause, not backbone capacity. This is the strongest literature-differentiating opportunity identified.

**Gap 6 — SubspaceNet's extreme-parameter-efficiency pattern (41,761 params via learned covariance-surrogate feeding a classical back-end) has never been applied to joint 2D AoA/AoD estimation.**
[EVIDENCE: SubspaceNet report — 1D DOA only]. Combining this pattern with the project's already-implemented Gridless-Unfold complex soft-threshold layer (which already targets the same quantization root cause as Gap 5, but under-performed at only 8 blocks/30,201 params, Pd=0.2216) [EVIDENCE: PROJECT `RESULTS.md`] is a concrete, low-risk architecture direction with no precedent in either the literature or the project's completed work.

**Gap 7 — Architecture search and compression have never been combined, in the literature or in this project.**
[EVIDENCE: Meneses2026 does adaptation/fine-tuning but not architecture search; no Tier-1 paper combines pruning with an alternative backbone]. The project's two tracks (4-block screening: FNO/SIREN/WindowAttention/GridlessUnfold vs. pruning: magnitude/SNR-aware) have run fully independently [EVIDENCE: PROJECT §7 item 7]. Pruning the current-best alternative (FNO) or scaling up Gridless-Unfold is an open item on both sides of the evidence base simultaneously.

**Gap 8 — No paper reports standalone inference latency for a joint 2D AoA/AoD estimator on real hardware.**
[EVIDENCE: Meneses2026 — real Jetson Orin Nano, but only training/fine-tuning energy reported, not clean inference latency; BeamSeek — real 60GHz hardware but 1D-only, and its own paper reports no latency numbers at all despite what an earlier synthesis stage wrongly attributed to it — see §6]. The project has working Teacher/Student/FNO checkpoints ready to benchmark on real hardware but has not yet done so — a genuinely open item on both sides, and cheap to close given existing artifacts.

**Gap 9 — Zero of the 6 Tier-1 papers use any Transformer/Attention component** (per the Architecture Component Matrix, §2). Resolving the project's own unresolved SIREN/WindowAttention training-failure question [EVIDENCE: PROJECT §3, §7 item 1 — loss flat from epoch 1 for both, confounded by untuned LR/init (SIREN) and missing warmup + 4× smaller batch (WindowAttention)] directly determines whether attention mechanisms are worth pursuing for the dense-bottleneck replacement in Gap 2, an axis the entire reviewed literature has not explored at all. This remains genuinely open (not a settled negative) pending a documented-but-unrun LR/warmup sweep.

**Gap 10 — No statistical significance / cross-seed variance is reported anywhere in the Tier-1 literature** [EVIDENCE: Lloria TVT2026 Sec M — 1000 realizations per point but no CI across independent training runs]. The project's evaluator-determinism check (13+ significant figures reproducibility given fixed weights) [EVIDENCE: PROJECT `baseline_models/README.md`] confirms pipeline determinism but not training-run-to-training-run variance, which is also unmeasured in the project. A cheap multi-seed addition would differentiate a rigor-focused paper from the entire reviewed set.

---

## 6. CONTRADICTIONS — flagged, with resolution or preserved uncertainty

| # | Contradiction | Resolution / how to preserve as open uncertainty |
|---|---|---|
| 1 | Lloria TVT2026 internal: Sec II-A states AoA/AoD ~ Uniform[0,2π]; Sec IV-B states Uniform[0,π] — self-contradictory paper text (both independently VERIFIED verbatim) | The code implements [0,π] — treat Sec IV-B + code as the operative definition; cite the [0,2π] statement only as a noted internal inconsistency, never as the effective distribution |
| 2 | Lloria TVT2026 paper text says U-Net ConvBlock uses k=5; the released code's default (and the only value the shipped inference pipeline ever exercises) is k=3 | **Unresolved** — recommend extracting the split `.h5` weight archive (`inf_model_007_256_unet.7z.001-005`) and reading `model.summary()` before citing this paper's U-Net numbers as a locked baseline. Preserve as open until then; do not assume either value without weight-file inspection |
| 3 | Lloria TVT2026 Fig. 3/4/Sec III-C describe the Dense/bottleneck block as a core structural element; the code path that actually loads the released weights (`ae=0` default) **skips it entirely** | **Unresolved, HIGH severity** — same resolution path as #2 (extract and inspect `.h5`). This directly determines whether the paper's central architectural narrative matches what was actually trained/evaluated. Flag explicitly in any downstream citation of this paper's U-Net results |
| 4 | Lloria TVT2026 training SNR: paper text says [−10,25]dB; the officially released code implements [−15,24]dB | Treat as a real reproducibility gap in the *official* repo, not a downstream error — anyone retraining "per the paper text" will not match the released model's actual training distribution. Disclose both figures whenever this project's own data generator (which inherits the code's range) is used for comparisons |
| 5 | Lloria TVT2026's P_D metric silently excludes fully-missed-detection trials from both numerator and denominator (rather than counting them as failures) | Not a "bug," but a non-standard convention that can inflate reported PD, especially at low SNR/high L. Preserve as an open caveat on every PD number quoted from this paper or from this project's own evaluator (which inherits the same convention) |
| 6 | The Robust DoA DAE-DNN paper (Chen et al. 2022) reports RMSE<0.5° in its own simulations, but Koh & Lee 2026's independent re-implementation on a different (BLE) dataset gets MAE 30°(sim)/>70°(real) | Resolve as a domain/dataset-mismatch effect, not a factual error in either paper — but preserve as a standing caution: single-paper self-reported accuracy claims for this architecture family should not be trusted without independent re-implementation evidence, exactly the caution both verification agents applied here |
| 7 | The **computed task brief** (not any paper) attributed specific latency figures ("MLP 0.63ms, CNN 1.26ms, deep-MPDR 0.89ms") to BeamSeek — these numbers and the "deep-MPDR" method do not exist anywhere in the actual BeamSeek paper | This is a fabrication introduced upstream in the pipeline (not by BeamSeek's authors). **Excluded from all citable numbers in this report.** Any future use of this project's own evidence chain should trace latency claims back to primary-source verification before citing them |
| 8 | PROJECT_STATUS.md (item 5) describes PIA-Net's issue as "real/ReLU-constrained ISTA state vs. arbitrary complex phase"; `BUJHO_SHOHOJ_VABE.md` documents 4 separate, differently-described bugs (data starvation, dictionary sign, ISTA step-size, coordinate mismatch) | Neither document cross-references the other's specific claim. **Preserve as open uncertainty** — do not assume these are the same issue without checking the v4 notebook directly if PIA-Net work is resumed. All current PIA-Net numbers are known-stale regardless (never retrained after the 4th bug fix) |
| 9 | This project's SIREN/WindowAttention core-block results (near-zero Pd, flat loss) could be read as "these architecture families don't work" for this task | **Explicitly not a settled result** — both runs are confounded (SIREN: untuned LR/init for a conv-stack context, not its native coordinate-MLP use case; WindowAttention: no warmup + 4× smaller batch from an OOM-driven auto-probe). Preserve as unresolved pending the recommended but unrun LR/warmup sweep; do not cite as a negative result for attention mechanisms in general |
| 10 | Gupta et al. TCOMM2025 is the only Tier-1 paper explicitly *not* performing joint AoA/AoD estimation, yet was included in this literature set | Not a contradiction to resolve, but a scope note to preserve: Gupta 2025 should be cited only as a classical-baseline/complexity reference (its 3D-BL O(Gτ³Gν³Gθ³) per-iteration cost is a legitimate, quotable target to beat with a single forward pass), never in a head-to-head joint-AoA/AoD accuracy table |
| 11 | The verification agent for Lloria TVT2026 found the specialist's "π/6 minimum pairwise separation" detail could not be located in the paper text, though the specialist had attributed it partly to Sec IV-B | Corrected: attribute π/6 to **code only** (`dldoa_dataset_generation.py`), never to the paper text, in any downstream citation |
| 12 | Meneses-Albalá 2026's specialist report characterizes the paper as performing "joint" AoA/AoD estimation; the verification agent found the literal word "joint" never appears in that paper's text | Not a factual error — the underlying behavioral claim (single 2D map, single forward pass, both angles from one peak) is independently verified as accurate. Preserve the distinction: the *behavior* is joint; the *terminology* is this report's own gloss, not the paper's own language — flag this explicitly if directly quoting the paper |
| 13 | TVT2026's officially released `Resnet()` function has no `M` parameter and cannot produce the "ResNet-512" variant plotted in Fig. 5/8, while U-Net's code does expose an M∈{256,512} switch | Real, unresolved reproducibility gap in the official repo — the ResNet-512 curve in the paper's own figures is not independently reconstructible from the released code as-is. Flag this whenever citing Fig. 5/8's ResNet-512 numbers as "reproducible" |

---

## Summary assessment for the project's lightweight joint AoA/AoD design

The literature converges on three concrete, evidence-backed target weaknesses this project is already positioned to exploit with existing artifacts: **(1)** publish real params/FLOPs/latency numbers no Tier-1 paper reports (Gap 1 — zero new work, data already exists); **(2)** replace the literature's shared dense/flatten bottleneck with a non-dense alternative, building directly on the project's own executed FNO result (Gap 2); **(3)** target the diagnosed Jacobian-singularity Pd ceiling with a differentiable set-prediction head (Gap 5 — the single most evidence-convergent unbuilt idea across both the literature and the project's own experiments). The clearest remaining open technical question before finalizing an architecture is whether attention-based components are viable at all for this task (Gap 9, Contradiction 9) — a cheap LR/warmup sweep should be run before that door is closed, since the entire Tier-1 literature has left this architecture family completely unexplored.