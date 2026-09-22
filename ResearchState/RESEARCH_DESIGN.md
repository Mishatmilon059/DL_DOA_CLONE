# Agent: architecture

# Agent-R1 — Architecture Researcher: Candidate Neural Architecture Directions

**Role note:** Per protocol §11, this is Loop 3 (architecture generation) output — a *spread of directions*, not a final design or a selection. No novelty claim below is settled; each is flagged `[HYPOTHESIS]` and must still pass Agent-R7 (Novelty Researcher) and the Adversarial Review Team (§15, §18) before being treated as defensible. Citations use the evidence tags from the literature synthesis; project-internal evidence is tagged `[EVIDENCE: PROJECT]`.

---

## A1. Attention-Bottleneck ResNet (CNN + lightweight Transformer)

**Motivation:** Every Tier-1 paper in the matrix uses a dense/flatten bottleneck (Lloria TVT2026 ResNet+U-Net) or a pure MLP (Naoumi). Zero of the 6 Tier-1 papers use attention at all — Gap 9. The dense bottleneck is also the paper's own self-flagged worst component: TVT2026 Table IV identifies it as the only quadratic-in-(P'Q') term [EVIDENCE: PAPER — Lloria TVT2026 Sec K/N].

**Borrowed components:** ResNet-shell conv stem/residual blocks from Lloria TVT2026/PIMRC2024 [EVIDENCE: PAPER]; general "replace global bottleneck with attention" pattern motivated by the transformer literature (not present in any reviewed AoA/AoD paper — this is an import from outside the reviewed set, flagged explicitly since no paper here validates it for this task).

**Novel twist [HYPOTHESIS]:** Replace only the bottleneck (not the whole backbone) with a small windowed/linear-attention block operating on the ResNet's downsampled feature map, keeping the conv encoder/decoder shell identical to the paper's for direct ablation comparability. This isolates "does attention fix the bottleneck problem" as a single-variable test, directly resuming the project's own unresolved Window-Attention question (Gap 9 / Contradiction 9) under a *fixed* LR-warmup + matched-batch protocol this time — the project's prior Window-Attention run was confounded (no warmup, 4× smaller batch) [EVIDENCE: PROJECT — PROJECT_STATUS.md].

**Expected strengths:** Global receptive field without the O(quadratic) dense cost; directly answers an open literature-wide gap; cheap to ablate against the project's existing FNO/GridlessUnfold screening runs since the outer shell is unchanged.

**Expected weaknesses:** Attention blocks are known to need LR warmup and larger batches to train stably — exactly the confound that sank the project's earlier attempt; risk of re-failing for a training-recipe reason rather than an architectural one if the sweep isn't done properly first.

**Failure modes:** Flat/non-descending loss (as seen before) if warmup is skipped again; attention map collapsing to near-uniform weighting under low SNR (no clear signal to attend to), degenerating to worse-than-dense performance exactly where robustness matters most (Gap 3).

---

## A2. Dual-Backbone Fusion (ResNet + U-Net)

**Motivation:** Lloria TVT2026 itself reports *both* a ResNet and a U-Net variant for the same task with different accuracy/robustness trade-offs (ResNet .925/.253, U-Net .945/.230 PD/RMSE @20dB) [EVIDENCE: PAPER — Lloria TVT2026 Table II], but never fuses them. Meneses-Albalá 2026 independently confirms the U-Net's edge-deployment cost is a real concern ("larger than ResNet," authors' own admission) [EVIDENCE: PAPER — Meneses-Albalá 2026 Gap Matrix].

**Borrowed components:** ResNet residual-block encoder (Lloria) as the primary feature extractor; U-Net-style skip connections (Lloria/Meneses-Albalá) added *only* at 1–2 resolution levels rather than the full encoder/decoder mirror, to keep params low.

**Novel twist [HYPOTHESIS]:** A "ResNet trunk with sparse U-Net skips" — most of the network stays ResNet (cheap, already validated at 469K params), and only 1–2 skip connections are added at the resolution level nearest the diagnosed Jacobian-singularity failure zone (near end-fire angles), to recover spatial precision without paying for a full second decoder path.

**Expected strengths:** Directly targets the ceiling behavior with minimal added parameters; both parent architectures are already independently validated on this exact task/dataset, reducing implementation risk.

**Expected weaknesses:** Not clearly "novel" in the strict sense (§15 test 3 — "merely replacing one layer") since it's a straightforward combination of two published components from the *same* paper; Novelty Researcher will likely classify this `PARTIALLY NEW` at best.

**Failure modes:** Skip connections could reintroduce the same dense-bottleneck cost they're meant to avoid if placed carelessly; marginal-gain risk similar to Meneses-Albalá's own fine-tuning result (only 3.6% RMSE / +1pp PD gain, mostly at the most severe impairment level) [EVIDENCE: PAPER — Meneses-Albalá 2026 Table 2], i.e. the fusion may not move the needle enough to justify the added complexity.

---

## A3. Channel/Spatial-Attention ResNet (CNN + attention, non-Transformer)

**Motivation:** A lower-risk attention variant than A1 — squeeze-excite / CBAM-style channel-and-spatial attention adds negligible parameters and doesn't carry the warmup/batch-size training instability that sank the project's Window-Attention run, while still probing the "is attention useful at all" question from a different angle.

**Borrowed components:** ResNet conv-block shell (Lloria) with lightweight channel-attention gating inserted between residual blocks — a well-established efficient-CNN pattern, not itself drawn from any reviewed AoA/AoD paper (no paper in the set uses it — a genuine literature gap, not an import from a competing paper).

**Novel twist [HYPOTHESIS]:** Use the attention weights themselves as an auxiliary SNR/impairment-severity signal — hypothesize that channel-attention activations correlate with local SNR, and expose this as an interpretable diagnostic alongside the AoA/AoD outputs (testable against the project's existing robustness-test protocol, Gap 3/Notebook 4).

**Expected strengths:** Nearly free in parameter/FLOP budget (squeeze-excite blocks are typically <1% overhead); no known training-instability failure mode, unlike A1; easy to ablate cleanly against the plain-ResNet baseline already in the project's registry.

**Expected weaknesses:** Small capacity increase may produce only a marginal accuracy gain, weakening the novelty argument (§15 test 5 — "does it solve an actual limitation?"); does not address the dense-bottleneck root cause at all, so it may leave the Pd ceiling (~0.972) [EVIDENCE: PROJECT] essentially untouched.

**Failure modes:** Attention weights collapsing to near-uniform (no discriminative signal) under the hardest robustness conditions, making the "SNR-correlation" interpretability claim untestable/false.

---

## A4. Complex-Valued Front-End + Real-Valued Lightweight Backbone

**Motivation:** Naoumi et al. (both JSTSP2024 and GCWkshps2023) show a complex-valued MLP front-end achieves strong complexity reduction (6.5×/10.3× fewer multiplies than a 2D parametric algorithm; exact 8,584-param, 4.321 MMACs footprint) [EVIDENCE: PAPER — Naoumi JSTSP2024, GCWkshps2023], but their architecture is a pure complex MLP throughout, which the same authors admit saturates at high SNR. No Tier-1 joint-AoA/AoD paper combines a complex-valued *front-end* with a real-valued *conv* backbone.

**Borrowed components:** Complex-valued layer structure (magnitude/phase-aware operations) from Naoumi's MLP [EVIDENCE: PAPER]; real-valued ResNet-shell backbone from Lloria for the bulk of the network.

**Novel twist [HYPOTHESIS]:** Keep only the first 1–2 layers complex-valued (operating directly on the raw complex received signal Y, preserving phase information the current real-valued-from-the-start pipeline discards immediately), then convert to a real-valued magnitude/phase-channel representation before the ResNet/FNO backbone — hypothesized to fix the Naoumi-reported high-SNR saturation (their own admitted open weakness) by giving the backbone a cleaner high-SNR-relevant fine-grained representation, without paying for complex arithmetic throughout the whole network.

**Expected strengths:** Directly targets a literature-documented, author-admitted weakness (Naoumi's high-SNR saturation) with a low-cost architectural change; complex-valued front-end is exactly the input-reduction philosophy Naoumi credits for their complexity win, so params should stay low.

**Expected weaknesses:** Complex-to-real conversion point is a design choice with no literature precedent to justify placement — high risk of information loss if done wrong; complex-valued ops require careful, non-standard autodiff/framework support (implementation risk, not just architectural risk).

**Failure modes:** If the complex front-end's phase information is discarded too early, this degenerates to the exact same real-valued pipeline already tried (no gain); if kept complex too long, inherits Naoumi's own architecture-family limitations (their paper's real 2D-algorithm baseline outperforms their MLP beyond 13dB SNR) [EVIDENCE: PAPER — Naoumi JSTSP2024].

---

## A5. Multi-Scale CNN with Shared AoA/AoD Representation + Dual Lightweight Heads

**Motivation:** All 3 Lloria-family papers (TVT2026, PIMRC2024, Meneses-Albalá 2026) commit to a *single* joint 2D heatmap output that entangles AoA and AoD into one representation, then extracts both angles via a shared non-differentiable peak search — which is exactly the mechanism identified (via the project's own ceiling test + diffusion-sharpening ablation) as the source of the Pd≈0.972 ceiling [EVIDENCE: PROJECT]. No paper tests a shared-*encoder*-but-*split*-heads design for this specific joint task.

**Borrowed components:** Multi-scale conv feature pyramid pattern (generic efficient-CNN design, not specific to any one reviewed paper); shared-encoder-plus-task-heads pattern loosely analogous to Koh & Lee 2026's DNN/CNN benchmark family structure [EVIDENCE: PAPER — Koh & Lee 2026], adapted from 1D AoA-only to a 2D joint setting.

**Novel twist [HYPOTHESIS]:** Instead of one 2D heatmap decoded by blob search, use a shared multi-scale conv trunk feeding two lightweight regression heads (one per angle) that each directly regress a continuous value (or a coarse bin + offset, see A6/A7) rather than searching a joint image for a peak — removing the peak-search dependency architecturally rather than only at the loss/matching level.

**Expected strengths:** Directly attacks Gap 5 (the single most evidence-convergent literature+project finding) from the representation side; dual heads are individually cheap (few added params over the shared trunk).

**Expected weaknesses:** Splitting the representation risks losing the implicit AoA-AoD correlation the joint 2D map currently encodes (which is *why* the paper's approach is "genuinely joint" per the literature matrix) — could regress toward the non-joint behavior Gupta 2025 deliberately chose and that this project's objective explicitly wants to avoid.

**Failure modes:** If the shared trunk under-represents the AoA-AoD coupling, accuracy could degrade specifically on closely-spaced or correlated-path cases (an axis no Tier-1 paper stress-tests — Gap 3/Robustness Matrix), i.e. failure would be invisible until a dedicated multi-target test is run.

---

## A6. Coarse-to-Fine Estimation (Classification Grid + Regression Refinement)

**Motivation:** This directly targets the diagnosed root cause of the Pd ceiling — pixel quantization amplified by the Jacobian singularity near end-fire angles, d(π cos ψ)/dψ = −π sin ψ [EVIDENCE: PROJECT — ceiling test]. A coarse grid classification stage sidesteps sub-pixel quantization error at the coarse level, then a fine regression stage estimates the residual offset with much finer effective resolution than the raw 256×256 heatmap grid.

**Borrowed components:** Heatmap/grid output philosophy (Lloria TVT2026 [EVIDENCE: PAPER]) for the coarse stage; the general two-stage classification→regression idea is explicitly listed as a candidate pattern in the protocol itself (§11) rather than drawn from a specific reviewed paper — no Tier-1 or Tier-2 paper in the matrix implements coarse-to-fine for AoA/AoD.

**Novel twist [HYPOTHESIS]:** Make the "coarse" stage operate on a deliberately *coarser* grid than the current 256×256 (e.g., 32×32 or 64×64) specifically to reduce the Jacobian-amplification effect (which worsens as grid resolution increases near end-fire angles, since the mapping from angle to pixel becomes more compressed there), then let the fine regression head recover precision via direct coordinate regression rather than further pixel search.

**Expected strengths:** Architecturally the most direct response to Gap 5; the coarse stage can reuse most of the existing ResNet backbone almost unchanged, keeping implementation risk low relative to A9 (full set-prediction redesign).

**Expected weaknesses:** Two-stage training/inference adds complexity and a potential error-compounding path (coarse-stage errors propagate into the fine stage) — the opposite failure mode of what it's meant to fix, if not calibrated carefully at the coarse/fine boundary.

**Failure modes:** If the coarse grid is too coarse, closely-spaced sources (small L separations, the one axis all 3 Lloria papers *do* test) could collapse into the same coarse bin and become unrecoverable by the fine stage — likely to hurt exactly the multipath/L-sweep robustness axis these papers already report on, making any regression here immediately visible and damaging in a direct comparison.

---

## A7. Classification + Regression Hybrid (Anchor-Free Set Head)

**Motivation:** A generalization of A6 that explicitly frames peak detection as a differentiable classification-of-existence + regression-of-location problem per candidate, rather than a fixed two-stage pipeline — closer in spirit to modern anchor-free detection heads. This is listed as a candidate hybrid pattern in the protocol (§11) and is the most direct architectural cousin of the project's own most evidence-backed unbuilt idea, `My_Proposed_Architecture.md`'s Stage 3 (DETR-style set prediction + Hungarian matching) [EVIDENCE: PROJECT].

**Borrowed components:** Heatmap-based existence confidence from Lloria's blob-detection paradigm [EVIDENCE: PAPER]; the differentiable-matching concept itself is not present in any reviewed paper (Gap 5 — none of the 3 Lloria-family papers, despite sharing the identical output representation, diagnose or fix the non-differentiable peak-search step).

**Novel twist [HYPOTHESIS]:** A per-cell "objectness" classification head (does a source exist near this coarse cell?) combined with a per-cell continuous (AoA, AoD) regression offset, trained end-to-end with a differentiable matching loss (Hungarian or simpler greedy-nearest assignment) — this is essentially A6's coarse/fine split made differentiable end-to-end instead of two separately-trained stages, directly removing the non-differentiable step identified as the ceiling's cause.

**Expected strengths:** This is the *architecturally* closest candidate to the project's own most evidence-convergent unbuilt idea (three independent project experiments — ceiling test, diffusion-sharpening ablation, and the shared literature failure pattern — all converge on this exact fix) [EVIDENCE: PROJECT]. If it works, it has the strongest, most defensible novelty argument of any candidate here, since it targets a root cause no paper in the entire reviewed set (Tier-1 or Tier-2) even diagnoses.

**Expected weaknesses:** Highest implementation complexity of any candidate in this set (matching loss, variable-cardinality output, training-stability concerns known from DETR-style heads generally); hardest to keep genuinely "lightweight" if not carefully scoped, working against the project's efficiency objective.

**Failure modes:** DETR-style heads are documented (outside this literature set) to converge slowly and need many training steps/careful loss balancing — could fail not because the idea is wrong but because it needs substantially more training budget than the project's other screened candidates (FNO/GridlessUnfold/SIREN/Window-Attention) got, risking an unfair architecture comparison if not step-matched (a concern the protocol explicitly flags, §16/PROJECT §7 item 2).

---

## A8. Physics-Informed Feature Extraction (Steering-Vector/DFT Front-End) + Lightweight DL Backbone

**Motivation:** Gupta 2025's classical Sparse-Bayesian-Learning/MUSIC pipeline (non-DL) shows real gains from grounding estimation in the physical steering-vector model [EVIDENCE: PAPER — Gupta TCOMM2025, 5dB/15dB gain vs OMP/FOCUSS], and this project's own DFT-SIC classical baseline independently validates that a physics-grounded front-end reproduces near-paper accuracy without any learning at all (Pd 0.884 vs paper's 0.891 @20dB) [EVIDENCE: PROJECT]. Naoumi's coarse-timing/IFFT-based input reduction is a concrete precedent for a lightweight physics-informed front-end feeding a small learned model [EVIDENCE: PAPER — Naoumi JSTSP2024/GCWkshps2023].

**Borrowed components:** DFT-based front-end / classical angle-hypothesis dictionary (project's own DFT-SIC baseline + Gupta's steering-vector model) [EVIDENCE: PROJECT, PAPER]; the general "physics branch + learned refinement branch" fusion idea validated as sound reasoning in the project's PIA-Net work (even though the current PIA-Net implementation is bug-affected/stale) [EVIDENCE: PROJECT — BUJHO_SHOHOJ_VABE.md].

**Novel twist [HYPOTHESIS]:** Use a fixed (non-learned) steering-vector/DFT projection as a *feature-reduction* front-end — replacing several early conv layers with a physics-derived transform that has zero trainable parameters — then feed the reduced representation into a small learned backbone (ResNet-lite, FNO, or GridlessUnfold) for the actual joint AoA/AoD regression. This directly chases Gap 1 (params/FLOPs/latency together, a claim no Tier-1 paper makes) since the physics front-end is essentially free in parameter count.

**Expected strengths:** Strong efficiency story (front-end adds ~zero parameters); grounded in two independent validations already in hand (Gupta's classical result and this project's own DFT-SIC reproduction), lowering implementation risk versus a purely learned front-end.

**Expected weaknesses:** A fixed, non-learned front-end may throw away exactly the information a learned model would otherwise exploit under distribution shift or hardware impairment — Gupta's own paper explicitly does *not* test hardware-impairment/distribution-shift robustness, so this project would be extending into genuinely untested territory (Gap 3) with an approach that has no robustness precedent in the literature at all.

**Failure modes:** Under phase-shifter/gain-mismatch impairment (a condition where the true steering vectors deviate from the assumed model — the exact failure mode Meneses-Albalá 2026 studies as its primary focus [EVIDENCE: PAPER]), a fixed physics front-end could systematically mis-project the signal with no way to adapt, unlike a fully learned model that can in principle absorb some impairment through training-distribution exposure.

---

## A9. Global-Spectral + Learned-Sparsity Bottleneck (FNO + GridlessUnfold Fusion)

**Motivation:** This is the one direction built entirely from the project's *own* two best-performing screened alternatives rather than from an external paper, addressing Gap 6 and Gap 7 (architecture search and compression/combination have never been tried together, in the literature or in this project). FNO is the current best alternative core block from scratch (Pd 0.6235, 88% of teacher's, 334K params) [EVIDENCE: PROJECT — RESULTS.md]; GridlessUnfold directly targets the same pixel-quantization root cause as Gap 5 via a phase-preserving complex soft-threshold layer, but under-performed only because of shallow depth (8 blocks, Pd 0.2216, 30K params) [EVIDENCE: PROJECT].

**Borrowed components:** FNO/spectral-convolution block (project screening result, no direct literature precedent in the reviewed set — a genuine gap); GridlessUnfold's complex soft-threshold layer (project-internal, inspired by the PIA-Net physics-informed deep-unfolding concept) [EVIDENCE: PROJECT]; the extreme-parameter-efficiency pattern of "learned covariance/representation surrogate feeding a lighter back-end" is directly borrowed from SubspaceNet's 41,761-param design [EVIDENCE: PAPER — Shmuel et al. 2025 TVT], applied here to a 2D joint setting for the first time.

**Novel twist [HYPOTHESIS]:** Use FNO blocks for the global-receptive-field early/mid layers (where the input Y is already frequency-domain, giving FNO a natural fit), and swap only the final bottleneck/output stage for the GridlessUnfold complex soft-threshold layer at increased depth — hypothesizing that FNO's global context plus GridlessUnfold's precision mechanism (matched to more capacity than the 8-block screening run gave it) together resolve both the accuracy gap and the pixel-quantization ceiling simultaneously.

**Expected strengths:** Directly combines two already-independently-validated project components rather than starting from zero; addresses two gaps (6 and 7) in a single design; has the most concrete prior evidence trail of any candidate (both parent components have real numbers on the exact same 8000-sample bank).

**Expected weaknesses:** FNO's own documented weakness (Pd drops from ~0.80 at 10dB to 0.70 at 25dB — degrading exactly as SNR improves, the opposite of every other model in the registry) [EVIDENCE: PROJECT — RESULTS.md] is unexplained and unresolved; fusing it with GridlessUnfold doesn't obviously fix a high-SNR degradation whose cause hasn't been diagnosed yet (§7 item 5, still open).

**Failure modes:** If the FNO high-SNR weakness is a fundamental property of the spectral representation losing fine-grained precision once noise stops being the bottleneck (the project's own working hypothesis, unconfirmed), fusing it with GridlessUnfold could simply inherit that ceiling rather than fix it — this candidate's success is contingent on an open, currently-undiagnosed question (Gap 5/§7 item 5) rather than a validated mechanism.

---

## Cross-cutting note: fusion mechanism is separable from backbone choice

A7's differentiable set-prediction head (or A6's coarse/fine split) is not mutually exclusive with A1/A3/A4/A8/A9's backbone choices — the Assembly stage should treat "backbone family" and "output/fusion mechanism" as two independent axes when constructing full candidates (per §17's Candidate A/B/C pattern), rather than only evaluating these nine directions as fixed, monolithic packages.

---

## Summary table for the Assembly stage

| ID | Direction | Primary literature/project anchor | Gap addressed | Key risk |
|---|---|---|---|---|
| A1 | Attention-bottleneck ResNet | Lloria TVT2026 dense-bottleneck flag; Gap 9 | 2, 9 | Training instability (warmup/batch) |
| A2 | ResNet+U-Net sparse-skip fusion | Lloria TVT2026 Table II; Meneses-Albalá 2026 | 5 (partial) | Weak novelty (same-paper combo) |
| A3 | Channel/spatial-attention ResNet | Generic efficient-CNN pattern (no direct paper precedent) | 9 (low-risk variant) | Marginal-gain risk |
| A4 | Complex front-end + real backbone | Naoumi JSTSP2024/GCWkshps2023 high-SNR saturation | High-SNR saturation (Naoumi's own admitted gap) | Complex/real boundary design risk |
| A5 | Multi-scale shared-encoder dual heads | Koh & Lee 2026 shared-encoder pattern (adapted) | 5 | Loses AoA-AoD coupling |
| A6 | Coarse-to-fine classification+regression | Protocol-listed pattern; project ceiling test | 5 | Error compounding coarse→fine |
| A7 | Anchor-free classification+regression set head | `My_Proposed_Architecture.md` Stage 3; ceiling test | 5 (strongest anchor) | Training complexity/step budget |
| A8 | Physics front-end + lightweight DL | Gupta TCOMM2025; project DFT-SIC baseline | 1 (params/FLOPs story) | No impairment-robustness precedent |
| A9 | FNO + GridlessUnfold fusion | Project's own FNO/GridlessUnfold results; SubspaceNet efficiency pattern | 6, 7 | FNO's undiagnosed high-SNR weakness |

All nine are candidates for the Assembly Team (§17) and Adversarial Review Team (§18) to combine, prune, and stress-test — none is selected here, per §11's explicit instruction not to settle on one architecture at this stage, and per §15's requirement that Agent-R7 (Novelty Researcher) independently classify each `[HYPOTHESIS]` novel twist (NEW / PARTIALLY NEW / KNOWN COMBINATION / NOT NOVEL / UNCERTAIN) before any of them can be treated as a defensible contribution.

# Agent: lightweight

# Agent-R2 Report — Lightweight/Compression Researcher

**Scope note:** This report evaluates the 12 efficiency techniques named in Protocol §12 against this project's own already-measured numbers (`baseline_models/README.md`, `RESULTS.md`) and the literature synthesis, per Protocol §19 evidence-labeling rules. No new code was run; all parameter/FLOP/accuracy estimates below are `[INFERENCE]` (derived from established ML literature scaling laws, not measured on this project's exact architecture) unless tagged `[EVIDENCE: PROJECT]` or `[EVIDENCE: PAPER]`. None should be read as a substitute for an actual ablation run.

**Reference points held fixed throughout:**
- Teacher (64-block ResNet): 469,393 params, Pd 0.7103 `[EVIDENCE: PROJECT baseline_models/README.md]`
- Magnitude-pruned Student (r=8, 64-block, fine-tuned from teacher): 314,513 params (67% of teacher), Pd 0.6956 (Δ −0.0147) `[EVIDENCE: PROJECT]`
- FNO (8-block, trained from scratch): 334,321 params (71% of teacher), Pd 0.6235 (88% of teacher's Pd) `[EVIDENCE: PROJECT]`
- GridlessUnfold (8-block, from scratch): 30,201 params, Pd 0.2216 `[EVIDENCE: PROJECT]`
- Naoumi GCWkshps2023 complex-MLP: 8,584 params `[EVIDENCE: PAPER]` (different task framing — bistatic ISAC MLP, LOW comparability to this project's ResNet/heatmap pipeline, see literature synthesis §1 comparability notes)
- SubspaceNet: 41,761 params vs >21M black-box CNN for 1D DOA `[EVIDENCE: PAPER]` (NOT COMPARABLE — 1D task)

---

## 1. Depthwise Separable Convolution

**Mechanism:** Replace each standard k×k conv (C_in→C_out) with a depthwise k×k conv (per-channel) + pointwise 1×1 conv. Reduces per-layer params/FLOPs by roughly `1/C_out + 1/k²` (e.g. ~8–9× for k=3, C_out=64, a MobileNet-standard result).

**Estimated param effect:** Applied to the teacher's 64-block ResNet's 3×3 conv layers, a plausible reduction is 3–6× on the conv-heavy portion of the network `[INFERENCE]` — i.e. a candidate in the tens-of-thousands to ~150K param range depending on how many blocks are converted vs kept dense (early/late layers often kept dense in practice to protect accuracy).

**Estimated accuracy effect:** Depthwise separable convs are known to lose some accuracy relative to full convs at matched depth, particularly hurting the model's ability to mix spatial + channel information in one op — historically 1–3pp top-1 drop on ImageNet-scale tasks at similar compression ratios `[INFERENCE, from general CNN-efficiency literature, not this task]`. For this project's specific failure mode (Jacobian-singularity/pixel-quantization Pd ceiling near end-fire angles — PROJECT_STATUS.md, RESULTS.md), depthwise separable convs do **not** address the root cause; if anything, reduced channel-mixing capacity per layer could make fine-grained peak localization *harder*, i.e. **could interact badly with, not just be orthogonal to, the diagnosed ceiling**. `[HYPOTHESIS]`

**Build-on vs. duplicate:** **NOT YET TRIED.** This is architecturally closest to the *architecture-screening track* (FNO/SIREN/WindowAttention/GridlessUnfold swapped into the ResNet shell) rather than the *pruning track* — it should be run as a 5th core-block candidate under the same step-matched, 8-block, from-scratch protocol already established, not as a modification of the pruned student. **Directly duplicates Gap 7** in the literature synthesis (architecture search and compression never combined) if paired with subsequent pruning of the result.

---

## 2. Bottleneck Blocks (1×1 reduce → 3×3 → 1×1 expand)

**Mechanism:** ResNet-50-style bottleneck: squeeze channel dimension before the expensive 3×3 conv, then re-expand. Reduces the 3×3 conv's FLOPs/params quadratically in the squeeze ratio while keeping a wide input/output channel count for representational capacity.

**Estimated param effect:** At a typical 4:1 squeeze ratio, could cut the teacher's conv-block params by roughly 2–4× `[INFERENCE]`, likely landing in a similar range to the magnitude-pruned student (~300K) or somewhat below, depending on how aggressively squeezed.

**Estimated accuracy effect:** Bottleneck blocks are generally **less damaging to accuracy than depthwise separable** at a matched compression ratio, because the 1×1 layers still perform full channel mixing — this is the reason ResNet-50/101 bottleneck designs became standard over ResNet-34's plain blocks in classification `[INFERENCE, general literature]`. For this project, a moderate risk: bottleneck squeeze on the channels feeding into the heatmap output could interact with the diagnosed pixel-quantization ceiling if the squeeze happens in later layers close to the output (reduced channel capacity right before the localization-critical layers) — recommend squeezing only early/mid blocks, keeping late blocks at full width. `[HYPOTHESIS]`

**Build-on vs. duplicate:** **PARTIALLY BUILDS ON existing pruning work.** The project's magnitude pruning already reduces channel width 12→8 uniformly across all 64 blocks (`r=8`); bottleneck blocks are a *structural, trainable-from-scratch* alternative to uniform post-hoc channel pruning, not the same technique — bottleneck squeeze ratios can be learned/optimized per-block rather than applied uniformly. This is the most direct, low-risk "next architecture" to try because it is conceptually a middle ground between the teacher (full width) and the FNO screening result (different block entirely), reusing the *same* ResNet skeleton and thus comparable in the existing evaluation harness without new architecture-screening infrastructure.

---

## 3. Grouped Convolution

**Mechanism:** Split C_in/C_out channels into G groups, each group convolving independently (ResNeXt-style). Reduces conv params/FLOPs by ~1/G, sitting between full conv (G=1) and depthwise (G=C).

**Estimated param effect:** At G=4 or G=8, roughly 4–8× reduction on conv layers `[INFERENCE]` — tunable, unlike depthwise's fixed extreme.

**Estimated accuracy effect:** Empirically, grouped convolution with a moderate G (4–8, as in ResNeXt) tends to preserve accuracy notably better than depthwise (G=C) at comparable compute, because it retains partial cross-channel mixing within each group `[INFERENCE, general literature]`. This makes it a **more attractive candidate than pure depthwise** for this project if channel-mixing capacity near the output matters for angle localization precision, as hypothesized under item 1 and 2.

**Build-on vs. duplicate:** **NOT YET TRIED**, and is a genuinely new axis relative to both the pruning track (post-hoc weight removal) and architecture-screening track (block-type replacement) — it modifies the conv structure itself rather than either pruning existing weights or swapping the entire block type. Worth flagging as the **best risk-adjusted first efficiency experiment to run** among the untried structural options, because G is a single tunable knob that can be swept (G∈{2,4,8}) cheaply against the existing 8000-sample eval bank without touching the data pipeline.

---

## 4. Low-Rank Factorization

**Mechanism:** Decompose a weight matrix/conv kernel W (C_out×C_in×k×k) into a product of two smaller factors (rank-r approximation), reducing params roughly proportional to r/min(C_in,C_out).

**Estimated param effect:** Depends entirely on chosen rank; SVD-based rank selection can target a specific compression ratio directly (this is the appeal of this axis).

**Estimated accuracy effect:** Low-rank approximation error accumulates layer-by-layer in deep stacks (the teacher has 64 blocks), and truncation error specifically degrades the network's ability to represent sharp/high-frequency spatial features — plausibly compounding the project's *already-diagnosed* pixel-quantization/Jacobian-singularity problem near end-fire angles, since that failure mode is fundamentally about losing precision in fine spatial localization. `[HYPOTHESIS, but directly informed by EVIDENCE: PROJECT's ceiling-test finding]`

**Build-on vs. duplicate: DIRECT DUPLICATE OF ALREADY-SCOPED WORK.** `[EVIDENCE: PROJECT — PROJECT_STATUS.md §"low-rank factorization compression axis"]`: this exact technique was mathematically validated locally via SVD reconstruction and then **explicitly dropped from the plan on 2026-09-21 for scope reasons**, not because it failed. Per the prior-work analysis's explicit guidance (§5 "What should NOT be repeated"): *"revisit only if compression is being revisited as its own topic, not by re-deriving the math."* Recommendation: if compression is now a first-class topic again (which this task implies), reuse the existing SVD validation rather than re-deriving it, and prioritize items 2/3 above it since they have zero prior scoping investment already sunk-and-abandoned here.

---

## 5. Further Pruning (beyond r=8)

**Mechanism:** More aggressive magnitude or SNR-aware channel pruning (e.g. r=6 or r=4) applied to the already-pruned or original teacher.

**Estimated param effect:** Roughly linear in channel width squared for conv-heavy layers — r=8→r=6 could yield ~175–200K params `[INFERENCE, extrapolating from the r=8 result's 67% ratio]`; r=4 could approach ~80–100K but with steeply rising risk.

**Estimated accuracy effect:** The project's own r=8 result already shows a real, non-trivial gap opening (Δ −0.0147 Pd) `[EVIDENCE: PROJECT]`. Pruning-accuracy curves are typically **not linear** — degradation accelerates sharply past some critical width, and this project has no r<8 data point to anchor the extrapolation. A cheap, low-risk experiment (**directly extending, not duplicating**, existing infrastructure) would be an r∈{4,6,8,10} sweep to characterize the actual degradation curve rather than guessing.

**Build-on vs. duplicate:** **DIRECTLY BUILDS ON existing infrastructure** — same pruning code, same evaluator, same eval bank. This is the lowest-implementation-cost item on this entire list. However, it does **not** address the diagnosed Pd ceiling (pruning removes capacity, it doesn't change the non-differentiable blob-detection output representation that Gap 5 identifies as the actual bottleneck) — flagged so it isn't oversold as solving the project's core open problem, only as a pure efficiency lever.

**Separately — the unresolved SNR-aware pruning loop `[EVIDENCE: PROJECT]`:** the project's own comparison at 1200-sample subsample scale found SNR-aware pruning *beats* magnitude pruning (ΔPd −0.0190 vs −0.0221 `[EVIDENCE: PROJECT]`), but the SNR-aware student's weights were lost twice on Kaggle and never recovered, so the full-8000-sample number for the supposedly-better method is unknown. **This is a duplicate-avoidance flag, not a technique recommendation**: before running *any* new pruning experiment, recovering/retraining the SNR-aware student (Prior-Work §7 item 3) should take priority, since it closes an existing open loop rather than opening a new one.

---

## 6. Quantization

**Mechanism:** Reduce weight/activation numeric precision (FP32→INT8 or lower), post-training or quantization-aware.

**Estimated param effect:** Does not reduce parameter *count* — reduces model *size* (bytes) and can accelerate inference on supporting hardware, typically ~4× size reduction (FP32→INT8) and often measurable latency wins on CPU/edge hardware `[INFERENCE, general literature]`.

**Estimated accuracy effect:** INT8 post-training quantization on well-behaved CNNs is usually near-lossless (<0.5pp) `[INFERENCE]`, but this project's regression output (continuous angle values via heatmap peak location) is more precision-sensitive than a classification softmax — and the project has *already* diagnosed a pixel-quantization-driven Pd ceiling as its core unsolved problem. Standard weight/activation quantization is a **different kind of quantization** (numeric precision) from the diagnosed problem (spatial/output quantization), but the naming coincidence is worth flagging explicitly so it is not confused with, or assumed to worsen, the existing diagnosed issue — they are largely orthogonal effects. `[HYPOTHESIS]`

**Build-on vs. duplicate:** **COMPLETELY UNTRIED, zero prior investment.** This is the only technique on the list that targets model *size*/*latency* directly rather than param count or Pd, which directly closes literature **Gap 1** and **Gap 8** (no paper in the entire reviewed set reports params+FLOPs+latency+memory together, and none report standalone inference latency on real hardware for a joint AoA/AoD model `[EVIDENCE: PROJECT synthesis §5]`). Combined with the project's existing Teacher/Student/FNO checkpoints (already trained, ready to benchmark per Gap 8), a quantized-inference-latency benchmark on real hardware would be one of the cheapest, most literature-differentiating additions available — no retraining required, only a benchmarking pass.

---

## 7. Knowledge Distillation

**Mechanism:** Train a small student network to match the teacher's soft outputs (heatmap distributions, not just hard labels), typically via a distillation loss (e.g. KL divergence or MSE on the heatmap) alongside the standard supervised loss.

**Estimated param effect:** Student size is a free design choice under distillation — could target the same 314,513-param footprint as the existing magnitude-pruned student, or go smaller (e.g. matching FNO's 334K, or an 8-block-from-scratch size like GridlessUnfold's 30K, for a distillation-vs-scratch-training controlled comparison).

**Estimated accuracy effect:** Distillation from a strong teacher (Pd 0.7103) is well-established to often outperform training the same small architecture from scratch or via magnitude pruning alone, because it transfers the teacher's full output *distribution* (including near-miss peak shapes) rather than just its final weights `[INFERENCE, general distillation literature — e.g. distilled students often close 30–60% of the accuracy gap to teacher versus scratch training of the same architecture]`. This is a **plausible direct lever on Gap 2/5**: if the teacher's heatmap output near end-fire angles encodes useful "soft" shape information that gets clipped away by hard peak-detection labels alone, a distillation loss on the raw heatmap (not just the post-blob-detection Pd metric) could transfer some of that precision to a smaller student. This is speculative and untested. `[HYPOTHESIS]`

**Build-on vs. duplicate:** **COMPLETELY UNTRIED — the single largest gap in the project's own compression track.** The project has run pruning (post-hoc weight removal from a trained teacher) and from-scratch architecture screening (FNO, GridlessUnfold, etc.) as two independent tracks, but never distillation, which is conceptually a third, distinct compression paradigm (transfer teacher *knowledge*, not teacher *weights*, into an arbitrary — possibly much smaller or structurally different — student). This directly answers **Gap 7** (architecture search and compression never combined) in the strongest possible way: distilling the teacher into an FNO-shaped or GridlessUnfold-shaped student combines both tracks in one experiment. **Recommended as the highest-value new experiment among all 12 techniques**, given (a) zero prior work, (b) direct literature-gap relevance, (c) the teacher checkpoint is already trained and ready to serve as the distillation source with no new data-generation cost.

---

## 8. Feature Compression (bottlenecked intermediate representation / autoencoder-style squeeze)

**Mechanism:** Insert a narrow intermediate representation (e.g. compress the channel-wise activation volume at some mid-network point) forcing the model to learn a compact encoding before expanding back out for the heatmap decode.

**Estimated param effect:** Depends on where and how narrow the bottleneck is placed; if placed centrally in a U-Net-style architecture, this is structurally the same idea Lloria's own U-Net uses (dense/flatten bottleneck) — which the literature synthesis's Gap 2 flags as the **literature's own self-identified worst component** (`[EVIDENCE: PAPER — TVT2026 Table IV, only quadratic-in-(P'Q') term in its complexity table]`).

**Estimated accuracy effect:** A naive dense/flatten bottleneck is specifically the component this project's own FNO screening result was designed to test as a *replacement* for — and the FNO result (non-dense spectral bottleneck, 0.6235 Pd, 88% of teacher's) is direct, already-executed evidence that moving *away* from a dense/flatten feature-compression bottleneck toward a structured (convolutional/spectral) one is the right direction, not a dense compression bottleneck itself. `[EVIDENCE: PROJECT RESULTS.md]`

**Build-on vs. duplicate:** **PARTIALLY DUPLICATES existing FNO work if implemented as a dense bottleneck** (exactly the pattern the project already has evidence against), but **would be a genuinely new experiment if implemented as a structured/convolutional feature-compression layer distinct from both FNO's spectral approach and a naive dense layer** — e.g. a learned 1×1-conv channel squeeze at the teacher's mid-depth, kept structurally intact (not flattened to a vector) to avoid the exact quadratic-cost / spatial-information-loss failure mode Gap 2 already identifies. Recommend framing any feature-compression experiment explicitly against this prior evidence rather than as a fresh idea.

---

## 9. Shared Encoder with Dual Lightweight Heads

**Mechanism:** A single shared backbone produces a shared representation, then two small, separate output heads (one for AoA, one for AoD) branch off, rather than a single monolithic path to a joint 2D heatmap.

**Estimated param effect:** Head parameters are typically small relative to the backbone (a few conv/FC layers each), so total param count is dominated by the shared encoder — could be similar to or slightly larger than the current single-path teacher (469,393 + two small heads vs. one combined head), or smaller if the shared encoder itself is reduced (e.g. combined with item 2 or 3 above). `[INFERENCE]`

**Estimated accuracy effect:** This is architecturally a significant departure from the current pipeline. Note carefully: Lloria's base paper (`[EVIDENCE: PAPER]`) already outputs a **single joint 2D heatmap** (Eq. 9–11) where AoA and AoD are read off the *same* peak location, not from separate heads — this is central to why it counts as genuinely joint (not two independent 1D estimates that happen to be paired). Splitting into dual heads risks **regressing the joint-estimation property itself** unless the heads still share enough representation and are trained with a coupling loss to preserve the correlation between AoA and AoD at a given detected path — otherwise this recreates exactly the kind of "joint in name only" pattern the literature synthesis flags as a red flag in Contradiction #12 (Meneses-Albalá's paper never actually uses the word "joint" despite behaving jointly). `[HYPOTHESIS — this is a genuine risk to flag, not just an efficiency question]`

**Build-on vs. duplicate:** **COMPLETELY UNTRIED in this project.** This is architecturally the biggest deviation from the current single-2D-heatmap paradigm of any item on this list, and — notably — it would sidestep the diagnosed Jacobian-singularity ceiling in a *different* way than the proposed Stage-3 DETR-style set-prediction fix (Gap 5): separate 1D outputs per angle don't suffer the same end-fire 2D-heatmap pixel-quantization geometry, but at the cost of potentially losing the multi-path *pairing* correctness (knowing which AoA goes with which AoD when multiple paths exist) that the current joint 2D representation guarantees implicitly by construction. This should be flagged to the Architecture Researcher (Agent-R1) and Novelty Researcher (Agent-R7) as a **fundamental design-philosophy question, not just a compression technique** — it interacts with the project's core "genuine joint estimation" claim.

---

## 10. Early Exit

**Mechanism:** Attach auxiliary output heads at intermediate depths, allowing "easy" samples (e.g. high SNR, well-separated angles) to exit early without running the full network, reducing *average* inference cost.

**Estimated param effect:** Adds a small number of extra parameters (auxiliary head(s)), net roughly neutral to slightly increasing total param count relative to a non-early-exit version of the same backbone — it's primarily a *latency/compute* technique, not a parameter-count technique. `[INFERENCE]`

**Estimated accuracy effect:** Full-depth performance is generally preserved (the exit heads are typically only used at inference for the confident-exit fraction), so worst-case accuracy shouldn't degrade much for a well-tuned exit threshold `[INFERENCE, general literature]` — but this project's SNR range (per the base paper, −15…24 dB per code / −10…25 dB per paper text `[EVIDENCE: PAPER, Contradiction #4]`) spans conditions where "easy" and "hard" samples differ enormously, meaning an early-exit threshold tuned on this data could plausibly show a real average-latency win with limited average-Pd cost, IF the confidence signal used for exiting is well-calibrated. This is unverified.

**Build-on vs. duplicate:** **COMPLETELY UNTRIED**, and orthogonal to every existing track (pruning, from-scratch architecture screening) — it is a *dynamic inference-time* strategy layered on top of any of the other architectures rather than a static compression of the network itself. Directly relevant to closing **Gap 8** (no standalone inference latency ever reported for this task) since early exit's entire value proposition is expressed in *average* latency, which nobody in the reviewed literature measures at all. Low priority relative to items 6/7 above because it requires a confidence-calibration study that doesn't yet exist in the project, but worth flagging as a distinct, unexplored axis.

---

## 11. Reduced Resolution

**Mechanism:** Downsample the input (Y) and/or output heatmap resolution (e.g. from 256×256 to 128×128 or 64×64), reducing compute roughly quadratically in the linear resolution ratio.

**Estimated param effect:** Conv-layer parameter counts (kernel weights) are resolution-*independent* — this technique mainly reduces FLOPs/activation memory/latency, not stored parameter count. Any param reduction would only come indirectly (e.g. smaller final FC/flatten layers if present). `[INFERENCE]`

**Estimated accuracy effect:** **This is the highest-risk item on the entire list for this specific project**, because the project's own diagnosed core failure mode — a Pd ceiling ≈0.972 caused by pixel-quantization amplified by a Jacobian singularity near end-fire angles (`d(π cosψ)/dψ = −π sinψ`) `[EVIDENCE: PROJECT — ceiling test + diffusion-sharpening ablation]` — is **directly and severely worsened by reducing resolution**, since coarser pixels mean larger angular quantization steps exactly where the model is already most fragile. The diffusion-sharpening ablation already showed that *even attempts to improve effective precision* at fixed 256×256 resolution made Pd worse, not better, because misses are location errors, not blur. Reducing resolution moves in the opposite (wrong) direction from what the project's own diagnostic work indicates is needed.

**Build-on vs. duplicate:** **NOT recommended as a standalone technique for this project** given the above — but flagged because it directly **contradicts** Gap 5's proposed fix (a differentiable DETR-style set-prediction head that removes discrete peak-search entirely, side-stepping the pixel-quantization problem rather than making it coarser). If reduced resolution is pursued at all, it should only be combined with the set-prediction approach (continuous coordinate regression rather than a discrete heatmap grid), not with the current blob-detection pipeline. This is the clearest case among the 12 techniques where a compression technique directly conflicts with an already-diagnosed project-specific failure mode.

---

## 12. Sparse Processing

**Mechanism:** Exploit sparsity in either weights (unstructured/structured sparse pruning, different from the project's existing *channel*-level magnitude pruning) or activations (skip computation on near-zero regions of the input/feature maps), potentially via sparse-conv kernels.

**Estimated param effect:** Unstructured weight sparsity can achieve very high nominal sparsity ratios (50–90%+) `[INFERENCE, general literature]`, but without specialized sparse hardware/kernels this often doesn't translate to real latency wins — a known practical gap between "sparsity ratio" and "actual speedup," which should be flagged explicitly rather than assumed away.

**Estimated accuracy effect:** Unstructured sparsity typically preserves accuracy better than structured (channel) pruning at a matched *nominal* param-reduction ratio, because it can remove individually unimportant weights rather than entire channels — but the practical benefit is capped by hardware support. `[INFERENCE]`

**Build-on vs. duplicate:** **DIFFERENT AXIS from existing pruning work, not a duplicate.** The project's existing pruning (magnitude and SNR-aware) is *structured* (channel-width r=12→r=8), which is what actually yields real speedups on standard hardware. Unstructured sparse processing would be a genuinely new, complementary experiment, but its practical value is contingent on whether the eventual deployment target (Gap 6/8 — real hardware, e.g. a Jetson-class device as in Meneses-Albalá 2026 `[EVIDENCE: PAPER]`) actually supports sparse kernels; this should be resolved with Agent-R6 (Hardware/Deployment Researcher) before investing effort here, since an unsupported sparsity format would be a wasted compression axis in practice despite looking good on paper.

---

## Summary Table

| # | Technique | Param effect | AoA/AoD accuracy effect | Status vs. project |
|---|---|---|---|---|
| 1 | Depthwise separable conv | ~3–6× reduction (est.) | Likely mild-moderate loss; may compound existing ceiling | **Untried** — belongs on architecture-screening track |
| 2 | Bottleneck blocks | ~2–4× reduction (est.) | Likely smaller loss than depthwise at matched ratio | **Untried**, low-risk — closest analog to existing structured pruning |
| 3 | Grouped convolution | Tunable via G | Better retained accuracy than depthwise at moderate G | **Untried** — best risk-adjusted first experiment |
| 4 | Low-rank factorization | Tunable via rank | Risk of compounding diagnosed pixel-quantization ceiling | **Already scoped-out and dropped** (2026-09-21) — do not re-derive |
| 5 | Further pruning (r<8) | ~80–200K depending on r | Non-linear degradation past unknown critical width | **Directly extends** existing pruning infra; recover SNR-aware student first |
| 6 | Quantization | No param-count change; ~4× size | Likely near-lossless for weights/activations | **Completely untried**, cheapest high-value addition (closes Gaps 1, 8) |
| 7 | Knowledge distillation | Fully flexible target size | Plausibly better than scratch/pruning alone | **Completely untried**, highest-recommended new experiment (closes Gap 7) |
| 8 | Feature compression | Depends on placement | Risk of repeating literature's own flagged dense-bottleneck flaw | **Partial duplicate** if dense; new if structured/conv-based |
| 9 | Shared encoder, dual heads | Roughly neutral | Risk of breaking genuine joint-estimation property | **Completely untried** — flag to R1/R7 as design-philosophy question |
| 10 | Early exit | Roughly neutral | Minimal if confidence calibration is sound | **Completely untried**, orthogonal to all other tracks |
| 11 | Reduced resolution | Minimal (FLOPs, not params) | **Likely worsens** the diagnosed Pd ceiling directly | **Not recommended standalone**; conflicts with Gap 5 findings |
| 12 | Sparse processing | High nominal sparsity possible | Likely better than structured pruning at matched ratio | **New axis**, but hardware-support-dependent — check with R6 first |

## Top 3 recommendations (by evidence strength + gap relevance + implementation cost)

1. **Knowledge distillation (item 7)** — zero prior investment, directly closes literature Gap 7 (architecture search × compression never combined), teacher checkpoint already available as distillation source.
2. **Quantization + real-hardware latency benchmark (item 6)** — zero prior investment, directly closes Gaps 1 and 8 (no paper reports params+FLOPs+latency+memory together; no standalone inference latency on real hardware for this task), and requires no retraining — only benchmarking of existing Teacher/Student/FNO checkpoints.
3. **Grouped convolution sweep (item 3)** — cheapest new *structural* experiment, tunable compression ratio, avoids depthwise's steep accuracy cliff, fits into the existing architecture-screening harness without new infrastructure.

**Explicitly deprioritized:** low-rank factorization (already scoped out — re-running would duplicate abandoned work) and reduced resolution (directly conflicts with the project's own diagnosed root-cause finding).

# Agent: robustness

# Agent-R3 — Robustness Stress-Test Plan (Joint AoA/AoD Lightweight Architecture)

**Role:** Principal Research Agent, Robustness Researcher (per `MULTI-AGENT RESEARCH LOOP ENGINEERING PROTOCOL.md` §13)
**Consumes:** Paper Head's Master Synthesis Report + Previous-Result Analysis Agent's report (both supplied)
**Feeds:** Mother Agent → `MASTER_STATE.Robustness Evidence`, `EXPERIMENT_PLAN.md`, `ROBUSTNESS_MATRIX.csv`, and Final Report §24 (Robustness Study)

Every condition below carries an evidence tag per protocol §19 (`[EVIDENCE: PAPER]`, `[EVIDENCE: CODE]`, `[EVIDENCE: EXPERIMENT]` = this project's own prior runs, `[GENUINE GAP]` = nobody has tested this, `[HYPOTHESIS]`). A `COMPARABILITY` note is attached wherever a proposed test could later be quoted against a paper's own number.

---

## 0. Framing note (not a repeat of Notebook 4)

The project has already run one genuine beyond-the-paper robustness experiment: a 4th-path nuisance-interference test (200 scenes × 2 SNR × 3 nuisance-power levels) where the **magnitude-pruned student's principal-path Pd (0.6174) exceeded the teacher's (0.6065)** at 0dB SNR + strongest (0dB) nuisance `[EVIDENCE: EXPERIMENT — PROJECT_STATUS.md §"Notebook 4"]`. This plan does **not** re-run that test. It treats it as Tier-1 prior evidence that compression may not cost — and can even help — robustness under stress, and designs the *remaining* axes (hardware impairment, distribution shift, angular geometry) that Notebook 4 did not cover, so the eventual candidate architecture(s) get a robustness profile no single reviewed paper currently has (per Robustness Matrix §3: no paper covers more than 3/7 axes; this plan targets covering 6/7 for a *joint 2D* estimator, which no paper has done at all).

All runs must use `evaluate_on_bank` (`DL_DOA/src/TVT_Blob_Inference.py`) on frozen banks, and must report **both** the paper-style Pd (drops missed detections from numerator/denominator) **and** a strict per-trial Pd (missed = 0) side by side, per Gap 4 / Contradiction 5 — this dual-metric reporting is itself a literature differentiator no paper in the set provides.

---

## 1. SNR sweep

**Grid:** `{-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35}` dB, in-distribution core = `{-15…24}` dB (5dB steps), extrapolation tails = `{-25,-20}` and `{30,35}` dB.

**Justification:**

| Range | Source | Tag |
|---|---|---|
| −15…24dB (code-trained range) | Lloria TVT2026 **code** (not paper text, which says −10…25dB — Contradiction #4) `[EVIDENCE: CODE]` | this project's data generator inherits this range, so it is the true in-distribution baseline |
| −10…25dB | Lloria PIMRC2024 `[EVIDENCE: PAPER]` | HIGH comparability to Lloria-family numbers |
| −5…30dB (eval), 5…40dB (train) | Naoumi JSTSP2024 `[EVIDENCE: PAPER]` | LOW comparability (different metric, MSE rad² not RMSE/Pd) |
| −9…30dB | Naoumi GCWkshps2023 `[EVIDENCE: PAPER]` | LOW comparability |
| −5…25dB | Gupta TCOMM2025 `[EVIDENCE: PAPER]` | NOT COMPARABLE (not joint, not DL) |
| −10,−5,0,…,25dB (8 pts) | Meneses-Albalá 2026 `[EVIDENCE: PAPER]` | HIGH comparability (same generator family) |
| [−20,10]dB | Sub6GHz CNN/UNet `[EVIDENCE: PAPER]` | LOW (AoA/AoD are classical inputs there, not DL outputs) |

The union of all Tier-1/Tier-2 tested ranges is roughly [−20, 30]dB. The proposed grid matches that union exactly at its core and adds a −25dB and 35dB tail that **no paper in the set tests** `[GENUINE GAP]` — these two points are the actual OOD-SNR distribution-shift condition (§4).

---

## 2. Hardware-impairment conditions

### 2a. Phase-shifter error (δ/γ)
**Grid:** `{0°, 1°, 2°, 5°, 10°, 15°}` (first three are literature-anchored, last two are OOD extrapolation).

- δmax ∈ {1,2,5}° tested by Lloria TVT2026 (Sec J of extraction) `[EVIDENCE: PAPER]` — HIGH comparability, same phase-error injection convention.
- γmax ∈ {1,2,5}° is the **primary focus** of Meneses-Albalá 2026, with fine-tuning shown to help mainly at γ=5° `[EVIDENCE: PAPER]` — HIGH comparability, same underlying DL_DOA generator family.
- {10°, 15°} — **no Tier-1 or Tier-2 paper tests phase error beyond 5°** `[GENUINE GAP]`. This directly probes whether Meneses-Albalá's finding ("gains from adaptation materialize mainly at the most severe *tested* level") continues to hold or the model collapses once impairment exceeds every published test point.

### 2b. Gain mismatch
**Grid:** amplitude error `{0, 0.5, 1, 2, 4} dB` per element (log-normal or uniform, matching the phase-error injection style already in the code path).

- `[GENUINE GAP]` — Meneses-Albalá 2026's own Gap Matrix entry explicitly lists "amplitude-error robustness" as **their own stated future work**, never executed `[EVIDENCE: PAPER — Gap Matrix row 6]`. No Tier-1 paper tests gain mismatch for joint 2D estimation. This is the single most directly-licensed gap to close (the paper that owns this system model names it as unfinished).
- Not comparable to SubspaceNet's "array miscalibration" test (1D DOA, different task family — Cross-paper comparability notes rate this NOT COMPARABLE).

### 2c. Array-element position perturbation
**Grid:** per-element position error σ ∈ `{0, 0.01λ, 0.02λ, 0.05λ, 0.1λ}`.

- `[GENUINE GAP]` for joint 2D AoA/AoD. SubspaceNet is the only paper in the set that explicitly tests array perturbation/miscalibration `[EVIDENCE: PAPER]`, but it is 1D DOA only, rated NOT COMPARABLE to any joint-estimation paper per the synthesis's own comparability table. Chen et al. 2022 (Robust DoA DAE-DNN) tests a related-but-distinct impairment (mutual coupling, ρ≤0.8), also 1D and — per its own re-implementation contradiction (Koh & Lee 2026: 30–70° MAE vs the paper's claimed <0.5°) — a caution against trusting self-reported robustness numbers for this architecture family without independent verification `[EVIDENCE: PAPER, flagged contradiction #6]`. Design point: borrow SubspaceNet's *perturbation-injection mechanism* (report §6, "promising component"), not its numbers.

### 2d. RF-chain / calibration error
**Tier 4 (stretch only).** `[GENUINE GAP]`, not tested by any paper in the set for this task family. Low priority: no literature anchor to justify a specific severity grid, and the project's system model (single-user analog BF, no explicit RF-chain-count variable in the current data generator) would need a new generator parameter before this is even runnable. Flag to Mother Agent as **research gap, not experiment**, pending an Architecture Researcher decision on whether the final candidate models RF-chain count at all.

---

## 3. Channel conditions

### 3a. Number of paths (L)
**Grid:** in-distribution `L ∈ {1,2,3,4,5,6}`, OOD `L ∈ {7,8,10}`.

- L∈{1..6} tested by both Lloria TVT2026 and PIMRC2024 `[EVIDENCE: PAPER]` — HIGH comparability, identical convention.
- Naoumi (both papers) test "#targets-per-peak" (1,2,5) — a related but not identical framing (targets sharing a resolution cell vs. independent multipath count) `[EVIDENCE: PAPER]` — LOW comparability, different unit.
- L∈{7,8,10} — `[GENUINE GAP]`, no paper tests beyond L=6. This is also the project's own currently-unexplored "unseen path count" distribution-shift axis named explicitly in the protocol (§13).

### 3b. Angular separation between paths
**Grid:** Δθ ∈ `{30°, 20°, 10°, 5°, 2°, 1°}`, i.e. sweeping down toward and below the array's Rayleigh resolution limit.

- Naoumi GCWkshps2023 trains/tests inside a narrow fixed [20°,40°] window and **explicitly names "closely-spaced-target stress test" as a missing experiment in its own Gap Matrix entry** `[EVIDENCE: PAPER — Gap Matrix row 4]` — this condition is directly licensed by that paper's own stated gap.
- Lloria's minimum pairwise separation (π/6, i.e. 30°) is a **code-only** convention (Contradiction #11 — not in the paper text) `[EVIDENCE: CODE]`. Testing at and below 30° is therefore testing *below the code's own training floor*, i.e. genuine OOD for the Lloria-family generator this project inherits.
- Below ~5–10°, treat results as diagnostic-only (physically approaching the array's resolution limit; failure here is expected and informative, not necessarily a model defect) — flag this explicitly in reporting per protocol §19 (do not present resolution-limited failure as a robustness defect without the caveat).

### 3c. Path-power imbalance / pseudo-NLOS / dominant-vs-weak-LOS
**Grid:** principal-path-to-secondary-path power ratio ∈ `{0dB (balanced), −5dB, −10dB, −15dB, −20dB (near-NLOS)}`.

- `[GENUINE GAP]` — **no paper in the entire 14-paper set uses an explicit LOS/NLOS categorical framing or power-imbalance sweep** for this task; all Tier-1 multipath models are power-balanced synthetic paths. This is the one condition in the whole plan with zero literature anchor, so treat every result here as exploratory (`[HYPOTHESIS]`-tagged), not literature-comparable.
- Directly reuses and extends the project's own Notebook 4 protocol, which already implements a "sweep-power nuisance path" mechanism `[EVIDENCE: EXPERIMENT]` — the power-ratio grid above is a natural parameterization of the *existing* nuisance-path generator rather than new code, keeping this cheap (~17min per condition-block per the Notebook 4 baseline runtime).
- Correlated/coherent paths (e.g. from a common scatterer) are a related but distinct condition tested only by SubspaceNet (1D, NOT COMPARABLE) `[EVIDENCE: PAPER]` — mark as **Tier 4 / out of scope** unless the final architecture explicitly claims coherent-source robustness.

---

## 4. Distribution-shift conditions

| Condition | Design | Justification | Tag |
|---|---|---|---|
| Unseen SNR | Train on code-inherited [−15,24]dB; **evaluate only** at {−25,−20} and {30,35}dB, never train on these | Gap 3: "zero papers in the Tier-1 set test genuine OOD robustness" `[EVIDENCE: PAPER — Sec J audits]` + this is the protocol's own explicit distribution-shift axis (§13) | `[GENUINE GAP]` |
| Unseen impairment level | Train with δ/γ ≤5° (matching Lloria/Meneses' tested ceiling); **evaluate only** at {10°,15°} | Meneses-Albalá 2026 is the closest approach (fine-tunes across severities up to 5°) but never generalizes past its own trained ceiling `[EVIDENCE: PAPER]` | `[GENUINE GAP]` |
| Unseen path count | Train L≤6; evaluate L∈{7,8,10} | No paper trains/tests beyond L=6 | `[GENUINE GAP]` |
| Unseen angular range / end-fire density probe | Train on the code's actual angle distribution (Uniform[0,π] per Contradiction #1 resolution `[EVIDENCE: CODE]`); evaluate with a **densified** sample near end-fire (θ,φ → 0° and 180°) | Directly targets this project's own diagnosed Jacobian-singularity Pd ceiling (~0.972) — a project-only finding absent from every Lloria-family paper's limitations section despite sharing the identical output representation `[EVIDENCE: EXPERIMENT — ceiling test + diffusion-sharpening ablation]` (Gap 5). This is the highest-value distribution-shift test in the plan because it is the one condition where the project already has a *mechanistic* hypothesis (pixel quantization near end-fire, not blur) to confirm or falsify against the eventual candidate | `[EVIDENCE: EXPERIMENT]` + `[HYPOTHESIS]` |
| Unseen channel realization (coherent/correlated paths) | Optional, stretch only | SubspaceNet tests this for 1D DOA (NOT COMPARABLE) `[EVIDENCE: PAPER]` | `[GENUINE GAP]`, Tier 4 |

---

## 5. Priority tiers (given single-RTX-5090 local compute + occasional Kaggle, and the Kaggle weight-loss incident already logged in prior-work §3 — every Tier 2+ run must be checkpointed with explicit "Save Version → Save & Run All" if run on Kaggle)

**Tier 1 — cheap, literature-anchored, must run for every candidate + baseline registry (Teacher, magnitude-pruned Student, FNO, eventual final candidate):**
1. SNR sweep, in-distribution core [−15,24]dB (5dB steps) — reproduces/extends Lloria & Meneses tables directly, HIGH comparability.
2. Phase error δ/γ ∈ {0,1,2,5}° — reproduces Lloria & Meneses directly, HIGH comparability.
3. Path count L ∈ {1..6} — reproduces Lloria directly, HIGH comparability.

**Tier 2 — moderate cost, project-novel, builds on existing Notebook 4 machinery:**
4. Angular separation sweep Δθ ∈ {30°…1°} — licensed by Naoumi's own named gap.
5. Path-power-imbalance / pseudo-NLOS sweep — extends Notebook 4's existing nuisance-path generator.
6. Unseen-SNR extrapolation tails {−25,−20,30,35}dB — cheap, reuses Tier-1 infrastructure, first genuine OOD test in the whole evidence base.

**Tier 3 — genuine literature gaps, higher implementation cost (new impairment-injection code):**
7. Gain mismatch {0,0.5,1,2,4}dB — directly closes Meneses-Albalá's own named future-work item.
8. Array-element-position perturbation σ ∈ {0.01λ…0.1λ} — adapts SubspaceNet's mechanism to the joint-2D setting.
9. Unseen path count L ∈ {7,8,10} and unseen impairment level {10°,15°}.
10. End-fire density probe — directly tests the project's own Jacobian-singularity hypothesis against the new candidate; run this whenever Stage-3 (differentiable set-prediction head, per `My_Proposed_Architecture.md`) is implemented, since that is the component specifically designed to remove this failure mode.

**Tier 4 — optional/stretch, no literature anchor or out of current system-model scope:**
11. RF-chain/calibration error, coherent/correlated paths, mobility/Doppler (explicitly out of scope: current system model is static estimation, Doppler is Lim et al. 2021's axis on a completely different task — LSTM beam tracking, not comparable per synthesis).

---

## 6. Reporting requirements for every condition (protocol §19 / §23 compliance)

- Every number must carry an evidence tag; nothing labeled a "robustness result" unless it came from an actual run (`[EVIDENCE: EXPERIMENT]`), never inferred or assumed.
- Every cross-paper comparison must carry a COMPARABILITY level (HIGH/MEDIUM/LOW/NOT COMPARABLE), inherited from the synthesis's own comparability table (§1 above) — never silently upgraded.
- Both paper-style and strict per-trial Pd reported at every condition (Gap 4).
- Catastrophic-degradation check (protocol §24 "Robustness" selection criterion): flag any condition where Pd drops by more than, e.g., 50% relative to the matched in-distribution point — this threshold should be confirmed with the Mother Agent/Critical Reviewer before use, since no paper in the set defines a formal catastrophic-failure threshold `[ASSUMPTION — needs Mother Agent sign-off]`.
- Genuine-gap conditions (§2b, §2c, §3c, §4 rows marked `[GENUINE GAP]`) must be explicitly labeled as such in the Final Report's Robustness Study section — these are the plan's actual novelty contribution and must not be quietly presented as "expected" literature reproductions.

---

## 7. Open items to hand back to Mother Agent

1. The catastrophic-degradation threshold (§6) is an assumption, not literature-derived — needs explicit adjudication.
2. Whether Tier 4 conditions (RF-chain mismatch, coherent paths, mobility) are in scope depends on the Architecture Researcher's (Agent-R1) final system-model decision — flagging as a cross-agent dependency, not resolving unilaterally, per protocol §21 (feed-forward rule).
3. The end-fire density probe (§4) should be sequenced *after* Stage-3 differentiable set-prediction is implemented, since it is the condition most likely to distinguish that architectural change from a capacity-only improvement — recommend the Experimental Design Researcher (Agent-R8) place it in the ablation plan alongside a "Stage-3 removed" variant.

# Agent: signal_processing

# Agent-R4 (Signal Processing Researcher) — Front-End Compression Assessment

## Framing

The question is the **inverse** of what the literature actually contains. Every classical+DL hybrid found in the synthesis (§2, Architecture Component Matrix) chains in one direction only: **DL front end → classical back end**.

- **SubspaceNet** (Shmuel et al. 2025): CNN-DCNN autoencoder learns a denoised/regularized surrogate covariance matrix from raw signal, then hands it to an *unmodified* classical subspace algorithm (MUSIC / Root-MUSIC / ESPRIT / MVDR) for the actual angle extraction. Result: 41,761 params (vs >21M for a black-box CNN doing the whole task) and ~60× RMSPE improvement over classical Root-MUSIC alone (12.48°→0.20°, M=2 coherent). The DL only fixes what breaks the classical method's assumptions (coherence, low SNR, miscalibration); the classical algorithm still does the estimation.
- **Koh & Lee 2026**'s DAE+MUSIC entry in their 11-model BLE benchmark is the same pattern: learned denoising autoencoder front end, classical MUSIC back end.

Nobody in the reviewed set runs classical processing **first** to shrink a downstream *learned* estimator. That is a genuine, citable gap (extends Gap 6/7 in the synthesis) — the ideas below have no direct precedent to validate against, only adjacent evidence to reason from. I flag this explicitly rather than overstate confidence.

The strongest adjacent evidence for feasibility is actually **this project's own baseline**, not a paper: the classical 2D-DFT+SIC estimator reproduces **Pd 0.884 vs the 64-block ResNet's 0.925 at 20 dB — 95.6% of the DL model's accuracy, with zero learned parameters** (PROJECT_STATUS.md §3). That ~4pp gap is the *entire* burden currently placed on a 469,393-parameter network. If a classical stage can already close most of the distance, the DL backbone's job can shrink from "estimate the angles" to "correct the residual" — a strictly easier, lower-capacity task.

---

## Candidate 1 — Beamspace/DFT-codebook pre-reduction + residual-correction backbone

**Mechanism:** Project the raw sensor-domain input onto a coarse DFT beamspace codebook (the same P×Q angular grid Lloria's ResNet/U-Net already outputs over) *before* the network, producing a low-resolution coarse heatmap via classical DFT+SIC (already implemented and validated in this project). Feed that coarse heatmap — not the raw antenna-domain tensor — into a much shallower backbone (e.g., 8–16 ResNet blocks instead of 64) whose only job is residual sharpening/de-aliasing around the classical peaks, not full-image angle detection from scratch.

**Precedent:** Naoumi et al. (GCWkshps 2023/JSTSP 2024) already do input-side classical reduction — a coarse-timing IFFT front end that cut multiplications 6.5×/10.3× vs. the full 2D parametric algorithm at Nr=8/16 — though that comparison is DL-vs-classical-cost, not a literal chained hybrid. Sub6GHz CNN/UNet (arXiv 2025) also feeds classically-pre-estimated AoA/AoD as *input* to a CNN/UNet (for a different downstream task, SE gain, not angle refinement) — validating architecturally that DL backbones can profitably consume classical-domain features rather than raw IQ.

**Why it should shrink the backbone specifically:** The dense/flatten bottleneck the synthesis's Gap 2 already flags as the *literature's own self-identified worst component* (Lloria's Table IV quadratic-in-(P'Q') term) exists partly because the network must learn a global mapping from raw-antenna to angle-space. Pre-placing the signal in angle-space via DFT removes that mapping burden entirely — directly attacking the same bottleneck Gap 2 targets, but from the input side rather than the architecture side. This is complementary to, not competing with, the FNO-bottleneck result (Gap 2/PROJECT `RESULTS.md`).

**Risk:** The DFT-SIC classical stage's own O(PQ) cost must be counted in any total-latency claim — Gap 1 already notes *no* Tier-1 paper reports full pipeline params+FLOPs+latency together, so this would need fresh measurement, not an assumed win.

---

## Candidate 2 — SubspaceNet-pattern learned covariance-correction + classical 2D-MUSIC back end, re-purposed for compression instead of accuracy

**Mechanism:** Instead of inverting SubspaceNet's direction, keep its direction (small NN → classical back end) but change the *goal*: SubspaceNet used a tiny NN (41.7K params) to make classical MUSIC accurate enough to compete with a black-box CNN. Here, use the identical pattern — small learned covariance-correction network feeding a **2D-MUSIC** (joint AoA/AoD extension) back end — as a wholesale *replacement* for Lloria's 469K-parameter heatmap network, since 2D-MUSIC's continuous spectral peak search is not confined to a 256×256 pixel grid.

**Why this matters beyond just "smaller":** This directly attacks **Gap 5**, the project's own strongest evidence-convergent finding — the diagnosed Pd ceiling (~0.972) caused by Jacobian singularity/pixel quantization in the heatmap+blob-detection representation, independently confirmed by the ceiling test and the diffusion-sharpening ablation (which ruled out blur, confirming *location* error is the mechanism). A classical subspace back end has no pixel grid to quantize onto — it removes the failure mode's root cause architecturally, as an alternative (or complement) to the project's own not-yet-built DETR-style set-prediction Stage 3.

**Precedent and its limits:** SubspaceNet's 41,761-param model beats classical Root-MUSIC alone by ~60× RMSPE and is dramatically smaller than a 21M-param black-box CNN — direct, verified evidence this pattern scales down parameter count while *improving* accuracy over the pure-classical method, which is exactly the "shrink without sacrificing accuracy" ask. But SubspaceNet is **1D DOA only** (single ULA) — extending the joint-2D-covariance-correction + 2D-MUSIC pipeline to Lloria's joint AoA/AoD problem is unbuilt anywhere in the reviewed literature (this is literally Gap 6, unmodified). Also relevant: Gupta et al. 2025's MUSIC-based Bayesian pipeline is explicitly **not** joint (AoD eliminated by a ZF beamformer) and reports **no hardware-impairment/domain-shift test at all** — so a joint-2D-MUSIC extension inherits an untested robustness gap that would need to be closed before trusting it under Lloria-style phase-error conditions.

**Estimated payoff:** If SubspaceNet's ratio holds even loosely (41.7K vs teacher's 469K is an ~11× reduction while *improving* on the classical baseline), this is the single most aggressive shrink candidate of the three — but it is also the least de-risked, since nothing in the literature validates it for 2 angles simultaneously.

---

## Candidate 3 — Classical coarse-support localization (ESPRIT / 2D-MUSIC) gating a cropped, narrow-field backbone

**Mechanism:** Run a cheap classical coarse localization pass (ESPRIT, or a low-resolution 2D-MUSIC scan) to identify the small angular sub-region(s) likely containing the true peaks — feasible because the project's own data generator already enforces a known minimum pairwise separation (π/6, code-only per Contradiction #11) and L∈{1..6} well-separated sources. Crop the DL backbone's receptive field to operate only on that sub-region (e.g., a 32×32 window instead of the full 256×256 grid) rather than convolving over the whole angular space.

**Precedent:** This is architecturally closest to BeamSeek's single-RF-chain beam-switching constraint, which narrows the estimation problem before any learned component runs — though BeamSeek is 1D-azimuth-only and its own paper reports **no latency numbers at all** (Contradiction #7 explicitly flags a prior pipeline stage as having fabricated BeamSeek latency figures — excluded here). Naoumi's narrow [20°,40°] training/test window (GCWkshps 2023) is an *unintentional* version of the same idea (fixed rather than adaptive cropping), and its own authors flag that the well-separated-targets assumption breaks under limited temporal resolution — a direct warning sign for Candidate 3's closely-spaced-source failure mode.

**Risk (most severe of the three):** Coarse localization errors are compounding, not just costly — if ESPRIT misplaces the candidate window, the cropped backbone never sees the true peak at all (worse than a full-field miss, which a full-field model can sometimes still partially correct). Given the project's own diagnosed Jacobian-singularity problem is *worst near end-fire angles* (Gap 5), a coarse classical localization stage is likely to be *least reliable* exactly where the DL backbone most needs help — this candidate's risk profile is inversely correlated with where it would add value, which is a real concern, not a minor caveat.

---

## Recommendation ranking

| Candidate | Precedent strength | Novelty vs. literature | Directly attacks a named project gap | Risk |
|---|---|---|---|---|
| 1. Beamspace/DFT pre-reduction + residual backbone | Medium (Naoumi, Sub6GHz UNet — adjacent, not identical) | Low-medium | Gap 2 (bottleneck), Gap 1 (report full pipeline cost) | Low — builds on already-validated project DFT-SIC baseline |
| 2. SubspaceNet-pattern covariance-correction + 2D-MUSIC | High for the *pattern* (SubspaceNet verified numbers), zero for the *2D extension* | High (Gap 6, literally unaddressed anywhere) | Gap 5 (Pd ceiling root cause), Gap 6 | Medium — pattern proven at 1D, unproven at 2D+joint AoA/AoD |
| 3. ESPRIT/MUSIC coarse-crop gating | Weak/indirect (BeamSeek, Naoumi's narrow-window as unintentional analog) | Medium | Partial (compute reduction) but works against Gap 5's own failure geometry | High — compounding localization error, worst exactly where it's needed most |

**My assessment as Agent-R4:** Candidate 1 is the safest near-term win — it reuses an already-validated project artifact (DFT-SIC at 95.6% of teacher Pd) and directly targets the literature's own self-flagged bottleneck component (Gap 2), with low implementation risk. Candidate 2 is the highest-ceiling idea and the most scientifically interesting because it could resolve Gap 5 architecturally rather than by patching around it (DETR-style set prediction), but it requires validating SubspaceNet's pattern at 2D joint-angle scale first — nobody has done this, so it should be scoped as an experiment, not assumed to work. Candidate 3 I would deprioritize or drop: its failure mode is anti-correlated with exactly the region (end-fire angles) where the project's evidence says help is most needed.

None of these should be read as "classical replaces DL" — consistent with the brief, all three keep DL as the accuracy-critical component and use classical processing only to reduce what the learned model has to do from scratch.

# Agent: novelty

# Agent-R7 — Novelty/Gaps Assessment (Skeptical Pass)

## 0. Method note and a hard caveat before any verdict

I ran the classification below against the 14-paper synthesis **and** four targeted supplementary web searches (DETR/Hungarian-matching + DOA; joint AoA/AoD lightweight DL 2025–2026; transformer/attention + DOA; classical-subspace + DL hybrids). The supplementary searches surfaced material the 14-paper corpus does **not** cover, some of it directly on-topic. This changes several verdicts from what the synthesis alone would suggest — the corpus's "literature-wide gap" framing (esp. for attention/Transformer use) is partly an artifact of a narrow paper-selection scope, not a true gap in the field. Details in §4. Every "NEW" verdict below is conditional on those items being checked by a dedicated literature agent before anyone writes "no prior work does X."

## 1. Direct answers to the framing questions

**Has "a lightweight hybrid architecture for joint AoA/AoD combining known building blocks" already been published, in spirit?** Yes, effectively. Naoumi et al. (GCWkshps 2023 / JSTSP 2024) already ship a genuinely lightweight (8,584-param) joint AoA/AoD estimator built from a known block (complex-valued MLP) plus a known classical algorithm for input reduction. Meneses-Albalá 2026 already does an edge-oriented, adapted variant of Lloria's joint 2D architecture on real hardware. So the bare claim "we built something lightweight that jointly estimates AoA/AoD by combining known parts" is **not new as a genre** — it's the default move this sub-field has already made twice.

**Is combining two known architectures automatically novel?** No. A reviewer's first question for any "Block A + Block B" paper is: what does the combination do that neither block does alone, and is there an ablation proving the combination — not just the union — is necessary? Simple concatenation/grafting of known components (e.g., "ResNet backbone + attention bottleneck", "pruning + a different backbone") is standard engineering iteration, not a contribution, unless it is (a) motivated by a *specific, diagnosed* failure mode of the existing components, (b) shown via ablation that the combination fixes that failure mode where either component alone does not, or (c) paired with a measurement/rigor contribution nobody else has made. Absent one of these, "combining known blocks" reads as re-labeling.

**What would make a contribution here scientifically meaningful?**
1. Tie the architectural choice to a *specific diagnosed root cause* already evidenced by this project's own experiments (e.g., the Jacobian-singularity/pixel-quantization ceiling), not a generic "let's try attention/FNO" swap.
2. Report a measurement axis genuinely absent from the field (params+FLOPs+**real hardware inference latency**+memory together, for the *joint 2D* task) — this is thin but real, low-effort novelty.
3. Run an evaluation protocol (OOD, multi-seed variance, strict-PD) that the field's self-reported numbers don't support, and show the architecture's ranking changes under it — that's a claim about *evidence quality*, which is defensible even when the architecture itself is unoriginal.
4. Where a combination is used, ablate it against each block alone on the identical task/data, so "hybrid beats components" is demonstrated, not asserted.

## 2. Candidate novelty angles — classification

| # | Angle | Verdict | Justification |
|---|---|---|---|
| A | Generic "lightweight hybrid architecture for joint AoA/AoD, combining known blocks" (as literally stated in the brief) | **KNOWN COMBINATION** | Naoumi (8,584-param complex MLP, already lightweight, already joint) and Meneses2026 (edge-adapted joint 2D net) already occupy this exact genre. Standalone, this is not a defensible thesis contribution. |
| B | Params + FLOPs + **real-hardware standalone inference latency** + memory, reported together for a joint AoA/AoD model, using the project's inference-time/pilot-rate latency framing (Gap 1 + Gap 8) | **PARTIALLY NEW** | No Tier-1/2 paper reports all four together for the joint 2D task. But Meneses2026 already did real-edge-hardware profiling (Jetson) for this paper family — for training/fine-tuning *energy*, not clean inference *latency*. The novel slice is narrow (inference latency specifically, on a genuinely from-scratch-lightweight model) and is a measurement contribution, not an architectural one — fine as a supporting result, thin as a headline claim. |
| C | Replace literature's dense/flatten bottleneck (TVT2026's own self-flagged quadratic term) with FNO/spectral or attention block (Gap 2, Gap 9) | **PARTIALLY NEW, conditional on §4** | The specific weakness (dense bottleneck) and its fix direction (non-dense) are real and unaddressed within the 14-paper corpus, and the project already has an executed FNO result to build on. But §4 found multiple published lightweight/attention-based 2D-DOA and DOA architectures outside this corpus (Triple-Attention 2D-DOA for L-shaped arrays, DACL-Net, two-stage transformer DOA) — "attention works for compact angle estimation" is not a field-wide open question, only an open question *within this narrow joint-AoA/AoD sub-literature*. Frame the contribution as "first systematic dense-vs-non-dense bottleneck comparison for *joint 2D AoA/AoD specifically*," not "first use of attention for angle estimation." |
| D | Differentiable DETR-style set prediction + Hungarian matching to remove the non-differentiable peak-search step diagnosed as the Jacobian-singularity/quantization ceiling cause (Gap 5) | **UNCERTAIN, most promising but unverified** | This is the single most evidence-backed angle *inside* the project (three converging internal experiments point at the same root cause), and nothing in the 14-paper corpus or my supplementary search found DETR/set-prediction applied to DOA/AoA estimation specifically. But my search was not exhaustive — DETR-style set prediction is an extremely popular pattern across detection domains since 2020, and radar/array literature adopts CV techniques quickly; absence-of-evidence here is weak evidence of absence. **Do not claim novelty on this angle without a dedicated, deeper search** (e.g. "query-based DOA", "set prediction radar angle estimation", "learned NMS direction finding") before committing. If confirmed absent, this becomes the strongest angle in the set because it targets a mechanism, not a block swap. |
| E | SubspaceNet-style learned-covariance-surrogate + classical back-end, extended to joint 2D and combined with the project's Gridless-Unfold layer (Gap 6) | **KNOWN COMBINATION at the pattern level, PARTIALLY NEW at the task level** | "Model-based deep learning" (learned front end feeding a classical resolver) is an established paradigm (SubspaceNet is one instance of it; my search found others, e.g. deep-unrolling+GNN hybrids for DOA). Applying an established paradigm to a new task variant (2D instead of 1D) is a domain-transfer contribution, not a conceptual one — legitimate as an engineering result but weak as a "novelty" headline unless paired with the Gap 5 mechanism story. |
| F | Combine architecture search/screening with pruning/compression (Gap 7) | **NOT NOVEL as a general technique; PARTIALLY NEW only as domain-first** | NAS+compression combinations are mature, widely published in general efficient-DL literature. The only possible novelty is "first time applied to joint AoA/AoD," which is a thin, non-mechanistic claim. |
| G | Resolve the SIREN/Window-Attention LR/warmup confound (Gap 9) and report whether attention is viable for this bottleneck | **UNCERTAIN (contingent, not a claim yet)** | This is a diagnostic ablation, not a contribution by itself. It only becomes a novelty-relevant result if it flips the current near-zero-Pd outcome into a positive one **and** that positive result is then shown to beat the literature's dense/flatten baseline specifically for *joint 2D* AoA/AoD (per angle C's caveat — attention-for-DOA broadly is already known elsewhere). |
| H | Strict per-trial PD (missed detections counted as failures) reported alongside the field's own inflated convention (Gap 4) | **PARTIALLY NEW** | Real and easy to establish as absent from the corpus, but it is evaluation-hygiene, not an architectural or algorithmic contribution — useful as a credibility differentiator in a paper, not sufficient as its central claim. |
| I | Multi-seed / cross-training-run statistical significance reporting (Gap 10) | **PARTIALLY NEW** | Same category as H — genuinely absent from this literature, standard practice in ML broadly, so it reads as "finally doing due diligence" rather than a discovery. |
| J | Classical front-end (2D-DFT/MUSIC-style) feeding a lightweight DL refinement stage for joint 2D AoA/AoD (the "classical-front-end+lightweight-DL gap" named in the brief) | **UNCERTAIN** | Within the 14-paper corpus this direction (classical→DL, as opposed to SubspaceNet's DL→classical) is untested for the joint 2D task. But my supplementary search surfaced an explicit hybrid pattern in this exact space ("DNN-based beamformer + MUSIC subspace tracking," alternating DL/classical roles across time), which is close enough in spirit that a dedicated search specifically for "classical-front-end, DL-refinement, joint AoA AoD" is needed before treating this as open. |

## 3. Prior art surfaced outside the given corpus — must be checked before any novelty claim is finalized

The literature-gathering stage scoped itself to 14 papers built around the Lloria/Naoumi/Gupta lineage. A quick supplementary search found several concretely on-topic items **not in that corpus**, which the R7 role is obligated to flag rather than let a downstream "novelty" claim rest on an incomplete search:

- **"Joint Single-Shot AoD/AoA Estimation in mmW Systems and Analysis Under Hardware Impairments" (IEEE, 2023, doc 10279062)** — joint AoA/AoD **and** explicitly studies hardware impairments. This directly overlaps Gap 3 (robustness/OOD) and possibly Gap 1 (the project's "nobody reports X" claims). Architecture/params unknown from title alone — **must be pulled and read** before any claim that "no paper tests joint AoA/AoD under hardware impairment" survives.
- **"Deep Machine Learning-Based AoD Map and AoA Map Construction for Wireless Networks" (IEEE, 2024, doc 10683141)** — another joint AoD/AoA paper missed by the corpus; unknown whether lightweight.
- **"SABER: Symbolic Regression-based Angle of Arrival and Beam Pattern Estimator" (arXiv 2510.26340, Oct 2025)** — very recent, and symbolic regression is by construction far lighter than any NN discussed in the synthesis. Directly threatens any "ours is the lightweight one" framing if not distinguished (different technique family, but same efficiency pitch).
- **Cluster of attention/transformer-based DOA and 2D-DOA papers** (DACL-Net; "Efficient 2D-DOA Estimation Based on Triple Attention Mechanism for L-Shaped Array"; "Two-Stage Transformer Framework for Sparse-Array DOA Estimation"; "Gridless coherent polarization-DOA estimation with dilated neighborhood attention and TCN") — none are joint AoA/AoD, but all directly undercut the synthesis's framing of "zero Tier-1 papers use attention" as evidence of a *field-wide* gap (Gap 9). It is a gap only in this specific narrow sub-literature. The Gridless+attention item is particularly relevant to Gap 6 (Gridless-Unfold combination ideas) and should be pulled.
- **"Hybrid deep unrolling and graph neural networks for super-resolution DOA in physics-informed antenna arrays" (Frontiers, 2026)** — relevant context for the PIA-Net concept: physics-informed deep-unfolding for DOA is not a blue-sky technique family; it already has other instances outside this project.
- Two entries titled close to "Deep Learning Based AoA and AoD Estimation for Millimeter Wave MIMO Systems" / "Deep-Learning-Based AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems" (ACM 2024 / ResearchGate) may be duplicates or close variants of the Lloria paper family already in the corpus — needs a dedup check, not an assumption either way, before being cited as separate prior art.

None of this is definitive — I did not read any of these papers' full text, only titles/abstract snippets — but their existence means the corpus's implicit "we checked the literature" claim is **not exhaustive**, and any of the classifications above (especially C, D, and G, which lean on "the field hasn't tried attention/set-prediction here") could be downgraded further once these are actually read.

## 4. Bottom line

- The idea as stated in the brief ("lightweight hybrid combining known blocks for joint AoA/AoD") is, by itself, **not novel** — it restates work the field (Naoumi, Meneses) has already done.
- The only angle with a real, mechanism-level (not just block-swap) justification is **Gap 5's differentiable set-prediction head**, because it targets a specific, multiply-confirmed internal root-cause diagnosis rather than a generic architecture change — but it is currently **UNCERTAIN**, not confirmed novel, pending a dedicated DETR/set-prediction-for-DOA literature search.
- The **cheapest, most defensible, least risky** novelty available to this project right now is not architectural at all: it is the *measurement and rigor* bundle (Gaps 1, 4, 8, 10 — real inference latency + strict PD + multi-seed variance, reported together for the joint 2D task, which genuinely nobody in the reviewed set does). This should be treated as a supporting pillar of any paper, not its sole contribution, but it is close to zero-risk.
- Before this project commits to an architecture-novelty story, someone should (a) pull and read the four newly surfaced papers in §4, and (b) run a dedicated search for "query-based / set-prediction DOA" and "DETR angle estimation" beyond what this pass covered — both are cheap and materially change the confidence of the D and C/G verdicts above.

# Agent: experiment_design

# Agent-R8 — Experimental Design Researcher: Evaluation Matrix for Final Candidate Architecture

Grounded in the Paper-Head literature synthesis (6 Tier-1 + 8 Tier-2 papers) and the Previous-Result Analysis of `D:\ai_ml_project`, and verified directly against this project's code (`DL_DOA/src/TVT_Blob_Inference.py`, `scripts/generate_frozen_banks.py`, `DL_DOA/src/tvt_data_generation_v3.py`, `baseline_models/README.md`, `notebooks/*`).

**Verification note on the brief itself:** the computed task text asserted FLOPs/latency measurement "does not currently exist in this project's notebooks." Code inspection shows this is *partially* wrong and I'm correcting it rather than passing it upward uncritically (per Mother Agent rule 1–7): `notebooks/proposed pin architecture.ipynb` already has a wall-clock `benchmark_latency()` function (`time.perf_counter`, 20 runs, single-sample batch, TF `.count_params()`), used for ResNet/U-Net/PIA-Net. FLOPs/MACs counting, however, is genuinely absent project-wide (no `thop`/`ptflops`/`fvcore`, no manual FLOP-counting code found anywhere). This distinction matters for what's "reuse" vs "new instrumentation" below.

---

## 0. What the existing pipeline gives you for free vs. what must be built new

`evaluate_on_bank(model, data_arr, feat_arr, meta_arr, batch_size, max_deg_error=1.0)`, redefined identically across `DLDOA_Architecture_FullEval_Local.ipynb`, `DLDOA_Compression_0{1,2,3}*.ipynb`, `snr.ipynb`, built from the original (non-reimplemented) `TVT_Blob_Inference.py` primitives (`get_blob_detector`, `get_blob_peaks`, `peaks_to_angles`, `prepare_for_metric`, `get_ang_difference`, `filter_angles`):

| Given directly by `evaluate_on_bank` on `frozen_banks/eval_bank.npz` (reuse as-is) | Confirmed by code |
|---|---|
| Combined angular RMSE per SNR (8 points, −10…25dB step 5) | `get_ang_difference` flattens ψ (AoA) and φ (AoD) errors together (`H = angle(exp(i·gt)·exp(-i·pred))`, `.flatten()`) — **RMSE is AoA+AoD pooled, not per-axis** |
| Pd per SNR, paper-style convention | `filter_angles(ang_dif_flat, max_deg_error=1.0)` splits good/bad at 1°; trials where `peaks_to_angles` under-detects (`np.isnan(pred).any()`) are `continue`d — **excluded from both numerator and denominator**, reproducing the exact non-standard convention flagged as literature Contradiction 5 / Gap 4 |
| Deterministic, cross-hardware-reproducible (13+ sig-figs, GPU vs CPU) | `baseline_models/README.md` |
| Params via `model.count_params()` | Already used per-model in the registry and in `proposed pin architecture.ipynb` |
| A crude wall-clock inference latency benchmark | `benchmark_latency()` in `proposed pin architecture.ipynb`, TF/Keras only, 1-sample batch, mean±std over 20 runs |

**Fixed by construction in `frozen_banks/eval_bank.npz`** (per `scripts/generate_frozen_banks.py`): `L=[3]` (single path count, no multipath sweep), `P=Q=16`, `nt=nr=16`, `sigma=0.07`, phase/gain error = 0 (the generator is called with no impairment kwargs). **Only the SNR axis varies.** So every metric that isn't "RMSE/Pd vs SNR at L=3, no impairment" needs a **new bank** — not new code, since `validation_data_generator` (imported unmodified from `tvt_data_generation_v3.py`, already supports `error_deg` for both TX and RX beamforming per `create_tx_bf_matrix`/`create_rx_bf_matrix`) already supports it. This is "new instrumentation = a new `generate_*_bank.py` call with different `conditions`," not new physics code.

| Missing from the pipeline entirely (new instrumentation required) | Why |
|---|---|
| Separate AoA-only RMSE / AoD-only RMSE | `get_ang_difference` pools both axes before RMSE; needs a variant that keeps ψ/φ separate before the `sqrt(mean(x²))` |
| MAE | Trivial add (`mean(abs(good_angles))`) alongside existing RMSE — same arrays already computed |
| P95 angular error | Not computed anywhere; must decide whether it's computed over *all* trials (including currently-discarded misses) or only "good" ones — see §6 design note, this materially changes the number |
| Strict/per-trial PD (missed detections counted as failures) | Requires **not** `continue`-ing NaN trials — a ~3-line change to `evaluate_on_bank`, but a real code change, and the two PD numbers must be reported side-by-side per Gap 4 |
| False-detection rate | Not measured — current pipeline only checks "is a true angle matched within 1°," never checks for spurious *extra* peaks (over-detection). Needs new logic against `get_blob_peaks`' raw peak count vs `L` |
| FLOPs/MACs | No tool anywhere in repo; needs `thop`/`ptflops`/`fvcore` (PyTorch) or manual layer-wise FLOP accounting (TF/Keras) added fresh |
| Model size on disk (MB) | Trivial (`os.path.getsize` on `.h5`/checkpoint) but not currently tabulated anywhere except informally |
| Peak/steady memory (RAM/VRAM) | Not measured anywhere; needs `tf.config.experimental.get_memory_info` or `torch.cuda.max_memory_allocated` instrumentation |
| Rigorous end-to-end latency + throughput | Existing `benchmark_latency` is single-sample, TF-only, no batch-size sweep, no CPU-vs-GPU split, no percentile (p50/p99) reporting |
| Multipath (L≠3) sweep | Frozen bank fixes L=3; needs a new bank with `L_VALUES=[1,2,3,4,5,6]` (matches Lloria TVT2026/PIMRC2024's own L∈{1..6} sweep for direct comparability) |
| Angular separation sweep | Not parameterized in current generator calls at all; would need explicit control over path angle spacing at generation time (check whether `validation_data_generator` exposes this — if not, this is genuinely new generator logic, not just new conditions) |
| Phase/gain error sweep | Generator supports `error_deg` (phase) already; **gain error is not implemented anywhere found** (`create_tx_bf_matrix`/`create_rx_bf_matrix` only inject phase, not amplitude, perturbation) — gain-error robustness is a real gap, matches literature Gap: Meneses-Albalá 2026 also flags gain error as future work, so this project would be first to close it |
| Array perturbation (element position/mutual coupling) | Not implemented; SubspaceNet is the only literature reference point (§3 Robustness Matrix) — would need new steering-vector perturbation code |
| Distribution shift (OOD SNR/L/angle-range) | Not implemented as a *held-out* protocol; Notebook 4's nuisance-interference test is the closest existing precedent (novel vs. literature, reusable methodology) but tests interference power, not train/test distribution mismatch |

---

## 1. Accuracy metrics

| Metric | Definition | Reuse vs. new | Comparable to (from literature matrix) | Comparability level |
|---|---|---|---|---|
| **AoA RMSE (ψ only)** | RMSE of matched ψ errors only, degrees | New: split `get_ang_difference` output by axis before pooling | Lloria TVT2026/PIMRC2024 report combined RMSE, not split — so even after building this, cross-paper comparison stays approximate | LOW vs. Lloria (they don't split either); enables project-internal ablation at HIGH confidence |
| **AoD RMSE (φ only)** | Same, φ axis | New (same code path) | Same caveat | Same |
| **Combined RMSE** (current convention) | Pooled ψ+φ angular error, deg | **Reuse directly** — `evaluate_on_bank` | Lloria TVT2026 Table II (ResNet .253, U-Net .230 @20dB); Meneses-Albalá 2026 Table 2 (.27–.28) | **HIGH** vs. Meneses-Albalá (identical generator lineage, identical convention); **MEDIUM** vs. Lloria TVT2026 (same family, PH vs. blob detector differs in PIMRC2024 only — TVT2026 also uses blob) |
| **MAE** | `mean(abs(error))` on same matched-angle arrays | New (trivial, same data) | Naoumi (JSTSP2024/GCWkshps2023) report MSE in rad², not MAE in degrees — unit mismatch | **NOT COMPARABLE** cross-paper (different units/metric family); project-internal only |
| **P95 angular error** | 95th percentile of `\|error\|` | New — **must decide denominator**: all trials (harsher, penalizes misses) vs. matched-only (current pipeline's implicit convention). Recommend reporting **both**, mirroring the strict-vs-paper-style PD split below, since no paper in the set reports P95 at all — this is a project-original robustness-of-tail-error metric | No paper reports P95 | Not comparable to literature; useful as an internal tail-risk indicator, directly relevant to Gap 5 (the Jacobian-singularity end-fire ceiling shows up in tail error, not mean RMSE) |

---

## 2. Detection metrics

| Metric | Definition | Reuse vs. new | Comparable to | Comparability level |
|---|---|---|---|---|
| **PD (paper-style)** | Fraction of matched-angle pairs within 1° threshold, computed only over trials where enough peaks were found (current convention) | **Reuse directly** | Lloria TVT2026 Table II; Meneses-Albalá 2026 Table 2 | HIGH vs. Meneses-Albalá; MEDIUM vs. Lloria TVT2026 (same convention family, per Contradiction 5) |
| **PD (strict/per-trial)** | Same threshold, but missed-detection trials counted as failures in the denominator | New (remove the `continue` on NaN trials; ~3-line change) | No paper reports this — it's the literature-gap-closing metric identified in Gap 4 | Not comparable to any paper directly, but should be reported *alongside* paper-style PD on every table so the gap between the two quantifies how much the paper-style convention inflates results |
| **Missed-detection rate** | 1 − (fraction of GT angles matched at all, regardless of 1° threshold) | New — needs peak-count vs. `L` bookkeeping before the 1° filter | No direct literature analogue at this granularity | Project-internal |
| **False-detection rate** | Fraction of detected peaks beyond `L` per scene (over-detection) | New — not measured anywhere currently | No paper reports this cleanly either (most fix `L` a priori and don't test over-detection) | Project-internal, but relevant since a lighter model's blob detector could hallucinate peaks under noise — this is a genuine failure mode the current pipeline is blind to |

---

## 3. Efficiency metrics

| Metric | Definition | Reuse vs. new | Comparable to | Comparability level |
|---|---|---|---|---|
| **Params** | `model.count_params()` (TF) / `sum(p.numel())` (PyTorch) | **Reuse directly** — already tabulated in `baseline_models/README.md` (Teacher 469,393 / Student 314,513 / FNO 334,321) | Naoumi GCWkshps2023 (8,584, exact); SubspaceNet (41,761, exact) | HIGH as a raw number (params are architecture-agnostic), but **task framing differs** (Naoumi's 8,584-param MLP solves a much lower-dimensional bistatic problem; SubspaceNet's 41,761 is 1D DOA) — report side-by-side but flag task mismatch explicitly, don't claim "beats Naoumi 55×" without that caveat |
| **FLOPs/MACs** | Forward-pass multiply-accumulates | **New** — no tool in repo. Recommend `ptflops`/`thop` for a PyTorch reimplementation, or manual per-layer accounting for the existing TF/Keras models (Conv2D FLOPs = `2 × k_h × k_w × C_in × C_out × H_out × W_out`) | Naoumi GCWkshps2023 (4.321 MMACs fwd, exact); Naoumi JSTSP2024 (headline 6.5×/10.3× reduction vs. classical 2D algo, no absolute number); Lloria TVT2026 (asymptotic O(PQ) only, no concrete FLOPs) | HIGH vs. Naoumi GCWkshps2023 numerically once built (both report a single fwd-pass MACs number); this project would be the *first* to report FLOPs for the Lloria architecture family at all — directly closes Gap 1 |
| **Model size (disk, MB)** | Serialized weights file size | New (trivial — `os.path.getsize`) but not currently tabulated anywhere as a standing metric | No paper reports this | Project-internal, cheap |
| **Peak memory (RAM/VRAM, MB)** | Peak allocation during a forward pass at a fixed batch size | New — needs `tf.config.experimental.get_memory_info('GPU:0')` or `torch.cuda.max_memory_allocated()` instrumentation, none present today | No paper reports this at all (only Meneses-Albalá reports *energy*, not memory) | No literature comparison possible; still needed for the "lightweight" claim in the research objective (§0 of the protocol lists memory explicitly) |
| **Inference latency (ms, single-sample)** | Wall-clock per forward pass | **Partially reuse** — `benchmark_latency()` pattern from `proposed pin architecture.ipynb`; needs generalizing to the current baseline registry's models (currently only wired for ResNet/U-Net/PIA-Net in that one notebook) and to whichever framework the final candidate uses | No Tier-1 paper reports concrete latency (all asymptotic-only); Meneses-Albalá 2026 reports **training/fine-tuning energy on Jetson Orin Nano**, not inference latency | **NOT COMPARABLE** to any paper's number directly — this is Gap 8: reuse Meneses-Albalá's *profiling methodology* (real edge hardware) but for genuine standalone inference latency, since no paper (including theirs) reports it |
| **End-to-end latency** | Forward pass + blob-detection + peak-to-angle conversion (the *whole* pipeline a deployed system would pay) | New — `benchmark_latency` only times the NN forward pass, not `get_blob_peaks`/`peaks_to_angles`, which run on CPU via OpenCV and could dominate at small model sizes | No paper measures this either (the blob-detector cost is universally ignored in the literature) | Project-original metric; directly relevant if Stage-3 differentiable set-prediction (My_Proposed_Architecture.md, targeting Gap 5) replaces the non-differentiable blob detector — this metric is exactly what would demonstrate that stage's efficiency payoff, not just its accuracy payoff |
| **Throughput (samples/sec)** | Batch-size-swept, steady-state | New — no batch-size sweep exists today (`benchmark_latency` is fixed at batch=1) | No paper reports this | Project-internal |

---

## 4. Robustness conditions

| Condition | Frozen-bank coverage today | Reuse vs. new | Literature grounding | Comparability if built |
|---|---|---|---|---|
| **SNR sweep** | Full: −10…25dB, step 5 (8 pts) | **Reuse directly** | Matches Lloria TVT2026's *code* range exactly (paper text says −10…25, code says −15…24 — Contradiction 4); matches Meneses-Albalá 2026's 8-point grid exactly | HIGH vs. Meneses-Albalá; note the text-vs-code range discrepancy explicitly whenever quoting "matches Lloria's range" |
| **Multipath (L)** | Not covered — bank fixes `L=[3]` | New — regenerate bank with `L_VALUES=[1,2,3,4,5,6]`, reusing the same unmodified `validation_data_generator` | Lloria TVT2026/PIMRC2024 both sweep L∈{1..6} | HIGH once built (same generator lineage, same L range) |
| **Angular separation** | Not parameterized at all in current generator calls | New — check whether `validation_data_generator` exposes angle-spacing control; if not, this needs new generator logic, not just new call args (flag before assuming it's cheap) | Naoumi JSTSP2024 sweeps #targets-per-peak (proxy for separation); Naoumi GCWkshps2023 explicitly flags well-separated-target assumption as untested at failure — a project result here would directly speak to a literature-acknowledged weakness | MEDIUM (different system framing — bistatic ISAC vs. this project's single-user analog BF, per literature's own LOW-comparability note on Naoumi-vs-Lloria) |
| **Phase error** | Generator supports it (`error_deg` param, TX+RX), frozen bank has it at 0 | New bank, existing code — reuse `create_tx_bf_matrix`/`create_rx_bf_matrix` unmodified, generate at `δmax∈{1,2,5}°` | Lloria TVT2026 (δmax∈{1,2,5}°, exact match); Meneses-Albalá 2026 (γmax∈{1,2,5}°, PRIMARY focus, exact match) | **HIGH** — literally the same three severity levels as both papers, same generator lineage |
| **Gain error** | **Not implemented anywhere in the codebase** | New — genuinely new physics code (amplitude perturbation on `create_tx_bf_matrix`/`create_rx_bf_matrix`, not present) | Meneses-Albalá 2026 explicitly flags gain error as future work (their own gap); no Tier-1 paper tests it | Would be a literature-first if built — no comparability baseline exists, but strengthens the "this project closes documented gaps" narrative |
| **Array perturbation** | Not implemented | New — steering-vector/element-position perturbation, no existing code found | SubspaceNet only (1D DOA, not joint 2D) | NOT COMPARABLE (different task per literature's own note) — project-internal robustness signal only |
| **Distribution shift** | Not implemented as held-out train/test split; closest precedent is Notebook 4's nuisance-interference test (different axis: interference power, not train/test mismatch) | New protocol — reuse Notebook 4's *methodology* (200 scenes, sweep design) but redefine the swept variable as unseen SNR/L/angle-range rather than nuisance power | Zero Tier-1 papers test genuine OOD (Gap 3); this is the single most literature-differentiating robustness axis available | Not comparable to any paper (none report it) — but exactly the axis that would make this project's evaluation more rigorous than the entire reviewed literature |

---

## 5. Ablation plan (remove one component at a time)

Grounded in `My_Proposed_Architecture.md`'s 3-stage design (Stage 1: physics-informed deep-unfolding sparse recovery; Stage 2: subspace attention/transformer; Stage 3: differentiable DETR-style set prediction + Hungarian matching) and the project's own screened core blocks (FNO, GridlessUnfold, SIREN, WindowAttention):

| Ablation | What's removed | Predicts / tests | Metrics to watch | Reuse vs. new eval |
|---|---|---|---|---|
| **Full candidate** | — (baseline) | — | All of §1–§3 | Reuse `evaluate_on_bank` for accuracy/PD; new for efficiency |
| **− Stage 3 (set-prediction head, revert to blob detector)** | Differentiable DETR-style query + Hungarian matching → back to non-learned peak search | Directly tests whether Stage 3 is what fixes the diagnosed Jacobian-singularity Pd ceiling (~0.972, Gap 5) — the single most evidence-convergent unbuilt idea in the project | PD (paper-style + strict), P95 angular error near end-fire angles specifically (not just mean RMSE — the ceiling is a *tail* effect) | Reuse `evaluate_on_bank` for the blob-detector variant; the set-prediction variant needs a new eval path since its output isn't a heatmap for `get_blob_peaks` to consume |
| **− Stage 2 (subspace attention/transformer)** | Removes attention block, keep Stage 1+3 | Tests whether attention contributes given Gap 9 (zero Tier-1 papers use attention at all) and the project's own unresolved SIREN/WindowAttention confound (§7 item 1 of prior-work analysis — untuned LR/warmup, not a settled negative) | Accuracy metrics + params/FLOPs delta (attention is usually FLOP-heavy relative to its param count) | Reuse accuracy pipeline; FLOPs comparison is new instrumentation either way |
| **− Stage 1 (physics-informed deep-unfolding front end)** | Revert to raw Y as direct backbone input | Tests the PIA-Net dual-branch fusion idea in isolation from the 4 known implementation bugs (`BUJHO_SHOHOJ_VABE.md`) — only valid once Stage 1 is *bug-free and retrained*, since all current PIA-Net numbers are known-stale (unresolved item, §7.4 of prior-work analysis) | RMSE/PD at low SNR specifically (physics priors should help most where noise dominates) | Reuse |
| **− dense/flatten bottleneck → non-dense (FNO/conv) bottleneck** | Swaps the literature's self-flagged worst component (TVT2026 Table IV's only quadratic-in-(P'Q') term) | Direct extension of the project's own already-executed FNO screening result (0.6235 mean Pd, 88% of teacher, from scratch, Gap 2) | Params, PD, RMSE — this ablation already has a partial answer in `RESULTS.md`; treat as replication/extension, not a fresh unknown | Reuse `evaluate_on_bank` fully — no new instrumentation needed, this is the cheapest ablation to run |
| **− pruning (compare full-width vs. pruned candidate)** | Channel pruning (magnitude or SNR-aware) applied post-hoc to the final candidate | Tests Gap 7 — architecture search and compression have never been combined in this project or the literature; also resolves the open loop of recovering/re-running the SNR-aware student (§7.3 of prior-work analysis) at full scale | Params, PD, robustness-test delta under Notebook 4's nuisance-interference protocol (the only point in the project where a compressed model beat its teacher) | Reuse `evaluate_on_bank` + reuse Notebook 4's robustness protocol directly |
| **− learned front end, signal-processing-only front end** | Tests the Signal Processing Researcher's (Agent-R4) question: can a classical front end (DFT codebook / covariance features) shrink the DL model while preserving accuracy | Params, FLOPs, PD | New FLOPs instrumentation required regardless of which ablation |

**Note on ablation validity:** every ablation above must be evaluated on the **identical** `frozen_banks/eval_bank.npz` (or its multipath/phase-error/gain-error extensions once built) with the **identical** `evaluate_on_bank` call, matching the prior-work analysis's explicit warning (§5) not to conflate the stricter per-source Hungarian-matching metric used in early exploratory notebooks with the paper-Table-II-comparable metric used in `baseline_models/`. Ablations involving Stage 3 (set-prediction head) are the one exception requiring a genuinely different eval code path, since its output format isn't a heatmap.

---

## 6. Design decisions this task surfaced that need a Mother Agent / Validation Head ruling before implementation

1. **P95 and MAE denominator**: compute over all trials (including currently-discarded misses, penalizing them at a large fixed error) or only over matched/"good" trials (current implicit convention)? These give materially different numbers. Recommend reporting both, exactly as recommended for PD (paper-style vs. strict), for consistency.
2. **Strict-PD code change**: removing the `continue` on NaN trials in `evaluate_on_bank` changes a function reused by 5+ existing notebooks. Recommend adding a `strict: bool` parameter rather than mutating shared behavior, so all existing paper-comparable numbers in `baseline_models/README.md` remain reproducible unchanged.
3. **Angular-separation sweep feasibility**: unconfirmed whether `validation_data_generator` exposes angle-spacing control at all — this should be checked directly (read `tvt_data_generation_v3.py`'s generator signature in full) before this axis is scheduled as "cheap new instrumentation" in any experiment timeline; it may be classical-algorithm-level new work rather than a new bank-generation call.
4. **Gain-error and array-perturbation robustness** are real new-physics-code items (not present anywhere in the repo), not new-config items like multipath/phase-error — timeline estimates should reflect that difference explicitly.

## Files consulted
`D:\ai_ml_project\DL_DOA\src\TVT_Blob_Inference.py`, `D:\ai_ml_project\scripts\generate_frozen_banks.py`, `D:\ai_ml_project\DL_DOA\src\tvt_data_generation_v3.py`, `D:\ai_ml_project\baseline_models\README.md`, `D:\ai_ml_project\My_Proposed_Architecture.md`, `D:\ai_ml_project\notebooks\proposed pin architecture.ipynb`, `D:\ai_ml_project\notebooks\DLDOA_Architecture_FullEval_Local.ipynb`, `D:\ai_ml_project\MULTI-AGENT RESEARCH LOOP ENGINEERING PROTOCOL.md` (§16).