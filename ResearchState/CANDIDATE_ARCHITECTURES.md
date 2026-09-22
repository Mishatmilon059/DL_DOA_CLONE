# Candidate A (refined after critique: True)

## Final specification

# SpectraSet — Candidate A Architecture Specification (REVISED, post-critique)

*(Architecture Assembly Agent, Candidate A — independent of Candidates B/C — Revision 2, responding to Critique #1 and Critique #2)*

## 1. Name and Motivation

**SpectraSet: FNO→GridlessUnfold Precision Chain with Shared-Query Joint Set-Prediction Head**

The literature synthesis and this project's own diagnostics converge on two separable failure points that no reviewed paper (and no single project screening run) has addressed *together*: (1) the dense/flatten bottleneck is the literature's own self-flagged worst component (TVT2026 Table IV, Gap 2), and (2) the discrete 256×256 heatmap + non-differentiable blob-search output is the diagnosed root cause of the project's own Pd ceiling (~0.972), confirmed by two independent internal experiments (the ceiling test and the diffusion-sharpening ablation, which ruled out blur and confirmed the failure is a *location* error from pixel quantization near end-fire angles, Gap 5). Separately, the project has two building blocks that each partially address one of these problems: FNO (global receptive field, non-dense, reaches 88% of the teacher's Pd at 334K params in the project's screening run — but see the explicit comparability caveat below, added in this revision) and GridlessUnfold (a complex soft-threshold layer that directly targets pixel-quantization by construction, but was starved of depth in its screening run — 8 blocks, 30K params, Pd 0.2216).

**Comparability caveat (new in this revision, responding to Critique #2 item 4):** the "88% of teacher's Pd" figure compares an 8-block, from-scratch-trained FNO against a teacher documented in the project's own `RESULTS.md` as a "64-block, pretrained reference — NOT equal-depth." This is not a like-for-like comparison: the teacher is 8× deeper and benefits from pretraining, not just architecture. It is used here only as a rough ceiling reference to argue FNO is *directionally* competent, not as evidence that FNO closes 88% of a fair capacity/training-budget gap to the teacher. Every subsequent use of this figure in this document (§8) is qualified the same way.

SpectraSet's core bet is that these two problems and these two components are the *same* problem seen from two angles: FNO's global context is good at coarse localization but may be losing fine-grained precision once noise stops dominating (its own undiagnosed high-SNR weakness); GridlessUnfold's sparse-coding mechanism is built to recover exactly that precision but was never given enough capacity or a global context to refine. SpectraSet chains them for the first time — FNO first (global, complex-domain, cheap given Y is already frequency-domain), GridlessUnfold second (precision refinement, still complex-valued) — and then replaces the discrete heatmap output entirely with a small differentiable set-prediction head, so the Jacobian-singularity/pixel-quantization mechanism (Gap 5) cannot occur at inference time by construction, rather than being patched around after the fact. (This specific "by construction" claim — that a discrete grid argmax simply does not exist in the inference path — is the one architectural claim in this document that both critiques left unchallenged; the claim that query-sharing *guarantees correct AoA/AoD pairing* "by construction" is a separate, weaker claim, and is walked back in §3/§9 below.)

## 2. Borrowed Components (with source)

| Component | Source | Evidence tag |
|---|---|---|
| ResNet-style complex-aware conv stem (2 blocks) | Lloria TVT2026 / PIMRC2024 conv shell | [EVIDENCE: PAPER] — used only for the input stem, to keep early-layer comparability with the paper's evaluated shell |
| FNO / spectral-convolution block | This project's own architecture-screening result (RESULTS.md, 334,321 params, Pd 0.6235, 8 blocks) | [EVIDENCE: PROJECT] — no direct precedent in the 14-paper corpus (R1 A9). **Caveat (new):** that screening run used 8 blocks feeding a 256×256 MSE heatmap head; SpectraSet's FNO sub-stage uses 6 blocks feeding a completely different query/matching head, so the screening result is weak prior evidence of viability, not validation of the deployed configuration (see §3 item 1 and §8). |
| GridlessUnfold complex soft-threshold layer | This project's own screening result (30,201 params, Pd 0.2216, 8 blocks), conceptually inspired by PIA-Net's physics-informed deep-unfolding | [EVIDENCE: PROJECT]. **Caveat (new):** the "starved of depth" diagnosis motivating the 8→20 block jump is itself an untested assumption — no intermediate depth (e.g. 16 blocks) or channel/dictionary-size variant has been run to confirm depth, specifically, is the bottleneck rather than the heatmap head/loss mismatch it shared with FNO's screening run. Added as ablation item in §10. |
| Grouped convolution (ResNeXt-style, G=4) | General efficient-CNN literature, not from any reviewed AoA/AoD paper | R2's top-ranked, untried, risk-adjusted structural technique (item 3) |
| Differentiable set prediction (learned queries + cross-attention read-out + Hungarian/greedy matching loss) | DETR-style pattern (outside the 14-paper corpus), sketched as Stage 3 of `My_Proposed_Architecture.md` | [EVIDENCE: PROJECT sketch, unbuilt] + R1's A7 direction; **R7 (novelty) explicitly rates this UNCERTAIN, not confirmed absent from the DOA literature** — flagged, not overstated |
| Fixed 2D sinusoidal positional encoding at the query read-out | **New in this revision**, standard DETR-style component (Vaswani et al. positional-encoding formula, adapted to 2D) | [EVIDENCE: STANDARD PRACTICE] — added directly in response to Critique #2 item 3 (see §3 item 4, §4) |
| Explicit 96→48 channel-reduction (bottleneck) conv at the complex→real handoff | **New in this revision**, standard bottleneck-layer practice | [EVIDENCE: STANDARD PRACTICE] — added directly in response to Critique #1/#2 item 1 (see §3 item 5, §6) |
| Complex-valued representation kept through the physics-motivated stages | Loosely informed by Naoumi's complex-front-end philosophy (R1 A4), but not a literal component import | Adapted-philosophy borrow only |

## 3. Novel Component(s), Stated Precisely

1. **FNO → GridlessUnfold role-split chaining.** FNO occupies the first ~23% of complex-domain trunk blocks by count (6 of 26; see the corrected depth accounting in §6, which replaces the original document's inaccurate "~35%" figure — that number did not match the design's own block counts under any consistent denominator, per Critique #1 item 7); GridlessUnfold occupies the remaining 20 blocks, deepened from 8 in its screening run, and stays complex-valued (not converted to real) so its soft-threshold operator still acts on genuine complex coefficients. This specific chaining — global-then-precision, both never mixed before — has not been tried in this project or in the reviewed literature (Gap 6/7). **Novelty scope narrowed (new):** as Critique #2 correctly notes, the *pattern* of chaining a global block into a local/precision-refinement block is a well-established "coarse-then-fine" design (cascade/refinement networks, model-based unfolding chains such as SubspaceNet's own DL→classical chain). I accept this rebuttal — see the revised §11 — and no longer implicitly lean on "never chained before" as if it were pattern-level novelty; the novelty claim is restricted to the specific instantiation (these two components, this task, this order).
2. **Shared-query joint AoA/AoD head.** A fixed set of N_q=8 learned query embeddings cross-attend once into the final complex→real converted, channel-reduced feature map. Each query feeds three sibling MLPs — objectness, AoA offset, AoD offset — that all read the *identical* query vector. **Claim strength corrected (responding to Critique #1 item 5):** the original document described this as guaranteeing correct AoA–AoD pairing "by construction." That is overstated and is withdrawn. Sharing one query vector across the three heads is an architecturally real, inspectable *inductive bias* toward joint estimation — it is not a mathematical guarantee that the AoA-head and AoD-head outputs for a given query correspond to the same physical path, the way a single-heatmap argmax does. Nothing in the mechanism forbids a query's AoA branch and AoD branch from specializing to different paths under adversarial or dense-path conditions. This is now listed explicitly as failure mode §9.6 ("pairing decoupling"), and an ablation against a dual-head design with independently-pooled (not query-identity-shared) branches is added to §10 to test whether identity-sharing is *measurably* stronger than late-stage information sharing over the same backbone — the original document asserted superiority over the "A5" dual-head design without this ablation; that gap is now closed as an experiment, not an assumption.
3. **Grouped-conv bridge at the complex→real conversion point**, positioned deliberately *after* both physics-motivated (FNO, GridlessUnfold) stages rather than immediately after the stem — moving the complex→real design risk R1 flagged for A4 to the latest defensible point in the pipeline, so both precision-critical stages retain phase information.
4. **Fixed 2D sinusoidal positional encoding, added to the cross-attention keys/values (new component, responding to Critique #2 item 3).** The original document's query read-out attended into a flattened feature map with no positional signal, which the critique correctly identified as a gap: SpectraSet's entire motivation is fixing a *location* error, yet its own new head had no explicit mechanism for queries to know *where* a token came from, relying entirely on implicit position-encoding baked in by preceding conv layers — precisely the kind of unexamined risk the architecture claims to eliminate. A standard fixed sinusoidal 2D positional encoding (no learned parameters, applied as an elementwise add to the 32×32×48 feature map before the K/V projections) is now part of the design. This is deliberately the *fixed, parameter-free* variant rather than a learned positional embedding table (which would cost ≈49K extra params for a 32×32×48 grid) — a conscious param/precision trade-off, stated explicitly rather than left implicit.
5. **Explicit 96→48 channel-reduction bottleneck (new component, responding to Critique #1/#2 item 1).** The original §6/§7 arithmetic silently computed the grouped-conv bridge at C=48 while §4's data flow stated the bridge receives 96 channels (magnitude + phase) — a real, uncaught ~4x costing error correctly identified as fatal by both reviewers. Rather than accepting the bridge at its stated 96 channels (which the reviewers computed would push total params to ≈403K, ~86% of the teacher, eliminating most of the efficiency claim), this revision adds a genuine, explicitly-diagrammed 1×1 conv layer that projects 96→48 channels immediately after the complex→real conversion and before the grouped-conv bridge. This is a real architectural decision, not a bookkeeping fix: it is a deliberate lossy bottleneck, its cost is now correctly counted (§6/§7), and its risk (discarding cross-stage magnitude/phase correlation at the reduction point) is now listed as failure mode §9.7. This is the single largest substantive design change made in response to critique.

## 4. Data Flow

```
Input Y (complex, per-antenna received signal)
  │
  ▼ [Stem — SHARED, complex-valued]
Complex-aware conv stem (2 blocks; ResNet-shell parity for comparability)
  │
  ▼ [FNO stage — SHARED, complex, 6 of 26 complex-domain trunk blocks (23%)]
6 FNO blocks (depthwise spectral conv, k1=k2=8 modes, C=48; complex 1×1 mix) — global receptive field
  │
  ▼ [GridlessUnfold stage — SHARED, complex, precision refinement]
20 deepened complex soft-threshold unfolding blocks (vs. 8 in the screening run)
  │
  ▼ [Complex → real conversion]
Magnitude + phase, 2×C=96 channels
  │
  ▼ [Channel-reduction bottleneck — SHARED, real — NEW]
1×1 conv, 96→48 channels (explicit, costed; see §3 item 5, §6)
  │
  ▼ [Grouped-conv bridge — SHARED, real, now correctly operating at its stated C=48]
8 grouped residual blocks (G=4) + pointwise mix — prepares features for query read-out
  │
  ▼ [Fixed 2D sinusoidal positional encoding — NEW]
Added elementwise to the 32×32×48 feature map (zero learned params)
  │
  ▼ [Query read-out — SHARED]
N_q=8 learned queries, single lightweight cross-attention layer (4 heads, d=48) into the position-encoded feature map
  │
  ├──────────────┬───────────────┬───────────────┐
  ▼              ▼               ▼               ▼
Objectness    AoA branch     AoD branch     (all three MLPs read the SAME
head (MLP)   (MLP, 2 layer) (MLP, 2 layer)   query vector — an inductive
  │              │               │            bias toward joint estimation,
  ▼              ▼               ▼            NOT a structural pairing
existence    ψ̂ᵢ (AoA)       φ̂ᵢ (AoD)          guarantee — see §3.2, §9.6)
score            │               │
  └──────┬───────┴───────┬───────┘
         ▼               ▼
   Hungarian/greedy matching vs. ≤L ground-truth (ψ,φ) pairs
         │
         ▼
   Final output: set of matched (AoA, AoD) pairs — continuous, no heatmap, no blob detector
```

**Shared layers:** stem, FNO stage, GridlessUnfold stage, complex→real conversion, channel-reduction bottleneck, grouped-conv bridge, positional encoding, and the query embeddings/cross-attention are all fully shared — there is no branching until the three per-query heads.
**AoA branch / AoD branch:** the two small sibling MLPs after the shared query read-out.
**Fusion mechanism (claim revised):** identity-sharing of the query vector across all three heads is an inductive bias for joint estimation, reinforced by an optional coupling-loss term (§5) discouraging distinct queries from collapsing to the same path — this is explicitly not claimed to be a structural guarantee equivalent to a single-heatmap argmax (see §3.2).

## 5. Loss Function

After Hungarian (or greedy nearest-cost) matching between predicted query tuples (existence, ψ̂, φ̂) and ground-truth (ψ, φ) pairs:

- **L_obj**: binary cross-entropy / focal loss on the objectness logit (matched query → 1, unmatched → 0)
- **L_angle**: Huber loss on matched (ψ̂−ψ) and (φ̂−φ), using the project's own circular-angle-difference convention (`H = angle(exp(i·gt)·exp(-i·pred))` from `get_ang_difference`) rather than raw subtraction, to avoid wraparound artifacts
- **L_coupling** (optional, DETR-standard trick): penalizes high cosine similarity between distinct matched queries' embeddings, to discourage query collapse onto the dominant path

**L_total = λ_obj·L_obj + λ_angle·L_angle + λ_coupling·L_coupling**, default weights 1 : 5 : 0.1 (angle regression weighted highest since it drives the primary Pd/RMSE metrics; unvalidated, needs a sweep).

No change to the loss formulation itself was required by either critique; both critiques' substantive findings targeted §6–7 arithmetic, §9 failure-mode completeness, and §11 novelty framing.

## 6. Parameter Count Estimate (arithmetic shown, corrected)

**Stated conventions (new — responding to Critique #1 item 2 / Critique #2 item 2, "internally inconsistent complex accounting"):** every layer this document calls complex-valued (stem, FNO spectral+mix, GridlessUnfold) is defined as holding two real weight tensors (real part, imaginary part), each shaped exactly as a real-valued layer with the same channel counts would be. So: **complex params = 2 × real-equivalent params**, where real-equivalent params is computed with the standard real-conv formula. This convention is now applied to *every* complex layer, including the stem and the FNO mix layer, which the original document left at their real-equivalent (unmultiplied) values — the specific inconsistency both reviewers flagged. GridlessUnfold's block cost is not a formula but the project's own measured checkpoint size (30,201 params / 8 blocks), so it already reflects whatever real/complex representation the actual implementation uses and needs no formula adjustment — only the arithmetic slip in scaling it to 20 blocks is fixed (Critique #1 item 7: 30,201/8 = 3,775.125/block, not 3,775; ×20 = 75,502.5, rounded to 75,503, not the original 75,510).

Design hyperparameters (unchanged): trunk channels C=48, FNO uses a depthwise/channel-parametrized spectral-conv variant.

| Stage | Arithmetic | Params |
|---|---|---|
| Stem (complex; real-equiv. 9×2×32+9×32×48=14,400, ×2 for complex) | 14,400 × 2 | 28,800 |
| FNO (6 blocks; spectral already complex per original: 2×48×8×8=6,144; mix now correctly complex: 2×(48×48)=4,608; sum 10,752/block) | 10,752 × 6 | 64,512 |
| GridlessUnfold (20 blocks, measured/scaled, arithmetic slip fixed) | 3,775.125 × 20 | 75,503 |
| **Channel-reduction bottleneck (NEW, real, 96→48, 1×1)** | 96×48 | 4,608 |
| Grouped-conv bridge (8 blocks, G=4, now correctly operating at its stated C=48, justified by the new bottleneck) | per block: (48/4)²×9×4 + 48×48 = 5,184+2,304=7,488; ×8 | 59,904 |
| Positional encoding (fixed sinusoidal, NEW) | no learned parameters | 0 |
| Query embeddings + cross-attention (N_q=8, d=48, 4 heads, +FFN) | 8×48 + 4×(48×48) + 2×(48×96) = 384+9,216+9,216 | 18,816 |
| 3 per-query heads (48→24→1, ×3) | 3×(48×24+24×1) = 3×1,176 | 3,528 |
| **Total** | | **≈255,671 params** |

For context: this is **≈54.5% of the teacher (469,393)**, not the originally claimed ~47% — the corrected complex-accounting and the explicit bottleneck layer together raise the total, but the design remains **below the magnitude-pruned student (314,513, at ≈81% of its size)**, above GridlessUnfold-alone (30,201), and well below both reviewers' worst-case recomputation (≈403K–686K) because the channel-width bug is fixed with a genuine, cheap 4,608-param bottleneck layer rather than by accepting the bridge at its full stated 96-channel cost. Not directly comparable to Naoumi's 8,584-param complex MLP or SubspaceNet's 41,761 (different task framing, per the comparability table).

**Depth accounting corrected (Critique #1 item 7):** FNO is 6 of 26 complex-domain trunk blocks (23.1%), 6 of 34 stem+trunk+bridge blocks (17.6%), and by parameter share within the complex-domain stages, 64,512 / (28,800+64,512+75,503) = 64,512/168,815 ≈ 38.2%. The original "~35% of trunk depth" figure is replaced with these three explicit, individually-labeled figures rather than one unlabeled approximation.

## 7. FLOP Estimate (arithmetic shown, forward pass, single sample, corrected)

Assumed feature-map spatial resolution H×W=32×32=1,024. MACs counted ×2 for FLOPs (real-valued layers). For complex-valued layers, following the convention in §6, a complex multiply-accumulate costs 4 real multiplies (standard (a+bi)(c+di) expansion, no Karatsuba-style 3-multiply optimization assumed — this is a conservative/upper-bound estimate, flagged as such): **complex FLOPs = 4 × real-equivalent FLOPs**, applied to the matrix/conv portions of complex layers. FFT FLOPs in the FNO stage already assume the standard complex-FFT operation count (5·N·log₂N per channel) and are left unscaled since that formula is inherently a complex-domain formula, not a real one needing a multiplier.

| Stage | Arithmetic | FLOPs |
|---|---|---|
| Stem (complex) | real-equiv. ≈29.5M (as original) × 4 | ≈118.0M |
| FNO (6 blocks: FFT already complex ≈4.9M/block, unscaled; mix now correctly complex, real-equiv. 4.72M × 4 = 18.87M/block) | (4.9+18.87)=23.77M/block × 6 | ≈142.6M |
| GridlessUnfold (20 blocks; dominant pointwise term real-equiv. ≈4.72M × 4 = 18.87M, + ≈0.15M threshold term left unscaled as a real-magnitude operation) | 19.02M/block × 20 | ≈380.5M |
| **Channel-reduction bottleneck (NEW, real)** | 2×96×48×1024 | ≈9.4M |
| Grouped-conv bridge (8 blocks, real, now correctly at C=48) | per block: 2×9×12×12×4×1024 + 2×48×48×1024 = 10.6M+4.7M=15.3M; ×8 | ≈122.4M |
| Positional encoding add (NEW, negligible) | 1024×48 elementwise adds | ≈0.05M |
| Cross-attention (N_q=8, HW=1024, d=48) | as original | ≈11.3M |
| Per-query heads | negligible | ≈0.06M |
| **Total** | | **≈784.3M FLOPs (≈0.78 GFLOPs)** |

This is **≈2.4× the original (flawed) 0.32 GFLOPs estimate** — higher than even Critique #2's own corrected figure (≈0.69 GFLOPs), because that recomputation fixed only the channel-width bug (item 1) and not the complex-accounting inconsistency (item 2); this revision fixes both together, per Critique #1's explicit request that "this inconsistency likely under-counts params/FLOPs throughout the physics-motivated stages, not just the bridge." As before, Agent-R8 confirms no FLOP-counting tool exists anywhere in the project's codebase, so this remains a hand-derived estimate that must be replaced with `ptflops`/`thop`/manual-instrumentation measurement before any headline efficiency claim (§10, experiment 7).

### 7.5 Latency (new section — responding to Critique #1 item 6 and Critique #2 item 6, "latency never estimated")

The original document estimated only params/FLOPs and implicitly let that stand in for deployment cost. This is now explicitly flagged as insufficient, and no latency number is fabricated here — the honest position is:

- **Hungarian matching** (used at minimum during training, and at inference if exact bipartite assignment is kept rather than simple thresholding) is polynomial in N_q (O(N_q³) ≈ O(512) per sample at N_q=8, i.e., not a FLOP-scale concern), but it is CPU-bound, non-differentiable, and executed per-sample rather than batched on-GPU like the rest of the network — so its wall-clock cost does not track its FLOP cost and could dominate small-batch latency. A greedy nearest-cost matching variant is a cheaper drop-in for inference-time deployment (already listed as an option in §4) and should be the default outside training.
- **FFT-based spectral convolution (FNO)** typically carries higher real hardware latency per FLOP than dense real convolution, due to kernel-launch and library overhead not captured by a FLOP count — this is a known, general property of FFT-based layers, not a claim specific to this project's hardware, and is flagged rather than quantified here.
- **No wall-clock number is given in this document.** Per Critique #2's own suggested proxy, first-pass latency should be measured on the existing FNO-alone and GridlessUnfold-alone screening checkpoints (which already exist and are runnable) before any "lightweight" or "efficient" claim is finalized. This is now a required item in §10 (experiment 9), not an optional follow-up.

## 8. Expected Advantages and Weaknesses

**Advantages:**
- Directly targets Gap 5 (the single most evidence-convergent project+literature finding) architecturally — no discrete pixel grid exists at inference, so the diagnosed Jacobian-singularity failure mode cannot occur by construction. (This specific claim, unlike the pairing-guarantee claim, was not challenged by either critique and is retained as-is.)
- Reuses two project components with weak prior evidence of viability at the component level (FNO, GridlessUnfold) — but see the corrected framing below; this is a lower-implementation-risk starting point than a from-scratch design, not a validated one.
- Shared-query fusion is a genuine, inspectable inductive bias toward preserving AoA–AoD pairing, now explicitly tested by an added ablation (§10) rather than asserted to be structurally superior to the "A5" dual-head design.
- ≈255.7K params (≈54.5% of the teacher, still below the pruned student's 314,513) is lighter than the teacher and the pruned student even after correcting the arithmetic errors both critiques identified — the direction of the original efficiency claim survives, at a smaller margin than originally stated.

**Weaknesses (revised):**
- Highest implementation complexity of any single-backbone candidate: Hungarian/greedy matching, careful loss-weight tuning, DETR-style training instability documented outside this literature.
- The central hypothesis (that chaining FNO→GridlessUnfold fixes FNO's own undiagnosed high-SNR degradation) is unverified — if that degradation comes from information already discarded at FNO's mode-truncation step, no downstream stage can recover it.
- Chaining three previously-independent components for the first time is a compounding risk: a bad interaction (e.g., under-mixing at the new channel-reduction bottleneck, §3 item 5) could mask or worsen either component's individual behavior in ways neither screening run would reveal. The new bottleneck layer is itself an added risk surface (§9.7), traded deliberately against the alternative of a much larger, unjustified bridge.
- The "reuses validated components" framing is weaker than it reads: FNO's screening result (334,321 params, 8 blocks, heatmap+MSE head) and GridlessUnfold's screening result (30,201 params, 8 blocks, same head) validate those components only in a configuration SpectraSet does not use (different depth, and a completely different output head/loss). This is weak prior evidence of directional viability, not validation of the deployed configuration — restated here per Critique #1 item 3.
- The "88% of teacher's Pd" figure supporting FNO's inclusion is not a controlled comparison (depth- and pretraining-mismatched teacher); restated with this caveat per Critique #2 item 4 rather than used as unqualified support.
- Corrected FLOP estimate (≈0.78 GFLOPs) is markedly higher than originally claimed, and latency (as opposed to FLOPs) remains entirely unmeasured (§7.5) — neither the params nor the FLOPs figures should be read as a proxy for deployed wall-clock cost until experiment 9 (§10) is run.

## 9. Concrete Failure Modes

1. **Query collapse**: with N_q=8 and no strong diversity pressure beyond the optional coupling loss, multiple queries could converge onto the dominant path, leaving weaker/farther paths undetected — expected to surface first under high-L (L≥5) or small angular-separation (Δθ<10°) conditions.
2. **FNO high-SNR degradation persisting**: if it is a mode-truncation information loss (not a downstream-precision problem), the GridlessUnfold stage cannot fix it — this is the single largest unverified assumption in the design.
3. **Complex→real conversion point** (after GridlessUnfold, before the channel-reduction bottleneck and grouped-conv bridge) could still discard cross-stage phase correlations useful downstream, even though it is better-motivated than converting immediately after the stem.
4. **Slow/unstable convergence** typical of small-query set-prediction heads — may need substantially more than the project's standard 20,000-step budget, risking an unfair comparison against the other screened candidates if not step-matched.
5. **Query-capacity ceiling**: N_q=8 covers in-distribution L≤6 with margin, but the robustness plan's OOD path-count sweep (L∈{7,8,10}) exceeds N_q by construction — this candidate cannot even represent more than 8 simultaneous paths without redesign.
6. **Pairing decoupling (new — responding to Critique #1 item 5).** Because shared-query identity is an inductive bias and not a structural guarantee, the AoA branch and AoD branch for a given query could specialize to different underlying paths under partial, non-collapse interference between queries — a failure mode the original document's fusion claim implicitly denied by calling the mechanism "by construction." Expected to surface first under the same dense-path / small-Δθ conditions as query collapse (item 1), but is a distinct failure (mis-pairing survives even where existence detection is correct), and needs its own diagnostic in §10.
7. **Channel-reduction information loss (new — responding to the added §3 item 5 component).** The new 96→48 bottleneck is a single lossy linear projection with no residual/skip path around it. If the magnitude and phase channels carry complementary, non-redundant information that a single 1×1 conv cannot compress losslessly into 48 channels, this could reintroduce a milder version of the precision loss the two physics-motivated stages were meant to preserve — this is the direct cost of resolving the §6/§7 arithmetic bug with a cheap layer instead of a full-width (96-channel) bridge, and should be checked empirically (§10 item 9 below extends to a bottleneck-width sensitivity check).

## 10. Experiments Needed to Validate

1. Ablation: full SpectraSet vs. (− query head, project final features to a heatmap + blob detector) — isolates backbone-chain contribution from head contribution.
2. Ablation: FNO-only backbone + query head vs. GridlessUnfold-only backbone + query head vs. the full chain — since these two components have never been composed before. **Depth-matched control added (responding to Critique #2 item 5):** each solo arm must be run both (a) at its original screening depth (6 or 20 blocks respectively) and (b) expanded to match the full chain's total block count (26 complex-domain blocks), step-count-matched to the full chain's training budget — without (b), a full-chain win cannot be distinguished from simply having ~3-4x more total blocks than either solo screening run, which is exactly the "improvement might just be more parameters" trap the review protocol asks to guard against.
3. High-SNR sweep (15–35dB) specifically testing whether FNO's own documented degradation is resolved by the chaining (the central unverified hypothesis).
4. End-fire density probe (Robustness R3 §4) — the most direct test of whether Gap 5's root cause is actually removed.
5. Path-count (L∈{1..6}, OOD {7,8,10}) and angular-separation (Δθ down to 1°) sweeps — directly tests the query-collapse and query-capacity-ceiling failure modes, and, cross-tabulated per-query (not just per-sample), directly tests the pairing-decoupling failure mode (§9.6).
6. Step-count-matched training comparison against the project's existing from-scratch candidates (FNO, GridlessUnfold, ResNet baseline), per R1's explicit fairness warning for DETR-style heads.
7. Real FLOP/param instrumentation (`ptflops`/`thop` or manual accounting) to replace the estimates in §6–7 with measured numbers before any efficiency claim is finalized.
8. Grouped-conv G sweep (G∈{2,4,8}) at the bridge stage specifically, per R2's recommendation, rather than assuming G=4 is optimal.
9. **New — first-pass latency measurement (responding to §7.5).** Measure wall-clock inference latency on the existing FNO-alone and GridlessUnfold-alone screening checkpoints as a proxy, before any latency or "lightweight" claim is finalized for the full chain; separately time Hungarian-matching overhead at training batch sizes to quantify its CPU-bound cost outside the GPU-side FLOP budget.
10. **New — shared-query vs. independently-pooled dual-head ablation (responding to §3 item 2 / §9.6).** Train a variant identical to SpectraSet except that AoA and AoD branches read independently-pooled features over the same backbone (the "A5"-style design) instead of one shared query vector, and compare pairing-error rate (not just per-angle RMSE) between the two — this is the direct test the original document's fusion claim needed but did not include.
11. **New — GridlessUnfold intermediate-depth check (responding to §2's caveat).** Run GridlessUnfold-alone at 16 blocks (between the screened 8 and the deployed 20) to test whether the "starved of depth" diagnosis is actually depth-limited or whether channel width / dictionary size / the heatmap-head mismatch is the real bottleneck.
12. **New — channel-reduction bottleneck width sensitivity (responding to §9.7).** Sweep the 96→C reduction target (C∈{48, 64, 80, 96-i.e.-no-reduction}) to check whether the chosen 48-channel bottleneck measurably costs Pd/RMSE relative to a wider (more expensive) reduction, quantifying the params/precision trade-off made in §3 item 5 rather than leaving it asserted.

## 11. Novelty Argument — Self-Classified

**UNCERTAIN, leaning PARTIALLY NEW** (narrowed scope relative to the original document, responding to Critique #2's rebuttal).

Justification, applying R7's own explicit framework (combining known blocks is not automatically novel unless tied to a diagnosed root cause *and* shown by ablation to be necessary, not just asserted):

- **Backbone chaining (FNO→GridlessUnfold)**: this exact pair has never been composed, in this project or the reviewed literature (R1's A9 flags it as unbuilt anywhere). However, per Critique #2's correctly-raised rebuttal, the *pattern* — chaining a global learned block into a precision-correction block — is a well-established, mature paradigm (model-based deep learning generally, and SubspaceNet's own DL→classical chaining specifically as a structurally analogous "coarse-then-fine" design). **I accept this rebuttal in full and revise the classification accordingly**: this is **KNOWN COMBINATION at the pattern level** — no longer hedged as "partially new at the pattern level" — and **at most PARTIALLY NEW at the task/instantiation level**, conditional on ablation evidence (§10 item 2) that the specific ordering and role-split (not just "more blocks total") is what drives any observed improvement.
- **Shared-query joint AoA/AoD head**: directly targets a diagnosed root cause (Gap 5) rather than being a generic block swap, satisfying R7's criterion (1) for a meaningful contribution. But two things narrow this further in this revision: (a) the DETR/set-prediction-for-DOA angle itself remains **UNCERTAIN, not confirmed novel** by R7, who explicitly warned *"Do not claim novelty on this angle without a dedicated, deeper search"* — unchanged, still deferred to; and (b) the "by construction" pairing-guarantee language that made this component sound structurally stronger than it is has been withdrawn (§3 item 2, §9.6) — the component is now presented as an inductive bias with an untested comparative advantage over simpler dual-head designs, which is a *weaker* novelty claim than the original document made, not a stronger one.
- **Shared-query identity as the AoA/AoD fusion device** remains a narrow, incremental reconciliation of R2's item-9 risk — not independently sufficient to claim NEW on its own, and now has a concrete ablation (§10 item 10) that did not exist before, rather than being permanently untested.
- **Channel-reduction bottleneck and positional encoding (new components in this revision)** are both standard, off-the-shelf design elements (bottleneck convs, sinusoidal positional encoding) added to fix flaws the critique identified — they are correctness/completeness fixes, not novelty claims, and are not presented as such.
- No ablation proving the combination (not just the union of its parts, and not just more total depth) is necessary has been run — this still fails R7's criterion (2), so the strongest defensible label remains **UNCERTAIN/PARTIALLY NEW, conditional on**: (a) the dedicated set-prediction-for-DOA literature search R7 itself calls for, and (b) the depth-matched ablations in §10 items 2, 10, and 11 actually showing the chained combination — at matched depth and training budget — outperforms either component alone or a simpler dual-head/no-reduction variant.

I am explicitly not claiming NEW. This revision narrows the novelty claim relative to the original document (accepting Critique #2's pattern-level rebuttal in full) rather than defending the original framing, consistent with R7's own unresolved hedge on the component this design leans on most heavily.

---

## Changes made in response to critique

1. **Fixed the fatal channel-width bug (§3 item 5, §4, §6, §7)** — Critique #1 item 1 / Critique #2 item 1. Added an explicit, diagrammed 96→48 channel-reduction bottleneck conv (4,608 params, ≈9.4M FLOPs) instead of letting the grouped-conv bridge silently operate at the wrong channel count. This is a genuine design change, not a footnote: it is now listed as its own data-flow stage, costed, and given its own failure mode (§9.7) and sensitivity-sweep experiment (§10 item 12).
2. **Fixed the internally inconsistent complex-parameter/FLOP accounting (§6, §7)** — Critique #1 item 2 / Critique #2 item 2. Adopted and stated one explicit convention (complex params = 2× real-equivalent; complex FLOPs = 4× real-equivalent for matrix/conv terms) and applied it uniformly to the stem and FNO's mix layer, which the original document had left uncorrected. This raised total params to ≈255.7K (from the flawed 223K) and total FLOPs to ≈0.78 GFLOPs (from the flawed 0.32 GFLOPs) — both corrections go further than either reviewer's own recomputation, since fixing item 2 compounds with item 1.
3. **Re-scoped the "already-validated component" claims (§2, §8)** — Critique #1 item 3. FNO's and GridlessUnfold's screening results are now explicitly caveated as validating a different depth/head/loss configuration than SpectraSet deploys — weak prior evidence, not validation of the actual deployed components.
4. **Added a depth-matched ablation control (§10 item 2)** — Critique #2 item 5. The FNO-only and GridlessUnfold-only ablation arms must now also be run at the full chain's total block count and step budget, so a full-chain win cannot be attributed to "more blocks" alone.
5. **Withdrew the "by construction" pairing-guarantee claim (§3 item 2, §4, §9.6)** — Critique #1 item 5. Reframed shared-query fusion as an inductive bias, added pairing-decoupling as an explicit failure mode, and added a direct ablation against an independently-pooled dual-head design (§10 item 10).
6. **Added latency treatment (§7.5, §10 item 9)** — Critique #1 item 6 / Critique #2 item 6. No latency number is fabricated; Hungarian-matching's CPU-bound/non-batched cost and FFT kernel-overhead are flagged qualitatively, and a first-pass measurement on existing checkpoints is now a required experiment rather than an absent deliverable.
7. **Fixed the minor arithmetic slip and the inaccurate depth-percentage claim (§6)** — Critique #1 item 7. GridlessUnfold scaling corrected to 75,503 params; the "~35% of trunk depth" claim replaced with three explicit, correctly-labeled figures (23.1% of complex-domain blocks, 17.6% of all trunk+bridge blocks, 38.2% by param share).
8. **Added the missing positional encoding (§3 item 4, §4)** — Critique #2 item 3. A fixed, zero-parameter 2D sinusoidal positional encoding was added to the query cross-attention, directly closing the gap between SpectraSet's stated motivation (fixing a location error) and its original head design (a flattened feature map with no positional signal).
9. **Re-stated the teacher comparison with its depth/pretraining caveat (§1, §8)** — Critique #2 item 4. The "88% of teacher's Pd" figure is now explicitly flagged everywhere it is used as a mismatched (8-block vs. 64-block pretrained) comparison, used only as a rough directional reference.
10. **Narrowed the novelty self-classification (§11)** — Critique #2's "what holds up" section. Accepted the rebuttal that global-into-precision chaining is a known pattern-level design (citing model-based deep learning / SubspaceNet's own chaining as the analogous case) and revised the classification to reflect that the pattern-level claim was never defensible as stated, tightening the novelty argument to the task/instantiation level only.
11. **Item not changed, with rebuttal:** Critique #1 item 4 ("confounded ablations mean the motivating narrative is overstated") is addressed by making §1's and §8's language more explicitly conditional throughout (rather than a structural change to the design), and by the new depth-matched ablation (§10 item 2) — I judge the original ablation plan's *design* (§10 items 1–2) was already correctly scoped for this purpose per both critiques' own assessments; what was missing was the depth-matching detail, which is now added, not a redesign of the plan itself.

## Adversarial review

### Critic 1 (fatal_flaw_found=True)

ADVERSARIAL REVIEW — SpectraSet (Candidate A). I verified the candidate's numbers against the project's own files (notebooks/outputs_full_eval/RESULTS.md, baseline_models/README.md, My_Proposed_Architecture.md, PROJECT_STATUS.md). Several of the candidate's cited facts (FNO 334,321 params/Pd 0.6235, GridlessUnfold 30,201/0.2216, Teacher 469,393/0.7103, student 314,513) check out exactly. But the quantitative core of the "lightweight" and "reuses validated components" claims does not survive scrutiny.

1) ARITHMETIC BUG THAT INVALIDATES THE HEADLINE "LIGHTWEIGHT" CLAIM (fatal on its own).
Section 4's data flow states the complex→real conversion produces "2×C=96 channels," and the grouped-conv bridge consumes that output. But §6/§7's arithmetic for the bridge explicitly computes params/FLOPs "approximated at C=48" — i.e. it silently halves the real channel count with no channel-reduction layer shown anywhere in the diagram to justify it. Because both the grouped conv and the 1×1 mix scale ~quadratically in channel width, this is not a ~2x error but a ~4x undercount for that stage:
 - Corrected bridge params: (96/4)²×9×4 + 96×96 = 20,736 + 9,216 = 29,952/block × 8 = 239,616 (vs. the stated 59,904).
 - Corrected total params: ≈223K − 59,904 + 239,616 ≈ 403K — i.e. ~86% of the teacher's 469,393 params, not "~47%," and no longer "above GridlessUnfold-alone, below the student" as claimed — it's essentially on par with the teacher.
 - The same 4x scaling applies to §7's FLOP estimate for the bridge stage (122.4M → ~490M), raising total FLOPs from the claimed ≈0.32 GFLOPs to ≈0.69 GFLOPs.
This single unexplained channel-width substitution is doing most of the work in making SpectraSet look "comfortably lighter than the teacher" in §8; corrected, that advantage largely evaporates. This must be fixed (or justified with an explicit channel-reduction layer) before any efficiency claim is usable.

2) Inconsistent complex/real parameter accounting compounds (1). The FNO spectral weights are explicitly doubled for real/imaginary parts ("2×48×8×8"), but the 1×1 "mix" convolutions inside the same nominally complex-valued FNO and GridlessUnfold stages are costed as plain real ops (48×48, no ×2 or ×4 factor for complex multiplication). Since the architecture states these stages stay complex-valued through 26 of the 34 trunk blocks (§2, §4), this inconsistency likely under-counts params/FLOPs throughout the "physics-motivated" stages, not just the bridge.

3) The "already-validated component" claim for FNO and GridlessUnfold does not transfer to what SpectraSet actually uses. The screening run that produced FNO's 0.6235 Pd / 334,321 params and GridlessUnfold's 0.2216 Pd / 30,201 params (RESULTS.md) used 8 blocks each, trained against an MSE loss on a 256×256×1 Gaussian-blob heatmap — the very output paradigm SpectraSet is designed to eliminate. SpectraSet's own FNO sub-stage uses only 6 blocks and 50,688 params (≈15% of the validated 334,321-param configuration) feeding into a completely different head (learned queries + Hungarian matching, no heatmap at all). Neither the much smaller capacity nor the swapped output head and loss were tested in the cited screening run, so the "88% of teacher's Pd" and "distant second" framing borrow credibility from a result that doesn't describe the component actually being deployed. Likewise, GridlessUnfold's "starved of depth" diagnosis (used to justify jumping 8→20 blocks) is an untested assumption — no intermediate depth (e.g. 16 blocks) was run to confirm depth is really the bottleneck rather than channel width, dictionary size, or the same head/loss mismatch.

4) Confounded to-be-run ablations mean any future improvement claim is pre-compromised. GridlessUnfold's block count is more than doubled (8→20, params 30,201→75,510) in the same step that also adds a global FNO front-end. §10's ablations (items 1–2) are the right design to disentangle "more capacity" from "the chaining hypothesis," but they have not been run yet — so §1's central motivating narrative ("these are the same problem seen from two angles") is presented with more rhetorical confidence than the (candidate's own, honestly flagged) evidence supports. This is self-acknowledged in §8's weaknesses, which is good practice, but the self-classification in §11 should be read as even more provisional than stated once (1)–(3) above are accounted for.

5) The "joint" AoA/AoD claim is weaker than "by construction" implies. Sharing one query embedding across three sibling MLP heads is an inductive bias, not a structural guarantee: nothing mathematically forces the AoA-head and AoD-head outputs for a given query to correspond to the same physical path the way a single-heatmap argmax does by construction. Two independently-pooled branches over a shared backbone feature map (the "A5" design this claims to improve on) already share nearly as much information; the paper doesn't establish that query-vector identity-sharing is meaningfully stronger fusion than that, and no ablation tests it. This exact risk (pairing decoupling under partial, non-collapse query interference) is also absent from §9's failure-mode list, despite being the most direct test of the paper's own headline fusion claim.

6) Latency/system-cost conflation. The spec never estimates wall-clock latency at all — only FLOPs/params. That's a gap relative to the review brief: Hungarian matching (used at least in training, cubic-time, CPU-bound, non-differentiable) and the thresholding/deduplication logic needed to turn N_q=8 query outputs into a final variable-size path set at inference are entirely uncosted, and complex-valued FFT-based spectral convolution (FNO) typically carries higher real hardware latency per FLOP than dense real convs due to kernel/library overhead — none of this is acknowledged, so the "≈0.32 GFLOPs" figure (itself likely wrong per points 1–2) should not be read as a proxy for deployed latency.

7) Minor: the GridlessUnfold reuse multiplication has a small slip (30,201/8 = 3,775.125/block × 20 = 75,502.5, not the stated 75,510), and the claim "FNO occupies the first ~35% of trunk depth" doesn't match the paper's own block counts (6 of 34 trunk blocks = 17.6%; 6 of 26 physics-stage blocks = 23%; even by param share it's ~40%, not 35%).

Recommendation: the core idea (chain a global block into a precision-refinement block, replace the heatmap with set prediction) is a reasonable direction and the self-critique in §8/§11 is unusually honest, but the document's quantitative claims — "lightweight" (~223K/~0.32GFLOPs), "reuses validated components," and "≈35% depth split" — are not currently trustworthy as stated and should not be forwarded as-is. Fix the channel-width bug in §6/§7 (point 1), make complex-value accounting consistent (point 2), and caveat or re-derive the "validated component" claims given the head/loss/capacity mismatch (point 3) before this proceeds to implementation or is compared against other candidates on efficiency grounds.

### Critic 2 (fatal_flaw_found=True)

ADVERSARIAL REVIEW — SpectraSet (Candidate A) — Reviewer #2 verdict: REJECT AS WRITTEN (fixable, but the headline "lightweight" claim as currently computed is not defensible)

I recomputed the arithmetic in §6–7 from the candidate's own stated channel counts rather than trusting its summary rows, and found a channel-count error that materially breaks the document's central efficiency claim. I also found a missing architectural component that undercuts the document's central novelty/motivation claim, and an unflagged comparability problem in the evidence it leans on hardest.

1) FATAL: Grouped-conv bridge params/FLOPs are computed at the wrong channel width (~4x undercount).
The data-flow (§4) and complex→real conversion row (§6) both state the bridge receives 2×C = 96 channels ("Magnitude + phase, 2×C=96 channels"). But the bridge's own param/FLOP rows then silently substitute C=48 ("96→96 approximated at C=48 post-mix") and compute grouped-conv arithmetic on 48 channels, not 96. Grouped-conv cost scales as C²/G, so this is not a rounding choice — it undercounts by exactly (96/48)² = 4x.
- Params: doc's 59,904 → corrected ≈ 8 × [(96/4)²×9×4 + 96×96] = 8 × (20,736 + 9,216) = 239,616. New total ≈ 223,000 − 59,904 + 239,616 ≈ 402,700 params (≈86% of the teacher's 469,393), not the "≈223,000 (~47% of teacher)" claimed, and it now exceeds the magnitude-pruned student (314,513) that §6 claims it beats.
- FLOPs: doc's 122.4M → corrected ≈ 8 × [2×9×24×24×4×1024 + 2×96×96×1024] ≈ 8 × 61.3M ≈ 490.6M. New total ≈ 318M − 122.4M + 490.6M ≈ 686M FLOPs (≈0.69 GFLOPs), more than double the claimed "≈0.32 GFLOPs."
This single error removes most of the parameter/FLOP advantage §8 lists as the design's primary selling point relative to the teacher and the pruned student. Whether SpectraSet is still "lightweight" after this correction is now an open question, not a settled one — the candidate should not proceed to the report stage with the current numbers stated as fact.

2) Compounding, unresolved: complex-valued parameter accounting is internally inconsistent.
FNO's spectral weights correctly carry an explicit ×2 factor for complex (real+imag) coefficients. But the "complex-aware" stem (§6, computed as a plain real conv: 9×2×32 + 9×32×48, no ×2) and FNO's own 1×1 "mix" layer (48×48, no ×2) do not, even though §4 states the entire trunk through GridlessUnfold "stays complex-valued (not converted to real)." If those layers are genuinely complex (separate real/imag weight banks, as complex convolution requires), the stem and FNO-mix rows are each undercounted by roughly 2x too — on top of finding (1). The document needs one consistent, stated convention for what "complex-aware" means parameter-wise, not two different conventions in the same table.

3) Missing component that undercuts the core motivation (Gap 5): no positional encoding for the query read-out.
SpectraSet's entire justification (§1, §8, §9-item… ) is that removing the discrete heatmap fixes a *location* error. But the cross-attention query read-out (§4) attends into a flattened feature map with no mentioned positional embedding. Standard DETR-style set prediction requires positional encodings precisely because a bare flattened feature map + attention gives queries no way to know *where* a token came from. Without it, there is a real risk the new head reintroduces a location-precision problem by a different mechanism (relying entirely on preceding conv layers to bake in enough implicit positional information), which is exactly the failure category this architecture claims to eliminate "by construction." This is a concrete gap the candidate's own §9 failure-mode list does not include, i.e. the failure-mode enumeration is not complete.

4) Fairness problem in the evidence used to motivate keeping FNO: the "88% of teacher's Pd" figure is not a like-for-like comparison, and the document doesn't flag this.
I checked the underlying project result file (notebooks/outputs_full_eval/RESULTS.md): FNO reaches Pd 0.6235 at 334,321 params (8 blocks, trained from scratch), versus the teacher's 0.7103 at 469,393 params — but the teacher is explicitly documented in that same file as "64-block, pretrained reference... NOT equal-depth" (8x deeper, and pretrained rather than trained under the same budget as the screened candidates). SpectraSet's §1/§8 use "88% of teacher's Pd" as a load-bearing reason to keep FNO in the design without surfacing that the teacher comparison is depth- and pretraining-mismatched. This is precisely the kind of comparison the review charge asks to interrogate, and it should be re-stated with the caveat or dropped as supporting evidence.

5) Ablation plan (§10) doesn't isolate "more depth/params" from "the specific chaining," despite §8 admitting this is the central unverified risk.
§10-item 2 proposes FNO-only+head vs GridlessUnfold-only+head vs full chain, but never states whether the "FNO-only" arm is held at 6 blocks (matched to the chain's FNO portion) or expanded to the chain's full ~34-block depth. If it stays shallow, a full-chain win could simply reflect having ~4x more total blocks than either solo screening run (6+20+... vs 8), not a benefit of the FNO→GridlessUnfold ordering itself — the exact "could the improvement just come from more parameters" trap the review is asked to check for. A step-count-matched *and* depth-matched control arm is missing from the experiment list.

6) Latency is never estimated at all (neither pure inference latency nor end-to-end latency), despite being one of the protocol's required efficiency axes and an explicit review question. §7 stops at FLOPs; nothing in the document gives even an order-of-magnitude latency number or flags real hardware measurement as required before any latency claim — this should be listed as a missing deliverable, not silently absent.

What holds up: the joint-AoA/AoD claim is architecturally real (both angle MLPs read one shared query vector, which is a genuine, inspectable joint-estimation mechanism, not just two independently pooled branches); the novelty self-classification (UNCERTAIN/PARTIALLY NEW) is appropriately hedged and, if anything, could be read even more conservatively — chaining a global block into a local/precision-refinement block is a well-established "coarse-then-fine" pattern (e.g., cascade/refinement designs, model-based unfolding chains), so "KNOWN COMBINATION at the pattern level" is at least as defensible as "PARTIALLY NEW," and the document already concedes most of this. The Hungarian-matching loss formulation and query-collapse/query-capacity-ceiling failure modes (§9 items 1 and 5) are honestly reported, including the fact that N_q=8 cannot even represent the OOD L∈{7,8,10} sweep the project's own robustness plan calls for.

Recommendation: do not carry the §6–7 numbers or the "comfortably lighter than the teacher" claim forward into the final report as-is. Require: (a) a recomputation of §6–7 with a single, explicitly stated complex-parameter convention applied consistently to the stem/FNO/bridge, and the bridge computed at its stated 96 channels; (b) an explicit statement of whether the resulting model is still meaningfully lighter than the teacher/pruned student after correction; (c) a positional-encoding component added to the query read-out, or an explicit justification for why it's unnecessary here; (d) the teacher comparison in §1/§8 re-stated with its depth/pretraining caveat; (e) a depth-matched control added to the ablation plan; (f) a first-pass latency number (even measured on the existing FNO/GridlessUnfold screening checkpoints as a proxy) before any "lightweight"/"latency" claim is finalized.

## Original (pre-refinement) specification

# SpectraSet — Candidate A Architecture Specification

*(Architecture Assembly Agent, Candidate A — independent of Candidates B/C)*

## 1. Name and Motivation

**SpectraSet: FNO→GridlessUnfold Precision Chain with Shared-Query Joint Set-Prediction Head**

The literature synthesis and this project's own diagnostics converge on two separable failure points that no reviewed paper (and no single project screening run) has addressed *together*: (1) the dense/flatten bottleneck is the literature's own self-flagged worst component (TVT2026 Table IV, Gap 2), and (2) the discrete 256×256 heatmap + non-differentiable blob-search output is the diagnosed root cause of the project's own Pd ceiling (~0.972), confirmed by two independent internal experiments (the ceiling test and the diffusion-sharpening ablation, which ruled out blur and confirmed the failure is a *location* error from pixel quantization near end-fire angles, Gap 5). Separately, the project has two already-validated but never-combined building blocks that each partially address one of these problems: FNO (global receptive field, non-dense, 88% of teacher's Pd at 334K params, but with an unexplained high-SNR degradation) and GridlessUnfold (a complex soft-threshold layer that directly targets pixel-quantization by construction, but was starved of depth in its screening run — 8 blocks, 30K params, Pd 0.2216).

SpectraSet's core bet is that these two problems and these two components are the *same* problem seen from two angles: FNO's global context is good at coarse localization but may be losing fine-grained precision once noise stops dominating (its own undiagnosed high-SNR weakness); GridlessUnfold's sparse-coding mechanism is built to recover exactly that precision but was never given enough capacity or a global context to refine. SpectraSet chains them for the first time — FNO first (global, complex-domain, cheap given Y is already frequency-domain), GridlessUnfold second (precision refinement, still complex-valued) — and then replaces the discrete heatmap output entirely with a small differentiable set-prediction head, so the Jacobian-singularity/pixel-quantization mechanism (Gap 5) cannot occur at inference time by construction, rather than being patched around after the fact.

## 2. Borrowed Components (with source)

| Component | Source | Evidence tag |
|---|---|---|
| ResNet-style complex-aware conv stem (2 blocks) | Lloria TVT2026 / PIMRC2024 conv shell | [EVIDENCE: PAPER] — used only for the input stem, to keep early-layer comparability with the paper's evaluated shell |
| FNO / spectral-convolution block | This project's own architecture-screening result (RESULTS.md, 334,321 params, Pd 0.6235, 8 blocks) | [EVIDENCE: PROJECT] — no direct precedent in the 14-paper corpus (R1 A9) |
| GridlessUnfold complex soft-threshold layer | This project's own screening result (30,201 params, Pd 0.2216, 8 blocks), conceptually inspired by PIA-Net's physics-informed deep-unfolding | [EVIDENCE: PROJECT] |
| Grouped convolution (ResNeXt-style, G=4) | General efficient-CNN literature, not from any reviewed AoA/AoD paper | R2's top-ranked, untried, risk-adjusted structural technique (item 3) |
| Differentiable set prediction (learned queries + cross-attention read-out + Hungarian/greedy matching loss) | DETR-style pattern (outside the 14-paper corpus), sketched as Stage 3 of `My_Proposed_Architecture.md` | [EVIDENCE: PROJECT sketch, unbuilt] + R1's A7 direction; **R7 (novelty) explicitly rates this UNCERTAIN, not confirmed absent from the DOA literature** — flagged, not overstated |
| Complex-valued representation kept through the physics-motivated stages | Loosely informed by Naoumi's complex-front-end philosophy (R1 A4), but not a literal component import | Adapted-philosophy borrow only |

## 3. Novel Component(s), Stated Precisely

1. **FNO → GridlessUnfold role-split chaining.** FNO occupies the first ~35% of trunk depth (global, low-frequency, complex-domain); GridlessUnfold occupies the back half, deepened from 8 to 20 blocks, and stays complex-valued (not converted to real) so its soft-threshold operator still acts on genuine complex coefficients. This specific chaining — global-then-precision, both never mixed before — has not been tried in this project (the 4-way screening ran each block type independently, never composed) or in the reviewed literature (Gap 6/7).
2. **Shared-query joint AoA/AoD head.** A fixed set of N_q=8 learned query embeddings cross-attend once into the final complex→real converted feature map. Each query feeds three sibling MLPs — objectness, AoA offset, AoD offset — that all read the *identical* query vector. This is the fusion mechanism (§4): because both angle heads share one query identity rather than being independently pooled (as in A5's dual-head design), the AoA–AoD pairing correctness the literature's single-heatmap output guarantees "by construction" is preserved architecturally, directly answering R2's item-9-flagged risk ("splitting into dual heads risks regressing the joint-estimation property") for a query-based split specifically — a reconciliation not discussed by any R1 candidate.
3. **Grouped-conv bridge at the complex→real conversion point**, positioned deliberately *after* both physics-motivated (FNO, GridlessUnfold) stages rather than immediately after the stem — moving the complex→real design risk R1 flagged for A4 to the latest defensible point in the pipeline, so both precision-critical stages retain phase information.

## 4. Data Flow

```
Input Y (complex, per-antenna received signal)
  │
  ▼ [Stem — SHARED]
Complex-aware conv stem (2 blocks; ResNet-shell parity for comparability)
  │
  ▼ [FNO stage — SHARED, complex, ~35% depth]
6 FNO blocks (depthwise spectral conv, k1=k2=8 modes, C=48) — global receptive field
  │
  ▼ [GridlessUnfold stage — SHARED, complex, precision refinement]
20 deepened complex soft-threshold unfolding blocks (vs. 8 in the screening run)
  │
  ▼ [Complex → real conversion]
Magnitude + phase, 2×C=96 channels
  │
  ▼ [Grouped-conv bridge — SHARED, real]
8 grouped residual blocks (G=4) + pointwise mix — prepares features for query read-out
  │
  ▼ [Query read-out — SHARED]
N_q=8 learned queries, single lightweight cross-attention layer (4 heads, d=48) into the feature map
  │
  ├──────────────┬───────────────┬───────────────┐
  ▼              ▼               ▼               ▼
Objectness    AoA branch     AoD branch     (all three MLPs read the SAME
head (MLP)   (MLP, 2 layer) (MLP, 2 layer)   query vector — this identity
  │              │               │            sharing IS the fusion mechanism)
  ▼              ▼               ▼
existence    ψ̂ᵢ (AoA)       φ̂ᵢ (AoD)
score            │               │
  └──────┬───────┴───────┬───────┘
         ▼               ▼
   Hungarian/greedy matching vs. ≤L ground-truth (ψ,φ) pairs
         │
         ▼
   Final output: set of matched (AoA, AoD) pairs — continuous, no heatmap, no blob detector
```

**Shared layers:** stem, FNO stage, GridlessUnfold stage, complex→real conversion, grouped-conv bridge, and the query embeddings/cross-attention are all fully shared — there is no branching until the three per-query heads.
**AoA branch / AoD branch:** the two small sibling MLPs after the shared query read-out.
**Fusion mechanism:** identity-sharing of the query vector across all three heads (not late concatenation of independently-pooled branches), plus an optional coupling-loss term (§5) discouraging distinct queries from collapsing to the same path.

## 5. Loss Function

After Hungarian (or greedy nearest-cost) matching between predicted query tuples (existence, ψ̂, φ̂) and ground-truth (ψ, φ) pairs:

- **L_obj**: binary cross-entropy / focal loss on the objectness logit (matched query → 1, unmatched → 0)
- **L_angle**: Huber loss on matched (ψ̂−ψ) and (φ̂−φ), using the project's own circular-angle-difference convention (`H = angle(exp(i·gt)·exp(-i·pred))` from `get_ang_difference`) rather than raw subtraction, to avoid wraparound artifacts
- **L_coupling** (optional, DETR-standard trick): penalizes high cosine similarity between distinct matched queries' embeddings, to discourage query collapse onto the dominant path

**L_total = λ_obj·L_obj + λ_angle·L_angle + λ_coupling·L_coupling**, default weights 1 : 5 : 0.1 (angle regression weighted highest since it drives the primary Pd/RMSE metrics; unvalidated, needs a sweep).

## 6. Parameter Count Estimate (arithmetic shown)

Design hyperparameters (my own choices, stated explicitly): trunk channels C=48, feature-map spatial size assumed downsampled to H×W not needed for params (only for FLOPs, §7), FNO uses a depthwise/channel-parametrized spectral-conv variant (not full dense C×C mode mixing — necessary to stay lightweight; a dense variant would be ~15–30× larger, noted as a design risk).

| Stage | Arithmetic | Params |
|---|---|---|
| Stem (2 conv blocks, 2→32→48, 3×3) | 9×2×32 + 9×32×48 = 576 + 13,824 | 14,400 |
| FNO (6 blocks, depthwise spectral C=48, k1=k2=8, + 1×1 mix) | per block: 2×48×8×8 (spectral) + 48×48 (mix) = 6,144+2,304=8,448; ×6 | 50,688 |
| GridlessUnfold (20 blocks, reusing project's own measured ~3,775 params/block from the 30,201/8 screening run) | 3,775×20 | 75,510 |
| Grouped-conv bridge (8 blocks, G=4, 96→96 approximated at C=48 post-mix, 3×3 grouped + 1×1 mix) | per block: (48/4)²×9×4 + 48×48 = 5,184+2,304=7,488; ×8 | 59,904 |
| Query embeddings + cross-attention (N_q=8, d=48, 4 heads, +FFN) | 8×48 (table) + 4×(48×48) (QKVO) + 2×(48×96) (FFN) = 384+9,216+9,216 | 18,816 |
| 3 per-query heads (48→24→1, ×3) | 3×(48×24+24×1) = 3×1,176 | 3,528 |
| **Total** | | **≈223,000 params** |

For context (per R2's reference table): this is ~47% of the teacher (469,393), below the magnitude-pruned student (314,513) and FNO-alone (334,321), above GridlessUnfold-alone (30,201). Not directly comparable to Naoumi's 8,584-param complex MLP or SubspaceNet's 41,761 (different task framing, per the comparability table).

## 7. FLOP Estimate (arithmetic shown, forward pass, single sample)

Assumed feature-map spatial resolution H×W=32×32=1,024 (a downsampled representation consistent with a stride-8-reduced grid). MACs counted ×2 for FLOPs.

| Stage | Arithmetic | FLOPs |
|---|---|---|
| Stem | 2×9×2×32×1024 + 2×9×32×48×1024 = 1.18M+28.3M | ≈29.5M |
| FNO (6 blocks: FFT overhead ~O(C·HW·log(HW)) + pointwise mix) | per block ≈4.9M (FFT) + 4.7M (mix) = 9.6M; ×6 | ≈57.6M |
| GridlessUnfold (20 blocks, pointwise-dominated per the param-derived structure) | per block ≈2×48×48×1024 + elementwise threshold ≈4.87M; ×20 | ≈97.4M |
| Grouped-conv bridge (8 blocks) | per block: 2×9×12×12×4×1024 + 2×48×48×1024 = 10.6M+4.7M=15.3M; ×8 | ≈122.4M |
| Cross-attention (N_q=8, HW=1024, d=48) | QKᵀ+attn×V ≈1.57M + K/V/O/Q projections ≈9.6M + FFN ≈0.15M | ≈11.3M |
| Per-query heads | negligible (8 tokens only) | ≈0.06M |
| **Total** | | **≈318M FLOPs (≈0.32 GFLOPs)** |

This is a rough order-of-magnitude estimate built from standard conv/attention FLOP formulas — Agent-R8 confirms **no FLOP-counting tool exists anywhere in this project's codebase**, so this number should be replaced with an actual `ptflops`/`thop`/manual-layer instrumentation pass before being used in any headline claim (also listed under §10, experiment 7).

## 8. Expected Advantages and Weaknesses

**Advantages:**
- Directly targets Gap 5 (the single most evidence-convergent project+literature finding) architecturally — no discrete pixel grid exists at inference, so the diagnosed Jacobian-singularity failure mode cannot occur by construction.
- Reuses two already-validated project components rather than a from-scratch novel block — lower implementation risk than A7 alone.
- Shared-query fusion structurally answers R2's own flagged risk (dual-head designs losing AoA–AoD pairing correctness) without relying solely on an auxiliary loss.
- ~223K params / ~0.32 GFLOPs (estimated) is comfortably lighter than the teacher while retaining more total depth (34 blocks across 3 stages) than either FNO-alone or GridlessUnfold-alone achieved individually.

**Weaknesses:**
- Highest implementation complexity of any single-backbone candidate: Hungarian/greedy matching, careful loss-weight tuning, DETR-style training instability documented outside this literature.
- The central hypothesis (that chaining FNO→GridlessUnfold fixes FNO's own undiagnosed high-SNR degradation) is unverified — if that degradation comes from information already discarded at FNO's mode-truncation step, no downstream stage can recover it.
- Chaining three previously-independent components for the first time is a compounding risk, not merely additive: a bad interaction (e.g., grouped-conv under-mixing at the complex→real handoff) could mask or worsen either component's individual behavior in ways neither screening run would reveal.

## 9. Concrete Failure Modes

1. **Query collapse**: with N_q=8 and no strong diversity pressure beyond the optional coupling loss, multiple queries could converge onto the dominant path, leaving weaker/farther paths undetected — expected to surface first under high-L (L≥5) or small angular-separation (Δθ<10°) conditions.
2. **FNO high-SNR degradation persisting**: if it is a mode-truncation information loss (not a downstream-precision problem), the GridlessUnfold stage cannot fix it — this is the single largest unverified assumption in the design.
3. **Complex→real conversion point** (after GridlessUnfold, before the grouped-conv bridge) could still discard cross-stage phase correlations useful downstream, even though it is better-motivated than converting immediately after the stem.
4. **Slow/unstable convergence** typical of small-query set-prediction heads — may need substantially more than the project's standard 20,000-step budget, risking an unfair comparison against the other screened candidates if not step-matched.
5. **Query-capacity ceiling**: N_q=8 covers in-distribution L≤6 with margin, but the robustness plan's OOD path-count sweep (L∈{7,8,10}) exceeds N_q by construction — this candidate cannot even represent more than 8 simultaneous paths without redesign.

## 10. Experiments Needed to Validate

1. Ablation: full SpectraSet vs. (− query head, project final features to a heatmap + blob detector) — isolates backbone-chain contribution from head contribution.
2. Ablation: FNO-only backbone + query head vs. GridlessUnfold-only backbone + query head vs. the full chain — since these two components have never been composed before.
3. High-SNR sweep (15–35dB) specifically testing whether FNO's own documented degradation is resolved by the chaining (the central unverified hypothesis).
4. End-fire density probe (Robustness R3 §4) — the most direct test of whether Gap 5's root cause is actually removed.
5. Path-count (L∈{1..6}, OOD {7,8,10}) and angular-separation (Δθ down to 1°) sweeps — directly tests the query-collapse and query-capacity-ceiling failure modes.
6. Step-count-matched training comparison against the project's existing from-scratch candidates (FNO, GridlessUnfold, ResNet baseline), per R1's explicit fairness warning for DETR-style heads.
7. Real FLOP/param instrumentation (`ptflops`/`thop` or manual accounting) to replace the estimates in §6–7 with measured numbers before any efficiency claim is finalized.
8. Grouped-conv G sweep (G∈{2,4,8}) at the bridge stage specifically, per R2's recommendation, rather than assuming G=4 is optimal.

## 11. Novelty Argument — Self-Classified

**UNCERTAIN, leaning PARTIALLY NEW.**

Justification, applying R7's own explicit framework (combining known blocks is not automatically novel unless tied to a diagnosed root cause *and* shown by ablation to be necessary, not just asserted):

- **Backbone chaining (FNO→GridlessUnfold)**: this exact pair has never been composed, in this project or the reviewed literature (R1's A9 flags it as unbuilt anywhere) — but R7 classifies the *pattern* of chaining a global learned block into a precision-correction block as a mature, established paradigm (model-based deep learning, e.g., SubspaceNet's own DL→classical chaining). So this is **KNOWN COMBINATION at the pattern level, PARTIALLY NEW at the task/instantiation level** (R7's own classification of the structurally analogous item E).
- **Shared-query joint AoA/AoD head**: directly targets a diagnosed root cause (Gap 5) rather than being a generic block swap, satisfying R7's criterion (1) for a meaningful contribution. But the DETR/set-prediction-for-DOA angle itself is explicitly rated **UNCERTAIN, not confirmed novel** by R7, who ran only a partial supplementary search and explicitly warned: *"Do not claim novelty on this angle without a dedicated, deeper search."* I defer to that hedge rather than overriding it.
- **Shared-query identity as the AoA/AoD fusion device** (both angle heads reading one query vector, rather than independently pooled branches) is a narrow, incremental reconciliation of R2's item-9 risk — not independently sufficient to claim NEW on its own, and untested by ablation.
- No ablation proving the combination (not just the union of its parts) is necessary has been run — this fails R7's criterion (2), so the strongest defensible label right now is **UNCERTAIN/PARTIALLY NEW, conditional on**: (a) the dedicated set-prediction-for-DOA literature search R7 itself calls for, and (b) the ablations listed in §10 items 1–2 actually showing the chained combination outperforms either component alone under matched training budget.

I am explicitly not claiming NEW, consistent with the instruction not to assert novelty without justification, and consistent with R7's own unresolved hedge on the one component (set prediction) this design leans on most heavily.

---

# Candidate B (refined after critique: True)

## Final specification

# Candidate B — IABR-Net: Impairment-Adaptive Beamspace Residual-Correction Network with Coupled Dual-Head Refinement (Revision 2, post-adversarial-review)

## 1. Name and motivation

**IABR-Net** (Impairment-Adaptive Beamspace Residual-correction Network). Every Tier-1 paper in the matrix estimates angles from the raw antenna-domain tensor using a learned model end-to-end (dense/flatten or conv bottleneck → heatmap → blob search). This project's own DFT-SIC classical baseline already reproduces 95.6% of the 469K-parameter teacher's accuracy (Pd 0.884 vs 0.925 @20dB) with **zero learned parameters** [EVIDENCE: PROJECT], which means the current network spends most of its 469K parameters re-deriving a mapping a fixed transform already gets most of the way to. At the same time, the literature's one paper that actually names this system model's calibration weaknesses (Meneses-Albalá 2026) admits it never tested amplitude/gain-mismatch robustness, and no Tier-1 paper tests phase error beyond 5° [EVIDENCE: PAPER, `[GENUINE GAP]` per Agent-R3 §2a/2b]. A purely fixed classical front end (Agent-R4 Candidate 1/A8) inherits exactly this weakness: it has no way to adapt when the true steering vectors deviate from the assumed model.

**Revised motivation framing (see §12, item 2).** Reviewer #1 and Reviewer #2 both correctly show, by direct recomputation of §7's own FLOP table, that the DFT+correction front end costs <0.02% of total FLOPs while the 10-block trunk is >99% of compute *regardless of what feeds it* — so a "the classical front end gives a FLOP efficiency win" claim, as stated in the original draft, is **not supported by the candidate's own arithmetic**, and is withdrawn. What the arithmetic *does* support, and what this revision now argues instead, is a **parameter**-efficiency claim: a hypothetical raw-antenna dense/flatten front end producing an equivalent 16×16×2 representation (512→512 fully connected) would cost 262,656 parameters on its own — more than the entire rest of this model (see §6a and the new §10 ablation). The fixed DFT transform (0 params) plus a small correction module (≈250 params, §3a/§6a) avoids that cost. This is the concrete, arithmetic-grounded efficiency claim this design now makes; the compute-efficiency framing is dropped. The robustness motivation — closing the untested gain-error and >5° phase-error gaps — is unchanged in substance but its language is now hedged consistently with the design's actual current state (see §12, item 3): this capability is **designed to be buildable** on top of the existing pipeline, but is **not yet substantiated**, because gain-error injection code does not exist in the repository today (confirmed by code audit, Critique #2) and must be built as a Phase-0 prerequisite before any gain-robustness claim is made.

This remains a robustness-first, signal-processing-first design, not a backbone-family swap — but its headline efficiency claim is now about parameters, stated with the arithmetic that actually supports it, not compute.

## 2. Borrowed components (with source)

| Component | Source | Role here |
|---|---|---|
| 2D-DFT / beamspace codebook projection (fixed, zero-param) | This project's own DFT-SIC classical baseline, reproducing Gupta TCOMM2025's steering-vector model [EVIDENCE: PROJECT, PAPER] | Front-end feature reduction; **parameter**-efficiency contribution, not FLOP-efficiency (revised, §1/§12) |
| ResNet residual-block shell (conv stem, skip connections) | Lloria TVT2026 / PIMRC2024 [EVIDENCE: PAPER] | Shallow (10-block) residual-correction trunk; unchanged, remains ≈99% of FLOPs and ≈97.6% of params — this dominance is now stated plainly rather than downplayed |
| Joint single 2D heatmap output read by one peak search (not split 1D outputs) | Lloria TVT2026 Eq. 9–11 [EVIDENCE: PAPER] | Preserves the "genuinely joint" property flagged as at-risk by Gap 9/Contradiction 12 |
| Squeeze-excite-style channel attention between blocks | Generic efficient-CNN pattern, no direct paper precedent in the reviewed set (per A3) | Cheap capacity + SNR-correlated diagnostic signal |
| Phase-error injection convention (`error_deg` in TX/RX beamforming matrices) | Lloria TVT2026 code / `tvt_data_generation_v3.py` [EVIDENCE: CODE, PROJECT] | Training-time impairment augmentation grid |
| "Small learned network corrects a classical algorithm's inputs" pattern | SubspaceNet (Shmuel et al. 2025 TVT), 41,761-param covariance-surrogate network [EVIDENCE: PAPER] | Structural inspiration for the correction module (corrects *before* the classical step here, vs. SubspaceNet's *after*, for joint 2D vs. 1D DOA) |
| Deep-Sets / PointNet-style shared-weight per-element MLP with pooled global context | Generic permutation-equivariant set-processing pattern, no direct paper precedent in the reviewed set | **New in this revision** — replaces the original monolithic 64→32→64 MLP to fix the identifiability bottleneck (Critique #1 item 4, Critique #2 item 7; see §3a, §12 item 4) |
| Coarse-to-fine peak refinement via a fixed-size top-K batched crop, not a per-sample dynamic-shape loop | Protocol §11-listed coarse-to-fine pattern (A6), re-engineered for vectorized GPU execution | **Revised in this revision** to make NN-only latency well-defined (Critique #1 item 7, Critique #2 item 3; see §4, §12 item 6) |

## 3. Novel component(s), stated precisely

**(a) Impairment-Adaptive Beamspace Correction module, v2 (IABC-v2) — redesigned for identifiability.**

*Original design and the flaw in it.* The original IABC module was a single monolithic MLP (64→32→64) that pooled magnitude/phase statistics from all 32 antenna elements (16 RX + 16 TX) into one 64-dim vector, squeezed it through a 32-unit hidden layer, and reconstructed all 64 per-element correction outputs (δ, γ for each element) jointly. Both reviewers independently flagged this as an unexamined, tightly-determined inverse problem: recovering 64 real values from a 32-dim bottleneck gives the module no architectural reason to keep per-element corrections disentangled rather than collapsing to a smoothed/averaged correction (Critique #1 item 4; Critique #2 item 7). This critique is accepted as valid, not rebutted — the original design genuinely had no mechanism to prevent this collapse, and it exactly mirrors the pixel-quantization/Jacobian-singularity risk (Gap 5) that the original document flagged elsewhere but failed to apply to its own new module.

*Fix (architectural change, not a defensive paragraph).* IABC-v2 replaces the monolithic MLP with a **shared-weight, per-element MLP** in the Deep-Sets / PointNet family: for each of the 32 antenna elements independently, the *same* small MLP (weights shared across elements) maps that element's own pooled (magnitude, phase) pair — concatenated with a low-dimensional **global context embedding** shared by all elements — to that element's own (δ̂, γ̂) correction. The global context is computed once per sample (mean/std of magnitude and phase pooled over all 32 elements, dim 4) and projected through a tiny shared linear layer to an 8-dim embedding, giving each element's forward pass access to array-wide information (e.g., a coarse SNR proxy) without ever forcing all 64 outputs through one narrow shared bottleneck. Each element's correction is now produced from its *own* 2 inputs + the 8-dim shared context (10 inputs total) through a 16-unit hidden layer to 2 outputs — a per-element inverse problem with a favorable input/output ratio (10→2, not 64→64-through-32), which directly removes the specific failure mode raised in the critique rather than merely acknowledging it.

Trained with a **synthetic paired-consistency loss** (renamed from "self-supervised," see §12 item 5): for each training scene the data generator produces both an impaired realization (random `δ,γ` drawn from the Tier-1 grid) and, at zero extra data-generation cost, the same channel realization with `δ=γ=0`; the IABC-v2 module is trained so its corrected beamspace map matches the clean one. This is ordinary supervised regression against a synthetically-paired target — no external signal is derived from the impaired sample alone — and is a training-time-only signal; no clean reference is available or needed at inference (verified by construction: the module's forward pass at inference takes only the impaired `Y` as input, never the paired clean sample — this must be checked in code review per Failure Mode 5, §9).

This module is **designed to be capable of** addressing Meneses-Albalá's named gain-error gap and extending phase-error testing past every paper's 5° ceiling, **conditional on** gain-error injection code being built first (it does not exist in the repository today — confirmed by code audit in both critiques). This claim is now stated with that condition attached everywhere it appears (§8, §9 Failure Mode 2, §10 item 1), not asserted confidently in §1/§8 while being hedged only in §9/§10 as the original draft did (Critique #2 item 2).

**(b) Path-token coupling bottleneck** *(renamed from "novel fusion mechanism," see §12 item 8)*. Immediately after the trunk, a single shared 1×1-conv "path-token" layer produces one shared feature map that both the coarse joint-heatmap head *and* the two crop-refinement heads read from — the coarse heatmap is retained (not replaced by two independent 1D heads) specifically so the AoA–AoD pairing information the joint 2D representation currently encodes implicitly is not lost, addressing the risk A5/A9's "shared-encoder, split-heads" idea creates per Gap 9's own caution. This is a **standard shared-bottleneck multi-task pattern with index-based pairing**, not a new fusion mechanism — the original draft's confident "novel fusion mechanism" language directly contradicted its own §11 concession that this is "KNOWN COMBINATION at the pattern level," and that inconsistency is corrected here (Critique #2 item 5). What is being claimed is narrower and more defensible: this specific combination, applied to preserve joint AoA/AoD pairing while adding local continuous refinement, is not present in the reviewed corpus (see §11).

**(c) Peak-conditioned local crop-refinement heads over a fixed-size, vectorized top-K gather** *(revised for latency measurability, see §12 item 6)*. Unlike A6 (which coarsens the *entire* grid, risking closely-spaced-source collapse), a small window is cropped from the path-token map around each detected coarse peak and fed to two lightweight per-angle regression heads that output a continuous sub-pixel offset. The original draft described this as operating "around each coarse peak" without specifying how a data-dependent, variable count of peaks (L=1–6, up to 10 OOD) is handled in a batched GPU forward pass — Critique #2 item 3 correctly identifies this as a latency-measurability gap: a dynamic-shape/gather-per-peak pattern's real GPU latency is not captured by a static FLOP estimate, and if implemented as a Python loop it becomes CPU-bound and would silently corrupt any "NN-only latency" claim. This revision specifies the implementation precisely: refinement always operates over a **fixed-size top-K=10 batched gather** (the maximum L supported, matching the Tier-3 OOD ceiling), with unused slots masked and excluded from the loss and from peak search output; this keeps the refinement stage a single fixed-shape, vectorized GPU operation regardless of the true L, making its latency well-defined and reproducible. This differs from A6/A7 by leaving the coarse detection stage untouched (still blob-search-compatible) and only adding local continuous correction, keeping compatibility with the existing `evaluate_on_bank` pipeline with a small, well-specified extension.

## 4. Data flow

```
Input Y (raw complex antenna-domain signal, Nr x Nt = 16x16)
   │
   ├─► [FIXED] 2D-DFT beamspace projection F_codebook^H · Y  →  Y_bs  (P×Q=16×16, complex, 0 params)
   │
   ├─► [LEARNED, per-element shared-weight, IABC-v2]
   │        Global context: pool (mean|mag|, mean∠, std|mag|, std∠) over all 32 elements → FC(4→8)
   │        Per-element (×32, shared weights): own (mag,phase) [2] ⊕ context [8] → FC(10→16)→ReLU
   │              → FC(16→2) → (δ̂ phase, γ̂ gain) for this element
   │        → applied multiplicatively (per-element, TX/RX) to Y_bs
   │        →  Y_bs_corrected (P×Q, 2-channel real/imag)
   │
   ├─► [SHARED TRUNK] 10-block shallow ResNet shell (C=32, 3x3 convs + SE channel-attention per block)
   │        operating on Y_bs_corrected, spatial size held at 16x16 throughout (no down/upsampling —
   │        classical front end already put the signal in angle-space, so no global-mapping burden;
   │        this trunk is ≈97.6% of params and ≈99% of FLOPs — stated plainly, not downplayed, §1/§12)
   │        →  shared feature map F_shared (16×16×32)
   │
   ├─► [FUSION — shared-bottleneck, index-paired, not a novel mechanism, §3b] Path-token 1x1 conv
   │        (32→32) on F_shared → F_token (16×16×32), read by all heads below
   │
   ├──────────────┬───────────────────────────────┬─────────────────────────────┐
   │              │                                │                             │
   ▼              ▼                                ▼                             ▼
[Coarse joint   [AoA crop-refine head]        [AoD crop-refine head]      [SE-attention→SNR
 2D heatmap      Fixed top-K=10 batched         Fixed top-K=10 batched      diagnostic head]
 head]           7x7 gather around each         7x7 gather around each      pooled SE gate
 1x1 conv        of the top-10 coarse-          of the top-10 coarse-       → 1 scalar, aux-only
 32→1, blob-     score locations (padded/       score locations (padded/
 detector-       masked if true L<10)           masked if true L<10)
 compatible      → 1x1 conv 32→16 → FC          → 1x1 conv 32→16 → FC
                 → Δψ per slot (masked slots    → Δφ per slot (masked slots
                   excluded from loss/output)     excluded from loss/output)
   │                    │                                │
   ▼                    ▼                                ▼
Coarse (ψ,φ) via   ψ_final = ψ_coarse + Δψ        φ_final = φ_coarse + Δφ
 blob search        (only for unmasked slots)      (only for unmasked slots)
   │                    │                                │
   └────────────────────┴──────────── AoA output ────────┴──── AoD output
                    (paired by shared coarse peak index — joint pairing preserved)
```

The fusion mechanism remains index-based pairing through a shared bottleneck (§3b) — this is stated as a known, standard pattern, not claimed as novel (correcting the original draft's inconsistency, Critique #2 item 5).

## 5. Loss function

```
L = λ1 · L_heatmap        (BCE/MSE, joint 2D coarse heatmap vs Gaussian-blurred GT peak map — Lloria-style)
  + λ2 · L_offset          (Smooth-L1 on Δψ, Δφ at each matched, unmasked top-K slot, nearest-GT
                             assignment valid since L≤10 (top-K ceiling) well-separated sources —
                             no Hungarian matcher needed)
  + λ3 · L_pair_consist    (MSE between IABC-v2-corrected beamspace map and the clean (δ=γ=0)
                             beamspace map for the same channel realization — SYNTHETIC PAIRED
                             SUPERVISION, training-time only; renamed from "self-supervised," §12
                             item 5 — this is ordinary regression against a simulator-paired target,
                             not self-supervision in the standard sense)
  + λ4 · L_SE_snr           (small-weight auxiliary regression: pooled SE-gate activation → true per-
                             sample SNR label; ablatable, tests A3's "attention correlates with SNR"
                             hypothesis directly; does not affect angle outputs if ablated)
```

`λ1..λ4` to be tuned; recommend starting `λ1=1.0, λ2=1.0, λ3=0.5, λ4=0.1` and treating `λ3, λ4` as ablation switches (§10), with an explicit loss-weighting sensitivity sweep added to the experiment plan (§10 item 8; this was flagged as missing in Critique #1 item 8c).

## 6a. Parameter count estimate (arithmetic shown, revised)

| Component | Arithmetic | Params |
|---|---|---|
| DFT beamspace front end | fixed transform, no weights | 0 |
| IABC-v2 context projection | in=4, out=8: 4·8+8 | 40 |
| IABC-v2 shared per-element MLP (weights shared across all 32 elements) | in=10 (own 2 + context 8), hidden=16, out=2: (10·16+16)+(16·2+2) | 210 |
| **IABC-v2 subtotal** | 40 + 210 | **250** |
| Trunk stem conv | 3×3, Cin=2 (real/imag), Cout=32: 3·3·2·32 + 32 | 608 |
| Trunk: 10 residual blocks, C=32 | per block = 2×(3·3·32·32) convs + 2×BN(2·32 trainable): 2·(9·1024)+128 = 18,560; ×10 blocks | 185,600 |
| SE channel-attention, r=8, C=32→4 | per block: (32·4+4)+(4·32+32)=292; ×10 blocks | 2,920 |
| Path-token 1×1 conv, 32→32 | 32·32+32 | 1,056 |
| Coarse heatmap head, 1×1 conv 32→1 | 32·1+1 | 33 |
| AoA crop-refine head (7×7×32 crop → 1×1 conv 32→16 → FC 16·49→1) | conv: 32·16+16=528; FC: 784·1+1=785 | 1,313 |
| AoD crop-refine head (identical shape) | same | 1,313 |
| SE→SNR diagnostic head (tiny FC, 10→1 pooled over blocks) | 10+1 | 11 |
| **Total (estimate)** | 250+608+185,600+2,920+1,056+33+1,313+1,313+11 | **≈ 193,104 ≈ 193K** |

This is **≈41.1%** of the teacher's 469,393 params (revised down from the original's ≈42%/200K estimate — the IABC-v2 redesign is both a correctness fix and a modest additional param saving, since 250 params replaces the original's 4,192-param monolithic MLP). It remains below the magnitude-pruned student (314,513) and FNO (334,321), and is now ≈6.4× GridlessUnfold's 30,201 params (193,104/30,201≈6.39) — this comparison is now stated with the same caveat applied to §6b/§7 (array size, L-range, and task framing differ across all these baselines; none of these four numbers should be read as a like-for-like accuracy/param trade-off without matching those conditions — Critique #1 item 3 correctly noted this caveat was present for the FLOP table but missing here, and it is added here for consistency). No accuracy trade-off argument for being 6.4× larger than GridlessUnfold is offered yet; this is explicitly listed as an open comparison to be resolved empirically, not asserted in this document's favor (Critique #2 item 4). Channel width (C=32), block count (10), and crop size (7×7) remain design choices, not measured — this is explicitly an ESTIMATE, and the true count depends on final hyperparameter choices, which should be locked before any comparison is quoted.

**Supporting arithmetic for the reframed parameter-efficiency claim (§1, §12 item 2):** a counterfactual raw-antenna dense/flatten front end that flattens `Y` (16×16 complex → 512 real values) and projects it with a single fully-connected layer to the same 16×16×2 representation IABC-v2 produces would cost `512·512+512 = 262,656` parameters — **more than the entire rest of this model combined** (193,104 − 250 = 192,854 for the trunk+heads). This is the concrete number behind the claim that the classical-front-end choice is parameter-efficient; it is not a claim about FLOPs (see §6b). This counterfactual is proposed as an actual ablation, not just an arithmetic aside — see §10 item 5.

## 6b. FLOP estimate (arithmetic shown, forward pass, single sample, revised)

Using Agent-R8's stated convention: Conv2D FLOPs = `2 × k_h × k_w × C_in × C_out × H_out × W_out`. Spatial size held at 16×16 throughout (P=Q=Nr=Nt=16 per the project's generator, no down/upsampling in this shallow trunk). **The IABC-MLP row's arithmetic error flagged by both reviewers is fixed below** (§12 item 1): the original document's formula evaluated to 8,192 but the table printed 16,768 (a ~2.05× discrepancy, apparently a copy/scale error); this is corrected, and the IABC-v2 redesign additionally changes the module's actual cost.

| Component | Arithmetic | FLOPs |
|---|---|---|
| DFT front end (via 2D-FFT, not naive DFT) | O(Nr·Nt·log₂(Nr·Nt)) ≈ 256·8, ×4 for complex arithmetic | ≈ 8,192 |
| IABC-v2 context projection | 2×(4·8) | 64 |
| IABC-v2 shared per-element MLP (×32 elements, shared weights, applied once per element) | per element: 2×(10·16)+2×(16·2) = 320+64 = 384; ×32 elements | 12,288 |
| **IABC-v2 subtotal** | 64 + 12,288 | **12,352** |
| Trunk stem conv | 2·9·2·32·16·16 | 294,912 |
| Trunk: 10 blocks × 2 convs each, Cin=Cout=32 | per conv: 2·9·32·32·16·16 = 4,718,592; ×2 convs ×10 blocks | 94,371,840 |
| SE blocks (global pool + 2 tiny FC ×10) | negligible, ≈ 292×2×10 | ≈ 5,840 |
| Path-token 1×1 conv | 2·1·1·32·32·16·16 | 524,288 |
| Coarse heatmap head 1×1 conv | 2·1·1·32·1·16·16 | 16,384 |
| Crop-refine heads (fixed top-K=10, 7×7 crop, ×2 heads) | conv: 2·1·1·32·16·7·7=50,176; FC: 2·784·1=1,568; ×2 heads; **now over a fixed K=10 batched gather, not a variable-L loop** — see §3c/§4 | ≈ 103,488 |
| **Total (estimate)** | | **≈ 95,337,296 ≈ 95.3M FLOPs ≈ 47.67M MACs** |

The corrected IABC arithmetic changes the total by roughly 4,000–16,000 FLOPs out of 95.3M — immaterial to the bottom line, exactly as both reviewers noted, but the error itself is fixed here rather than left standing, because it directly falsified the document's "arithmetic shown" claim.

**The trunk dominates (>99%), confirmed and now stated as the headline fact, not something the motivation section talks around (§1, §12 item 2).** The front end (DFT + IABC-v2) costs ≈20,544 FLOPs total, ≈0.022% of the 95.3M budget. Swapping it for the counterfactual raw-dense front end (§6a) would cost 524,288 FLOPs — 503,744 more — which is itself only ≈0.53% of the trunk's budget. **This confirms, not merely concedes, Critique #1 item 2 and Critique #2 item 3 in full**: front-end choice has no material effect on total compute; the trunk's width/depth is the compute cost driver and is an independent design choice. The efficiency argument this design now makes rests on §6a's parameter arithmetic, not on this table. Direct FLOP comparison against Naoumi's 4.321 MMACs remains **NOT COMPARABLE** (different task framing, bistatic MLP vs. this project's spatial-conv joint task, per the literature synthesis's own comparability table). No Tier-1 paper reports a concrete FLOP figure for the Lloria architecture family at all, so this would be a first for that family (closes Gap 1), not a beat-the-literature claim.

## 7. Expected advantages and weaknesses

**Advantages:**
1. The front end (DFT + IABC-v2) is near-zero in FLOPs and adds only ≈250 params, avoiding the ≈262,656 params a comparably-capable dense raw-antenna projection would cost (§6a) — a demonstrated, arithmetic-grounded parameter saving. This is now the specific claim made; the earlier general "efficiency win" claim is withdrawn (§1, §12 item 2).
2. **Conditionally** targets two named, unaddressed literature gaps — Meneses-Albalá's own admitted gain-error gap and phase-error robustness beyond every paper's tested ceiling (5°) — conditional on gain-error injection code being built first (§3a, §9 Failure Mode 2, §10 item 1). This is stated as a capability the design enables, not a result already demonstrated.
3. The joint 2D heatmap is retained, so this does not inherit A5/A9's risk of breaking the "genuinely joint" property.
4. Stays compatible with the existing `evaluate_on_bank`/blob-detector pipeline; the fixed top-K=10 refinement gather (§3c/§4) is specified precisely enough to be latency-measurable in a well-defined way, unlike a full DETR-style redesign (A7) or the original draft's unspecified variable-L crop loop.
5. IABC-v2's per-element shared-weight design (§3a) directly addresses the identifiability bottleneck both reviewers flagged in the original monolithic MLP, rather than leaving it as an acknowledged-but-unfixed risk.

**Weaknesses:**
1. IABC-v2's generalization is still bounded by the synthetic paired-consistency loss's training distribution — if trained only on the Tier-1 grid (phase ≤5°, no gain-error data exists in the codebase today), it will not extrapolate to the Tier-3 OOD grid (10–15° phase, 0.5–4dB gain) without the Phase-0 gain-error injection code being built first (confirmed absent by code audit — new physics code, not new config).
2. It still relies on a discrete coarse P×Q=16×16 grid for initial peak localization, so it may partially inherit the diagnosed Jacobian-singularity/pixel-quantization Pd ceiling (Gap 5) unless the crop-refinement head's continuous regression genuinely escapes grid discretization — unconfirmed, and now specifically testable end-to-end since the refinement path has a fixed, well-defined shape (§3c).
3. Sequential coarse→correct→refine pipeline has three points where error can compound (front-end miscorrection → wrong coarse peak → refinement head has no valid anchor).
4. Only ≈0.13% of the parameter budget (250/193,104) is the IABC-v2 module itself; the trunk remains a generic 10-block ResNet unrelated to impairment-adaptivity. Any accuracy gain over the zero-param DFT-SIC baseline cannot be attributed to "impairment-adaptivity" specifically versus "adding a large generic learned trunk" without the capacity-matched control now added to the ablation plan (§10 item 4; Critique #1 item 6, Critique #2 item 6). This confound is **not resolved by this document** — it is resolved only by running that ablation.
5. This design is ≈6.4× larger than GridlessUnfold (30,201 params) with no accuracy trade-off argument yet offered (Critique #2 item 4) — an open comparison, not a settled advantage.

## 8. Concrete failure modes

1. **Correction overshoot under untrained severity.** At phase error 15° (never seen in training, which only spans 0–5° per the Tier-1 grid), IABC-v2's learned correction — extrapolating outside its training manifold — could systematically *over*-correct, actively moving the coarse peak into the wrong DFT beam bin entirely, producing a worse result than the uncorrected fixed front end (A8's exact flagged risk, now inherited by the correction module rather than avoided).
2. **Gain-error blind spot.** Because gain-error training data does not exist in the project pipeline today, if this architecture is trained only on the currently-available phase-error grid, `λ3`'s consistency loss never sees gain perturbations at all — IABC-v2 could learn to ignore the gain channel of its output entirely (a degenerate solution), silently failing the exact Meneses-Albalá gap it was designed to close, unless new gain-error generator code is built and used in training before any claim is made. **This is a Phase-0 blocking dependency, not a caveat** (§10 item 1).
3. **Closely-spaced-source collapse.** At Δθ<5° (below the array's Rayleigh limit, `[GENUINE GAP]` per Agent-R3 §3b), two true peaks can fall inside the same 16×16 coarse bin; the crop-refinement head only sees one coarse anchor and cannot recover two angles from it — same failure category as A6, and the crop-based local design does not fix this (only global re-gridding or a set-prediction head would).
4. **Coupling leakage at high L.** The path-token bottleneck is designed to preserve AoA–AoD pairing, but this is untested above L=6 (the current training distribution's ceiling); at L∈{7,8,10} (Tier-3 OOD, per Agent-R3 §3a) it is unconfirmed whether the shared token still keeps the correct angle pairs together or starts swapping AoA/AoD across paths. The fixed top-K=10 slot design (§3c) accommodates L up to 10 structurally, but pairing correctness at that range is still unverified.
5. **Consistency-loss shortcut / train-test mismatch.** If `L_pair_consist` is implemented carelessly (e.g. the clean-reference pairing leaks information not derivable from the impaired sample alone at inference), the module could learn a shortcut that only works because a paired clean sample existed during training — this must be checked by confirming inference-time inputs are strictly `Y` alone, no clean reference.
6. **False-positive coarse peaks / hallucinated refinement.** *(New — Critique #1 item 8a.)* When the coarse heatmap head produces a spurious peak with no real path underneath it, the crop-refinement head has no mechanism to detect this and may output a confident refined angle for a non-existent path. Mitigation must be tested: does the coarse heatmap's own confidence score suppress low-confidence slots before they reach refinement, and is this sufficient?
7. **Low-SNR-specific IABC-v2 misbehavior.** *(New — Critique #1 item 8b.)* IABC-v2's own inputs (per-element magnitude/phase statistics) become unreliable at low SNR, compounding with the existing SNR-extrapolation gap (Tier-3 tails, −25 to −20dB and 30–35dB). This interaction is not yet characterized and should be tested as its own condition, not folded into the general SNR sweep.
8. **Loss-weighting sensitivity.** *(New — Critique #1 item 8c.)* `λ1–λ4` are stated as "to be tuned" with no sensitivity analysis. If the multi-task balance is wrong (e.g., `λ3` too high relative to `λ1/λ2`), IABC-v2 could over-fit to consistency matching at the expense of final angle accuracy, or vice versa. A sensitivity sweep is now added to §10 item 8 rather than left unaddressed.

## 9. Experiments needed to validate

0. **Phase-0 prerequisite (blocking, elevated from an original caveat to an explicit gate — Critique #2 item 2):** build gain/phase calibration-error injection code in the data generator (confirmed absent today). No claim about closing the Meneses-Albalá gain-error gap may be made before this exists and is used in training.
1. **Ablation battery** (on identical `frozen_banks/eval_bank.npz`, reusing `evaluate_on_bank`):
   a. − IABC-v2 (revert to fixed, uncorrected DFT front end) evaluated across the Tier-1 phase/gain grid, to isolate IABC-v2's actual contribution.
   b. − crop-refinement heads (coarse heatmap + blob detector only) vs. full model, with P95 tail-error reported specifically near end-fire angles (Gap 5 test).
   c. − SE/`L_SE_snr` auxiliary loss, to test whether A3's "attention correlates with SNR" hypothesis actually holds for this task or is a false lead.
   d. **Capacity-matched control (new — Critique #1 item 6, Critique #2 item 6):** replace IABC-v2 with a same-parameter-count module of identical shape (per-element shared MLP, 2→16→2, ≈250 params) trained with *only* the task loss (`λ3=0`, no clean-reference pairing at all). If this control performs comparably to the full IABC-v2, the gain is attributable to added capacity, not impairment-adaptivity, and the design's central claim must be revised.
   e. **Front-end swap ablation (new — Critique #1 item 2, supports §6a/§1):** replace the DFT+IABC-v2 front end with the counterfactual raw-dense projection (512→512 FC, 262,656 params) feeding the same trunk, holding the trunk fixed. Measure resulting total params, FLOPs, and accuracy, to empirically confirm or refute the reframed parameter-efficiency claim rather than leaving it as arithmetic alone.
2. **Full Tier-1 robustness battery** per Agent-R3 (SNR sweep in-distribution core, phase δ/γ∈{0,1,2,5}°, L∈{1..6}) as the baseline registry comparison point.
3. **Genuine-gap extrapolation tests** (depends on item 0): phase 10–15°, gain 0–4dB, L∈{7,8,10}, Δθ down to 1°, and the SNR extrapolation tails (−25,−20,30,35dB) — all flagged `[GENUINE GAP]`; this design's core claim rests on results here, not on in-distribution numbers.
4. **End-fire density probe** (Agent-R3 §4) to directly test whether the crop-refinement mechanism actually escapes the diagnosed pixel-quantization ceiling, run alongside a "refinement-heads-removed" variant for direct comparison.
5. **Two-number latency instrumentation (revised — Critique #1 item 7, Critique #2 item 3):** measure and report separately, never merged: (i) **NN-only forward latency**, using the fixed top-K=10 vectorized gather (§3c/§4) so the measurement has no CPU-bound dynamic-shape component; (ii) **full end-to-end pipeline latency**, including the CPU-side blob-search peak-detection step, explicitly labeled as such. Use `ptflops`/`thop` for FLOPs (none exist in-repo today) and generalize `benchmark_latency()` (already present at `notebooks/proposed pin architecture.ipynb` line 664, confirmed by Critique #2 but not previously used) for both latency numbers. Also report peak memory.
6. **Multi-seed variance run** (Gap 10) given the correction/attention modules introduce additional stochastic training dynamics beyond the teacher's.
7. **Notebook-4-style nuisance-path stress test**, extended to this architecture, since it is the project's only existing evidence that a smaller model can *beat* the teacher under stress — worth checking whether IABR-Net's correction mechanism reproduces or exceeds that effect.
8. **Loss-weighting sensitivity sweep (new — Critique #1 item 8c).** Grid or coarse random search over `λ1–λ4` (at minimum: `λ3∈{0,0.1,0.5,1.0}`, `λ4∈{0,0.1,0.3}`), reporting final angle-accuracy sensitivity, to characterize rather than assert the "to be tuned" claim.

## 10. Novelty argument

**Self-classification: PARTIALLY NEW, with the specific-application claim UNCERTAIN pending a dedicated search — unchanged from the original assessment in substance, but the reasoning is now consistent with the toned-down §3b/§4 language (Critique #2 item 5).**

Justification, following Agent-R7's framework directly:

- At the **pattern level**, "a small learned network corrects/regularizes inputs to a classical or near-classical processing stage" is an established paradigm — SubspaceNet is a verified instance of it (learned covariance surrogate → classical resolver) [EVIDENCE: PAPER], and Agent-R7 independently found other instances (deep-unrolling+GNN DOA hybrids) outside the reviewed corpus. IABR-Net's IABC-v2 module is structurally this same family (now additionally a Deep-Sets/PointNet-style shared-weight instance of it, §3a), just placed *before* a classical transform rather than *after* one, and targeting *hardware impairment* rather than *covariance denoising*. By Agent-R7's own test 3 (§1), a block-for-block application of an established pattern to a new task variant is a **domain-transfer contribution, not a conceptual one** — so at this level this remains **KNOWN COMBINATION at the pattern level**. The path-token coupling (§3b) is likewise now explicitly described as a standard shared-bottleneck pattern, not a novel fusion mechanism, removing the inconsistency Critique #2 item 5 identified between the confident §3b/§4 language and this section's own concession.
- At the **specific-application level**, the combination of (a) a fixed DFT beamspace reduction, (b) a synthetic-paired-consistency-trained, per-element identifiability-aware correction module specifically targeting Meneses-Albalá's own named gain-error gap, and (c) a path-token-coupled coarse+local-refine dual head that preserves joint AoA/AoD pairing over a fixed-size vectorized gather, applied to the *joint 2D* task, is not present in the 14-paper corpus. Agent-R4's Candidate 1 remains the closest literature-corpus anchor and is explicitly a *fixed*, non-adaptive front end (A8's own flagged weakness). This matches Agent-R7's angle-J verdict (**UNCERTAIN**): within the corpus this direction is untested for the joint 2D task, but a dedicated search specifically for "classical front-end, learned impairment correction, joint AoA/AoD" is still required before this is treated as settled — that requirement is unchanged by this revision.
- Per Agent-R7's explicit methodological requirement (§1, item 1: tie the architectural choice to a specific diagnosed failure, not a generic swap), this design does clear that bar for the correction module — but the capacity-matched control (§10 item 1d) is now explicitly required before the "impairment-adaptivity, not just added capacity" version of that claim can be made (Critique #1 item 6, Critique #2 item 6). Until that ablation runs, the novelty argument rests on architectural motivation and identifiability design, not yet on demonstrated attribution.

**Verdict: PARTIALLY NEW** (domain-transfer of an established model-based-DL pattern to a genuinely under-tested robustness axis for the joint-2D task), **with the "is this exact combination unpublished" claim held at UNCERTAIN** until Agent-R7's recommended dedicated searches are actually run, **and with the "impairment-adaptivity specifically, not just added capacity" claim held as unconfirmed until the capacity-matched control ablation (§10 item 1d) is run.** This specification does not claim NEW.

---

## Changes made in response to critique

1. **Fixed the IABC FLOP arithmetic error** (both reviewers, Critique #1 item 1 / Critique #2 item 1): the original §7 row evaluated its own stated formula to 8,192 but printed 16,768. The IABC-v2 redesign additionally changes this module's true cost to 12,352 FLOPs / 250 params; the full FLOP and parameter tables (§6a, §6b) are recomputed and now reproduce exactly from their own shown formulas.

2. **Withdrew the unsupported "compute efficiency win" framing and replaced it with an arithmetic-grounded parameter-efficiency claim** (Critique #1 item 2): §1 and §7 (Advantages) no longer claim the classical front end saves compute — the front end is confirmed to be ≈0.022% of total FLOPs and swapping it for a raw-dense alternative saves only ≈0.53% of the trunk's budget (§6b). Instead, §6a now shows that a counterfactual dense front end would cost 262,656 parameters — more than the rest of the model — and this is the number the "lightweight front end" claim now rests on. A front-end-swap ablation (§10 item 1e) is added to test this empirically rather than leave it as arithmetic alone.

3. **Made the gain-error-robustness claim's dependency on unbuilt code explicit and consistent everywhere it appears** (Critique #2 item 2): §1, §3a, §7 (Advantages), §8 (Failure Mode 2), and §10 now all state that this capability is conditional on Phase-0 gain-error injection code being built, elevated from a caveat buried in §9/§10 to an explicit blocking prerequisite (§10 item 0), removing the confidence/hedge mismatch the critique identified.

4. **Redesigned IABC to fix the identifiability bottleneck rather than defend it** (Critique #1 item 4, Critique #2 item 7): the monolithic 64→32→64 MLP is replaced with IABC-v2, a shared-weight per-element MLP (Deep-Sets/PointNet-style) that maps each element's own 2 inputs plus an 8-dim shared context to its own 2 outputs, removing the 64-into-32-bottleneck-back-to-64 collapse risk. This is an architectural change, not an added paragraph, and it also reduces the module's parameter count from 4,192 to 250.

5. **Corrected the "self-supervised" mislabeling** (Critique #1 item 5): `L_pair_consist` (renamed from `L_impair_consist`) is now consistently described as synthetic paired supervision / ordinary regression against a simulator-generated clean target, in §3a, §5, and §10, and the novelty argument (§10 of this revision) no longer leans on "self-supervised" as a distinguishing property.

6. **Added a capacity-matched control to the ablation plan** (Critique #1 item 6, Critique #2 item 6): §10 item 1d trains a same-parameter-count generic module with no impairment-specific loss, to isolate whether accuracy gains come from impairment-adaptivity specifically or from added capacity alone — the exact confound both critiques identified as uncontrolled.

7. **Specified the latency measurement protocol precisely, and re-engineered the refinement path to make it measurable** (Critique #1 item 7, Critique #2 item 3): crop-refinement now always operates over a fixed-size top-K=10 vectorized batched gather (§3c, §4) rather than an unspecified variable-L loop, and §10 item 5 requires reporting NN-only forward latency and full end-to-end pipeline latency as two separate numbers, never merged — directly addressing the "confusing end-to-end system latency with NN inference time" risk both reviews flagged, using the already-existing but previously unused `benchmark_latency()` tool.

8. **Added the missing param-comparison caveat and the GridlessUnfold trade-off note** (Critique #1 item 3, Critique #2 item 4): §6a now carries the same comparability caveat (array size, L-range, task framing) that §6b already had for the Naoumi comparison, and explicitly states the ≈6.4× size difference vs. GridlessUnfold with no accuracy trade-off yet argued, rather than presenting the parameter table without qualification.

9. **Downgraded "novel fusion mechanism" language to match §11's own concession** (Critique #2 item 5): §3b and §4 now describe the path-token bottleneck as a standard shared-bottleneck, index-paired pattern, consistent with the novelty section's "KNOWN COMBINATION at the pattern level" verdict, removing the confident/hedged language mismatch.

10. **Completed the failure-mode list** (Critique #1 item 8): added false-positive coarse peak hallucination (§9.6), low-SNR-specific IABC-v2 misbehavior (§9.7), and loss-weighting sensitivity (§9.8), plus a corresponding sensitivity-sweep experiment (§10 item 8).

**Not changed / rebutted rather than revised:** The trunk's dominance of FLOPs and params (>99% / ≈97.6%) is confirmed as correct by recomputation, not disputed — this revision reframes the efficiency claim around parameters (where the arithmetic supports it) rather than disputing the FLOP finding (where it does not). The §6a param-count arithmetic for the unchanged components (stem, trunk, SE, path-token, heads) was independently verified correct in both critiques and is carried forward unmodified.

## Adversarial review

### Critic 1 (fatal_flaw_found=True)

ADVERSARIAL REVIEW — Candidate B (IABR-Net), Reviewer #1

VERDICT: Not fatal to the underlying idea, but there ARE two substantiation-level flaws serious enough that the candidate should not proceed to comparison/selection unmodified — its central "lightweight, efficiency-win" thesis is not actually supported by its own numbers, and one of its "arithmetic shown" figures is internally inconsistent. Below is what survives scrutiny and what does not.

1) ARITHMETIC ERROR IN THE FLOP TABLE (confirmed by recomputation).
I recomputed every row of §7. All rows check out exactly EXCEPT the IABC MLP row. The formula given, "2×(64·32) + 2×(32·64)", literally evaluates to 2×2048 + 2×2048 = 8,192 — not the "≈16,768" printed in the table (off by ~2.05×). Every other row (DFT front end 8,192; stem 294,912; trunk 4,718,592/conv ×20 = 94,371,840; path-token 524,288; coarse head 16,384; crop-refine heads 103,488) reproduces exactly from its own shown formula. The mis-stated IABC figure is carried into the ≈95.3M total, so the total itself is off by ~8,500 FLOPs (~0.009% of the total) — immaterial to the bottom-line "lightweight" number, but it directly falsifies the document's own claim that "arithmetic is shown" and invites doubt about whether the rest of the estimate was actually re-derived or just asserted.

2) THE "LIGHTWEIGHT / EFFICIENCY-WIN" THESIS IS NOT SUPPORTED BY THE CANDIDATE'S OWN TABLE.
The motivation section's entire pitch is "take the efficiency win of a classical front end." But by the document's own FLOP table, the DFT front end + IABC module together cost ≈12,384–16,768 FLOPs — about 0.01–0.02% of the 95.3M total. The 10-block ResNet trunk (94.37M FLOPs) is 99.0% of the entire budget and runs at the SAME full 16×16 spatial resolution regardless of what feeds it. No ablation or baseline is given showing what the trunk would have cost with a raw-antenna-domain or dense/flatten front end instead — so the claimed "efficiency win" of the classical front end is asserted, not demonstrated. As currently written, swapping the front end for a fixed DFT has essentially no effect on total compute; the real cost driver (trunk width/depth) is an independent design choice unrelated to the paper's headline efficiency argument. This is a genuine "is it really lightweight — check the arithmetic" failure: the arithmetic shows the opposite of what the motivation implies.

3) INCONSISTENT RIGOR BETWEEN THE TWO COMPARISON TABLES.
The FLOP comparison against Naoumi (§7) is explicitly and correctly flagged "NOT COMPARABLE — different task framing." The parameter comparison against the teacher/pruned-student/FNO/GridlessUnfold (§6) receives no equivalent caveat about matching array size, L-range, or task framing, even though the same comparability concerns apply. Selectively caveating one comparison and not the structurally identical other is a fairness/rigor inconsistency a reviewer should catch.

4) UNEXAMINED ARCHITECTURAL BOTTLENECK IN THE CORRECTION MODULE.
The IABC MLP is 64→32→64: it must recover 64 independent per-element correction values (32 elements × {phase, gain}) after squeezing through only 32 hidden units. No justification is given for why this bottleneck is sufficient to disentangle per-element corrections rather than just producing a smoothed/averaged correction — this is exactly the kind of failure mode (a Jacobian/identifiability squeeze) the document is careful to flag elsewhere (Gap 5, coarse-grid quantization) but does not apply to its own new module.

5) "SELF-SUPERVISED" IS MISLABELED. The clean/impaired consistency loss (λ3) uses paired (clean, impaired) simulator outputs of the *same* channel realization with an MSE target — this is ordinary supervised regression with synthetically generated pairs, not self-supervised learning in the standard sense (no external signal is derived from the impaired sample alone). This isn't fatal but is a terminology overclaim that should be corrected before the novelty argument in §11 leans on it.

6) IMPROVEMENT-ATTRIBUTION CONFOUND NOT CONTROLLED. Going from the 0-parameter classical baseline (Pd 0.884) to a 197K-parameter network is a huge capacity jump; the ablation plan (§10a) isolates IABC's contribution relative to the fixed front end, but nothing in the experiment plan controls for "any comparable-capacity add-on, not specifically impairment-adaptive, would do about as well." So the eventual claim "our impairment-adaptive design helps" risks actually being "adding 197K learned parameters to a 0-parameter baseline helps," which is a much weaker and less interesting claim than advertised.

7) LATENCY INSTRUMENTATION AMBIGUITY (flagged honestly, but incompletely specified). §10-item-5 proposes generalizing `benchmark_latency()`, but the pipeline (DFT → IABC → trunk → blob search on the coarse heatmap → per-peak cropping → refinement heads) includes a classical peak-search step that is potentially CPU-bound and non-vectorized. The plan does not specify whether the eventual latency number will isolate pure NN forward-pass time or measure the full pipeline including peak search — exactly the "confusing end-to-end system latency with NN inference time" risk the review brief asked me to check for. This is currently unresolved, not wrong, but should be nailed down before any latency claim is made.

8) FAILURE-MODE LIST IS NOT COMPLETE. §9's five failure modes are good and honestly adversarial against the design itself (overshoot, gain blind spot, closely-spaced collapse, coupling leakage, consistency shortcut). Missing: (a) false-positive coarse peaks — what does the crop-refinement head do when it's handed a spurious peak with no real path underneath it (does it hallucinate a confident refined angle)? (b) low-SNR-specific IABC misbehavior — the correction module's magnitude/phase input statistics become unreliable at low SNR, and this interaction with the existing SNR-extrapolation gap is not called out as its own risk; (c) loss-weighting sensitivity (λ1–λ4) is called "to be tuned" with no sensitivity analysis or failure discussion if the multi-task balance is wrong.

9) WHAT DOES HOLD UP: the joint-pairing design (shared coarse-peak index as pairing key, retained 2D heatmap rather than split 1D heads) is a coherent and reasonably well-justified answer to Gap 9/A5/A9's risk, and is the strongest part of the submission. The param-count arithmetic in §6 is fully correct on recomputation. The novelty self-assessment (§11) is unusually honest and already downgrades itself to "partially new / uncertain," which is appropriate given SubspaceNet is a clear structural precedent — I'd push the self-assessment even further given point (5) above, but it does not overclaim novelty.

NET: The design is not incoherent and most of its stated risks are self-acknowledged with a real validation plan — but its own headline claim ("lightweight, efficiency-win front end") is undermined by its own FLOP table (front end is <0.02% of total compute, trunk dominates, no comparative ablation proves the front end saved anything), and one of its "arithmetic shown" figures does not reproduce from its own formula. Both are things the report explicitly asked me to check, and both fail as currently written. Recommend: fix the IABC FLOP arithmetic, add a same-trunk-different-front-end FLOP ablation to actually support (or drop) the efficiency-win framing, add the missing param-comparison caveat, and add a capacity-matched (non-impairment-adaptive) control to the ablation battery before this candidate is scored against the others.

### Critic 2 (fatal_flaw_found=True)

Verified against the repo: DL_DOA/src/tvt_data_generation_v3.py confirms phase-error injection (error_deg) exists but no gain/amplitude-mismatch injection exists anywhere (only a fixed amps_type='ones' and per-path signal "gains", not hardware calibration error) — corroborating the candidate's own admission. benchmark_latency() already exists in notebooks/proposed pin architecture.ipynb (line 664) but was not used.

Key flaws found:
1. FLOP arithmetic error: §7's IABC-MLP line shows the formula "2×(64·32)+2×(32·64)" which evaluates to 8,192, but the document reports 16,768 (= 4× the 4,192 param count, an apparent copy/scale error). Immaterial to the 95.3M total (trunk is >99%) but undercuts the "arithmetic verified" framing of the whole section.
2. The central motivating claim (§1, §8-Adv.2: "directly and mechanically" closes the Meneses-Albalá gain-error gap) currently has zero evidentiary basis — gain-error injection code does not exist in the repo (confirmed by grep), so the self-supervised consistency loss can never have seen a gain perturbation, and per the document's own Failure Mode 2, the module could trivially learn to ignore the gain channel entirely. Confident motivating language in §1/§8 contradicts the hedged admission in §9/§10 that this capability is unbuilt.
3. The "lightweight" claim (§6-§8) is asserted from static param/FLOP counts only; no actual latency was measured despite the tooling (benchmark_latency()) already sitting in the repo. This matters because the crop-refinement heads run over a variable, data-dependent number of ROIs (L=1–6, up to 10 OOD) — a dynamic-shape/gather pattern whose real GPU latency is not captured by a ~103K-FLOP (0.1% of budget) static estimate. This exact FLOPs-vs-latency conflation, which the review brief explicitly asked about, is absent from the document's own advantages/failure-mode lists.
4. The comparison is selectively framed: IABR-Net is favorably compared to the teacher/pruned/FNO baselines but is also ~6.5× larger than GridlessUnfold (30,201 params) with no accuracy trade-off argument offered.
5. The "novel" path-token coupling mechanism is a standard shared-1x1-conv multi-task trunk read by three parallel heads at the same spatial index, with pairing enforced only by index-matching, not a learned interaction — the confident "novel fusion mechanism" language in §3b/§4 sits in tension with §11's own concession that this is "KNOWN COMBINATION at the pattern level."
6. Parameter/attribution confound: only ~2% of the budget (4.2K/197K) is the purportedly headline IABC module; 94% is a generic ResNet trunk. The ablation plan tests IABC-alone and crop-refine-alone but never a full factorial, so any accuracy gain over the zero-param DFT-SIC baseline cannot yet be attributed to "impairment-adaptivity" versus simply adding a large generic learned trunk — the exact critique the design claims to avoid in §1.
7. Identifiability risk unaddressed: the IABC module must recover 64 real impairment parameters (32 phase + 32 gain, TX+RX) from a 64-dim pooled statistic of the same signal, a tightly-determined inverse problem where pooling may destroy the information needed to disambiguate impairment from genuine signal content at low SNR.

These are not fatal to the underlying idea, but are fatal to approving the specification unmodified: the headline gain-error-robustness claim needs real training data/code before being asserted, the lightweight claim needs an actual latency benchmark (especially under variable-L crop refinement) before being used as a selling point, and the FLOP arithmetic and novelty framing need correction.

## Original (pre-refinement) specification

# Candidate B — IABR-Net: Impairment-Adaptive Beamspace Residual-Correction Network with Coupled Dual-Head Refinement

## 1. Name and motivation

**IABR-Net** (Impairment-Adaptive Beamspace Residual-correction Network). Every Tier-1 paper in the matrix estimates angles from the raw antenna-domain tensor using a learned model end-to-end (dense/flatten or conv bottleneck → heatmap → blob search). This project's own DFT-SIC classical baseline already reproduces 95.6% of the 469K-parameter teacher's accuracy (Pd 0.884 vs 0.925 @20dB) with **zero learned parameters** [EVIDENCE: PROJECT], which means the current network spends most of its 469K parameters re-deriving a mapping a fixed transform already gets most of the way to. At the same time, the literature's one paper that actually names this system model's calibration weaknesses (Meneses-Albalá 2026) admits it never tested amplitude/gain-mismatch robustness, and no Tier-1 paper tests phase error beyond 5° [EVIDENCE: PAPER, `[GENUINE GAP]` per Agent-R3 §2a/2b]. A purely fixed classical front end (Agent-R4 Candidate 1/A8) inherits exactly this weakness: it has no way to adapt when the true steering vectors deviate from the assumed model. IABR-Net's motivation is to take the efficiency win of a classical front end while removing its single biggest liability — by inserting a small, learned, *impairment-adaptive* correction stage between the fixed transform and a shallow residual backbone, so the backbone only ever has to sharpen an already-approximately-correct beamspace map rather than learn angle estimation from raw antenna data. This is a robustness-first, signal-processing-first design, not a backbone-family swap.

## 2. Borrowed components (with source)

| Component | Source | Role here |
|---|---|---|
| 2D-DFT / beamspace codebook projection (fixed, zero-param) | This project's own DFT-SIC classical baseline, reproducing Gupta TCOMM2025's steering-vector model [EVIDENCE: PROJECT, PAPER] | Front-end feature reduction, replaces several early conv layers |
| ResNet residual-block shell (conv stem, skip connections) | Lloria TVT2026 / PIMRC2024 [EVIDENCE: PAPER] | Shallow (10-block) residual-correction trunk |
| Joint single 2D heatmap output read by one peak search (not split 1D outputs) | Lloria TVT2026 Eq. 9–11 [EVIDENCE: PAPER] | Preserves the "genuinely joint" property flagged as at-risk by Gap 9/Contradiction 12 |
| Squeeze-excite-style channel attention between blocks | Generic efficient-CNN pattern, no direct paper precedent in the reviewed set (per A3) | Cheap capacity + SNR-correlated diagnostic signal |
| Phase-error injection convention (`error_deg` in TX/RX beamforming matrices) | Lloria TVT2026 code / `tvt_data_generation_v3.py` [EVIDENCE: CODE, PROJECT] | Training-time impairment augmentation grid |
| "Small learned network corrects a classical algorithm's inputs" pattern | SubspaceNet (Shmuel et al. 2025 TVT), 41,761-param covariance-surrogate network [EVIDENCE: PAPER] | Direct structural inspiration for the correction module (though SubspaceNet corrects *after* the classical step for 1D DOA; here it corrects *before*, for joint 2D) |
| Coarse-to-fine peak refinement (crop around a coarse peak, then regress) | Protocol §11-listed pattern (A6), scoped down to a local crop rather than a global re-grid | Second-stage precision without full DETR-style set prediction complexity |

## 3. Novel component(s), stated precisely

**(a) Impairment-Adaptive Beamspace Correction (IABC) module.** After the fixed DFT beamspace projection `Y_bs = F_codebook^H Y` (zero trainable parameters), a small MLP consumes per-antenna-element magnitude/phase summary statistics of the raw signal (not the beamspace map itself) and outputs a predicted per-element phase offset `δ̂` and gain offset `γ̂` for both TX and RX arrays. These are applied as a multiplicative complex correction to `Y_bs` *before* it enters the trunk. It is trained with a **self-supervised clean/impaired consistency loss**: for each training scene the data generator produces both an impaired realization (random `δ,γ` drawn from the Tier-1 grid) and, at zero extra data-generation cost, the same channel realization with `δ=γ=0`; the IABC module is trained so its corrected beamspace map matches the clean one. This is a training-time-only signal — no clean reference is available or needed at inference. This directly and precisely targets Meneses-Albalá's own named future-work gap (gain-error robustness, never tested by any Tier-1 paper) and extends phase-error testing past every paper's tested ceiling of 5°, by construction rather than by assumption.

**(b) Path-token coupling bottleneck.** Immediately after the trunk, a single shared 1×1-conv "path-token" layer produces one shared feature map that both the coarse joint-heatmap head *and* the two crop-refinement heads read from — the coarse heatmap is retained (not replaced by two independent 1D heads) specifically so the AoA–AoD pairing information the joint 2D representation currently encodes implicitly is not lost, addressing the risk A5/A9's "shared-encoder, split-heads" idea creates per Gap 9's own caution. This is the fusion mechanism (see §4).

**(c) Peak-conditioned local crop-refinement heads (not global re-grid).** Unlike A6 (which coarsens the *entire* grid, risking closely-spaced-source collapse), only a small window around each coarse-detected peak is cropped from the path-token map and fed to two lightweight per-angle regression heads that output a continuous sub-pixel offset. This differs from A6/A7 by leaving the coarse detection stage untouched (still the paper's blob-search-compatible heatmap) and only adding local continuous correction, which keeps it compatible with the existing `evaluate_on_bank` pipeline with a small extension rather than requiring a wholly new eval path.

## 4. Data flow

```
Input Y (raw complex antenna-domain signal, Nr x Nt)
   │
   ├─► [FIXED] 2D-DFT beamspace projection F_codebook^H · Y  →  Y_bs  (P×Q=16×16, complex, 0 params)
   │
   ├─► [LEARNED, small] IABC module: pooled per-element mag/phase stats of Y → MLP(64→32→64)
   │        → per-element (δ̂ phase, γ̂ gain) corrections → applied multiplicatively to Y_bs
   │        →  Y_bs_corrected (P×Q, 2-channel real/imag)
   │
   ├─► [SHARED TRUNK] 10-block shallow ResNet shell (C=32, 3x3 convs + SE channel-attention per block)
   │        operating on Y_bs_corrected, spatial size held at 16x16 throughout (no down/upsampling —
   │        classical front end already put the signal in angle-space, so no global-mapping burden)
   │        →  shared feature map F_shared (16×16×32)
   │
   ├─► [FUSION] Path-token 1x1 conv (32→32) on F_shared → F_token (16×16×32), read by all heads below
   │
   ├──────────────┬───────────────────────────────┬─────────────────────────────┐
   │              │                                │                             │
   ▼              ▼                                ▼                             ▼
[Coarse joint   [AoA crop-refine head]        [AoD crop-refine head]      [SE-attention→SNR
 2D heatmap      1x1 conv (32→16) on            1x1 conv (32→16) on         diagnostic head]
 head]           a 7x7 crop around each         a 7x7 crop around each      pooled SE gate
 1x1 conv        coarse peak (from F_token)     coarse peak (from F_token)  → 1 scalar, aux-only
 32→1, blob-     → FC → Δψ (continuous          → FC → Δφ (continuous
 detector-       sub-pixel offset)              sub-pixel offset)
 compatible
   │                    │                                │
   ▼                    ▼                                ▼
Coarse (ψ,φ) via   ψ_final = ψ_coarse + Δψ        φ_final = φ_coarse + Δφ
 blob search
   │                    │                                │
   └────────────────────┴──────────── AoA output ────────┴──── AoD output
                    (paired by shared coarse peak index — joint pairing preserved)
```

The **fusion mechanism** is the path-token bottleneck: both angle heads and the coarse heatmap head read the *same* shared feature map location for a given detected path, so AoA and AoD for path *i* are never computed from independent, unpaired representations — the coarse heatmap's peak index is the pairing key, exactly preserving Lloria's "one peak, both angles" joint property while still letting each angle get its own lightweight continuous correction.

## 5. Loss function

```
L = λ1 · L_heatmap        (BCE/MSE, joint 2D coarse heatmap vs Gaussian-blurred GT peak map — Lloria-style)
  + λ2 · L_offset          (Smooth-L1 on Δψ, Δφ at each matched coarse peak, nearest-GT assignment
                             valid since L≤6 well-separated sources — no Hungarian matcher needed)
  + λ3 · L_impair_consist  (MSE between IABC-corrected beamspace map and the clean (δ=γ=0) beamspace
                             map for the same channel realization — training-time only, self-supervised)
  + λ4 · L_SE_snr           (small-weight auxiliary regression: pooled SE-gate activation → true per-
                             sample SNR label; ablatable, tests A3's "attention correlates with SNR"
                             hypothesis directly; does not affect angle outputs if ablated)
```

`λ1..λ4` to be tuned; recommend starting `λ1=1.0, λ2=1.0, λ3=0.5, λ4=0.1` and treating `λ3, λ4` as ablation switches (§9).

## 6. Parameter count estimate (arithmetic shown)

| Component | Arithmetic | Params |
|---|---|---|
| DFT beamspace front end | fixed transform, no weights | 0 |
| IABC MLP | in=64 (per-element mag+phase pooled stats, Nr+Nt=32 elements × 2), hidden=32, out=64 (δ,γ per element): (64·32+32) + (32·64+64) | 4,192 |
| Trunk stem conv | 3×3, Cin=2 (real/imag), Cout=32: 3·3·2·32 + 32 | 608 |
| Trunk: 10 residual blocks, C=32 | per block = 2×(3·3·32·32) convs + 2×BN(2·32 trainable): 2·(9·1024)+128 = 18,560; ×10 blocks | 185,600 |
| SE channel-attention, r=8, C=32→4 | per block: (32·4+4)+(4·32+32)=292; ×10 blocks | 2,920 |
| Path-token 1×1 conv, 32→32 | 32·32+32 | 1,056 |
| Coarse heatmap head, 1×1 conv 32→1 | 32·1+1 | 33 |
| AoA crop-refine head (7×7×32 crop → 1×1 conv 32→16 → FC 16·49→1) | conv: 32·16+16=528; FC: 784·1+1=785 | 1,313 |
| AoD crop-refine head (identical shape) | same | 1,313 |
| SE→SNR diagnostic head (tiny FC, 10→1 pooled over blocks) | 10+1 | 11 |
| **Total (estimate)** | | **≈ 197,046 ≈ 200K** |

This is **~42% of the teacher's 469,393 params**, below both the magnitude-pruned student (314,513) and FNO (334,321), above GridlessUnfold (30,201). Channel width (C=32), block count (10), and crop size (7×7) are design choices, not measured — this is explicitly an ESTIMATE, and the true count depends on final hyperparameter choices, which should be locked before any comparison is quoted.

## 7. FLOP estimate (arithmetic shown, forward pass, single sample)

Using Agent-R8's stated convention: Conv2D FLOPs = `2 × k_h × k_w × C_in × C_out × H_out × W_out`. Spatial size held at 16×16 throughout (P=Q=Nr=Nt=16 per the project's generator, no down/upsampling in this shallow trunk).

| Component | Arithmetic | FLOPs |
|---|---|---|
| DFT front end (via 2D-FFT, not naive DFT) | O(Nr·Nt·log₂(Nr·Nt)) ≈ 256·8, ×4 for complex arithmetic | ≈ 8,192 |
| IABC MLP | 2×(64·32) + 2×(32·64) (MAC×2 for FLOPs) | ≈ 16,768 |
| Trunk stem conv | 2·9·2·32·16·16 | 294,912 |
| Trunk: 10 blocks × 2 convs each, Cin=Cout=32 | per conv: 2·9·32·32·16·16 = 4,718,592; ×2 convs ×10 blocks | 94,371,840 |
| SE blocks (global pool + 2 tiny FC ×10) | negligible, ≈ 292×2×10 | ≈ 5,840 |
| Path-token 1×1 conv | 2·1·1·32·32·16·16 | 524,288 |
| Coarse heatmap head 1×1 conv | 2·1·1·32·1·16·16 | 16,384 |
| Crop-refine heads (7×7 crop, ×2 heads) | conv: 2·1·1·32·16·7·7 = 50,176; FC: 2·784·1 = 1,568; ×2 heads | ≈ 103,488 |
| **Total (estimate)** | | **≈ 95.3M FLOPs ≈ 47.7M MACs** |

The trunk dominates (>99%). Direct FLOP comparison against Naoumi's 4.321 MMACs is **NOT COMPARABLE** (different task framing, bistatic MLP vs. this project's spatial-conv joint task, per the literature synthesis's own comparability table) and is flagged as such rather than presented as a competing number. No Tier-1 paper reports a concrete FLOP figure for the Lloria architecture family at all, so this would be a first for that family (closes Gap 1), not a beat-the-literature claim.

## 8. Expected advantages and weaknesses

**Advantages:** (1) front-end cost is near-zero in both params and FLOPs, so essentially the entire budget goes to the correction/refinement job rather than re-deriving the angle→antenna mapping; (2) directly and mechanically targets two named, unaddressed literature gaps simultaneously — Meneses-Albalá's own admitted gain-error gap and phase-error robustness beyond every paper's tested ceiling (5°); (3) the joint 2D heatmap is retained, so this does not inherit A5/A9's risk of breaking the "genuinely joint" property; (4) it stays compatible with the existing `evaluate_on_bank`/blob-detector pipeline (only the offset-refinement path needs a new eval branch), unlike a full DETR-style redesign (A7).

**Weaknesses:** (1) the IABC module's generalization is bounded by the self-supervised consistency loss's training distribution — if trained only on the Tier-1 grid (phase ≤5°, no gain-error data exists in the codebase today per Agent-R8), it may not extrapolate to the Tier-3 OOD grid (10–15° phase, 0.5–4dB gain) without new impairment-injection code being built first (gain-error injection is **not implemented anywhere in the repository**, confirmed by code audit — this is new physics code, not new config); (2) it still relies on a discrete coarse P×Q=16×16 grid for initial peak localization, so it may partially inherit the diagnosed Jacobian-singularity/pixel-quantization Pd ceiling (Gap 5) unless the crop-refinement head's continuous regression genuinely escapes grid discretization — unconfirmed; (3) sequential coarse→correct→refine pipeline has three points where error can compound (front-end miscorrection → wrong coarse peak → refinement head has no valid anchor).

## 9. Concrete failure modes

1. **Correction overshoot under untrained severity.** At phase error 15° (never seen in training, which only spans 0–5° per the Tier-1 grid), the IABC module's learned correction — extrapolating outside its training manifold — could systematically *over*-correct, actively moving the coarse peak into the wrong DFT beam bin entirely, producing a worse result than the uncorrected fixed front end (A8's exact flagged risk, now inherited by the correction module rather than avoided).
2. **Gain-error blind spot.** Because gain-error training data does not exist in the project pipeline today, if this architecture is trained only on the currently-available phase-error grid, `λ3`'s consistency loss never sees gain perturbations at all — the IABC module could learn to ignore the gain channel of its output entirely (a degenerate solution), silently failing the exact Meneses-Albalá gap it was designed to close, unless new gain-error generator code is built and used in training before any claim is made.
3. **Closely-spaced-source collapse.** At Δθ<5° (below the array's Rayleigh limit, `[GENUINE GAP]` per Agent-R3 §3b), two true peaks can fall inside the same 16×16 coarse bin; the crop-refinement head only sees one coarse anchor and cannot recover two angles from it — same failure category as A6, and the crop-based local design does not fix this (only global re-gridding or a set-prediction head would).
4. **Coupling leakage at high L.** The path-token bottleneck is designed to preserve AoA–AoD pairing, but this is untested above L=6 (the current training distribution's ceiling); at L∈{7,8,10} (Tier-3 OOD, per Agent-R3 §3a) it is unconfirmed whether the shared token still keeps the correct angle pairs together or starts swapping AoA/AoD across paths.
5. **Consistency-loss shortcut / train-test mismatch.** If the self-supervised `L_impair_consist` loss is implemented carelessly (e.g. the clean-reference pairing leaks information not derivable from the impaired sample alone at inference), the module could learn a shortcut that only works because a paired clean sample existed during training — this must be checked by confirming inference-time inputs are strictly `Y` alone, no clean reference.

## 10. Experiments needed to validate

1. **Ablation battery** (on identical `frozen_banks/eval_bank.npz`, reusing `evaluate_on_bank`): (a) − IABC (revert to fixed, uncorrected DFT front end) evaluated across the Tier-1 phase/gain grid, to isolate IABC's actual contribution; (b) − crop-refinement heads (coarse heatmap + blob detector only) vs. full model, with P95 tail-error reported specifically near end-fire angles (Gap 5 test); (c) − SE/`L_SE_snr` auxiliary loss, to test whether A3's "attention correlates with SNR" hypothesis actually holds for this task or is a false lead.
2. **Full Tier-1 robustness battery** per Agent-R3 (SNR sweep in-distribution core, phase δ/γ∈{0,1,2,5}°, L∈{1..6}) as the baseline registry comparison point.
3. **Genuine-gap extrapolation tests**: phase 10–15°, gain 0–4dB (requires building the missing gain-error injection code first — new physics work, not new config, per Agent-R8), L∈{7,8,10}, Δθ down to 1°, and the SNR extrapolation tails (−25,−20,30,35dB) — all flagged `[GENUINE GAP]` and this design's core claim rests on results here, not on in-distribution numbers.
4. **End-fire density probe** (Agent-R3 §4) to directly test whether the crop-refinement mechanism actually escapes the diagnosed pixel-quantization ceiling, run alongside a "refinement-heads-removed" variant for direct comparison.
5. **New instrumentation**: FLOPs (via `ptflops`/`thop`, none exist in-repo today), peak memory, and standalone inference latency (generalizing `benchmark_latency()` from `proposed pin architecture.ipynb`) — needed to make the "lightweight" claim concrete and to close Gap 1/Gap 8 as a supporting (not headline) contribution.
6. **Multi-seed variance run** (Gap 10) given the correction/attention modules introduce additional stochastic training dynamics beyond the teacher's.
7. **Notebook-4-style nuisance-path stress test**, extended to this architecture, since it is the project's only existing evidence that a smaller model can *beat* the teacher under stress — worth checking whether IABR-Net's correction mechanism reproduces or exceeds that effect.

## 11. Novelty argument

**Self-classification: PARTIALLY NEW, with the specific-application claim UNCERTAIN pending a dedicated search.**

Justification, following Agent-R7's framework directly rather than asserting novelty:

- At the **pattern level**, "a small learned network corrects/regularizes inputs to a classical or near-classical processing stage" is an established paradigm — SubspaceNet is a verified instance of it (learned covariance surrogate → classical resolver) [EVIDENCE: PAPER], and Agent-R7 independently found other instances (deep-unrolling+GNN DOA hybrids) outside the reviewed corpus. IABR-Net's IABC module is structurally this same family, just placed *before* a classical transform rather than *after* one, and targeting *hardware impairment* rather than *covariance denoising*. By Agent-R7's own test 3 (§1), a block-for-block application of an established pattern to a new task variant is a **domain-transfer contribution, not a conceptual one** — so at this level, per Agent-R7's own angle-E verdict, this is **KNOWN COMBINATION at the pattern level**.
- At the **specific-application level**, the combination of (a) a fixed DFT beamspace reduction, (b) a self-supervised clean/impaired consistency-trained correction module specifically closing Meneses-Albalá's own named gain-error gap, and (c) a path-token-coupled coarse+local-refine dual head that preserves joint AoA/AoD pairing, applied to the *joint 2D* task, is not present in the 14-paper corpus (Agent-R4's Candidate 1 is the closest literature-corpus anchor and is explicitly a *fixed*, non-adaptive front end — A8's own flagged weakness). This matches Agent-R7's angle-J verdict (**UNCERTAIN**) almost exactly: within the corpus this direction is untested for the joint 2D task, but Agent-R7's own supplementary search surfaced an adjacent pattern ("DNN-based beamformer + MUSIC subspace tracking, alternating DL/classical roles") close enough in spirit that a dedicated search specifically for "classical front-end, learned impairment correction, joint AoA/AoD" is required before this is treated as settled.
- Per Agent-R7's explicit methodological requirement (§1, item 1: tie the architectural choice to a specific diagnosed failure, not a generic swap), this design does clear that bar — the correction module is motivated by a *named, evidenced* gap (Meneses-Albalá's own admitted future work), not a generic "let's add a learned layer" choice. That is the strongest part of the novelty case.

**Verdict: PARTIALLY NEW** (domain-transfer of an established model-based-DL pattern to a genuinely under-tested robustness axis for the joint-2D task), **with the "is this exact combination unpublished" claim held at UNCERTAIN** until Agent-R7's recommended dedicated searches ("learned impairment-adaptive beamspace correction DOA", "classical front-end learned calibration angle estimation") are actually run — this specification does not claim NEW, consistent with the protocol's requirement that no `[HYPOTHESIS]` novel-twist claim self-certify without the Novelty Researcher's and Adversarial Review Team's independent pass.

---

# Candidate C (refined after critique: True)

## Final specification

# Candidate C — Architecture Specification (REVISED v2 — post-critique)

## SpectraSparse‑CoFi (SSC‑Net) — Grouped‑FNO Trunk + Depth‑Extended GridlessUnfold Precision Head, Coarse‑to‑Fine Joint Heatmap Fusion

*Revision note: this version responds point‑by‑point to Critique #1 and Critique #2. Several design decisions were changed outright (not just re‑argued); others are defended with reasoning. See "Changes made in response to critique" at the end for a compact diff.*

---

### Motivation (one paragraph)

Two of this project's own from‑scratch screening results point in the same direction: the FNO spectral‑conv core block is the current best alternative to the dense/flatten bottleneck the literature itself flags as its worst component (Pd 0.6235, 88% of teacher's Pd, at 334,321 params — 71% of teacher) `[EVIDENCE: PROJECT RESULTS.md]`, and the GridlessUnfold complex soft‑threshold layer directly targets the diagnosed Jacobian‑singularity/pixel‑quantization root cause of the project's own Pd≈0.972 ceiling, but under‑performed (Pd 0.2216, 30,201 params) at a shallow 8‑block screening depth `[EVIDENCE: PROJECT]`. **Revised framing:** whether that under‑performance was capacity‑limited or mechanism‑limited is *not yet known* (Critique #1, item 1) — this specification no longer asserts capacity‑limitation as fact. SSC‑Net remains a proposal to fuse these two components into a coarse‑to‑fine pipeline, but the depth‑extension of GridlessUnfold is now **explicitly gated behind a cheap diagnostic** (new Experiment 0, below) rather than committed to up front. The pipeline still targets Gap 5 (pixel‑quantization ceiling) and Gap 2 (dense bottleneck) and still adds a post‑hoc structural‑pruning pass (Gap 7), but this revision **retracts the "lightweight" framing as a headline claim** (Critique #1‑2, Critique #2‑1/2/3/4) until the numbers below are validated, and narrows the novelty claim (Critique #1‑4, Critique #2‑6).

---

### Borrowed components (cited)

| Component | Source | Role in SSC-Net | Status after revision |
|---|---|---|---|
| FNO / spectral-convolution block | Project's own architecture-screening result `[EVIDENCE: PROJECT RESULTS.md]` | Trunk, coarse global-receptive-field extraction | unchanged |
| GridlessUnfold complex soft-threshold layer | Project-internal, PIA-Net-inspired `[EVIDENCE: PROJECT]` | Precision refinement head | **now conditional** — depth extension gated on Experiment 0 |
| ResNet-shell conv stem / residual pattern | Lloria et al., TVT 2026 / PIMRC 2024 `[EVIDENCE: PAPER]` | Input stem, output decode | unchanged, but now costed as its **own explicit line item** (see Param section) |
| Grouped convolution (ResNeXt-style, G=4) | Generic efficient-CNN pattern, flagged by Agent-R2 item 3, untried in this project | Channel-mixing compression inside FNO trunk | unchanged, still untested on spectral weight tensors — flagged uncertainty widened |
| SubspaceNet-pattern (learned surrogate → lighter resolver) | Shmuel et al. 2025 TVT `[EVIDENCE: PAPER]` | Design-philosophy justification | unchanged |
| Structured (magnitude/SNR-aware) channel pruning | Project's compression track, r=8 result (314,513 params, ΔPd −0.0147 vs teacher) `[EVIDENCE: PROJECT baseline_models/README.md]` | Post-hoc trunk compression | **retention ratio no longer transplanted as-is** — see revised Step 5 |
| Coarse-to-fine grid rationale | Protocol §11 (Agent-R1's A6) | Output-representation design | **downgraded**: acknowledged as a well-known pattern in the broader CV literature (cascaded/coarse-to-fine heatmap refinement — stacked-hourglass / cascaded-pyramid-style designs), not DOA-specific novelty `[INFERENCE — general field knowledge, not covered by the reviewed 14-paper corpus]` |

---

### The novel component(s), stated precisely — **reduced from 5 to 4, one re-weighted down**

Critique #1‑4 and Critique #2‑6 both correctly note that "single joint heatmap, not split AoA/AoD heads" (old component 3) is not a design choice this spec makes — it's the pre-existing default representation already used by the FNO/GU screening runs and the Lloria pipeline being reused as anchors. **This is removed from the novel-component list and moved to Borrowed Components above**, where it belongs. The remaining components:

1. **Grouped-channel-mixing FNO blocks (G=4).** Untried on FNO's spectral channel-mixing tensor anywhere in this project or the reviewed literature. Uncertainty on the achievable reduction factor is now stated as a **range**, not a single "conservative 3×" (see Param section) — Critique #2‑1/2 correctly noted the original single-point estimate carried false precision.
2. **Depth-extended, resolution-decoupled GridlessUnfold head — now CONDITIONAL.** Repositioning GU from whole-network core block to a post-trunk precision head, and deepening it 8→16 blocks, is **no longer asserted as capacity-limited-therefore-safe-to-deepen**. It is gated behind Experiment 0 (a GU-only depth sweep). If Experiment 0 shows GU's Pd does not improve materially with depth (i.e., the poor 8‑block result is mechanism‑limited, not capacity‑limited), this component is **dropped from the design** and SSC‑Net reduces to Grouped‑FNO‑trunk + a shallow (8‑block) GU head, with the rest of the pipeline unchanged. This is a genuine architectural fallback path, not a caveat bolted onto an unchanged design.
3. **Jacobian‑aware loss reweighting — reframed.** Critique #2‑5 correctly argues that a loss‑level reweighting term cannot manufacture pixel resolution that a fixed 256×256 grid does not have; the *representational* fix is the coarse‑to‑fine architecture, not the loss. This spec now describes the term as **an auxiliary training‑dynamics mitigation** ("encourages the network to spend more of its fixed representational capacity accurately localizing the peak within existing bins near end‑fire, analogous to hard‑example reweighting in general heatmap‑regression literature `[INFERENCE]`"), not a "direct countermeasure to the singularity." We partially disagree that this makes the term mechanically inert (see Rebuttal R2 in the Changes section) — it is kept, but its causal claim is downgraded and it is no longer listed as attacking Gap 5 on its own; only the architecture is credited with that.
4. **Post‑hoc SNR‑aware structured pruning applied only to the trunk, after fusion training, with the retention ratio re‑derived on the actual grouped‑FNO trunk (not transplanted from the teacher run), plus a mandatory post‑pruning fine‑tune pass.** The fine‑tune pass is a new addition (Critique #1 Failure Mode 3 / Critique #2‑2): the fusion gate is trained against the *unpruned* trunk's feature statistics, so a short fine‑tune of the fusion gate + decode stack after pruning is now a required pipeline step, not an omission.

---

### Data flow

```
Input Y (raw antenna-domain, complex, nt=nr=16)
        │
        ▼
  Conv stem (ResNet-style, shared) ── borrowed from Lloria TVT2026
  [NOW COSTED AS A SEPARATE, ONE-TIME PARAM LINE ITEM — see Param section]
        │
        ▼
  Grouped-FNO trunk (6 blocks, G=4, spectral-domain)
  [FFT/IFFT cost per block now explicitly included in FLOP accounting]
        │
        ▼
  Coarse joint 2D heatmap decode (64×64), deep-supervised with L_coarse
        │
        ▼
  GridlessUnfold precision head — DEPTH GATED BY EXPERIMENT 0
  (16 blocks if Exp-0 validates capacity-limitation; else 8 blocks, unchanged
  from screening depth, if Exp-0 shows mechanism-limitation)
        │
        ▼
  Fusion/gating 1×1 conv (trunk coarse-stage skip ⊕ GU fine-stage feature)
  [RETRAINED/FINE-TUNED after post-hoc pruning — new required step]
        │
        ▼
  Shared output decode conv stack → joint 2D heatmap (256×256)
        │
        ▼
  Non-differentiable blob detection → peak location
        │
        ├──► AoA output (ψ, peak x-coordinate)
        └──► AoD output (φ, peak y-coordinate)
```

Shared stem/trunk/decode, single joint map, no AoA/AoD split until peak read‑off — unchanged from the original, but no longer claimed as a *novel* contribution (see above).

**Joint‑ness claim downgraded to a hypothesis.** Critique #2‑8 is accepted: a per‑pixel loss does not demonstrably enforce AoA–AoD statistical coupling beyond spatial co‑location, and the same outcome could plausibly be reached by a dual‑head design with concatenated outputs. "Preserves the genuinely joint property" is moved out of *Expected advantages* and into a **testable hypothesis**, validated only by the new correlated‑error experiment (see Experiments Needed, item 8).

---

### Loss function

```
L = L_coarse(H_64, GT_64)
  + λ1 · L_fine(H_256, GT_256; w(ψ,φ))      # Jacobian-aware reweighting — reframed, see novel component 3
  + λ2 · L_sparsity(GU soft-threshold activations)
  + λ3 · L_distill(H_256, H_256^teacher)     # OPTIONAL
```

Unchanged formulas, but:
- `w(ψ,φ) = 1/max(|sinψ|, ε)` is now documented as a **training‑dynamics aid, not a representational fix** (see novel component 3).
- λ1, λ2, λ3, ε together constitute a 4‑dimensional hyperparameter search that was not costed in the original spec — this is now listed explicitly as Failure Mode 6 (new) and constrained by the staged experimental protocol below (loss terms are turned on one at a time, not searched jointly from the start).

---

### Parameter count estimate (with arithmetic) — **restructured to fix the double-counting error (Critique #2‑1)**

**The structural bug.** The original spec computed a "per‑block average" for FNO (41,790 params/block) that the anchor text itself says "includes its share of stem+head" — i.e., a one‑time, non‑repeating cost was baked into a per‑block figure and then multiplied by 6. This is corrected below by separating **Stem (one‑time) + Trunk (repeats ×6) + Head (repeats ×16, or ×8 if Experiment 0 fails) + Fusion + Decode** into non‑overlapping components.

**Honest limitation, stated up front:** we do not have the literal stem‑only or head‑only parameter counts for the existing FNO‑8/GU‑8 checkpoints — only their totals. Per Protocol §31 rule 2 ("never invent missing hyperparameters") and §6 ("always inspect code when available"), the correct fix is to **read the actual model‑definition code** and report exact stem/block/head counts. **This is now Required Experiment 0a (blocking)** — no headline parameter number should be trusted until it is done. Pending that, we replace the previous false‑precision single number with an explicit **sensitivity range** over the unknown stem+head fraction `f` of the FNO‑8 total:

| f (stem+head share) | Stem (one-time) | FNO per-block (net) | Grouped-FNO block (×0.5333, G=4 3× mixing reduction) | Trunk ×6 |
|---|---:|---:|---:|---:|
| 10% | 33,432 | 37,611 | 20,057 | 120,342 |
| 20% (central) | 66,864 | 33,432 | 17,831 | 106,986 |
| 30% | 100,296 | 29,253 | 15,602 | 93,612 |

(0.5333 = 0.70×[1/3] + 0.30, same conservative 3× mixing-reduction assumption as the original, itself still unvalidated — see Weaknesses.)

GU head (same fixed-fraction correction applied to the 30,201-param, 8-block anchor; ×16 blocks if Exp‑0 validates depth extension):

| f | GU per-block (net) | Head ×16 |
|---|---:|---:|
| 10% | 3,398 | 54,363 |
| 20% (central) | 3,020 | 48,320 |
| 30% | 2,643 | 42,285 |

Fusion (4,160) + decode (8,000) unchanged, order-of-magnitude, still least-certain line item.

**Pre-pruning total, central (f=20%):** 66,864 + 106,986 + 48,320 + 4,160 + 8,000 = **≈234,330 params**
**Full sensitivity range (f: 10–30%): ≈220,297 – 248,353 params**

This is **higher**, not lower, than the original's uncorrected 206,290 — the original's error happened to under-state the true single-stem cost relative to a properly separated accounting; the direction of the correction is not something we could have predicted without redoing the arithmetic, which is exactly the point of fixing it rather than defending the original number.

**Step 5 — Pruning, re-derived (Critique #1‑3, Critique #2‑2).** The teacher's r=8 magnitude‑pruning retention ratio (67%) is **no longer applied as the expected value**. A grouped‑convolution trunk has already removed much of the redundancy that magnitude pruning typically exploits, so it should be *less* prunable than a dense teacher, not equally prunable. **Required Experiment 0b (blocking):** run magnitude pruning directly on the trained grouped‑FNO trunk and measure the actual retention ratio. Pending that, we use a sensitivity range that treats 67% as an *unvalidated upper bound* (least conservative) rather than the estimate:

| Retention ratio r | Trunk post-pruning (f=20% central) | Total post-pruning |
|---|---:|---:|
| 90% (conservative, expected for an already-compressed trunk) | 96,287 | ≈223,631 |
| 80% | 85,589 | ≈212,933 |
| 67% (teacher-derived, unvalidated for this architecture) | 71,681 | ≈199,025 |

**Revised headline: ≈199K–224K params (central estimate), full range ≈185K–258K across all stated uncertainties** — not "162,158." A mandatory post‑pruning fine‑tune of the fusion gate + decode stack (new pipeline step, addressing Critique #1 Failure Mode 3) is added before any post‑pruning accuracy number is reported.

**Required control (Critique #1‑2, "apples-to-apples"):** a **pruned‑FNO‑only baseline**, pruned by the *same* re‑derived methodology at the *same* measured ratio, is now a required comparison point (Experiment 0c) before any "SSC‑Net is X% smaller than FNO" claim is made. Against the *unpruned* 334,321‑param FNO baseline, the revised range is ~60–75% of its size (not 48%) — a materially weaker efficiency story than originally headlined, and this spec no longer leads with "lightweight" as a result.

---

### FLOP estimate (with arithmetic) — **FFT/IFFT cost now included (Critique #2‑3)**

The original `FLOPs = 2×params×H×W` formula is a conv/pointwise approximation only; it omits the FFT and inverse FFT that FNO's spectral‑convolution mechanism performs at **every trunk block** — its namesake operation, not a rounding error. Standard FFT cost estimate: `FLOPs_FFT ≈ 5·N·log2(N)` per complex 2D transform of size N=H×W, ×2 for forward+inverse, ×channel width C (unknown — not in project evidence, **flagged as an ASSUMPTION**, swept over a plausible range):

At the coarse 64×64 stage (N=4096, log2N=12), per block: `2·C·5·4096·12 = 491,520·C`. For 6 trunk blocks: `2.949M·C`.

| Assumed trunk channel width C | FFT cost, 6 blocks | Trunk conv FLOPs (central, r=80%) | FFT as % of trunk conv cost |
|---|---:|---:|---:|
| 32 | 94.4M | 701.1M | +13% |
| 64 | 188.8M | 701.1M | +27% |
| 128 | 377.6M | 701.1M | +54% |
| 256 | 755.2M | 701.1M | +108% (dominant) |

This directly validates Critique #2‑3's core point: the FFT term is **not negligible and could be dominant** depending on the (currently unknown) channel width. The candidate no longer claims to "close Gap 1" with the previous single number; that framing is retracted. Only real instrumentation (`ptflops`/`thop` **with a custom hook for `torch.fft.fft2`/`ifft2`**, since standard autograd‑based profilers typically miss FFT ops entirely) can produce a trustworthy figure — this is now stated as a hard precondition, not a nice‑to‑have (Required Experiment, unchanged position but strengthened language).

Using C=32 (a plausible but unverified central assumption for a "lightweight" trunk) and the revised param figures:

- Trunk (64×64, r=80% pruning): 701.1M (conv) + 94.4M (FFT) ≈ **795.5M**
- GU head (256×256, revised 48,320 params): 2×48,320×65,536 ≈ **6.33B**
- Fusion+decode (256×256): 2×12,160×65,536 ≈ **1.59B**

**Total ≈ 8.72 GFLOPs (central, C=32), range ≈ 8.4B–9.4B+ depending on C** — revised down slightly from the original's 10.24 GFLOPs (due to the corrected, smaller head param count) while gaining an explicit, non‑trivial uncertainty band from the newly‑included FFT term. Still **not comparable** to Naoumi's MMACs figure (different task/resolution/convention, rated NOT COMPARABLE) — this caveat is unchanged and still required.

**Latency is not FLOPs (Critique #2‑4).** Grouped convolutions can underperform their theoretical FLOP/param savings on real GPUs due to memory‑access patterns and kernel‑launch overhead at low per‑group channel counts. **Partial rebuttal:** at G=4 with C∈{32,…,128}, per‑group channel counts are 8–32, which is generally within the range where modern cuDNN grouped‑conv kernels retain reasonable utilization — severe degradation is more consistently reported at per‑group counts below ~4–8 `[INFERENCE — general hardware/kernel knowledge, not project-specific evidence]`. This is a reason for cautious optimism, **not** a substitute for measurement. No latency number is claimed anywhere in this spec; "lightweight" is explicitly **not** asserted on any axis (params, FLOPs, or latency) until Experiment 3 below is run on real hardware, with NN‑only and end‑to‑end latency reported **separately** (see next point).

---

### NN-only vs. end-to-end latency — new, explicit distinction (Critique #1‑5)

The pipeline retains a non‑differentiable blob‑detection step after two very differently‑shaped network stages (64×64 trunk, 256×256 head). The benchmark extension to `benchmark_latency()` must report:

- `L_NN` — forward pass only (trunk + head + fusion + decode), and
- `L_e2e` — `L_NN` + the blob‑detection peak‑search step,

as **two separate numbers**, never conflated. Any deployment claim must cite which one it means.

---

### Expected advantages

- Reuses two independently‑validated project components with the most concrete prior‑evidence trail of any single‑mechanism design.
- Attacks Gap 2, Gap 5, Gap 6, Gap 7 with one pipeline — but the Gap‑5 claim is now credited to the **architecture** (coarse‑to‑fine resolution), not the loss term (corrected per Critique #2‑5).
- Post‑hoc pruning reuses existing infrastructure — but now with a re‑derived ratio and a mandatory fine‑tune, not a transplanted number.
- The GU‑depth‑vs‑mechanism question, previously an unstated assumption, is now an explicit, cheap, first‑run diagnostic (Experiment 0) — this makes the design **falsifiable early and cheaply**, before the expensive full build, which is itself a methodological improvement over the original plan.

### Expected weaknesses

- Inherits FNO's own unexplained high‑SNR degradation (Pd ~0.80→0.70, 10dB→25dB) `[EVIDENCE: PROJECT RESULTS.md]` — grouping further reduces trunk capacity, risk unchanged from v1.
- Still relies on non‑differentiable blob detection; does not architecturally remove the peak‑search step (A7/DETR does). Unchanged limitation.
- **"Lightweight" is now an open question, not a claim.** Params, FLOPs, and latency all carry unresolved uncertainty ranges (params ±25%, FLOPs FFT-term ±100%+, latency unmeasured). This spec explicitly does not assert efficiency superiority until Experiments 0a/0b/0c/3 are complete.
- Grouped spectral convolution remains untested on any FNO weight tensor anywhere in the reviewed corpus; the 3× mixing‑reduction figure is still an extrapolation from non‑spectral conv literature.
- Total processing depth (6 grouped‑FNO + up to 16 GU = up to 22 blocks) is ~2.75× either single‑mechanism screening baseline (8 blocks) — a capacity/depth confound that the ablation ladder must now explicitly control for (see Experiments, item 1).

---

### Concrete failure modes

1. **Grouping collapses FNO's global‑mixing advantage** — unchanged from v1.
2. **Coarse‑stage under‑detection at high L** (closely‑spaced sources collapsing into one 64×64 bin) — unchanged from v1.
3. **Pruning/fusion‑gate distribution shift** — **now mitigated by design** (mandatory post‑pruning fine‑tune added), but the fine‑tune itself could under‑correct if the shift is large; residual risk retained, not eliminated.
4. **Jacobian‑reweighted loss overfitting to end‑fire tail cases** — unchanged risk, now understood as a training‑dynamics risk on a *reframed* (not representational) mechanism.
5. **FLOP‑cost‑confounded step‑matched comparisons** — unchanged, now compounded by the FFT‑cost uncertainty above.
6. **(NEW) Hyperparameter‑search burden.** λ1, λ2, λ3, ε form an uncosted 4‑D search space; the staged experimental protocol below (loss terms activated one at a time) is the mitigation, not a guarantee it stays cheap.
7. **(NEW) Grouped spectral mixing changing effective frequency coverage.** Grouping the spectral channel‑mixing tensor may alter which frequency bands each group can represent, potentially interacting badly with the aliasing/end‑fire geometry already diagnosed as the project's core failure mode — a second‑order risk beyond simple "capacity loss," not previously listed.
8. **(NEW) Ablation‑ladder cost/compounding‑error risk.** With up to 5 candidate axes (grouping, GU repositioning/depth, Jacobian loss, pruning, distillation), a full factorial ablation is expensive and may not be fully run before a paper claim is made, in which case a negative result cannot be cleanly attributed to a single component. Mitigated by the staged protocol below, which resolves axes sequentially rather than assuming the full ladder will be completed.

---

### Experiments needed to validate — **restructured into a staged protocol (Critique #2‑7)**

The original "ablation ladder" changed architecture, loss, and compression simultaneously, confounding any observed gain. Experiments are now **explicitly sequenced**, each stage gating the next:

**Stage 0 (new, blocking, cheapest, run first):**
0. **GU‑only depth sweep** (8/12/16/24 blocks, standard uniform loss, no fusion, no trunk changes) — resolves whether novel component 2 (deepened GU head) is worth building at all.
0a. **Code inspection** of existing FNO‑8/GU‑8 checkpoints for literal stem/block/head param counts (resolves the Param‑section uncertainty directly, cheaper than any training run).
0b. **Magnitude pruning run directly on the trained grouped‑FNO trunk** to measure its actual retention ratio (resolves Param Step‑5 uncertainty).
0c. **Pruned‑FNO‑only control**, same methodology/ratio as 0b applied to the ungrouped FNO‑8 baseline (resolves the "apples‑to‑apples lightweight" comparison).

**Stage 1 — architecture‑only comparison, fixed uniform loss, no Jacobian weighting, no sparsity term, no distillation, no pruning:** dense FNO‑8, GU‑8 (or GU‑N per Stage 0's result), a **depth‑matched control** (single‑mechanism network extended to the same ~14–22 total blocks SSC‑Net uses, to isolate depth/capacity from the coarse‑to‑fine combination — Critique #1‑6/Critique #2‑7), and the unpruned Grouped‑FNO‑trunk+GU‑head combination.

**Stage 2 — add loss terms one at a time:** + Jacobian weighting only, then + sparsity term, on top of the Stage‑1 winner.

**Stage 3 — add pruning**, using the Stage‑0b‑derived ratio, with the mandatory post‑pruning fine‑tune.

**Stage 4 — distillation on/off** (λ3=0 vs. λ3>0), using the existing teacher checkpoint.

**Stage 5 — full battery on the Stage‑4 winner**, unchanged from v1: FLOPs/params/latency/memory benchmark (with FFT‑aware instrumentation and separate `L_NN`/`L_e2e`), Tier‑1 robustness (SNR, δ/γ phase error, L‑sweep), end‑fire density probe, angular‑separation sweep.

8. **(NEW) Joint‑coupling test (Critique #2‑8).** Correlated‑error analysis (e.g., Pearson correlation or mutual information between AoA and AoD residuals) comparing SSC‑Net's single joint map against a matched‑capacity dual‑head control from Stage 1, to test — not assert — whether the joint representation produces measurably tighter AoA–AoD coupling than a dual‑head design with concatenated outputs.

---

### Novelty argument — self-classification: **PARTIALLY NEW (narrowed), one component downgraded to KNOWN COMBINATION**

Critique #1‑4 and Critique #2‑6 both attack the novelty count from different angles, and both are accepted in part:

- **The coarse‑to‑fine, single‑joint‑map backbone pattern is reclassified as KNOWN COMBINATION**, not merely "known within the DOA literature" (R1/R7's original framing) but known well beyond it — cascaded/coarse‑to‑fine heatmap refinement is a decades‑old, extremely well‑established pattern in the broader 2D keypoint/pose‑estimation literature (stacked‑hourglass‑ and cascaded‑pyramid‑style designs) `[INFERENCE — general CV field knowledge, outside the reviewed 14‑paper corpus and not independently verified against it here]`. The original novelty search scoped only against the project's own reports and the DOA‑specific corpus, which Critique #2‑6 correctly identifies as a favorably narrow comparator set. This spec does not claim novelty for the backbone pattern.
- **What remains independently defensible, per‑component (not as a package):**
  - Grouped convolution inside an FNO spectral weight tensor — still untried anywhere in the reviewed corpus or general literature as far as searched; **UNCERTAIN, not confirmed novel**, same caveat as before (absence of evidence in a non‑exhaustive search is weak evidence of absence).
  - GU repositioned as a conditional, diagnostically‑gated precision head — now explicitly *not* claimed unless Experiment 0 supports it; if it is dropped by that gate, this component disappears from the final candidate entirely.
  - The measurement‑rigor bundle (params + FFT‑aware FLOPs + separately‑measured NN/end‑to‑end latency + memory, multi‑seed variance, staged deconfounded ablation) — still the single most defensible, near‑zero‑risk novelty pillar, and arguably *more* valuable after this revision, since the staged protocol and FFT‑inclusive FLOP accounting are themselves methodological contributions largely absent from the reviewed corpus.
- **Rebuttal to the implication that a downgraded backbone sinks the whole candidate:** per Agent‑R7's own stated criteria, novelty should be assessed per‑component, not only at the top‑level pattern. A KNOWN‑COMBINATION backbone does not retroactively make an untested component‑level modification (grouped spectral mixing) or a rigor‑bundle contribution non‑novel; it only means the *headline* claim must not be "a new architecture," which this spec no longer asserts.

**Overall: PARTIALLY NEW, narrowly** — defensible as "grouped‑spectral‑FNO trunk (component‑level, uncertain‑novel) conditionally combined with a repositioned precision head (gated, may not survive), evaluated with a measurement‑rigor bundle largely absent from the literature" — explicitly **not** "a new architecture family," and explicitly **not** validated until the Stage 0–4 protocol above has run.

---

## Changes made in response to critique

**Design changes (not just added caveats):**
1. GU depth‑extension (8→16 blocks) is now **conditional**, gated behind a new Experiment 0 (GU‑only depth sweep); the design has an explicit fallback (keep GU at 8 blocks) if the diagnostic fails. — *Critique #1‑1*
2. "Single joint heatmap" removed from the novel‑component list and reclassified as a borrowed/default representation. Novel‑component count reduced 5→4. — *Critique #1‑4, #2‑6*
3. Parameter accounting restructured into non‑overlapping Stem/Trunk/Head/Fusion/Decode components instead of an amortized per‑block average multiplied by block count; false‑precision single number ("162,158") replaced with an explicit sensitivity range (≈185K–258K, central ≈199K–224K) and a blocking code‑inspection experiment (0a). — *Critique #2‑1*
4. Pruning retention ratio no longer transplanted from the teacher's r=8 run; re‑derivation on the actual grouped trunk is now a blocking experiment (0b), the teacher's 67% is relabeled an unvalidated upper bound, and a mandatory post‑pruning fine‑tune of the fusion gate is added as a new pipeline step. — *Critique #1‑3, #2‑2*
5. A same‑methodology pruned‑FNO‑only control (0c) is added as a precondition for any "X% smaller than FNO" claim. — *Critique #1‑2*
6. FLOP formula extended to include FFT/IFFT cost per FNO block, with a channel‑width sensitivity table showing the term ranges from +13% to fully dominant; the "closes Gap 1" claim is retracted pending real FFT‑aware instrumentation. — *Critique #2‑3*
7. Latency benchmarking now requires two separately reported numbers (`L_NN`, `L_e2e`), never conflated. — *Critique #1‑5*
8. The Jacobian‑aware loss term's causal claim is downgraded from "direct countermeasure to the singularity" to "auxiliary training‑dynamics mitigation"; the architecture, not the loss, is now credited with the Gap‑5 fix. — *Critique #2‑5*
9. The ablation ladder is restructured into a staged, sequential protocol (Stage 0 → 4) that resolves architecture, loss, and compression axes one at a time instead of in parallel, plus an explicit depth‑matched control. — *Critique #1‑6, #2‑7*
10. A new joint‑coupling experiment (correlated‑error analysis vs. a matched dual‑head control) is added; "genuinely joint" is downgraded from an asserted advantage to a tested hypothesis. — *Critique #2‑8*
11. Three failure modes added: hyperparameter‑search burden; grouped spectral mixing's interaction with aliasing/end‑fire geometry; ablation‑ladder cost/compounding‑error risk. — *Critique #2‑9*

**Rebuttals (reasoned disagreement, no design change):**
- R1 — *Grouped‑conv GPU underperformance (Critique #2‑4):* partially rebutted. At G=4 with plausible channel widths (32–128), per‑group channel counts (8–32) are generally above the range where severe kernel‑efficiency loss is typically reported; this justifies cautious optimism but does not replace the required hardware benchmark, which is kept as a hard precondition for any "lightweight" claim.
- R2 — *Jacobian loss claimed mechanically inert (Critique #2‑5):* partially rebutted. While the term cannot create pixel resolution a fixed grid lacks, per‑pixel reweighting plausibly can still improve effective sub‑pixel localization precision within existing bins by reallocating training‑time supervisory emphasis near end‑fire — a distinct and more modest mechanism than "fixing the singularity," which is why the causal language was downgraded rather than the term being deleted.
- R3 — *Downgraded backbone novelty sinking the whole candidate (Critique #2‑6):* rebutted. Novelty is assessed per‑component per Agent‑R7's own framework; a known‑combination backbone does not by itself invalidate the still‑untested grouped‑spectral‑mixing modification or the measurement‑rigor‑bundle contribution, both of which are argued and evaluated independently above.

## Adversarial review

### Critic 1 (fatal_flaw_found=True)

Adversarial review of SSC-Net (SpectraSparse-CoFi). I found several defensible weaknesses that the candidate itself already surfaces honestly, plus a few it does not, and I judge the composite serious enough that the design should not proceed unmodified until specific diagnostic experiments are added ahead of the full build.

1) UNVALIDATED LOAD-BEARING ASSUMPTION (most serious). The entire justification for novel component 2 (depth-extending GridlessUnfold 8→16 blocks and repositioning it as a precision head) rests on the claim that GU's poor screening result (Pd 0.2216 at 8 blocks) was "capacity-limited, not mechanism-limited." This is asserted, not evidenced — no ablation, scaling curve, or diagnostic run is cited to distinguish "the mechanism doesn't work at this task" from "it just needed more blocks." An equally plausible reading of the same evidence (Pd 0.2216 vs. FNO's 0.6235 and teacher's ~0.972) is that GridlessUnfold's complex soft-threshold mechanism is simply poorly suited as a coarse global estimator, in which case doubling its depth and repurposing it downstream of a coarse map could still fail for the same underlying reason. The experiments list buries this question inside a general ablation ladder rather than proposing it as a targeted, first, cheap diagnostic (e.g., a GU-only depth-sweep at 8/12/16 blocks on the existing task) before committing to the full fused build. Until that specific test is run, this is a load-bearing assumption stacked under two of the five "novel" components.

2) "Lightweight" claim is comparison-mismatched, and the document's own arithmetic proves it. The headline "≈162K params, ~35% of teacher, ~48% of FNO baseline" compares a post-pruning SSC-Net against an unpruned FNO baseline (334,321 params) — not an apples-to-apples comparison unless FNO is pruned by the same methodology for the comparison to be meaningful. More importantly, the FLOP arithmetic the candidate itself computes (10.24 GFLOPs, with GU head consuming ~77% of FLOPs from only ~37% of params, because it runs at 256×256 vs. the trunk's 64×64) directly contradicts the "lightweight" framing headline gives most prominence to. No FLOP figure exists anywhere in the project or literature for the teacher or the FNO/GU baselines, so the 10.24 GFLOPs number cannot actually be shown to be smaller, larger, or comparable to anything — the "lightweight" story is a param-count-only claim dressed as an efficiency win, while the document's own compute analysis argues against it. This is somewhat self-flagged ("Expected weaknesses," "genuine design tension") but the headline result line still leads with the favorable framing.

3) Pruning-ratio transfer is a second unjustified compounding assumption. The 67% width-retention ratio being applied to the trunk was measured on a magnitude-pruning run of a different (denser, presumably teacher-scale) architecture (r=8, 314,513/469,393 params). Grouped convolutions already remove a large fraction of the redundant channel-mixing weights that magnitude pruning typically exploits; assuming a lean, already-compressed grouped-FNO trunk has the same prunable redundancy as a dense architecture is optimistic and unvalidated. Combined with the 3× "conservative" grouped-conv reduction assumption (itself extrapolated from generic, non-spectral, non-FNO literature, and explicitly labeled as never tested on an FNO spectral weight tensor anywhere in this project or the reviewed corpus), the final ≈162K-param estimate is the product of three independent, uncertain, and all-favorably-directioned assumptions. If any one of the three is off (e.g., grouped spectral mixing only gets 1.5× instead of 3×, or the trunk only prunes to 85% instead of 67%), the "lightweight" headline number moves substantially, and there's no sensitivity range given despite the arithmetic being presented with false precision (e.g. "162,158").

4) Novel component #3 is mislabeled. "Single-joint-heatmap coarse-to-fine fusion (not split AoA/AoD heads)" is presented as a novel design choice, but it is simply the existing default representation already used by the project's own Lloria-style pipeline and by the FNO/GU screening runs being reused as anchors. Not changing something is not a novel contribution — this inflates the apparent count of genuinely new ideas from effectively 3 (grouped-FNO, GU repositioning, Jacobian loss) to a padded 5, which matters when weighing "PARTIALLY NEW" against reviewer scrutiny.

5) Latency vs. system latency is not disambiguated. The pipeline explicitly retains a non-differentiable blob-detection step after the network. Experiment item 3 ("FLOPs/params/latency/memory benchmark... extending benchmark_latency()") does not state whether the measured latency includes this post-hoc, non-learned step or only the neural forward pass. Given the network now has two very differently-shaped stages (64×64 trunk vs 256×256 head) plus an external peak-search step, conflating NN-only inference time with end-to-end system latency would materially change any deployment claim, and the document does not commit to measuring these separately.

6) Depth confound in the ablation plan. SSC-Net's combined trunk+head totals 22 processing blocks (6 grouped-FNO + 16 GU) versus 8 blocks in either single-mechanism screening baseline being cited as "already-validated" priors. The proposed ablation ladder removes whole components (pruning, loss term, head, grouping) but never isolates total block-count/depth as a controlled variable against the original 8-block runs, so any observed gain from the fused model over the two solo screening runs is confounded with simply having ~2.75× more processing stages than either prior baseline — undermining the "reuses already-validated components, doesn't add raw capacity" framing that is central to the candidate's own stated advantage.

None of these individually is necessarily disqualifying on its own — the candidate is unusually self-aware about several of its own risks (FLOP/param decoupling, untested grouped-spectral-conv, non-differentiable peak search) — but taken together (an unvalidated capacity-vs-mechanism assumption load-bearing for the head design, a pruning-ratio transfer between structurally dissimilar architectures, a mismatched baseline comparison underlying the headline efficiency claim, and a depth confound left uncontrolled in the validation plan) they constitute a fatal flaw in the sense that the candidate should not proceed to full implementation unmodified: at minimum it needs (a) a cheap GU-depth-sweep diagnostic run before committing to 16 blocks, (b) a pruned-FNO-only control at the same pruning ratio to make the "lightweight" comparison fair, and (c) an explicit depth-matched control in the ablation ladder, all added or run before the candidate can be considered validated rather than merely internally consistent on paper.

### Critic 2 (fatal_flaw_found=True)

ADVERSARIAL REVIEW — SSC-Net (Candidate C). Several flaws are serious enough that the spec should not proceed unmodified; the strongest ones are the parameter-accounting error, the unjustified pruning-ratio transplant, and the missing FFT cost in the FLOP estimate, each of which undermines a headline numeric claim ("~162K params," "48% of FNO baseline," "10.24 GFLOPs").

1) PARAMETER ARITHMETIC IS INTERNALLY INCONSISTENT (double-counting risk). The anchor explicitly states the 41,790 params/block figure for FNO "includes its share of stem+head" — i.e., the whole 334,321-param network (stem + 8 blocks + decode head) was divided by 8 to get a per-block average that already amortizes the *one-time, non-repeating* stem and head cost into every block. Step 2 then multiplies this stem/head-inclusive average by 6 (133,728), which incorrectly scales a fixed one-time cost as if it repeated per block. Step 4 then adds a *separate* ~8,000-param decode stack and a fusion gate on top, with no accounting for whether the stem is present at all in the final total. The result is a total that is simultaneously (a) missing an explicit stem line item and (b) probably inflating the trunk figure by folding a fixed cost into a per-block number that gets multiplied by block count. This isn't a minor rounding issue (the arithmetic ops themselves are computed correctly, e.g. 206,288 vs. stated 206,290 is trivial) — it's a structural bookkeeping error that could shift the "≈162K" headline number in either direction by a non-trivial amount. The spec should not be trusted on "lightweight" until params are recomputed with stem, trunk, head as three cleanly separated, non-overlapping components.

2) THE 67% PRUNING RETENTION RATIO IS TRANSPLANTED WITHOUT JUSTIFICATION. The spec applies the teacher's r=8 magnitude-pruning ratio (314,513/469,393 ≈ 0.67) to a structurally different network — an already-compressed, grouped-convolution trunk — assuming the same fraction of width survives pruning. This is unfair/unfounded: grouped convolutions already remove a large share of the redundancy that magnitude pruning exploits, so a pre-compressed trunk should be expected to tolerate *less* pruning than the dense teacher, not the same ratio. Reusing this number without re-deriving it on the actual grouped-FNO trunk is an optimistic, uncontrolled extrapolation, and it silently assumes pruning costs zero accuracy (using the teacher's ΔPd −0.0147 by analogy) despite the spec itself (Failure Mode 3) admitting the fusion gate will see a distribution-shifted input post-pruning with no retraining step specified anywhere in the pipeline. The "~162K, comparable degradation" claim is therefore not credible as stated.

3) THE FLOP ESTIMATE OMITS FNO'S OWN DOMINANT COST (FFT/IFFT). The FLOP formula used, `2 × params × H × W`, is a conv/pointwise-op approximation. It does not account for the FFT and inverse FFT that FNO's spectral-convolution mechanism requires at every trunk block — this is FNO's namesake operation and typically a first-order compute cost, not a rounding error, especially since the spec's own motivation is that Y is "already frequency-domain" and doing spectral mixing "naturally." A FLOP estimate for an FNO-based trunk that omits FFT cost is not "the first FLOPs number for this problem," it is an undercount, and the candidate should not claim to be "closing Gap 1" with this figure — indeed the spec's own Experiments-Needed section concedes the number is an estimate needing real instrumentation, directly contradicting the earlier claim that it "closes Gap 1" as if it were settled.

4) LATENCY IS CONFLATED WITH FLOPs/PARAMS, AND THE CONFLATION CUTS AGAINST THE CANDIDATE, NOT JUST FOR IT. The spec is commendably explicit that FLOPs and params decouple (GU head ≈37% of params but ≈77% of FLOPs). But it doesn't go far enough: grouped convolutions frequently underperform their FLOP/param savings on real GPU hardware due to poor memory-access patterns and kernel-launch overhead at low per-group channel counts — so the "lightweight" story could fail a third time (params low, FLOPs high, wall-clock latency worse than either predicts) once actually benchmarked. No latency number is reported; only a promise to measure it later. Until measured, "lightweight" is unsupported on any of the three axes that matter for deployment.

5) THE MECHANISM CLAIM FOR THE JACOBIAN-AWARE LOSS DOESN'T MATCH THE DIAGNOSED ROOT CAUSE. The spec frames `w=1/|sinψ|` reweighting as a "direct countermeasure" to the pixel-quantization/Jacobian-singularity ceiling. But the diagnosed root cause is a *representational* one — insufficient pixel resolution near end-fire on a fixed 256×256 grid — not a training-dynamics one. Upweighting the loss near end-fire cannot manufacture additional bins/resolution where the output grid geometrically has none; at best it makes the network try harder to be precise within an already information-limited representation, and at worst (as the spec's own Failure Mode 4 admits) it destabilizes training via large gradients from a small fraction of end-fire samples. Calling this a fix "directly informed by" and "a direct countermeasure to" the singularity overstates what a loss reweighting term can mechanically deliver against a fundamentally geometric/representational problem — the coarse-to-fine architecture change (if anything) is the real lever on this axis, and the loss term is better described as a secondary, unvalidated mitigation.

6) NOVELTY CHECK IS SCOPED TOO NARROWLY, WEAKENING THE "PARTIALLY NEW" VERDICT ITSELF. The self-classification searches only the project's own reports and a 14-paper DOA-specific corpus. But "coarse global map → refine to full resolution via a second-stage head" is one of the most standard patterns in the broader 2D heatmap/keypoint-estimation literature (cascaded/coarse-to-fine pose estimation is decades old and extremely well known outside this narrow DOA corpus). The spec's borrowed-components table and novelty argument never check against this much larger body of prior art, only against the in-project and DOA-specific search set — so even the modest "PARTIALLY NEW, domain-first" claim may be generous; the core backbone pattern could be an even more directly "known combination" than R7's own citations suggest, once compared against the correct (larger) reference class. This is exactly the kind of comparison-fairness gap the review is meant to catch: the self-assessment grades itself against a favorably narrow set of comparators.

7) IS THE GAIN REALLY FROM THE ARCHITECTURE, OR FROM MORE CAPACITY / DIFFERENT TRAINING RECIPE? The candidate simultaneously changes (a) architecture (grouped FNO + repositioned/deepened GU head + fusion gate), (b) the loss function (added Jacobian weighting, sparsity term, optional distillation), and (c) a post-hoc compression step — three confounded axes at once. The ablation ladder proposed (item 1 in Experiments Needed) is the right instrument to disentangle this, but as specified the candidate has *not yet run it*, so any claimed improvement over plain FNO or plain GridlessUnfold at this stage would be unattributable: it could come from the distillation term (extra teacher supervision — literally "more information," not "better architecture"), from simply training longer/deeper (16 vs 8 GU blocks is a depth/capacity increase, not purely a re-purposing), or from the architecture combination itself. The spec is honest that ablations are needed, but the review brief specifically asks to flag this, and it is a real, currently-unresolved confound.

8) "GENUINELY JOINT" AoA/AoD IS ASSERTED BUT ONLY WEAKLY ENFORCED. The joint property reduces to reading x/y off one peak in a shared 2D map trained with pixelwise (per-pixel) MSE/focal loss. Per-pixel losses do not explicitly enforce any AoA–AoD statistical coupling beyond spatial co-location in the same map — the same outcome could largely be achieved by two heads whose outputs are concatenated into one coordinate pair before the loss is applied. The claim that this design is meaningfully more "joint" than a dual-head design (beyond avoiding a literal parameter split) is asserted rather than demonstrated, and no experiment in the validation list directly tests joint-coupling strength (e.g., correlated-error analysis between AoA and AoD residuals) versus a matched dual-head baseline.

9) FAILURE-MODE LIST IS ARGUABLE BUT NOT COMPLETE. It omits: (a) training-instability/convergence risk from stacking three loss terms with three new tunable weights (λ1–λ3) plus the Jacobian weight's own ε, a hyperparameter-search burden not costed anywhere; (b) the possibility that grouped spectral mixing changes FNO's *effective* frequency coverage in a way that interacts badly with the aliasing/end-fire geometry itself (a second-order effect beyond the "collapses global mixing" failure mode already listed); and (c) the compounding-error risk of stacking un-validated components (grouped-FNO, repositioned-GU, fusion gate, pruning, Jacobian loss) such that a negative ablation result cannot cleanly identify which of five simultaneous novel components is at fault without the full ladder being run — the spec does mention the ladder but doesn't flag the practical risk that five-way ablation is expensive and may not be fully executed before any paper claim is made.

Net verdict: the candidate is a defensible, well-cited engineering combination, but as written it (i) contains a real parameter-accounting inconsistency that puts its headline size claim in doubt, (ii) borrows a pruning-survival ratio from an unrelated architecture without justification, (iii) reports a FLOP estimate that omits FNO's core FFT cost while simultaneously claiming to "close" a literature gap with that same unmeasured number, and (iv) overstates what its loss-level fix can mechanically do against a representational root cause. These are fatal to the spec's current numeric/mechanistic claims (not necessarily to the underlying idea) and should be corrected — with the ablation ladder actually run — before this candidate is presented as validated.

## Original (pre-refinement) specification

# Candidate C — Architecture Specification

## SpectraSparse‑CoFi (SSC‑Net): Grouped‑FNO Trunk + Depth‑Extended GridlessUnfold Precision Head, Coarse‑to‑Fine Joint Heatmap Fusion

### Motivation (one paragraph)

Two of this project's own from-scratch screening results point in the same direction but were never combined: the FNO spectral-conv core block is the current best alternative to the dense/flatten bottleneck the literature itself flags as its worst component (Pd 0.6235, 88% of teacher's Pd, at 334,321 params — 71% of teacher) `[EVIDENCE: PROJECT RESULTS.md]`, and the GridlessUnfold complex soft-threshold layer directly targets the diagnosed Jacobian-singularity/pixel-quantization root cause of the project's own Pd≈0.972 ceiling, but under-performed (Pd 0.2216, 30,201 params) only because it was screened at a shallow 8 blocks, not because the mechanism failed `[EVIDENCE: PROJECT]`. SSC-Net fuses these two already-validated components into a single differentiable coarse-to-fine pipeline — a coarse grid (deliberately coarser than 256×256, reducing the Jacobian-amplification effect per Agent-R1's A6 rationale) is produced by a grouped-convolution-compressed FNO trunk, then refined by a depth-extended GridlessUnfold head operating on the same joint 2D representation (not split into separate AoA/AoD heads, preserving the "genuinely joint" property the literature synthesis identifies as central to this task family) — and the whole trunk is then structurally pruned post-hoc, directly answering the literature-and-project gap that architecture search and compression have never been combined (Gap 7) while attacking Gap 5 (pixel-quantization ceiling) and Gap 2 (dense bottleneck) mechanistically rather than by adding raw capacity.

---

### Borrowed components (cited)

| Component | Source | Role in SSC-Net |
|---|---|---|
| FNO / spectral-convolution block | Project's own architecture-screening result `[EVIDENCE: PROJECT RESULTS.md]` — no direct literature precedent in the reviewed set | Early/mid trunk, coarse global-receptive-field feature extraction |
| GridlessUnfold complex soft-threshold layer | Project-internal implementation, inspired by PIA-Net's physics-informed deep-unfolding concept `[EVIDENCE: PROJECT]` | Precision refinement head, extended from 8→16 blocks |
| ResNet-shell conv stem / residual pattern | Lloria et al., TVT 2026 / PIMRC 2024 `[EVIDENCE: PAPER]` | Input stem, output heatmap decode conv stack |
| Grouped convolution (ResNeXt-style, G=4) | Generic efficient-CNN pattern; flagged by Agent-R2 (item 3) as "best risk-adjusted first efficiency experiment," untried in this project | Channel-mixing compression inside FNO trunk blocks |
| Extreme-parameter-efficiency "learned surrogate → lighter resolver" pattern | SubspaceNet, Shmuel et al. 2025 TVT (41,761 params vs >21M black-box CNN, 1D DOA) `[EVIDENCE: PAPER]` | Design-philosophy justification for keeping the trunk small and letting the precision head do proportionally more work, applied here to a 2D joint setting for the first time |
| Structured (magnitude / SNR-aware) channel pruning | Project's own compression track, r=8 magnitude result (314,513 params, ΔPd −0.0147 vs teacher) `[EVIDENCE: PROJECT baseline_models/README.md]` | Post-hoc compression pass applied to the trunk only |
| Coarse-to-fine grid rationale (deliberately coarser first stage to reduce Jacobian amplification) | Protocol §11-listed candidate pattern (Agent-R1's A6), motivated directly by the project's own ceiling test `[EVIDENCE: PROJECT]` | Output-representation design, independent of backbone choice per R1's cross-cutting note |

---

### The novel component(s), stated precisely

1. **Grouped-channel-mixing FNO blocks (G=4).** No prior work (in this project or the reviewed literature) applies grouped convolution to FNO's spectral channel-mixing weight tensor. This is a genuinely untried structural modification, not just a parameter-count knob applied elsewhere.
2. **Depth-extended, resolution-decoupled GridlessUnfold head.** GridlessUnfold is deepened from 8→16 blocks and moved from "whole-network core block" (its original screening role) to a dedicated post-trunk *precision head* that only the coarse heatmap passes through — a re-purposing of an existing component into a new architectural position, directly informed by the project's own diagnosis that its 8-block screening result was capacity-limited, not mechanism-limited.
3. **Single-joint-heatmap coarse-to-fine fusion (not split AoA/AoD heads).** The trunk emits a *coarse* (e.g. 64×64) joint 2D heatmap; the GridlessUnfold head refines that same joint map to full resolution. AoA and AoD are still read off one peak location at the end, preserving the joint-estimation property both R1 (A5) and R2 (item 9) flag as at-risk in dual-head designs — this is a deliberate design choice to get coarse-to-fine's Jacobian benefit without dual-head's joint-coupling risk.
4. **Jacobian-aware loss reweighting.** The fine-stage loss is explicitly reweighted by ∝1/|sin ψ| (and the AoD analogue) near end-fire angles, a direct, loss-level countermeasure to the exact singularity term the ceiling test derived (`d(π cos ψ)/dψ = −π sin ψ`) — this is a loss-function-level novel component, distinct from the architecture-level fixes and not discussed anywhere in the literature synthesis or R1–R8 reports as far as searched.
5. **Post-hoc SNR-aware structured pruning applied only to the trunk, after fusion training.** This is the first experiment in the project (and, per Gap 7, in the literature) to combine a from-scratch-screened alternative backbone with a compression pass, rather than treating architecture search and compression as independent tracks.

---

### Data flow

```
Input Y (raw antenna-domain, complex, nt=nr=16)
        │
        ▼
  Conv stem (ResNet-style, shared) ── borrowed from Lloria TVT2026
        │
        ▼
  Grouped-FNO trunk (6 blocks, G=4 channel mixing, spectral-domain
  early/mid layers — natural fit since Y is already frequency-domain)
        │
        ▼
  Coarse joint 2D heatmap decode (e.g. 64×64), deep-supervised
  with L_coarse  ──── this IS the shared representation; no AoA/AoD
  split occurs here — both angles still live in one map
        │
        ▼
  GridlessUnfold precision head (16 blocks, depth-extended from the
  project's 8-block screening run), phase-preserving complex
  soft-threshold operating on the upsampled coarse map, producing
  the full-resolution (256×256) joint heatmap
        │
        ▼
  Fusion/gating 1×1 conv (merges trunk's coarse-stage skip feature
  with GU head's fine-stage feature, small, shared — not per-angle)
        │
        ▼
  Shared output decode conv stack → single joint 2D heatmap (256×256)
        │
        ▼
  Non-differentiable blob detection (unchanged from Lloria/project
  pipeline, reused as-is) → peak location
        │
        ├──► AoA output (ψ, read from peak x-coordinate)
        └──► AoD output (φ, read from peak y-coordinate)
```

**AoA branch / AoD branch:** there is no separate AoA/AoD branch until the very last, non-learned peak-read-off step — both angles are extracted from the *same* peak in the *same* joint map at both the coarse and fine stages. This is a deliberate rejection of A5/R2-item-9's dual-head pattern, chosen specifically to avoid its flagged joint-coupling-loss risk, at the cost of not architecturally removing the discrete peak-search step the way A7's DETR-style head would (that trade-off is stated explicitly in Expected Weaknesses below).

**Shared layers:** stem, trunk, and the final decode stack are all shared between "AoA" and "AoD" — there is no point in the network where the two angles are computed independently.

**Fusion mechanism:** a small 1×1 gated conv that combines (a) the trunk's own coarse-stage feature map (skip connection) with (b) the GridlessUnfold head's refined feature map, before the final joint-heatmap decode — analogous to a lightweight U-Net-style skip (per A2's rationale) but placed at exactly one resolution level, not mirrored across the whole depth, to keep the added parameter cost minimal.

---

### Loss function

```
L = L_coarse(H_64, GT_64)                                  # deep supervision on trunk output
  + λ1 · L_fine(H_256, GT_256; w(ψ,φ))                      # main joint-heatmap loss, Jacobian-weighted
  + λ2 · L_sparsity(GU soft-threshold activations)          # encourages gridless/sparse support, native to GU
  + λ3 · L_distill(H_256, H_256^teacher)                    # OPTIONAL, teacher checkpoint already trained
```

- `L_coarse`, `L_fine`: pixel-wise weighted MSE (or focal-style) against Gaussian-rendered ground-truth peaks, matching Lloria's Eq. 9–11 heatmap convention so the eval pipeline (`evaluate_on_bank`) stays reusable unchanged.
- `w(ψ,φ) = 1 / max(|sin ψ|, ε)` (and the AoD analogue) — the Jacobian-aware reweighting term (novel component 4 above); `ε` prevents blow-up exactly at end-fire.
- `L_sparsity`: L1 penalty on GU's soft-threshold magnitudes, already part of GridlessUnfold's original design intent.
- `L_distill` (optional ablation arm): MSE between SSC-Net's fine heatmap and the frozen 64-block teacher's heatmap — zero additional data-generation cost since the teacher checkpoint already exists, and this is Agent-R2's #1-ranked untried technique (knowledge distillation), folded in here as an ablatable auxiliary term rather than a separate experiment, directly extending Gap 7's "combine tracks" argument to a third track (distillation).
- λ1, λ2, λ3 are tunable; λ3=0 recovers a pure from-scratch training run for apples-to-apples comparison against the existing FNO/GridlessUnfold registry entries.

---

### Parameter count estimate (with arithmetic)

All figures are **estimates**, built from ratios anchored to the project's own measured totals, with assumptions stated explicitly.

**Anchors (measured, not estimated):**
- FNO: 8 blocks → 334,321 params total ⇒ **41,790 params/block** (average, includes its share of stem+head)
- GridlessUnfold: 8 blocks → 30,201 params total ⇒ **3,775 params/block** (average)

**Step 1 — Grouped-FNO block cost.** Grouped convolution at G=4 gives a 4–8× reduction on the *conv/channel-mixing* portion of a layer (Agent-R2 item 3, general literature). Assume conservatively that ~70% of an FNO block's params sit in the channel-mixing spectral weight tensor and 30% in group-invariant components (norms, biases, pointwise projections):
- Mixing portion: 0.70 × 41,790 = 29,253 → ÷3 (conservative net reduction, below the literature's 4–8× ceiling to account for the group-invariant remainder) = 9,751
- Fixed portion: 0.30 × 41,790 = 12,537
- **Grouped-FNO block ≈ 22,288 params**

**Step 2 — Trunk (6 blocks).** 6 × 22,288 = **133,728 params**

**Step 3 — GridlessUnfold head, depth-extended 8→16 blocks (linear scaling assumption, since GU blocks are structurally repeated).** 16 × 3,775 = **60,400 params**

**Step 4 — Fusion gate + shared decode stack.** Estimated as a small 1×1 conv (≈64→64 channels: 64×64+64 ≈ 4,160) plus an output decode stack comparable in scale to a typical 2-layer upsample/projection head (≈8,000 params, order-of-magnitude estimate, no precise project analogue to anchor this against — flagged as the least-certain line item).

**Pre-pruning total:** 133,728 + 60,400 + 4,160 + 8,000 = **≈ 206,290 params**

**Step 5 — Post-hoc SNR-aware structured pruning, trunk only.** Applying the project's own empirical magnitude-pruning retention ratio (r=8 result: 67% of original width retained) `[EVIDENCE: PROJECT]` to the trunk portion only (GU head and decode stack left unpruned, since they are already near-minimal):
- Trunk post-pruning: 0.67 × 133,728 = 89,598
- **Total post-pruning ≈ 89,598 + 60,400 + 4,160 + 8,000 = 162,158 params**

**Result: ≈ 162K params, roughly 35% of the teacher's 469,393 and roughly 48% of the FNO screening baseline's 334,321** — smaller than either prior project result, at the cost of the added GridlessUnfold-head and fusion-layer overhead the original FNO run didn't carry.

---

### FLOP estimate (with arithmetic)

Using the standard conv FLOP identity `FLOPs = 2 × params × H_out × W_out` (Agent-R8's recommended formula), applied per stage at its own operating resolution — this is the step that most concretely differentiates SSC-Net from a simple param-count story.

- **Trunk** (coarse stage, operates at ~64×64): 2 × 89,598 (post-pruning) × (64×64) = 2 × 89,598 × 4,096 ≈ **734.1M FLOPs**
- **GridlessUnfold head** (fine stage, operates at full **256×256**): 2 × 60,400 × (256×256) = 2 × 60,400 × 65,536 ≈ **7.92B FLOPs**
- **Fusion gate + decode stack** (also full-res, 256×256): 2 × 12,160 × 65,536 ≈ **1.59B FLOPs**

**Total ≈ 734.1M + 7.92B + 1.59B ≈ 10.24 GFLOPs per forward pass.**

**Explicit caveat (required, not optional):** this number is *not comparable* to Naoumi's 4.321 MMACs figure — different task framing (bistatic ISAC vs. this project's system), different resolution, different metric convention (MACs vs. FLOPs), rated LOW/NOT COMPARABLE per the literature synthesis's own comparability table. No paper in the reviewed set reports a FLOPs figure for the Lloria/joint-2D task family at all, so this would be the first such number for this specific problem (directly closing Gap 1), not a number that currently has any apples-to-apples literature anchor.

**The arithmetic itself surfaces a real design tension worth flagging up front:** the GridlessUnfold head holds only ~37% of total (post-pruning) params but consumes ~77% of total FLOPs, because it runs at 16× the trunk's spatial resolution. Param-count efficiency and compute efficiency decouple sharply in this design — a genuine, arithmetic-derived risk, not a hand-waved one.

---

### Expected advantages

- Reuses two already-independently-validated project components rather than starting from zero, giving this candidate the most concrete prior-evidence trail of any single-mechanism design (mirroring A9's stated strength).
- Attacks Gap 2 (dense bottleneck), Gap 5 (Jacobian-singularity ceiling), Gap 6 (SubspaceNet-pattern 2D extension), and Gap 7 (architecture search × compression) simultaneously with one coherent pipeline, rather than one gap at a time.
- The Jacobian-aware loss term is cheap to add/remove and gives a clean single-variable ablation independent of the architectural changes.
- Post-hoc pruning reuses existing, already-working infrastructure (same pruning code, same evaluator) — lowest-implementation-cost compression axis available per Agent-R2.
- Preserves the "genuinely joint" AoA/AoD estimation property that the literature synthesis identifies as central to why this task family is meaningfully different from independent 1D estimation (Contradiction #12's discussion of what "joint" actually requires).

### Expected weaknesses

- Inherits FNO's own **unexplained** high-SNR degradation (Pd drops ~0.80→0.70 from 10dB→25dB) `[EVIDENCE: PROJECT RESULTS.md]` — grouping the FNO trunk further reduces its capacity, which could worsen (not fix) this open, undiagnosed issue rather than resolve it, exactly the risk A9 itself flags.
- Still relies on non-differentiable blob detection at the very end — unlike A7's DETR-style set-prediction head, SSC-Net does **not** architecturally remove the peak-search step that Gap 5 ultimately identifies as the ceiling's true cause; it only makes the *heatmap feeding into* that step more precise. This is a genuine, stated limitation relative to the project's own most evidence-convergent unbuilt idea.
- FLOP/param decoupling (above): the "lightweight" params claim is compute-heavy in practice, weakening any headline "cheap to deploy" story unless the GU head is also run at reduced resolution — a change this spec does not make, and one that would reintroduce Gap 5's own resolution-vs-precision trade-off (Agent-R2 item 11's explicit warning against reducing resolution given this project's diagnosed failure mode).
- Grouped convolution inside FNO's spectral weight tensor is architecturally untested anywhere (in this project or the reviewed literature) — the 3× reduction assumed above is an extrapolation from general (non-spectral, non-FNO) conv literature, not a validated number for this specific operation.

---

### Concrete failure modes

1. **Grouping collapses FNO's global-mixing advantage.** If G=4 channel grouping breaks the exact global cross-channel spectral mixing that gives FNO its receptive-field advantage in the first place, the trunk could regress toward a purely local conv model — defeating the entire rationale for choosing FNO over a plain ResNet stem.
2. **Coarse-stage under-detection at high L.** At the coarse 64×64 resolution, closely-spaced sources (small angular separation, per Agent-R3's Δθ sweep) could collapse into the same coarse bin before the GU head ever sees them separately — an unrecoverable loss, exactly A6's flagged failure mode, and specifically damaging on the one axis (L-sweep, angular separation) every Lloria-family paper already reports on.
3. **Pruning interacting badly with the fusion gate.** Because pruning is applied only to the trunk post-training, the fusion gate (trained against the *unpruned* trunk's feature statistics) could receive a distribution-shifted input at inference, degrading fusion quality in a way neither the original FNO nor the original pruning experiments would have exposed (this is a genuinely new failure surface created by the combination itself, not inherited from either parent).
4. **Jacobian-reweighted loss overfitting to end-fire tail cases.** Upweighting `1/|sin ψ|` near end-fire could destabilize training (large loss gradients from a small fraction of samples) — a standard hard-example-reweighting risk not previously encountered in this project's loss designs.
5. **GU head's full-resolution FLOP cost making step-matched training comparisons unfair.** Per Protocol §16/PROJECT §7 item 2's explicit warning (raised originally about A7/DETR heads): if SSC-Net's forward pass is markedly more expensive per step than FNO/GridlessUnfold's original 8-block runs, a wall-clock-matched (rather than step-matched) comparison could make SSC-Net look worse than it structurally is, or vice versa — must be controlled for explicitly.

---

### Experiments needed to validate

1. **Component ablation ladder** (per Agent-R8 §5, reusing `evaluate_on_bank` unchanged wherever the output stays a heatmap): Full SSC-Net → −pruning → −Jacobian-weighted loss (uniform loss) → −GU head (trunk-only coarse output, no fine stage) → −grouped conv (dense FNO trunk instead). Each isolates exactly one of the five novel components against the same eval bank.
2. **Step-matched (not just wall-clock-matched) re-run against the existing FNO and GridlessUnfold registry entries**, at the same total gradient-step budget, to avoid the exact confound flagged in Failure Mode 5 and previously identified as a project-wide risk (§7 item 2 of the prior-work analysis).
3. **FLOPs/params/latency/memory benchmark** on real hardware, extending `benchmark_latency()` (already present in `proposed pin architecture.ipynb`) and adding the currently-absent FLOP-counting instrumentation (`thop`/`ptflops`, per Agent-R8 §0) — necessary to make the 10.24 GFLOPs estimate above an actual measurement, not an estimate.
4. **Tier-1 robustness battery** (SNR sweep, phase error δ/γ∈{0,1,2,5}°, L∈{1..6}) per Agent-R3, run on this candidate alongside Teacher/Student/FNO for direct comparability.
5. **End-fire density probe** (Agent-R3 §4/§7 item 3), specifically because this candidate's core novelty claim (Jacobian-aware loss + GU precision head) is falsifiable exactly at this condition — if SSC-Net doesn't outperform plain FNO+pruning near end-fire, the mechanism-level claim collapses to a capacity-only claim.
6. **Distillation-on/off ablation** (λ3=0 vs. λ3>0) using the existing teacher checkpoint, to determine whether the optional distillation term is pulling meaningful weight or is dead weight in the loss.
7. **Angular-separation sweep** (Δθ∈{30°…1°}) to directly test Failure Mode 2 (coarse-bin collapse for closely-spaced sources) — this requires confirming first whether `validation_data_generator` exposes angle-spacing control at all (Agent-R8 §6 item 3, unresolved as of the research inputs).

---

### Novelty argument — self-classification: **PARTIALLY NEW**

**Justification (not asserted, argued):**

- The core backbone pattern (FNO trunk + GridlessUnfold precision head) is **structurally identical to Agent-R1's A9**, which Agent-R7 did not independently classify in isolation, but whose closest analogues (E: SubspaceNet-pattern extended to 2D; F: architecture-search × compression combined) were explicitly rated **"KNOWN COMBINATION at the pattern level, PARTIALLY NEW at the task level"** and **"NOT NOVEL as a general technique; PARTIALLY NEW only as domain-first,"** respectively. SSC-Net inherits that verdict directly — it is a domain-first application of established patterns (model-based DL front-end→resolver; NAS+compression), not a conceptually new paradigm.
- The specific components that are genuinely untried anywhere in the reviewed corpus — grouped convolution inside an FNO spectral block, and depth-extending GridlessUnfold as a dedicated post-trunk head rather than a whole-network core block — are incremental engineering combinations of existing techniques applied to a task-specific diagnosed failure mode (Gap 5), which per Agent-R7's §1 criteria ("tie the architectural choice to a specific diagnosed root cause… + ablate the combination against components alone") is exactly the bar for a *defensible but modest* contribution, not a strong one.
- The Jacobian-aware loss reweighting term is the one piece with no identified precedent anywhere in the 14-paper corpus, the project's prior work, or the R1–R8 supplementary searches — but per Agent-R7's own explicit caution (§0, §4), **absence of evidence in a non-exhaustive search is weak evidence of absence**; this specific term was not itself subjected to a dedicated literature search (unlike Gap 5's DETR angle, which R7 did flag for a dedicated follow-up search). It should be treated as **UNCERTAIN, not confirmed novel**, pending exactly that kind of targeted check (e.g., "Jacobian-weighted loss DOA angle estimation," "hard-angle reweighting end-fire array processing").
- Critically, per Agent-R7's bottom line, the single most defensible novelty pillar available to *any* candidate built from these inputs is not architectural at all — it is the measurement/rigor bundle (params+FLOPs+latency+memory together, strict-vs-paper-style PD, multi-seed variance). SSC-Net is designed to produce all of these cleanly (per the Experiments Needed section), and that rigor contribution should be presented as a **co-equal, not secondary, pillar** of this candidate's paper-worthiness — it is close to zero-risk novelty, whereas every architectural element above is PARTIALLY NEW at best and contingent on ablations that have not yet been run.

**Overall: PARTIALLY NEW.** The strongest legitimate claim is "first systematic combination of FNO-trunk / GridlessUnfold-head / structured pruning for the joint-2D AoA/AoD task, evaluated with a measurement rigor bundle absent from the entire reviewed literature" — not "a new architecture family." This candidate should not be pitched as NEW without first running the ablation ladder (item 1 above) to show the combination — not just the union of already-known parts — earns its keep, per Agent-R7's explicit reviewer-question framing in §1.