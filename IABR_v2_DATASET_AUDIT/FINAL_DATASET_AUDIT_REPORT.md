# IABR-v2 Dataset-Generation Pipeline: Forensic Audit (Final Report)

Scope: how the data used by IABR-v2 is generated (inputs, labels, training stream, tuning and test banks). Model accuracy is out of scope.
Labels: FACT / DERIVATION / INFERENCE / ASSUMPTION / CONCERN / VERIFIED BUG. Status: 🟢 verified, 🟡 uncertain, 🔴 problem.

**Process note (honest disclosure).** 11 specialist reports (01-11) and 3 of 4 cross-examination panels (X1, X3, X4) completed. Panel X2 (reviewer vs math and validity), the separate senior-review agent and the completeness critic were stopped because of the token budget. This report is a synthesis written directly from the specialist reports and the X1/X3/X4 resolution tables, where later panels overrode earlier specialist numbers. Where a claim is not re-verified here it says "see report NN".

---

## 1. Executive summary

- The dataset is **synthetic and generated on the fly**. Training uses an endless stream (48,000 steps x 256 = 12.29M scenes, never stored, no epochs). Evaluation uses frozen banks: the authors' seed-42 `eval_bank.npz` (8,000 samples, L=3, SNR -10..25 dB), T2_d1/d2/d5 (8,000 each, seeds 9101/9102/9105), tuning banks (seeds 9901-9904) and 47 of 56 v1 robustness banks (500 each).
- The channel and angle sampler are **the base paper's (Lloria et al., TVT 2026) generator**: `dldoa_dataset_generation.py` is md5-identical to the authors' code. Regeneration is bit-exact: seed 42 equals `eval_bank.npz`, seed 7 equals the `dldoa_dataset` test split.
- **Project-original additions:** per-antenna gain error (uniform in dB, 3 dB max), the 32x32 wrap-around Gaussian heat-map labels, impairment-projection targets, and the 20% clean fraction and training ranges.
- 🟢 **No train/test leakage found** at sample or checkpoint level. Training uses `default_rng([seed, worker, os.urandom(4)])` and `train()` has no evaluation or checkpoint selection.
- 🟡 **Design-level exposure of the official bank:** the v2 design was motivated by looking at results on it. No effect is detectable on the unseen `dldoa_test` bank down to about 0.006-0.008 Pd (INFERENCE).
- 🔴 **Main concern (attribution and interpretation).** NOMP alone (no network) scores 0.7581 on the official bank against 0.7551 for IABR-v2. The clean-bank gain over the U-Net comes from the physics decoder, and the notebook does not cite that algorithm. The base paper (Lloria) is also not named in the notebook.
- No plagiarism evidence was found. Insufficient evidence to claim uncited borrowing beyond standard, well-known algorithms.

## 2. Full flow, step by step

1. Draw L (1..9 in training; 3 in the official bank), SNR (-15..24 dB in training; -10..25 step 5 in the official bank), clean flag (20% in training), phase max delta (0..8 deg) and gain max (0..3 dB).
2. Draw angles (psi, phi) uniformly on [0, pi]^2 by sequential rejection with Euclidean separation pi/6 in the angle plane (`generate_points`).
3. Draw alpha ~ CN(0, 1/L), sorted strongest first and paired with placement order.
4. Channel H = sqrt(NtNr) * sum_l alpha_l a_r(psi_l) a_t^H(phi_l), with a(theta)_k = exp(-j pi k cos theta)/sqrt(16).
5. Codebooks: unitary 16x16 DFT for W and F. With impairments D = g * exp(j eps) applied per antenna to both codebooks. Y = W^H H F.
6. Noise Z ~ CN, added after the combiner at the requested SNR.
7. Labels: a 32x32 wrap-around Gaussian (sigma = 1 cell, max over paths) at cells q = G*wrap(-pi cos psi)/2pi and p = G*wrap(pi cos phi)/2pi. Impairment targets have the mean removed (and the linear ramp for phase).
8. Input to the network: normalised Y (real, imag, log-magnitude).

## 3. Derivation (key results, DERIVATION)

- Y = fft2(H)/N when the codebooks are unitary DFTs (checked to about 4e-15 against `generate_channel_v2`).
- Impaired channel: H~ = conj(D_r) H D_t. This is rank-1 separable per path, with 16+16 degrees of freedom.
- Unidentifiable subspace is exactly 6-D: for phase, the mean and linear ramp on each side (2+2). A ramp is indistinguishable from an angle shift, and for gain the mean (2).
- Phase-error leakage relative to signal: 2 - 2(sin d / d)^2 gives -22.9 dB at d = 5 deg and -36.9 dB at d = 1 deg. So T2_d1 and T2_d2 are effectively clean banks.
- Gain SNR bonus: per side E[g^2] = sinh(x)/x, x = gamma ln10/10. Both sides: +0.08 / 0.31 / 0.68 / 1.20 dB at gamma = 1 / 2 / 3 / 4 dB.

## 4. Worked examples

Report 10 contains 12 worked examples: a seeded RNG trace, a 4x4 toy channel and full Y table, the impairment leak (relative change 0.1917), realised noise SNR, ramp-to-angle-shift slopes, the aliasing example (||a(5 deg) - a(175 deg)|| = 0.2098 for N = 16), and the sampler batch trace. Panel X4 independently re-ran all of them and **reproduced every number** (🟢). The only nit is a realised SNR of 12.04 dB rather than 12.05 dB.

## 5. Visuals

Report 11 and the `figures/` folder hold the 16 figures (pipeline diagram, occupancy, closeness, ramp floor, ground-truth power check and others). X4 re-derived the numbers in figs 09, 12, 13 and 16. Figs 06, 09 (rendering), 11 and 14 were not visually re-viewed (insufficient evidence on rendering). fig09's "random" baseline is not a 0 dB null (angle-based random gives 3.3-3.7 dB).

## 6. Parameter table (FACT, `notebook_src.py` CFG and `iabr2_sim.py`)

| Parameter | Training | Official bank | T2_d1/d2/d5 |
|---|---|---|---|
| Steps x batch | 48,000 x 256 | n/a | n/a |
| L | 1..9 | 3 | 3 |
| SNR (dB) | -15..24 | -10..25 step 5 | as spec |
| Phase max | U(0, 8 deg) | none | 1/2/5 deg |
| Gain max | U(0, 3 dB) | none | none |
| Clean fraction | 20% | n/a | n/a |
| Grid / sigma | 32 / 1 cell | n/a | n/a |
| Size | endless | 8,000 (seed 42) | 8,000 each (9101/9102/9105) |

Paper text vs code: the paper says SNR -10..25 and L = 1..10, while the code uses -15..24 and 1..9. The pi/6 separation appears only in the code, not in the paper's Table I (checked on the rendered page).

## 7. Distribution audit (after cross-examination)

- **Angles are uniform in psi, so cos psi follows an arcsine law.** About 32% of paths sit within one DFT cell of end-fire (cell 8 of 16), which is an occupancy of 3.6-3.9x uniform. Analytic values: 1 cell 32.2%, half cell 22.6%, within 5 deg of either end-fire 5.56% per path.
- **Beam-space merging (X1, X3, X4 all agree):** with the wrap respected, about 3-4% of L=3 scenes contain a path pair closer than one beam cell (3.15% in the official bank; the training mix is worse at 15.18%). Report 04's "0.018%" was wrong. It used a non-wrapped metric.
- **Heat-target merge on the 32-grid:** L=3 0.33-0.37%, L=9 5.4%, equal-weight mean about 1.9%.
- **Sampler ordering bias at high L:** at L=9, later-placed paths lie near end-fire more often (0.221 first vs 0.266 last, p about 2e-6). Negligible at L=3, so the official bank is unaffected. It shapes the training distribution only.
- **Ramp-shift floor.** A phase ramp is a physical ambiguity: at 5 deg, about 1.5% of components are displaced (non-alias), about 2.55% when the end-fire wrap is counted. All methods are equally affected. I rate this Medium for impaired banks.
- **Gain banks are slightly easier** (the SNR bonus in section 3).

## 8. Question bank (for supervisor and reviewer)

1. Why is the noise added after the combiner, and is the SNR per-entry? (Yes: total signal over total noise energy over 256 entries equals the per-entry SNR. The 24 dB gain applies to estimation variance, not to an integrated SNR metric.)
2. Why are the impairments beam-independent? (Modelling choice. Real phase shifters are state-dependent. ASSUMPTION, Low.)
3. Why is the phase redrawn per sample? (The authors' own code does this despite its "static" comment. Whether Meneses' Table 2 data does this is unresolved.)
4. Why 3 dB gain, uniform in dB? (Project assumption. Neither paper models gain error; Meneses lists it as future work.)
5. Why is L given as an oracle? (It is the base paper's own protocol, applied to all methods. It limits absolute claims.)
6. Why does NOMP alone match IABR-v2 on clean banks? (See section 13.)
7. Why do the T2_d1 and d2 numbers barely differ from clean? (Leakage of -37 dB and -31 dB.)
8. Was the frozen bank ever seen in design? (Design-level only; see section 10.)

## 9. Bug audit

- **No sign, conjugation or transposition bug** in Y = W^H H F, nor in the impairment targets (🟢, X1 row 10). The simulator matches the original generator.
- **Seed collisions (VERIFIED BUG in inherited v1 bank construction, Low):**
  - Seed 2020: `gain_g2_snr0` and `gain_g0.5_snr15` have identical angle scenes (500/500) and a shared unit-noise draw. The two cells are correlated, not independent.
  - Seeds 1000/1015: `phase_d0_snr{0,15}` share 4 and 3 scenes with `nuis_*`. The `nuis_*` banks are not used in v2, so this is immaterial to the v2 tables.
- **Draw order differs from the authors' generator** (points, then alpha; the authors draw alpha, then points, then noise). v2 banks are seed-exact within v2 but not stream-identical to the authors'. Statistical equivalence is an ASSUMPTION.
- **Non-wrap near-coincident pairs** exist in the sampler (6e-6 per pair). Seeing 0 in the official bank is the expected outcome, not structural.
- **Corrections to specialist claims:** 04's 0.018% and 1.3% figures are not reproducible; 05's "30% within 5 deg of end-fire" is wrong (5.56% per path, about 29% per 6-angle sample); 05's "end-fire is nearly free for the 1-deg criterion" is **inverted** (the criterion is tighter in beam space there).

## 10. Validity audit

- 🟢 **No sample-level leakage.** The training-stream seed cannot collide with a bank seed except with probability 2^-32 per worker start. 0 first-path key hits across 10,240 training scenes against 64 banks.
- 🟢 **No checkpoint or eval contact inside `train()`.** No stale caches: files that predate `final.pt` are only the fixed-weight U-Net predictions.
- 🟡 **Design-level peeking (Low-Medium).** The NOMP diagnostic used a 300-per-SNR subset of the official bank. On the unseen `dldoa_test` bank, IABR-v2 scores 0.7531 against 0.7551 official, and the gap to U-Net is 0.0177 against 0.0185. Overfitting to the official bank above about 0.006-0.008 is excluded.
- 🟡 **Max cosine 0.915 between official and T2_d1** is not a duplicate. Only the strongest path coincides (within 0.5-1.1 deg). Independent fresh draws reach 0.80-0.89 (X3 row 2).
- 🟡 **Oracle L and the metric.** The U-Net's paper-style Pd drops samples with fewer than L blobs (0.7366 paper vs 0.6940 strict), which flatters the U-Net. The metric choice is therefore conservative for v2.
- 🟡 **"DFT-SIC" is not the paper's DFT-CEA** (N_DFT 512 vs 1024, different iteration). Do not equate them.

## 11. Provenance audit

- FACT: the channel and sampler are the base paper's public code (md5-identical generator copies). The authors' official weights are evaluated on the authors' own seed-42 generator.
- FACT: the notebook credits Meneses-Albalá and a repository URL. It has 0 matches for "Lloria" and "TVT", so "the authors' official U-Net weights" reads as Meneses'.
- FACT: the notebook has 0 citations for NOMP (Mamandipoor, Ramasamy, Madhow), focal loss (CornerNet/CenterNet), SE blocks (Hu 2018), ResNet, or the TSDCE codebook. NOMP is named at `:527` and focal loss at `:686-691` without references. These are standard, public techniques, so this is an attribution gap, **not plagiarism**. Whether the thesis text cites them: insufficient evidence.
- FACT: the gain-error model is the project's own extension. The topic is literature-adjacent (both papers cite Bakr and Johnson on amplitude/phase errors) but neither paper models it.
- FACT: the log-magnitude input channel and the IABC estimator are project designs.
- Insufficient evidence for any wrongful copying.

## 12. Reproducibility audit

- 🟢 Official bank regeneration: bit-exact (data, features, meta; sha1 53a7172f4dc3). Seed 7 equals the zip test split; seed 42 does not.
- 🟢 Seven v2 banks reproduce bit-exact (arrays and file md5).
- 🟢 The 47 v1 banks used: 8+8+4+16+7+4 = 47; the 9 `nuis_*` and 3 `v1_clean_*` banks are unused.
- 🔴 **The training stream is not reproducible** (`os.urandom`). The trained weights cannot be regenerated from the seed alone. Caches are keyed by name only (`make_bank` returns if the file exists), which is an integrity risk, though no staleness was observed.
- 🟡 Whether the actual run resumed from a checkpoint: the log shows only step 48000/48000, 108 min, 191,702 params. Insufficient evidence.
- 🟡 `dldoa_dataset/test` files are not on this machine, but the generator and seed 7 were identified by regeneration.

## 13. Most serious concerns (ranked)

1. **Attribution / interpretation (Medium).** The clean-bank improvement over the U-Net comes from the NOMP decoder, not the network. NOMP alone is 0.7581 against IABR-v2 0.7551 (official) and 0.7564 against 0.7531 (`dldoa_test`). The network's benefit, if any, is on impaired banks, or in cost. This was not tested in this audit. The thesis should not present the network as the source of the clean-bank gain.
2. **Uncited algorithms and the unnamed base paper (Low-Medium).** NOMP, focal loss, SE and ResNet are uncited in the notebook, and Lloria is not named.
3. **Beam-space merging (Medium).** About 3-4% of L=3 official scenes (15% of training scenes) contain two paths within one beam cell. Any claim about resolution needs to state this.
4. **Ramp ambiguity floor (Medium).** Angle labels keep the ramp-induced shift that the impairment targets project away, capping the 1-deg detection rate on impaired banks for all methods. Quote one definition (about 2.55% per axis at 5 deg, wrap-consistent).
5. **Design-level exposure of the official bank (Low-Medium)**, bounded at about 0.006-0.008 Pd.
6. **Non-reproducible training stream (Low)** and name-keyed caches.
7. **Seed-collision pairs in inherited v1 banks (Low)**: correlated cells, immaterial to the v2 tables.

## 14. Reviewer verdict

**Accept the results as genuine, with corrections to how they are framed.** The dataset generation is faithful to the base paper's public generator, is bit-exact reproducible for all banks, and shows no train/test leakage. The extensions (gain error, heat-map labels, projection targets) are documented as project-original. A hostile reviewer would push on: (a) the NOMP-alone comparison, since the network is not the source of the clean-bank gain; (b) oracle L and the Pd metric; (c) the uncited standard methods; (d) the beam-merging and ramp-ambiguity floors, which make some ground truth ambiguous; (e) the non-reproducible training stream.

Not established in this audit (insufficient evidence): whether the authors' U-Net saw the seed-42 stream; whether Meneses' Table 2 impairments are drawn per sample; whether the thesis text cites NOMP, focal loss or Lloria; whether the dedicated X2 panel would have overturned any specialist claim.

## Appendix: files

Specialist reports: 01 pipeline, 02 math, 03 physics, 04 bug hunter, 05 hostile reviewer, 06 validity, 07 leakage, 08 reproducibility, 09 provenance, 10 worked examples, 11 visualization. Panels: X1, X3, X4. Figures in `figures/`. Scratch scripts in `scratch/` (not pushed).
