# X4 Cross-examination: reproducibility (08) vs assumptions, worked examples (10), visualization (11), math bugs (04)

Scope: dataset generation only (inputs, labels, training data, tuning/test banks) of the IABR-Net v2 pipeline. Model accuracy is out of scope. Base paper: Lloria et al. (TVT 2026). Comparison paper: Meneses-Albala et al. (J. Supercomputing 2026).
Nothing in the project was modified. All numbers below were re-computed by scratch scripts in `scratch/x4_*.py` (index at the end).

Labels: FACT (read in code or reproduced exactly), DERIVATION (follows from stated math), INFERENCE, ASSUMPTION, CONCERN, VERIFIED BUG. "Insufficient evidence" is stated where it applies.

## 0. What the authors did (short reference, from `iabr2_sim.py`)

- Channel: $H=\sqrt{N_tN_r}\sum_l \alpha_l\, a_r(\psi_l)a_t^H(\phi_l)$, $a(\theta)_k=e^{-j\pi k\cos\theta}/\sqrt N$, 16x16 ULAs (`iabr2_sim.py:45-47`).
- Measurement: $Y=W^H H F+Z$ with unitary DFT codebooks, so $Y=\mathrm{fft2}(H)/N$; impairment $D=g\,e^{j\varepsilon}$ applied to both codebooks (`:48-53`); white noise added after the combiner (`:54-57`).
- Labels: 32x32 wrap-around Gaussian, $\sigma=1$, max over paths, cell $q=G\,\mathrm{wrap}_{2\pi}(-\pi\cos\psi)/2\pi$, $p=G\,\mathrm{wrap}_{2\pi}(\pi\cos\phi)/2\pi$ (`:71-104`). Impairment targets have the mean and (for phase) the linear ramp projected out (`:76-89`).
- Angles: `DG.generate_points` sequential rejection, Euclidean separation $\pi/6$ in the $(\phi,\psi)$ plane; $\alpha\sim\mathcal{CN}(0,1/L)$ sorted strongest first and paired with placement order (`:28-38`).
- Training config (`notebook_src.py:82-86`): 48000 steps, batch 256, $L\in[1,9]$, SNR $\in[-15,24]$ dB, phase max 8 deg, gain max 3 dB, clean fraction 0.2, grid 32, $\sigma=1$. Training stream is seeded with `os.urandom` (`iabr2_sim.py:127`), so training data is not exactly reproducible.

## 1. Resolution table

| # | Claim | Challenger | Defence | Resolution | Evidence |
|---|---|---|---|---|---|
| A1 | "Only 0.018% of scenes have a pair closer than one DFT beamwidth" (04) vs 3.15-3.3% (01/03/05/06/11) | 11/03/05 (wrapped metric) | 04: its metric is non-wrapped $\lvert u_i-u_j\rvert$ | **REVISED.** 04's number is right for its metric, but the metric misses the $\pm\pi$ (end-fire) alias. Wrapped square-<1-cell rate: eval_bank 3.86%, T2_d5 4.00%, fresh L=3 4.01% (non-wrapped 0.013/0.037/0.013%). Euclidean-disc (11): 3.15% on both banks (`x4_09`), consistent because a disc is smaller than a square. 04's "negligible" conclusion is wrong; beam-space merging is real, ~3-4% of L=3 scenes. | `x4_01_closeness.py`, `x4_09_fig16.py` (FACT) |
| B1 | Ramp-shift floor at 5 deg: 1.44% (02), ~2.0% (06), 2.07% (11), 2.71% (03), 1.4% (10) | mutual | each report handles $\lvert\cos'\rvert>1$ differently | **REVISED to one definition.** Same Monte Carlo, different alias handling (2M paths): 5 deg: non-alias only 1.46%, alias fraction 1.09%, clip variant 2.03%, true-wrap variant 2.55%. 02 and 10 = non-alias; 06 and 11 = clip; 03's 2.71% coincides with the non-alias value at 8 deg. No real conflict. Physically correct (wrap) value ~2.55% per axis at 5 deg; ~14% of L=3 scenes have at least one of 6 components displaced >1 deg (independent-sampler approximation). | `x4_02_ramp.py` (DERIVATION) |
| B2 | 11 rates the ramp floor "CONCERN Low" | 11 | 11: the ramp is a physical ambiguity | **REVISED to Medium** for impaired banks: the ramp is removed from impairment targets but not from angle labels, so it puts a ceiling on the 1-deg detection rate. | `iabr2_sim.py:76-89`, `x4_02` (CONCERN) |
| C1 | Heat-cell collision by L (1.3, 0.6, 2.9, 2.3, 7.4, 4.2, 2.2% for L=3..9) in 04 vs 5.1%/4.7% at L=9, ~1.7-1.8% overall in 02/06 | 02/06 | 04: 150 scenes per L | **04 REVISED (small-sample noise); 02 and 06 UPHELD.** N=20000 per L, 32-grid merged-scene rate: L=2 0.17, 3 0.37, 4 0.76, 5 1.25, 6 2.02, 7 2.94, 8 3.92, 9 5.42%; equal-weight mean 1.87%. On the 16-grid L=3 is 1.31% and L=9 17.5%; 04's L=3 = 1.3% matches the 16-grid, not the 32-grid. 06's L=3 0.6% is slightly high (0.37% correct). | `x4_03_heat_bias.py` (FACT) |
| D1 | Sampler ordering bias: 04 retracts (KS p 0.12-0.92 at L=3); 03 finds bias at L=9 (0.221 -> 0.260) | 03 | 04's retraction | **03 UPHELD; 04 REVISED to L=3 only.** P(within 20 deg of end-fire), first vs last placed point: L=3 0.227 vs 0.232 (KS p=0.40); L=4 p=0.021; L=6 p=0.006; L=9 0.221 vs 0.266 (p=1.7e-6). Uniform baseline 0.222, SE ~0.003. 03's 0.260 vs my 0.266 is sampling difference. Consequence: weakest paths at high L are pushed toward end-fire. Because $\alpha$ is sorted and paired with placement order, the strongest path is the unbiased one. | `x4_03_heat_bias.py` (FACT) |
| E1 | "End-fire cell gets 3.8x uniform occupancy, 23-24% within one bin of 8" (11) | 11 | 11 | **Numbers UPHELD, wording REVISED.** Analytic uniform-$\psi$ occupancy of cell 8 (16-grid) = 22.6% (3.62x); eval_bank 3.72x; training mix ($L$ 1..9) 24.2% (3.88x). "Within one bin" is wrong: it is one cell; $\pm1$ cells is 39.6%. "Cells 0/8 depending on sign convention" is wrong: in this code end-fire = cell 8 (16-grid) / 16 (32-grid), broadside = cell 0. | `x4_04_occ.py`, `x4_05_trainocc.py` (FACT) |
| E2 | 05: "30% within 5 deg of end-fire" | - | 05: per-sample figure | **UPHELD as ambiguous but derivable**: per path 5.56%; per sample of 6 angles $1-(1-0.0556)^6=29.0\%$. Should be labelled per-sample. | DERIVATION |
| F1 | 05 item 4: near end-fire the 1-deg criterion is "nearly free" and end-fire mass "inflates Pd" (outside the four assigned reports) | this panel | none | **REJECTED (direction inverted).** $d\theta=du/(\pi\sin\theta)$, so 1 deg spans only 0.024 cell at 10 deg from end-fire: the tolerance is tighter in $u$, i.e. harder, not free. 05's own D6 (bias 0.29 deg at 10 deg vs 0.05 deg at broadside) agrees with this direction. Severity Medium (05 had High, inverted). | DERIVATION / CONCERN |
| G1 | 08: official bank regeneration - "Insufficient evidence" (output empty when 08 was written) | 08 | 08 | **UPHELD and upgraded to FACT.** `r08_official.out` now complete: authors' `validation_data_generator` (conds L=3, SNR -10..25 step 5, 1000 each, sigma 0.07, amps 'ones'), seed 42 equals eval_bank on data, feat, meta (sha1 53a7172f4dc3 both sides); seed 7 equals the zip test split on data and feat; seed 42 != zip test split. | `r08_official.py`, `r08_official.out` |
| G2 | 08: generator copies are md5-identical | 08 | - | **UPHELD.** e3c507165f629728abd49b2f747acf91 for both files. | `md5sum` (FACT) |
| G3 | 08: 7 v2 banks bit-exact reproducible | 08 | - | **UPHELD** (arrays and file md5 equal). | `r08_regen_v2.out` (FACT) |
| G4 | 08 labels positive results as "VERIFIED BUG" | this panel | - | **CONCERN, Low** (label hygiene, no dataset impact). | 08 |
| G5 | Seed collisions 1000/1015: 08 "scenes not identical"; 04/01 "only the first scene shared" | 06 | 04, 08 (row-aligned comparison) | **06 UPHELD; 04 and 08 REVISED.** `phase_d0_snr0` (L=3) vs `nuis_p0_snr0` (L=4) share 4 scenes: rows (0,0), (81,13), (164,26), (241,38). snr15 pair shares 3: (0,0), (408,64), (427,67). The RNG streams re-synchronise, so shared scenes sit at different row indices; a row-aligned comparison misses them. Seed 2020: `gain_g2_snr0` and `gain_g0.5_snr15` have identical $\psi$ in all 500 rows. 49 distinct v1 seeds across 56 banks. `nuis_*` banks are identical to each other by design (200 rows). | `MANIFEST.json` + slice comparison (FACT) |
| G6 | "47 of 56 v1 banks used" | 08 | - | **UPHELD**: 8+8+4+16+7+4 = 47; the 9 `nuis_*` and 3 `v1_clean_*` are unused. Only `phase_d0_snr{0,15}` from the colliding pair enter the v2 tables, so the collision is immaterial to v2 results. Resolves 01 vs 07. | `notebook_src.py:402-407` (FACT) |
| G7 | v2 draw order (points then alpha) differs from authors' (alpha, points, noise) | 08 | - | **UPHELD.** Banks are seed-exact within v2; not stream-identical to the authors' generator (statistically equivalent, ASSUMPTION). | `iabr2_sim.py:28-38` |
| H1 | 10: all worked-example arithmetic (Ex 1-12) | this panel re-ran independently | - | **UPHELD, all reproduced.** Ex 1 rng(10) trace and $\alpha$; Ex 2 $H[1,2]=-1.0731+0.9617j$; Ex 4 full $Y$ table, $\lvert Y\rvert[3,1]=1.1840$, $[3,3]=4.2285$, $Y=\mathrm{fft2}(H)/4$; Ex 5 relative change 0.1917, $Y_{imp}[3,2]=-0.0739-1.0000j$, path-1 leak 0.4364; Ex 6 $Y_{final}[3,3]=3.6894+2.1583j$, mean$\lvert Z\rvert^2=0.0875$, realised SNR 12.04 dB (10 prints 12.05; ratio 16.016 gives 12.045, negligible); Ex 7 $\tilde H[0,0]=1.1530-0.5805j$; Ex 8 slopes 1.8 and -0.6 deg/el, energy left 46.0%/93.3%, $\Delta\psi=[0.659,0.600]$ deg, $\Delta\phi=[0.221,0.267]$ deg; Ex 9 recovered 75.522/41.410 deg (G=8) and 71.790/46.567 deg (G=32), $\lVert a(5^\circ)-a(175^\circ)\rVert=0.2098$ (N=16); Ex 10 block sizes [3,4,4,4,4,5,4,4,4,4,5,4,4,4,4,3]; Ex 11 rng(2026) batch (L, SNR, clean, $\delta$, $\gamma$) and phase half-sums ~1e-9; Ex 12 T2_d5 sample 0 and eval_bank sample 7000 meta [3,25,16,16]. | `x4_06_toy.py`, `x4_07_ex11_12.py` (FACT) |
| H2 | 10 ASSUMPTION: `feat` order = [$\psi$; $\phi$] | 10 | 04 (LS fit), 11 (GT-cell check) | **Upgraded to FACT.** High-SNR eval_bank samples: top-$\lvert Y\rvert$ bin within 1 cell of a GT path for 100% with `feat`=[$\psi$;$\phi$] vs 15% if swapped. | `x4_07` |
| H3 | 10 ASSUMPTION: `meta` = [L, SNR, P, nt] | 10 | - | **Upgraded to FACT** (`dldoa_dataset_generation.py:536-537`; cols 2-3 constant 16,16). | FACT |
| H4 | 10 Ex 12 calls the end-fire cell "the wrap edge" | this panel | - | **REVISED wording**: end-fire alias is cell 8 of 16, not the 0/16 wrap. Digest's "1.4% at 5 deg, 2.7% at 8 deg" is the non-alias-only figure (see B1). | `x4_04` |
| I1 | 11 fig09: GT 17.3 dB, swapped 8.2 dB, random 3.9 dB | this panel | 11 | **UPHELD with a baseline caveat.** GT 17.34, swapped 8.24 dB reproduced. "Random" is not a 0 dB null: uniform-random cells give -1.22 dB; shuffled GT angles 3.25 dB; fresh uniform angles 3.73 dB (both axes cluster at cell 8). The 9 dB GT-vs-swapped gap remains valid and is independently confirmed by the 100% vs 15% peak-hit test. | `x4_08_gtratio.py` (FACT) |
| I2 | 11 fig12, fig13, fig16 match captions | this panel (viewed) | 11 | **UPHELD.** fig16: train 15.18%, eval_bank 3.15%, T2_d5 3.15% for <1 cell, reproduced (`x4_09`); fig13(d) SNR spread left tail consistent with Gamma(L,1/L). fig06, fig09 (numbers only), fig11, fig14 were not visually re-viewed: Insufficient evidence on their rendering. | viewed + `x4_09` |
| I3 | 11 training histograms are a replica of `make_batch` | 11 | stated limitation | **UPHELD**: seed-20260929 replica, not the actual run (training is `os.urandom`-seeded). | `iabr2_sim.py:127` |
| I4 | "48000 x 256 is CFG only" (11) | this panel | - | **REVISED to confirmed**: `outputs_train_stdout.log` shows step 48000/48000, batch 256, 191702 params, 108 min. Whether the run resumed mid-way: Insufficient evidence. | log (FACT) |
| I5 | 11: impairment is "physically correct, beam-independent" | 03 | 11 | **REVISED to modelling choice**: per-antenna weights identical for every beam (`iabr2_sim.py:52`); real phase shifters are state-dependent (03). ASSUMPTION, Low. | `iabr2_sim.py:52` |
| I6 | 11: "dldoa seed-7 file not found" | this panel | - | **Resolved**: the file is inside the zip; seed-7 regeneration equals the zip test split. | `r08_official.out` (FACT) |

## 2. New issues found by the panel

1. **Wrapped beam-space merging is under-reported by 04.** About 3-4% of L=3 scenes have a path pair within one beam cell once the $\pm\pi$ alias is respected (04: 0.013-0.037%). Training scenes with L>1 are worse (fig16: 15.18% for <1 cell in the training mix). Medium.
2. **Five different ramp-floor numbers** (1.44/2.0/2.07/2.71/1.4%) come from different alias handling; unify to the wrap-consistent 2.55% per axis at 5 deg and ~14% of L=3 scenes affected. Medium (caps the achievable 1-deg detection on impaired banks).
3. **05's "end-fire is nearly free for the 1-deg criterion" is inverted.** The criterion is tighter in beam space near end-fire. Medium.
4. **Sampler ordering bias at L>=4** pushes weak paths toward end-fire (P(within 20 deg): 0.221 first vs 0.266 last at L=9); negligible at L=3 (the bank scenario). Low-Medium: it shapes the training distribution, not the L=3 banks.
5. **Shared scenes in seed-collision pairs are 4 and 3, not 1**; and 04/08 row-aligned checks are blind to them. Low (immaterial to v2 tables).
6. **fig09 "random" baseline is not a null** (3.25-3.73 dB for angle-based random). Low.
7. **08's "VERIFIED BUG" label is used for positive results.** Low (hygiene).
8. **End-fire cell terminology** ("within one bin", "wrap edge", "cells 0/8") is wrong in 10/11; end-fire = cell 8 (16-grid) / 16 (32-grid). Low.
9. Realised SNR in Ex 6 is 12.04 dB, not 12.05 dB. None.

## 3. Not resolved / Insufficient evidence

- Whether the actual training run resumed from a checkpoint (only the log's final line is available).
- Rendering of fig06, fig09, fig11, fig14 was not re-viewed visually in this session; fig09's numbers were re-derived.
- Whether the v2 and the authors' generators are statistically identical despite different draw order (ASSUMPTION; not tested here).
- Independent-sampler estimate of "~14% of L=3 scenes affected by the ramp" ignores sampler rejection correlation (INFERENCE).

## 4. Verified script index (`scratch/`)

- `x4_01_closeness.py` wrapped vs non-wrapped beam closeness
- `x4_02_ramp.py` ramp-shift exceedance, four alias variants
- `x4_03_heat_bias.py` heat-cell merge rate and sampler bias by L
- `x4_04_occ.py` cell-8 occupancy (analytic and eval_bank)
- `x4_05_trainocc.py` training-mix occupancy
- `x4_06_toy.py` independent 4x4 toy, Ex 1-9, 10
- `x4_07_ex11_12.py` Ex 11 batch, Ex 12 bank samples, feat-order test
- `x4_08_gtratio.py` fig09 GT-power ratios
- `x4_09_fig16.py` fig16 <1 cell rates (Euclidean, wrapped) and unit-cell counts
- Reused: `r08_official.py/.out`, `r08_regen_v2.out`, `MANIFEST.json`, `outputs_train_stdout.log`
