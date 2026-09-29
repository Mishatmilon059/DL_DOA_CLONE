# X1 Cross-examination: Math (02) vs Physics (03), Bug Hunter (04) vs Pipeline (01)

Scope: investigation only. No project file was modified, no git was run. Only scratch script written: `scratch/x1_checks.py` (imports `dldoa_dataset_generation` read-only, uses `DG.generate_points`, N=40000 scenes for L=3, 6000 scenes for L=9, seed 12345).

Ground truth re-read this session: `iabr2_sim.py` (l.28-130) and `dldoa_dataset_generation.py` (l.46-103 `generate_points`, l.249-279 `generate_channel_v2`, l.539-600 `validation_data_generator`).

## 1. What the authors / pipeline actually do (recap, FACT)

- Angles: `generate_points` draws (phi, psi) uniformly in $[0,\pi]^2$ sequentially, rejecting any point within Euclidean distance $\pi/6$ of an earlier one (`dldoa_dataset_generation.py:77-87`). The $\pi/6$ separation is in the angle plane, not in the beamspace.
- Channel: $H=\sqrt{N_tN_r}\sum_l\alpha_l a_r(\psi_l)a_t^H(\phi_l)$ (`:271-279`), $Y=W^HHF+Z$ (`:565-567`); v2 adds per-antenna gain/phase errors to both codebooks (`iabr2_sim.py:49-53`), noise after the combiner (`:54-57`).
- v2 targets: 32x32 wrap-around Gaussian, sigma=1 cell, on rounded cells (`iabr2_sim.py:92-104`).

## 2. Resolution table

| # | Claim | Challenger | Defence | Resolution | Evidence |
|---|-------|-----------|---------|-----------|----------|
| 1 | Pairs closer than one DFT beamwidth in both u and v occur in 3-4% of L=3 scenes (0.9-1.1% within 0.5 cell) | 04 says 0.018% | 01/02/03/06 say 3-4% | 01/02/03 UPHELD, 04 REJECTED | x1_checks: box metric 1 cell = 3.82% (no wrap), 3.87% (torus); Euclid 3.08/3.11%; 0.5 cell box 1.07%, Euclid 0.87%. 04's number is not reproducible under any metric tried. |
| 1b | The high pair-closeness rate is driven by the $\pm\pi$ wrap | Panel hypothesis (from summary) | - | REJECTED | Wrap changes the rate by only 0.05 percentage points (3.815 vs 3.865). The cause is the arcsine density of $u=\pi\cos\psi$ piling paths at end-fire, not wrap. |
| 2 | L=3 heat-target merge fraction | 04 (1.3%) vs 02 (0.36%), 01 (28/8000=0.35%) | - | 02/01 UPHELD, 04 REJECTED | x1_checks: 132/40000 = 0.330% (coincident rounded 32-grid cells). 04's 1.3% is not reproduced. |
| 3 | Rejection sampler biases later placements toward end-fire at L=9 | 04 retracted a bias using a KS test at three points (L=3) | 03: P(within 20 deg of end-fire) 0.221 to 0.260 over index | 03 UPHELD; 04 not contradicted (different L, different test) | x1_checks L=9: psi 0.220 to 0.264, phi 0.222 to 0.264 (uniform expectation 0.222; SE about 0.005, so the rise is about 8 SE). Cause: INFERENCE, boundary points of the square have fewer excluded neighbours so later placements survive rejection more often there. |
| 4 | End-fire fractions | 02 32.2% (1 cell), 03 22.6% (half cell), 04 29.1% ($|\cos\psi|>0.9$), 05 "30% within 5 deg" | - | 02, 03 UPHELD (exactly the arcsine law); 04 REVISED (analytic 28.7%, 0.4 pp gap is sampling/threshold rounding); 05 REJECTED as stated | Analytic: 1 cell 32.17%, half cell 22.63%, $|\cos|>0.9$ 28.71%, within 5 deg of either end-fire 5.56%. "30% within 5 deg" is inconsistent with uniform angles; likely a threshold in cosine or cell units. |
| 5 | Gain SNR bonus | 05 quotes 1.0817, 06 quotes 1.178 | - | REVISED (definitions differ) | Per side $E[g^2]=\sinh(x)/x$, $x=\gamma\ln10/10$: 1.0089 / 1.0357 / 1.0814 / 1.1475 for gamma=1/2/3/4 dB. Both sides: 1.018 / 1.073 / 1.170 / 1.317 (+0.08 / +0.31 / +0.68 / +1.20 dB). 05's 1.0817 matches one side at 3 dB; 06's 1.178 is the two-side value at 3 dB to within 0.7% (small residual, Monte Carlo or rounding; not resolved). |
| 6 | Ramp-induced angle-shift fractions (0.13 to 7.1% in 01, 0.38 to 2.7% in 02, 0.55/1.10/2.71% in 03) | Mutual | Each report used a different definition | UNRESOLVED (definition mismatch) | The reports do not share a definition (AoA only vs AoA+AoD, alias-only vs any shift over a threshold). One consistent definition computed here (per side, LS slope of uniform $\pm\delta$ errors, P(path crosses end-fire) = alias flip): delta=1/2/5/8/15 deg gives 0.48 / 0.69 / 1.10 / 1.39 / 1.90%. Slope std 0.00055 / 0.00109 / 0.00273 / 0.00437 / 0.00819 rad/element. Which of the earlier figures is "right" depends on the definition; magnitude order of about 1% at delta about 5 deg agrees with 02 and 03. |
| 7 | SNR framing: "-10 dB nominal = +14 dB integrated / 24 dB processing gain" | 02 (per-entry SNR is $1/\sigma^2$) | 03 (framing for CRLB) | 03 REVISED | Total signal energy over total noise energy over 256 entries equals the per-entry SNR. The 24 dB gain applies to estimation variance (CRLB) and to a single path's peak beam, not to a defined integrated-SNR metric. |
| 8 | $\psi,\phi\in[0,2\pi]$ vs $[0,\pi]$ | 02: equivalent for the data | 03/01: physically redundant, ambiguous labels | Both UPHELD, reconciled | $\cos$ has the same distribution on both ranges, so $Y$ statistics are equal, but $\psi$ and $2\pi-\psi$ give the same steering vector, so the label would be ambiguous. The code uses $[0,\pi]$ (`:78-79`), so this is a paper-text issue only. |
| 9 | Test-set provenance (02 item 13: dldoa test not the seed-42 set) | 02 | 01: bit-exact seed-7 result | 02 REVISED (superseded) | 01 bit-exact regeneration of the `dldoa_dataset/test_*` protocol with seed 7; 02 and 04 bit-exact for `eval_bank.npz` seed 42. |
| 10 | No sign / conjugation / transposition bug in $Y=W^HHF$ and impairment targets | 04 (numerical), 02 (LS convention test) | 01, 03 | UPHELD | `iabr2_sim.py:47,53,86`: $\tilde H=\bar D_rHD_t$ gives Rx phase $-\varepsilon_r$, Tx $+\varepsilon_t$, consistent with `impairment_targets`. |
| 11 | Simulator equals the original `generate_channel_v2` | 04, 02 | 01 | UPHELD | Agreement to about 4e-15 reported by 02 and 04; the code paths match by inspection (`:271-279` vs `iabr2_sim.py:45-47`). |
| 12 | Rx/Tx phase ramp equals a common angle shift; unidentifiable subspace is exactly 6-D | 02 | 03 | UPHELD | `project_impairment` (`iabr2_sim.py:76-80`) removes mean and linear ramp for phase and mean for log-gain: 2+2+2 = 6. |
| 13 | Per-sample signal-power std at L=1 (0.96 vs 0.99) | 02/04 | - | REVISED (sampling noise) | Theory: $|\alpha|^2\sim\text{Exp}(1)$, std = 1. Not a defect. |
| 14 | Sign of ramp shift ($\cos\psi\pm a/\pi$) | 02 vs 10 | - | REVISED (definitional) | Sign depends on the direction of the ramp definition; the magnitude is unaffected. |
| 15 | v1 seed 2020 collision (identical scenes) | 04 | 01 | UPHELD | Reported by 04 seed audit, consistent with 01. Not re-derived here beyond reading (insufficient independent evidence this session). |
| 16 | Paper text vs code (SNR -10..25 vs -15..24; L 1..10 vs 1..9; $\pi/6$ code-only) | 01 | 04 | UPHELD | `iabr2_sim.py:108-113`; `dldoa_dataset_generation.py:555`. |

## 3. New issues found by the panel

1. **CONCERN (Medium):** The 04 statistics (0.018% pair closeness, 1.3% heat merge) are not reproducible; any downstream report that relied on 04 for "pairs are virtually never unresolvable" must be corrected. 3-4% of L=3 scenes contain a pair within one DFT cell, and 0.33% have coincident 32-grid targets.
2. **CONCERN (Low):** The sequential rejection sampler makes later-placed (weaker after sorting is independent, so this is a geometry not an amplitude effect) paths more end-fire at large L (0.222 to 0.264 at L=9). Effect is a modest increase in the number of end-fire paths, which alias between $\psi=0$ and $\pi$.
3. **CONCERN (Low):** Report 05's "30% within 5 deg of end-fire" should be corrected (analytic 5.56% for 5 deg).
4. **CONCERN (Low):** The ramp-shift fraction is quoted with at least three incompatible definitions across reports 01-03; a shared definition should be stated before citing.

## 4. Limits

Insufficient evidence for the exact source of 04's 0.018% and 1.3% figures, and for the 1.178 vs 1.1695 gap in row 5. Seed-2020 collision was not independently re-run.
