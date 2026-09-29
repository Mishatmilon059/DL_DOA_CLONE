# Agent 06: Dataset Validity Audit, IABR-Net v2 data-generation pipeline

**Scope.** This audit covers only the generation of model inputs, labels and targets: the training stream, the tuning banks, the test banks (v2 `T2_*`, v2 `tune_*`, the v1 robustness banks reused by v2, the frozen "official" `eval_bank.npz`) and the separate `dldoa_dataset` test set. Model accuracy is out of scope.

**Method.** I read the source files myself. I then **regenerated every bank from its seed** and compared bit for bit. I also replayed the RNG stream of the frozen bank to recover the path gains, which are not stored, and computed statistics on all saved banks. All scripts are in `D:\ai_ml_project\IABR_v2_DATASET_AUDIT\scratch\` (`s1_frozen.py` … `s7_dldoa.py`, `s4g.py`). No project file was modified.

**Labels.** FACT = read directly from code, paper or data. DERIVATION = follows mathematically. VERIFIED BUG = a defect reproduced numerically. INFERENCE = a reasoned conclusion that is not directly observed. ASSUMPTION = taken as given without proof. CONCERN = a validity risk that is not a code defect.

---

## 1. What the authors actually did

### 1.1 Signal model (training stream and all banks)

| Item | Implementation | Source |
|---|---|---|
| Array / codebook | ULA, $\lambda/2$, $n_t=n_r=P=Q=16$; $a(\theta)=\frac1{\sqrt{16}}[e^{-j\pi k\cos\theta}]_{k=0}^{15}$; DFT codebooks from the original `beamforming_vector_generation_P/Q` | `iabr2_sim.py:11-13,45-46`; `dldoa_dataset_generation.py:111-205` |
| Channel | $H=\sqrt{n_tn_r}\sum_l\alpha_l a_r(\psi_l)a_t^H(\phi_l)$ | `iabr2_sim.py:47` |
| Path gains | $\alpha_l\sim\mathcal{CN}(0,1/L)$, sorted by $|\alpha|$ descending. The angle list is **not** permuted, so column 0 = strongest path | `iabr2_sim.py:35-37` |
| Angles | `DG.generate_points(L, π/6)`: sequential rejection sampling in $[0,\pi]^2$ with **Euclidean** distance $\sqrt{\Delta\phi^2+\Delta\psi^2}\ge\pi/6$ in the joint (AoD, AoA) plane | `dldoa_dataset_generation.py:46-103` (line 81-84) |
| Impairment | $W=D_rW_0$, $F=D_tF_0$ with $D=\mathrm{diag}(g\,e^{j\epsilon})$. $\epsilon\sim U(-\delta,\delta)$ i.i.d. per antenna. $20\log_{10}g\sim U(-\gamma,\gamma)$. Independent draw for Rx and Tx, redrawn **per sample** | `iabr2_sim.py:48-52` |
| Observation | $Y=W^HHF+Z$, $Z_{qp}\sim\mathcal{CN}(0,10^{-\mathrm{SNR}/10})$, added after combining and independent of the impairment | `iabr2_sim.py:53-57` |
| Network input | real/imag of the 16×16 $Y$ (`to_ri`), float32 | `iabr2_sim.py:63-64,115` |
| Heat target | 32×32 wrap-around separable Gaussian, $\sigma=1$ cell, peak 1 at the **rounded** cell of $q=32\,\mathrm{wrap}(-\pi\cos\psi)/2\pi$, $p=32\,\mathrm{wrap}(\pi\cos\phi)/2\pi$; paths combined with `max` | `iabr2_sim.py:71-73,92-104` |
| Impairment target | antenna-domain distortion $\bar D_r H D_t$: phase of $(-\angle D_r, +\angle D_t)$ with mean and linear ramp removed; log-gain with mean removed | `iabr2_sim.py:76-89` |
| Training batches | $L\sim U\{1..9\}$, SNR $\sim U\{-15..24\}$ (integers), 20 % clean, else $\delta\sim U(0,8^\circ)$, $\gamma\sim U(0,3\text{ dB})$; fresh every step (48 000 × 256 samples), never stored | `iabr2_sim.py:107-117`, CFG `notebook_src.py:81-92` |
| Worker RNG | `default_rng([seed, worker_id, os.urandom(4)])` | `iabr2_sim.py:127` |

### 1.2 Evaluation banks

| Bank | Generator | Content | Source |
|---|---|---|---|
| `official` (`eval_bank.npz`) | original `DL_DOA/src/tvt_data_generation_v3.validation_data_generator`, seed 42, via `scripts/generate_frozen_banks.py` | L=3, SNR −10..25 step 5, 1000/SNR, clean; stores `data` (64×64×2 zoomed), `feat` (ψ;φ), `meta` | `scripts/generate_frozen_banks.py:28-65` |
| `T2_d1/d2/d5` | v2 `make_bank`, seeds 9101/9102/9105 | L=3, SNR −10..25, 1000/SNR, phase δ = 1/2/5° (no gain) | `notebook_src.py:356-372` |
| `tune_*` | v2 `make_bank`, seeds 9901-9904 | L=3 (L=6 for `tune_L6`), 150/SNR, used **only** to pick self-cal (λ, τ) from 7 grid points | `notebook_src.py:373-378,811-827` |
| v1 robustness (47 used) | v1 suite `make_standard_bank` / `make_separation_bank`, seeds in `MANIFEST.json` | phase, gain, OOD phase, L, separation, SNR tails; 500 each | v1 notebook cell 9; `notebook_src.py:402-407` |
| `dldoa_test` | zip `dldoa_dataset-20260911T150638Z-1-001.zip`; generator not identified | L=3, SNR −10..25, 1000/SNR | `infer_dldoa_test.py:15-25` |

Decoders are given the true $L$ of each test sample (`notebook_src.py:796-806`). This follows the paper's protocol, where the number of desired paths is set to $L$.

---

## 2. Mathematical and physical correctness checks

| # | Check | Result | Label |
|---|---|---|---|
| C1 | `eval_bank.npz` regenerated with the **original** `validation_data_generator` (seed 42, the 8 conditions, 1000 each) | `data`, `feat`, `meta` are **bit-identical** (max diff 0.0). The reproduction `dldoa_dataset_generation.validation_data_generator` is also bit-identical | FACT (`s1_frozen.py`) |
| C2 | All 7 v2 banks regenerated with a copy of `make_bank` | bit-identical `Y16`, `psi`, `phi` (0.0) | FACT (`s2_v2banks.py`) |
| C3 | v2 impaired `simulate` vs original impaired codebooks (`error_deg`) fed the same per-antenna errors | max diff $3.0\times10^{-15}$ | FACT (`s6_imp_equiv.py`) |
| C4 | Codebooks unitary | $\lVert W_0^HW_0-I\rVert_\infty=2.3\times10^{-15}$, F likewise | FACT (`s4g.py`) |
| C5 | Antenna-domain inverse $W_0YF_0^H=\bar D_rHD_t$ (noise-free, 5° phase + 2 dB gain) | $1.3\times10^{-14}$. The sign convention $-\angle D_r$ for rows and $+\angle D_t$ for columns in `impairment_targets` is correct | FACT / DERIVATION |
| C6 | Heat-target peak cell vs 32-point FFT peak of the antenna-domain channel, L=1, 3000 trials | exact match 100 % | FACT (`s4g.py`) |
| C7 | angle → cell → angle round trip (`angles_to_cells` ↔ decoder `U=-2πq/G`, `V=2πp/G`, `uv_to_angles`) | $\le 1.7\times10^{-11}$ deg | FACT |
| C8 | Padded paths (L < lmax): `nan_to_num`, α = 0, heat mask | L=1 padded to 9 gives exactly 1 positive; padded α exactly 0 | FACT |
| C9 | Impairment targets: mean and ramp removed | residual mean $<10^{-9}$, ramp $<4\times10^{-8}$ | FACT |
| C10 | `upsample64`/`downsample16` vs `scipy.ndimage.zoom(order=0)` | exact on `eval_bank` and `dldoa_test` (0.0). Note the zoom is **non-uniform**: row/column block widths `[3,4,4,4,4,5,4,4,4,4,5,4,4,4,4,3]` | FACT (`s4_more.py`) |
| C11 | Noise power per entry vs claim $10^{-\mathrm{SNR}/10}$ | within 1 % at every SNR in every v2 bank (e.g. 25 dB: 0.003153 vs 0.003162). `dldoa_test` LS residual matches within 1-2 % | FACT |
| C12 | Mean signal power per entry $\lVert G\rVert_F^2/256$ | 0.96-1.04 (clean/phase banks). This is the paper's $\rho=1$, $\mathrm{SNR}=1/\sigma_n^2$ definition | FACT |
| C13 | Impairment distortion-to-signal ratio $\lVert G_{imp}-G_{id}\rVert^2/\lVert G_{id}\rVert^2$ | δ=1°: −36.9 dB; 2°: −30.9; 5°: −22.9; gain 3 dB: −10.5. Matches the closed form $2-2(\sin\delta/\delta)^2$ | FACT + DERIVATION |

**Conclusion on correctness.** I found no conjugation, transpose, degree/radian, wrap or normalization bug in the v2 simulator, its targets, or the bank loaders (`load_bank('official')` maps `feat[:,0]`→ψ, `feat[:,1]`→φ and `meta[:,0]`→L, `meta[:,1]`→SNR correctly; `notebook_src.py:386-388`). The frozen bank is exactly the paper's seed-42 protocol.

---

## 3. Bank statistics

### 3.1 Composition (FACT, `s2_v2banks.py`, `s3_stats.py`)

- `official`, `T2_d1/2/5`: 8000 samples each, L=3 only, SNR ∈ {−10,…,25}, exactly 1000 per SNR.
- `tune_clean`: 600 (150 × 4 SNRs). `tune_d5`, `tune_g3`, `tune_L6`: 300 each.
- Stored impairment extremes equal the specification: max |ε| = 1.000 / 2.000 / 5.000° and max |gain| = 3.000 dB.

### 3.2 Angle distribution and separation (FACT, `s3_stats.py`)

| Bank | min Euclid. sep. (deg) | pairs < 0.5 beam cell in wrapped (u,v) | pairs < 1 cell | AoA within 5° of endfire |
|---|---|---|---|---|
| official | 30.01 | 0.92 % of samples | 3.15 % | 5.8 % |
| T2_d1 / d2 / d5 | 30.00 / 30.02 / 30.01 | 0.83 / 0.92 / 0.88 % | 3.1-3.3 % | 5.6-5.9 % |
| v1 L8 (snr0/15) | 30.00 | 8.8 % | 27 % | 6.4 % |
| v1 L10 | 30.01 | 15 % | 44-46 % | 6.1 % |
| v1 sep_20deg | 28.28 | 0 | 0 | 0 (angles restricted to [20°,160°]) |
| v1 sep_5deg | 7.07 | 4 % | 100 % | 0 |

- KS test of ψ and φ against $U(0,\pi)$: p = 0.16-0.93 for official and the T2 banks, both for path 0 (strongest) and path 2. Two-sample KS of each T2 bank vs official: p ≥ 0.39. **The v2 banks are statistically indistinguishable from the frozen official bank in angle distribution, SNR grid, L and power**. They differ only by the impairment (FACT). The original generator draws α before the points and v2 draws the points first. This changes which samples appear but not their distribution (DERIVATION).
- **Every** pair closer than 1 beam cell in the official bank (255 pairs in 252 samples) comes from the $\pm\pi$ **spatial-frequency wrap at endfire**: 154 from ψ wrap, 101 from φ wrap, 0 without wrap (FACT, `s4_more.py`). Example: ψ₁ = 5°, ψ₂ = 175° are 170° apart in angle but only 0.06 beam cells apart in $u=-\pi\cos\psi$.

### 3.3 Path strength and effective SNR (official bank, α recovered by RNG replay with 0 mismatches)

- Sorting is verified: $|\alpha_0|\ge|\alpha_1|\ge|\alpha_2|$ in all 8000 samples (FACT).
- $\sum|\alpha|^2$: mean 1.008, std 0.58, 5-95 % range 0.28-2.11. So the **realized** per-entry SNR spreads about −5.6…+3.2 dB around nominal. The mean realized SNR in dB is ≈0.8 dB below nominal (FACT, C12 table in `s2_v2banks.py` output).
- The per-path post-beamforming SNR is $10\log_{10}(256|\alpha_l|^2)+\mathrm{SNR}$ (DERIVATION: matched DFT beams give coherent gain $n_tn_r$ = 24.1 dB). Median strongest / weakest path is 11.4 / 3.2 dB at −10 dB nominal and 46.3 / 37.9 dB at 25 dB. The fraction of samples whose **weakest path is below 0 dB** is 28.1 % (−10 dB), 10.3 % (−5), 2.6 % (0), 1.0 % (5) and ≈0 for SNR ≥ 15 (FACT).
- Strongest/weakest power ratio: median 8.1 dB, p90 16.2 dB.

### 3.4 Training stream (10 240 samples drawn with `make_batch`, `s5_train_dup.py`)

- L counts are uniform over 1..9. 20.2 % of samples are clean (impairment target all zero).
- Impairment targets: |phase| max 0.196 rad (11.2°), rms 0.039 rad; |log-gain| max 0.44, rms 0.099.
- Heatmaps with **fewer exact-1 positives than L** (two paths merged at the rounded 32-grid cell): 1.7 % overall; 0.6 % at L=3, 4.7 % at L=9 (FACT).
- Per-sample input RMS spans 21.9 dB (p1-p99), which supports the per-sample normalization C7.
- `generate_points` never failed (0/300) for L = 9, 10 or 12.

---

## 4. Train/test independence, leakage and duplicates

| # | Finding | Label | Severity |
|---|---|---|---|
| I1 | The training stream uses `default_rng([seed, wid, os.urandom(4)])`, which is independent of every bank seed (42, 43, 1000-7025, 9101-9105, 9901-9904). Scene-level overlap between the stream and any test bank has probability ≈0 (continuous angles). **No sample-level train/test leakage.** | FACT (code) + DERIVATION | None |
| I2 | Scene duplicates across all 63 banks (key = first-path (ψ,φ) to 1e-5): `official` ∩ any v2 bank = 0; `dldoa_test` ∩ `official` = 0; `dldoa_test` ∩ v2 banks = 0; no exact duplicate Y within any bank. Maximum cosine similarity of normalized Y between official and T2_d1 at 25 dB is 0.915 (within-official maximum 0.802), so no near-duplicates | FACT (`s5`, `s7`) | None |
| I3 | **Seed collision in the v1 robustness suite:** `gain_g2_snr0` and `gain_g0.5_snr15` both use seed 2020 (`MANIFEST.json`, formula `2000+int(10g)+snr`). The 500 scenes are **identical** (max |Δψ| = |Δφ| = 0), as are the underlying uniform draws of the gain errors (scaled differently) and the unit noise. Both banks are in v2's `ROBUST_BANKS`, so these two table cells are not independent evidence. (The analogous formula also gives 1000/1015 collisions between `phase_d0_snr*` and `nuis_*`: 4 and 3 shared scenes. `nuis_*` is not used by v2.) | VERIFIED BUG (in v1 bank construction, inherited read-only) | Low |
| I4 | `dldoa_dataset` validation set (seed-42 stream) and the official test set (seed-42 stream) are carved from **the same PRNG stream**. Val #595 (L=8) has the same first 3 paths as official #1509 (L=3). v2 does not use this val set. It matters only for models validated on it | FACT (`s7`) | Low (not v2) |
| I5 | **Test-set reuse during development.** The v2 design (C1-C8) is motivated explicitly by results on the official bank and the v1 robustness banks: a NOMP diagnostic on a 300/SNR subset of the official test set (`notebook_src.py:104-117`) and a decoder sanity check on official samples (`:670-674`). Only the self-cal (λ, τ) is chosen on separate tuning banks. Mitigation: on the independent `dldoa_test` draw, IABR-v2 scores 0.7531 vs 0.7551 on official and the U-Net 0.7354 vs 0.7366, a difference within sampling error. This gives no sign of adaptive overfitting for clean L=3 | FACT + INFERENCE | Low-Medium |
| I6 | The tuning banks come from the same distribution as the test banks (L=3; SNR and impairment levels overlapping T2 and v1). This is legitimate validation. The grid has only 7 points, so selection bias is small | FACT / INFERENCE | None |

---

## 5. Validity issues and concerns (most severe first)

### V1. The paper-Table-2 impairments are nearly invisible at most SNRs (CONCERN, High for interpretation)

$E|e^{j(\epsilon_r+\epsilon_t)}-1|^2=2-2(\sin\delta/\delta)^2$ gives distortion-to-signal ratios of −36.9 / −30.9 / −22.9 dB for δ = 1 / 2 / 5° (DERIVATION, confirmed on the banks, C13). Relative to the per-entry noise, the distortion in `T2_d5` is −33 dB at −10 dB SNR, −13 dB at 10 dB and only +2 dB at 25 dB. In `T2_d1` it is below noise at **every** SNR (−12 dB at 25 dB) (FACT). Consequences (INFERENCE):
- `T2_d1` and `T2_d2` are effectively clean banks. Pd differences between them and `official` (e.g. U-Net 0.736 on official vs 0.736 on d=1) are dominated by sampling noise: the SE of Pd per SNR point is ≈0.005-0.009 for 1000 × 3 components. The banks are **not paired** with official (different scenes), so cross-bank "degradation" numbers mix scene sampling noise with the impairment effect.
- Only 20-25 dB in `T2_d5` stresses impairment robustness. Claims of "robustness to hardware impairments" from Table-2 banks are weak evidence. The v2-added gain errors (≥2 dB gives −14…−7.7 dB) and OOD phase 10-15° (−17 / −13.4 dB) are the only strong impairment tests.
- This follows the paper's own model and levels (Meneses-Albalá Sec. 2.2; TVT Sec. IV-D). It is not a code defect.

### V2. Unidentifiable linear phase ramp gives a Pd ceiling on impaired banks (DERIVATION, Medium)

A linear ramp $\epsilon_n=cn$ on the Rx array gives $e^{-jn(\pi\cos\psi-c)}$ for every path. It is an exact common shift of all spatial frequencies, so no estimator can separate it from the angles. `project_impairment` correctly removes it from the IABC targets (`iabr2_sim.py:76-80`), but the angle labels stay the true angles. A Monte Carlo over the uniform per-antenna model (`s4_more.py` (h)) gives these fractions of AoAs shifted by more than 1° **irreducibly**: 0.13 % (δ=1°), 0.5 % (2°), 2.0 % (5°), 3.5 % (8°), 4.6 % (10°), 7.1 % (15°), with the same values for AoDs. This caps the achievable Pd in `T2_d5` and `oodphase_*` below 1 even at infinite SNR. It affects every method equally, so it is not a leak, but it is a floor that should be reported with those tables.

### V3. Minimum separation is enforced in angle space, not in the resolvable spatial-frequency domain (FACT / CONCERN, Medium)

The π/6 constraint is Euclidean in (φ, ψ) degrees (`dldoa_dataset_generation.py:81-84`). The observation depends on $u=-\pi\cos\psi$ and $v=\pi\cos\phi$, which are circular in $2\pi$. Results:
- 3.1 % of official test samples contain a pair closer than one Rayleigh cell ($2\pi/16$) and 0.9 % one closer than half a cell. **All** of them come from the endfire wrap (§3.2). These samples are physically near-unresolvable, and ψ≈0° vs ψ≈180° is an aliasing ambiguity.
- Near endfire, a 1° angle tolerance means $\Delta u=\pi\sin\psi\cdot\pi/180$. For 9.6 % of AoAs this is under 0.02 beam cells (median 0.098 cells). A tiny u error across ±π maps to a ≈180° angle error. The angle-domain Pd metric with uniform-in-angle sampling therefore puts ~6 % of components into an extreme-precision, wrap-ambiguous regime, and hence controls the high-SNR Pd ceiling.
- Inherited from the original code. The **paper text does not state a π/6 separation**: TVT Sec. IV-B says only that angles are uniform in [0, π], and Table I lists only optimizer, learning rate, batch, epochs and data sizes (FACT, page 10206 rendered). The reproduction docstring (`dldoa_dataset_generation.py:16-28`) attributes π/6, L ∈ {1..9} and SNR [−15, 24] to "Section IV-B & Table I". The paper text instead says SNR between −10 and 25 dB and L from 1 to 10. **The docstring misattributes code values to the paper** (FACT). v2 follows the code ("same as the original training generator", `notebook_src.py:85`), which matches the released weights' training.

### V4. The training prior makes close-path banks out of distribution (FACT, Medium for S3 claims)

Training always has Euclidean separation ≥ 30°. The v1 `sep_Xdeg` banks offset **both** AoA and AoD by X (Euclidean $X\sqrt2$), and only in the positive direction. They also restrict angles to [20°, 160°], which excludes the endfire regime present in training and test. `sep_20deg` (28.3°) and below are therefore outside the training prior, and `sep_30deg` (42.4°) is easier than the typical test pair. Resolution claims (S3/C2 in the notebook) are measured on a family with a different geometry from the in-distribution test set.

### V5. Low-SNR Pd is bounded by physically undetectable weak paths; high-SNR is "easy" in SNR terms (FACT / INFERENCE, Medium for interpretation)

α ~ CN(0, 1/L) with no dominant (LoS) path gives a weakest-path array SNR below 0 dB in 28 % of samples at −10 dB. Pd at −10 and −5 dB is therefore largely a detection-probability-of-weak-paths measurement. Methods that always return L peaks (NOMP, IABR-v2) differ from methods that drop samples (U-Net: `pd_paper` excludes samples with fewer than L blobs, `TVT_Blob_Inference.py:126-131,256`). At ≥ 15 dB every path is above 10 dB array SNR, so the remaining errors come from endfire precision (V3), resolution and impairments.
- **SNR definition.** Nominal SNR is the per-entry average SNR after unit-norm beams. The coherent per-path SNR is 24.1 dB higher (DERIVATION). "−10 dB" here is not comparable to per-antenna SNR definitions in other literature.
- The realized per-sample SNR varies over roughly ±5 dB around nominal because $\sum|\alpha|^2\sim\Gamma(L,1/L)$ (FACT, §3.3).

### V6. Impairment realism: the separable per-antenna model matches the v2 architecture exactly (INFERENCE / CONCERN, Medium)

- The model applies one static multiplicative error per antenna, identical for every codeword, i.i.d. uniform. In the antenna domain this gives exactly $\bar D_rHD_t$ (C5), a rank-1-separable distortion. v2's inverse codebook, IABC and self-calibration are built to invert exactly this structure. Codeword-dependent (state-dependent) phase-shifter errors, phase quantization (low-bit shifters), gain-phase coupling, mutual coupling and element patterns are not simulated. Robustness conclusions may therefore not transfer (INFERENCE).
- The impairment is **redrawn for every sample** (`iabr2_sim.py:49-50`; the original code also redraws per call, `tvt_data_generation_v3.py:81,107`, even though its comment calls it "static"). Test banks therefore average over devices. A deployed device has one fixed error (FACT / INFERENCE).
- Noise is added after combining and not scaled by the impaired combiner norm. With gain errors the signal power rises by $10\log_{10}E[g_r^2]E[g_t^2]$: +0.68 dB at 3 dB, +1.19 dB at 4 dB (`tune_g3` measured 1.13-1.23 vs 1.0). **Gain-impairment banks therefore carry a small SNR bonus** (FACT / DERIVATION, Low).
- The P=Q=n_t=n_r DFT codebook is exactly unitary. v2's "stage 0 exact inverse" depends on this idealization, and no non-square or quantized codebook is tested (FACT / INFERENCE).

### V7. Missing channel-realism components (CONCERN, Medium, inherited)

Not modeled: narrowband single-carrier beyond the flat-per-subcarrier assumption; channel variation and phase noise/CFO across the **256 sequential pilot slots** (the antenna-domain inverse and NOMP rely on phase coherence across all Y entries); path clusters and angular spread; near-field effects; LoS K-factor; ADC quantization; non-Gaussian interference. Angles are uniform over the full [0, π] including endfire, where real ULA elements have little gain (INFERENCE). All of these are shared with the base paper, so comparisons stay fair, but absolute Pd values are optimistic.

### V8. Label construction details (FACT, Low)

- Heat targets place the peak at the **rounded** 32-grid cell (quantization ≤ 0.25 beam cell). Positives in the focal loss are exact-1 cells only (`notebook_src.py:686-691`). Paths that round to the same cell merge: 0.6 % of L=3 and 4.7 % of L=9 training samples. This is a small amount of label noise, and sub-cell accuracy comes from the model-based decoder, not the labels.
- Strongest-first ordering: column 0 of `psi/phi` is always the strongest path in the frozen bank and the v2 banks (verified for official). No v2 target or metric uses the order (heatmap is permutation-invariant, metric uses Hungarian matching, `TVT_Blob_Inference.py:123-142`), so no ordering shortcut exists (INFERENCE). The rejection sampler makes the first point exactly uniform and later points slightly edge-biased; KS tests detect no effect (p ≥ 0.06).
- Test L is fixed at 3 and the decoder is **given L** (oracle model order). This is a protocol-level shortcut shared by all methods and inherited from the paper. Nothing in the data tests model-order estimation (FACT).

### V9. Coverage of the parameter space (FACT)

| Parameter | Training | Test / robustness | In distribution? |
|---|---|---|---|
| L | 1..9 uniform | 3 (main), 1-8, **10** | L=10 is OOD |
| SNR | integers −15..24 | −10..**25**; tails −25, −20, **30, 35** | 25 dB is 1 dB beyond; tails OOD |
| Phase δ | 0 (20 %) or U(0, 8°) | 1, 2, 5, 10, 15° | 10 and 15° OOD |
| Gain γ | 0 or U(0, 3 dB) | 0.5, 1, 2, 4 dB | 4 dB OOD |
| Separation | ≥ 30° Euclidean | sep banks 1.4-42° | < 30° OOD |
| P=Q, n | 16 only (the original code also trains 32) | 16 | yes |

Diversity: 48 000 × 256 ≈ 12.3 M fresh samples, so memorization is impossible (DERIVATION).

---

## 6. Reproducibility and provenance

| # | Finding | Label |
|---|---|---|
| R1 | `eval_bank.npz` provenance is established: `scripts/generate_frozen_banks.py` using the original `validation_data_generator` (seed 42, L=3, SNR −10..25, P=nt=16, 1000/cond). Bit-exact regeneration confirmed (C1) | FACT |
| R2 | All v2 banks are bit-exactly regenerable from their seeds (C2) | FACT |
| R3 | The **training stream is not reproducible**: `os.urandom` enters the seed (`iabr2_sim.py:127`), and resumption reseeds with `seed + 17*start` (`notebook_src.py:716`). Retraining gives a different model; seed-to-seed variance of v2 is not reported | FACT |
| R4 | `make_bank` returns early if the file exists (`notebook_src.py:357-358`). No spec hash or manifest is stored for v2 banks, and the files store `seed/phase_deg/gain_db` but not `n_per/snrs`. A changed spec would silently reuse stale banks. The current banks match their specs (C2) | FACT |
| R5 | The original and reproduction generators draw the impairment phase with the **global, unseeded** `np.random.uniform` (`tvt_data_generation_v3.py:81,107`; `dldoa_dataset_generation.py:161,197`). Any impaired validation set built with them (e.g. the paper's Table II / Meneses Table 2 data) is not reproducible from `seed`. v2 avoids this (rng-based) | FACT |
| R6 | `dldoa_dataset` (zip dated 2026-09-11) is a **different draw** from `eval_bank` (0 shared scenes, data differ), yet it has the same distribution (KS p = 0.60, min separation 30.001°, noise variance matches). The generator is **not identifiable**: `save_test_dataset` with its default seed 42 would reproduce `eval_bank` exactly (C1), and the zip contains `train_meta.npz`, which the current `save_training_dataset` does not write (`dldoa_dataset_generation.py:781-782`). It must come from another script version or seed. Insufficient evidence for which | FACT + INFERENCE |

---

## 7. Numerical stability

- float32 storage of Y and angles: at 35 dB (σ² ≈ 3e-4) the noise rms is still > 1e4 × float32 eps relative to the signal (DERIVATION), so there is no precision issue.
- `np.angle` in `impairment_targets` would wrap only for |ε| > 180°. The maximum used is 15°, so this is safe (FACT).
- `arccos` inputs are clipped in the decoders. `generate_points` has a 10 000-attempt cap with no failures up to L=12 (FACT).
- The non-uniform nearest-neighbour zoom (C10) is lossless (exact downsample) and identical to what the released U-Net was trained on. It is not an instability (FACT).

---

## 8. Insufficient evidence

- Exact generator and seed of `dldoa_dataset/test_*` (R6).
- How the Meneses-Albalá Table 2 test data were generated. Their text describes the same per-antenna uniform phase model, but the seeds and paired or unpaired status are unknown, and the extracted text of their angle range is garbled.
- Whether the authors' U-Net was validated or model-selected on a seed-42 set that shares the PRNG stream with the seed-42 test set (I4 shows this for the project's reproduction; the authors' own validation seed is not in the repo code).

---

## 9. Bottom line

The v2 data-generation code is **mathematically and physically consistent** with the base paper's official code. Every convention I checked holds, all banks are bit-reproducible, and there is **no train/test sample leakage**. The frozen official bank is exactly the paper's seed-42 test set, and the v2 T2 banks share its distribution apart from the impairment.

The main validity risks are interpretive:
1. The paper-level phase impairments (1-5°) are 23-37 dB below the signal, so the Table-2 comparison is almost a clean-data comparison except at 20-25 dB with δ=5°.
2. An unidentifiable phase ramp imposes an irreducible Pd ceiling on impaired banks.
3. The π/6 separation is in angle space, so ~3 % of test samples contain endfire-aliased, near-unresolvable pairs, and ~6-10 % of angles sit in an extreme-precision endfire regime.
4. The impairment model is exactly the separable structure v2 is designed to invert. It is redrawn per sample, and gain banks carry a +0.7-1.2 dB SNR bonus.
5. The v1 seed collision makes `gain_g2_snr0` and `gain_g0.5_snr15` identical scene sets.
6. The training stream is non-reproducible (`os.urandom`).
7. The reproduction docstring attributes code parameters (π/6, L 1-9, SNR −15..24) to the paper, whose text states SNR −10..25 and L 1..10 and does not mention π/6.

---

## 10. Re-verification pass (second run of this audit)

I re-ran `s1_frozen.py`, `s2_v2banks.py` and `s3_stats.py` and re-checked two claims directly against source. The results are unchanged.

- **C1/C2 (FACT):** `eval_bank.npz` regenerates bit-identically (max diff 0.0) with both the original and the reproduction `validation_data_generator`. All 7 v2 banks regenerate bit-identically from their seeds.
- **C11/C12/V1 (FACT):** Per-entry noise power matches $10^{-\mathrm{SNR}/10}$ within about 1 %. `T2_d5` distortion-to-noise is −33.1 dB at −10 dB SNR and +2.1 dB at 25 dB. `tune_g3` shows E[g_r² g_t²] = 1.178 and mean signal power 1.13-1.23 (the gain SNR bonus of V6).
- **I3 (FACT):** In `MANIFEST.json`, `gain_g2_snr0` and `gain_g0.5_snr15` share seed 2020. Seeds 1000 and 1015 are each shared by `phase_d0_snr*` and three `nuis_*` banks.
- **V3 (FACT):** Lloria Sec. IV-B (`scratch/a01_lloria.txt` lines 655-665) says SNR in [−10, 25] dB and L from 1 to 10, with angles uniform in [0, π]. It does not state a π/6 separation. The authors' code `DL_DOA/src/tvt_data_generation_v3.py:364-365,401` uses `randint(1,10)`, `randint(-15,25)` and `sep = np.pi/6`. Paper text and released code therefore differ, and v2 follows the code.
