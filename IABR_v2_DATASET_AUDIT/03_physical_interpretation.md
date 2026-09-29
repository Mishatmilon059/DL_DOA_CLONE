# Agent 03: Physical Interpretation Audit of the IABR-Net v2 Dataset-Generation Pipeline

**Scope.** This audit covers only the physics of the data generator: signal model, conventions, labels, impairments, noise and SNR, and the saved banks. It does not assess model accuracy.
**Method.** I read all the code myself. I checked conventions numerically with scratch scripts in `D:\ai_ml_project\IABR_v2_DATASET_AUDIT\scratch\`:
- `a03_phys_checks.py`: toy cases, conventions, energy, end-fire, separation, impairment ramp, upsampling, heat target
- `a03_bank_checks.py`: statistics of the saved banks and the frozen bank, plus a label-convention fit
- `a03_crlb_ceiling.py`: a Pd ceiling limited by the CRLB and by end-fire geometry
- `a03_rsa_bias.py`: bias of the angle sampler by placement order

I modified no project file.
**Re-verification (second pass):** all four scripts were re-run from scratch and reproduced every number quoted below (codebook unitarity ~2e-15, toy-case argmax = label, no-conj alternative moves the peak, SNR/energy bookkeeping, end-fire 22.6 %, torus-separation table, ramp-bias 0.0055/0.011/0.027, CRLB ceiling table, sampler bias). The paper-text contradictions ([0,2π] in Sec. II vs [0,π] in Sec. IV) were re-checked in `scratch\tvt.txt`.
**Labels used:** FACT (read directly from code or paper), DERIVATION (math or numerical result I computed), INFERENCE, ASSUMPTION, CONCERN, VERIFIED BUG.

---

## 0. Executive summary

1. **No sign, conjugation or transposition bug** in the v2 simulator, the label mapping (`angles_to_cells`, `heat_targets`), the antenna-domain transform (`to_ant`), the NOMP atoms, or the angle inversion (`uv_to_angles`). All were checked numerically on toy cases (DERIVATION). Dropping the conjugate on $a_t$ would visibly move the peak, and the code does not drop it. The frozen "official" bank follows the same convention: with the true angles, the LS residual equals the nominal noise variance at every SNR (DERIVATION).
2. **What "SNR" means.** SNR is $1/\sigma_n^2$ per entry of $Y$, with $\rho=1$ and $\mathbb E\|H\|_F^2 = n_t n_r$. It is a per-element SNR measured *before* array gain. The best beam pair of a path sees about $+24\,\text{dB} - 10\log_{10}L$ more, and coherent processing of all 256 pilots gains 24 dB. So "−10 dB" is not a low-SNR regime in the usual link-budget sense. The SNR label is also the *nominal ensemble* SNR: the realised $\sum|\alpha_l|^2$ spreads the per-sample SNR by about ±3–4 dB for L=3 and by about 13 dB (p10 to p90) for L=1 (DERIVATION).
3. **The end-fire geometry dominates the angle-domain metric.** Angles are drawn uniformly in $[0,\pi]$, but spatial frequency $u=\pi\cos\psi$ is uniform in the beamspace. As a result, 22.6 % of angles lie within half a beam cell of the $\pm\pi$ alias edge. There $\psi\approx0$ and $\psi\approx\pi$ are *physically indistinguishable*: $|a(2^\circ)^H a(178^\circ)|=0.9998$. Consider an idealised single-path estimator that reaches the CRLB. Scored with the paper's 1° criterion, it reaches only about 0.95 per-component Pd at 15 dB and 0.985 at 25 dB. Restricting angles to $[20^\circ,160^\circ]$ raises this to 0.994 and 0.999 (DERIVATION). Part of the high-SNR Pd gap in the results is therefore geometry, not estimator quality (INFERENCE).
4. **The minimum separation of π/6 is Euclidean in the (φ, ψ) *angle* plane.** It has no direct physical resolvability meaning. After mapping to the spatial-frequency torus, 3.3 % (L=3), 15.7 % (L=6) and 35 % (L=9) of samples still contain a pair closer than one DFT beam cell. The constraint also allows alias-coincident pairs, such as (2°, 2°) and (178°, 178°) (DERIVATION, CONCERN).
5. **The impairment model is a static, beam-independent, diagonal per-antenna complex error on both TX and RX codebooks.** It matches the base code's phase-error model exactly; the gain error is a v2 addition. This is a *calibration-offset* model. The phase-state-dependent error that datasheet "RMS phase error" refers to (it varies with the programmed phase, i.e. with each codebook column) is not modelled, and neither is mutual coupling. Because of the diagonal model, the impairment factorises exactly as $\tilde H = D_r^* H D_t$ in the antenna domain, and IABR-Net v2's IABC and self-calibration are built on exactly this structure. The method is therefore evaluated on the one impairment family it was designed to invert (CONCERN, High for any claim of hardware robustness).
6. **The linear-phase-ramp part of the impairment cannot be identified, and it biases the angle labels.** At $\delta_{max}=5^\circ$, 2.7 % of angle components in `T2_d5` have an *apparent* angle more than 1° from the label, and 1.15 % flip across the end-fire alias. So even a perfect estimator of what the data contain would have a Pd ceiling of about 0.97 at δ=5° (about 0.989 at 2°, about 0.995 at 1°) (DERIVATION).
7. **Path power profile.** $\alpha_l\sim\mathcal{CN}(0,1/L)$ i.i.d. gives equal-mean-power Rayleigh paths with no LoS/Rician dominance and no delay or path-loss structure. For L=3 the weakest path is a median 8 dB below the strongest (p10: 16 dB); for L=9 the medians are 15 dB and p10 24 dB. Every path, however weak, gets a unit-peak label. This is physically optimistic and label-noisy for large L (CONCERN, Medium).
8. **Noise model.** White noise is added after combining. This is exact for phase-only errors, because the codebook stays unitary. With gain errors (a v2 addition) it is an approximation, because noise generated before an erroneous gain stage would be scaled and correlated across beams. Gain errors also raise the signal power: +0.68 dB at 3 dB, so the effective SNR of `tune_g3` is above its label (DERIVATION, Low).

---

## 1. What the authors actually implemented

### 1.1 v2 simulator (`D:\ai_ml_project\IABR_v2_extracted\IABR_v2\iabr2_sim.py`)

| Item | Implementation | Ref |
|---|---|---|
| Arrays | ULA, $N_T=N_R=16$, $P=Q=16$ | l.11 (FACT) |
| Steering | $a_r[n]=e^{-j\pi n\cos\psi}/\sqrt{N_R}$, $a_t[m]=e^{-j\pi m\cos\phi}/\sqrt{N_T}$ | l.45–46 (FACT) |
| Channel | $H=\sqrt{N_TN_R}\sum_l\alpha_l a_r(\psi_l)a_t^H(\phi_l)$ (conj on $a_t$) | l.47 (FACT) |
| Codebooks | `F_IDEAL`, `W_IDEAL` from `DG.beamforming_vector_generation_P/Q` | l.12–13 (FACT) |
| Impairment | $\varepsilon\sim U(-\delta,\delta)$ per element, $20\log_{10}g\sim U(-\gamma,\gamma)$ dB, $D=g\,e^{j\varepsilon}$, $W=D_rW_0$, $F=D_tF_0$ (same on every column), drawn independently for TX and RX and per sample | l.49–52 (FACT) |
| Observation | $Y = W^H H F + Z$, $Z_{qp}\sim\mathcal{CN}(0,10^{-\mathrm{SNR}/10})$, i.i.d. | l.53–56 (FACT) |
| Paths | `DG.generate_points(L, π/6)` → (φ, ψ) ∈ [0,π]², α ~ CN(0,1/L) sorted by \|α\| descending, assigned in placement order | l.28–38 (FACT) |
| Labels | `angles_to_cells`: row $=n\cdot\mathrm{wrap}(-\pi\cos\psi)/2\pi$, col $=n\cdot\mathrm{wrap}(\pi\cos\phi)/2\pi$; `heat_targets`: 32×32 wrap-around separable Gaussian, σ = 1 cell, peak 1 at the *rounded* cell, max over paths; `impairment_targets`: (phase, log-gain) of $D_r^*$ rows / $D_t$ cols with mean removed (both) and linear ramp removed (phase) | l.71–104 (FACT) |
| Training draw | L ~ U{1..9}, SNR ~ U{−15..24} (integers), 20 % clean, else δ ~ U(0,8°), γ ~ U(0,3 dB) | l.107–117; CFG `notebook_src.py` l.84–86 (FACT) |

### 1.2 Test and tuning banks (`notebook_src.py`)
- `make_bank` (l.356–368) uses the same `sample_paths` and `simulate`. `T2_d{1,2,5}`: L=3, SNR −10..25 step 5, 1000 per SNR, phase only, seeds 9101/9102/9105 (l.372). Tuning banks: seeds 9901–9904 (l.373–378). FACT; confirmed by the saved keys (`phase_deg`, `gain_db`, `seed` inside each npz).
- 'official' = frozen `eval_bank.npz`. `scripts\generate_frozen_banks.py` builds it with the *original* `DL_DOA/src/tvt_data_generation_v3.validation_data_generator` (L=3, SNR −10..25, P=nt=16, seed 42, 1000 per condition). Stored keys: data (8000,64,64,2), feat (8000,2,3) = [ψ; φ], meta = [L, SNR, P, nt], sigma = 0.07, M = 256, seed = 42 (FACT, from the script header and the loaded file). Whether this exact script produced the file on disk is for Agent 04 to settle. Physically, the file is fully consistent with it (§3).
- The v1 robustness banks come from the same simulator (v1 suite `simulate`, copy in `scratch\v1_suite_src.py` l.306–). `make_separation_bank` places path 2 at $(\psi_1+d,\phi_1+d)$ with angles in [20°,160°] (l.459–475) (FACT).

### 1.3 Base-paper generator (`dldoa_dataset_generation.py`, identical in physics to `DL_DOA/src/tvt_data_generation_v3.py`)
- `ev` l.111–131, `beamforming_vector_generation_P/Q` l.134–205. Phase error: `np.random.uniform` (global RNG), one vector per call, the same on all columns (FACT).
- `generate_noise(1.0, SNR, …)` l.208–239; `generate_channel_v2` l.249–279; GT l.313–372 (not periodic, grid $[-3\sigma, 2\pi+3\sigma)$) (FACT).

---

## 2. Component-by-component physical interpretation

### 2.1 ULA and half-wavelength spacing
- FACT: the phase $\pi n\cos\theta = 2\pi (d/\lambda) n\cos\theta$ with $d=\lambda/2$, and $\theta$ is measured from the **array axis** (θ = 90° is broadside, θ = 0 or π is end-fire). Paper Eqs. (2)–(3) match (tvt.txt l.182–186).
- DERIVATION: with $d=\lambda/2$ the spatial frequency $u=\pi\cos\theta\in[-\pi,\pi]$ exactly fills one period, so no grating lobe appears inside the visible region. The two end-fire directions coincide, however: $a(0)=a(\pi)$ exactly, and $|a(2^\circ)^Ha(178^\circ)|/N = 0.9998$ (`a03_phys_checks.py` §2).
- INFERENCE: other physics not modelled: isotropic elements (real patch elements lose roughly 10+ dB toward end-fire), mutual coupling, narrowband far field with no beam squint, a single 1-D azimuth cut (no elevation, no cone-of-confusion effects), and a channel with no phase noise or CFO that stays static over the 256 pilot slots. This is standard for this literature and matches the base paper, but it limits any "realistic hardware" claim.

### 2.2 Angle range [0, π] and its end-fire implications
- FACT: the code draws φ, ψ ~ U[0, π] (`DG.generate_points` l.78–79). The paper is **internally inconsistent**: Sec. II says "uniform ∈ [0, 2π]" (tvt.txt l.232–233) and Sec. IV says "[0, π]" (tvt.txt l.703). The JSC paper also says [0, 2π] (jsc.txt l.123).
- DERIVATION: [0, π] is the physically correct, non-redundant range for a ULA, because cos θ = cos(2π−θ). Sampling from [0, 2π] would draw each spatial frequency twice. The code is right and the paper text is wrong.
- DERIVATION: uniform angle gives an arcsine density of $u$, heavily concentrated at ±π. With 16 beams, 22.6 % of angles fall within half a cell of the alias edge (within 20.4° of end-fire), 15.9 % within a quarter cell and 11.3 % within an eighth.
- DERIVATION (Jacobian): a 1° angle error corresponds to Δu = 0.14 of a beam cell at 90°, 0.071 at 30°, 0.025 at 10°, 0.013 at 5° and 0.006 at 2°. A fixed Δu = 0.01 rad error already puts 11.7 % of uniformly drawn angles beyond 1°, and in 2.5 % the estimate wraps to the opposite end-fire (error > 90°).
- DERIVATION (`a03_crlb_ceiling.py`, single path, 2-D tone CRLB $\mathrm{var}(u)=6/(\rho\,M N(N^2-1))$, $\rho=|\alpha|^2/\sigma^2$, $|\alpha|^2\sim\mathrm{Exp}(1/3)$, inter-path interference ignored). Per-component probability of an error of at most 1°:

| SNR (dB) | −10 | 0 | 10 | 15 | 20 | 25 | 40 |
|---|---|---|---|---|---|---|---|
| angles U[0°,180°] | 0.406 | 0.742 | 0.914 | 0.952 | 0.973 | 0.985 | 0.996 |
| angles U[20°,160°] | 0.485 | 0.853 | 0.980 | 0.994 | 0.998 | 0.999 | 1.000 |

  INFERENCE: the observed high-SNR saturation (NOMP 0.973 and IABR-v2 0.968 at 25 dB on the dldoa test set, `infer_dldoa_test.log`) sits close to this geometric ceiling. The Pd metric in angle space is dominated by end-fire paths that no ULA estimator can resolve to 1°. This is a property of the base paper's protocol, which v2 inherits, and it should be stated wherever Pd ceilings are discussed.

### 2.3 Sign and conjugation conventions (verified numerically)

| Check | Result | Label |
|---|---|---|
| `F_IDEAL[m,p]` $= e^{-j2\pi pm/16}/4$, `W_IDEAL[n,q]` $= e^{+j2\pi qn/16}/4$; both unitary (error ≈ 1e−15) | so $Y = W^HHF$ is the 2-D DFT of $H$ (DFT on rows and on columns, each scaled by 1/16) | DERIVATION |
| RX beam q looks at cos ψ = −2q/16 (wrapped); TX beam p looks at cos φ = +2p/16 | the asymmetry is intentional and matches paper Eq. (9): $\omega_q=-\pi\cos\bar\psi_q$, $\omega_p=\pi\cos\bar\phi_p$ | FACT/DERIVATION |
| Single-path toys (ψ, φ) = (60, 120), (100, 30), (10, 170), (90, 90) deg | argmax \|Y\| equals the rounded `angles_to_cells` label every time; peak \|Y\| = 16\|α\| on-grid | DERIVATION |
| Alternative without the conj on $a_t$, for (100°, 30°) | peak moves from column 7 to column 9, so the conj matters, and the code has it (l.47) | DERIVATION |
| `to_ant(Y)` = $W_0 Y F_0^H$ = H | error ≈ 1e−14; row phase step = −π cos ψ, column step = +π cos φ | DERIVATION |
| NOMP atom $a[n,m]=e^{-jnu+jmv}$, $u=\pi\cos\psi$, $v=\pi\cos\phi$ (`notebook_src.py` l.533–535) | fits H with relative error 1.5e−15 and returns exactly α | DERIVATION |
| `heat_peaks` inverse $u=-2\pi\,\text{idx}/G$ (l.594); FFT init $u=-2\pi k/NF$ (l.581); `uv_to_angles` (l.623–625) | round trip exact | DERIVATION |
| Evaluator `peaks_to_angles`: ψ = arccos(−f_row/π), φ = arccos(f_col/π) (TVT_Blob_Inference.py l.108–118) | consistent with the GT rows = AoA (−π cos ψ) and cols = AoD (π cos φ) | FACT |
| Impairment sign: $W=D_rW_0$ gives $W^H = W_0^H D_r^*$, so $\tilde H = D_r^* H D_t$; targets use $-\angle D_r$ (rows) and $+\angle D_t$ (cols) (l.83–89) | correct | DERIVATION |
| v2 phase model vs base code: DG multiplies each column by $e^{j\varepsilon}$ (l.168, l.204), the same as $D W_0$ | identical | FACT |

Minor FACT: for ψ = 90° exactly, `angles_to_cells` can return exactly `n` (16.00) because of the wrap of −0. Every consumer applies `% G`, so the value is harmless (`a03_phys_checks.py` §2).

### 2.4 TX/RX dimension consistency
- FACT/DERIVATION: $H\in\mathbb C^{N_R\times N_T}$ (einsum `bl,bln,blm->bnm`, n = RX, m = TX); $W\in\mathbb C^{N_R\times Q}$; $F\in\mathbb C^{N_T\times P}$; $Y\in\mathbb C^{Q\times P}$ with rows = RX (AoA) and cols = TX (AoD). All dimensions are 16, so a transposition bug would stay silent. I checked the conventions with asymmetric (ψ ≠ φ) toys, and the swapped-label hypothesis fails on the banks (§3). No transposition bug.
- FACT: `DG.training_data_generator` calls `generate_noise(1.0, SNR, P, Q)` with arguments (P, Q) where (Q, P) is meant (l.460). This is harmless because P = Q. The base paper also trains with P ∈ {16, 32} and n_t ∈ {16, 32}, which v2 does not (v2 uses P = 16 only). That is a distribution difference, not a physics error.

### 2.5 Channel normalisation $\sqrt{N_TN_R}$
- DERIVATION: with unit-norm steering vectors, $\mathbb E|h_{nm}|^2=\sum_l\mathbb E|\alpha_l|^2=1$, so $\mathbb E\|H\|_F^2=N_TN_R$. This is the standard normalisation. Unitary codebooks give $\|G\|_F^2=\|H\|_F^2$. Measured mean per-entry $|G|^2$ is 0.995/0.997/0.999 for L = 1/3/9, against the expected value of 1.
- DERIVATION: in the antenna domain every entry of $\tilde H$ has amplitude $|\alpha_l|$ per path and noise variance σ² (a unitary transform preserves white noise).

### 2.6 Path gains $\alpha\sim\mathcal{CN}(0,1/L)$, sorted strongest first
- FACT: the model and the sorting follow the base code and the paper (tvt.txt l.230–231, l.702–703).
- DERIVATION: because the sum over paths is permutation-invariant, sorting has no effect on Y. It only sets label order. It affects the data only through the sampler: the strongest path is the *first placed point*. The sequential rejection sampler slightly pushes later (weaker) points toward the edges of the square. For L = 9, the probability of lying within 20° of end-fire rises from 0.221 (strongest) to 0.260 (weakest); for L = 3 it goes from 0.223 to 0.230 (`a03_rsa_bias.py`). Severity: Low.
- DERIVATION (power spread, `a03_phys_checks.py` §4). Median (p10) ratio of weakest to strongest path power: L=2: −4.8 (−12.8) dB; L=3: −8.0 (−16.3) dB; L=6: −12.8 (−21.1) dB; L=9: −15.3 (−23.6) dB. The weakest path's coherent 256-pilot gain $10\log_{10}(256|\alpha_{min}|^2)$ has a median of 3.4 dB and a p10 of −4.8 dB for L=9. At SNR = −15 (the training minimum), such a path is 10–20 dB below noise even after full coherent integration, yet `heat_targets` still gives it a unit peak (l.100–103).
- INFERENCE (realism): measured mmWave channels (NYUSIM, 3GPP 38.901) show a dominant LoS or strongest cluster with power decaying by delay, often Rician K of 5–10 dB, and clusters with angular spread (not single rays). I.i.d. equal-mean-power Rayleigh rays are a stylised sparse model. It is acceptable for reproducing the paper, but it is not "realistic mmWave". Severity: Medium for the realism claim; label noise at large L.
- DERIVATION: the realised per-sample SNR offset $10\log_{10}\sum|\alpha|^2$ has p10/p50/p90 of −9.8/−1.7/+3.6 dB for L=1, −4.4/−0.5/+2.5 dB for L=3 and −2.2/−0.2/+1.6 dB for L=9. Any "per-SNR" curve for small L therefore averages over a wide spread of true SNRs.

### 2.7 Path sparsity L ∈ {1..9}
- FACT: `make_batch` draws L ~ U{1..9} (l.108), matching the original code (tvt_data_generation_v3.py l.364). The paper text states "L = 1 to 10" and "SNR −10 dB to 25 dB" for training (tvt.txt l.422–424), which differs from the code's 1..9 and −15..24. The docstring of `dldoa_dataset_generation.py` (l.19–20) says the paper specifies −15..24. The text extraction I have does not contain Table I, so whether Table I says otherwise: *Insufficient evidence*. v2 follows the code, not the paper text.
- DERIVATION: 4L real parameters (up to 36) against 512 real observations, so the problem is well determined. Resolvability, not identifiability, is the limit (§2.8).

### 2.8 Minimum separation π/6 in the (φ, ψ) plane
- FACT: `generate_points` rejects a candidate when `math.hypot(Δφ, Δψ) < δ`, i.e. a 2-D **Euclidean distance in radians of angle**, not wrapped and not in spatial frequency (DG l.81–84; original l.37). Bank minima are 30.00–30.13°, so the constraint is honoured.
- DERIVATION: resolvability is set by distance in $(u,v)$ on the $2\pi$-torus relative to the beam cell $2\pi/16$. Because $\Delta u\approx\pi\sin\theta\,\Delta\theta$, 30° is about 4.1 cells at broadside but only about 1.07 cells at end-fire. Opposite end-fires alias, so the constraint allows (2°, 2°) and (178°, 178°), whose Euclidean distance is 249° but whose spatial frequencies essentially coincide.
- DERIVATION (`a03_phys_checks.py` §6). Minimum pairwise torus distance, in 16-beam cells:

| L | p5 | median | P(<1 cell) | P(<0.5 cell) | P(some pair inside a 1×1-cell box) |
|---|---|---|---|---|---|
| 2 | 2.19 | 6.21 | 0.010 | 0.003 | 0.014 |
| 3 | 1.21 | 4.13 | 0.033 | 0.010 | 0.042 |
| 6 | 0.51 | 2.02 | 0.157 | 0.049 | 0.197 |
| 9 | 0.32 | 1.27 | 0.353 | 0.115 | 0.417 |

  CONCERN (Medium): the constraint does not guarantee resolvable paths. For L ≥ 6 a large share of training and tuning samples (`tune_L6`) contain sub-Rayleigh pairs, and the heatmap target (σ = 1 cell on a 32-grid, max-merged) asks for two separate peaks there. The v1 `sep_*deg` banks define separation differently again (both AoA and AoD offset by d, angles in [20°,160°]), so e.g. "sep 5°" spans 0.24–0.7 cells depending on location (DERIVATION from `make_separation_bank`). Any "resolution vs separation" statement should be made in cell units, not degrees.

### 2.9 Hardware impairment model
- FACT: the phase error is per antenna, i.i.d. $U(-\delta,\delta)$, the same on every codebook column (a static offset in each element's path), drawn independently for TX and RX and freshly per sample. This matches the base code (tvt_data_generation_v3.py l.78–91, l.104–117). Both papers describe the error as "altering the array responses in Eqs. (2)–(3)" (tvt.txt ~l.515–520; jsc.txt l.215–217). The code applies it to the *beamformer*, which is the physically right place. It is mathematically equivalent to perturbing the steering vector by $D^*$, and distributionally equivalent because ε is symmetric (DERIVATION).
- FACT: the gain error $20\log_{10}g\sim U(-\gamma,\gamma)$ dB is a v2/v1-suite addition that neither paper has. The v1 suite marks it as an [ASSUMPTION] (v1_suite_src.py l.255–258).
- DERIVATION: RMS phase error = δ/√3, i.e. 0.58°, 1.15°, 2.89° and 4.62° for δ = 1, 2, 5, 8°. RMS gain error for γ = 3 dB is 1.73 dB.
- INFERENCE, from published 28 GHz beamformer ICs: RMS phase error about 1–3.2° and RMS gain error about 0.3–0.6 dB for 6-bit shifters ([Sensors 2024, GaN 6-bit PS](https://doi.org/10.3390/s24041087), [PMC10346209, 28 GHz CMOS BFIC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346209/), [8-ch 3–28 GHz RX BF](https://www.researchgate.net/publication/397278685_An_8-Channel_3-28_GHz_High-Linearity_Phased-Array_Receive_Beamformer_With_Gain_Compensation_for_6G_FR3_Systems)). So δ ≤ 5° is realistic to pessimistic, δ = 8° (the training maximum) is beyond typical ICs, and γ = 3 dB is about 3× worse than current ICs.
- CONCERN (High, realism and method validity): datasheet "RMS phase error" measures the **state-dependent** error across the programmed phase states. In a real array that error differs per codebook column, $W_{nq}=w^0_{nq}g_{n}(\theta_{nq})e^{j\varepsilon_n(\theta_{nq})}$. The simulator models only the beam-independent part (calibration offsets from feed lines, LNA/PA mismatch). With the beam-independent diagonal model the distortion is exactly $\tilde H=D_r^*HD_t$, a rank-preserving row and column scaling. IABR-Net v2's IABC (`notebook_src.py` l.496–507) and `selfcal` (l.597–620) estimate and invert precisely that structure. State-dependent errors, quantisation of a non-DFT codebook, mutual coupling (non-diagonal) or element-pattern errors would break the factorisation. The tests therefore measure robustness only inside the model's own impairment class.
- DERIVATION: for a 16-element DFT codebook all weights are multiples of 22.5°, so a phase shifter with 4 or more bits reproduces it exactly and quantisation error is physically absent here. That is consistent with not modelling it, but it would not hold for oversampled or off-grid codebooks.
- FACT/INFERENCE: an error that is static within a sample but redrawn per sample models an *ensemble of uncalibrated devices*, not one device over time. This is consistent with the base code's `validation_data_generator` (errors redrawn per sample).
- **Unidentifiable components (the labels).** `project_impairment` removes the mean (it trades off against ∠α or |α|) and the linear phase ramp (it trades off against a common shift of all $u_l$ or $v_l$) (l.76–80). This is correct identifiability reasoning (DERIVATION). The consequence is that the ramp part *moves the apparent angles* while the labels keep the true angles. The LS ramp slope has std 0.157°/element at δ = 5 (0.007 cell); it is the same for all paths on one side, and it cannot be removed by any estimator. On the actual `T2_d*` angles (both AoA and AoD, per component, `a03_bank_checks.py`):

| δ_max | fraction of components with apparent − label > 1° | of which alias flips (> 90°) |
|---|---|---|
| 1° | 0.0055 | 0.0048 |
| 2° | 0.0110 | 0.0072 |
| 5° | 0.0271 | 0.0115 |

  CONCERN (Medium): the Table-2-style banks carry an intrinsic label ceiling of about 0.97–0.995 per component, concentrated at end-fire. Comparisons with the Meneses-Albalá Table 2 inherit this ceiling. The 87.5 % of the phase-error variance that *is* identifiable is what IABC can learn.
- DERIVATION: gain errors raise the mean signal power by $\mathbb E[g^2]$ per side (+0.34 dB per side at 3 dB, +0.68 dB total; measured +0.679 dB). The effective SNR of gain-impaired banks is therefore slightly above the label.

### 2.10 Noise model and SNR definition
- FACT: $Z_{qp}$ i.i.d. $\mathcal{CN}(0,\sigma^2)$, $\sigma^2=10^{-\mathrm{SNR}/10}$ (l.55–56), matching `generate_noise(var_alpha=1)` and the paper's "SNR = ρ/σ_n² with ρ = 1" (tvt.txt l.230–236, l.419–420).
- DERIVATION: **what the SNR is taken relative to.** With $\mathbb E\sum|\alpha_l|^2=1$ and unitary codebooks, $\mathbb E|G_{qp}|^2$ averaged over all 256 beam pairs is 1. So SNR = (mean per-entry signal power of Y) / (noise power per entry), which equals the **per-antenna-pair (pre-beamforming) SNR** $\mathbb E|h_{nm}|^2/\sigma^2$. It excludes array gain. At the best on-grid beam pair a path has $|G|^2 = N_TN_R|\alpha_l|^2$, a gain of $10\log_{10}(256/L)$ = 19.3 dB for L=3 (median measured peak-to-mean 19.8 dB). Coherent processing of all $QP=256$ pilots gives $10\log_{10}256=24$ dB of processing gain. The nominal −10 dB point is therefore about +14 dB integrated SNR for the whole channel.
- DERIVATION: independent noise per (q, p) matches a physical sequential beam sweep with one RF chain: 256 pilot slots, independent thermal noise in each. The noise does not depend on the TX beam, which is correct. With phase-only errors, $W=D_rW_0$ still has orthonormal columns, so "white noise after combining" is exact.
- CONCERN (Low): with RX gain errors, noise created at the LNA before the erroneous gain stage would be $W_0^HD_r^*n$, with covariance $W_0^H|D_r|^2W_0$: scaled by mean $g^2$ and correlated across beams (the off-diagonals are the DFT of $g_n^2$). The simulator keeps it white and unit-scaled. The effect is small at γ ≤ 3 dB, but the "gain" banks are slightly optimistic.
- FACT: `simulate(..., want_clean=True)` returns `Y_clean` = ideal codebooks **plus the same noise** (l.59), i.e. impairment-free but not noise-free. `make_batch` does not use it. Naming caveat only.
- FACT: SNR draws are integers (`rng.integers`, l.109), as in the base code.

### 2.11 Label construction (physical meaning)
- DERIVATION: the v2 `heat_targets` grid is periodic (wrap distance, l.98–99), which matches the $2\pi$-periodic beamspace. It is used together with circular padding in the network. Broadside ($u=0$ ↔ ψ = 90°) sits at cell 0 and wraps correctly: a path at 89° gets a peak at row 0 with neighbours at rows 31 and 1 (value 0.607 each).
- CONCERN (Low–Medium, base-paper GT only, i.e. the U-Net baseline): DG `generate_gt` (l.354–368) is **not periodic**. Frequencies are wrapped to [0, 2π) and the Gaussians drawn on the extended line $[-3\sigma,2\pi+3\sigma)$ without images. Broadside in either dimension (ψ or φ ≈ 90°) lands at the image *edges and corners*, and a path at $u=+0.01$ versus $u=-0.01$ rad appears on opposite sides of the image although it is physically adjacent. The U-Net's labels inherit this. v2's own labels do not.
- FACT: the v2 heat peak sits at the rounded 32-grid cell (quantisation ±π/32 rad in u). This is a coarse detection target, and sub-cell precision is left to the NOMP decoder. It is physically reasonable, and it means the heatmap alone cannot reach the 1° criterion (consistent with "network only" Pd ≈ 0.43 at high SNR in `infer_dldoa_test.log`) (INFERENCE).

### 2.12 Nearest-neighbour upsampling (U-Net input only)
- DERIVATION: `scipy.ndimage.zoom(order=0)` with 16 → 64 does **not** make uniform 4× blocks. The per-row source replication counts are [3,4,4,4,4,5,4,4,4,4,5,4,4,4,4,3] (`a03_phys_checks.py` §8), so the paper's 64×64 input has non-uniform beam-cell widths. This is the base paper's preprocessing and has no physical meaning. `downsample16` inverts it exactly: I checked `upsample64(downsample16(x)) == x` over all 8000 frozen samples (max error 0.0), so the v2 16×16 input loses nothing.

---

## 3. Bank-level physical verification (`a03_bank_checks.py`)

For every bank and SNR (300 samples per group) I fitted LS gains in the antenna domain using the **stored label angles**, then compared the residual power with $10^{-\mathrm{SNR}/10}$:

| Bank | residual / nominal σ² (typical) | Interpretation |
|---|---|---|
| official (frozen) | 1.00–1.03 at every SNR (25 dB: 0.0032 vs 0.0032) | labels and data follow exactly the v2 convention (ψ = feat[:,0], φ = feat[:,1]), and noise is exactly $1/\sigma^2$ per entry, **so the SNR definition is identical to v2's** |
| tune_clean | 1.00–1.03 | same |
| T2_d1/d2 | ≈ 1.0–1.25 at 25 dB | small impairment mismatch |
| T2_d5 | 25 dB: 0.0082 vs 0.0032 → excess 0.0050 | the prediction for **both-sided** phase error, $2\cdot\frac{15}{16}\mathrm{var}(\varepsilon)=0.0048$, matches. One-sided would give 0.0024. This confirms TX and RX are both impaired |
| tune_g3 | 15 dB: 0.120 vs 0.032 → excess 0.089 | the prediction for both-sided 3 dB log-uniform gain, $\approx2\,\mathrm{var}(\ln g)=0.080$, matches. Mean \|Y\|² is inflated (+0.7–0.9 dB) |

Other checks:
- The swapped (ψ↔φ) label hypothesis places the argmax within 1 cell of the path-0 label in only 6–19 % of cases, against 80–85 % for the correct one. The 80–85 % is below 100 % because of off-grid scalloping (up to about 3.9 dB per dimension) and multipath, so the strongest *path* is not always the brightest *beam* (DERIVATION).
- LS gains come out sorted strongest-first in 98–99 % of cases at 20–25 dB (official, T2), which confirms the sorted-α column ordering in the stored labels.
- Angle marginals: mean ψ ≈ 90°, and P(ψ < 20°) = 0.113–0.116 against 0.111 expected, i.e. uniform on [0, π] as coded. The minimum Euclidean separation is ≥ 30.00° in every bank.
- The banks store neither α nor $D_r$, $D_t$ (keys: Y16, psi, phi, L, snr, phase_deg, gain_db, seed). Impairment targets and path powers for test banks can only be reconstructed by re-simulating from the seed (FACT; limits post-hoc audit).

---

## 4. Ranked concerns (physics only)

| # | Severity | Label | Concern |
|---|---|---|---|
| C1 | High | CONCERN | The impairment is only static, beam-independent and diagonal (calibration offset). Real phase-shifter errors depend on the phase state (they vary per codebook column), and coupling is non-diagonal. v2's IABC and self-calibration invert exactly the simulated structure, so robustness claims hold only within this model class. |
| C2 | Medium | DERIVATION/CONCERN | End-fire geometry with U[0, π] angles and the 1° angle criterion. About 22 % of components sit near the alias edge, and the CRLB-limited Pd ceiling is 0.95 at 15 dB and 0.985 at 25 dB. Pd saturation should not be read as estimator weakness. |
| C3 | Medium | DERIVATION | The unidentifiable phase ramp biases labels: a per-component ceiling of 0.9945/0.989/0.973 at δ = 1/2/5°, about half of it from alias flips. |
| C4 | Medium | CONCERN | The π/6 separation is Euclidean in angle, not in (u, v). It does not ensure resolvability: 35 % of L = 9 and 16 % of L = 6 samples have a sub-cell pair, and alias-coincident pairs are allowed. |
| C5 | Medium | INFERENCE | Equal-mean-power i.i.d. Rayleigh rays: no LoS or cluster structure, weakest paths 15–24 dB down at L = 9, and unit-peak labels for sub-noise paths. |
| C6 | Medium | DERIVATION | SNR is the per-element, pre-array-gain nominal SNR. It must be reported as such (+24 dB processing gain) when comparing with papers that define post-beamforming SNR. The per-sample realised SNR spread is large for small L. |
| C7 | Low–Med | DERIVATION | The base-paper GT (U-Net baseline) is non-periodic, with broadside at the image edges and corners. v2's heat target is periodic and correct. |
| C8 | Low | DERIVATION | With gain errors, the noise is not propagated through the erroneous RX gains, and the signal power is inflated by +0.68 dB at 3 dB. |
| C9 | Low | DERIVATION | The sequential rejection sampler makes weaker paths slightly more likely near end-fire (L = 9: 0.221 → 0.260). |
| C10 | Low | FACT | The paper text contradicts itself and the code: angles [0, 2π] (Sec. II) vs [0, π] (Sec. IV); training SNR −10..25 and L 1..10 (text) vs −15..24 and 1..9 (code). The code's choices are the physically sensible ones. The [0, 2π] statement is physically wrong for a ULA. |
| C11 | Info | FACT | The zoom order-0 upsampling is non-uniform (3/4/5 replications). It is a base-paper preprocessing quirk, handled exactly by `downsample16`. |

**VERIFIED BUGS:** none found in the physics of the v2 generator. The conjugation, sign, transposition, normalisation and unit conventions (degrees → radians via `np.deg2rad`, dB → amplitude via `10**(x/20)`, SNR dB → power via `10**(-x/10)`, complex noise with variance split /2 per quadrature) are all correct and verified numerically.

## 5. Insufficient evidence
- The contents of Table I of the TVT paper (the training SNR and L ranges quoted in the DG docstring). My text extraction does not include the table.
- The exact test set behind Meneses-Albalá Table 2 (their seeds and generator). The `T2_*` banks match their stated distribution (L = 3, δ ∈ {1, 2, 5}, both arrays, i.i.d. uniform) but not their realisations.
- `dldoa_dataset/test_*.npz` is not on this machine, so I could not verify its physics directly. `infer_dldoa_test.py` l.20 asserts it is a zoom-4 16×16 observation, and it would come from `DG.save_test_dataset`, which uses the same physics as the frozen bank.

## Sources
- [A 28 GHz GaN 6-Bit Phase Shifter MMIC with Continuous Tuning Calibration Technique (Sensors 2024)](https://doi.org/10.3390/s24041087)
- [A Multimode 28 GHz CMOS Fully Differential Beamforming IC for Phased Array Transceivers (PMC10346209)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346209/)
- [An 8-Channel 3–28 GHz High-Linearity Phased-Array Receive Beamformer With Gain Compensation](https://www.researchgate.net/publication/397278685_An_8-Channel_3-28_GHz_High-Linearity_Phased-Array_Receive_Beamformer_With_Gain_Compensation_for_6G_FR3_Systems)
