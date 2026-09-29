# 01 — Dataset Pipeline Reconstruction (IABR-Net v2)

Agent 01, Dataset Pipeline Reconstructor. Scope: everything that produces model inputs, labels/targets, training data and tuning/test banks for IABR-Net v2. Model accuracy is out of scope.

Label legend: **FACT** (read directly in code/data/paper), **DERIVATION** (follows by algebra from FACTs), **VERIFIED** (checked by a computation I ran), **INFERENCE** (a reasoned conclusion that is not proven), **ASSUMPTION** (an assumption made by the authors or by me), **CONCERN** (possible problem), **VERIFIED BUG** (a defect demonstrated by computation). "Insufficient evidence" is stated where it applies.

Verification scripts (all mine, read-only on project data) are in `D:\ai_ml_project\IABR_v2_DATASET_AUDIT\scratch\a01_*.py`. The test split of the dataset zip was extracted to `scratch\dldoa_dataset\`.

---

## 0. Headline results

| # | Result | Label |
|---|---|---|
| H1 | The **frozen official bank** `eval_bank.npz` was written by `D:\ai_ml_project\scripts\generate_frozen_banks.py` (`generate_eval_bank`), which calls the **original authors'** `DL_DOA/src/tvt_data_generation_v3.validation_data_generator(conditions=(L=3, SNR∈{-10..25 step 5}, P=16, nt=nr=16), examples_per_condition=1000, seed=42, sigma=0.07, M=256, amps='ones')`. Regenerating it with that function gives a **bit-identical** `data` (8000×64×64×2) and `feat` (8000×2×3). The project copy `dldoa_dataset_generation.validation_data_generator` also reproduces it bit-exactly. | VERIFIED (`a01_regen_official.py`: `data==eval_bank True feat== True` for both generators) |
| H2 | This is the same protocol and seed as the authors' `run_inference_and_metrics_unet/resnet`, which regenerate the test set on the fly with `seed=42`. So `eval_bank.npz` is, bit for bit, the authors' own published test set. The PyTorch U-Net port reproduces the published Pd to within 2·10⁻⁴ per SNR on it (`outputs/tables/V_unet_port.md`). | FACT (TVT_Blob_Inference.py:290-313) + VERIFIED (H1) + INFERENCE (same numpy PCG64 stream) |
| H3 | `dldoa_dataset/test_*` is **not** missing from the machine. It is inside `D:\ai_ml_project\dldoa_dataset-20260911T150638Z-1-001.zip` (Google-Drive export, 11 Sep 2026). Its test split equals `validation_data_generator(generate_test_conditions_fig5_fig6(), 1000, seed=7, sigma=0.07, M=256)`, bit-exact in data, features and meta. So it is the official protocol with **seed 7 instead of 42**: an independent replicate of the official test distribution, disjoint from `eval_bank`. | VERIFIED (`a01_seedscan.py` found seed 7; `a01_regen_seed7.py`: `data equal True, max abs diff 0.0, feat equal True, meta equal True`) |
| H4 | The zip's test split is the file set that `infer_dldoa_test.py` used. Re-running the log's DFT-SIC on it gives Pd 0.26017 at −10 dB and 0.92467 at 25 dB. The log shows 0.2602 and 0.9247. | VERIFIED (`a01_dftsic_check.py`) + INFERENCE (DFT-SIC is deterministic, so a match at 4 d.p. on two SNRs identifies the set) |
| H5 | The script that produced the zip is **not in the repo**. The zip has `train_meta.npz` and `val_features.npy`, while the current `save_training_dataset` saves no train meta, and the Colab run in `DLDOA_Colab.ipynb` wrote `val_features.pkl`. Its `val_meta` also differs from `generate_validation_conditions()` (seed 123). Only the test split's generator is established. | FACT (zip listing; dldoa_dataset_generation.py:750-833; DLDOA_Colab.ipynb cell 11/13 outputs) + VERIFIED (val_meta mismatch). Insufficient evidence for the train/val splits' generator. |
| H6 | Every bank used by v2 (official, dldoa-test, T2_d1/2/5, the 4 tune banks) is **label-consistent with its observation**. Least-squares fitting of the known-angle model gives a residual noise power per beamspace entry of $10^{-\mathrm{SNR}/10}$ within ±0.1 dB on clean banks, and a mean fitted signal power per entry of ≈0 dB. | VERIFIED (`a01_consistency.py`) |
| H7 | v1 bank seed collisions: `gain_g0.5_snr15` and `gain_g2_snr0` both use seed 2020 and have **identical angle sets** (ψ, φ). Both are in v2's `ROBUST_BANKS`. `phase_d0_snr{0,15}` share seeds 1000/1015 with the nuisance banks, which v2 does not use. | VERIFIED (`a01_banks.py`) |

---

## 1. Source inventory and data flow

```
                       physical parameters (L, SNR, delta_max, gamma_max, seed)
                                           |
      +------------------+-----------------+------------------+----------------------+
      | (a) BatchStream  | (b) v2 make_bank| (c) v1 suite     | (d) eval_bank (e) dldoa test
      | iabr2_sim.py     | notebook §5     | make_standard/   | original validation_data_generator
      | fresh, unseeded  | T2_*, tune_*    | separation/nuis  | seed 42 / seed 7
      +--------+---------+--------+--------+---------+--------+-----------+----------+
               |                  |                  |                    |
      sample_paths -> simulate (Y = W^H H F + Z, D_r,D_t impairments)    DG loop: alpha, points, H, G+Z
               |                  |                  |                    |  -> get_real_imag -> zoom x4 (64x64x2)
            to_ri               to_ri (Y16)         to_ri (Y16)          float32 data + feat [psi;phi] + meta
               |                  |                  |                    |
     y (B,2,16,16) + heat (B,32,32) + imp (B,32,2)   |              downsample16 (DOWN_IDX) -> Y16
               |                  +---------+--------+--------------------+
               v                            v
        training loop (§8)        load_bank -> Y16, Y (complex), psi, phi, L, snr
                                            |
                         net_forward: model.front (to_ant, IABC) ... / unet_predict: upsample64
                                            |
                          decoders + metrics(prediction, bank psi/phi/L)
```

Where things live:
- (a) `iabr2_sim.py:107-129` (also `notebook_src.py:242-264`, written by `%%writefile`).
- (b) `notebook_src.py:356-380`. Files in `IABR_v2\data\banks\`.
- (c) v1 suite `IABR_Net_Full_Test_Suite.ipynb` cell 9 (extracted to `scratch\v1_suite_src.py:448-537`). Files and sha256 `MANIFEST.json` in `IABR_Net_TestSuite\data\generated_banks\`. All 56 hashes match (VERIFIED). v2 uses 47 of the 56 (notebook_src.py:402-407; the 6 `nuis_*` and 3 `v1_clean_*` banks are excluded; the run log says "47 v1 robustness banks").
- (d) `scripts\generate_frozen_banks.py:44-68` → `DL_DOA\src\tvt_data_generation_v3.py:validation_data_generator`.
- (e) zip → `validation_data_generator(..., seed=7)`, read by `infer_dldoa_test.py:16-23`.

**Module identity (FACT/VERIFIED).** `iabr2_sim` imports `dldoa_dataset_generation` from `os.environ['IABR2_SUITE']` (iabr2_sim.py:8-9). On the run machine that was `PKG/new/new/IABR_Net_TestSuite` (notebook_src.py:35; run log line 9 shows the `new\new\...` path). All 5 copies of `dldoa_dataset_generation.py` on this machine have the same md5 (`e3c50716...`), and the `DL_DOA/src` files are identical between `D:\ai_ml_project\DL_DOA` and the TestSuite copy. The copy on the run machine is not available (Insufficient evidence). However, the run's physics check "simulator == original generate_channel_v2: 4.78e-15" passed (infer_dldoa_test.log:5).

---

## 2. The common physical model: stage by stage

The observation model is shared by (a), (b), (c) (vectorised `simulate`) and by (d), (e) (the per-sample loop in `validation_data_generator`). The two implementations are numerically identical for clean data: the run check gives 4.78e-15, and I checked that bank (d) regenerates exactly. Each stage below lists input, output, shape, operation, physical meaning, why it is performed, what is preserved, what is discarded, assumptions, where the output goes next, and a numerical example.

### S1. Path count, SNR and impairment level (sample-level parameters)

- **Operation (training, FACT, iabr2_sim.py:108-112):**
  - $L\sim\mathcal U\{1,\dots,9\}$ (`rng.integers(1, 9+1)`)
  - $\mathrm{SNR}\sim\mathcal U\{-15,\dots,24\}$ dB, integers
  - `clean` ~ Bernoulli(0.2)
  - if not clean: $\delta_{\max}\sim\mathcal U(0,8^\circ)$ and $\gamma_{\max}\sim\mathcal U(0,3\,\mathrm{dB})$, drawn independently; both are always present together
  - if clean: $\delta_{\max}=\gamma_{\max}=0$
- **Banks:** fixed per bank (e.g. T2_d5: L=3, δ=5°, γ=0, SNR∈{−10,…,25}, 1000 per SNR).
- **Physical meaning:** scene sparsity, operating SNR, hardware quality.
- **Assumptions:**
  - Training uses P=Q=nt=nr=16 only. The base paper's training draws P∈{16,32} (FACT, dldoa_dataset_generation.py:423; paper §IV-B).
  - The training ranges L 1..9 and SNR −15..24 match the authors' *code* (tvt_data_generation_v3.py `data_generation`). They do **not** match the paper text, which says "SNR in the range between −10 dB and 25 dB, … L = 1 to 10" (a01_lloria.txt:654-656). **CONCERN (documentation):** the project docstring (dldoa_dataset_generation.py:16-21) attributes the code values to "Paper Section IV-B & Table I". Table I's content could not be extracted from the PDF (Insufficient evidence for Table I).
  - Test SNR 25 dB is 1 dB outside the training range (DERIVATION).
- **Example (seed 0, B=4):** L=[8,6,5,3], SNR=[−3,−14,−12,−15], clean=[F,F,F,F], δ_max=[4.35, 7.48, 6.53, 0.02]°, γ_max=[2.57, 0.10, 2.19, 0.53] dB.
- **Next:** S2 (sample_paths), S6 (impairment), S7 (noise).

### S2. Angle generation (`DG.generate_points`, sep = π/6)

- **Input:** L, δ=π/6, rng. **Output:** L pairs (AoD φ, AoA ψ) ∈ [0,π]².
- **Operation (FACT, dldoa_dataset_generation.py:46-103):** sequential rejection sampling. Draw $(x,y)\sim\mathcal U[0,\pi]^2$ and accept if $\sqrt{(x-x_i)^2+(y-y_i)^2}\ge\pi/6$ for every accepted point. At most 10,000 attempts, otherwise a ValueError. Tuple order is `(x, y) = (AoD φ, AoA ψ)`. `sample_paths` maps `phi = p[0]`, `psi = p[1]` (iabr2_sim.py:37), and the DG loop uses the same order (dldoa_dataset_generation.py:449-451).
- **Physical meaning:** the uniform AoA/AoD of the scatterers, with a minimum **Euclidean separation in the (φ,ψ) angle plane** of 30°.
- **Why:** so that the L paths are resolvable (paper §II-A, "likely to be separated").
- **Preserved / discarded:** continuous angles in float64 inside the simulator. Stored labels are float32 (bank `psi`/`phi`, `feat`). The float32 quantisation of ~1e-7 rad is negligible against the 1° threshold (DERIVATION).
- **Assumptions / concerns:**
  - The π/6 separation is **code-only**: the paper text has no "π/6" and no minimum separation (grep over a01_lloria.txt). Paper §II-A says "uniform ∈ [0, 2π]" and §IV-B says "[0, π]", so the paper contradicts itself; the code uses [0,π] (FACT). Over [0,2π], ψ and 2π−ψ would give identical steering vectors. The code avoids that ambiguity (DERIVATION).
  - Sequential rejection sampling is not the exact uniform distribution on the constrained set. Configurations are slightly biased, more so for larger L (INFERENCE, standard random-sequential-adsorption property; not quantified).
  - The separation is enforced in **angle** space, not in beamspace. Because $u=\pi\cos\psi$ compresses near end-fire, paths separated by ≥30° can still be close in beamspace. VERIFIED (`a01_stats.py`):
    - official bank: 0.92% of samples have two paths < 0.5 cell apart (16-grid, wrapped); 3.15% < 1 cell; minimum 0.058 cell
    - training stream (L>1): 4.7% of samples < 0.5 cell apart; 15.0% < 1 cell
  - Wrap-around aliasing: ψ=2° and ψ=178° are 176° apart in angle but map to beamspace rows 16.0097 and 15.9903 (32-grid). Their spatial frequencies are $\pm\pi\cos 2^\circ$, which are 0.0038 rad apart on the circle (VERIFIED).
- **Example (seed 0, sample 0, L=8):** ψ = [97.5, 76.1, 22.4, 116.5, 69.1, 176.6, 129.9, 160.1]°, φ = [155.4, 53.9, 5.1, 120.7, 110.8, 179.5, 24.3, 87.5]°. The rest of the lmax=9 row is padded with NaN.
- **Next:** S3/S5 (steering vectors, channel), S10 (labels).

### S3. Array model and steering vectors

- **Operation (FACT, iabr2_sim.py:45-46; DG `ev`, :111-131):**
  $a_r(\psi)=\tfrac{1}{\sqrt{n_r}}[e^{-j\pi k\cos\psi}]_{k=0}^{15}$ and $a_t(\phi)=\tfrac{1}{\sqrt{n_t}}[e^{-j\pi k\cos\phi}]_{k=0}^{15}$. This is a ULA with half-wavelength spacing and unit norm, matching paper Eqs. (2)-(3).
- **Shapes:** `a_r`, `a_t`: (B, Lmax, 16) complex128.
- **Meaning:** the far-field narrowband phase progression. Spatial frequencies are $u=\pi\cos\psi$ and $v=\pi\cos\phi$, both in [−π,π].
- **Padded paths:** NaN angles are replaced by 0 via `nan_to_num`. Their α=0, so they contribute nothing (DERIVATION).
- **Next:** S5.

### S4. Path gains

- **Operation (FACT, iabr2_sim.py:35-36):** $\alpha_l=\sqrt{1/L}\,(n_1+jn_2)/\sqrt2$ with $n_i\sim\mathcal N(0,1)$, i.e. $\alpha_l\sim\mathcal{CN}(0,1/L)$ i.i.d. The gains are then sorted by descending $|\alpha_l|$, and the angles stay attached to the draw order.
- **Order issue:** in `sample_paths` the points are drawn *before* α. In DG's `validation_data_generator`, α is drawn first. So the same seed gives different samples in the two frameworks. This is irrelevant to correctness and only matters for reproducibility cross-checks (FACT; iabr2_sim.py:34-35 vs dldoa_dataset_generation.py:549-556).
- **Sorting:** the sort is a permutation of i.i.d. draws. It does not change the joint distribution of (α, angle) pairs, because the angles are i.i.d. and exchangeable apart from the separation constraint (DERIVATION).
- **Meaning:** Rayleigh fading with average total power $E\sum|\alpha_l|^2=1$.
- **Consequence (VERIFIED):**
  - strongest-to-weakest path power ratio in training: L=3 median 8.0 dB (p90 15.7 dB); L=9 median 15.3 dB
  - per-sample *realised* SNR at nominal 10 dB (L=3): median 9.53 dB, p10 5.76 dB, p90 12.57 dB; the mean over the linear values is 10.04 dB
  - the SNR label is therefore an **ensemble-average SNR**, not the per-sample SNR
- **Example (seed 0, sample 0):** |α|² = [0.380, 0.302, 0.251, 0.171, 0.129, 0.121, 0.033, 0.004].
- **Next:** S5.

### S5. Channel matrix

- **Operation (FACT, iabr2_sim.py:47):** $H=\sqrt{n_tn_r}\sum_l\alpha_l\,a_r(\psi_l)a_t^H(\phi_l)$, shape (B,16,16) complex128. `einsum('bl,bln,blm->bnm', alpha, a_r, conj(a_t))` is exactly the outer product $a_r a_t^H$ (the conjugation is on $a_t$). This matches paper Eq. (1) and DG `generate_channel_v2`: the check value 4.78e-15 was logged, and the v1 E0a check gives 3e-15.
- **Antenna-domain form (DERIVATION):** $H[n,m]=\alpha\,e^{-jn u}e^{+jm v}$ for a single path.
- **Preserved:** all path parameters. **Discarded:** nothing (noise-free).
- **Next:** S6.

### S6. Analog codebooks and per-antenna impairments

- **Ideal codebooks (FACT, DG :134-205; iabr2_sim.py:12-13):**
  - `cosp = angle(exp(j2πp/P))/π` = [0, .125, …, 1, −.875, …, −.125]
  - $F_{:,p}=a_t(\arccos(\text{cosp}_p))$, and $W_{:,q}=a_r(\arccos(\text{cosq}_q))$ with cosq = −cosp
- **VERIFIED:** $F=\overline{\mathrm{DFT}_{16}}/4$ and $W=\mathrm{DFT}_{16}/4$ (error ≈3e-15). Both are unitary (error 2.4e-15). Hence $W^HHF=\mathrm{fft2}(H)/16$ (error 2.3e-14) and $WYF^H=16\,\mathrm{ifft2}(Y)$ (error 2.2e-14).
- **Impairment (FACT, iabr2_sim.py:48-52):**
  - $\varepsilon_{r,n},\varepsilon_{t,m}\sim\mathcal U(-\delta_{\max},\delta_{\max})$ (degrees converted to radians)
  - $20\log_{10}g\sim\mathcal U(-\gamma_{\max},\gamma_{\max})$
  - $D=\mathrm{diag}(g\,e^{j\varepsilon})$, $W=D_rW_{\text{ideal}}$, $F=D_tF_{\text{ideal}}$
  - the draws are independent per antenna and per sample, and the same across all 16 beams (static hardware)
- **Phase part matches the original code** `beamforming_vector_generation_{P,Q}(error_deg)`: `F[:,p] = f_ideal * exp(j*phase_error)` applied to all beams (FACT; original :73-118; v1 E0a check "original phase error is per-antenna").
- **Gain part is a project extension.** The v1 suite explicitly marks the symmetric-uniform-in-dB law as an ASSUMPTION (v1_suite_src.py:256-258). The Meneses/Lloria papers model phase only (paper §2.2 / §IV-D).
- **Why:** to study robustness to phase-shifter and gain mismatch.
- **Physical meaning in the antenna domain (VERIFIED, 8.7e-15):** $\tilde H=W_{\text{ideal}}YF_{\text{ideal}}^H=D_r^{*}HD_t+W_{\text{ideal}}ZF^H_{\text{ideal}}$. Rows are distorted by $g_re^{-j\varepsilon_r}$ and columns by $g_te^{+j\varepsilon_t}$.
- **Unidentifiable components (DERIVATION):**
  - the common phase and common log-gain are absorbed into α
  - a **linear phase ramp** across the array is identical to a spatial-frequency shift ($u\to u+c$) that is common to all paths
  - the remaining identifiable part is exactly what `project_impairment` keeps (S12)
- **Consequence for the labels (VERIFIED, `a01_stats.py`):** the labels are the geometric angles, but the observation "sees" the angles shifted by the ramp. The least-squares slope has standard deviation $\sigma_c=(\delta_{\max}/\sqrt3)/\sqrt{\sum n_c^2}$ with $\sum n_c^2=340$:

  | δ_max | slope std (rad/element) | median AoA shift | fraction of components shifted > 1° |
  |---|---|---|---|
  | 1° | 0.00055 | 0.011° | 0.13% |
  | 2° | 0.00110 | — | 0.55% |
  | 5° | 0.00273 | 0.056° | 1.99% |
  | 8° | 0.00437 | — | 3.5% |
  | 15° | 0.00818 | — | 7.1% |

  The shift is amplified near end-fire, where $d\psi=du/(\pi\sin\psi)$. **CONCERN:** this is an irreducible label/observation mismatch. No estimator can remove it, so it caps achievable Pd on the impaired banks (T2_d5, oodphase) at a few percent below 1. It is inherent to the phase-error model (the paper's Table 2 has it too), not a code bug.
- **Model-mismatch floor (VERIFIED):**
  - theory: $2\delta_{\max}^2/3$ relative to signal, i.e. −36.9 / −30.9 / −22.9 dB for δ = 1/2/5°
  - measured on T2_d5 at nominal 25 dB: LS residual −20.8 dB, so the effective SINR ceiling is ≈21 dB
  - tune_g3 (γ=3 dB) at nominal 15 dB: residual −9.2 dB
- **CONCERN (gain model, DERIVATION + VERIFIED):** noise is added *after* the impaired combiner as white noise (S7). Paper Eq. (4) puts noise at the antennas ($w_q^Hn$). With phase-only errors, $W^HW=I$, so the two are equivalent. With gain errors they are not: the physical noise covariance is $\sigma^2W_{\text{ideal}}^H\mathrm{diag}(g_r^2)W_{\text{ideal}}$. For γ=3 dB (one draw) the diagonal mean is 1.17 and the off-diagonal/diagonal energy ratio is 0.128, so the noise should be coloured and slightly stronger. Gain errors also raise the mean signal power: $E[g^2]$ per side is 1.009/1.035/1.082/1.146 for γ = 1/2/3/4 dB, i.e. +0.08/+0.30/+0.69/+1.18 dB of signal power over both sides. The gain-impaired banks therefore have a slightly higher effective SNR than their label and white noise. The effect is small, but the "gain" banks are a stylised model.
- **Next:** S7.

### S7. Noise

- **Operation (FACT, iabr2_sim.py:54-56; DG `generate_noise` :208-239):** $Z_{qp}\sim\mathcal{CN}(0,\sigma_n^2)$ i.i.d. with $\sigma_n^2=10^{-\mathrm{SNR}/10}$; the real and imaginary parts each have variance $\sigma_n^2/2$. It is added in the **beamspace** (after combining).
- **SNR definition (DERIVATION):** $E|G_{qp}|^2=E\|H\|_F^2/(QP)=n_tn_rE\sum|\alpha_l|^2/256=1$, so the SNR is $1/\sigma_n^2$ per entry. This equals paper §IV-B "SNR definition is 1/σ²_n" with ρ=1.
- **VERIFIED (`a01_consistency.py`, 200 samples per SNR):** the estimated noise power on eval_bank is 10.08, 5.06, 0.04, −4.94, …, −24.95 dB, against 10, 5, 0, …, −25 expected. The signal power per entry is between −0.27 and +0.28 dB.
- **Minor (FACT):** in DG `training_data_generator` and the original `data_generation`, `generate_noise(1.0, SNR, P, Q)` passes (P,Q) where (Q,P) is expected. This is harmless because P=Q. It is not used by v2.
- **Next:** S8.

### S8. Observation

- **Operation:** $Y=W^HHF+Z$, (B,16,16) complex128. `einsum('bnq,bnm,bmp->bqp', conj(W), H, F)` is $W^HHF$, with the conjugate on $W$ only (FACT, iabr2_sim.py:53).
- **Single-path closed form (DERIVATION):** $Y_{qp}=\frac{\alpha}{16}\sum_n e^{-jn(u+2\pi q/16)}\sum_m e^{jm(v-2\pi p/16)}$, which peaks at $q^*=16\,\mathrm{wrap}_{2\pi}(-u)/2\pi$ and $p^*=16\,\mathrm{wrap}_{2\pi}(v)/2\pi$. This is exactly `angles_to_cells` (rows = AoA, cols = AoD) and matches paper Eqs. (9)-(10).
- **Numerical example (VERIFIED):** ψ=60°, φ=100°, α=1, SNR=300 dB.
  - $u=\pi/2$ and $v=-0.5455$, giving continuous cells (12.00, 14.61)
  - the argmax of |Y| is at (12, 15)
  - $|Y|_{\max}=12.31$ against 16 on-grid: an off-grid scalloping loss of 0.77 (−2.3 dB)
- **`want_clean` (FACT):** returns $W^H_{\text{ideal}}HF_{\text{ideal}}+Z$ with the same noise. v2 banks and the training stream do not use it.

### S9. Transformations of the observation

| Transform | Code | Input → output | Math | Preserved / discarded | Example |
|---|---|---|---|---|---|
| `to_ri` | iabr2_sim.py:63-64 | (B,16,16) c128 → (B,16,16,2) f32 | $[\Re Y,\Im Y]$ | everything except float64→float32 precision | $Y_{00}=1.08\cdot10^{-15}-4.2\cdot10^{-16}j$ → [0., −0.] |
| training layout | iabr2_sim.py:115 | (B,16,16,2) → (B,2,16,16) | transpose(0,3,1,2) | lossless | — |
| `upsample64` | :18-20 | (N,16,16,2) → (N,64,64,2) | gather `UP_SRC` = `scipy.ndimage.zoom(order=0, ×4)` index map | lossless (each source appears ≥3 times) | see below |
| `downsample16` | :23-25 | (N,64,64,2) → (N,16,16,2) | first occurrence of each source (`DOWN_IDX`) | exact inverse on zoom-generated data | DOWN_IDX[:5] = flat 0,3,7,11,15 → (0,0),(0,3),(0,7),(0,11),(0,15) |
| `to_ant` | notebook_src.py:274-276; model `front` :496-507 | beamspace (B,16,16) → antenna domain | $\tilde H=W_{\text{ideal}}YF_{\text{ideal}}^H=16\,\mathrm{ifft2}(Y)$ | lossless (unitary) | single path: row phase step −1.5708 = −π cos60°; col step −0.5455 = π cos100° |
| per-sample RMS norm + log-power channel | notebook_src.py:511-512 | $Y_c$ → (B,3,16,16) | $Y_n=Y_c/\sqrt{\mathrm{mean}|Y_c|^2}$; channels $[\Re Y_n,\Im Y_n,\log(|Y_n|^2+10^{-3})]$ | **discards absolute power**, so the network cannot see the SNR level directly | — |

**`UP_SRC` structure (VERIFIED).** `scipy.ndimage.zoom(order=0)` with the default `grid_mode=False` does **not** make uniform 4×4 blocks. Along each axis the source rows are replicated [3,4,4,4,4,5,4,4,4,4,5,4,4,4,4,3] times. `UP_SRC` reproduces this map exactly (`zoom==gather True`, 256 distinct sources). This is a FACT about the original U-Net input format: the input has uneven pixel blocks. v2 preserves it bit-exactly, and U-Net inputs rebuilt from the Y16 banks are identical to the authors' pipeline (VERIFIED: `upsample64(downsample16(x))==x` on all of eval_bank and on the dldoa test set).

### S10. Angle labels (`feat`, bank `psi`/`phi`)

- **Official / dldoa:** `features = np.stack([psi_l, phi_l])`, shape (2,L), float32, **in generation order** (sorted by |α|, strongest first). The bank stores `feat` (8000,2,3), and v2 loads `psi = feat[:,0,:]`, `phi = feat[:,1,:]` (FACT, notebook_src.py:387-388).
- **v2 / v1 banks:** `psi`, `phi` (N, Lmax) float32, NaN-padded (FACT). The metric slices `[:L]`, so the NaN padding is never read by `metrics` (FACT, notebook_src.py:321-322).
- **L is an oracle input at test time.** The decoders and the U-Net top-L blob selection receive `bank['L']` (FACT, notebook_src.py:796-806, 916). This is the same as the paper's protocol (TVT_Blob_Inference.py:347-348).

### S11. Heatmap target `heat_targets` (v2 training label)

- **Operation (FACT, iabr2_sim.py:92-104):**
  - $(q,p)=G\cdot\mathrm{wrap}_{2\pi}(-\pi\cos\psi,\ \pi\cos\phi)/2\pi$ with G=32
  - round to the nearest integer mod 32
  - separable Gaussian with **wrap-around** distance: $h_{ij}=\max_l\exp(-(d_q^2+d_p^2)/2\sigma^2)$, σ = 1 cell
  - padded paths are masked through `~isnan(psi)`
- **Shape:** (B,32,32) float32, peak exactly 1 at the rounded cell.
- **Meaning:** a detection map on a grid of 2× beam resolution, i.e. half-cells of 2π/32 = 0.196 rad in spatial frequency. σ=1 cell equals 0.196 rad. The paper's U-Net target uses σ=0.07 rad (≈2.8× narrower).
- **Discarded:**
  - (i) the sub-cell position: rounding error mean 0.236 cell, max 0.5 cell = 0.098 rad (VERIFIED); v2 recovers it with Newton refinement on $\tilde H$
  - (ii) path power: all peaks are 1 regardless of |α|, same as the paper's `amps='ones'`
  - (iii) coincident paths: `np.maximum` merges them into one positive cell
- **VERIFIED:** 1.73% of training samples have fewer positive cells than L (rising from 0.09% at L=2 to 5.1% at L=9). On the official bank, 28/8000 samples have two paths in the same rounded 32-cell. The focal loss uses `pos = target ≥ 1−1e−6` (notebook_src.py:688), so in these samples a path is silently not a separate positive. **CONCERN (minor):** the label is not per-path in close-path cases; this is intentional for a detection map, but the counts are nonzero.
- **Example (VERIFIED):** ψ=60°, φ=100° → continuous (24.00, 29.22) → peak at (24,29). Its 4-neighbours are $e^{-0.5}=0.61$. End-fire example: ψ=2° and 178° both round to row 16.

### S12. Impairment target `impairment_targets` (v2 IABC supervision)

- **Operation (FACT, iabr2_sim.py:76-89):**
  - phase: $[\Pi(-\angle D_r),\ \Pi(+\angle D_t)]$, where $\Pi(x)=x-\bar x-\frac{x\cdot n_c}{n_c\cdot n_c}n_c$ with $n_c=k-7.5$ (removes the mean and the linear ramp)
  - log-gain: $\ln|D|$ minus its mean, separately for rows and columns (no ramp removed)
- **Output:** (B,32,2) float32 = [16 rows ; 16 cols] × (phase, log-gain).
- **Sign convention (VERIFIED):** $\tilde H$ rows carry $g_re^{-j\varepsilon_r}$, so the row target is $-\varepsilon_r$; columns carry $e^{+j\varepsilon_t}$, so the column target is $+\varepsilon_t$. With the correction $c=\exp(-\text{lg}-j\,\text{ph})$ (notebook_src.py:502-503), the residual distortion after an ideal correction is **affine in phase** (max deviation from a line 6e-9 rad) and **constant in gain** (peak-to-peak 1.7e-7). So only the unidentifiable part remains, and the target is consistent with the model's correction.
- **`project_impairment` toy example (VERIFIED):** const + ramp → 5.6e-17. A spike of 0.1 at n=5 → 0.0919 at n=5 and small negative residuals elsewhere.
- **Assumption / CONCERN:** `np.angle(D)` is used without unwrapping. This is safe because |ε| ≤ 8° in training and ≤ 15° in the banks (DERIVATION). The log-gain ramp is kept, which is correct: a real exponential ramp is not interchangeable with a steering shift.

### S13. U-Net target `generate_gt` (base paper; not used to train v2)

- **Operation (FACT, dldoa_dataset_generation.py:313-372; original :268-331):** $X=\sum_l \frac{1}{2\pi\sigma^2}\exp(-[(\omega_p-\tilde\omega_{\phi_l})^2+(\omega_q-\tilde\omega_{\psi_l})^2]/2\sigma^2)$ with σ=0.07. The axes are `linspace(-3σ, 2π+3σ, 256, endpoint=False)`, so the pixel spacing is 0.026184 rad, not the 2π/256 = 0.02454 of paper Eq. (11). The map is a **sum** over paths (not a max), with **no wrap**.
- **VERIFIED:**
  - peak 32.469 (theory 1/(2πσ²) = 32.48)
  - for $\tilde\omega=0.05$ the peak is at column 10
  - the physically identical alias location $2\pi+0.05$ (column 250) lies inside the grid's margin but has GT value **0**
- **CONCERN (baseline only):** the margin pixels duplicate physical frequencies, but only one copy is labelled. `peaks_to_angles` inverts the same margin mapping and wraps (TVT_Blob_Inference.py:108-118), so the evaluation is self-consistent. The paper's Eq. (11) text (no margin) does not describe the code. The Meneses paper does mention "an extended range beyond [0, 2π]" (a01_meneses.txt:221).
- **Where used:** by the authors' U-Net training. In the v2 notebook, the U-Net "fine-tune energy" measurement uses **random tensors** `xu = randn`, `gu = rand` rather than generated data (FACT, notebook_src.py:1235). That affects cost only.

---

## 3. Per-source reconstruction

### (a) Endless training stream: `make_batch` / `BatchStream`

1. S1 draws (L, SNR, clean, δ_max, γ_max) per sample (iabr2_sim.py:108-112).
2. `sample_paths(L, rng, lmax=9)` gives ψ, φ (B,9) NaN-padded and α (B,9) with zero padding.
3. `simulate(...)` gives Y (B,16,16) plus $D_r$, $D_t$ (B,16).
4. Outputs: `y = to_ri(Y).transpose(0,3,1,2)` (B,2,16,16) f32; `heat = heat_targets(ψ,φ,32,1.0)` (B,32,32) f32; `imp = impairment_targets(D_r,D_t)` (B,32,2) f32. VERIFIED shapes/dtypes for B=4.
5. `BatchStream.__iter__` (:126-129): `rng = default_rng([seed, worker_id, urandom(4 bytes)])`. The DataLoader has `batch_size=None` (each item is already a batch of 256), 3 persistent workers, and prefetch 4 (notebook_src.py:716-718).
6. Volume: 48,000 steps × 256 ≈ 12.3 M fresh samples (DERIVATION from CFG). Nothing is stored.

- **FACT / CONCERN (reproducibility):** the `os.urandom` entropy makes the training data **non-reproducible** even with fixed `seed`. `torch.manual_seed` and `np.random.seed` (notebook_src.py:707) do not affect it. After a resume, the seed is `seed + 17*start`, still mixed with urandom.
- **INFERENCE:** the angles are continuous random draws, so the probability that any training sample coincides with a bank sample is zero. There is no sample-level leakage from the stream into the tune/test banks.
- **Training-vs-test distribution:**
  - training has P=16 only, L 1..9, SNR −15..24 (integers), and phase and gain impairments always jointly present (80%)
  - pure phase-only or gain-only conditions (like T2_d5 or the v1 gain banks) occur in training only as limits where the other draw is near 0 (DERIVATION)
  - out of distribution: L=10, SNR −25/−20/30/35, δ = 10°/15°, γ = 4 dB, and separation < 30° Euclidean (v1 `sep_{≤20}deg`)

### (b) v2 banks T2_* and tune_* (`make_bank`, notebook_src.py:356-380)

- **Loop:** for each L and each SNR in order, `sample_paths([L]*n_per, rng, lmax=max(Ls))` then `simulate(..., phase_deg_max=d, gain_db_max=g)`, all drawn from **one** `rng = default_rng(seed)`.
- **Stored fields:** `Y16` (N,16,16,2) f32, `psi`/`phi` (N,Lmax) f32, `L` int32, `snr` f32, `phase_deg`, `gain_db`, `seed`.
- **Skip rule:** `make_bank` returns early if the file exists, so the files on disk are whatever the first run wrote. Their stored seed/phase/gain fields match the spec (VERIFIED below).

| Bank | N | L | SNR (dB) | δ_max | γ_max | seed | min Euclid sep (°) |
|---|---|---|---|---|---|---|---|
| T2_d1 | 8000 | 3 | −10..25 step 5, 1000 each | 1° | 0 | 9101 | 30.003 |
| T2_d2 | 8000 | 3 | same | 2° | 0 | 9102 | 30.015 |
| T2_d5 | 8000 | 3 | same | 5° | 0 | 9105 | 30.011 |
| tune_clean | 600 | 3 | −5, 5, 15, 25 (150 each) | 0 | 0 | 9901 | 30.13 |
| tune_d5 | 300 | 3 | 0, 15 | 5° | 0 | 9902 | 30.086 |
| tune_g3 | 300 | 3 | 0, 15 | 0 | 3 dB | 9903 | 30.056 |
| tune_L6 | 300 | 6 | 0, 15 | 3° | 1 dB | 9904 | 30.072 |

(All values VERIFIED from the files; `a01_banks.py`.)

- **Purpose (FACT):** the tune banks are used only by `tune_selfcal` to choose (λ, τ) from a 7-point grid by mean Pd over the 4 banks (notebook_src.py:811-831). The result was (1.0, 5.0) (run log line 16). The T2 banks reproduce the Meneses Table-2 grid (L=3, δ_max ∈ {1,2,5}°, SNR −10..25).
- **Disjointness (INFERENCE):** tune and test use different seeds and continuous draws, so their samples are disjoint. tune_d5 is **the same condition** as T2_d5 at 0/15 dB (condition-matched tuning on disjoint samples). That is legitimate, but it means the self-calibration was selected on the test condition's distribution.
- **Comparison to Meneses Table 2 (CONCERN):** v2's T2 banks are **fresh geometries** (seeds 9101-9105) with 1000 per SNR, which v2 assumes is "same as the official test set" (notebook_src.py:89).
  - The Meneses paper does not state the per-SNR count or the seed of its Table-2 test data (a01_meneses.txt; Insufficient evidence).
  - If the authors used the original `validation_data_generator(..., seed=42, error_deg=d)`, their phase errors come from the **global unseeded `np.random`** (tvt_data_generation_v3.py:79-81, 104-106), so their test set is not reproducible. Also, because the phase-error draw does not consume `rng`, their geometry and noise would be **identical to the clean seed-42 bank** (DERIVATION from code).
  - v2's T2 numbers and the paper's Table 2 are therefore computed on different realisations. At Pd ≈ 0.9 with 3000 components per SNR, the binomial standard error alone is ≈0.5 pt (DERIVATION; components within a sample are correlated, so it is larger). Cell-by-cell "wins" within ±0.5 pt (notebook_src.py:1054) are inside that noise.

### (c) Reused v1 robustness banks (v1 suite cell 9)

- **`make_standard_bank(n, L, snr, seed, phase_deg, gain_db)`:** one `default_rng(seed)`; `sample_paths([L]*n)` then `simulate` with fixed δ/γ. Stored fields are the same as the v2 banks.
- **`make_separation_bank(n, sep_deg, snr=15, seed)`:**
  - L=2; ψ₁, φ₁ ~ U[20°, 160°−d]; the second path is (ψ₁+d, φ₁+d)
  - so the **Euclidean** angle separation is $d\sqrt2$ and "sep_d" is a per-axis offset (FACT, v1_suite_src.py:459-474; VERIFIED Δψ = 5.000002° in `sep_5deg`)
  - α ~ CN(0, 1/2), sorted, with the angles reordered to match
  - no π/6 constraint; angles are restricted to [20°, 160°], so end-fire is excluded, unlike every other bank
  - relative to training's 30° minimum Euclidean separation: `sep_30` (42.4°) is in distribution; `sep_20` (28.3°) and below are out of distribution (DERIVATION)
- **`make_nuisance_bank`:** 3 principal paths plus 1 nuisance path at `power_db` relative to the principal total, with the scenes paired across power levels. Not used by v2.

**BANK_SPECS (FACT; seeds VERIFIED against MANIFEST; n_robust = 500):**

| Family | Seed rule | Conditions |
|---|---|---|
| `phase_d{d}_snr{s}` | 1000+10d+s | d∈{0,1,2,5}, s∈{0,15} |
| `gain_g{g}_snr{s}` | 2000+int(10g)+s | g∈{0.5,1,2,4} |
| `oodphase_d{d}_snr{s}` | 3000+d+s | d∈{10,15} |
| `L{L}_snr{s}` | 4000+10L+s | L∈{1,2,4,5,6,7,8,10} |
| `sep_{d}deg_snr15` | 5000+d | d∈{30,20,10,5,3,2,1} |
| `tail_snr{s}` | 6000+s | s∈{−25,−20,30,35} |
| `v1_clean_snr{s}` | 7000+s | not used by v2 |
| `nuis_p{pw}_snr{s}` | 1000+s | 200 scenes; not used by v2 |

The C3 L-sweep uses `phase_d0_snr{s}` as its L=3 point (notebook_src.py:1076-1077).

- **VERIFIED seed collisions:**
  - {1000: phase_d0_snr0 + 3 nuis_*_snr0}
  - {1015: phase_d0_snr15 + 3 nuis_*_snr15}
  - {2020: gain_g2_snr0, gain_g0.5_snr15}
- **VERIFIED BUG (bank independence, low severity):** `gain_g0.5_snr15` and `gain_g2_snr0` have identical ψ, φ arrays. The uniform and normal draw sequence is identical: path draws, then ε_r, ε_t, g_r, g_t, then noise normals, with only the scale factors differing. Both are in v2 `ROBUST_BANKS`, so those two table cells are evaluated on the same scenes with the same noise pattern (scaled). The phase_d0/nuisance collision shares only the first uniform draws (0.5% of first-path ψ equal) and does not affect v2.

### (d) Official frozen bank `eval_bank.npz`

- **Generator (FACT, `scripts/generate_frozen_banks.py`; created 15 Sep 2026, oldest copy `D:\ai_ml_project\frozen_banks\eval_bank.npz`):**
  - loop over conditions (L=3, SNR, 16, 16) × 1000
  - per sample: $F$, $W$ ideal (no impairment); α via `rng.standard_normal` ×2 (sorted); `generate_points(3, π/6, rng)`; $H$; $G=W^HHF$; `generate_noise(1.0, SNR, P, Q, rng)`; $Y=G+Z$ → `get_real_imag` (16,16,2) → `zoom(order=0, ×4)` per channel → (64,64,2) f32
  - `generate_gt` (σ=0.07, 256×256) is computed and **discarded** ("the original evaluator never reads the GT heatmap")
  - `feat = [ψ; φ]` (2,3) f32; `meta = [L, SNR, P, nt]` f32
- **Stored:** `data` (8000,64,64,2) f32, `feat` (8000,2,3) f32, `meta` (8000,4) f32, `sigma` 0.07, `M` 256, `seed` 42. SNR blocks are contiguous: samples 0-999 are −10 dB, …, 7000-7999 are 25 dB.
- **Hashes (FACT):** all 5 copies on disk have md5 `184372b8…`.
- **VERIFIED:** bit-exact regeneration with both the original and the project generator (H1). In v2 it is loaded via `downsample16` (lossless, VERIFIED).
- **Companion:** `calibration_bank.npz` (seed 43, 75 per SNR, with GT float16) comes from the same script. It is not used by v2.

### (e) `dldoa_dataset/test_*`

- **What is established (VERIFIED):** `test_data.npz` (8000,64,64,2) f32, `test_meta.npz` (8000,4) with L=3, 1000 per SNR in {−10,…,25}, P=nt=16, and `test_features.npy` (object array of (2,3) f32). It is produced by `validation_data_generator(generate_test_conditions_fig5_fig6(), 1000, seed=7, sigma=0.07, M=256)`, bit-exact. It shares no first-path angle with eval_bank (0 overlaps); only `meta` is identical.
- **Also established:** it is the set on which `infer_dldoa_test.log` was computed (H4). Its results replicate the official-bank results within ~0.1–0.2 pt mean Pd for every method (A_official.json vs results_dldoa_test.json, e.g. U-Net 0.7366 vs 0.7354, IABR-v2 0.7551 vs 0.7531). That is expected for an independent replicate.
- **What cannot be established:**
  - which script and session wrote the zip (H5), or why seed 7 was chosen
  - how its `train`/`val` splits were generated: val conditions ≠ `generate_validation_conditions`; train has 10,000 samples with P∈{16,32} (5067/4933) and nt∈{16,32}, matching the base-paper training law in `meta`
  - v2 never uses the zip's train/val splits (FACT: no reference in notebook_src.py / infer_dldoa_test.py)
- **CONCERN (reporting):** `RESULTS_iabr_v2_test.md` and `infer_dldoa_test.py` call it "a separate pre-existing test set" without its seed. It is the **official protocol with seed 7**, which should be stated, and it is not a different distribution.

---

## 4. Clean mathematical data-generation story

For each sample $i$:

1. Draw $L_i$, the nominal $\mathrm{SNR}_i$, and the impairment levels $(\delta_i,\gamma_i)$ (training: random; bank: fixed).
2. Draw $\{(\phi_{il},\psi_{il})\}_{l=1}^{L_i}\subset[0,\pi]^2$ by sequential rejection with pairwise Euclidean distance ≥ π/6. Draw $\alpha_{il}\overset{iid}{\sim}\mathcal{CN}(0,1/L_i)$, sorted by magnitude.
3. $H_i=\sqrt{256}\sum_l\alpha_{il}a_r(\psi_{il})a_t^H(\phi_{il})$, equivalently $H_i[n,m]=\sum_l\alpha_{il}e^{-jnu_{il}+jmv_{il}}$ with $u=\pi\cos\psi$ and $v=\pi\cos\phi$.
4. $D_{r,i}=\mathrm{diag}(g_{r}e^{j\varepsilon_r})$ and $D_{t,i}$ likewise, with $\varepsilon\sim\mathcal U(\pm\delta_i)$ and $20\log_{10}g\sim\mathcal U(\pm\gamma_i)$.
5. $Y_i=W_{\text{ideal}}^HD_{r,i}^{*}H_iD_{t,i}F_{\text{ideal}}+Z_i=\tfrac1{16}\mathrm{fft2}(D_{r,i}^{*}H_iD_{t,i})+Z_i$ with $Z_i\sim\mathcal{CN}(0,10^{-\mathrm{SNR}_i/10}I)$.
6. **Input:** $\mathrm{to\_ri}(Y_i)\in\mathbb R^{16\times16\times2}$. The U-Net path uses the nearest-neighbour ×4 map with uneven 3/4/5 blocks, giving $\mathbb R^{64\times64\times2}$. Inside v2, $\tilde H_i=16\,\mathrm{ifft2}(Y_i)$.
7. **Labels:** $(\psi_{il},\phi_{il})$ as float32; a heatmap on the 32×32 wrapped grid at the rounded $(16\,\mathrm{wrap}(-u)/\pi,\ 16\,\mathrm{wrap}(v)/\pi)$ with σ = 1 cell; the impairment target $\Pi(\mp\varepsilon)$ and the mean-free $\ln g$. The model order $L_i$ is given at test time.

---

## 5. Paper vs code discrepancies (dataset-relevant)

| Item | Paper (Lloria TVT) | Authors' code / project | Label |
|---|---|---|---|
| Angle domain | §II-A "[0, 2π]"; §IV-B "[0, π]" | [0, π] | FACT (paper internally inconsistent) |
| Minimum separation | not stated | π/6 Euclidean in (φ,ψ) | FACT |
| Training SNR / L | −10..25 dB, L 1..10 (text) | −15..24, L 1..9 | FACT (Table I not extractable) |
| GT grid | $\omega_m=2\pi m/M$ | extended by ±3σ, spacing 0.02618 rad | FACT/VERIFIED |
| Phase error | uniform per antenna, Tx and Rx | same; but drawn from unseeded global `np.random` | FACT → CONCERN (reproducibility of the paper's impaired sets) |
| Gain error | not modelled | v1/v2 extension, U(±γ) dB | ASSUMPTION (flagged in v1 notebook) |
| Noise location | at antennas, $w^Hn$ | after combiner, white | equivalent for phase-only; CONCERN for gain |
| Meneses | AoA/AoD "uniform within [0, 2π]"; test size/seed not stated | v2 T2: 1000/SNR, seeds 9101-9105 | Insufficient evidence on the paper's test set |

---

## 6. Findings list (for the orchestrator)

1. VERIFIED: eval_bank = original `validation_data_generator(seed=42)`, bit-exact; the generator script is `scripts/generate_frozen_banks.py`.
2. VERIFIED: dldoa_dataset/test (in the zip) = the same protocol with seed 7, bit-exact; the log's results were computed on it.
3. VERIFIED: all v2 banks are label-consistent; SNR = 1/σ² per entry holds.
4. VERIFIED BUG (low): seed collision gain_g0.5_snr15 / gain_g2_snr0, whose scenes are identical; both are in v2 robustness tables.
5. CONCERN (medium): the unidentifiable phase ramp shifts the apparent angles relative to the labels; >1° for ~2% of components at δ=5° and ~7% at δ=15°. This caps Pd on impaired banks and is not reported anywhere.
6. CONCERN (medium): T2 banks use different realisations from the Meneses Table 2 (whose test data are unspecified and, with the original code, unseeded). Sub-point comparisons are within sampling noise.
7. CONCERN (low-medium): the gain-impairment model adds white post-combiner noise (non-physical colouring ignored) and raises signal power by up to +1.2 dB at 4 dB.
8. CONCERN (low): the 20%/80% training mix always couples phase and gain; the test banks isolate them.
9. CONCERN (low): the heat target merges coincident paths (1.7% of training samples, 28/8000 official samples).
10. CONCERN (low): training data is non-reproducible (urandom).
11. CONCERN (low): tuning is condition-matched to T2_d5 (disjoint samples).
12. CONCERN (low, baseline): the U-Net GT leaves alias margin pixels unlabelled; the U-Net energy measurement uses random tensors.
13. FACT: π/6 separation in angle space still admits beamspace-close and end-fire-aliased pairs (0.9% of official samples < 0.5 cell apart).
14. FACT: nearest-neighbour ×4 upsampling has uneven 3/4/5-pixel blocks; v2 reproduces it exactly.
15. Insufficient evidence: the generator of the zip's train/val splits and the reason for seed 7; the DG copy on the run machine; the Meneses Table-2 test set construction.
