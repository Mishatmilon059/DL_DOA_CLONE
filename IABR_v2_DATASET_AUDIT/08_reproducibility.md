# Agent 08 - Reproducibility audit of the IABR-Net v2 dataset-generation pipeline

Scope: dataset generation only (inputs, labels, training stream, tuning banks, test banks). Model accuracy is out of scope.
Rules followed: investigation only. No project file was modified, no git command was run. Scratch scripts and outputs are in `IABR_v2_DATASET_AUDIT/scratch/r08_*`.
Labels: FACT / DERIVATION / INFERENCE / ASSUMPTION / CONCERN / VERIFIED BUG. Checklist tags: [Explicitly specified] (E), [Inferable] (I), [Missing] (M), [Ambiguous] (A). "Insufficient evidence" is stated where it applies.

Paths: `SIM` = `D:\ai_ml_project\IABR_v2_extracted\IABR_v2\iabr2_sim.py`; `NB` = `...\IABR_v2\notebook_src.py`; `DG` = `D:\ai_ml_project\IABR_Net_TestSuite\dldoa_dataset_generation.py` (md5-identical to `D:\ai_ml_project\dldoa_dataset_generation.py`); `TVT` = `D:\ai_ml_project\DL_DOA\src\tvt_data_generation_v3.py`.

---
## 1. Headline

| Question | Answer | Label |
|---|---|---|
| Can the v2 tuning/test banks (T2_d1, T2_d2, T2_d5, tune_*) be regenerated from the stated spec? | Yes, **bit-exact** (all arrays, and the compressed `.npz` file md5), using only the code, the seed, `n_per`, `Ls`, `snrs`, `phase_deg`, `gain_db`. | VERIFIED (section 3.1) |
| Can the 56 v1 robustness banks that v2 also reads be regenerated? | Yes, 56/56 bit-exact, and every regenerated sha256 equals `MANIFEST.json`. | VERIFIED (section 3.2) |
| Can `eval_bank.npz` and the dldoa test split be regenerated? | Yes, bit-exact, but only from the authors' `validation_data_generator` with an undocumented-in-v2 (seed, sigma, M, amps_type) call. Confirmed by Agent 01 (seed 42 and seed 7). | VERIFIED by Agent 01 (`scratch/a01_regen_seed7.out`); see 3.4 for my own run status |
| Is the **training** data reproducible? | No. `BatchStream` mixes `os.urandom(4)` into every worker seed, so two runs with identical config see different data. It is reproducible **in distribution** only. | VERIFIED (section 3.5) |
| Are the v2 simulator and the authors' generator the same distribution? | Same physics, to 3.6e-15 for identical inputs, but **different RNG stream order**, so the same seed gives different scenes. Scenes are distribution-equivalent, not bit-equivalent. | VERIFIED (section 3.3) |

---
## 2. What the authors did (reported first)

### 2.1 Base authors (Lloria et al., IEEE TVT 2026; repo SandraRoger/DLDOA)

- FACT (`scratch/a01_lloria.txt` lines 646-680): training data are generated on the fly ("infinite dataset"), SNR "between -10 dB and 25 dB", $L$ "randomly selected from 1 to 10", $Q=P\in\{16,32\}$, $\alpha_l$ zero-mean complex Gaussian with variance $1/L$, AoA and AoD uniform in $[0,\pi]$, target = 2-D Gaussians with $\sigma_1=\sigma_2=0.07$, validation "fixed", 1000 test realisations per SNR.
- FACT (`DG:46-103`): minimum pairwise separation $\pi/6$ is imposed by sequential rejection sampling in the $(\phi,\psi)\in[0,\pi]^2$ plane with Euclidean distance, `max_attempts=10000`, an internal re-check, and no edge margin.
- FACT (`DG:249-279`, `DG:134-205`): $H=\sqrt{N_TN_R}\sum_{l}\alpha_l\,a_r(\psi_l)a_t^H(\phi_l)$, with $a(\theta)=\frac{1}{\sqrt N}[1,e^{-j\pi\cos\theta},\dots,e^{-j\pi(N-1)\cos\theta}]^T$; $Y=W^HHF+Z$; DFT codebooks with $\cos\theta_p=\frac1\pi\angle e^{j2\pi p/P}$ for $F$ and the mirrored (negative-phase) rule for $W$.
- FACT (`DG:208-239`): $Z_{ij}$ has i.i.d. real and imaginary parts $\mathcal N\!\big(0,\tfrac12 10^{-\mathrm{SNR}/10}\big)$, with $\rho=1$.
- FACT (`DG:313-372`): ground truth is a $256\times256$ sum of unit-amplitude Gaussians, $\sigma=0.07$, on a grid `linspace(-3σ, 2π+3σ, 256, endpoint=False)` (hidden default `margin_factor=3.0`), `normalize_gt=False`.
- FACT (`DG:585-588`): 16x16 (or 32x32) $Y$ is upsampled to 64x64 with `scipy.ndimage.zoom(., 4 (or 2), order=0)`, real and imaginary parts separately.
- FACT (`DG:421-423`, `DG:539-566`): the training generator uses **unseeded global** `np.random`/`random` and $P\in\{16,32\}$. The validation generator uses `np.random.default_rng(seed)` and draws in the order alpha, points, noise.
- FACT (`DG:159-161`, `TVT` equivalent): hardware impairment (`error_deg`) is a phase-only error, one `np.random.uniform(-d,d,size=nt)` per call multiplied onto every beam (column), independent for $F$ and $W$, using the **unseeded global** `np.random`. So an impaired authors' bank is not reproducible even with `seed` fixed. No gain error exists.

### 2.2 Comparison paper (Meneses-Albala et al., J. Supercomputing 2026)

- FACT (`scratch/a01_meneses.txt` lines 148-158, 308-347): $\delta_i\sim U[-\delta_{max},\delta_{max}]$ per antenna element, $\delta_{max}\in\{1^\circ,2^\circ,5^\circ\}$; fine-tuning set 10,000 impaired samples; validation 4,800 samples over a grid of (L, SNR, PQ, $\delta_{max}$); Table 2 compares base vs fine-tuned models.
- Insufficient evidence: how the **Table 2 test set** was generated (size, seeds, whether $F$ and $W$ have independent errors, whether one error vector is shared across columns). The text says only that the error alters "array responses", while the authors' code puts it on the codebooks (Agent 09 notes the two are equal in distribution).

### 2.3 What v2 does differently in data generation (FACT)

- `SIM:28-38` `sample_paths` calls `DG.generate_points` (identical rejection sampler) and then draws alpha. The stream order per sample is **points, then alpha**, the reverse of `validation_data_generator` (alpha, points).
- `SIM:41-60` `simulate` is a vectorised re-implementation of $Y=W^HHF+Z$ that adds a **per-antenna complex distortion** $D=g\,e^{j\varepsilon}$ to both codebooks, with
  $$\varepsilon\sim \mathcal U(-1,1)\,\delta_{\max},\qquad g=10^{\,u\,G_{\max}/20},\;u\sim\mathcal U(-1,1),$$
  one vector per antenna per sample, constant across beams. This is a superset of the authors' phase-only model (it adds a gain error).
- `SIM:107-117` `make_batch` (training): $L\sim\mathrm{randint}[1,9]$, SNR $\sim\mathrm{randint}[-15,24]$ dB, 20% clean samples, otherwise $\delta_{\max}\sim\mathcal U(0,8)^\circ$ and $G_{\max}\sim\mathcal U(0,3)$ dB, $P=Q=16$ only.
- `SIM:92-104` `heat_targets`: $32\times32$ wrap-around Gaussian, $\sigma=1$ cell, peak 1 at the rounded cell. This is a **different label** from the authors' $256\times256$, $\sigma=0.07$ rad map.
- `SIM:83-89` `impairment_targets`: 32 (phase, log-gain) values per sample with the mean (both) and the linear ramp (phase only) projected out.

---
## 3. Regeneration experiments

### 3.1 v2 banks (`scratch/r08_regen_v2.py`, `.out`)
FACT. I re-ran `make_bank` (NB:356-368) from scratch in a clean directory with numpy 2.3.5 / scipy 1.17.0 (the original run used numpy 2.4.2, see section 8).

| Bank | Spec | All arrays equal | Compressed file md5 equal |
|---|---|---|---|
| T2_d1 | n_per 1000, L=[3], SNR -10..25 step 5, seed 9101, phase 1 | yes (max abs diff 0.0) | yes (887751f0...) |
| T2_d2 | seed 9102, phase 2 | yes | yes (7f560e0e...) |
| T2_d5 | seed 9105, phase 5 | yes | yes |
| tune_clean, tune_d5, tune_g3, tune_L6 | n_per 150, seeds 9901-9904 | yes | yes |

The summary check (`v2_regen_ok.npy`) covers all 7 banks. The notebook (`.ipynb`) code cell, `NB` and `SIM` were diffed by `r08_nbdiff.py` and are identical.
The regeneration is therefore bit-exact across a numpy minor-version change (2.4.2 to 2.3.5). DERIVATION: PCG64 streams and `standard_normal` (ziggurat) are stable across these versions, and the results confirm it.

### 3.2 v1 suite banks used by v2 (`scratch/r08_regen_v1.py`, `.out`)
FACT. I executed the v1 suite's own bank-builder cell (`IABR_Net_Full_Test_Suite.ipynb`, `make_standard_bank`, `make_separation_bank`, `make_nuisance_bank`, `BANK_SPECS`, 56 specs) with the v2 `sample_paths`/`simulate` and compared with `data/generated_banks`.
- regenerated sha256 equals `MANIFEST.json` for 56/56; regenerated `.npz` bytes equal the saved file for 56/56; all arrays equal 56/56; saved-file sha256 equals `MANIFEST` for 56/56.
- So the v1 suite `sample_paths`/`simulate` are functionally the same as v2's. (Regeneration time 78 s.)

### 3.3 Equivalence with the authors' generator (`scratch/r08_equiv.py`, `.out`)
- FACT: for one scene (alpha, psi, phi and noise taken from `DG` with `rng=default_rng(5)`), $\max|Y_{v2}-Y_{DG}|=3.6\times10^{-15}$ (with $\max|Y|=9.9$). So the v2 channel/codebook/observation maths equals `DG.generate_channel_v2` and the DG codebooks to float64 rounding.
- FACT: with `seed=42`, `sample_paths` gives first path $(\psi,\phi)=(1.3788,2.4315)$ while `validation_data_generator` gives $(2.4695,2.3912)$. Same seed, different scenes, because of the stream order (DERIVATION: `DG:549-566` draws alpha first, `SIM:34-35` draws points first).
- FACT: `upsample64` equals `scipy.ndimage.zoom(order=0)` exactly, and `downsample16(upsample64(y)) == y` bit-exactly.

### 3.4 `eval_bank.npz` and dldoa test (provenance)
- FACT (`scripts/generate_frozen_banks.py`): `eval_bank.npz` is created with the original `validation_data_generator`, `L=[3]`, `SNR=-10..25 step 5`, `P=[16]`, `nt=nr=[16]`, `seed=42`, `sigma=0.07`, `M=256`, `amps_type='ones'`, 1000 per condition, storing only `data`, `feat`, `meta`. The file also stores `sigma`, `M`, `seed` scalars (checked: keys `data, feat, meta, sigma, M, seed`).
- VERIFIED (Agent 01, `01_pipeline_reconstructor.md` sections H1, (d), (e)): bit-exact regeneration of `eval_bank` (seed 42) and of `dldoa_dataset/test_*` (seed 7; `scratch/a01_regen_seed7.out`: data, features and meta all equal). The two share no first-path angle.
- Missing (M): the generator script and session that wrote the `dldoa_dataset` zip, and the reason for seed 7. Its train/val splits have no recoverable generator. v2's log calls it "a separate pre-existing test set" without giving the seed (CONCERN, reporting).
- My own independent re-run (`scratch/r08_official.py`) was still running when I finalised this report (the machine was shared with other agents' jobs and the output is written only at exit). I therefore rely on Agent 01's result for these two items and I do not claim an independent confirmation. Insufficient evidence for my own confirmation.

### 3.5 Training-stream determinism (`scratch/r08_checks.py`, `.out`)
- FACT (`SIM:125-129`): worker RNG is `default_rng([self.seed, worker_id, int.from_bytes(os.urandom(4),'little')])`.
- VERIFIED: two `BatchStream` objects with identical seed give different first batches (False), while patching `os.urandom` to a constant makes them identical (True). `make_batch(rng=5)` is itself deterministic.
- DERIVATION: `torch.manual_seed(1000+seed); np.random.seed(1000+seed)` (`NB:707`) fixes weight init and dropout-style torch RNG, but does not affect the data workers. No `cudnn.deterministic`/`use_deterministic_algorithms` flags exist in `NB` (grep found none), so GPU kernels are also not pinned. Result: training is non-reproducible in two independent ways.
- FACT (`NB:716`): the stream seed is `seed + 17*start`, so a resumed run gets a different (already non-deterministic) stream.

### 3.6 Other checks (`r08_checks.out`, `r08_seedscan.out`)
- FACT: `scipy.ndimage.zoom(order=0)` 16 to 64 is **not** a block repeat: the row/column run lengths are 3,4,4,4,4,5,4,4,4,4,5,4,4,4,4,3 (10 rows differ from `np.repeat(.,4)`). The map is separable and invertible, `UP_SRC`/`DOWN_IDX` recover the 16x16 array exactly.
- FACT: `F_IDEAL` is deterministic and unitary (error 2.4e-15). `cosp` = [0, .125, ..., 1.0, -.875, ..., -.125].
- FACT: `generate_points` never failed in 300 trials each for L = 3, 6, 9, 10, 12 (`max_attempts=10000` is not binding at these densities).
- FACT (seed scan, 56 v1 banks): only 49 distinct seeds. Seed 2020 is shared by `gain_g2_snr0` and `gain_g0.5_snr15`, and their `psi`/`phi` arrays are **identical**, while `Y` differs (different SNR and gain). Seeds 1000 and 1015 are shared by `phase_d0_snr{0,15}` and the three `nuis_*` banks each, but shapes differ (500x3 vs 200x4) and the scenes are not identical.
- FACT: no v2 or official seed (42, 7, 9101, 9102, 9105, 9901-9904) collides with any v1 seed (range 1000-7025).

---
## 4. Reproducibility checklist

Format: item, value, source, tag.

### 4.1 Random distributions

| # | Item | Value | Source | Tag |
|---|---|---|---|---|
| D1 | Number of paths, training | integer uniform on {1,...,9} (`rng.integers(1, 9+1)`) | `SIM:108`, `NB:85` | E (code). Paper says 1 to 10: mismatch, see section 7 |
| D2 | Number of paths, test banks | fixed L=3 (official, T2, tune_clean/d5/g3), L=6 (tune_L6) | `NB:372-377` | E |
| D3 | SNR, training | integer uniform on {-15,...,24} dB, per sample | `SIM:109`, `NB:85` | E (code). Paper -10 to 25: mismatch |
| D4 | SNR, banks | grid -10,-5,...,25 (official, T2); {-5,5,15,25} tune_clean; {0,15} other tune | `NB:371-377` | E |
| D5 | Angle pair | rejection sampling, x=AoD $\phi$, y=AoA $\psi$, each $\mathcal U[0,\pi]$; accept if Euclid distance to all earlier points $\ge\pi/6$ | `DG:46-93` | E (code); A in paper (range $[0,2\pi]$ in Sec. II vs $[0,\pi]$ in Sec. IV-B; metric not stated) |
| D6 | Separation metric | Euclidean in the $(\phi,\psi)$ radian plane, no wrap, no per-axis separation, no edge margin | `DG:81-84` | I from code, M in paper |
| D7 | Path gain | $\alpha_l=\sqrt{1/L}(n_1+jn_2)/\sqrt2$, $n_i\sim\mathcal N(0,1)$, then sorted by $|\alpha|$ descending | `SIM:35-36` | E |
| D8 | Noise | real and imag $\mathcal N(0, 10^{-\mathrm{SNR}/10}/2)$ per entry, added in beamspace | `SIM:55-56`, `DG:236-238` | E |
| D9 | Clean flag | Bernoulli(0.2): `rng.uniform(size=B) < clean_frac` | `SIM:110` | E |
| D10 | Phase envelope per sample | $\delta_{\max}\sim\mathcal U(0,8)^\circ$, else 0 if clean | `SIM:111` | E |
| D11 | Gain envelope per sample | $G_{\max}\sim\mathcal U(0,3)$ dB, else 0 if clean; drawn independently of $\delta_{\max}$ | `SIM:112` | E |
| D12 | Per-antenna phase error | $\varepsilon_n=u\,\delta_{\max}$, $u\sim\mathcal U(-1,1)$, separate draws for RX (16) and TX (16), constant across beams | `SIM:49` | E |
| D13 | Per-antenna gain error | $g_n=10^{uG_{\max}/20}$, $u\sim\mathcal U(-1,1)$, separate RX/TX | `SIM:50` | E |
| D14 | Impairment draws when amplitude is 0 | The four uniform arrays are drawn even when $\delta_{\max}=G_{\max}=0$, so they still consume RNG state | `SIM:49-50` | I (DERIVATION) |
| D15 | Codebook size mixture | none in v2 (P=Q=16 only); authors' training uses `random.choice([16,32])` | `SIM:11` vs `DG:423` | E (mismatch) |

### 4.2 Constants, dimensions, normalisation

| # | Item | Value | Source | Tag |
|---|---|---|---|---|
| C1 | Array sizes | $N_T=N_R=P=Q=16$ | `SIM:11` | E |
| C2 | Channel scale | $\sqrt{N_TN_R}=16$ | `SIM:47` | E |
| C3 | Steering vector | $e^{-j\pi k\cos\theta}/\sqrt N$, half-wavelength ULA | `SIM:45-46` | E |
| C4 | Codebooks | $F=$ `DG.beamforming_vector_generation_P(16,16)`, $W=$ `..._Q(16,16)`; DFT with $\cos\theta_p=\frac1\pi\angle e^{\pm j2\pi p/P}$ | `SIM:12-13`, `DG:134-205` | E (code); A in paper (refers to [16], not reproduced) |
| C5 | Signal reference | $\rho=1$, $\mathrm{E}\sum|\alpha|^2=1$; SNR $=1/\sigma_n^2$ | `DG:236`, paper | E |
| C6 | Input layout | `Y16`: (N,16,16,2), rows = RX combiner q, columns = TX beam p, last axis = (real, imag); trained as (B,2,16,16) | `SIM:63-64,115` | E |
| C7 | Numeric precision | simulation in float64/complex128; stored `Y16` float32; `psi`/`phi` stored float32 | `SIM:64`, `NB:366-367` | E |
| C8 | Generator-side normalisation of $Y$ | none. Model-side per-sample RMS normalisation and a $\log(|Y|^2+10^{-3})$ channel (`NB:511-512`) | `NB:511-512` | E (model side, not data) |
| C9 | Upsampling to 64x64 (only for the official bank) | `scipy.ndimage.zoom(order=0)`, run lengths 3,4,4,4,4,5,4,4,4,4,5,4,4,4,4,3; inverse `DOWN_IDX` | `SIM:14-25` | E; the non-uniform map is not documented in the paper (A) |
| C10 | Sort order of paths in stored banks | as returned by `generate_points` (acceptance order), alpha sorted strongest first; unused slots are NaN padded | `SIM:31,37` | E |

### 4.3 Labels

| # | Item | Value | Source | Tag |
|---|---|---|---|---|
| T1 | Heat grid | $G=32$ | `NB:84` | E |
| T2 | Heat coordinates | rows $=G\cdot\mathrm{wrap}_{2\pi}(-\pi\cos\psi)/2\pi$, cols $=G\cdot\mathrm{wrap}_{2\pi}(\pi\cos\phi)/2\pi$ | `SIM:71-73` | E |
| T3 | Cell rounding | `np.round` (half to even), modulo $G$ | `SIM:96` | E. Sub-cell position is discarded in the label |
| T4 | Heat kernel | $\exp(-d^2/2\sigma^2)$, $\sigma=1$ cell, circular distance, path-wise maximum, peak exactly 1 | `SIM:98-103`, `NB:84` | E |
| T5 | Authors' GT | $256^2$ grid, $\sigma=0.07$ rad, `margin_factor=3.0` shifts the grid start to $-0.21$, `amps='ones'`, `normalize_gt=False` | `DG:313-372` | E (default hidden in a function signature); not used by v2 |
| T6 | Impairment target | (phase, log-gain), shape (B,32,2): rows use $-\angle D_r$, columns $+\angle D_t$; mean removed from phase and log-gain, linear ramp $(n-7.5)$ removed from phase only | `SIM:76-89` | E |
| T7 | Label sign convention for rows | derived assumption $\tilde H=\bar D_rHD_t$ | `SIM:84` docstring | E (a derivation, not checked here; Agent 02 owns the maths) |

### 4.4 Seeds and splits

| # | Item | Value | Source | Tag |
|---|---|---|---|---|
| S1 | T2 banks | seeds 9100+$\delta$, i.e. 9101, 9102, 9105; unpaired (different scenes per $\delta$, verified by different `psi` sha1) | `NB:372` | E |
| S2 | Tuning banks | 9901, 9902, 9903, 9904 | `NB:374-377` | E |
| S3 | Official | 42 (stored in the npz as `seed`) | `generate_frozen_banks.py`; `eval_bank.npz['seed']` | E |
| S4 | Calibration bank | 43, 75 per condition, gt float16 | `generate_frozen_banks.py` | E (not used by v2 dataset code) |
| S5 | dldoa test | seed 7 | Agent 01 (bit-exact match) | I (found by seed scan, not stated in any v2 file) |
| S6 | Training stream | `default_rng([seed+17*start, worker_id, urandom(4)])`, workers=3 | `SIM:127`, `NB:716`, `CFG workers=3` | E (code); the entropy word makes it M as a reproducible seed |
| S7 | Torch/numpy seeds | `1000+seed` | `NB:707` | E |
| S8 | v1 suite seeds | 49 distinct values in 56 banks, see 3.6 | `MANIFEST.json`, `BANK_SPECS` | E |
| S9 | Train/val/test split | there is no static training set and no train/val split in v2: training is an endless stream, validation is by the fixed banks. The tuning banks (S2) are separate seeds from every test bank | `NB` | E |
| S10 | Bank RNG order | for L in Ls: for snr in snrs: [per sample: points then alpha] then [one vectorised block: $\varepsilon_r,\varepsilon_t,g_r,g_t$, noise] | `NB:361-365`, `SIM:32-60` | I (DERIVATION from code; confirmed by bit-exact regeneration) |
| S11 | Prefix stability | Not prefix-stable: the vectorised block draws depend on `n_per`, so bank content changes if `n_per` or the SNR order changes | `SIM:49-56` | I (DERIVATION) |
| S12 | Skip-if-exists | `make_bank` returns without regenerating if the file already exists, so a stale bank would be silently reused | `NB:357-358` | E |
| S13 | Bank lookup fallback | `load_bank` falls back to the v1 suite directory if the file is absent from `BANK_DIR` | `NB:391` | E |

### 4.5 Hidden and default parameters

| # | Item | Value | Source | Tag |
|---|---|---|---|---|
| H1 | `generate_points` `max_attempts` | 10000 (not binding for L up to 12, 0/300 failures) | `DG:46`, `r08_checks.out` | E |
| H2 | `sigma` default | 0.07 in `DG` (paper 0.07); `TVT` default is 0.1 | `DG:314`, `TVT` | E; mismatch between repo copies |
| H3 | `M` default | 256 | `DG:379,501` | E |
| H4 | `margin_factor` | 3.0 | `DG:314` | E (hidden) |
| H5 | `generate_points` `rng=None` | falls back to an unseeded `default_rng()`. `training_data_generator` calls it without `rng` | `DG:71-72,448` | E |
| H6 | `error_deg` | phase-only, global `np.random`, unseeded | `DG:159-161,195-197` | E |
| H7 | `os.urandom` in `BatchStream` | 4 bytes | `SIM:127` | E |
| H8 | `amps_type` | 'ones' | `DG:379,501` | E |
| H9 | v2 heat sigma | 1.0 cell | `NB:84` | E |

### 4.6 Preprocessing, filters and transformations

| # | Item | Value | Tag |
|---|---|---|---|
| P1 | Filters / outlier rejection on generated scenes | none (only the min-separation check) | E |
| P2 | Per-sample scaling of $Y$ in the data | none; RMS scaling occurs inside the model (`NB:511`) | E |
| P3 | float32 cast | at `to_ri`; heat targets use the float64 $\psi,\phi$ in training but float32 in banks (rounding differences below 1e-7 rad, can change a cell only at an exact half-cell boundary; DERIVATION) | E / I |
| P4 | Official bank `downsample16` | lossless inverse of the 4x nearest-neighbour up-map (VERIFIED) | E |
| P5 | Bank NaN padding | `psi`/`phi` NaN for slots $\ge L$ | E |

---
## 5. Classification of reproducibility

### 5.1 Bit-exact reproducible (VERIFIED unless stated)
- All v2 banks: T2_d1, T2_d2, T2_d5, tune_clean, tune_d5, tune_g3, tune_L6 (arrays and file md5).
- All 56 v1 banks read by v2 (arrays, file bytes and sha256 in `MANIFEST.json`).
- `eval_bank.npz` (seed 42) and `dldoa_dataset/test_*` (seed 7): bit-exact per Agent 01, with the authors' generator. The exact call arguments have to be known (they are recorded in `generate_frozen_banks.py`, and `sigma`, `M`, `seed` are stored in the npz for `eval_bank` only).
- The codebooks $F_{ideal},W_{ideal}$ and the up/down-sampling maps.

### 5.2 Reproducible only in distribution
- The **training stream** (all of `make_batch`): fixed by the spec, but seeded by `os.urandom`. Distribution is reproducible (L, SNR, impairment mixture, angle sampler), samples are not.
- The authors' **impaired** banks (`error_deg` set): global unseeded RNG, so even `seed` fixed does not reproduce them. The Meneses Table 2 test set is unspecified.
- Cross-generator comparisons: official bank (authors' stream order) versus v2 banks (`sample_paths` order). Same distribution (equal to 3.6e-15 for a given scene), different scenes for the same seed. This matters when one compares $\delta=0$ (official) with $\delta>0$ (T2 banks, seeds 9100+$\delta$): the scenes are independent samples, not a paired design (DERIVATION; sampling standard error of $P_d$ is up to $\sqrt{0.25/1000}=1.6$ percentage points per SNR).
- The v1 gain-sweep banks with the shared seed 2020 are partially paired (identical scenes for two SNR/gain levels), so the corresponding results are correlated (VERIFIED).

### 5.3 Missing information
- M1: the script that produced the `dldoa_dataset` zip (train, val and test), and why seed 7. Only the test split is regenerable. Insufficient evidence for the train and val splits.
- M2: the Meneses Table 2 test-set generator (size, seed, independence of TX/RX errors, per-column sharing).
- M3: the training-stream seed of the reported v2 model (fresh entropy each run, not logged). `outputs/logs/config.json` records CFG, torch 2.10.0+cu126, platform Windows-10-10.0.19045, RTX 3050, but not the drawn `os.urandom` words.
- M4: the authors' own trained-model training run (the DLDOA repo has no training script; only the generators).
- M5: the numpy/scipy versions used to write the banks are not stored inside the banks (numpy 2.4.2 from the log; my regeneration at 2.3.5 matched, so this is currently harmless, but nothing pins it).

---
## 6. Provenance of the training and test sources (short table)

| Source | Generator | Seeded | Regenerable | Tag |
|---|---|---|---|---|
| Training stream (v2) | `SIM:make_batch` | seed + os.urandom | distribution only | VERIFIED |
| T2_d*, tune_* | `NB:make_bank` | yes | bit-exact | VERIFIED |
| v1 banks (56) | v1 suite cell | yes | bit-exact | VERIFIED |
| eval_bank (seed 42) | authors' `validation_data_generator` | yes | bit-exact (Agent 01) | VERIFIED (Agent 01) |
| dldoa test (seed 7) | same, seed 7 | yes | bit-exact (Agent 01) | VERIFIED (Agent 01) |
| dldoa train/val | unknown | unknown | no | Insufficient evidence |
| Authors' impaired banks | `error_deg` path | global unseeded | no | FACT |

---
## 7. Paper versus code versus v2 discrepancies (what may be problematic)

| Item | Paper (Lloria 2026) | Authors' code | v2 |
|---|---|---|---|
| L range | 1 to 10 | randint(1,10) gives 1 to 9 | 1 to 9 (says "same as the original") |
| SNR range | -10 to 25 dB | randint(-15,25) gives -15 to 24 | -15 to 24 |
| Codebook size | P=Q in {16,32} | random.choice([16,32]) | 16 only |
| Angle range | internally inconsistent: Sec. II says AoA/AoD uniform in $[0,2\pi]$ (`a01_lloria.txt` line 226), Sec. IV-B says $[0,\pi]$ (line 663-664) | [0,pi] with Euclid separation pi/6 | same |
| sigma | 0.07 (rad) | 0.07 in `DG`, 0.1 default in `TVT` | 1 cell of a 32-grid (different label) |
| Impairment | phase-only in test (not trained with) | phase-only, global RNG | phase and gain, trained |
| Training noise | AWGN | AWGN | AWGN |

- CONCERN: the v2 claim "same as the original training generator" (`NB:85`) is true of the code, but the code differs from the published text on L and SNR (mismatch of one unit at the top of L and of five dB at the bottom of SNR). The first is DERIVATION from `randint` upper-exclusive semantics, the second from `randint(-15,25)`.
- CONCERN: `NB:89` says the T2 banks are "same as the official test set" in size; but the stream order, seeds and impairment model differ, so they are not the same population as any Meneses Table 2 set (Missing M2).
- CONCERN: the v2 training set has $P=Q=16$ only, so v2 is trained on a strict subset of the authors' data distribution with respect to the codebook size, and on a broader one with respect to impairments (gain error, up to 8 degrees).
- CONCERN: labels differ from the paper (32x32, $\sigma=1$ cell, rounded cell). Results are not comparable with the paper's heatmap-target training; only the angle estimates (Pd, RMSE) can be compared.
- CONCERN: T3 rounding discards sub-cell position from the label (up to half a cell = $\pi/32$ of $u$). This is a design choice, not an error, but it limits how tight the supervised heat label can be.
- CONCERN (bank hygiene): `make_bank` silently skips existing files (S12) and `load_bank` falls back to another directory (S13). A stale or wrong bank would be used without a warning. I found no such case: every bank I regenerated equals the saved one.
- CONCERN (statistics): seeds are shared across some v1 banks (S8), giving identical scenes across two robustness conditions (VERIFIED for seed 2020).

No VERIFIED BUG was found in the dataset-generation code itself: every generation path I re-ran matched its saved artefact bit for bit.

---
## 8. Environment

| Component | Original run | My regeneration |
|---|---|---|
| numpy | 2.4.2 (`outputs_train_stdout.log`) | 2.3.5 |
| scipy | not logged | 1.17.0 |
| torch | 2.10.0+cu126 | 2.7.1+cu118 (only for import) |
| Platform / GPU | Windows-10-10.0.19045, RTX 3050 Laptop | same Windows machine family, CPU |
| Training length | 48000 steps, batch 256, 3 workers, about 108 min | not re-run |

FACT for the original run (`config.json`, log). The bank contents are independent of torch.

---
## 9. Bottom line

1. FACT: The evaluation side (all banks) is fully reproducible bit for bit from the code plus seeds, including across a numpy minor version.
2. FACT: The training side is not reproducible: `os.urandom` in `BatchStream`, resume-dependent stream seeds, and no deterministic-GPU flags.
3. FACT: The v2 stream order differs from the authors' generator, so equal seeds do not give equal scenes. Pd differences between the official bank and the T2 banks are therefore between independent scene samples.
4. Insufficient evidence: origin of the dldoa train/val splits, the Meneses Table 2 test set, and my own independent confirmation of the eval_bank and seed-7 regeneration (Agent 01 provides it).
