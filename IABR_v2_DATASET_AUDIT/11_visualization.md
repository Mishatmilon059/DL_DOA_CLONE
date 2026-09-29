# 11 - Visualization Report: IABR-Net v2 dataset-generation pipeline

Agent 11 (Visualization). Scope: dataset generation only (not model accuracy). Investigation only: no project file was modified, no fix is proposed, git was not run.

All figures are in `D:\ai_ml_project\IABR_v2_DATASET_AUDIT\figures\`, produced by scripts in `D:\ai_ml_project\IABR_v2_DATASET_AUDIT\scratch\` (`a11_common.py` plus `a11_fig01_pipeline.py`, `a11_fig02_05.py`, `a11_fig06_08.py`, `a11_fig09_11.py`, `a11_fig12_16.py`, `a11_fig14_15.py`, `a11_train_draw.py`). Every PNG carries a footer stating its concept and its data source.

Legend for labels: FACT (read in a file or measured), DERIVATION (computed by hand or by script from stated inputs), INFERENCE, ASSUMPTION, CONCERN, VERIFIED BUG. Where evidence is missing the text says "Insufficient evidence".

Source files read (read-only): `IABR_v2_extracted\IABR_v2\iabr2_sim.py` (simulator, referred to as `sim`), `IABR_v2_extracted\IABR_v2\notebook_src.py` (training configuration), `D:\ai_ml_project\dldoa_dataset_generation.py` (`DG`, the angle sampler), report `01_pipeline_reconstructor.md`. Data read: `IABR_v2\data\banks\{T2_d1,T2_d2,T2_d5,tune_clean,tune_d5,tune_g3,tune_L6}.npz`, `IABR_Net_TestSuite\data\frozen_banks\eval_bank.npz`, `IABR_Net_TestSuite\data\generated_banks\` (56 npz plus `MANIFEST.json`).

---

## 1. Executive summary

* The generation pipeline is a physically standard 4-step chain: (i) draw $L$ paths with angles and complex gains, (ii) build $H=\sqrt{N_tN_r}\sum_l\alpha_l\,a_r(\psi_l)a_t(\phi_l)^H$, (iii) observe through two unitary DFT codebooks with per-antenna hardware impairment, $Y=W^HHF+Z$, (iv) convert to a 16x16 Re/Im image, with labels being a 32x32 Gaussian heat map and a mean/ramp-removed impairment profile.
* The figures confirm (numerically, not just by reading code) that the ground-truth cell formula puts labels on the physical peaks (17.3 dB above the map mean, versus 8.2 dB if AoA/AoD are swapped and 3.9 dB for random angles).
* Visual evidence supports several audit concerns already raised, quantified below: label merging in beamspace (15.2 % of training scenes with $L>1$ have two paths closer than 1 beam cell even though the angle-space separation is at least $\pi/6$), nominal SNR being only an ensemble average, test SNR 25 dB outside the training range (max 24), and v1 robustness banks with impairment beyond the training bounds.
* No dimension, conjugation, or FFT-convention error was found in the equations visualised. Limits: the training stream is not reproducible (`os.urandom`), so training histograms are a faithful replica, not the actual run.

---

## 2. Flow diagrams (ASCII)

### 2.1 Whole pipeline (fig01)

```
 scene draw                    channel                 observation                    tensor / labels
 ----------                    -------                 -----------                    ---------------
 L ~ U{1..9}  ------+
 SNR ~ U{-15..24} --|-----+
 delta ~ U(0,8 deg) |     |
 gamma ~ U(0,3 dB)  |     |
 clean w.p. 0.2     |     |
                    v     |
 DG.generate_points: |     |
  (phi, psi) in [0,pi]^2  |
  reject if dist < pi/6   |
  alpha_l ~ CN(0,1/L), sort by |alpha|
                    |     |
                    v     |
      H = sqrt(NtNr) sum_l alpha_l a_r(psi_l) a_t(phi_l)^H     (16x16 complex)
                    |
                    v
      F = D_t F0,  W = D_r W0     D = diag(g e^{j eps})  (eps ~ U(+-delta), 20log g ~ U(+-gamma dB))
                    |
                    v
      Y = W^H H F + Z,   Z ~ CN(0, 10^{-SNR/10}) per entry     (16x16 complex, beamspace)
                    |
                    +--> to_ri  --> y (2,16,16) float32     [network input]
                    +--> angles_to_cells --> 32x32 Gaussian heat (sigma=1)   [detection label]
                    +--> project_impairment --> imp (32,2) [phase, log-gain] [impairment label]
```

### 2.2 Splits (fig14)

```
 TRAIN : BatchStream (endless, os.urandom seed, 48000 steps x 256 = 12,288,000 draws per CFG)
 TUNE  : tune_clean, tune_d5, tune_g3, tune_L6      (seeds 9901-9904, 1500 samples total)
 TEST  : eval_bank (seed 42, 8000), T2_d1 (9101), T2_d2 (9102), T2_d5 (9105) - 8000 each
         v1 robustness banks: 56 specs, 26,200 samples (47 used in v2 tables, per report 01)
 split is by generator seed; no shared pool is carved.
```

### 2.3 Tensor shapes (fig15)

```
 make_batch(B):   y (B,2,16,16) f32  |  heat (B,32,32) f32  |  imp (B,32,2) f32
 v2 bank npz:     Y16 (N,16,16,2)   psi (N,9?) phi (N,9?)  L (N)  snr (N)  phase_deg  gain_db  seed
 eval_bank npz:   data (N,2,64,64)  feat (N,2,Lmax)  meta (N,4)=[L,SNR,16,16]  sigma  M  seed
```
Exact shapes and dtypes are printed on fig15 from the files themselves; the lines above are a reading aid, not a substitute.

---

## 3. Stage-by-stage visualization plan and what was rendered

For each stage: the plan (what the picture must show), the figure that implements it, its data source, what to look at, and any audit-relevant observation. "Report what the authors did first, then possible problems."

### 3.1 Pipeline overview - fig01_pipeline_overview.png
* Plan: one schematic showing sampling, channel, observation, tensor and labels with array shapes.
* Data: `sim` symbols and `notebook_src.py` CFG (lines 82-92). No random data.
* Authors' design: as in the diagram of section 2.1.

### 3.2 Scene to paths - fig02_scene_paths.png
* Plan: show the $(\phi,\psi)$ square $[0,\pi]^2$ with sampled points, the minimum-distance rule, and the path-gain magnitudes sorted by size.
* Data: `DG.generate_points`, `sim.sample_paths` (real function calls).
* Authors: angles are uniform on $[0,\pi]^2$ by sequential rejection with Euclidean minimum separation $\pi/6$ in angle space (FACT, `DG`); tuple order is (AoD $\phi$, AoA $\psi$); $\alpha_l\sim\mathcal{CN}(0,1/L)$ sorted by magnitude.
* Possible problem: the separation is in angle space, not in beam space (see 3.11, fig16). CONCERN.

### 3.3 Steering vectors - fig03_steering_vectors.png
* Plan: real/imag parts of $a(\theta)$ over antenna index, and the phase progression versus index.
* Equation (FACT, `sim`): $a(\theta)_k=\dfrac{1}{\sqrt{16}}e^{-j\pi k\cos\theta}$, $k=0,\dots,15$, half-wavelength spacing.
* Checks: dimension 16; $\|a\|_2=1$ (16 entries of magnitude $1/4$); the per-antenna phase step is $-\pi\cos\theta$ rad (radians, not degrees); the negative sign is a convention only, chosen consistently on TX and RX (see 3.8 for the cell formula that inherits it).
* Note: the DFT beam index $p$ that responds to this vector is $p=n\,\mathrm{wrap}(\pi\cos\theta)/2\pi$ (TX) and the RX form has the opposite sign because of the conjugate in $W^H$. The sign asymmetry between `angles_to_cells` rows and columns is therefore expected, not a bug (validated numerically in 3.9).

### 3.4 Path combination - fig04_path_combination.png
* Plan: show single-path rank-1 matrices, their weighted sum, and the sum's SVD spectrum.
* Equation: $H=\sqrt{N_tN_r}\sum_{l=1}^{L}\alpha_l\,a_r(\psi_l)\,a_t(\phi_l)^{H}$ with $N_t=N_r=16$, so $\sqrt{N_tN_r}=16$.
* Dimension check: $a_r$ is $16\times1$, $a_t^H$ is $1\times16$, so each term is $16\times16$; $\|a_ra_t^H\|_F=1$, so $E\|H\|_F^2=256\cdot\sum E|\alpha_l|^2=256$. Mean entry power is $1$ (DERIVATION; confirmed empirically: fig07 bottom-right shows the SNR-independent floor near 1).
* Consequence: the rank of $H$ is $L$ in the noiseless model; with $L>16$ impossible here since $L\le9$.

### 3.5 The 4x4 channel - fig05_channel_matrix_4x4.png
* Plan: a teaching figure with a small array (4x4 sub-array of the same construction) so that individual entries, the phase progression across a row, and the rank-1 structure are visible by eye; entries are drawn in the complex plane.
* Note: the 4x4 is a pedagogic reduction using the same formula with $N=4$; the real dataset is 16x16. It is not used for any number in this report.

### 3.6 TX/RX impairment - fig06_impairment.png
* Plan: show per-antenna phase error $\epsilon_k$, gain error $g_k$, the diagonal matrices, and their effect on the beam pattern; plus the label side (mean and linear ramp removed).
* Equation (FACT, `sim`): $F=D_tF_0$, $W=D_rW_0$, $D=\mathrm{diag}(g_ke^{j\epsilon_k})$, $\epsilon_k\sim U(-\delta,\delta)$, $20\log_{10}g_k\sim U(-\gamma,\gamma)$ dB. The same $D$ is shared by all beams (physically correct: a hardware fault is per antenna, not per beam).
* Observation (measured, Monte Carlo on the figure): the linear-ramp component of the phase error, which the label removes via `project_impairment`, is itself real physical content: a ramp $\epsilon_k=\beta k$ is a beam steering error. The apparent angle is shifted by $\Delta(\cos\theta)=\beta/\pi$. At $\delta=5^{\circ}$ the shift exceeds 1 degree for 2.07 % of path components (DERIVATION, Monte Carlo on fig06). This means the heat label (from the true angle) and the image peak (from the steered angle) are inconsistent for a small fraction of samples. CONCERN, Low.

### 3.7 Noise and SNR - fig07_noise_snr.png
* Plan: same scene with noise scale from 24 dB to -15 dB; noise-power check; realised versus nominal SNR; empirical noise floor in a real bank.
* Equation (FACT): $Z_{qp}\sim\mathcal{CN}(0,\sigma^2)$, $\sigma^2=10^{-\mathrm{SNR}/10}$, added after the combiner (in beam space). Per-component std is $\sqrt{\sigma^2/2}$: 0.04 at 24 dB, 0.71 at 0 dB, 3.98 at -15 dB (matches the panel titles; DERIVATION $\sqrt{10^{1.5}/2}=3.98$).
* Checks (panel bottom-left): measured $\mathrm{var}(\mathrm{Re}Z)$ against $10^{-\mathrm{SNR}/10}/2$ lies on $y=x$ at five SNRs, 2000 repeats each. Dimension and factor-of-two are correct.
* Observation: since $E|Y_{qp}|^2\approx1$ (section 3.4), the SNR is per-entry signal power (average over the whole map) over noise power; it is not per-path or per-peak SNR. A peak has power about $256\cdot|\alpha|^2/(\text{beam count spread})$, i.e. far above 1, whereas the average entry is near 1. FACT/INFERENCE.
* Realised per-sample SNR (middle panel; 3000 noiseless draws per $L$, at nominal 10 dB): the spread is substantial for $L=1$ (because $|\alpha_1|^2$ is exponential; a share of samples fall 10 dB and more below nominal) and narrows as $L$ grows. Across the training mixture, realised minus nominal SNR has mean -0.72 dB, median -0.36 dB, 1st percentile -10.8 dB, 95th percentile +2.97 dB (DERIVATION, 6000-draw replica, `a11_train_draw.py`). So "SNR = X dB" is an ensemble label, not a per-sample guarantee. CONCERN, Low-Medium.
* Bottom-right: the real bank T2_d5 mean $|Y|^2$ per SNR block follows signal(1) plus noise $10^{-\mathrm{SNR}/10}$ closely; this confirms the noise scaling in a stored bank as well as in the simulator.

### 3.8 Observation, clean versus impaired versus noisy - fig08_clean_impaired_noisy.png
* Plan: same scene through three pipeline variants and their difference maps.
* Clean case (FACT, `sim`): with $F_0,W_0$ the unitary DFT matrices the clean observation reduces to $Y=\mathrm{fft2}(H)/16$. Verified numerically: the two computations agree to floating point precision (cross-check passed). Both codebooks are unitary (verified, $\|U^HU-I\|\approx10^{-15}$).
* Impaired: peaks broaden and side lobes appear along the row and column of each peak (the impairment is shared by all beams, so leakage lines up along rows/columns).
* Noisy: white noise on top; see 3.7.

### 3.9 Beamspace map with ground truth - fig09_beamspace_with_gt.png
* Plan: overlay the ground-truth cells on real bank samples, and validate the formula with a quantitative check.
* Equation (FACT, `angles_to_cells`): row $q=n\,\mathrm{wrap}(-\pi\cos\psi)/2\pi$ (AoA), column $p=n\,\mathrm{wrap}(\pi\cos\phi)/2\pi$ (AoD), $n=16$ for the observation and $n=32$ for the heat label. Wrap is to $[-\pi,\pi)$ then mapped to $[0,n)$.
* Validation (DERIVATION, on real banks): mean $|Y|^2$ at the ground-truth cells is 17.3 dB above the map mean for T2_d5 at 25 dB and 17.3 dB for eval_bank at 25 dB; if AoA and AoD are swapped it is 8.2 dB (eval_bank), and for random angles it is 3.9 dB. The orientation `feat[0]=psi`, `feat[1]=phi` puts labels on peaks. The formula is correct, including sign, radian units and the fftshift-free indexing. VERIFIED (not a bug).
* Detail: a cell coordinate is continuous. A path centred between two pixels leaks into both (DFT scalloping); an x at coordinates $\ge15.5$ is drawn wrapped to row/column 0.
* Occupancy: the ground-truth cell density peaks at row/column 8 (about 3.8 times uniform; measured 23-24 % of cells within one bin of 8, predicted 22.7 %). Reason: $u=\cos\theta$ with $\theta$ uniform gives an arcsine density for $u$ concentrated near $\pm1$ (endfire), and $q\propto\mathrm{wrap}(-\pi u)$ maps... see 3.12 for the angle-space view. The mapping $\theta\to$ cell has $|dq/d\theta|=n\sin\theta/2$, which is small near $\theta=0,\pi$ (cells 0/8 depending on the sign convention), so many angles map to few cells. DERIVATION. This is a property of uniform-angle sampling, not an error, but it means the dataset is not uniform over beam cells. CONCERN, Low.

### 3.10 Beamspace transform and upsampling - fig10_beamspace_transform.png
* Plan: 16x16 to 64x64 (`upsample64`) and back (`downsample16`) and the exact-recovery check.
* Authors: the official eval_bank stores `data` at 64x64 (upsampled) and the pipeline down-samples to 16x16 before the network. `UP_SRC` is derived from `scipy.ndimage.zoom(order=0)`, i.e. nearest-neighbour replication.
* Observation: since 64/16 = 4 is an integer, one would expect uniform 4x4 blocks, but the `zoom` index map used has uneven block widths 3, 4, 5 (measured from `UP_SRC`). The mapping is nonetheless exactly invertible: 16 to 64 to 16 recovers the input to floating-point equality (verified on the figure). FACT/VERIFIED. Consequence: none for correctness; the 64x64 file is a redundant representation.
* Dimension check: `data` is (N,2,64,64); after `downsample16` (N,2,16,16), matching the network input `y` (B,2,16,16).

### 3.11 Heat target - fig11_heat_target.png
* Plan: show the 32x32 target built from continuous cells, the `max` across paths, wrap-around, and the histogram of the offsets.
* Authors (FACT, `heat_targets`): rounded cell (integer), Gaussian $\exp(-d^2/2\sigma^2)$ with $\sigma=1$ cell on a 32x32 grid, wrap-around distance on both axes, `max` over paths.
* Observations: (i) rounding the centre to an integer cell introduces a label quantisation error of up to 0.5 target cell (0.25 observation cell); the histogram of fractional parts in fig11 is uniform except for a spike near 0 (retitled accordingly) - INFERENCE that the spike is the arcsine effect of section 3.9 and endfire angles; (ii) `max` over paths hides overlapping paths; coincident heat cells occur in 0.33 % of T2_d5 samples (DERIVATION). CONCERN, Low.

### 3.12 Train, tune and test split - fig14_train_tune_test.png
* Plan: bar chart of sample counts on a log axis plus a seed table.
* Counts (FACT from file shapes): tune 1500 in total, T2_d1/d2/d5 8000 each, eval_bank 8000, v1 robustness 26,200 over 56 banks, training 48000 x 256 = 12,288,000 per CFG. The training count is steps times batch and is the upper bound on unique draws; the actual steps in the executed run are not verified. Insufficient evidence.
* Seeds (FACT): eval_bank 42, T2_d1/d2/d5 9101/9102/9105, tune banks 9901-9904.
* Seed reuse in the v1 manifest: `gain_g2_snr0` and `gain_g0.5_snr15` share seed 2020 with L=3 and n=500 and have identical angles, but different SNR and gain: a paired design, not independent test sets. `phase_d0_snr0` and `nuis_p0_snr0` share seed 1000 but have different L (3 versus 4) and no angle overlap was found. The seed-1000 and seed-1015 groups have different L and no overlap. T2 banks and eval_bank angles are pairwise different (checked). CONCERN, Low: pooled averages over v1 banks would over-weight the paired scenes.
* The dldoa seed-7 test set mentioned in report 01: no materialised file was found under `IABR_Net_TestSuite`; not plotted. Insufficient evidence.
* "47 of 56 used" is taken from report 01; it was not independently verified here.

### 3.13 Angle distributions - fig12_angle_distributions.png
* Plan: histograms of AoA and AoD in degrees, the joint scatter, the pairwise separation, from training replica and real banks.
* Training replica: `a11_train_draw.py` reproduces `make_batch` call order with my own seed (6000 samples); the real stream is `os.urandom`-seeded, so exact reproduction is impossible (FACT, `BatchStream`).
* Expected: $\psi,\phi$ uniform on $[0,\pi]$ marginally, slightly perturbed near the borders by rejection. Observed as expected.

### 3.14 Parameter distributions - fig13_parameter_distributions.png
* Plan: $L$, integer SNR, impairment $\delta$ and $\gamma$, clean fraction.
* Training (FACT, CFG at `notebook_src.py:82-92`): $L\sim U\{1..9\}$, SNR integer in $\{-15,\dots,24\}$, clean fraction 0.2, $\delta\sim U(0,8^\circ)$, $\gamma\sim U(0,3\text{ dB})$, grid 32, $\sigma_{heat}=1$, batch 256, 48000 steps.
* Test versus training coverage (FACT/DERIVATION):
  * eval_bank: $L=3$ only; SNR $\{-10,\dots,25\}$ step 5, 1000 samples per SNR. 25 dB is outside the training range (max 24). CONCERN, Low (1 dB beyond).
  * T2 banks: $\delta=1,2,5^\circ$, $\gamma=0$ (inside).
  * v1 robustness banks: include $\delta=10,15^\circ$ and $\gamma=4$ dB, beyond the training bounds $8^\circ$ and $3$ dB. This is an extrapolation test, presumably intentional; the tables should not describe it as in-distribution. CONCERN.
  * Bank L values: tune_L6 uses $L=6$, other banks $L=3$; the training range includes both.

### 3.15 Final dataset structure - fig15_final_dataset_structure.png
* Plan: cards for the training batch, the v2 bank, the official bank and the v1 bank with shapes and dtypes read from the files, plus one real sample (T2_d5 #7003) in Re, Im, magnitude.
* The network sees only the (2,16,16) image; angles, $L$, and SNR are held back for scoring. In the training batch no scene metadata is kept at all (`y`, `heat`, `imp`).

### 3.16 Beamspace closeness - fig16_beamspace_closeness.png
* Plan: for scenes with $L>1$, histogram of the minimum inter-path distance in beam cells (wrap-around Euclidean on the 16x16 grid), compared with the angle-space separation.
* Reason: the separation constraint is $\pi/6$ in $(\phi,\psi)$ (Euclidean), but two paths may differ mostly in one coordinate, and the cell coordinate depends on $\cos\theta$; near endfire a large angle difference is a small cell difference. So an angle-space rule does not guarantee beam-space resolvability.
* Result (DERIVATION, reproduced independently of the earlier audit):

| Set | closer than 0.5 cell | closer than 1 cell | closer than 2 cells |
|---|---|---|---|
| Training replica, $L>1$ | 4.66 % | 15.18 % | 42.4 % |
| eval_bank | 0.92 % | 3.15 % | 11.8 % |
| T2_d5 | 0.88 % | 3.15 % | 11.7 % |

  248 of 5324 replica training scenes ($4.7$ %) have angle-space separation of at least 30 degrees but beam distance below 0.5 cell. This reproduces the earlier audit's 0.92 % (official set) and 15 % (training) figures. The training distribution has more hard pairs (because $L$ up to 9 crowds the plane) than any test set (L=3), so the test difficulty is lower than the training difficulty in this respect. FACT/INFERENCE.

---

## 4. Concept visualization plan (for the reader)

| Concept | Figure | What the reader should see |
|---|---|---|
| 4x4 channel | fig05 | entries as complex numbers; a row shows a constant phase step; rank equals number of paths |
| Phase progression | fig03 | linear phase $-\pi k\cos\theta$ along the array; larger $|\cos\theta|$ means faster rotation and a beam farther from the centre |
| Complex plane | fig05, fig06 | $\alpha_l$, $g_ke^{j\epsilon_k}$ as points; impairment is a small rotation and radial scaling |
| Gain/phase impairment | fig06 | per-antenna scatter and pattern distortion; shared by all beams |
| Noisy versus clean | fig07, fig08 | white noise in beam space fills the map at low SNR |
| Beamspace heat map | fig09, fig10, fig11 | paths as spots plus leakage crosses; labels sit on the spots |
| Angle, SNR and path-count distributions | fig12, fig13 | uniform in angle, integer SNR, $L\in\{1..9\}$ |

---

## 5. Equation audit (dimensions, conjugates, normalisation, units, FFT)

| Item | Statement | Check | Result |
|---|---|---|---|
| Steering vector | $a_k=e^{-j\pi k\cos\theta}/4$ | length 16, unit norm, radians | OK |
| Channel | $H=16\sum\alpha_la_r a_t^H$ | 16x16, $E\|H\|_F^2=256$, mean entry power 1 | OK |
| Path gain | $\alpha\sim\mathcal{CN}(0,1/L)$ | total gain power 1 for all $L$ | OK |
| Observation | $Y=W^HHF+Z$ | $W^H$ is 16x16, $H$ 16x16, $F$ 16x16 | OK |
| Codebooks | $W_0,F_0$ unitary DFT | $U^HU=I$ to $10^{-15}$ | OK |
| Clean shortcut | $Y=\mathrm{fft2}(H)/16$ | agrees with $W_0^HHF_0$ | OK |
| Noise | var per entry $10^{-\mathrm{SNR}/10}$ | measured on $y=x$ (fig07) | OK |
| Cells | $q=n\,\mathrm{wrap}(-\pi\cos\psi)/2\pi$, $p=n\,\mathrm{wrap}(\pi\cos\phi)/2\pi$ | GT 17.3 dB above map mean | OK |
| Resample | 16 to 64 to 16 | exact recovery, uneven 3/4/5 blocks | OK, cosmetic |
| Impairment label | remove mean and linear ramp | ramp is physical steering error | CONCERN, Low |

Nothing here was assumed correct; each row was checked by script or by reading.

---

## 6. Problems and concerns, ranked (report of what was seen in the pictures)

1. Beam-space closeness of paths (fig16): CONCERN, Medium. 15.2 % of training scenes with $L>1$ have a pair closer than one cell; the heat target then merges two peaks under `max`, and the physics of the image makes them indistinguishable. The official test sets have 3.15 % below one cell. Test and train difficulty are therefore not matched.
2. Nominal SNR is an ensemble mean (fig07): CONCERN, Low-Medium. Realised minus nominal has a 1st percentile of -10.8 dB, so a "SNR = 0 dB" bin contains samples that behave like -10 dB.
3. White noise after the combiner (fig07): CONCERN, Low. Noise is not shaped by the impairment or the codebook; equivalent to a receiver noise at the beam output rather than at the antennas. With unitary $W_0$ and white antenna noise the result would be statistically equal for the clean case, but the impairment $D_r$ (with gain error) would colour it. Not visualised as a separate case.
4. Ramp shift (fig06): CONCERN, Low. 2.07 % of components shift the apparent angle by more than 1 degree at $\delta=5^\circ$.
5. Test SNR 25 dB and v1 impairments beyond training bounds (fig13): CONCERN, Low.
6. Seed pairing in the v1 manifest (fig14): CONCERN, Low.
7. Non-reproducible training stream: CONCERN, Low for correctness, relevant for audit. `BatchStream` seeds with `os.urandom`, so the exact training data cannot be regenerated.
8. Label quantisation and heat-cell coincidence (fig11): CONCERN, Low (0.33 % coincident in T2_d5).

No VERIFIED BUG was found in the dataset generation.

---

## 7. Not producible / insufficient evidence

* The dldoa seed-7 test file was not found, so it was not visualised.
* The step count of the actually executed training run is unverified (the CFG says 48000).
* The training histograms are a replica with my seed; the real stream cannot be reproduced.
* "47 of 56 v1 banks used" comes from report 01 and was not re-derived.
* Per-path SNR, as opposed to per-entry SNR, was not measured on stored banks because the stored banks do not contain $\alpha$.
* fig12, fig13 and fig16 were re-rendered after a margin fix and were not re-viewed afterwards; the numbers in this report come from the scripts, not from reading the images.

## 8. Figure index

| File | Concept | Data source |
|---|---|---|
| fig01_pipeline_overview | end-to-end pipeline | code (`sim`, `notebook_src.py`) |
| fig02_scene_paths | angles and path gains | `DG.generate_points`, `sim.sample_paths` |
| fig03_steering_vectors | phase progression | `sim` steering function |
| fig04_path_combination | sum of rank-1 terms | `sim` channel builder |
| fig05_channel_matrix_4x4 | small-array teaching case | same formula, $N=4$ |
| fig06_impairment | gain/phase error and ramp shift | `sim.simulate`, Monte Carlo |
| fig07_noise_snr | noise power and realised SNR | `sim.simulate`, T2_d5.npz |
| fig08_clean_impaired_noisy | pipeline variants | `sim.simulate` |
| fig09_beamspace_with_gt | GT overlay and validation | T2_d5.npz, eval_bank.npz |
| fig10_beamspace_transform | 16/64 resampling | `sim.upsample64`, `downsample16` |
| fig11_heat_target | 32x32 target | `sim.heat_targets` |
| fig12_angle_distributions | angle histograms | training replica, banks |
| fig13_parameter_distributions | $L$, SNR, $\delta$, $\gamma$ | CFG, replica, banks |
| fig14_train_tune_test | split and counts | npz shapes, MANIFEST.json |
| fig15_final_dataset_structure | shapes and one sample | make_batch probe, T2_d5.npz |
| fig16_beamspace_closeness | beam-cell distance | replica, eval_bank, T2_d5 |
