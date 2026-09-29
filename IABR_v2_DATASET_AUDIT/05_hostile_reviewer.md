# Agent 05 - Hostile Peer Review of the IABR-Net v2 Dataset-Generation Pipeline

Scope: dataset generation only (simulator, labels, banks, train/test philosophy). Model accuracy is out of scope except where a dataset property makes a reported number uninterpretable.
Labels: FACT / DERIVATION / INFERENCE / ASSUMPTION / CONCERN / VERIFIED BUG. "Insufficient evidence" is stated where it applies.
File shorthand: `SIM` = `IABR_v2_extracted/IABR_v2/iabr2_sim.py`; `NB` = `notebook_src.py`; `DG` = `dldoa_dataset_generation.py`; `ORIG` = `DL_DOA/src/tvt_data_generation_v3.py`; `INF` = `infer_dldoa_test.py`; scripts `a05_*.py` and outputs `a05_*.out` are in `scratch/`.

No code fixes are proposed. This is an investigation.

---

## 0. Executive verdict

1. FACT: The pipeline is deterministic and faithful to the authors' generator where it claims to be. Regenerating banks `T2_d1` and `tune_g3` is bit-exact; the simulator matches the original `generate_channel_v2` to 4.78e-15 (`a05_banks.out`). The official bank equals the authors' generator with seed 42 and the dldoa test set equals seed 7 (Agent 01).
2. FACT: The mathematically delicate parts (identifiability projection, steering conjugation, cos-mapping to cells, up/down-sampling) are correct as coded (`a05_geom.out`; projection residual is exactly linear-phase and flat-gain to 1e-8 to 1e-15).
3. CONCERN (High): The simulator is a *matched-model* world. Simulator, physics decoder (NOMP + LS gains) and the impairment head all use the same narrowband, iid-per-antenna, half-wavelength-ULA steering model. A physics-only NOMP with no network matches or beats the full IABR-v2 (0.7581 vs 0.7551 on official; 0.7564 vs 0.7531 on the dldoa set; network-only decode 0.4475; `INF`, `infer_dldoa_test.log`, `a05_results.out`). Any "improvement" measured on this dataset is therefore an improvement on the simulator's own model, not on hardware.
4. CONCERN (High): The Table 2 comparison is not like-for-like. The v2 model trains on impairments drawn from U(0,8 deg) and U(0,3 dB) with 20 percent clean, so the test values d in {1,2,5} deg are inside its training distribution. The Meneses base U-Net was trained with no impairments (Meneses Table 1: "Pre-trained U-Net (no simulated impairments)"). The fine-tuned per-delta models are the like-for-like comparator, and those are the ones that use 10,000 samples.
5. CONCERN (Medium): The nominal SNR does not describe per-path detectability. With alpha ~ CN(0,1/L) and unit-norm-ish steering, the weakest of three paths carries a median 9.7 percent of total power (10th percentile 1.7 percent). The Pd ceiling (about 0.95 at 25 dB in the paper, 0.75 mean here) is set by fading draws, not by the estimator.
6. VERIFIED BUG: None found in the simulator arithmetic. One documentation-level inconsistency (paper text L 1..10, SNR -10..25 for training vs code L 1..9, SNR -15..24) is a paper/code discrepancy, not a code bug.

---

## 1. What the authors actually did (compact ground truth)

### 1.1 Channel and observation
- FACT: $H=\sqrt{N_TN_R}\sum_{l=1}^{L}\alpha_l\,a_r(\psi_l)a_t^H(\phi_l)$, $a(\theta)=\tfrac{1}{\sqrt N}[1,e^{-j\pi\cos\theta},\dots,e^{-j\pi(N-1)\cos\theta}]^T$ (`SIM:45-47`; Meneses Eq. 1-3).
- FACT: $Y=W^{H}HF+Z$ with $P=Q=16$ square unitary DFT codebooks `F_IDEAL, W_IDEAL` (`SIM:11-13,53`). `einsum('bnq,bnm,bmp->bqp', conj(W), H, F)` is $W^HHF$; the shape is (Q,P) = (rx-beams, tx-beams) with rows = AoA, columns = AoD.
- FACT: $\alpha_l\sim\mathcal{CN}(0,1/L)$, then sorted by magnitude, strongest first (`SIM:35-36`).
- FACT: Noise: each real and imaginary part has variance $\sigma^2/2$ with $\sigma^2=10^{-\mathrm{SNR}/10}$ (`SIM:55-56`). SNR is therefore $\rho/\sigma^2$ with $\rho=1$, computed per sample against the *expected* signal power, not the realized one.
- FACT: Angles are drawn by the original `DG.generate_points` with minimum Euclidean separation $\pi/6$ (`SIM:28-37`); angles lie in $[0,\pi]$ (Lloria Sec. IV-B), paths jointly rejection-sampled.

### 1.2 Training stream (`make_batch`, `SIM:107-117`)
- L ~ U{1..9} (`CFG train_L`), SNR ~ U{-15..24} integer dB, independent of L and angles; 20 percent clean (`clean_frac`); otherwise phase max ~ U(0,8) deg and gain max ~ U(0,3) dB *per sample*, then per-antenna errors uniform in $\pm$ that max.
- FACT: Per-antenna phase $\epsilon_i\sim U(-\delta,\delta)$ and gain $g_i=10^{U(-\gamma,\gamma)/20}$, iid across Rx and Tx antennas, one realization per sample, applied identically to every codebook column (`SIM:49-52`): $W=D_rW_{ideal}$, $F=D_tF_{ideal}$.
- FACT: Endless stream, RNG seeded with `[seed, worker_id, os.urandom(4)]` (`SIM:127`), i.e. not reproducible run to run.

### 1.3 Labels
- FACT: Heat target: $32\times32$ wrap-around grid, Gaussian $\sigma=1$ cell, peak 1 exactly at the rounded cell (`SIM:92-104`). Cell coordinates: rows $-\pi\cos\psi$, columns $+\pi\cos\phi$ wrapped to $[0,2\pi)$ (`SIM:71-73`).
- FACT: Impairment target: $32\times2$ (phase, log-gain) per antenna in the antenna domain with unidentifiable parts removed: mean (both) and linear ramp (phase only) (`SIM:76-89`). Sign conventions: rows use $-\angle D_r$ (from $\tilde H=\mathrm{conj}(D_r)HD_t$), columns $+\angle D_t$.
- FACT: The ramp is removed for phase but not for gain (gain has no ramp degeneracy; correct).

### 1.4 Evaluation banks
- FACT: `official` (authors' generator seed 42, frozen), dldoa test (seed 7, L=3, 8 SNR x 1000), `T2_d{1,2,5}` (seeds 9100+d, 8 SNR x 1000, phase-only impairments), 4 tuning banks (seeds 9901-9904, 150 per cell), 47 v1 robustness banks (500 each) (`NB:346-409`).
- FACT: Metrics: paper Pd (samples with fewer than L detections dropped), strict Pd, RMSE over errors $\le1^\circ$ after Hungarian matching.

---

## 2. Decision-by-decision interrogation

For each: intent / justification / alternatives / effect / evidence / bias.

### D1. Number of paths L ~ U{1..9}, test at L=3 only
- Intent (INFERENCE): copy the original generator's `randint(1,10)` (`ORIG`; `DG`). Paper text says 1..10 (Lloria IV; Agent 01), code 1..9. FACT: discrepancy paper vs code; `randint(1,10)` excludes 10.
- Scientific justification: none given for a uniform law on L. Measured mmWave channels (28/60 GHz) commonly show 2-5 clusters; L uniform to 9 puts 44 percent of mass at L in {6..9} where the 16x16 codebook (16 beams per dimension, resolution ~ 0.14 beam cell at broadside per degree) is heavily loaded. CONCERN (Medium): ASSUMPTION that L is a design choice, not a physical statistic; Insufficient evidence for a physical derivation.
- Effect: FACT (`a05_geom.out`): the rejection sampler with $\pi/6$ separation is non-uniform at high L; 12.6 percent of coordinates fall within 10 deg of endfire at L=9 vs 11.1 percent for a uniform marginal. High-L samples are also nearly unresolvable when $\pi/6$ separation is in angle but resolution is in $\cos$-domain (endfire compression, see D3).
- Test/train mismatch: FACT: all Table-2 style evaluation is L=3. The model trains on 1..9, so the headline number never tests the L=1,2 or L>=4 regime that the training distribution spends 89 percent of its mass on. CONCERN: Table 2 says nothing about the training L law; the training L law says nothing about Table 2.
- Alternative: match the train L law to the test law (L=3) or report per-L. Bias: mixed-L training diffuses capacity; the reported (L=3) mean is unaffected in either direction by a way that can be inferred - Insufficient evidence.

### D2. SNR law: U{-15..24} training, -10..25 dB test
- Intent: cover the paper grid. FACT: training excludes 25 dB (`train_snr` upper bound 24) although the test grid includes 25 dB. CONCERN (Low): edge extrapolation of 1 dB is benign for a network but the physics decoder has no such limit. The paper text states -10..25 (FACT, Lloria).
- Definition: FACT: SNR is $\rho/\sigma^2$ with $E|\alpha|^2$ normalized to 1 and codebooks unitary, so $E|Y_{qp}|^2=1/(NR)\cdot$... measured mean $|Y|^2 = 1+10^{-\mathrm{SNR}/10}$ (`a05_results.out`, VERIFIED). Note the per-element signal power is 1 only on average across a codebook; the realized power varies.
- VERIFIED (`a05_results.out`): per-sample noiseless power at L=3 has 5/50/95 percentiles of -5.7/-0.5/+3.3 dB, and at L=1 -13.2/-1.6/+4.8 dB. The "SNR" axis is therefore a *distributional* label; sample-level effective SNR spreads over ~9 to 18 dB around it. DERIVATION: because $|\alpha|^2\sim\mathrm{Exp}$ (mean 1/L each), the strongest path's power is $\ge$ the mean, the weakest is far lower.
- Consequence: Pd(SNR) saturates (0.95 at 25 dB) because of fading, not because of noise: at L=3 the weakest path has median share 0.097 of total power, median absolute power -11.2 dB, so it sits below the noise floor for SNR < ~10 dB in a fraction of samples independent of any estimator. CONCERN (Medium): using the same SNR label for L=1 and L=9 hides that per-path SNR falls as $10\log_{10}L$ (equal-power case).
- Alternative: per-path SNR, or condition on the weakest path. Bias: makes every estimator look worse at low SNR by the same amount; it compresses differences between methods (see D15).

### D3. Angle distribution (uniform in [0, pi] in angle, not in cos)
- FACT: Angles uniform in $\theta\in[0,\pi]$ (Lloria IV-B; Section II says $[0,2\pi]$ - internal inconsistency in the paper, FACT). With the cos mapping, $u=\pi\cos\theta$ is arcsine-distributed, so density piles up near endfire.
- VERIFIED (`a05_geom.out`, official bank): 30 percent of samples have an angle within 5 deg of endfire, 53 percent within 10 deg. (Uniform in angle would give ~ 5.6 percent per coordinate within 5 deg; multiple coordinates per sample compound it.)
- DERIVATION: $du/d\theta=-\pi\sin\theta$, so a 1 deg error is 0.14 beam cell at broadside (theta=90) but 0.024 at 10 deg and 0.005 at 2 deg from endfire (VERIFIED numerically). The paper's "error <= 1 deg" success criterion therefore has an angle-dependent difficulty; near endfire the criterion is nearly free, at broadside it is the hardest (hardest is 1 deg at broadside = 0.14 cell). CONCERN (High): the headline Pd/RMSE mixes a physically easy regime (endfire) with a hard one and the 30 percent endfire mass inflates both.
- Endfire ambiguity: FACT: $|a(5^\circ)^Ha(175^\circ)|=0.994$; at 10 deg it is 0.906. The ULA cos-mapping makes $\theta$ and $\pi-\theta$ identical *if* the range is $[0,2\pi]$; on $[0,\pi]$ the mapping is one-to-one but adjacent-to-degenerate near endfire. So the model must resolve ~0.006 cell differences there.
- Alternatives: uniform in $\cos\theta$ (physically isotropic scattering in the array plane is uniform in angle on the circle, but the reader would then note that endfire is de-emphasized). Bias: the angle-uniform choice creates the endfire mass that makes the 1-degree criterion easy for ~30 percent of samples.

### D4. 16 antennas; P=Q=16 square DFT codebooks (vs original random P in {16,32}, nt=nr in {16,32})
- FACT: The original generator draws P,Q ∈ {16,32} and nt=nr ∈ {16,32} (`ORIG`). v2 fixes P=Q=N=16 (`SIM:11`).
- Intent (INFERENCE): the U-Net input 64x64 = 4x16 nearest upsampling, and the official bank is 16x16.
- Effect: FACT/DERIVATION: with P=Q=N the observation is a *unitary transform* of $H$ (no compression), so $Y=W^HHF$ is invertible in noise-free case: the DFT beamspace image is a complete representation. This is the easiest observation model. The original P=32/N=16 case is an *oversampled* codebook (columns nonorthogonal) and P<N is *compressed sensing*. CONCERN (High): the reported performance is for the case where compressive-sensing difficulty is absent; hardware impairments then act on a lossless observation.
- Consequence for impairments: with unitary codebooks and multiplicative diagonal $D$, $Y=(D_rW)^HH(D_tF)$ exactly equals $W^H\tilde HF$ with $\tilde H=\mathrm{conj}(D_r)HD_t$ (`SIM:83-84`), so the impairment is *exactly invertible in the antenna domain* if $D$ were known. This is what makes the impairment head's target well-defined and the IABC step correct. It also means the impairment task is trivially separable, which is unlike real analog arrays where each beam column uses distinct phase-shifter states.

### D5. Impairment model: phase 0-8 deg, gain 0-3 dB, 20 percent clean
- Intent: cover Meneses $\delta_{max}\in\{1,2,5\}$ with margin (8 > 5) and add gain error the paper does not model (Meneses: amplitude only as future work; FACT).
- Justification: Meneses gives 1 deg = high-quality, 5 deg = pessimistic, "exclude higher values as infeasible". v2 trains up to 8 deg and gain up to 3 dB. CONCERN (Medium): 3 dB amplitude error has no cited source in the code or paper; Insufficient evidence for the range.
- Statistics (VERIFIED, `a05_results.out`): since $\delta\sim U(0,8)$ per sample and clean 20 percent, 30 percent of samples have $\delta<1^\circ$ (including clean), and 30 percent have $\delta\ge5^\circ$. RMS per-antenna error over the whole training distribution is 2.385 deg; test d=5 has RMS 2.888 deg. So d=1,2,5 are all interior to the training law.
- Gain: $E[g^2]=1.0817$ for U(-3,3) dB, i.e. the gain jitter adds an average +0.68 dB *effective SNR shift* to impaired samples (VERIFIED). CONCERN (Low): impaired samples are on average louder than clean ones at equal nominal SNR; a confound in any clean-vs-impaired comparison.
- Why uniform, why iid: uniform is the paper's model (FACT: Lloria/Meneses Sec. 2.2). It corresponds to a quantization-like bounded error; real phase-shifter errors are dominated by systematic terms (finite-resolution quantization on a lattice, temperature-dependent drift, per-state-dependent errors). iid per antenna is stated by the paper ("independent across antennas"). CONCERN (High): independent errors have no spatial correlation, so the *smooth* (low-order polynomial) component of the error - the part that shifts and defocuses beams - is small. The dangerous physical errors (linear ramp = pointing bias, quadratic = defocus) are precisely those iid uniform draws barely produce.
- Same error on every codebook column: FACT (`SIM:52`). Real analog beamforming programs a different phase-shifter *state* per codeword, and static per-element errors are only one component of the error (state-dependent error dominates in real phase shifters). With per-column-identical error, the imperfection collapses to a per-antenna scalar - an exactly antenna-domain diagonal distortion. CONCERN (High): this is the modelling choice that makes the problem solvable by IABC and makes NOMP+LS gains near-perfect. Under state-dependent errors, no antenna-domain correction exists.
- Independent Rx and Tx errors with the same $\delta$: FACT. Real Tx and Rx chains have different hardware; using the same max value for both is a simplification (ASSUMPTION).
- 20 percent clean: INFERENCE: prevents forgetting the clean regime. No ablation of the fraction in the package (Insufficient evidence).
- Bias on results: the impairment magnitude at test (d<=5 deg, iid) is small enough that the paper itself reports negligible degradation (Meneses Table 2: Pd 0.95 -> 0.93 at 25 dB). VERIFIED (`a05_results.out`, derived below) that the irreducible effect on the angle is at most a ramp-induced bias: the ramp is a pure angle shift.

### D6. Unidentifiable impairment components and the irreducible RMSE floor
- FACT: Removed: mean phase, mean log-gain (global scalars) and linear phase ramp (equivalent to an angle shift of each path; $e^{-j\pi\cos\theta n}\cdot e^{j\beta n}$ = steering at a shifted $\cos\theta$).
- DERIVATION/VERIFIED: the ramp coefficient of iid $U(-\delta,\delta)$ phases has std $\delta\sqrt{3}\cdot\ldots$; numerically the angle-bias std at $\delta=1,2,5,8,15^\circ$ is 0.010/0.020/0.050/0.080/0.149 deg at broadside and 0.057/0.115/0.287/0.459/0.860 deg at 10 deg from endfire (`a05_results.out`). The removed-mean-in-degrees std is 0.145/0.289/0.723/1.157/2.17.
- Effect: CONCERN (Medium): at $\delta=5$ deg and 10 deg from endfire the irreducible bias std is 0.29 deg, which is 29 percent of the 1 deg success threshold, for an estimator that cannot see the ramp *by construction*. Inflating $\delta$ or evaluating near endfire converts irreducible bias into apparent Pd loss. This bias is identical in every method, so it lowers absolute Pd but cannot rank methods - unless the method uses knowledge of the prior on the ramp (a Bayesian shrinkage prior would); Insufficient evidence that any does.
- The target definition (projected) is FACT-correct: residual is rank-1 removed exactly (err ~1e-8 to 1e-15). Remaining question: whether the *network* head is trained on a target that is a deterministic function of the input - yes, only up to the removed subspace, which is right.

### D7. Order of operations and normalization
- FACT: order: draw paths -> $H$ -> apply $D$ to codebooks -> $G=W^HHF$ -> add noise (noise is added *after* the impairment, i.e. white in beam domain and unaffected by gain errors). Real receivers apply the combiner gain errors to the noise too ($w^Hn$ with distorted $w$): noise would be colored by $|D_r|^2$ and correlated across beams. CONCERN (Medium): with unitary $W$ and $g_i\neq1$, $W_D=D_rW$ is no longer unitary: noise after combining should have covariance $W_D^HW_D\neq I$. The simulator adds i.i.d. noise of fixed variance regardless of the gain errors. FACT that this is what the code does (`SIM:53-56`, noise added to `G`). Effect: at gain 3 dB the true noise variance would change by a few percent relative to what is simulated; small but a fidelity gap, and it also removes the noise-correlation cue that real data would carry.
- Normalization of the input: no per-sample normalization of $Y$ was found in the simulator (FACT: `to_ri` casts to float32 only, `SIM:63-64`). The absolute scale of $Y$ is informative of SNR and of $L$ (mean $|Y|^2=1+\sigma^2$). Insufficient evidence on the network-side scaling (out of dataset scope).
- Heat target computed from continuous angles by rounding to a $32$-grid, then the peak is exactly 1 (`SIM:96-100`). Oversampling factor 2 relative to the 16-beam grid; sigma=1 fine cell = 0.5 beam cell. Paper: 256x256 with $\sigma=0.07$ rad (FACT). CONCERN (Medium): rounding quantizes the label to half a beam cell; at broadside that is ~3.6 deg of angle, at 10 deg from endfire ~21 deg. The regression to sub-cell angles is done later by NOMP, so the heat target is only a candidate generator. The heat peak lies within 0.5 beam cell of the $|Y|$ argmax in 500/500 checked cases (VERIFIED, `a05_geom.out`) - consistent, but shows the label is basically a copy of the DFT peak location.
- Collisions: VERIFIED: in training-like data 0.4 percent of paths are lost to same-cell collisions and 12.2 percent of samples have adjacent-cell path pairs (risk of NMS merge). Label noise of this kind is small but it affects only high-L samples, which are absent from the L=3 test.
- Sigma mismatch with the paper (0.07 rad vs 1/32 cell ~ 0.196 rad): FACT. The paper's target is on a $256^2$ grid over $[0,2\pi)$: 0.07 rad = 2.86 cells of 0.0245 rad. v2's $\sigma=1$ cell of 32 = 0.196 rad, 2.8x wider in angle. So the labels are not the paper's.

### D8. Independence and correlation of variables
- FACT: L, SNR, clean flag, $\delta$, $\gamma$, angles, $\alpha$ are drawn independently (`SIM:108-114`). The only coupling is the angle rejection sampler.
- CONCERN (Low/Medium): $\delta$ and $\gamma$ are independent of each other (a bad device would have both), and both independent of SNR (in reality calibration error interacts with power settings). ASSUMPTION: independence is a simplification; not testable here.
- FACT: $\alpha$ is sorted by magnitude *after* being drawn, but the angle/order pairing is by index. The angle draw and $\alpha$ draw are independent, so sorting does not bias the angle marginals. But the *labels* (psi, phi, order) are stored in strongest-first order, which is informative only if the network is asked to output ordered lists (Hungarian matching removes it). No issue for evaluation.

### D9. Train/test philosophy: infinite stream vs fixed banks
- FACT: training draws fresh samples forever with an `os.urandom` component in the seed (`SIM:127`). There is no fixed training set, no held-out validation set from the same generator, and final-checkpoint selection (Agent 01/06; INFERENCE). CONCERN (Medium): irreproducible training data stream; two runs with the same `seed` are different data. This blocks exact reproduction of the trained model's numbers, so any headline number has a training-noise variance that is not estimated (Insufficient evidence: no repeat-training results exist).
- Advantage (FACT): an infinite stream eliminates overfitting to a fixed set - a real benefit.
- Leakage between train and test: Insufficient evidence for leakage of angle values (continuous, infinite space); the official bank (seed 42) can in principle be reproduced by the stream only with negligible probability. Not a concern.
- Test set contamination through tuning: FACT: `tune_selfcal` selects (lam,tau)=(1.0,5.0) on 4 tuning banks with 150 samples per cell, seeds 9901-9904 (`NB:811-832`). The reported evaluation banks are seeded differently (9100+d, 42, 7), so no direct leakage. But CONCERN (Medium): the selection is noise-limited (below).

### D10. Sample counts and their statistical power
- 1000 per SNR cell: VERIFIED (binomial): SE of Pd up to ~0.016 near Pd=0.5 at n=1000 per cell. Pd is computed only on samples with $\ge L$ detections, so the effective n is smaller. CONCERN (Medium): differences of 0.002-0.02 between IABR-v2, NOMP and U-Net (0.7531 vs 0.7564 vs 0.7354 mean over 8 cells) are within or near the noise for a single cell, and only partly averaged in the 8-cell mean (8000 samples; mean SE ~0.005). The v2-vs-NOMP difference (0.0033) is < 1 SE: Insufficient evidence that IABR-v2 differs from NOMP at all. Paired tests are possible because both were run on the same samples; none are reported (FACT).
- 500 per robust bank: SE ~0.022 near 0.5; 47 banks with duplicated seeds (below) so the "47 points" are not 47 independent draws.
- 150 per tuning cell: SE about 0.03, comparable to differences among candidate configs (0.002-0.02; VERIFIED). The chosen config's mean is 0.7870 vs 0.7812 without self-calibration, a 0.006 gain, i.e. within selection noise. CONCERN (Medium): the self-calibration hyper-parameters are essentially chosen by noise; the "gain" reported for that step should not be trusted without a paired test on fresh data.
- Meneses fine-tuning uses 10,000 samples per delta and a 4,800-sample validation grid (FACT, Meneses Table 1). v2's effective training samples are orders of magnitude more (stream). Not like-for-like.

### D11. Seeds and bank duplication
- VERIFIED (`a05_banks.out`): v1 bank seeds collide: 2020 (`gain_g0.5_snr15` and `gain_g2_snr0`, identical psi/phi), 1000 and 1015 (`phase_d0_*` vs `nuis_*`). Effect: shared angle/alpha draws across banks make the robustness curve points correlated, and at the same seed different SNR/gain reuse noise-free signal; conclusions of "trend in gain" compare the same geometry - that is actually a *variance-reducing* pairing, but it is undisclosed and it makes the reported curve non-independent.
- FACT: Official = seed 42 of the authors' generator, T2 = 9100+d with a *different* generator (v2's `make_bank`), dldoa test = seed 7 of the authors' generator. CONCERN (Medium): Table-2 style banks (`T2_d*`) are generated by the reimplementation and not the authors' script; agreement of the reimplementation with the original is bit-level *for the noiseless model at matched seed* (4.78e-15), but the impairment draw order/RNG stream differs from Meneses' unpublished generator (Insufficient evidence on Meneses' RNG, distributions besides "uniform").
- Meneses applied the phase error to "the effective phase shift applied to the i-th element of both the Tx and Rx antenna arrays" (FACT text). Whether it acts on the steering vectors (channel) or on the codebook (hardware) is not specified in the paper text I read; v2 puts it on the codebook. Under the alternative (error on the array response) the effect on $Y$ is $\tilde D H\tilde D$, with conjugation differences on Rx. Insufficient evidence which was used by the authors; the sign convention of the Rx error (conjugated in $\tilde H$) is a modelling choice with no effect on distribution since $\epsilon$ is symmetric.

### D12. Label definition and identifiability
- FACT: Label = (heat map of rounded cells) + (projected impairments). Identifiable parts only. The removal of the global phase and of the ramp is mathematically correct; the residual is well-defined.
- CONCERN (Medium): the heat label encodes the *true* angle, but the input already contains the ramp-induced shift, so the label is not a function of the input: irreducible label noise of std given in D6, larger near endfire. The training loss thus contains an irreducible component that is heteroscedastic in angle and $\delta$. Not a bug; a property.
- CONCERN (Low): focal-loss positives only at the rounded cell (FACT, `SIM:96-100` peak exactly at one cell; NB focal loss) means neighbouring cells with 0.6 Gaussian value are negatives with soft weights; with rounding, a path at cell-boundary (0.5 offset) has two equally good neighbours but one is labelled positive. Effect on the dataset: unavoidable +/-0.5 cell label ambiguity for a fraction of paths; effect on results limited because NOMP refines.

### D13. Does the simulator represent real hardware?
- Not represented (FACT by reading `SIM`): quantized phase shifters (finite bits), phase-state-dependent errors, mutual coupling, frequency-dependent beam squint / wideband, non-uniform element patterns, array position errors (only ideal $\lambda/2$), I/Q imbalance, ADC quantization, phase noise / CFO across the 256 pilot slots (all 256 measurements share the same channel and the same static $D$), channel time variation, near-field, diffuse scattering (only L specular paths), Rician LOS factor, non-Gaussian path gains (Rayleigh only), correlated clusters (rays with angular spread), and AoA/AoD pairing structure (independent draws of $\psi_l,\phi_l$; physically AoA and AoD of the same path are geometrically coupled in a reflective environment).
- INFERENCE: therefore the simulator is a *synthetic benchmark* consistent with the paper, not a hardware proxy. Any claim about "hardware-impairment robustness" from this dataset is a claim about robustness to iid multiplicative antenna-domain errors, which the authors' own decoder inverts by construction (NOMP + LS gain). The 2.4 to 2.9 deg RMS errors are in the regime where the paper shows only 1-2 percent Pd change - Insufficient evidence that this regime discriminates methods.
- Frame: phase-shifter errors of 5 deg RMS-scale are often quoted as *good* hardware (5-6 bit quantization gives 2.8-5.6 deg steps); the paper's "pessimistic" label is questionable (ASSUMPTION; Insufficient evidence in-package).

### D14. Fixed vs random parameters
- Fixed: N=P=Q=16; $d=\lambda/2$; unit transmit power; $\sigma$ tied to SNR; up-sampling factor 4; grid 32; sigma 1.
- Random: L, SNR, clean flag, $\delta$, $\gamma$, angles, $\alpha$, antenna errors, noise.
- CONCERN (Medium): random per-sample $\delta$ but fixed *distribution family*. A model trained on this stream has never seen (a) correlated errors, (b) $N\neq16$, (c) $P\neq Q$, (d) $\rho$ scale drift, (e) other array spacing. The original generator's own P,Q,N randomization (used in the paper's training) is removed here, so v2 is a narrower task than the paper's. The comparison to the official U-Net weights, trained on the broader family, is therefore evaluated only on the sub-family where the official model is tested (16x16), which is fair to the U-Net, but the reverse claim ("v2 generalizes") is unsupported by the data.

### D15. Simulator-decoder circularity and what "beating the baseline" means
- FACT: NOMP shares the exact steering model, unitary codebooks and Gaussian noise. With $P=Q=N$, the ML estimator is a nonlinear least-squares fit of this model; NOMP is close to ML. Since the data are generated from this model, the ML estimator is near the Cramer-Rao bound apart from fading-induced misses.
- VERIFIED-consistent: NOMP no network 0.7564 vs v2 0.7531 on dldoa; 0.7581 vs 0.7551 on official. DFT-SIC 0.7313; U-Net 0.7354. Spread across all methods: 0.7313-0.7564, i.e. 2.5 percent. This is the same order as the Pd standard error (D10) and the fading ceiling (D2). CONCERN (High): the dataset is a *saturated* benchmark for L=3, P=Q=N=16; most methods sit within a few points of the model-based bound, so the benchmark has very little ability to separate a neural estimator from a classical one. That is a property of the dataset (matched, unitary, low-dimensional), not of the models.

### D16. Paper-code discrepancies affecting the dataset (FACT)
| Item | Paper (Lloria/Meneses) | Code | Comment |
|---|---|---|---|
| L range | 1..10 | 1..9 (`randint(1,10)`) | off-by-one |
| SNR range | -10..25 | -15..24 (`randint(-15,25)`) | code wider on the low side; 25 not trained |
| Angle range | Sec II: [0,2pi]; Sec IV-B: [0,pi] | [0,pi] | paper self-inconsistent |
| Heat sigma | 0.07 rad on 256x256 | 1 cell on 32x32 (0.196 rad) | v2 differs |
| Impairment | phase only, delta in {1,2,5} | phase + gain, 0..8 / 0..3 dB, 20% clean | v2 superset |
| Codebook | P,Q in {16,32} | P=Q=16 | v2 restricted |

---

## 3. Problems ranked

| # | Finding | Label | Severity |
|---|---|---|---|
| 1 | Simulator/decoder matched model; NOMP (no net) >= IABR-v2 on both banks | CONCERN | High |
| 2 | Table 2 not like-for-like: v2 trained on impairments including test range; Meneses base model has none | CONCERN | High |
| 3 | Per-column-identical, iid, antenna-domain diagonal error is trivially invertible; not hardware realistic (no state dependence, quantization, coupling) | CONCERN | High |
| 4 | Angle-uniform sampling puts 30/53 percent of samples within 5/10 deg of endfire where the 1 deg criterion is nearly free (0.005-0.024 cell) | VERIFIED (numbers), CONCERN (impact) | High |
| 5 | Square unitary P=Q=N=16 removes compressive difficulty, narrowing the paper's P,Q,N family | CONCERN | High |
| 6 | Nominal SNR vs realized per-path SNR: weakest path median share 9.7 percent; Pd ceiling set by fading | VERIFIED | Medium |
| 7 | Irreducible ramp bias: 0.29 deg std at d=5, 10 deg from endfire (29 percent of the 1 deg gate) | VERIFIED | Medium |
| 8 | Noise added after impairment; not colored by non-unitary distorted combiner | CONCERN | Medium |
| 9 | Self-calibration hyper-parameters selected on 150-sample cells (SE 0.03, differences 0.002-0.02); reported gain 0.006 | VERIFIED (arith), CONCERN | Medium |
| 10 | Pd differences between methods (<= 0.025 mean; v2 vs NOMP 0.003) are within sampling noise; no paired tests | CONCERN | Medium |
| 11 | Duplicate v1 seeds (2020, 1000, 1015): robustness points not independent | VERIFIED | Medium |
| 12 | Non-reproducible training stream (`os.urandom`), no held-out val set, final-checkpoint selection | FACT/CONCERN | Medium |
| 13 | Paper-code discrepancies (L, SNR, angle range, sigma) | FACT | Low/Medium |
| 14 | Gain impairment adds ~0.68 dB average effective SNR to impaired samples | VERIFIED | Low |
| 15 | Rejection sampler non-uniform at high L (12.6 vs 11.1 percent near endfire at L=9); 3.9 percent of L=3 official samples have a pair within one beam cell in both dimensions | VERIFIED | Low |
| 16 | Heat target quantization (half beam cell), 0.4 percent collisions, 12.2 percent adjacent-cell samples | VERIFIED | Low |
| 17 | Test L=3 only vs training L 1..9; 25 dB outside training range | FACT | Low |

## 4. Insufficient evidence
- The generator that produced the zip's train/val splits (Agent 01, H5), and the exact copy of `dldoa_dataset_generation.py` on the run machine (three local copies identical, hash 7812c91f7cc5773e).
- Meneses' impairment RNG, the exact placement of the error (channel vs codebook), and how L and SNR were distributed in the 10,000-sample fine-tuning set.
- Whether the 20 percent clean fraction and the 8 deg / 3 dB maxima were tuned or arbitrary: no ablation in the package.
- Training-run variance (no repeats).
- `a05_evalbank.py` output was empty at report time; no claim in this report depends on it.

## 5. Bias summary for the final results
- Upward bias on absolute Pd/RMSE: endfire mass (D3), impairment inside training support (D5), unitary codebooks (D4), matched model (D15).
- Downward bias on absolute Pd: fading ceiling with weak paths (D2), irreducible ramp bias (D6).
- Ranking ability: low. All strong methods (NOMP, IABR-v2, U-Net, DFT-SIC) fall within 0.73-0.76 mean Pd; differences of 0.003-0.02 are not distinguishable from sampling noise without paired tests.
- External validity: the dataset supports the claim "IABR-v2 performs like a model-based decoder on a simulator of its own model". It does not support a claim about real hardware.
