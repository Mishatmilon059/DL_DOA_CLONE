# Agent 07 - Data Leakage / Experiment Integrity Audit of the IABR-Net v2 dataset pipeline

Scope: dataset generation and evaluation protocol only (not model accuracy). Investigation only: no project file was modified, no fixes are proposed, no git was run. All scratch work is in `D:\ai_ml_project\IABR_v2_DATASET_AUDIT\scratch\` (scripts `a07_s1_banks.py`, `a07_s1b.py`, `a07_s2b_cache_only.py`, `a07_s3_train_vs_banks.py`; outputs `a07_s1_out.json`, `a07_s2b_out.json`, `a07_s3_out.json`).

Label legend: FACT (read in code or measured), DERIVATION (arithmetic / logic from facts), INFERENCE, ASSUMPTION, CONCERN, VERIFIED BUG. "Insufficient evidence" is used where I could not check.

Paths: `NB` = `D:\ai_ml_project\IABR_v2_extracted\IABR_v2\notebook_src.py`; `SIM` = `...\IABR_v2\iabr2_sim.py`; `OUT` = `...\IABR_v2\outputs`.

---

## 1. What the authors did (data flow, as implemented)

1. **Training data** are never stored. `BatchStream` (SIM:120-129) yields endless batches from `make_batch` (SIM:107-117). Every DataLoader worker seeds `np.random.default_rng([seed, worker_id, os.urandom(4)])` (SIM:127). Per scene: $L\sim U\{1..9\}$, SNR $\sim U\{-15..24\}$ dB (integer), 20 % clean, otherwise phase $U(0,8^\circ)$ and gain $U(0,3)$ dB; angles from `DG.generate_points(L, \pi/6)`; $\alpha\sim\mathcal{CN}(0,1/L)$; $Y = W^H H F + Z$ with per-antenna $D_r, D_t$ on both codebooks (SIM:28-60). [FACT]
2. **Targets**: (a) a 32x32 wrap-around Gaussian heatmap built from the same angles (SIM:92-104); (b) a 32x2 impairment target built from the exact $D_r, D_t$ with mean and (for phase) linear ramp projected out (SIM:83-89). Both are loss-side only. [FACT]
3. **Model input** is `Y16` only (NB:776-788 `net_forward` passes `b['Y16']`; the bank's `psi/phi/L/snr` are not fed to the network). Normalisation is per sample: `Yc / sqrt(mean(|Yc|^2, dims (1,2)))` (NB:511). BatchNorm runs in eval mode at inference (`load_model` returns `m.eval()`, NB:755; `torch.no_grad`, NB:775). [FACT]
4. **Decoder** (NOMP + Newton + cyclic re-estimation + optional self-calibration) has no learned weights. It receives `Hant` (from Y), the heatmap, and the **true $L$** (NB:792-806, `decode(Hsrc[jj], L, ...)`). [FACT]
5. **Test banks**: `official` = frozen `eval_bank.npz` (seed 42, L=3, 8 SNR x 1000, original generator); `dldoa_test` (seed 7, same protocol); `T2_d{1,2,5}` (seeds 9101/9102/9105); 4 tuning banks (9901-9904); 47 v1 robustness banks (seeds 1000-7035, n=500, sha256 in `MANIFEST.json`). [FACT: NB:347-407; `a07_s1_out.json`]
6. **Tuning**: `tune_selfcal` (NB:811-830) loops over `TUNE_BANKS` only and caches to `results/selfcal_tuning.json`. It chose $(\lambda,\tau)=(1.0,5.0)$. [FACT]
7. **Evaluation**: `metrics` (NB:316-330) uses verbatim-compiled evaluator functions from `TVT_Blob_Inference.py`; ground truth is read only there. U-Net = official Keras weights ported to PyTorch (Pd reproduces published values to ~2e-4, `OUT/RESULTS.md`). v1 baselines (`IABR-v1`, `Teacher`, `DFT-SIC`) are read from the v1 suite's prediction cache via `V1_NAME = {'official': 'E3_frozen'}` (NB:~922-948). [FACT]

---

## 2. Leakage-path verdict table

| # | Leakage path | Verdict | Inflation potential | Label |
|---|---|---|---|---|
| 1 | Ground truth (angles, alpha, D) entering the network input | ABSENT | none | FACT |
| 2 | Labels encoded in input features (SNR/L proxies) | ABSENT for L and angles; SNR is recoverable from Y power (legitimate observable) | none | FACT (measured) |
| 3 | Train/test share latent channel realisations (training stream vs any bank) | ABSENT (no exact or anomalous near-duplicates) | none | FACT (measured) |
| 4 | Seed-created duplicate structures inside / across banks | PRESENT among v1 robustness banks (paired by design or by seed collision); ABSENT in official / dldoa / T2 / tune banks | small, affects only robustness-table independence | FACT (measured) |
| 5 | Normalisation using test statistics | ABSENT (per-sample only; BN in eval mode) | none | FACT |
| 6 | Global preprocessing statistics computed on test data | ABSENT (none exist) | none | FACT |
| 7 | Simulator parameters leaking into labels | Impairment target uses exact $D_r,D_t$: legitimate supervision, not leakage; heat target from same angles: fine | none | FACT |
| 8 | Non-independent samples within a test bank | ABSENT within banks (0 duplicate scenes in 64 banks) | none | FACT (measured) |
| 9 | Evaluation data used during generation / tuning | ABSENT for parameter tuning (tune_selfcal uses TUNE_BANKS only). PRESENT for design (official-bank results and an NOMP diagnostic on a subset of the official bank shaped the v2 design) | small-to-moderate for "beats U-Net" narrative; not visible in independent banks | FACT + CONCERN |
| 10 | Model-specific information contaminating the dataset | ABSENT in data; PRESENT in protocol: true $L$ is an oracle input to decoders and evaluator | affects absolute Pd, applies equally to U-Net and paper protocol | FACT + CONCERN |
| 11 | v1 cached predictions / `V1_NAME` mapping | Alignment VERIFIED (positional, controls fail as expected); no staleness detected; v1 baselines were never tuned on v2 banks | none found | FACT (measured) |
| 12 | Train-stream reproducibility / stale caches | Training stream is non-reproducible by construction; v2 `pred__*` and `net__*` caches are keyed by name only | integrity risk, no observed inflation | FACT + CONCERN |
| 13 | Original U-Net possibly trained on data overlapping the seed-42 test bank | UNVERIFIABLE (one 3-path geometry coincidence found) | would inflate the baseline, not v2 | FACT (coincidence) / Insufficient evidence (training data) |

---

## 3. Evidence per path

### 3.1 Ground truth entering the input (path 1) - ABSENT
- FACT: `net_forward` builds `y` from `b['Y16']` only (NB:781-783). `run_decoder` passes `Hsrc`, `L`, `heat` (NB:792-806). The only reads of `bank['psi']/['phi']` are in `metrics` (NB:322) and the cache writers. Grep of `notebook_src.py` for `['psi']`/`['phi']` returned only lines 322, 939, 948.
- FACT: `IABC.features` uses per-sample `Ht` and `Y` (NB:440-460), no label tensors.
- FACT: the training input tensor is `to_ri(sim['Y'])` only (SIM:115); `heat` and `imp` are separate dict keys used only in the loss.
- Example of how this could have inflated performance (not present): if the decoder were initialised at bank `psi/phi` or the heatmap peaks were placed using `bank['psi']`, Pd would rise toward 1 at all SNR. The observed $\approx0.45$ for the network-only decoder and 0.22 U-Net-equal Pd at -10 dB are inconsistent with such a leak.

### 3.2 Labels encoded in features (path 2) - ABSENT (SNR recoverable, legitimate)
- Measured (`a07_s3_out.json`, 10,240 training scenes drawn with the training generator): corr(log mean power of Y, true L) = 0.038 (no L information in total power, as expected since $\|H\|_F^2 = N_tN_r\sum|\alpha_l|^2$ has mean 256 for every L because $\alpha\sim\mathcal{CN}(0,1/L)$); corr(log power, SNR) = -0.84 (noise floor dominates at low SNR). [FACT]
- DERIVATION: per-sample RMS normalisation (NB:511) removes the absolute scale, but the added channel $\log(|Y_n|^2+10^{-3})$ preserves the noise-floor shape, so SNR is partly recoverable. SNR is a physically observable quantity and is not a label of the task. Not leakage.
- Impairment: $D_r,D_t$ are in $Y$ through the codebooks (physics). The impairment head must estimate them from $Y$. No side channel found.

### 3.3 Train / test latent overlap (path 3) - ABSENT
- FACT (SIM:127, `a07_s3_out.json`): training rng entropy is a list `[seed, worker, urandom32]`; bank rngs are `default_rng(int)`. `SeedSequence(9101)` vs `SeedSequence([9101,0,1])` produce different draws (`seedseq_int_vs_list_same_first_draws: false`). A collision needs worker 0 and urandom = 0 for a bank seed equal to the train seed (numpy pads entropy with zeros: verified `SeedSequence(42) == [42,0] == [42,0,0]`, `a07_s1_out.json`). Probability $\approx 2^{-32}$ per worker start; the v2 training seed defaults to 0 (NB:701 `train(seed=0)`, stream seed `seed + 17*start`; the seed actually passed by `train_runner.py` was not separately verified), no bank uses seed 0 or 17*k.
- Measured: 10,240 training scenes vs first-path (psi, phi) keys (rounded 1e-5 rad) of 64 banks (eval_bank plus 63 generated banks): 0 hits (expected by chance 8e-4 per bank). Nearest-neighbour distance of eval-bank first paths to the 10,240 training first paths: median 0.0146 rad, identical to the train-to-train median 0.0145 rad. No anomalous proximity. [FACT]
- Caveat (ASSUMPTION): the training stream is effectively unbounded (48,000 steps x 256 = 12.3 M scenes). Scenes are drawn from the same continuous prior as the banks, so "nearby" scenes are unavoidable and equal to what the paper's protocol (train and test from one generator) also has. This is prior sharing, not sample sharing.

### 3.4 Seed-created duplicate structures (path 4) - PRESENT in v1 robustness banks only
- FACT (`a07_s1_out.json`, `seed_collisions`, `dup_detail`): seed 2020 is used by both `gain_g2_snr0` (2000+20+0) and `gain_g0.5_snr15` (2000+5+15). All 500 samples share identical L, all path angles and alphas; only the noise draw / gain differ (max |Y| difference 2.98). Seeds 1000 and 1015 are shared by `phase_d0`, `nuis_p{-20,-10,0}` at snr 0 and 15. Nuisance banks are paired to each other by design (NB v1 source: "scene sequence depends only on (seed, snr)"): 200 scenes identical across `nuis_p-20/-10/0`; and 3-4 scenes per pair coincide in the first path with `phase_d0` at unequal L and different indices (0/0, 13/81, ...), an RNG-stream offset artefact.
- FACT: `frozen:eval_bank` vs `dldoa:val` share exactly one first-path geometry, eval index 1509 (L=3, -5 dB) vs val index 595 (L=8, -8 dB): all 3 eval paths equal the first 3 val paths (printed by `a07_s1b.py`).
- FACT: within every one of the 64 banks there are 0 duplicate scenes.
- Inflation potential: DERIVATION - two banks from the same scenes give strongly correlated Pd (`gain_g2_snr0` and `gain_g0.5_snr15` look like independent evidence in a table but share geometry). With n=500 x 3 paths the binomial standard error at Pd ~0.9 is $\sqrt{0.9\cdot0.1/1500}\approx0.8$ pt, so differences < ~2 pt between v2 and a baseline on a single robustness bank are not significant, and the paired banks double-count the same scenes. Effect on headline results: none (official, dldoa_test, T2, tune are unaffected).

### 3.5 / 3.6 Normalisation and global statistics (paths 5, 6) - ABSENT
- FACT: the only normalisation is per sample at NB:511. `IABC` and the stem use no dataset means or stds (grep of `notebook_src.py`; no `mean_`/`std_` buffers loaded from data). Evaluation is `MODEL.eval()` (NB:755) so BatchNorm uses running statistics accumulated on the training stream, not on the test batch; evaluation batch (256) composition therefore does not affect outputs.
- Example of the risk that is absent: computing the RMS over the whole bank (or leaving BatchNorm in train mode at test time) would let an easy high-SNR sample lower the effective scale of a hard low-SNR sample in the same batch.

### 3.7 Simulator parameters leaking into labels (path 7) - NOT leakage
- FACT: `impairment_targets` (SIM:83-89) uses exact $D_r,D_t$ as supervised labels. At test time the model does not receive them. This is ordinary supervised learning of a nuisance, mirroring "a fitted estimator of $D$".
- DERIVATION: the projection (mean, linear ramp) removes the part that is unidentifiable from $Y$ (a global phase or a linear phase across the array is absorbed into the source angle and $\alpha$). Verified by the earlier scripts `s4g.py` (ramp-induced angle shifts). Not a leak; a label the model cannot see.
- Heat target from the same angles: the desired output of the detector. Fine. The peak is placed on the rounded cell, so the target is quantised to half of a beam cell, a training-signal choice, not a leak.

### 3.8 Non-independent samples (path 8) - ABSENT within banks
- FACT: each bank's scenes come from one sequential `default_rng(seed)`; 0 exact duplicates within 64 banks (`within_bank_dups`, `a07_s1_out.json`).
- Regeneration check: the seven v2 banks regenerate bit-exactly from their seeds (max|dY| = 0.0, max|dpsi| = 0.0; `v2_bank_regen_maxabs`), so no hidden shared state, and the stored `seed`, `phase_deg`, `gain_db` match the notebook spec.
- CONCERN (Low): `make_bank` returns immediately if the file already exists (NB:357). A bank produced by an older code version would silently persist. For the current banks the regeneration test rules this out.
- Serial-correlation: samples within a condition are consecutive draws from one PCG64 stream, which is standard and independent to any practical degree. The 1000 per SNR in the official bank are also sequential draws from one stream.

### 3.9 Evaluation data used during generation / tuning (path 9)
**(a) Parameter tuning: ABSENT.**
- FACT: `tune_selfcal` iterates `for tb in TUNE_BANKS` (NB:819-821) and never touches `official`, `T2_*`, or v1 banks. `TUNE_BANKS` seeds 9901-9904 are distinct from every evaluation bank seed (`bank_seeds`).
- FACT (timestamps, `OUT`): checkpoint dir 08:52, `selfcal_tuning.json` 08:54, `net__official.npz` 08:55, `pred__IABR-v2__official.npz` 08:55:48. Tuning happened after training and before the official evaluation. The log (`run_log.txt` lines 150-156) prints only tune-bank Pd.
- FACT: the tune grid has 7 settings; best mean 0.7870 vs 0.7812 without self-calibration (+0.6 pt on tune banks). On the official bank the full model gives 0.7551 vs 0.750 without (`RESULTS.md:310`), and 0.7531 vs 0.7486 on dldoa_test (`infer_dldoa_test.log`). The selection benefit is at most ~0.5 pt, the same size on fresh data. Not a test-set fit.
- CONCERN (Low): tuning banks are condition-matched to the headline test conditions (tune_d5 vs T2_d5; tune_g3 vs the gain sweep), so "never touched" holds, but their construction anticipates the test conditions; that is normal validation design.

**(b) Design-level peeking: PRESENT.**
- FACT (NB:101-132): the "Why v1 falls short" table (S1-S5) is built from v1 results on the official seed-42 test set, and a "Diagnostic run before this notebook" scored NOMP on "a 300-per-SNR subset of the official test set" (0.758 vs U-Net 0.737). The decoder design (C1, C3, C5) followed. `nomp_os=4, nomp_rc=3` (NB:88) have no recorded selection procedure: Insufficient evidence on how they were chosen. The decoder sanity check also runs on 40 official 25 dB samples (NB:670-674).
- Inflation example: the official bank is then not a pristine hold-out for the architecture choice; a design chosen after seeing that "NOMP gives 0.758 on this test set" is optimised for this test set family. DERIVATION on magnitude: the same v2 vs U-Net gap appears on fresh banks: official +1.85 pt (0.7551 vs 0.7366), dldoa_test (seed 7) +1.77 pt (0.7531 vs 0.7354), T2 d=1/2/5 +1.8/+1.4/+1.7 pt (`RESULTS.md` B). Because seed 7 and the T2 seeds were not seen at design time (dldoa test is evaluated only post hoc in `infer_dldoa_test.py`), design-level selection inflation on the headline gap is not detectable; it would appear as official >> dldoa_test, which is not observed.
- FACT: NOMP (no network, no learned parameters) scores 0.7581 on official and 0.7564 on dldoa_test, at or above IABR-v2. Any leakage into the trained network therefore cannot explain the headline number; the estimator carrying it has no trainable parameters.

### 3.10 Model-specific information in the protocol (path 10)
- FACT: `decode(..., L, ...)` and `peaks[np.argsort(-amps)[:L]]` in `unet_predict` (NB:~915-918) use the true per-sample $L$ from the bank. The paper's evaluator does the same (top-$L$ peaks), so the comparison is like-for-like.
- CONCERN (Medium, protocol): the reported Pd is conditional on known $L$. IABR-v2 and DFT-SIC/NOMP always return exactly $L$ detections, so `pd_paper == pd_strict` for them (0.7551/0.7551), whereas the U-Net has 0.7366 (paper) vs 0.6940 (strict) (`RESULTS.md`). The paper metric drops samples with fewer than $L$ detections; that drop favours the U-Net, not v2. Absolute Pd would drop for every method if $L$ had to be estimated. Effect size when $L$ is unknown: Insufficient evidence (never run).
- FACT: the 1-degree threshold decision uses the true angles only in the evaluator.

### 3.11 v1 cached predictions and `V1_NAME` (path 11)
- FACT (`a07_s2b_out.json`): for `IABR_s0`, `Teacher`, `DFT-SIC` on 6 banks (official mapped to `E3_frozen`, phase_d0_snr15, gain_g4_snr15, L8_snr15, sep_5deg_snr15, tail_snr35), prediction counts equal bank sizes (8000/500). Median angular error of aligned predictions vs true angles is 0.3-2.1 deg (e.g. Teacher official 1.30 deg), while the same predictions rolled by one sample give 38-85 deg. Alignment is therefore right; a mis-mapped bank would produce the large control value. Exceptions to note: `IABR_s0 | sep_5deg_snr15` median 45.5 deg vs its control 83.8, and `IABR_s0 | L8_snr15` 8.1 deg vs control 38.3; these are consistent with v1 failing on close/dense paths (documented S3/S4 in NB:101-132) rather than misalignment, since Teacher/DFT-SIC on the same banks are 1.7-2.1 deg.
- FACT: all v1 bank files in `IABR_Net_TestSuite\data\generated_banks` are byte-identical (md5) to the GPU-run copy (`v1_banks_identical_to_gpu_run_copy: all identical`); the MANIFEST sha256 of every bank matches the file on disk (`manifest_sha_mismatch: []`).
- FACT: the v2 network cache `net__*.npz` stores `Ht`; recomputing `Ht` from the locally regenerated banks gives relative error $\le 4.8\times10^{-7}$ for all 56 cached banks with equal sample counts (`v2_netcache_vs_local_bank`). So the banks the network was scored on are exactly the banks described here.
- FACT: v1 models were trained on the v1 stream (`default_rng([seed, worker])`, seeds 0-2) and, by the v1 source, never tuned on v2 banks. The v1 Teacher = the authors' 64-block ResNet baseline.
- CONCERN (Low): the v1 predictions are read positionally and only `official -> E3_frozen` is remapped; there is no hash check between cache and bank in the code. My alignment test replaces it and passed.
- Whether v1 IABR was trained/selected using the official bank: Insufficient evidence (I did not audit the v1 selection log, only the v1 source's training loop, which draws from a synthetic stream; the v1 E3 uses the bank for evaluation only).

### 3.12 Reproducibility / staleness (path 12)
- FACT: `os.urandom(4)` in the training rng (SIM:127) means two training runs use different data; the trained model cannot be regenerated bit-for-bit. The stream seed also changes on resume (`BatchStream(CFG, seed + 17*start)`, NB:716).
- FACT: prediction caches are named `pred__{model}__{bank}.npz` (NB:~935) and network caches `net__{bank}.npz` (NB:777) with no hash of weights, `SC`, or CFG. If the model or `SC` were changed after caching, old caches would be used silently. Current state: cache times (08:55) are later than the checkpoint (08:52) and the results JSONs are later than the caches; consistent. The rerun logs at 16:07-16:11 report "already trained" and reuse caches (`run_log.txt` lines 148-149, 174-175). No inconsistency found. [FACT / CONCERN Low]

### 3.13 Original U-Net vs the official bank (path 13)
- FACT: `eval_bank` sample 1509 and `dldoa` val sample 595 share the same three path angles (see 3.4). The dldoa train/val/test as saved have 0 exact input overlaps with the eval bank (train vs test 0, test vs eval 0, train vs eval 0 hash collisions; `a07_s1b.py`).
- Insufficient evidence: whether the authors' released U-Net was trained on data drawn from the same seed-42 stream as the official test bank. If it was, the baseline would be advantaged (again, not IABR-v2). One coincident scene among 8000 is 0.0125 % of the test set and different noise/SNR, so the direct effect is negligible; the concern is only that train/val/test streams in the authors' pipeline share a seed.

---

## 4. Other integrity observations (not leakage, but they bear on interpretation)

1. **T2 banks are not the Meneses et al. data.** [FACT] `T2_d*` are freshly simulated with `iabr2_sim.simulate` (NB:372) at $\delta_{max}\in\{1,2,5\}$ deg; the table row "paper base U-Net (Table 2)" is the published value, not evaluated on the same scenes. Comparison "paper vs ours" mixes data sets (see also Agent 01's report).
2. **Inverse crime on T2 / tune / v1 robustness banks.** [FACT/DERIVATION] Training and these banks are drawn by the same code (`sample_paths`, `simulate`); only the official bank and the dldoa banks come from the authors' generator (or the project's reproduction of it). Agreement of v2 on official (0.7551) and dldoa_test (0.7531) with T2 d=1 (0.754) indicates the two code paths are numerically consistent; I did not re-run a simulator-equivalence test in this audit. Insufficient evidence that the impairment model in `simulate` matches Meneses et al.'s phase-error model in every detail (uniform $\pm\delta$ on each antenna vs their definition).
3. **The trained network contributes little to the official result.** [FACT] `v2 no IABC` 0.758 and `NOMP (no network)` 0.7581 vs full 0.7551; network-only 0.451. The claim "v2 beats U-Net" is a physics-decoder claim conditioned on true $L$; the data pipeline does not need to leak for that result.
4. **Sample sizes.** [DERIVATION] Official: 24,000 paths (SE of mean ~0.3 pt; paired +1.85 pt is well outside noise). Robustness banks: 1,500 paths (SE ~0.8 pt at 0.9). Tune banks: 150 scenes per (bank, SNR): the 7-way grid selection has an SE ~1 pt per cell, so the chosen $(1.0, 5.0)$ over $(0.5, 5.0)$ (0.7870 vs 0.7851) is not statistically distinguishable; harmless for leakage.

---

## 5. Conclusions

- No leakage from ground truth, labels, normalisation statistics, or shared scenes into the IABR-v2 training, tuning, or headline evaluation was found (paths 1-3, 5-8: measured or code-verified absent).
- The self-calibration tuning is cleanly separated from the evaluation banks (path 9a).
- The genuine integrity weaknesses are (i) design-level exposure to the official bank (9b), (ii) the oracle-$L$ protocol (10), (iii) paired or seed-colliding v1 robustness banks (4), (iv) non-reproducible training stream and name-only cache keys (12). None was observed to inflate the headline v2 vs U-Net gap, which reproduces on unseen seed-7 data and on T2 banks.
- Unverifiable within this audit: the U-Net's own training data, the selection history of `nomp_os/nomp_rc`, the v1 model-selection log, and Pd when $L$ is unknown.
