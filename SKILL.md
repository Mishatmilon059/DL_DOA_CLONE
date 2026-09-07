# SKILL.md — DL_DOA_CLONE Project Memory

**Read this entire file before touching anything. Then VERIFY the claims below against
the actual repo (git log, file listing, notebook contents) before acting — this file
is a briefing, not a substitute for checking. Section 10 gives you a concrete
verification checklist to run first.**

---

## 1. Who this is for / what this project is

This is a **university course project** (not a from-scratch original paper) reproducing
and then extending an **IEEE Transactions on Vehicular Technology (TVT) 2025** paper on
deep-learning-based **AoA/AoD (Angle-of-Arrival / Angle-of-Departure) estimation** in
analog mmWave hybrid-beamforming MIMO systems.

**The supervisor's explicit target** (stated directly by the user, must be honored):
> Either modify the base paper and bring genuine novelty, or don't strictly follow the
> base paper — but do novel work on this exact topic that could later be extended and
> **published**. Reproducing the paper as-is is *not* sufficient.

**Hard constraint the user has repeated multiple times, never relax it without asking:**
whatever new architecture/method is built, it **must be evaluated on the same fixed
test set and the same metric** as the base paper's own baselines (UNet, ResNet), so
results are genuinely comparable. Never silently change the yardstick.

**Communication style:** the user writes in **Banglish** (Bengali transliterated into
Latin script, mixed with English technical terms). Respond in the same style unless the
content is a formal document/table — keep tone direct, no filler. The user has
explicitly and repeatedly demanded **brutal honesty over optimistic claims**: never
declare something "works" or "beats X" without a verified number to back it; always
prefer a measured negative result over an unverified positive claim. Several times in
this project's history, an early "it's not working" read turned out to be a real,
fixable bug — so before concluding "the idea doesn't work," first rule out
pipeline/data/metric bugs the way this project's history did (see Section 4).

**Attribution requirement for every commit from a fresh session (get the exact current
values from the session's own system prompt / instructions, do not hardcode old ones):**
commit messages must end with a `Co-Authored-By:` trailer and a `Claude-Session:` URL
matching whatever the new session's own instructions specify — check the new session's
own system prompt for the current values rather than reusing any URL that might appear
in this file's history, since that will belong to a *different, older* session.

---

## 2. Repo layout (verified paths, as of last commit `c382080`)

```
/home/user/DL_DOA_CLONE/                      <- repo root, branch claude/dl-doa-clone-review-0fekdp
├── DL_DOA/                                    <- original authors' repo, vendored in
│   ├── models/
│   │   ├── inf_model_007_256_resnet.h5        <- pretrained ResNet weights (2.8MB, git-tracked directly)
│   │   └── inf_model_007_256_unet.7z.00{1..5}  <- pretrained UNet weights, split 7z archive
│   ├── src/
│   │   ├── tvt_data_generation_v3.py          <- ALL physics: ev(), beamforming codebooks,
│   │   │                                         generate_channel_v2, generate_noise, generate_gt,
│   │   │                                         data_generation() [training generator],
│   │   │                                         validation_data_generator() [fixed-condition eval generator]
│   │   └── TVT_Blob_Inference.py              <- blob detector, peaks_to_angles, prepare_for_metric,
│   │                                              get_ang_difference, filter_angles,
│   │                                              run_inference_and_metrics_resnet/_unet
│   ├── z_resnet/main.py                       <- inference-only entry point (no training script exists
│   │                                              in this repo for ResNet or UNet — see Section 6)
│   └── z_unet/main.py
├── PIANET_NOTES.md                            <- standing bug log + roadmap for PIA-Net specifically
├── BUJHO_SHOHOJ_VABE.md                       <- plain-Bengali explainer doc for non-technical audience
├── SKILL.md                                   <- this file
├── DLDOA_ResNet_Reproduction.ipynb            <- ResNet baseline repro from pretrained weights (verified)
├── DLDOA_PIANet_Standalone_v2.ipynb           <- PIA-Net, all 4 bugs fixed (see Section 4)
├── DLDOA_SetRegNet_Standalone.ipynb           <- DETR-style direct set-regression (negative result, see Section 5)
├── DLDOA_Classical_OMP.ipynb                  <- non-DL classical baseline (self-contained, verified)
├── DLDOA_SE_ResNet_Lite.ipynb                 <- CURRENT ACTIVE WORK: SE-attention + block-count ResNet variant
├── DLDOA_Baseline_vs_PIANet.ipynb             <- older/earlier comparison notebook
├── DLDOA_NoClone_StepByStep.ipynb             <- earlier 78-cell full walkthrough
├── DLDOA_Colab.ipynb                          <- earliest end-to-end Colab reproduction
└── dldoa_dataset_generation.py                <- standalone dataset-gen script
```

---

## 3. The problem statement (physics + DL framing)

### Physical setup
- Transmitter: `Nt=16`-antenna half-wavelength ULA. Receiver: `Nr=16`-antenna ULA.
- Neither side observes the raw array — both use fixed **analog hybrid beamforming
  codebooks**: transmit precoder `F` (Nt×P), receive combiner `W` (Nr×Q), with
  `P=Q=16` in the standard evaluation condition. These are DFT-like, built from the
  same array steering vector `ev(n, angle) = (1/sqrt(n)) * exp(-j*pi*cos(angle)*k)`.
- `L` propagation paths (typically `L=3` at evaluation), each with an AoD `phi_l`, an
  AoA `psi_l`, and a complex gain `alpha_l`.
- Channel: `H = sqrt(Nt*Nr) * sum_l alpha_l * a_r(psi_l) * a_t(phi_l)^H`.
- **Observation (all that's ever seen):** `Y = W^H @ H @ F + Z`, a `16x16` complex
  matrix, `Z` = complex Gaussian noise at a given SNR. **Single snapshot** — no
  multi-snapshot covariance averaging is available (this matters a lot: see Section 7,
  SubspaceNet comparison).

### The estimation problem
Given only `Y` (noisy, compressed, single-snapshot), recover `{(phi_l, psi_l)}` for
`l=1..L`, with `L` itself not necessarily known in general (fixed to 3 for the
standard eval condition).

### Base paper's DL framing (image-to-image regression)
- Input to the network: `Y`'s real/imag parts stacked and zero-order-zoomed to a fixed
  `64x64x2` (zoom factor depends on P: 4x for P=16, 2x for P=32, 1x for P=64 — always
  lands on 64x64).
- Output: a `256x256x1` "heatmap" — a 2D grid over (AoD, AoA) space, with a small
  Gaussian blob (sigma=0.07 at eval time) rendered at each true path's location, ~zero
  elsewhere.
- Architecture: CNN encoder-decoder (ResNet: 64 stacked residual blocks, 469,393
  params; or UNet), trained with MSE loss against the Gaussian-blob heatmap.
- Post-processing (NOT part of the trained model): OpenCV `SimpleBlobDetector` finds
  peaks in the output heatmap → top-L peaks by amplitude → convert pixel coords back
  to angles via `arccos`.
- **This heatmap+blob-detector design is a CHOICE, not a necessity** — its real
  justification is that it handles a *variable, unknown* number of paths naturally (you
  just count blobs). Its cost: heavy upsampling decoder (measured ~80% of ResNet-family
  compute) and a pixel-quantization ceiling on achievable precision. This exact
  trade-off is the throughline motivating most of the redesign work in this project
  (Section 5).

### Fixed evaluation protocol (do not deviate without asking)
`L=3`, `SNR` swept over `[-10,-5,0,5,10,15,20,25]` dB, `P=Q=16`, `Nt=Nr=16`,
`sigma=0.07`, heatmap grid `M=256`. Metrics:
- **RMSE** (degrees) — computed only over predictions already within a 1° window
  (a *censored* statistic — cannot exceed 1.0 by construction; do not confuse with
  uncensored/raw error).
- **Pd (Probability of Detection)** — fraction of path estimates landing within 1° of
  ground truth, after Hungarian-matching predicted-to-true pairs.

### Verified hardcoded reference numbers (from real pretrained weights, reused everywhere)
```python
UNET_REF_RMSE = {-10:0.5498,-5:0.5069,0:0.4548,5:0.3777,10:0.3047,15:0.2556,20:0.2297,25:0.2086}
UNET_REF_PD   = {-10:0.2226,-5:0.4706,0:0.6788,5:0.8127,10:0.8883,15:0.9244,20:0.9439,25:0.9509}
RESNET_REF_RMSE = {-10:0.5532,-5:0.5118,0:0.4581,5:0.3920,10:0.3256,15:0.2792,20:0.2528,25:0.2377}
RESNET_REF_PD   = {-10:0.2043,-5:0.4366,0:0.6396,5:0.7790,10:0.8623,15:0.8987,20:0.9250,25:0.9378}
UNET_PARAMS, RESNET_PARAMS = 31_276_481, 469_393
```

---

## 4. PIA-Net — physics-informed deep-unfolding (the first thing built, now considered a dead end by the user)

`DLDOA_PIANet_Standalone_v2.ipynb`. Core idea: a `LearnedISTA` layer that unfolds
classical iterative sparse recovery (soft-threshold gradient steps, learned step-size
and threshold per iteration) over a physics dictionary, feeding into a decoder to the
256x256 heatmap. Four **real, root-caused, fixed** bugs, each with its own verification
cell in the notebook — this diagnostic discipline (never declare "architecture limit"
without first hunting for a pipeline bug) is the most important working norm of this
project and directly found real bugs three separate times:

1. **Data starvation** — model was scoring at chance level; root cause was simply not
   enough training samples reaching the model.
2. **Dictionary sign error** — steering-vector sign convention mismatch in the physics
   dictionary vs. the actual channel generator.
3. **ISTA step-size divergence** — the learned step size was ~725x above the
   convergence limit, causing the unfolded iterations to diverge instead of denoise.
4. **Coordinate-system mismatch ("Bug 4")** — the angle grid was built angle-uniform
   instead of omega-uniform (`omega = pi*cos(angle)`), causing a systematic ~119-pixel
   misalignment between the physics branch and the actual channel geometry. Fixed by
   switching the dictionary grid construction to sample uniformly in the omega
   (cos-angle) domain, matching the array's true non-uniform angular resolution
   (fine near broadside, coarse near endfire).

After all four fixes, PIA-Net produced genuine, measured learning (not chance-level),
but did **not** clearly beat ResNet/UNet on the fixed yardstick. The user has since
called this "useless" and the project pivoted away from it (see Section 5). Do not
resurrect PIA-Net work without the user asking — it's considered explored and closed
for now, though `PIANET_NOTES.md` has the full bug log if ever needed again.

**PIA-Net's `LearnedISTA` layer is conceptually important even though the project moved
on: it is literally a soft, differentiable, learned-step version of classical
Orthogonal Matching Pursuit (see Section 6's OMP notebook)** — this connection was
made explicit and is a useful framing device when discussing "why try deep-unfolding
at all."

---

## 5. The strategic pivot — reformulating beyond image-to-image

The user explicitly granted freedom to abandon the base paper's exact image-to-image
framing, **provided the fixed yardstick (Section 3's protocol + reference numbers)
stays the same**. Two major redesigns were tried:

### 5a. SetReg-Net (`DLDOA_SetRegNet_Standalone.ipynb`) — DETR-style direct set regression
No heatmap, no blob detector. Architecture: `LearnedISTA` branch (kept from PIA-Net) +
small CNN branch → concatenated features → learned query embeddings
(`QueryBroadcast` custom layer, `L_MAX=9` queries) → 2-layer cross-attention decoder →
per-query heads for `presence` (logit) and `(sin,cos)` pairs for both angles. Trained
with Hungarian-matching loss (`scipy.optimize.linear_sum_assignment`) + circular
`1-cos(pred-true)` angular loss + `EOS_COEF=0.4` no-object down-weighting (the DETR
trick).

**Result of the FULL run (verified, not a smoke test): 60 epochs, 384,000 samples.**
`ang` loss declined 0.143→0.123 and `pres` loss 0.368→0.299 — genuine, if modest,
optimization progress. But **`oracle_Pd` (Hungarian-matched to ground truth, i.e. the
best-case/cheating ceiling) stayed in the 0.0000-0.0100 range the entire 60 epochs,
never trending upward.** Root cause diagnosed: the Hungarian matcher performs a
"best-of-9" selection that makes the `ang` loss look deceptively low from step 0 (an
artifact of order statistics over 9 candidate slots, not real learning), while
continuous `(sin,cos)` regression fundamentally cannot reach sub-degree precision
without a spatial local-search structure the way heatmap+argmax naturally provides.
**Conclusion: this is a genuine, well-diagnosed negative result — not competitive,
and more training time will not fix it (the loss-vs-metric disconnect is structural).**
Do not resume SetReg-Net training expecting a different outcome; if pursued further, it
would need a redesign (Section 8's SimCC-classification idea), not more epochs.

### 5b. Classical OMP baseline (`DLDOA_Classical_OMP.ipynb`) — no DL at all
Implements textbook Orthogonal Matching Pursuit over an angular dictionary (the
non-learned ancestor of PIA-Net's `LearnedISTA`). Fully self-contained (all physics
functions inlined, no `sys.path`/repo dependency — see Section 9's Kaggle-path lesson).

**Verified real result, same protocol as Section 3:**
```
SNR   OMP RMSE  OMP Pd   | UNet Pd  ResNet Pd
-10   0.5695    0.2958   | 0.2226   0.2043      <- OMP WINS
 -5   0.5117    0.5042   | 0.4706   0.4366      <- OMP WINS
  0   0.4617    0.5667   | 0.6788   0.6396
  5   0.3761    0.7833   | 0.8127   0.7790
 10   0.3226    0.8583   | 0.8883   0.8623
 15   0.3035    0.8750   | 0.9244   0.8987
 20   0.2334    0.8833   | 0.9439   0.9250
 25   0.2582    0.9375   | 0.9509   0.9378
```
**Key finding: classical OMP (zero training) actually beats both UNet and ResNet at
low SNR (-10, -5 dB), is roughly tied through mid-range, and loses only at high SNR
(15-25 dB) where its fixed angular grid caps precision** — the exact same
quantization-ceiling phenomenon as the heatmap models, just in angle-space instead of
pixel-space. This is a genuinely important, citable finding for the writeup: DL's
measurable advantage over a *good* classical baseline in this problem is much narrower
and more SNR-localized than naive "DL beats everything" framing would suggest.

---

## 6. SE-ResNet-Lite (`DLDOA_SE_ResNet_Lite.ipynb`) — CURRENT ACTIVE WORK, most recent, not finished

This is the live experiment as of the last messages in this project. Modifies the base
paper's own ResNet (`Resnet_original`, verbatim-preserved in the notebook) along
multiple independently-toggleable levers:
- **Block count** (`N_BLOCKS`): fewer blocks = fewer params (lightweight goal).
- **Squeeze-and-Excitation (SE) attention** (`se_block`/`se_res_conv`): inserted into
  each residual block before the skip-connection Add, ~87 extra params/block
  (negligible overhead). Motivated as a low-SNR-robustness lever.
- **`SNR_LOW_BIAS`**: optional triangular-distribution training sampler favoring low
  SNR (built but *not currently recommended to enable* — see below, the diagnosed
  problem is now at high SNR, not low SNR).
- **Warm-start from pretrained ResNet** (`transfer_from_pretrained_resnet`,
  `find_pretrained_weights`): since inserting SE layers changes the Keras layer
  ordering, a direct `load_weights()` from the original pretrained `.h5` is impossible
  (position/name mismatch). Instead, a verified layer-by-layer copy function transfers
  weights for the structurally-identical Conv2D/BatchNorm/Conv2DTranspose layers
  (skipping the new SE Dense layers, which remain randomly initialized). Auto-detects
  the pretrained file path across `/kaggle/input`, `/kaggle/working`, `/content`, `.`.
  **Verified mechanism correctness** (not the real weight file, which isn't present
  in the sandbox): on freshly-built 6-block test models, layer counts matched exactly
  (26==26) and all copied weight values were bit-identical (0 mismatches).
- Both the physics/data-generation code and the evaluation/metric code are fully
  **self-contained** in this notebook (learned from the Kaggle path-resolution
  failures encountered earlier — see Section 9).

### Real, verified experimental history so far (chronological):

**Run 1 — 32 blocks, 40 epochs (160,000 samples), scratch-trained, full SNR sweep:**
```
SNR   SE-Lite RMSE  SE-Lite Pd | ResNet RMSE ResNet Pd | Pd gap
-10   0.5418        0.1707     | 0.5532      0.2043     -0.034  (RMSE actually BETTER than ResNet here)
 -5   0.5152        0.3462     | 0.5118      0.4366     -0.090
  0   0.4873        0.5220     | 0.4581      0.6396     -0.118
  5   0.4522        0.6308     | 0.3920      0.7790     -0.148
 10   0.4185        0.6993     | 0.3256      0.8623     -0.163
 15   0.3929        0.7550     | 0.2792      0.8987     -0.144
 20   0.3907        0.7765     | 0.2528      0.9250     -0.149
 25   0.3874        0.7893     | 0.2377      0.9378     -0.149
```
Param count at 32 blocks: 237,937 (49.3% fewer than original ResNet's 469,393).
**Diagnosis:** competitive-to-better at the lowest SNR, but RMSE visibly *plateaus*
from SNR=10 onward while ResNet's keeps improving — a capacity-ceiling signature.

**Continued training test (same 32-block model, +20 more epochs, +80,000 samples,
lower LR 5e-4):** loss stayed flat/noisy (~0.55-0.59), no further improvement. **This
ruled out "just needs more training" and confirmed a genuine capacity ceiling at 32
blocks** — motivated going to full 64 blocks next.

**Run 2 — 64 blocks (matching original ResNet depth) + SE attention, warm-started
from pretrained ResNet weights, 12 epochs fine-tune + 20 more continued epochs
(80,000 more samples), evaluated ONLY at SNR=[20,25] (a deliberate fast partial check,
not yet the full sweep):**
```
SNR   SE RMSE   SE Pd    | ResNet RMSE ResNet Pd | Pd gap
20    0.3023    0.8775   | 0.2528      0.9250     -0.0475
25    0.2861    0.8712   | 0.2377      0.9378     -0.0666
```
Param count at 64 blocks + SE: 474,961 (essentially the same size as original
ResNet's 469,393, +1.2% — a fair "same-size, attention-added" comparison).
**This is real, substantial progress: the high-SNR Pd gap narrowed from ~-0.149
(32 blocks) to ~-0.05/-0.07 (64 blocks + SE)** — roughly a 3x reduction — confirming
capacity was indeed the dominant factor. Loss again plateaued after the extra 20
epochs (0.55-0.57 range), so further pure training on this same config is unlikely to
close the remaining gap by itself.

**⚠️ OPEN / UNFINISHED AT TIME OF WRITING:** the 64-block+SE model has **only been
evaluated at SNR=20,25**. It has NOT yet been evaluated at the full sweep
(-10 through 25), so **it is not known whether the 32-block run's low-SNR advantage
survives in this larger, warm-started model.** Weights are saved to
`se_resnet_full64_v2.weights.h5` (also an earlier `se_resnet_full64.weights.h5` from
before the +20-epoch continuation). **The single most valuable next action, requiring
NO new training, is: load this saved checkpoint and run Part 4's evaluation with
`SNRS_EVAL = list(range(-10, 30, 5))` (the full sweep) instead of `[20, 25]`.** This
produces one single, internally-consistent, full-range comparison table — the user was
explicitly told this is the next concrete step and it had not been reported back as
done when this file was written.

A **composite table** was assembled once for discussion purposes only, explicitly
flagged as NOT a single model's result (rows 1-6 low-SNR from the 32-block run, rows
7-8 high-SNR from the 64-block run) — do not present that composite as a real result
in any final writeup; it was a stopgap while the real full-sweep-on-one-model number
was pending.

---

## 7. Literature reviewed (real papers, verified where noted)

- **SubspaceNet** (Shmuel, Merkofer, Revach, van Sloun, Shlezinger — IEEE TVT, vol.
  74, no. 3, March 2025, DOI 10.1109/TVT.2024.3496119). Read in full from the actual
  uploaded PDF. Solves classical DoA/AoA (single-array, receiver-only — **no AoD
  concept exists in this paper at all**, so "is it better at AoD" has no valid answer).
  Uses a differentiable learned-covariance surrogate feeding classical MUSIC/Root-
  MUSIC/ESPRIT. Verified real code exists and is substantial:
  github.com/ShlezingerLab/SubspaceNet. **Critical structural differences from this
  project's problem, confirmed from the actual code** (`data_handler.py`,
  `system_model.py`): raw antenna-domain snapshots (not beamformed/compressed),
  **multi-snapshot** (`T` snapshots per sample, explicitly needs T>1 for covariance
  estimation — their own "few snapshot" stress test still uses T=2, not T=1), explicit
  coherent-source handling, explicit array-miscalibration modeling. Their own reported
  numbers (Table I/II/III in the paper) are real and impressive *within their own
  easier setup* (e.g. RMSPE 0.20° at M=2 coherent sources, T=100 snapshots, SNR=10dB)
  but **cannot be validly compared to this project's single-snapshot numbers** — the
  metric itself also differs (their RMSPE is a raw/uncensored statistic over all
  Monte Carlo trials, unlike this project's censored within-1°-window RMSE).
- **DAE-DNN** (Chen, Shi, Xuemai Gu, Byonghyo Shim — IEEE Access 2022, DOI
  10.1109/ACCESS.2022.3164897). Read in full from an uploaded PDF. Denoising-
  autoencoder trick (corrupt input, reconstruct *original clean* target) plus
  β-weighted loss and J=9 parallel angle-gated decoders. Setup: ULA M=10, K=2 sources,
  **I=500 snapshots** (multi-snapshot averaging — same fundamental comparability
  caveat as SubspaceNet), covariance-matrix input (not raw beamformed signal), raw/
  uncensored RMSE metric. Not directly comparable to this project's numbers for the
  same reasons.
- **GAN-CNN Fusion** (Sensors 2025, DOI 10.3390/s26051676) — GAN-based signal
  enhancement + complex CNN DOA estimator, 72.2% accuracy / RMSE 3.9° at -10dB with
  500 snapshots. Cited as the factual basis for **recommending against pursuing
  GAN/generative-AI approaches** for this project (see Section 8).
- **SimCC** (Li et al., ECCV 2022) and **RTMPose** (arXiv:2303.07399) — coordinate-
  classification instead of heatmap regression or continuous regression, for
  human-pose estimation. Cited as the literature-grounded explanation for *why*
  SetReg-Net's continuous-regression approach likely failed on precision, and as the
  concrete, not-yet-tried fix (classify fine angle-bins instead of regressing
  continuous sin/cos) — see Section 8.
- **SwinIR** (arXiv:2108.10257) — verified real specs: 11.8M params, natural-image
  restoration (SISR/denoising/JPEG-artifact-removal), local-window self-attention.
  **Explicitly evaluated and rejected** for this project: pretrained weights won't
  transfer (radio-signal matrix vs. natural photo domain gap), local-attention
  inductive bias is the wrong fit for this problem's *global* physics-based
  input-output relationship, heavier than the lightweight goal, and channel-count
  mismatch (2 vs 3) would force discarding the first layer's pretrained weights anyway.

---

## 8. Explicitly rejected / deprioritized directions (with reasons — don't re-propose without new evidence)

- **GAN / diffusion / generative-AI approaches** — rejected because (a) this is a
  deterministic single-answer regression task, not a diverse/multi-modal generation
  task, so generative sampling fights the task's nature; (b) GANs/diffusion are harder
  and slower to train than a plain CNN, conflicting with this project's limited
  CPU/Kaggle-GPU compute budget (already observed: a 64-block CNN took ~80+ minutes to
  train adequately); (c) the one directly-relevant published GAN-DOA result (Sensors
  2025 paper above) isn't more impressive than what this project's own classical OMP
  already achieves, in an easier (multi-snapshot) setting besides; (d) generative
  models are also sometimes proposed for data augmentation, but this project already
  has an unlimited, physics-exact synthetic data generator (`data_generation()`) — no
  data-scarcity problem exists here for a generative model to solve.
- **SwinIR / transformer-based image-restoration transfer learning** — see Section 7.
- **256x256 spatial upsampling is not fundamentally required** — established as fact
  (not yet acted on): fine angular resolution in the *output* does not require the
  network's *internal* feature maps to be spatially upsampled all the way to 256x256.
  Three concrete alternatives were identified that avoid the heavy decoder entirely:
  direct continuous regression (tried: SetReg-Net, failed on precision), dictionary/
  correlation matching (tried: Classical OMP, works but has its own grid-quantization
  ceiling), and classification-over-discretized-bins (**not yet tried** — see below).

---

## 9. Hard-won engineering lessons (apply these by default in any new notebook)

1. **Every standalone notebook must be self-contained** — inline all physics
   (`ev`, `beamforming_vector_generation_P/Q`, `generate_noise`, `generate_channel_v2`,
   `generate_gt`, `data_generation`/`validation_data_generator`) and metric
   (`prepare_for_metric`, `get_ang_difference`, `filter_angles`, blob-detector) code
   directly, rather than `sys.path.insert` + `from src... import`. Repeated,
   time-costly failures occurred (`ModuleNotFoundError: src`) because Kaggle mounts
   the repo/dataset at unpredictable paths across sessions — self-contained code
   sidesteps the entire problem class. When a repo-relative path genuinely is needed
   (e.g. locating a pretrained `.h5` weight file), use a `find_*` helper that
   `os.walk`s a list of common roots (`.`, `/kaggle/input`, `/kaggle/working`,
   `/content`) rather than hardcoding one path.
2. **Custom training loops in a Jupyter/Kaggle context need a per-step heartbeat
   print**, not just an end-of-epoch print — otherwise a slow-but-working loop looks
   indistinguishable from a hang, especially combined with TensorFlow's noisy
   `complex64→float32` cast warning (which fires on *every* step in eager/custom loops
   that lack `@tf.function`, unlike `model.fit()` which traces once). Silence that
   specific warning with `tf.get_logger().setLevel('ERROR')` early in setup.
2b. **Skip any computation whose output the evaluation loop doesn't actually use** —
   a real, measured 5x evaluation speedup came from realizing `evaluate_model()` was
   generating an unused 256x256 ground-truth heatmap per sample; passing
   `compute_gt=False` cut eval time from 0.375s/sample to 0.071s/sample. Always check
   for this class of waste before assuming a slow eval loop just needs more patience.
3. **Verify every notebook by actually executing it before shipping** — the established
   practice is to extract all code cells into a flat `.py` script (via a small
   `json.load` + `''.join(cell['source'])` snippet) and run it standalone, ideally from
   an isolated directory with zero access to the repo, to catch exactly the kind of
   hidden-dependency bug described in lesson #1. Prefer a synthetic/reduced-scale
   smoke test (fewer blocks, fewer steps, fewer eval samples) for fast iteration, but
   always confirm the *actual* default config also at least compiles and the key
   functions (e.g. a weight-transfer function) are separately correctness-checked
   (e.g. the SE-ResNet layer-alignment count-and-value check in Section 6).
4. **Distinguish censored from uncensored/raw statistics before comparing across
   papers or models** — this project's own RMSE metric is censored (computed only over
   already-within-1° predictions, so it structurally cannot exceed 1.0 and does not
   represent "typical error"). Misreading this once already happened in this project's
   history (an early "RMSE 0.43° = near pixel-quantization limit" claim was wrong; the
   real uncensored median error was ~1-4.5°). Any external paper's RMSE must be checked
   for the same distinction before being placed in a comparison table.
5. **Multi-snapshot vs. single-snapshot is the single most common invalid-comparison
   trap** in this literature — always check how many snapshots (`T`) an external
   result assumes before comparing its numbers to this project's single-snapshot
   protocol.
6. **When a loss curve plateaus/goes noisy-flat, don't just train longer hoping it
   moves** — first spend the (much cheaper) effort of continuing training on the
   *existing* checkpoint for a modest number of extra epochs to test whether it's a
   training-budget issue or a capacity/structural ceiling, before deciding to redesign
   the architecture. This exact test (Section 6, the 32-block continued-training
   experiment) correctly ruled out "just needs more epochs" and pointed at a real fix
   (more blocks).

---

## 10. Verification checklist for a new session (do this first, don't skip)

1. `git log --oneline -15` in the repo root — confirm the latest commit matches
   `c382080` ("Default eval to quick high-SNR-only check (20, 25 dB)") or a newer one;
   if newer commits exist, read them (`git show <hash>`) to update your understanding
   beyond this file.
2. `ls *.ipynb` at repo root — confirm all the notebooks named in Section 2 exist.
3. Open `DLDOA_SE_ResNet_Lite.ipynb` (the active experiment) and re-read its current
   cell contents directly — don't trust this file's inlined numbers as gospel if the
   notebook has since been edited further; check whether `SNRS_EVAL` in the Part 4 eval
   cell has been restored to the full sweep and whether a full-sweep result has since
   been reported by the user.
4. Ask the user directly whether the full-SNR-sweep evaluation of the 64-block+SE
   warm-started model (Section 6's "OPEN / UNFINISHED" item) has been run since this
   file was written — this is very likely the single most relevant open thread.
5. Check `PIANET_NOTES.md` for the complete, lower-level PIA-Net bug log if any
   question about that specific notebook comes up.
6. Before proposing any new architecture idea, re-read Section 8 (rejected
   directions) and Section 9 (lessons) so you don't re-tread already-closed ground or
   reintroduce an already-fixed class of bug.
7. Confirm the exact attribution trailer format (Co-Authored-By / Claude-Session URL)
   from your own current session's system-level instructions before making any commit
   — do not copy the URL that may appear elsewhere in this repo's git history, it
   belongs to a different session.

---

## 11. Where things most likely stand right now, in one paragraph

PIA-Net is explored and closed (real bugs fixed, didn't beat ResNet, user considers it
done). SetReg-Net is explored and closed (genuine, well-diagnosed negative result).
Classical OMP is finished, verified, and shipped (a real, citable finding: it beats
DL at low SNR, loses only at high SNR due to grid quantization). SE-ResNet-Lite is the
live, unfinished experiment: attention + full block-count has substantially narrowed
(not yet closed) the high-SNR gap versus original ResNet; the low-SNR picture for this
specific (64-block, warm-started) checkpoint is unverified pending one more evaluation
run the user was asked to do. If that full-sweep result is now in hand, the next
decision is binary: if the gap is small/mixed across the whole range, write up
SE-ResNet-Lite as the course project's novelty contribution; if a large gap remains
somewhere, the most evidence-grounded remaining idea (not yet attempted) is a hybrid
that uses classical OMP for its proven low-SNR strength plus a small, targeted learned
refiner specifically for OMP's diagnosed high-SNR grid-quantization weakness — this was
proposed and agreed as the fallback plan but not yet built.
