# PIA-Net — bug log, verification checks, and roadmap

PIA-Net (Physics-Informed Attention Network) is a lightweight alternative to the
paper's UNet / ResNet heatmap regressors: a physics-informed deep-unfolded branch
(Learned-ISTA over a steering-vector dictionary) running in parallel with a
directly-learned CNN branch, fused by self-attention and decoded to the 256x256
angle heatmap.

Notebook: `DLDOA_PIANet_Standalone_v2.ipynb` (PIA-Net only; UNet/ResNet appear as
hardcoded reference numbers so the heavy baselines never need retraining).

---

## The failure this file exists to document

The first working version scored **at chance level** on the real dataset, but the
raw numbers looked merely "weak" rather than broken, which is why it went
unnoticed for several iterations. The tell was quantitative:

| SNR (dB) | -10 | -5 | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| RMSE | 0.558 | 0.583 | 0.572 | 0.574 | 0.595 | 0.584 | 0.586 | 0.581 |

Mean **0.5791**, essentially flat across 35 dB of SNR. The RMS of a *uniform*
error distribution over the 1-degree acceptance window is `1/sqrt(3) = 0.5774`.
The match means the "detections" were random hits that happened to fall inside
the window — not estimates. A model whose accuracy does not improve across 35 dB
of added signal quality is not reading the signal at all.

**Lesson:** for this task, always check the result against the chance floor and
against SNR-monotonicity before concluding anything. Both checks now ship in the
notebook as Part 12.

---

## Bug 1 — training data starvation (the dominant cause)

| | training data actually seen |
|---|---|
| UNet / ResNet (the reference numbers) | infinite generator, 10,000 fresh samples per epoch x 500 epochs ~= **5,000,000 sample presentations** (paper Table I) |
| PIA-Net (broken version) | the P=16 slice of the 1000-sample *validation* set = **428 samples**, reused every epoch |

A ~4 orders of magnitude deficit. The overfit was visible in the logs
(train loss 0.0071 vs val loss 0.0126) but was misread as "needs tuning".

**Fix:** the notebook now trains on the paper's own infinite generator
(`pia_sample_generator`, Section IV-B spec: L in 1..9, SNR in -15..24 dB,
alpha ~ CN(0, 1/L), min angular separation pi/6), so every batch is fresh and
428-sample memorization cannot happen. Train and val loss should now track each
other; a large gap between them means something regressed.

## Bug 2 — the physics dictionary was conjugated and used the wrong codebook

The steering vector was inlined as `exp(+1j*pi*n*cos(angle))` while the reference
implementation (`dldoa_dataset_generation.py:ev`) uses `exp(-1j*pi*cos(angle)*k)`
— a conjugate sign flip — and the codebooks used naive `linspace` angles instead
of the DFT-based `arccos` construction.

Measured on one noiseless single-path channel (true psi=2.000, phi=1.100 rad):

| dictionary | matched-filter peak | error |
|---|---|---|
| old | psi=0.639, phi=0.737 | **78 deg** off in AoA, 20.8 deg in AoD |
| fixed | psi=2.012, phi=1.129 | 0.7 / 1.7 deg (sub-grid-cell) |

So the physics branch was feeding the network meaningless features.

**Fix:** the exact reference functions are inlined in the notebook, and the
matched-filter check ships as a cell (`DICTIONARY CHECK: PASS`).

## Bug 3 — Learned-ISTA step size was 725x above the convergence limit

ISTA converges only when the step is below `1/L`, where `L` is the Lipschitz
constant of the gradient. For this dictionary:

```
L = c^2 * sigma_max(U)^2 * sigma_max(V)^2 = 16^2 * 2.6214^2 * 2.6214^2 = 12089
required step  < 1/L = 8.3e-5
step actually used     = 0.06          <-- 725x too large
per-iteration blow-up  = |1 - step*L| ~ 724x
over 6 unfolded iters  ~ 1e17x
```

An earlier "fix" added `tf.clip_by_value` to stop the resulting explosion. That
treated the symptom: the clipped sparse code became effectively binary — **85%
exactly zero, 13.5% pinned at the clip** — and both regions have zero gradient,
so the physics branch could neither contribute features nor learn. It looked
stable while being useless.

**Fix:** rescale `U` and `V` to unit spectral norm and normalize the observation
per sample. The problem becomes scale-free (`L = 1`), so a step near 0.9 is
correct. Step and threshold are softplus-reparametrized so the trainable
variables sit at O(1) — necessary because Adam at lr 1.5e-3 cannot meaningfully
tune a raw parameter whose correct value is 7e-5.

After the fix, on real noisy samples: max 0.47, **0.000% at the clip**, 96.5%
sparse; single-path localization error 0.032 rad against a 0.098 rad grid.
Ships as the Part 4.1 health check (`ISTA HEALTH: PASS`).

---

## Not a bug: the complex64 -> float32 warning

```
WARNING:tensorflow:You are casting an input of type complex64 to an incompatible
dtype float32. This will discard the imaginary part...
```

Emitted from the backward pass of `tf.cast(X, tf.complex64)`: TensorFlow casts
the incoming complex cotangent back to float32, keeping the real part. For a real
input embedded into the complex plane with a real-valued loss, taking the real
part is the mathematically correct pullback.

Verified against central finite differences:

| | max relative error |
|---|---|
| complex-cast path alone (float64) | 2.6e-10 |
| full 6-iteration ISTA, all 12 trainable params | 2.9e-7 |

Harmless. Silence with `tf.get_logger().setLevel('ERROR')` if desired.

Likewise, XLA's `Delay kernel timed out` and `slow_operation_alarm ... is taking
a while` messages are cuDNN **autotuning** during graph compilation (~45 s once at
startup), despite the `E` log prefix. Not errors.

---

## Standing verification checks (run these before trusting any result)

1. **Part 2.1** — `DICTIONARY CHECK: PASS` (physics matches the data model)
2. **Part 4.1** — `ISTA HEALTH: PASS` (sparse but not dead, 0% at clip, localizes)
3. **Part 12, Test A** — mean RMSE clearly below the 0.5774 chance floor
4. **Part 12, Test B** — Pd rises with SNR (corr > 0.7, range > 0.15)

Tests A and B are the ones that matter: they distinguish real estimation from
random hits. If they fail, the run is not a working method regardless of how the
loss curve looks.

---

## Result after all three fixes (60 epochs, 240,000 fresh samples, 8000-sample eval)

Both self-tests pass, so for the first time the architecture is actually being measured
rather than a bug:

```
TEST A (RMSE floor): chance = 0.5774, mean RMSE = 0.4787  -> PASS
TEST B (SNR trend) : corr(SNR, Pd) = 0.917, Pd range 0.430 -> PASS
```

| SNR | PIA RMSE | PIA Pd | UNet Pd | Pd as % of UNet |
|---|---|---|---|---|
| -10 | 0.561 | 0.190 | 0.223 | 85% |
| -5 | 0.526 | 0.339 | 0.471 | 72% |
| 0 | 0.510 | 0.463 | 0.679 | 68% |
| 5 | 0.475 | 0.526 | 0.813 | 65% |
| 10 | 0.447 | 0.576 | 0.888 | 65% |
| 15 | 0.442 | 0.608 | 0.924 | 66% |
| 20 | 0.437 | 0.620 | 0.944 | 66% |
| 25 | 0.431 | 0.620 | 0.951 | 65% |

Mean Pd 69% of UNet, 72% of ResNet. Mean RMSE 1.45x worse — but only **1.02x at
-10 dB** (statistically indistinguishable) versus 2.07x at 25 dB.

### Where the gap actually is

The error is not uniform, it is concentrated at high SNR, and Pd **saturates**:
PIA goes 0.608 -> 0.620 from 15 to 25 dB while UNet goes 0.924 -> 0.951. Once noise
stops being the binding constraint, something structural caps the model.

Two measurements identify it:

1. **Pixel quantization.** The 256 grid spans 6.703 rad, so one pixel is 0.478 deg at
   broadside (worse toward endfire). PIA's high-SNR RMSE of 0.431 deg is **0.90 pixel** —
   the integer-pixel limit. UNet reaches 0.209 deg = **0.44 pixel**, i.e. genuinely
   sub-pixel, because its blobs are clean enough for the detector's centroid to interpolate.
2. **Coarse physics grid.** G_GRID = 32 gives 0.098 rad (5.6 deg) spacing, and the decoder
   then has to upsample 8x to reach 256.

### Caveat that matters for any "lightweight" claim

| Model | Params | Pd@25dB | Pd per M params |
|---|---|---|---|
| UNet | 31,276,481 | 0.951 | 0.030 |
| ResNet | 469,393 | 0.938 | 1.998 |
| PIA-Net | 362,125 | 0.620 | 1.711 |

Comparing against UNet's 31M parameters flatters PIA-Net, but **ResNet is the relevant
efficient baseline: only 1.3x larger, far more accurate, and ahead on
accuracy-per-parameter too**. So "lightweight therefore competitive" does not hold today.
The honest claim is that the architecture works and its remaining gap has an identified,
addressable cause — not that it beats the baselines.

## Roadmap

### Improvement levers, in priority order

1. **DARK sub-pixel decoding — no retraining** (shipped as Part 12.6). A Gaussian blob is
   a quadratic in log space, so a 2nd-order Taylor expansion about the peak recovers the
   sub-pixel offset in closed form. Verified on synthetic blobs at this project's blob
   width (sigma = 2.67 px): 0.385 -> 0.000 px noiseless, 0.475 -> 0.044 px at noise 0.02
   (10.8x), 0.608 -> 0.154 px at noise 0.05 (3.9x). On an untrained model's real predicted
   heatmaps it still gave 3.4x. Directly attacks the measured 0.90-pixel ceiling.
2. **Training budget and a gentler LR schedule.** 240,000 samples versus the reference's
   ~5,000,000 (21x less), and ReduceLROnPlateau with patience=5 cut the LR six times in
   60 epochs, reaching 2.3e-5 by epoch 58 while val_loss was still creeping down. Raise
   STEPS_PER_EPOCH/EPOCHS, patience to ~12, and floor min_lr around 5e-5.
3. **SimCC-style coordinate classification** (ECCV 2022, arXiv 2107.03332). Splits each
   axis into bins instead of regressing a 2-D heatmap, which removes the 256x256 upsampling
   stack entirely (the source of both the memory pressure and the quantization error) and
   is sub-pixel by construction; the paper reports >55% fewer GFLOPs at higher accuracy,
   and explicitly targets low-resolution inputs — this project's input is 16x16.
   Caveat: SimCC assumes a fixed keypoint count, while L varies 1..9 here, so it needs an
   L_max-slot plus confidence head rather than a drop-in port.
4. **Finer physics grid (G_GRID 32 -> 48/64).** Halves or quarters the upsampling burden.
   Blocked on attention memory: the refine block is O(seq^2), so 32x32 -> 64x64 takes the
   attention matrix from 537 MB to 8.6 GB at batch 32. The attention has to be pooled or
   dropped first.

Change one thing at a time. Bundling changes is what hid the ISTA divergence behind the
value clip in the first place.

### Planned: port to PyTorch (after the current deadline)

TensorFlow was not chosen on merit — the whole base repo is TF (data generation,
`tvt_models.py`, `TVT_Blob_Inference.py`, and the pretrained `.h5` weights), and a
fair comparison against the UNet/ResNet reference numbers requires running in the
same pipeline.

PyTorch is the better fit for this particular model: it has native complex
autograd, so the artificial real/complex casts in `LearnedISTA` disappear along
with their warnings, and the layer becomes considerably harder to get wrong — note
that two of the three bugs above were complex-arithmetic bugs.

**Important constraint:** porting PIA-Net alone would invalidate the comparison.
The UNet and ResNet baselines have to be ported too, so that proposed model and
baselines share one pipeline and one metric implementation.

### Optional speed knob

The decoder currently holds 64 channels at full 256x256 resolution (537 MB per
activation tensor at batch 32), which dominates runtime. Tapering channels as
resolution grows — standard U-Net practice — cuts the top-level 3x3 conv from
2.42 to 0.34 GFLOP/sample:

```python
filt = 64
for _ in range(int(np.log2(m_out // g_grid))):
    x = L.Conv2DTranspose(filt, 3, strides=2, padding='same', activation='relu')(x)
    x = conv_block(x, filt)
    filt = max(filt // 2, 24)
```

Worth doing because throughput converts directly into fresh samples, which is the
dominant factor for accuracy here — but change it on its own, not together with
other changes, so the effect stays attributable.

---

## Honest status

The three bugs above are fixed and each fix is verified by a check that ships in
the notebook. Whether PIA-Net then reaches UNet-level accuracy is still an open
empirical question — do not claim it beats the baselines without Tests A and B
passing and the SNR curve to back it up. The defensible claim for this
architecture is *competitive accuracy at a fraction of the model size*
(362K params vs UNet's 31.3M, ~86x smaller), not superior accuracy.
