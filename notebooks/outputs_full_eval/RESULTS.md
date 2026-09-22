# DL-DOA 4-Way Architecture Screening — Full Local Evaluation Results

Full run on an RTX 5090 (local GPU), 8 blocks per architecture, 200 epochs x 100 steps each (20,000 gradient steps/architecture, training data generated live and synthetically via `training_data_generator` — never pre-generated or read from disk), evaluated on the complete frozen 8000-sample eval bank (`frozen_banks/eval_bank.npz`, 1000 samples x 8 SNR points from -10 to 25 dB, L=3 paths, nt=nr=16, Q=P=16).

- **Total wall-clock time:** 4.21 hours (all 4 architectures, training + full-bank evaluation, including one-time GPU JIT-compilation warmup on this machine's RTX 5090/Blackwell GPU)
- **Blocks per architecture:** 8 (not 64 — none of the 4 candidates has pretrained weights to warm-start from, so this keeps the comparison fair; the teacher row below is a 64-block, pretrained reference, NOT equal-depth)
- **Epochs x steps/epoch:** 200 x 100 = 20,000 gradient steps per architecture
- **Evaluation set:** 8,000 samples (full bank, no subsampling)

## Final Verdict — ranked by mean Pd across all 8 SNR points

| Rank | Architecture | Mean Pd | Params | Batch size used |
|---|---|---|---|---|
| 1 | FNO | 0.6235 | 334,321 | 256 |
| 2 | GridlessUnfold | 0.2216 | 30,201 | 256 |
| 3 | SIREN | 0.0744 | 3,565 | 256 |
| 4 | WindowAttention | 0.0707 | 7,537 | 64 |
| — | Teacher (ref, 64-block, pretrained) | 0.7103 | 469,393 | — |

**FNO is the clear winner** among the 4 screened candidates — its spectral (Fourier-domain) convolution body reaches 0.6235 mean Pd at only 8 blocks / 334K params, closing most of the gap to the 64-block pretrained teacher (0.7103) that none of the other 3 candidates come close to. GridlessUnfold is a distant second; SIREN and WindowAttention essentially did not learn a useful signal at this depth/epoch budget (mean Pd ~0.07, barely above chance).

## Per-SNR breakdown

### Detection probability (Pd) by SNR (dB)

| Architecture | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | Mean |
|---|---|---|---|---|---|---|---|---|---|
| FNO | 0.2031 | 0.4134 | 0.6104 | 0.7218 | 0.7960 | 0.7962 | 0.7460 | 0.7009 | **0.6235** |
| GridlessUnfold | 0.0815 | 0.1429 | 0.2119 | 0.2328 | 0.2691 | 0.2764 | 0.2747 | 0.2835 | **0.2216** |
| SIREN | 0.0285 | 0.0557 | 0.0787 | 0.0887 | 0.0885 | 0.0890 | 0.0838 | 0.0824 | **0.0744** |
| WindowAttention | 0.0293 | 0.0463 | 0.0762 | 0.0851 | 0.0822 | 0.0811 | 0.0885 | 0.0765 | **0.0707** |
| Teacher(ref,64blk) | 0.2040 | 0.4366 | 0.6393 | 0.7789 | 0.8623 | 0.8989 | 0.9253 | 0.9376 | **0.7103** |

### RMSE (degrees, on correctly-detected paths only) by SNR (dB)

| Architecture | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | Mean |
|---|---|---|---|---|---|---|---|---|---|
| FNO | 0.5546 | 0.5208 | 0.4784 | 0.4286 | 0.4151 | 0.4146 | 0.4313 | 0.4473 | **0.4613** |
| GridlessUnfold | 0.5752 | 0.5443 | 0.5472 | 0.5358 | 0.5373 | 0.5054 | 0.5258 | 0.5232 | **0.5368** |
| SIREN | 0.5777 | 0.5563 | 0.6043 | 0.5809 | 0.5637 | 0.5728 | 0.5828 | 0.5680 | **0.5758** |
| WindowAttention | 0.5767 | 0.5816 | 0.5800 | 0.5701 | 0.5893 | 0.5475 | 0.5877 | 0.5547 | **0.5735** |
| Teacher(ref,64blk) | 0.5526 | 0.5117 | 0.4578 | 0.3918 | 0.3255 | 0.2794 | 0.2534 | 0.2373 | **0.3762** |

## Training details per architecture

| Architecture | Wall-clock finish (h) | Batch | First-epoch loss | Min loss (epoch) | Final-epoch loss |
|---|---|---|---|---|---|
| FNO | 2.18 | 256 | 2.2142 | 0.6812 (ep 198) | 0.6892 |
| GridlessUnfold | 4.16 | 256 | 1.8302 | 1.2577 (ep 198) | 1.2667 |
| SIREN | 1.06 | 256 | 1.8335 | 1.6696 (ep 200) | 1.6696 |
| WindowAttention | 2.89 | 64 | 2.0985 | 1.7009 (ep 198) | 1.7344 |

Loss is MSE against the 256x256x1 ground-truth Gaussian-blob heatmap. Full per-epoch loss curves are plotted in `fulleval_loss_curves.png` (same folder) and the raw per-epoch values are in each architecture's own `checkpoints/<name>.done.json`.

## Files in this results folder (`notebooks/outputs_full_eval/`)

```
outputs_full_eval/
├── RESULTS.md                          <- this file
├── architecture_fulleval_results.json   <- raw machine-readable results (source of this report)
├── fulleval_loss_curves.png             <- training loss curves, all 4 architectures
└── checkpoints/
    ├── FNO.weights.h5                     <- trained Keras weights (334,321 params)
    ├── FNO.done.json                      <- full 200-epoch loss history + timing
    ├── GridlessUnfold.weights.h5          <- trained Keras weights (30,201 params)
    ├── GridlessUnfold.done.json           <- full 200-epoch loss history + timing
    ├── SIREN.weights.h5                   <- trained Keras weights (3,565 params)
    ├── SIREN.done.json                    <- full 200-epoch loss history + timing
    ├── WindowAttention.weights.h5         <- trained Keras weights (7,537 params)
    ├── WindowAttention.done.json          <- full 200-epoch loss history + timing
```

## Reproducing / loading a trained model

```python
# From notebooks/DLDOA_Architecture_FullEval_Local.ipynb, after running Cells 1-11
# (which rebuild the exact same architecture definitions):
models['FNO'].load_weights('outputs_full_eval/checkpoints/FNO.weights.h5')
```

## Scope caveats (from the notebook's own Cell 0)

This is a **screening study**, not a claim that any one of these 4 architectures beats the paper's full 64-block ResNet at full scale: all 4 use only 8 blocks (none has pretrained weights to warm-start from), Window-Attention is a local-neighborhood image-attention stand-in (not the literal antenna-element graph), and training ran for a fixed epoch count on synthetic data generated fresh every step (matching the paper's own 'regenerated every epoch, effectively infinite' training-set design, IEEE TVT 2025, Lloria et al., Section IV-B footnote 1) rather than the paper's full 500-epoch schedule.
