# Baseline models registry

Three reference models, all evaluated on the **identical full 8000-sample frozen eval
bank** (`frozen_banks/eval_bank.npz`) with the exact same evaluator
(`evaluate_on_bank` from `DL_DOA/src/TVT_Blob_Inference.py`), so their numbers are
directly, rigorously comparable — not mixed across different sample sizes or eval
code. Keep this folder as the reference point for any future architecture/compression
experiment: compare a new candidate against these 3, on this same eval bank, before
claiming it beats anything.

**Cross-validated:** Teacher was evaluated twice independently — once on GPU (RTX 5090,
the full-eval notebook run) and once on local CPU (this registry's own verification
pass) — and the two runs produced the **exact same mean Pd to 13+ significant figures**
(0.7103495885388549 both times). This confirms the evaluator is fully deterministic and
that mixing GPU-run and CPU-run numbers in the table below is valid.

## Comparison (full 8000-sample bank, mean Pd across all 8 SNR points)

| Model | Params | Mean Pd | vs Teacher | Blocks | Notes |
|---|---|---|---|---|---|
| **Teacher** | 469,393 | 0.7103 | — | 64 | Original pretrained ResNet (paper baseline), `inf_model_007_256_resnet.h5` |
| **Student (r8-magnitude)** | 314,513 (67% of teacher) | 0.6956 | −0.0147 | 64 | Channel-pruned (width 12→8, magnitude-based selection) + distilled from teacher, `DLDOA_Compression_02_PruneAndFinetune.ipynb` |
| **FNO (screening)** | 334,321 (71% of teacher) | 0.6235 | −0.0869 | 8 | Spectral-convolution architecture, trained from scratch (no warm start), `DLDOA_Architecture_FullEval_Local.ipynb` |

**Reading this table honestly:**
- The **Student** is currently the strongest of the two compressed/alternative candidates — smaller than FNO (314K vs 334K params) *and* closer to the teacher's Pd (only −0.0147 vs FNO's −0.0869). It also only needed fine-tuning from teacher weights, not training from scratch.
- **FNO** is still a genuinely strong result *given it trained from scratch* at only 8 blocks (vs the teacher's/student's 64) — the fair comparison for FNO isn't really "does it beat the pruned student," it's "how much can a fundamentally different, un-warm-started architecture close the gap in a fixed, cheap training budget." On that framing it did well (88% of teacher's Pd from scratch). But on a raw "which model should I ship" basis, the Student currently wins.
- Neither compressed/alternative model beats the teacher outright yet.

## Per-SNR breakdown

### Detection probability (Pd)

| Model | -10dB | -5dB | 0dB | 5dB | 10dB | 15dB | 20dB | 25dB |
|---|---|---|---|---|---|---|---|---|
| Teacher | 0.2040 | 0.4366 | 0.6393 | 0.7789 | 0.8623 | 0.8989 | 0.9253 | 0.9376 |
| Student | 0.1962 | 0.4334 | 0.6308 | 0.7592 | 0.8422 | 0.8828 | 0.9051 | 0.9155 |
| FNO | 0.2031 | 0.4134 | 0.6104 | 0.7218 | 0.7960 | 0.7962 | 0.7460 | 0.7009 |

Note the shape difference: Student tracks the Teacher closely at every SNR (small, consistent
gap). FNO tracks well up to ~10dB, then **falls further behind at high SNR (15-25dB)** where
the teacher pulls away — FNO's spectral representation may be losing fine-grained precision
that the teacher's deeper, spatial-conv stack captures once noise stops being the bottleneck.

Full RMSE breakdown and raw numbers: `comparison_full_bank.json`.

## Folder contents

```
baseline_models/
├── README.md                        <- this file
├── comparison_full_bank.json        <- raw Pd/RMSE numbers, all 3 models, machine-readable
├── teacher/
│   └── inf_model_007_256_resnet.h5
├── student_r8_magnitude/
│   └── student_r8_magnitude_lambda0.5.weights.h5
└── fno_screening/
    └── FNO.weights.h5
```

## How to reload each model (for any future comparison experiment)

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, Add, Conv2DTranspose, Layer
from tensorflow.keras.models import Model

# --- Teacher: original 64-block ResNet ---
from src.tvt_models import Resnet
teacher = Resnet(input_shape=(64, 64, 2))
teacher.load_weights('baseline_models/teacher/inf_model_007_256_resnet.h5')

# --- Student: 64-block pruned ResNet, internal width r=8 (of 12) ---
def res_conv_pruned(x, r, out_filters=12):
    skip = x
    x = Conv2D(r, 5, padding='same')(x); x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(out_filters, 5, padding='same')(x); x = BatchNormalization()(x)
    x = Add()([x, skip]); x = Activation('relu')(x)
    return x

def build_pruned_resnet(r, n_blocks=64, input_shape=(64, 64, 2), name=None):
    x_in = Input(shape=input_shape)
    x = Conv2DTranspose(12, (5, 5), strides=(2, 2), padding='same')(x_in)
    for _ in range(n_blocks):
        x = res_conv_pruned(x, r)
    x = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same')(x)
    return Model(x_in, x, name=name or f'PrunedResNet-r{r}')

student = build_pruned_resnet(8, n_blocks=64)
student.load_weights('baseline_models/student_r8_magnitude/student_r8_magnitude_lambda0.5.weights.h5')

# --- FNO: 8-block spectral-convolution architecture ---
# Full body_fn/SpectralConv2D definitions are in DLDOA_Architecture_FullEval_Local.ipynb
# (Cells 6 and 8) -- copy those two cells, then:
fno = build_model_with_body(fno_body, filters=12, n_blocks=8)
fno.load_weights('baseline_models/fno_screening/FNO.weights.h5')

# Evaluate any of them identically:
from src.TVT_Blob_Inference import get_blob_detector, get_blob_peaks, peaks_to_angles, prepare_for_metric, get_ang_difference, filter_angles
# ... use the same evaluate_on_bank() function every other notebook in this project uses,
# on frozen_banks/eval_bank.npz, for a directly comparable number.
```

## Adding a 4th (or 5th...) model to this registry

1. Evaluate it on the **full, unmodified** `frozen_banks/eval_bank.npz` with the same
   `evaluate_on_bank` function (not a subsample — the numbers above are all full-bank).
2. Add its weights file + a short note to this README's table.
3. Append its result to `comparison_full_bank.json`.
4. Re-run the Teacher's evaluation alongside it in the same session as a sanity
   cross-check (should reproduce 0.7103495885388549 to several significant figures) —
   this catches silent environment drift (a different TF version, a changed evaluator,
   a corrupted eval bank) before trusting the new model's number.
