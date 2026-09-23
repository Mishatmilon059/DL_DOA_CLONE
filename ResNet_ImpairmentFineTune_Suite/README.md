# ResNet Impairment-Aware Fine-Tuning — Freeze-Ratio Sweep

Self-contained folder: notebook + original code + baseline weights + frozen eval bank + this
README. Everything needed is here — nothing is downloaded at run time.

Adapts Meneses-Albalá et al.'s U-Net fine-tuning idea (freeze part of the network, fine-tune
the rest, on phase-impaired data) to this project's **ResNet** (Teacher, 64 blocks; Student,
also 64 blocks, pruned width r=8). ResNet has no encoder/decoder split — one upsampling layer,
then 64 structurally identical residual blocks at a fixed resolution, then one final
upsampling layer — so instead of "freeze encoder, tune decoder," this freezes the **first K of
64 blocks** and fine-tunes the last (64−K), the standard deep-CNN transfer-learning analogue.

**Sweep:** freeze ratio K/64 ∈ {0%, 20%, 50%, 80%, 90%} (the paper's own 5 points) × 2 base
models (Teacher, Student) = **10 fine-tuning runs**, same code path for both, 200 epochs each
with the paper's own hyperparameters (Adam, lr 1e-4, batch 32, plateau LR decay, 80-epoch
early stopping).

## 1. Quick start (Linux + NVIDIA GPU)

```bash
cd ResNet_ImpairmentFineTune_Suite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
jupyter lab notebooks/DLDOA_ResNet_ImpairmentFineTune.ipynb
```

The last line **must print a non-empty GPU list** before starting the real run.

1. **First, the smoke test.** In the notebook's first code cell, `SMOKE_TEST` is read from the
   `FT_SMOKE` environment variable (defaults to off). Either set `SMOKE_TEST = True` directly in
   that cell, or run headless with the env var:
   ```bash
   cd notebooks
   FT_SMOKE=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
       DLDOA_ResNet_ImpairmentFineTune.ipynb --output smoke_run.ipynb
   ```
   Takes a few minutes. Check `outputs_resnet_ft_smoke/RESULTS.md` — every run should say `done`,
   and `outputs_resnet_ft_smoke/logs/errors.log` should not exist (or be empty).
2. **Then the real run** — set `SMOKE_TEST = False` (or just don't set `FT_SMOKE`) and run all
   cells, or headless:
   ```bash
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
       DLDOA_ResNet_ImpairmentFineTune.ipynb --output full_run.ipynb
   ```

### Other operating systems

| OS | GPU support |
|---|---|
| **Linux** (native) | `pip install "tensorflow[and-cuda]"` bundles CUDA/cuDNN (already in `requirements.txt`). |
| **Windows, native** | **No GPU.** TensorFlow ≥ 2.11 dropped native-Windows GPU support. Use **WSL2** (Ubuntu) and follow the Linux steps inside it. |
| **macOS (Apple Silicon)** | `pip install tensorflow tensorflow-metal`, then the rest of `requirements.txt` minus the `[and-cuda]` extra. |

## 2. Folder layout

```
ResNet_ImpairmentFineTune_Suite/
├── README.md
├── requirements.txt
├── notebooks/
│   └── DLDOA_ResNet_ImpairmentFineTune.ipynb   <- open and run this
├── DL_DOA/
│   ├── src/
│   │   ├── TVT_Blob_Inference.py    (blob detector, Pd/RMSE metric, Hungarian matching)
│   │   ├── tvt_models.py            (Teacher ResNet architecture)
│   │   └── tvt_data_generation_v3.py (imported by the evaluator)
│   └── models/
│       └── inf_model_007_256_resnet.h5   (Teacher pretrained weights)
├── baseline_models/
│   └── student_r8_magnitude/
│       └── student_r8_magnitude_lambda0.5.weights.h5   (Student pretrained weights)
├── frozen_banks/
│   └── eval_bank.npz                (fixed 8000-sample eval bank — used for the reproducibility check)
└── dldoa_dataset_generation.py      (original physics/codebook code, reused for training-data generation)
```

No large dataset files are needed beyond `eval_bank.npz` (~19 MB) — training data is generated
synthetically on the fly.

## 3. Performance fix already applied (read before trusting a long unattended run)

An earlier version of the training-data generator built each batch **sample-by-sample in pure
Python**, measured live on Kaggle at **144s/epoch → ~8h for a single run** (of 10 planned) — the
GPU step itself is fast (469K-param model), the bottleneck was single-threaded CPU physics
simulation. Rewritten to build the whole batch at once with `einsum`, fixing the codebook size
at P=Q=nt=nr=16 (matching every eval bank and baseline comparison in this project already).

**The single biggest cost was the ground-truth heatmap**: computing the full 256×256 Gaussian
per path via broadcasting was 98.5% of a batch's time. Because the 2D Gaussian here is
separable, it's now built from two small 1D factors and combined with `einsum` — verified
numerically identical (float32 noise only) to the original full-2D computation, **~31x faster**
on just that step. Net effect measured on a local CPU: **~144s/epoch → ~5.9s/epoch (~24x)**,
projecting the full 10-run sweep at a few hours instead of ~80.

Both the vectorized physics (channel/observation/GT-heatmap math) and the exact
nearest-neighbor upsampling (bit-exact to `scipy.ndimage.zoom`, not approximated with
`np.repeat`, which was tested and found NOT equivalent) were verified against the original
per-sample functions before being trusted — see the notebook's own Cell 6 markdown for the
full explanation and what was checked.

A second real bug was also caught and fixed before handover: the freeze-mask boolean was
inverted, which froze everything except the input stem regardless of the configured freeze
ratio (caught because a 0%-freeze run showed only 612 trainable params — exactly the stem's own
param count — instead of ~the full model). Fixed and verified (0% freeze now correctly leaves
~466,321/469,393 Teacher params trainable). See the notebook's `set_freeze()` docstring.

## 4. Outputs (saved properly, nothing lost)

Everything lands under `outputs_resnet_ft/` (or `outputs_resnet_ft_smoke/` in smoke mode), next
to wherever the notebook is run from — a **real folder on this machine's disk**, not an
ephemeral cloud working directory, so unlike Kaggle there is no "forgot to Save Version" risk;
just copy the folder back when done.

```
outputs_resnet_ft/
├── RESULTS.md                 <- every table + every run's status in one file (read this first)
├── logs/
│   ├── run_log.txt            <- full timestamped log of everything printed during the run
│   ├── config.json            <- exact hyperparameters used for this run
│   └── errors.log             <- tracebacks of any FAILED run (empty/absent if none failed)
├── results/
│   ├── status.json            <- done / FAILED per run, at a glance
│   ├── BASE_<model>.json      <- base-model scores (impaired + clean)
│   ├── FT_<run>.json          <- fine-tune run summary (freeze ratio, K, params, best loss...)
│   └── sweep_rows.json        <- the full Base-vs-Fine-tuned comparison table, machine-readable
├── tables/sweep.md            <- the comparison table alone, as markdown
├── figures/
│   ├── freeze_sweep.png       <- delta-Pd vs freeze ratio, impaired + clean, both models
│   └── loss_curves.png        <- per-epoch training loss, all 10 runs overlaid
└── checkpoints/<model>_freeze<NN>/
    ├── best.weights.h5        <- lowest-training-loss checkpoint (what gets evaluated)
    ├── final.weights.h5       <- last-epoch weights
    ├── history.json           <- every epoch's loss/lr, for `loss_curves.png` and manual inspection
    ├── done.json              <- run summary (also feeds RESULTS.md)
    └── ckpt/                  <- TF checkpoint used to resume an interrupted run
```

Per-epoch progress prints to the console/log throughout (loss every epoch, a small live
accuracy check every 20 epochs), so you can watch a run without waiting for it to finish.

## 5. Resuming an interrupted run

Just re-run the notebook (or the same `jupyter nbconvert --execute` command). Finished
experiments load from `outputs_resnet_ft/results/*.json` instantly. An interrupted fine-tune
run resumes from its last checkpoint (saved every 20 epochs, optimizer state included) rather
than restarting from epoch 0. A failing run does not stop the rest of the sweep — its traceback
goes to `errors.log` and `RESULTS.md`'s status table shows it as `FAILED`; everything else
still completes.

## 6. Assumptions made (the paper doesn't specify these — see notebook Cell 0/6 for details)

- `delta_max = 5°` only (not swept over the paper's {1°,2°,5°}) — keeps the sweep to 10 runs,
  not 30. Change `CFG['delta_max_deg']` in the config cell to test another value.
- `steps_per_epoch = 40` (paper gives batch size but not steps/epoch) — fine-tuning from an
  already-converged model needs far fewer updates than training from scratch.
- LR-plateau and early-stopping use **training loss**, not a separate validation loss — a real
  validation pass needs a full blob-detection evaluation, too slow to run every epoch.
- Training data is fixed at P=Q=nt=nr=16 (see §3) instead of the original generator's random
  {16,32} — matches every eval bank and baseline comparison already used throughout this
  project.
