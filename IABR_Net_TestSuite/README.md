# IABR-Net Full Test Suite

A self-contained folder that builds, trains and tests **IABR-Net**, the architecture selected by
the multi-agent research loop (`ResearchState/FINAL_RESEARCH_REPORT.md` in the main repo). It also
runs every baseline against the same data, and writes all results, tables, figures and checkpoints
to `outputs/`.

Everything needed is in this folder: the notebook, the original evaluator and generator code, the
baseline weights, the fixed evaluation bank, and all generated test banks. Nothing is downloaded at
run time.

---

## 1. Quick start (Linux + NVIDIA GPU)

```bash
cd IABR_Net_TestSuite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
jupyter lab notebooks/IABR_Net_Full_Test_Suite.ipynb
```

The last `python -c` line **must print a non-empty GPU list** before you start the full run.

1. **First: the smoke test.** In the notebook's first code cell, set `SMOKE_TEST = True` and choose
   *Run → Run All Cells*. This checks the whole pipeline end to end in a few minutes on GPU and writes
   to `outputs_smoke/`. Every line of `outputs_smoke/RESULTS.md` under "Status of every experiment"
   should say `done`.
2. **Then: the real run.** Set `SMOKE_TEST = False` and *Run All* again. Results go to `outputs/`.
   Smoke results never mix with the real run.

You can also run it headless, for example over SSH or inside `tmux`:

```bash
cd notebooks
IABR_SMOKE=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 IABR_Net_Full_Test_Suite.ipynb --output smoke_run.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 IABR_Net_Full_Test_Suite.ipynb --output full_run.ipynb
```

`IABR_SMOKE=1` overrides the flag without editing the notebook.

### Other operating systems

| OS | GPU support |
|---|---|
| **Linux** (native) | `pip install "tensorflow[and-cuda]"` bundles CUDA/cuDNN (already in `requirements.txt`). This is what the earlier full-eval run used on the RTX 5090. |
| **Windows, native** | **No GPU.** TensorFlow ≥ 2.11 dropped native-Windows GPU support. Use **WSL2** (Ubuntu) and follow the Linux steps inside it. On native Windows the notebook still runs, but on CPU, which is far too slow for the full training. |
| **macOS (Apple Silicon)** | `pip install tensorflow tensorflow-metal`, then the rest of `requirements.txt` minus the `[and-cuda]` extra. |

---

## 2. Folder layout

```
IABR_Net_TestSuite/
├── README.md                      <- this file
├── requirements.txt
├── notebooks/
│   └── IABR_Net_Full_Test_Suite.ipynb   <- the ONE notebook: every test lives here
├── DL_DOA/src/                    <- original project code (evaluator, teacher ResNet), unmodified
│   ├── TVT_Blob_Inference.py      (blob detector, Pd/RMSE metric, Hungarian matching)
│   ├── tvt_models.py              (Teacher ResNet)
│   └── tvt_data_generation_v3.py  (imported by the evaluator)
├── dldoa_dataset_generation.py    <- original data generator (codebooks, channel, angle sampler)
├── baseline_models/               <- the 3 registry models (same files as the main repo's baseline_models/)
│   ├── teacher/inf_model_007_256_resnet.h5
│   ├── student_r8_magnitude/student_r8_magnitude_lambda0.5.weights.h5
│   └── fno_screening/FNO.weights.h5
├── data/
│   ├── frozen_banks/eval_bank.npz     <- FIXED 8000-sample bank (E3); every registry number comes from it
│   └── generated_banks/               <- 56 FIXED generated test banks + MANIFEST.json (sha256 of each)
└── outputs/                       <- created by the run (see section 5)
```

### Datasets

| Data | Kind | Used by |
|---|---|---|
| `data/frozen_banks/eval_bank.npz` | **Fixed** (the project's frozen bank, never regenerated). L=3, SNR −10…25 dB, 1000 per SNR. | E3, Abl-2, V0, E11, Abl-6, E9 |
| `data/generated_banks/*.npz` | **Fixed**, generated once with fixed seeds by the notebook's own bank cell (shipped pre-built). The notebook regenerates any missing file bit-for-bit. `MANIFEST.json` lists each file's sha256 and spec. | V1, E4–E8, SNR tails, E10, controls |
| Training data | **Generated on the fly**, never stored. Fresh random samples every step, same distribution as the original training generator, plus impairments (see below). | all training runs |

Generated bank families (500 samples each unless noted; robustness at SNR 0 and 15 dB):

| Banks | Condition |
|---|---|
| `phase_d{0,1,2,5}_snr{0,15}` | per-antenna phase error δmax (E4). `phase_d0` is the clean L=3 reference. |
| `gain_g{0.5,1,2,4}_snr{0,15}` | per-antenna gain error γmax in dB (E5; **new Phase-0 code**; 4 dB is OOD) |
| `oodphase_d{10,15}_snr{0,15}` | phase error beyond the training range (E6) |
| `L{1,2,4,5,6,7,8,10}_snr{0,15}` | path count (E7; only L=10 is OOD, see section 4) |
| `sep_{30,20,10,5,3,2,1}deg_snr15` | two paths with both AoA and AoD offset by the separation (E8) |
| `tail_snr{-25,-20,30,35}` | SNR outside the −15…24 dB training range |
| `nuis_p{-20,-10,0}_snr{0,15}` | 200 paired scenes each, 3 principal paths + 1 interferer (E10, Notebook-4 protocol) |
| `v1_clean_snr{0,15,25}` | clean L=3 banks for the generator-fidelity check V1 |

Banks store the compact 16×16 observation `Y`. The 64×64 input of the heatmap baselines is rebuilt
with the exact `scipy.ndimage.zoom(order=0)` index map of the original generator. E0a checks this is
bit-exact on the frozen bank.

---

## 3. What the notebook runs (in order)

| # | Experiment | What it answers |
|---|---|---|
| E0a | Phase-0 unit tests | Gain/phase injection is correct; the vectorized simulator equals the original generator (to 1e-15); codebooks are unitary; the inverse-codebook transform recovers `D_rᴴHD_t` exactly; the 16↔64 conversion is exact. **Assert:** if any check fails the run stops. |
| E0b | Model instantiation | IABR-Net's *measured* params, FLOPs and memory, replacing the report's estimates |
| V0 | Registry reproduction | The Teacher reproduces its registry mean Pd (0.7103495885) on the frozen bank |
| V1 | Generator fidelity | Teacher Pd on the new clean banks matches the frozen bank at SNR 0/15/25 |
| TRAIN | 7 full + 12 sweep runs | IABR-Net ×3 seeds, E1, E2, Abl-1, Abl-3 (20,000 steps each), Abl-6 sweep (5,000 steps each) |
| E3 | In-distribution SNR sweep | All models on the full frozen bank |
| Abl-2 | Refinement ablation | Coarse-only vs parabolic vs learned refinement, plus end-fire P95 (the Gap-5 ceiling probe) |
| Controls | E1 / E2 / Abl-1 / Abl-3 | One component changed, same data: is the gain from impairment-adaptivity, capacity, the front end, or SE? |
| E4 / E5 / E6 | Phase / gain / OOD-phase sweeps | Robustness to hardware impairment |
| E7 / E8 / tails | Path count, angular separation, SNR extrapolation | Channel conditions and distribution shift |
| E10 | Nuisance-path test | The project's one earlier "student beats teacher" robustness setting |
| E11 | Multi-seed variance | Std of IABR-Net Pd across 3 seeds |
| Abl-6 | Loss-weight sweep | λ3 ∈ {0, 0.1, 0.5, 1}, λ4 ∈ {0, 0.1, 0.3} |
| E9 | Efficiency | NN-only latency (batch 1/8/32/128, p50/p99), end-to-end latency including peak decoding, FLOPs, GPU peak memory, model size, for every model |

Every model is scored with the **original project evaluator**, which uses Hungarian matching and a
1° threshold. The notebook reports these metrics:

| Metric | Definition |
|---|---|
| `pd_paper` | The original component-level Pd. Samples with too few detections are dropped. Comparable to every earlier number. |
| `pd_strict` | Dropped samples count as misses. |
| `pd_source` | A path counts only if both AoA and AoD are within 1°. |
| `pd_aoa`, `pd_aod`, `rmse_aoa`, `rmse_aod` | AoA and AoD reported separately. |
| `p50`, `p95` | Absolute angle error percentiles. |
| `p95_endfire`, `p95_broadside` | P95 error for true angles within 30° of end-fire, and for the rest. |
| `pairing_error` | Share of paths where only one of AoA/AoD is within 1°. |

---

## 4. Decisions this suite had to make (the report left them open)

All are also marked **[ASSUMPTION]** in the notebook.

1. **Gain-error model.** Per-antenna amplitude error with `20·log10(g) ~ U(−γmax, γmax)`. This is
   symmetric-uniform, the same convention the original code uses for phase error.
2. **Stage 0 (Addendum A1).** Input is the observed 16×16 beamspace `Y`. Stage 0 is the inverse
   codebook `W_ideal·Y·F_idealᴴ`, which is exact because the 16×16 DFT codebooks are unitary. The
   agents' diagram wrongly assumed a raw antenna-domain input.
3. **IABC-v2 "own(mag, phase)" features.** The report does not define them. They are implemented as
   the element's log-energy ratio, plus its phase residual after removing the dominant path's phase
   and the linear trend. A linear phase slope is not identifiable, because it is the same as an
   angle shift.
4. **Training.**
   - Optimizer: Adam, lr 1e-3, 500-step warm-up, cosine decay to 1e-5, global-norm clip 5, batch 256,
     20,000 steps (the same step budget as the earlier screening).
   - Data per sample: L ∈ {1..9}, SNR ∈ {−15..24} dB (same as the original generator), phase
     δmax ~ U(0, 5°), gain γmax ~ U(0, 2 dB).
   - Loss: CenterNet-style focal loss on the heatmap, and offsets trained at ground-truth peak cells.
   - Trunk convolutions use circular padding, because DFT beamspace is periodic.
5. **OOD definitions.**
   - The original generator trains with **L ∈ {1..9}**, so L = 7 and L = 8 are *in-distribution*.
     Only L = 10 is OOD. This corrects the report, which called 7 and 8 OOD.
   - Other OOD conditions: gain 4 dB, phase 10°/15°, and SNR −25/−20/30/35 dB.
6. **DFT-SIC grid.** 512 (the sanity notebook used 1024) to keep evaluation time reasonable. It is
   scored with the same paper-style evaluator as the other models. The report's "Pd 0.884" came
   from a per-source metric in the sanity notebook, so the new number will differ.
7. **Fairness caveat.** Teacher, Student and FNO were trained on **clean** data by their original
   pipelines. IABR-Net and all its controls are trained with impairments. So the **controlled**
   robustness comparison is **IABR-Net vs E1 / Abl-1 / E2 / Abl-3**, which use the same data and
   differ in one component. The baselines' robustness numbers are reference points only.

---

## 5. Outputs

```
outputs/
├── RESULTS.md                 <- every table in one file, plus the status of every experiment  (read this)
├── results/<experiment>.json  <- full numbers per experiment (per-SNR, per-condition, all metrics)
├── results/status.json        <- done / FAILED / skipped per experiment
├── tables/*.md                <- each table on its own
├── figures/*.png              <- SNR curves, robustness curves, training loss
├── checkpoints/<run>/         <- final.weights.h5, done.json, history.json, resumable ckpt/
├── cache/                     <- cached predictions (re-running tables costs nothing)
└── logs/run_log.txt, errors.log, config.json
```

---

## 6. Resuming, re-running, and failures

- **Interrupted run** (power cut, closed laptop, killed kernel): just *Run All* again. Finished
  experiments load from `outputs/results/`. A training run resumes from its last checkpoint (every
  2,000 steps), with the optimizer state included.
- **Re-run one experiment:** delete `outputs/results/<name>.json` (and the matching
  `outputs/cache/pred__*` files, if you changed a model). To retrain a model, delete
  `outputs/checkpoints/<run>/`.
- **A failing experiment does not stop the suite.** Its traceback goes to `outputs/logs/errors.log`
  and the `RESULTS.md` status table shows it as `FAILED`. Experiments that depend on it are marked
  `skipped`. The only hard stop is E0a: if the physics checks fail, nothing downstream can be trusted.

## 7. Time

This suite has **not yet been run at full scale on any GPU**. On the local CPU it was verified only
in smoke mode, so there are no measured full-run timings. The notebook logs a **measured ETA** after
the first few hundred steps of every training run. Check that line before leaving the run
unattended. Expect training to be limited by on-the-fly data generation (Python/NumPy, 4 parallel
workers) rather than by the GPU, because IABR-Net itself is small.
