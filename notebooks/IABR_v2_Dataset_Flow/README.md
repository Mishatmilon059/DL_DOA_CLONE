# IABR-v2 dataset generation flow (CPU only)

Open `IABR_v2_Dataset_Flow.ipynb` - all outputs and figures are already saved inside it (no need to run).
Or open `IABR_v2_Dataset_Flow.html` in any browser for a static copy.

| Item | What it is |
|---|---|
| `IABR_v2_Dataset_Flow.ipynb` | executed notebook (numpy + matplotlib, runs on CPU in about 30 s) |
| `IABR_v2_Dataset_Flow.html` | same notebook as a self-contained web page |
| `figures/` | every figure of the notebook as PNG (`fig01..fig16`) + the train-vs-test pipeline diagram |
| `run_log.txt` | printed text outputs (tables, sanity checks: 10/10 pass) |
| `data/demo_scene_L3.npz` | the demo scene: angles, gains, clean/impaired/noisy Y, noise, impairment, labels |
| `data/train_recipe_batch64.npz` | 64 samples from the exact training recipe (y, heat, imp, L, SNR, ...) |

Re-run: `jupyter nbconvert --to notebook --execute --inplace IABR_v2_Dataset_Flow.ipynb` (or run all cells).
