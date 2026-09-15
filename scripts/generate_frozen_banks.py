"""
Generate frozen evaluation bank + calibration bank for the structured-compression
project, using the ORIGINAL repo's validation_data_generator (imported directly,
never reimplemented) so there is zero drift from the paper's own protocol.

Run locally (no TF needed -- pure numpy/scipy). Output goes to
D:\\ai_ml_project\\frozen_banks\\ -- zip that folder and upload as a Kaggle dataset.

Eval bank:  matches run_inference_and_metrics_resnet's exact protocol
            (L=[3], SNR=-10..25 step 5, P=[16], nt=nr=[16], seed=42,
            sigma=0.07, M=256, amps_type='ones', examples_per_condition=1000).
            Only (data, feat, meta) are stored -- the original evaluator never
            reads the GT heatmap, only the peak-derived angles vs feat, so this
            keeps the bank small (~256 MB instead of ~2 GB).

Calibration bank: same protocol but examples_per_condition=75 (600 total),
            WITH gt stored (float16) since Taylor-importance scoring in
            notebook 2 needs a GT-based loss gradient.
"""
import sys, os, time
import numpy as np

REPO_ROOT = r"D:\ai_ml_project"
DL_DOA_DIR = os.path.join(REPO_ROOT, "DL_DOA")
sys.path.insert(0, DL_DOA_DIR)  # so `from src.tvt_data_generation_v3 import ...` resolves

from src.tvt_data_generation_v3 import validation_data_generator  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "frozen_banks")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- exact protocol from TVT_Blob_Inference.run_inference_and_metrics_resnet ----
L_VALUES   = [3]
SNR_VALUES = list(range(-10, 30, 5))   # -10,-5,0,5,10,15,20,25
PQ_VALUES  = [16]
NTNR_VALUES = [16]
SIGMA = 0.07
M = 256
SEED = 42


def build_conditions(examples_per_condition):
    import itertools
    return list(itertools.product(L_VALUES, SNR_VALUES, PQ_VALUES, NTNR_VALUES)), examples_per_condition


def generate_eval_bank(examples_per_condition=1000):
    conditions, epc = build_conditions(examples_per_condition)
    gen = validation_data_generator(
        validation_conditions=conditions,
        examples_per_condition=epc,
        seed=SEED, M=M, normalize_gt=False, sigma=SIGMA, amps_type='ones',
    )
    total = len(conditions) * epc
    data_list, feat_list, meta_list = [], [], []
    t0 = time.time()
    for i, (data, gt, feat, meta) in enumerate(gen):
        data_list.append(data)              # (64,64,2) float32
        feat_list.append(feat)              # (2,L) float32  -- L=3 fixed here, uniform
        meta_list.append(meta)              # (4,) float32
        if (i+1) % 1000 == 0:
            print(f'  eval bank: {i+1}/{total}  ({time.time()-t0:.1f}s)')

    data_arr = np.stack(data_list).astype(np.float32)   # (N,64,64,2)
    feat_arr = np.stack(feat_list).astype(np.float32)   # (N,2,3)
    meta_arr = np.stack(meta_list).astype(np.float32)   # (N,4)

    path = os.path.join(OUT_DIR, 'eval_bank.npz')
    np.savez_compressed(path, data=data_arr, feat=feat_arr, meta=meta_arr,
                        sigma=SIGMA, M=M, seed=SEED)
    sz = os.path.getsize(path) / 1e6
    print(f'✅ eval_bank.npz saved: {data_arr.shape[0]} samples, {sz:.1f} MB')
    print(f'   data shape={data_arr.shape}  feat shape={feat_arr.shape}  meta shape={meta_arr.shape}')
    return data_arr, feat_arr, meta_arr


def generate_calibration_bank(examples_per_condition=75):
    conditions, epc = build_conditions(examples_per_condition)
    gen = validation_data_generator(
        validation_conditions=conditions,
        examples_per_condition=epc,
        seed=SEED + 1,   # different seed -- calibration bank must NOT overlap the eval bank
        M=M, normalize_gt=False, sigma=SIGMA, amps_type='ones',
    )
    total = len(conditions) * epc
    data_list, gt_list, feat_list, meta_list = [], [], [], []
    t0 = time.time()
    for i, (data, gt, feat, meta) in enumerate(gen):
        data_list.append(data)
        gt_list.append(gt.astype(np.float16))    # keep small -- calibration bank only
        feat_list.append(feat)
        meta_list.append(meta)
        if (i+1) % 200 == 0:
            print(f'  calibration bank: {i+1}/{total}  ({time.time()-t0:.1f}s)')

    data_arr = np.stack(data_list).astype(np.float32)
    gt_arr   = np.stack(gt_list).astype(np.float16)     # (N,256,256,1) float16
    feat_arr = np.stack(feat_list).astype(np.float32)
    meta_arr = np.stack(meta_list).astype(np.float32)

    path = os.path.join(OUT_DIR, 'calibration_bank.npz')
    np.savez_compressed(path, data=data_arr, gt=gt_arr, feat=feat_arr, meta=meta_arr,
                        sigma=SIGMA, M=M, seed=SEED+1)
    sz = os.path.getsize(path) / 1e6
    print(f'✅ calibration_bank.npz saved: {data_arr.shape[0]} samples, {sz:.1f} MB')
    print(f'   data shape={data_arr.shape}  gt shape={gt_arr.shape}  feat shape={feat_arr.shape}')
    return data_arr, gt_arr, feat_arr, meta_arr


if __name__ == '__main__':
    print('='*70)
    print('Generating EVAL bank (matches run_inference_and_metrics_resnet exactly)')
    print('='*70)
    generate_eval_bank(examples_per_condition=1000)

    print()
    print('='*70)
    print('Generating CALIBRATION bank (SNR-stratified, includes GT for Taylor scoring)')
    print('='*70)
    generate_calibration_bank(examples_per_condition=75)

    print()
    print('Done. Files in:', OUT_DIR)
    for f in os.listdir(OUT_DIR):
        p = os.path.join(OUT_DIR, f)
        print(f'  {f}  ({os.path.getsize(p)/1e6:.1f} MB)')
