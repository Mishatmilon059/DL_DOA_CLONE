"""
==========================================================================
DLDOA Dataset Generation
==========================================================================
Reproduces the synthetic dataset from:

    "Deep-Learning-Based AoA and AoD Estimation in Analog Millimeter Wave
     MIMO Systems"
    D. Lloria, S. Roger, C. Botella-Mascarell, M. Cobos
    IEEE Transactions on Vehicular Technology, Vol. 75, No. 6, June 2026
    DOI: 10.1109/TVT.2025.3637908

Paper Section IV-B & Table I:
  - Training set   : 10 000 synthetic samples per epoch (infinite generator)
  - Validation set : 1 000 fixed samples (seed=42)
  - L (paths)      : random in {1, ..., 9}
  - SNR            : random in [-15, 24] dB
  - Codebook P=Q   : random in {16, 32}
  - Antennas nt=nr : 16 if P=16; {16, 32} if P=32
  - alpha_l        : CN(0, 1/L)
  - AoA, AoD       : uniform in [0, pi], min pairwise separation pi/6
  - Ground truth   : sum of L 2-D Gaussians, sigma=0.07, grid M=N=256
  - Input size     : 64 x 64 x 2  (real + imag after nearest-neighbor upsample)
  - Output size    : 256 x 256 x 1
==========================================================================
"""

import math
import os
import random
import itertools

import numpy as np
import scipy.ndimage
from scipy.optimize import linear_sum_assignment


# =========================================================================
# Helpers
# =========================================================================

class _HermitianArray(np.ndarray):
    """ndarray subclass that exposes a Hermitian-transpose property (.H)."""
    @property
    def H(self):
        return self.conj().T


def _as_hermitian(arr):
    return arr.view(_HermitianArray)


def wrap_to_2pi(x):
    """Wrap angle(s) to [0, 2*pi)."""
    return np.mod(x, 2 * np.pi)


# =========================================================================
# Core signal-processing building blocks
# =========================================================================

def steering_vector(n_antennas, angle):
    """
    Half-wavelength ULA steering vector (Eq. 2-3 of paper).

        a(angle) = (1/sqrt(n)) * [1, e^{-j*pi*cos}, ..., e^{-j*pi*(n-1)*cos}]^T

    Returns shape (n_antennas, 1), complex.
    """
    k = np.arange(n_antennas)
    v = (1.0 / np.sqrt(n_antennas)) * np.exp(-1j * np.pi * np.cos(angle) * k)
    return v[:, np.newaxis]


def make_tx_codebook(P, nt, phase_error_deg=None):
    """
    Build the (nt x P) TX beamforming codebook F (DFT-based, Eq. 16 of TSDCE).

    phase_error_deg : max per-antenna static phase error in degrees (Section IV-D).
                      None = ideal codebook.
    """
    p = np.arange(P)
    cos_p = (1.0 / np.pi) * np.angle(np.exp(1j * (2 * np.pi / P) * p))
    phi_p = np.arccos(cos_p)

    phase_err = (np.random.uniform(-np.deg2rad(phase_error_deg),
                                    np.deg2rad(phase_error_deg), size=nt)
                 if phase_error_deg is not None else np.zeros(nt))

    F = np.zeros((nt, P), dtype=complex)
    for i, phi in enumerate(phi_p):
        f_ideal = np.squeeze(steering_vector(nt, phi), axis=-1)
        F[:, i] = f_ideal * np.exp(1j * phase_err)
    return F


def make_rx_codebook(Q, nr, phase_error_deg=None):
    """
    Build the (nr x Q) RX combining codebook W (mirror of TX, Eq. 16 of TSDCE).
    """
    q = np.arange(Q)
    cos_q = (1.0 / np.pi) * np.angle(np.exp(-1j * (2 * np.pi / Q) * q))
    phi_q = np.arccos(cos_q)

    phase_err = (np.random.uniform(-np.deg2rad(phase_error_deg),
                                    np.deg2rad(phase_error_deg), size=nr)
                 if phase_error_deg is not None else np.zeros(nr))

    W = np.zeros((nr, Q), dtype=complex)
    for i, phi in enumerate(phi_q):
        w_ideal = np.squeeze(steering_vector(nr, phi), axis=-1)
        W[:, i] = w_ideal * np.exp(1j * phase_err)
    return W


def generate_channel(nr, nt, aod_list, aoa_list, alpha_l):
    """
    Build the (nr x nt) mmWave MIMO channel matrix (Eq. 1 of paper):

        H = sqrt(nt*nr) * sum_l  alpha_l * a_r(aoa_l) * a_t^H(aod_l)
    """
    H = np.zeros((nr, nt), dtype=complex)
    for l, alpha in enumerate(alpha_l):
        at = _as_hermitian(steering_vector(nt, aod_list[l])).H   # (1, nt)
        ar = steering_vector(nr, aoa_list[l])                     # (nr, 1)
        H += alpha * (ar @ at)
    return np.sqrt(nt * nr) * H


def generate_noise(Q, P, SNR_dB, rng=None):
    """
    Complex AWGN noise matrix of shape (Q, P).

    Variance: sigma_n^2 = 10^(-SNR/10)  (rho=1, Eqs. 4-5 of paper).
    Real and imaginary parts are i.i.d. N(0, sigma_n^2 / 2).
    """
    if rng is None:
        rng = np.random.default_rng()
    var_noise = 10 ** (-SNR_dB / 10.0)
    sigma = np.sqrt(var_noise / 2.0)
    return sigma * (rng.standard_normal((Q, P)) + 1j * rng.standard_normal((Q, P)))


def generate_angles(L, min_sep=np.pi / 6, rng=None, max_attempts=10000):
    """
    Sample L (AoD, AoA) pairs uniformly from [0, pi]^2 with minimum
    pairwise Euclidean separation >= min_sep (paper: pi/6).

    Returns two lists: aod_list, aoa_list.
    """
    if rng is None:
        rng = np.random.default_rng()

    pts = []
    attempts = 0
    while len(pts) < L:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not place {L} points with min_sep={min_sep:.4f} "
                f"after {max_attempts} attempts."
            )
        x = rng.uniform(0, math.pi)
        y = rng.uniform(0, math.pi)
        if all(math.hypot(x - px, y - py) >= min_sep for px, py in pts):
            pts.append((x, y))
        attempts += 1

    aod_list = [p[0] for p in pts]
    aoa_list = [p[1] for p in pts]
    return aod_list, aoa_list


# =========================================================================
# Ground-truth heatmap (Eq. 11)
# =========================================================================

def _gaussian_2d(dist_x, dist_y, sigma):
    coeff = 1.0 / (2.0 * np.pi * sigma ** 2)
    return coeff * np.exp(-(dist_x ** 2 + dist_y ** 2) / (2.0 * sigma ** 2))


def generate_gt(aod_list, aoa_list, M=256, sigma=0.07, margin_factor=3.0):
    """
    Build the M x M ground-truth heatmap as the sum of L 2-D Gaussians.

    Spatial frequencies:
        omega_phi = pi * cos(aod_l)       [horizontal axis]
        omega_psi = -pi * cos(aoa_l)      [vertical axis]

    The grid spans [-margin, 2*pi + margin] to avoid edge clipping.
    Returns shape (M, M).
    """
    omega_phi = wrap_to_2pi(np.pi * np.cos(aod_list))
    omega_psi = wrap_to_2pi(-np.pi * np.cos(aoa_list))

    margin = margin_factor * sigma
    ax = np.linspace(-margin, 2 * np.pi + margin, M, endpoint=False)
    Wx, Wy = np.meshgrid(ax, ax)   # (M, M) each; Wx = horizontal, Wy = vertical

    heatmap = np.zeros((M, M))
    for phi, psi in zip(omega_phi, omega_psi):
        heatmap += _gaussian_2d(Wx - phi, Wy - psi, sigma)
    return heatmap


# =========================================================================
# Observation matrix preprocessing
# =========================================================================

def observation_to_input(Y, P):
    """
    Convert complex (Q, P) observation matrix to float32 (64, 64, 2) input.

    Real and imaginary parts are stacked on the last axis, then
    nearest-neighbor upsampled:
        zoom x4 if P=16  (16 -> 64)
        zoom x2 if P=32  (32 -> 64)
    """
    zoom = 4 if P == 16 else 2
    real_part = scipy.ndimage.zoom(np.real(Y), zoom, order=0)
    imag_part = scipy.ndimage.zoom(np.imag(Y), zoom, order=0)
    return np.dstack((real_part, imag_part)).astype(np.float32)


# =========================================================================
# Single-sample generator (shared core)
# =========================================================================

def _generate_one_sample(L, SNR, P, nt, sigma, M, rng, phase_error_deg=None):
    """
    Generate one (input, ground_truth, aod_list, aoa_list) sample.

    Used by both training and validation generators.
    """
    Q = P    # codebook size: always Q=P (square beamspace matrix)
    nr = nt  # symmetric array: nr=nt

    F = make_tx_codebook(P, nt, phase_error_deg)
    W = make_rx_codebook(Q, nr, phase_error_deg)

    # Path gains: alpha_l ~ CN(0, 1/L), sorted strongest-first
    alpha_l = (np.sqrt(1.0 / L) / np.sqrt(2.0)
               * (rng.standard_normal(L) + 1j * rng.standard_normal(L)))
    alpha_l = alpha_l[np.argsort(np.abs(alpha_l))[::-1]]

    aod_list, aoa_list = generate_angles(L, rng=rng)

    H = generate_channel(nr, nt, aod_list, aoa_list, alpha_l)
    G = _as_hermitian(W).H @ H @ F          # noiseless observation (Q, P)
    Z = generate_noise(Q, P, SNR, rng=rng)
    Y = G + Z

    data = observation_to_input(Y, P)
    gt = generate_gt(aod_list, aoa_list, M=M, sigma=sigma)
    gt = np.expand_dims(gt, axis=-1).astype(np.float32)  # (M, M, 1)

    return data, gt, aod_list, aoa_list


# =========================================================================
# Training data generator  (infinite, unseeded — varies every epoch)
# =========================================================================

def training_data_generator(sigma=0.07, M=256):
    """
    Infinite generator yielding (input, ground_truth) pairs for training.

    Randomisation follows Section IV-B of the paper:
      L   in {1, ..., 9},  SNR in {-15, ..., 24} dB,
      P=Q in {16, 32},     nt=nr <= P.

    Training data is intentionally not seeded so the effective training
    set is unlimited (paper footnote 1, Section IV-B).

    Yields
    ------
    data : np.ndarray (64, 64, 2)  float32
    gt   : np.ndarray (M, M, 1)   float32
    """
    while True:
        L   = np.random.randint(1, 10)
        SNR = np.random.randint(-15, 25)
        P   = random.choice([16, 32])
        nt  = 16 if P == 16 else random.choice([16, 32])

        rng = np.random.default_rng()
        data, gt, _, _ = _generate_one_sample(L, SNR, P, nt, sigma, M, rng)
        yield data, gt


# =========================================================================
# Validation / test data generator  (seeded, reproducible)
# =========================================================================

def validation_data_generator(conditions, examples_per_condition=1,
                               seed=42, sigma=0.07, M=256,
                               phase_error_deg=None):
    """
    Deterministic generator yielding (data, gt, features, meta) tuples.

    Parameters
    ----------
    conditions : list of (L, SNR, P, nt) tuples
    examples_per_condition : int
    seed : int
    sigma : float  (paper: 0.07)
    M : int        (paper: 256)
    phase_error_deg : float or None  (hardware impairment study, Section IV-D)

    Yields
    ------
    data     : np.ndarray (64, 64, 2)  float32
    gt       : np.ndarray (M, M, 1)   float32
    features : np.ndarray (2, L)       float32  — [aoa_list; aod_list]
    meta     : np.ndarray (4,)         float32  — [L, SNR, P, nt]
    """
    rng = np.random.default_rng(seed)

    for (L, SNR, P, nt) in conditions:
        for _ in range(examples_per_condition):
            data, gt, aod_list, aoa_list = _generate_one_sample(
                L, SNR, P, nt, sigma, M, rng, phase_error_deg
            )
            features = np.stack([aoa_list, aod_list]).astype(np.float32)
            meta = np.array([L, SNR, P, nt], dtype=np.float32)
            yield data, gt, features, meta


# =========================================================================
# Condition set generators  (paper-exact parameter grids)
# =========================================================================

def make_validation_conditions(n=1000, seed=123):
    """
    1 000 random (L, SNR, P, nt) conditions for the fixed validation set.
    Mirrors the training distribution with a fixed seed (paper Table I).
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L   = int(rng.integers(1, 10))
        SNR = int(rng.integers(-15, 25))
        P   = int(rng.choice([16, 32]))
        nt  = 16 if P == 16 else int(rng.choice([16, 32]))
        out.append((L, SNR, P, nt))
    return out


def make_test_conditions_snr_sweep():
    """
    Test conditions for Figs. 5-6 of the paper:
      L=3, nt=nr=16, P=Q=16, SNR in {-10, -5, 0, 5, 10, 15, 20, 25} dB.
    1 000 realisations per SNR point = 8 000 samples total.
    """
    return list(itertools.product([3], range(-10, 30, 5), [16], [16]))


def make_test_conditions_path_sweep():
    """
    Test conditions for Fig. 8 of the paper:
      L in {1,...,6}, nt=nr=16, P=Q=16, SNR in {0, 20} dB.
    1 000 realisations per (L, SNR) point = 12 000 samples total.
    """
    return list(itertools.product(range(1, 7), [0, 20], [16], [16]))


def make_test_conditions_hardware(error_deg_values=(0, 1, 2, 5)):
    """
    Conditions for the hardware impairment study (Table II, Section IV-D):
      L=3, nt=nr=16, P=Q=16, SNR in {0, 20} dB.
    Pass the returned conditions together with the matching phase_error_deg
    to validation_data_generator.
    """
    return list(itertools.product([3], [0, 20], [16], [16]))


# =========================================================================
# Dataset savers  (persist to .npz / .npy for offline use)
# =========================================================================

def save_training_dataset(output_dir, n_samples=10000, sigma=0.07, M=256):
    """
    Save n_samples training examples (one epoch snapshot) to disk.

    Files written:
        {output_dir}/train_data.npz  — inputs  shape (n, 64, 64, 2)
        {output_dir}/train_gt.npz   — targets shape (n, M, M, 1)
    """
    os.makedirs(output_dir, exist_ok=True)
    gen = training_data_generator(sigma=sigma, M=M)

    data_list, gt_list = [], []
    for i, (data, gt) in enumerate(gen):
        if i >= n_samples:
            break
        data_list.append(data)
        gt_list.append(gt)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_samples} training samples generated")

    data_arr = np.array(data_list)
    gt_arr   = np.array(gt_list)
    np.savez_compressed(os.path.join(output_dir, 'train_data.npz'), data=data_arr)
    np.savez_compressed(os.path.join(output_dir, 'train_gt.npz'),   data=gt_arr)
    print(f"Saved: inputs {data_arr.shape}, targets {gt_arr.shape}")
    return data_arr, gt_arr


def save_validation_dataset(output_dir, seed=42, sigma=0.07, M=256):
    """
    Save the fixed 1 000-sample validation set.

    Files written:
        {output_dir}/val_data.npz     — inputs    (1000, 64, 64, 2)
        {output_dir}/val_gt.npz      — targets   (1000, M, M, 1)
        {output_dir}/val_meta.npz    — conditions (1000, 4)
        {output_dir}/val_features.npy — angles    list of (2, L) arrays
    """
    os.makedirs(output_dir, exist_ok=True)
    conditions = make_validation_conditions()
    gen = validation_data_generator(conditions, seed=seed, sigma=sigma, M=M)

    data_list, gt_list, feat_list, meta_list = [], [], [], []
    for i, (data, gt, feat, meta) in enumerate(gen):
        data_list.append(data)
        gt_list.append(gt)
        feat_list.append(feat)
        meta_list.append(meta)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/1000 validation samples generated")

    data_arr = np.array(data_list)
    gt_arr   = np.array(gt_list)
    meta_arr = np.array(meta_list)
    np.savez_compressed(os.path.join(output_dir, 'val_data.npz'),  data=data_arr)
    np.savez_compressed(os.path.join(output_dir, 'val_gt.npz'),    data=gt_arr)
    np.savez_compressed(os.path.join(output_dir, 'val_meta.npz'),  data=meta_arr)
    np.save(os.path.join(output_dir, 'val_features.npy'),
            np.array(feat_list, dtype=object), allow_pickle=True)
    print(f"Saved: inputs {data_arr.shape}, targets {gt_arr.shape}")
    return data_arr, gt_arr


def save_test_dataset(output_dir, examples_per_condition=1000, seed=42,
                      sigma=0.07, M=256):
    """
    Save the test set for the SNR-sweep experiment (Figs. 5-6).

    Files written:
        {output_dir}/test_data.npz
        {output_dir}/test_gt.npz
        {output_dir}/test_meta.npz
        {output_dir}/test_features.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    conditions = make_test_conditions_snr_sweep()
    total = len(conditions) * examples_per_condition
    gen = validation_data_generator(
        conditions, examples_per_condition=examples_per_condition,
        seed=seed, sigma=sigma, M=M
    )

    data_list, gt_list, feat_list, meta_list = [], [], [], []
    for i, (data, gt, feat, meta) in enumerate(gen):
        data_list.append(data)
        gt_list.append(gt)
        feat_list.append(feat)
        meta_list.append(meta)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{total} test samples generated")

    data_arr = np.array(data_list)
    gt_arr   = np.array(gt_list)
    meta_arr = np.array(meta_list)
    np.savez_compressed(os.path.join(output_dir, 'test_data.npz'),  data=data_arr)
    np.savez_compressed(os.path.join(output_dir, 'test_gt.npz'),    data=gt_arr)
    np.savez_compressed(os.path.join(output_dir, 'test_meta.npz'),  data=meta_arr)
    np.save(os.path.join(output_dir, 'test_features.npy'),
            np.array(feat_list, dtype=object), allow_pickle=True)
    print(f"Saved: inputs {data_arr.shape}, targets {gt_arr.shape}")
    return data_arr, gt_arr


# =========================================================================
# TensorFlow dataset builders
# =========================================================================

def build_tf_training_dataset(batch_size=32, sigma=0.07, M=256):
    """
    Batched, prefetched tf.data.Dataset for training (paper Table I).
    Yields (input (64,64,2), gt (M,M,1)) batches.
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_generator(
        lambda: training_data_generator(sigma=sigma, M=M),
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(M, M, 1),   dtype=tf.float32),
        )
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_tf_validation_dataset(conditions, examples_per_condition=1,
                                 seed=42, batch_size=32, sigma=0.07, M=256):
    """
    Batched, prefetched tf.data.Dataset for validation / testing.
    Yields (data, gt, features, meta) batches.
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_generator(
        lambda: validation_data_generator(
            conditions, examples_per_condition=examples_per_condition,
            seed=seed, sigma=sigma, M=M
        ),
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(M, M, 1),   dtype=tf.float32),
            tf.TensorSpec(shape=(2, None),   dtype=tf.float32),
            tf.TensorSpec(shape=(4,),        dtype=tf.float32),
        )
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# =========================================================================
# Angle matching utility (used in evaluation)
# =========================================================================

def match_angle_pairs(A, B):
    """
    Optimal assignment between point sets A and B (Hungarian algorithm).
    Minimises total Euclidean distance.  Used to match predicted to true angles.

    A, B : array-like of shape (n, 2)
    Returns list of (a_point, b_point) pairs.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    return [(tuple(A[i]), tuple(B[j])) for i, j in zip(row, col)]


# =========================================================================
# CLI entry point
# =========================================================================

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description='DLDOA dataset generation — IEEE TVT 2026, Lloria et al.'
    )
    parser.add_argument(
        '--mode', default='demo',
        choices=['demo', 'save_train', 'save_val', 'save_test', 'save_all'],
    )
    parser.add_argument('--output_dir',      default='dataset')
    parser.add_argument('--n_train',         type=int, default=10000)
    parser.add_argument('--n_test_per_snr',  type=int, default=1000)
    args = parser.parse_args()

    if args.mode == 'demo':
        print("=== Training generator (5 samples) ===")
        gen = training_data_generator()
        t0 = time.time()
        for i in range(5):
            data, gt = next(gen)
            print(f"  [{i+1}] input={data.shape} range=[{data.min():.3f},{data.max():.3f}]"
                  f"  gt={gt.shape} max={gt.max():.4f}")
        print(f"  ({time.time()-t0:.2f}s)\n")

        print("=== Validation generator (3 samples) ===")
        conds = make_test_conditions_snr_sweep()
        gen_v = validation_data_generator(conds, examples_per_condition=1, seed=42)
        t0 = time.time()
        for i in range(3):
            data, gt, feat, meta = next(gen_v)
            L, SNR, P, nt = meta.astype(int)
            print(f"  [{i+1}] L={L} SNR={SNR}dB P={P} nt={nt}"
                  f"  input={data.shape} gt={gt.shape}")
        print(f"  ({time.time()-t0:.2f}s)\n")

        print(f"SNR-sweep test conditions : {len(make_test_conditions_snr_sweep())} scenarios")
        print(f"Path-sweep test conditions: {len(make_test_conditions_path_sweep())} scenarios")

    elif args.mode == 'save_train':
        save_training_dataset(args.output_dir, n_samples=args.n_train)

    elif args.mode == 'save_val':
        save_validation_dataset(args.output_dir)

    elif args.mode == 'save_test':
        save_test_dataset(args.output_dir, examples_per_condition=args.n_test_per_snr)

    elif args.mode == 'save_all':
        save_training_dataset(args.output_dir, n_samples=args.n_train)
        save_validation_dataset(args.output_dir)
        save_test_dataset(args.output_dir, examples_per_condition=args.n_test_per_snr)
        print(f"\nAll datasets saved to: {args.output_dir}")
