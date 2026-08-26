"""
==========================================================================
DLDOA Dataset Generation — Exact Reproduction
==========================================================================
Reproduces the synthetic dataset described in:

    "Deep-Learning-Based AoA and AoD Estimation in Analog Millimeter Wave
     MIMO Systems"
    D. Lloria, S. Roger, C. Botella-Mascarell, M. Cobos
    IEEE Transactions on Vehicular Technology, Vol. 75, No. 6, June 2026
    DOI: 10.1109/TVT.2025.3637908

Code derived from the authors' official repository:
    https://github.com/SandraRoger/DLDOA

Paper Section IV-B & Table I specify:
  - Training set   : 10 000 synthetic observation matrices per epoch
  - Validation set  : 1 000 fixed samples (seed=42)
  - L (paths)       : random in {1, ..., 9}  (training)
  - SNR             : random in [-15, 24] dB (training); [-10, 25] step 5 (test)
  - Codebook P=Q    : random in {16, 32}     (training)
  - Antennas nt=nr  : <= P  (16 if P=16; {16,32} if P=32)
  - Channel coeff   : alpha_l ~ CN(0, 1/L)
  - AoA, AoD        : uniform in [0, pi], min separation pi/6
  - Ground truth    : sum of L 2-D Gaussians, sigma=0.07, grid M=N=256
  - Loss            : MSE
  - Output size     : M x N x 1  (256 x 256 x 1)
  - Input size      : 64 x 64 x 2  (real + imag, after nearest-neighbor upsampling)
==========================================================================
"""

import numpy as np
import math
import random
import os
import itertools
import scipy.ndimage
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


# =====================================================================
# 1. CORE BUILDING BLOCKS  (from authors' tvt_data_generation_v3.py)
# =====================================================================

def generate_points(M, delta, max_attempts=10000, rng=None):
    """
    Generate M random 2-D points in [0, pi] x [0, pi] with minimum
    pairwise separation >= delta.

    This enforces the angular separation constraint described in the paper:
    AoA and AoD angles are drawn uniformly in [0, pi] with a minimum
    separation of pi/6 between any two paths.

    Parameters
    ----------
    M : int
        Number of points (= number of paths L).
    delta : float
        Minimum Euclidean distance between any pair of points (paper: pi/6).
    max_attempts : int
        Cap on random placement attempts.
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    list of (float, float)
        M points as (AoD, AoA) tuples in [0, pi]^2.
    """
    if rng is None:
        rng = np.random.default_rng()

    points = []
    attempts = 0

    while len(points) < M and attempts < max_attempts:
        x = rng.uniform(0, math.pi)
        y = rng.uniform(0, math.pi)
        new_point = (x, y)
        too_close = any(
            math.hypot(new_point[0] - p[0], new_point[1] - p[1]) < delta
            for p in points
        )
        if not too_close:
            points.append(new_point)
        attempts += 1

    if len(points) < M:
        raise ValueError(
            f"Could not place {M} points with delta={delta} "
            f"after {max_attempts} attempts."
        )

    pts = np.array(points)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < delta:
                raise ValueError(
                    f"Points {i} and {j} violate min delta: dist={d:.4f}"
                )
    return pts.tolist()


def wrapTo2Pi(x):
    """Wrap angle to [0, 2*pi)."""
    return np.mod(x, 2 * np.pi)


def ev(nt, angle):
    """
    Antenna array steering vector under half-wavelength separation.
    Implements Eqs. (2)-(3) of the paper:
        a(phi) = (1/sqrt(nt)) * [1, e^{-j*pi*cos(phi)}, ..., e^{-j*pi*(nt-1)*cos(phi)}]^T

    Parameters
    ----------
    nt : int
        Number of antenna elements.
    angle : float
        Angle in radians.

    Returns
    -------
    np.ndarray of shape (nt, 1)
        Complex steering vector.
    """
    k = np.arange(nt)
    vector = (1 / np.sqrt(nt)) * np.exp(-1j * np.pi * np.cos(angle) * k)
    return vector[:, np.newaxis]


def beamforming_vector_generation_P(P, nt, error_deg=None):
    """
    Create the P-column TX beamforming (codebook) matrix F of size (nt x P).
    Implements the DFT-based codebook from Eq.(16) of the TSDCE paper [16],
    used as the analog beamforming codebook described in Section II-B.

    Parameters
    ----------
    P : int
        Number of TX beamforming directions (codebook size).
    nt : int
        Number of transmit antennas.
    error_deg : float or None
        Maximum per-antenna phase error in degrees (Section IV-D).
        None = no hardware impairment.

    Returns
    -------
    np.ndarray of shape (nt, P)
        Complex TX beamforming matrix.
    """
    p = np.arange(P)
    cosp = (1 / np.pi) * np.angle(np.exp(1j * (2 * np.pi / P) * p))
    phi_p = np.arccos(cosp)

    if error_deg is not None:
        d_max = np.deg2rad(error_deg)
        phase_error = np.random.uniform(-d_max, d_max, size=nt)
    else:
        phase_error = np.zeros(nt)

    F = np.zeros((nt, P), dtype=complex)
    for idx_p in range(P):
        f_ideal = np.squeeze(ev(nt, phi_p[idx_p]), -1)
        F[:, idx_p] = f_ideal * np.exp(1j * phase_error)
    return F


def beamforming_vector_generation_Q(Q, nr, error_deg=None):
    """
    Create the Q-column RX combining matrix W of size (nr x Q).
    Mirror of beamforming_vector_generation_P for the receiver side.

    Parameters
    ----------
    Q : int
        Number of RX combining directions.
    nr : int
        Number of receive antennas.
    error_deg : float or None
        Maximum per-antenna phase error in degrees.

    Returns
    -------
    np.ndarray of shape (nr, Q)
        Complex RX combining matrix.
    """
    q = np.arange(Q)
    cosq = (1 / np.pi) * np.angle(np.exp(-1j * (2 * np.pi / Q) * q))
    phi_q = np.arccos(cosq)

    if error_deg is not None:
        d_max = np.deg2rad(error_deg)
        phase_error = np.random.uniform(-d_max, d_max, size=nr)
    else:
        phase_error = np.zeros(nr)

    W = np.zeros((nr, Q), dtype=complex)
    for idx_q in range(Q):
        w_ideal = np.squeeze(ev(nr, phi_q[idx_q]), -1)
        W[:, idx_q] = w_ideal * np.exp(1j * phase_error)
    return W


def generate_noise(var_alpha, SNR, Q, P, rng=None):
    """
    Generate (Q x P) complex AWGN noise matrix N.
    Variance scaled per SNR: var_noise = var_alpha * 10^(-SNR/10).
    Real and imaginary parts are i.i.d. N(0, var_noise/2).

    This implements the noise term in Eq.(4)-(5) of the paper:
        n ~ CN(0, sigma_n^2 * I),  SNR = rho / sigma_n^2

    Parameters
    ----------
    var_alpha : float
        Signal power scaling factor (rho = 1 in paper).
    SNR : float
        Signal-to-noise ratio in dB.
    Q : int
        Number of rows (RX combining directions).
    P : int
        Number of columns (TX beamforming directions).
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray of shape (Q, P)
        Complex Gaussian noise matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    var_noise = var_alpha * 10 ** (-SNR / 10)
    sigma = np.sqrt(var_noise / 2)
    Z = sigma * (rng.standard_normal((Q, P)) + 1j * rng.standard_normal((Q, P)))
    return Z


class _myarray(np.ndarray):
    """ndarray subclass with a Hermitian-transpose property."""
    @property
    def H(self):
        return self.conj().transpose()


def generate_channel_v2(nr, nt, angle_v, alpha_l):
    """
    Generate the (nr x nt) MIMO channel matrix H(theta).
    Implements Eq.(1) of the paper:
        H = sqrt(nt*nr) * sum_l  alpha_l * a_r(psi_l) * a_t^H(phi_l)

    Parameters
    ----------
    nr : int
        Number of receive antennas.
    nt : int
        Number of transmit antennas.
    angle_v : array-like of length 2*L
        First L entries are AoD (phi_l), next L are AoA (psi_l).
    alpha_l : array-like of length L
        Complex path gains.

    Returns
    -------
    np.ndarray of shape (nr, nt)
        Complex channel matrix.
    """
    L = len(alpha_l)
    Hl = np.zeros((nr, nt, L), dtype=complex)
    for l in range(L):
        qq = ev(nt, angle_v[l]).view(_myarray).H      # (1, nt)
        ww = ev(nr, angle_v[l + L])                    # (nr, 1)
        Hl[:, :, l] = alpha_l[l] * (ww * qq)
    Hl = np.sqrt(nt * nr) * Hl
    H = np.sum(Hl, axis=-1)
    return H


def gaus2d(dist_m, dist_n, sigma):
    """2-D Gaussian kernel value at (dist_m, dist_n)."""
    coeff = 1.0 / (2.0 * np.pi * sigma ** 2)
    exponent = -(dist_m ** 2 + dist_n ** 2) / (2.0 * sigma ** 2)
    return coeff * np.exp(exponent)


def get_real_imag(H):
    """Stack real and imaginary parts along axis=-1 → shape (..., 2)."""
    return np.dstack((np.real(H), np.imag(H)))


def permute_pairs(A, B):
    """
    Optimal assignment matching between point sets A and B by
    minimizing total Euclidean distance (Hungarian algorithm).
    Used to match predicted angles to ground-truth angles.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    dist_matrix = np.linalg.norm(
        A[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2
    )
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return [(tuple(A[i]), tuple(B[j])) for i, j in zip(row_ind, col_ind)]


# =====================================================================
# 2. GROUND TRUTH MAP GENERATION
# =====================================================================

def generate_gt(L, amps, f1, f2, num_points_rx=256, num_points_tx=256,
                sigma=0.07, return_axes=False, margin_factor=3.0):
    """
    Generate the M x N ground-truth image X as the sum of L 2-D Gaussians.
    Implements Eq.(11) of the paper:

        X_{m,n} = (1 / 2*pi*sigma1*sigma2) * sum_l
                   exp(-( (omega_m - tilde_omega_phi_l)^2 / (2*sigma1^2)
                        + (omega_n - tilde_omega_psi_l)^2 / (2*sigma2^2) ))

    where omega_m = m * 2*pi/M, omega_n = n * 2*pi/N,
    and tilde_omega are the wrapped-to-[0,2pi] spatial frequencies.
    The grid is extended by margin_factor * sigma beyond [0, 2*pi] to avoid
    edge clipping.

    Parameters
    ----------
    L : int
        Number of Gaussian components (= number of paths).
    amps : array-like of length L
        Amplitude of each Gaussian (paper uses all-ones).
    f1 : array-like of length L
        Horizontal spatial frequencies omega_phi = pi*cos(phi_l).
    f2 : array-like of length L
        Vertical spatial frequencies omega_psi = -pi*cos(psi_l).
    num_points_rx : int
        Number of pixels along the vertical (AoA) axis. Default 256.
    num_points_tx : int
        Number of pixels along the horizontal (AoD) axis. Default 256.
    sigma : float
        Std-dev of each 2-D Gaussian (paper: sigma1 = sigma2 = 0.07).
    return_axes : bool
        If True, also return the spatial-frequency axes.
    margin_factor : float
        Grid extension beyond [0, 2*pi] as multiple of sigma.

    Returns
    -------
    np.ndarray of shape (num_points_rx, num_points_tx)
        The ground-truth heatmap.
    """
    f1 = wrapTo2Pi(np.asarray(f1))
    f2 = wrapTo2Pi(np.asarray(f2))

    margin = margin_factor * sigma
    p = np.linspace(-margin, 2 * np.pi + margin, num_points_tx, endpoint=False)
    q = np.linspace(-margin, 2 * np.pi + margin, num_points_rx, endpoint=False)
    Wp, Wq = np.meshgrid(p, q)

    mod = []
    for l in range(L):
        dist_p = Wp - f1[l]
        dist_q = Wq - f2[l]
        gauss = amps[l] * gaus2d(dist_p, dist_q, sigma)
        mod.append(gauss)
    J = np.sum(mod, axis=0)

    if return_axes:
        return J, p, q
    return J


# =====================================================================
# 3. TRAINING DATA GENERATOR  (infinite, varies each epoch)
# =====================================================================

def training_data_generator(sigma=0.07, M=256, amps_type='ones',
                            normalize_gt=False):
    """
    Infinite generator yielding (input, ground_truth) pairs for training.

    This implements the dataset specification from Section IV-B of the paper:
      - L      : random in {1, 2, ..., 9}      (np.random.randint(1, 10))
      - SNR    : random in {-15, -14, ..., 24}  (np.random.randint(-15, 25))
      - P = Q  : random in {16, 32}
      - nt = nr: <= P  (16 if P=16; choice of {16,32} if P=32)
      - alpha_l: CN(0, 1/L), sorted by magnitude (strongest first)
      - AoA/AoD: uniform in [0, pi], min separation pi/6
      - rho    : 1  (transmit power)
      - Ground truth: L unit-amplitude 2-D Gaussians, sigma=0.07, M=256

    The observation matrix Y in C^{QxP} is decomposed into real and
    imaginary parts (2 channels), then upsampled via nearest-neighbor
    interpolation to 64x64x2 (beta=4 for P=16, beta=2 for P=32).

    Training data is NOT seeded — it varies every epoch, making the
    training set effectively infinite (paper Section IV-B, footnote 1).

    Parameters
    ----------
    sigma : float
        Gaussian width for ground truth (paper: 0.07).
    M : int
        Output spatial grid size (paper: 256).
    amps_type : str
        'ones' = unit amplitude (paper default), 'alpha_mag' = |alpha_l|.
    normalize_gt : bool
        If True, normalize GT to unit max.

    Yields
    ------
    data : np.ndarray of shape (64, 64, 2), dtype float32
        Real and imaginary parts of the upsampled observation matrix.
    gt : np.ndarray of shape (M, M, 1), dtype float32
        Ground-truth 2-D Gaussian heatmap.
    """
    while True:
        # --- Random configuration (paper Section IV-B) ---
        L = np.random.randint(1, 10)           # 1 to 9 paths
        SNR = np.random.randint(-15, 25)       # -15 to 24 dB
        P = random.choice([16, 32])            # codebook size
        Q = P

        if P == 64:
            nt = random.choice([16, 32, 64])
            nr = nt
        elif P == 32:
            nt = random.choice([16, 32])
            nr = nt
        else:
            nt = nr = 16

        # --- Beamforming codebooks (Eq. 16 of TSDCE [16]) ---
        F = beamforming_vector_generation_P(P, nt)
        W = beamforming_vector_generation_Q(Q, nr)

        # --- Complex path gains: alpha_l ~ CN(0, 1/L) ---
        alpha_l = (np.sqrt(1 / L)
                   * (np.random.randn(L) + 1j * np.random.randn(L))
                   / np.sqrt(2))
        s_idx = np.argsort(np.abs(alpha_l))
        alpha_l = alpha_l[s_idx[::-1]]         # strongest first

        # --- AoD (phi) and AoA (psi) angles ---
        sep = np.pi / 6                        # min angular separation
        points = generate_points(L, sep)
        phi_l = [pair[0] for pair in points]   # AoD
        psi_l = [pair[1] for pair in points]   # AoA
        angle_v = np.hstack([phi_l, psi_l])

        # --- Spatial frequencies (Eqs. 9-10) ---
        omega_phi_true = np.pi * np.cos(phi_l)
        omega_psi_true = -np.pi * np.cos(psi_l)

        # --- Channel + observation (Eqs. 1, 4-5) ---
        H = generate_channel_v2(nr, nt, angle_v, alpha_l)
        G = (W.view(_myarray).H @ H) @ F      # noiseless observation
        Z = generate_noise(1.0, SNR, P, Q)     # AWGN
        Y = G + Z                              # noisy observation

        Y = get_real_imag(Y)                   # (Q, P, 2)

        # --- Ground truth (Eq. 11) ---
        if amps_type == 'ones':
            amps = np.ones_like(alpha_l)
        elif amps_type == 'alpha_mag':
            amps = np.abs(alpha_l)
        else:
            raise ValueError(f"Unknown amps_type '{amps_type}'")

        gt = generate_gt(L, amps, omega_phi_true, omega_psi_true,
                         num_points_rx=M, num_points_tx=M, sigma=sigma)

        if normalize_gt:
            norm = np.max(gt)
            if norm > 0:
                gt = gt / norm

        # --- Nearest-neighbor upsampling to 64x64 (Section III-A) ---
        zoom_factor = 4 if P == 16 else (2 if P == 32 else 1)
        data_preal = scipy.ndimage.zoom(Y[:, :, 0], zoom_factor, order=0)
        data_pimag = scipy.ndimage.zoom(Y[:, :, 1], zoom_factor, order=0)
        data = np.dstack((data_preal, data_pimag))

        gt = np.expand_dims(gt, axis=-1)       # (M, M, 1)

        data = np.real(data)
        gt = np.real(gt)

        yield data.astype(np.float32), gt.astype(np.float32)


# =====================================================================
# 4. VALIDATION / TEST DATA GENERATOR  (fixed, reproducible)
# =====================================================================

def validation_data_generator(validation_conditions, examples_per_condition=1,
                              seed=42, normalize_gt=False, sigma=0.07,
                              M=256, amps_type='ones', error_deg=None):
    """
    Deterministic generator yielding (data, gt, features, meta) tuples.

    Used for:
      - Fixed validation set during training  (1000 samples, seed=42)
      - Evaluation / test set for reproducing paper figures

    Parameters
    ----------
    validation_conditions : list of (L, SNR, P, nt) tuples
        Each tuple defines a test scenario.
    examples_per_condition : int
        Samples to generate per condition.
    seed : int
        RNG seed for full reproducibility.
    normalize_gt : bool
        If True, normalize ground truth to unit max.
    sigma : float
        Gaussian width (paper: 0.07).
    M : int
        Output grid size (paper: 256).
    amps_type : str
        'ones' or 'alpha_mag'.
    error_deg : float or None
        Per-antenna phase error for hardware impairment study (Section IV-D).

    Yields
    ------
    data : np.ndarray (64, 64, 2) float32
        Input observation (real + imag).
    gt : np.ndarray (M, M, 1) float32
        Ground-truth heatmap.
    features : np.ndarray (2, L) float32
        True angles [psi_l; phi_l].
    meta : np.ndarray (4,) float32
        Condition vector [L, SNR, P, nt].
    """
    rng = np.random.default_rng(seed)

    for (L, SNR, P, ntnr) in validation_conditions:
        for _ in range(examples_per_condition):
            Q = P
            nt = nr = ntnr

            F = beamforming_vector_generation_P(P, nt, error_deg=error_deg)
            W = beamforming_vector_generation_Q(Q, nr, error_deg=error_deg)

            alpha_l = (np.sqrt(1 / L)
                       * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
                       / np.sqrt(2))
            s_idx = np.argsort(np.abs(alpha_l))
            alpha_l = alpha_l[s_idx[::-1]]

            sep = np.pi / 6
            points = generate_points(L, sep, rng=rng)
            phi_l = [pair[0] for pair in points]
            psi_l = [pair[1] for pair in points]
            angle_v = np.hstack([phi_l, psi_l])

            omega_phi_true = np.pi * np.cos(phi_l)
            omega_psi_true = -np.pi * np.cos(psi_l)

            H = generate_channel_v2(nr, nt, angle_v, alpha_l)
            G = (W.view(_myarray).H @ H) @ F
            Z = generate_noise(1.0, SNR, P, Q, rng=rng)
            Y = G + Z
            Y = get_real_imag(Y)

            if amps_type == 'ones':
                amps = np.ones_like(alpha_l)
            elif amps_type == 'alpha_mag':
                amps = np.abs(alpha_l)
            else:
                raise ValueError(f"Unknown amps_type '{amps_type}'")

            gt = generate_gt(L, amps, omega_phi_true, omega_psi_true,
                             num_points_rx=M, num_points_tx=M, sigma=sigma)

            if normalize_gt:
                norm = np.max(gt)
                if norm > 0:
                    gt = gt / norm

            zoom_factor = 4 if P == 16 else (2 if P == 32 else 1)
            data_preal = scipy.ndimage.zoom(Y[:, :, 0], zoom_factor, order=0)
            data_pimag = scipy.ndimage.zoom(Y[:, :, 1], zoom_factor, order=0)
            data = np.dstack((data_preal, data_pimag))

            gt = np.expand_dims(gt, axis=-1)
            data = np.real(data)
            gt = np.real(gt)

            features = np.stack([psi_l, phi_l])
            meta = np.array([L, SNR, P, ntnr], dtype=np.float32)

            yield (data.astype(np.float32),
                   gt.astype(np.float32),
                   features.astype(np.float32),
                   meta)


# =====================================================================
# 5. TENSORFLOW DATASET BUILDERS
# =====================================================================

def build_tf_training_dataset(batch_size=32, sigma=0.07, M=256,
                              amps_type='ones', normalize_gt=False):
    """
    Build a tf.data.Dataset for training.

    Paper Table I:
      - Batch size      : 32
      - Training size   : 10 000 per epoch
      - Epochs          : 500
      - Loss            : MSE

    Returns a batched, prefetched tf.data.Dataset yielding
    (input, ground_truth) pairs.
    """
    import tensorflow as tf

    def gen():
        return training_data_generator(
            sigma=sigma, M=M, amps_type=amps_type,
            normalize_gt=normalize_gt
        )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(M, M, 1), dtype=tf.float32),
        )
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_tf_validation_dataset(validation_conditions,
                                examples_per_condition=1, seed=42,
                                batch_size=32, sigma=0.07, M=256,
                                amps_type='ones', normalize_gt=False):
    """
    Build a tf.data.Dataset for validation / testing.

    For the paper's fixed validation set (1000 samples):
        validation_conditions = generate_validation_conditions()
        examples_per_condition = 1
        seed = 42

    For the paper's test evaluation (Figs. 5-8):
        validation_conditions = generate_test_conditions()
        examples_per_condition = 1000
    """
    import tensorflow as tf

    def gen():
        return validation_data_generator(
            validation_conditions=validation_conditions,
            examples_per_condition=examples_per_condition,
            seed=seed, normalize_gt=normalize_gt,
            sigma=sigma, M=M, amps_type=amps_type
        )

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(2, None), dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
        )
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# =====================================================================
# 6. CONDITION GENERATORS  (paper-exact parameter sets)
# =====================================================================

def generate_validation_conditions(n_samples=1000):
    """
    Generate 1000 random (L, SNR, P, nt) conditions for the fixed
    validation set.  Mirrors the training distribution but with a
    fixed seed so the set is deterministic.

    Paper Table I: Validation data size = 1000.
    """
    rng = np.random.default_rng(seed=123)
    conditions = []
    for _ in range(n_samples):
        L = int(rng.integers(1, 10))
        SNR = int(rng.integers(-15, 25))
        P = int(rng.choice([16, 32]))
        if P == 32:
            nt = int(rng.choice([16, 32]))
        else:
            nt = 16
        conditions.append((L, SNR, P, nt))
    return conditions


def generate_test_conditions_fig5_fig6():
    """
    Test conditions for reproducing Figs. 5-6 of the paper:
        L=3, nt=nr=16, Q=P=16, SNR in {-10, -5, 0, 5, 10, 15, 20, 25} dB
        1000 realizations per SNR point.
    """
    L_values = [3]
    SNR_values = list(range(-10, 30, 5))   # -10 to 25 step 5
    PQ_values = [16]
    ntnr_values = [16]
    return list(itertools.product(L_values, SNR_values, PQ_values, ntnr_values))


def generate_test_conditions_fig8():
    """
    Test conditions for reproducing Fig. 8 of the paper:
        L in {1,...,6}, nt=nr=16, Q=P=16, SNR in {0, 20} dB
        1000 realizations per (L, SNR) point.
    """
    L_values = list(range(1, 7))
    SNR_values = [0, 20]
    PQ_values = [16]
    ntnr_values = [16]
    return list(itertools.product(L_values, SNR_values, PQ_values, ntnr_values))


def generate_test_conditions_hardware(error_deg_values=None):
    """
    Test conditions for hardware impairment study (Table II, Section IV-D):
        L=3, nt=nr=16, Q=P=16, SNR in {0, 20} dB
        delta_max in {0, 1, 2, 5} degrees
    """
    if error_deg_values is None:
        error_deg_values = [0, 1, 2, 5]
    L_values = [3]
    SNR_values = [0, 20]
    PQ_values = [16]
    ntnr_values = [16]
    return list(itertools.product(L_values, SNR_values, PQ_values, ntnr_values))


# =====================================================================
# 7. NUMPY DATASET SAVER  (save to .npz for offline use)
# =====================================================================

def _to_ragged_object_array(arrays):
    """
    Build a 1-D object ndarray holding per-sample feature arrays of
    varying shape (features has shape (2, L) with L differing per
    sample). ``np.array(arrays, dtype=object)`` is unsafe here: NumPy
    first tries to infer a common regular shape across the whole list
    and raises "could not broadcast input array" as soon as two
    samples' L values disagree (e.g. whenever L is drawn randomly, as
    in the validation set). Filling a pre-allocated object array
    element-by-element sidesteps that shape inference entirely.
    """
    out = np.empty(len(arrays), dtype=object)
    for i, arr in enumerate(arrays):
        out[i] = arr
    return out


def save_training_dataset(output_dir, n_samples=10000, sigma=0.07,
                          M=256, amps_type='ones', normalize_gt=False):
    """
    Generate and save n_samples training examples to disk as .npz files.

    Paper Table I specifies 10 000 training samples per epoch.
    Since training uses an infinite generator that varies each epoch,
    saving a single epoch's worth of data is a snapshot — for full
    reproduction, use the generator directly with TensorFlow.

    Saves:
        {output_dir}/train_data.npz    — inputs  (n, 64, 64, 2)
        {output_dir}/train_gt.npz      — targets (n, M, M, 1)
    """
    os.makedirs(output_dir, exist_ok=True)
    gen = training_data_generator(sigma=sigma, M=M, amps_type=amps_type,
                                  normalize_gt=normalize_gt)

    all_data = []
    all_gt = []
    for i, (data, gt) in enumerate(gen):
        if i >= n_samples:
            break
        all_data.append(data)
        all_gt.append(gt)
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} training samples")

    all_data = np.array(all_data)
    all_gt = np.array(all_gt)

    np.savez_compressed(os.path.join(output_dir, 'train_data.npz'), data=all_data)
    np.savez_compressed(os.path.join(output_dir, 'train_gt.npz'), data=all_gt)
    print(f"Saved training data: {all_data.shape}, GT: {all_gt.shape}")
    return all_data, all_gt


def save_validation_dataset(output_dir, seed=42, sigma=0.07, M=256,
                            amps_type='ones', normalize_gt=False):
    """
    Generate and save the fixed 1000-sample validation set.

    Uses generate_validation_conditions() with seed=42 for the data
    generator (matching the author's code) and seed=123 for condition
    selection.

    Saves:
        {output_dir}/val_data.npz      — inputs    (1000, 64, 64, 2)
        {output_dir}/val_gt.npz        — targets   (1000, M, M, 1)
        {output_dir}/val_features.npz  — angles    (1000, 2, L)  [ragged, saved as list]
        {output_dir}/val_meta.npz      — conditions (1000, 4)
    """
    os.makedirs(output_dir, exist_ok=True)
    conditions = generate_validation_conditions(n_samples=1000)

    gen = validation_data_generator(
        validation_conditions=conditions,
        examples_per_condition=1, seed=seed,
        normalize_gt=normalize_gt, sigma=sigma, M=M, amps_type=amps_type
    )

    all_data = []
    all_gt = []
    all_features = []
    all_meta = []

    for i, (data, gt, features, meta) in enumerate(gen):
        all_data.append(data)
        all_gt.append(gt)
        all_features.append(features)
        all_meta.append(meta)
        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/1000 validation samples")

    all_data = np.array(all_data)
    all_gt = np.array(all_gt)
    all_meta = np.array(all_meta)

    np.savez_compressed(os.path.join(output_dir, 'val_data.npz'), data=all_data)
    np.savez_compressed(os.path.join(output_dir, 'val_gt.npz'), data=all_gt)
    np.savez_compressed(os.path.join(output_dir, 'val_meta.npz'), data=all_meta)
    np.save(os.path.join(output_dir, 'val_features.npy'), _to_ragged_object_array(all_features), allow_pickle=True)
    print(f"Saved validation data: {all_data.shape}, GT: {all_gt.shape}")
    return all_data, all_gt


def save_test_dataset(output_dir, examples_per_condition=1000, seed=42,
                      sigma=0.07, M=256, amps_type='ones'):
    """
    Generate and save the test dataset for reproducing Figs. 5-6.

    Conditions: L=3, nt=nr=16, Q=P=16, SNR in {-10,-5,0,5,10,15,20,25}
    1000 samples per SNR value = 8000 total samples.

    Saves:
        {output_dir}/test_data.npz
        {output_dir}/test_gt.npz
        {output_dir}/test_meta.npz
        {output_dir}/test_features.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    conditions = generate_test_conditions_fig5_fig6()

    gen = validation_data_generator(
        validation_conditions=conditions,
        examples_per_condition=examples_per_condition,
        seed=seed, normalize_gt=False, sigma=sigma, M=M, amps_type=amps_type
    )

    total = len(conditions) * examples_per_condition
    all_data = []
    all_gt = []
    all_features = []
    all_meta = []

    for i, (data, gt, features, meta) in enumerate(gen):
        all_data.append(data)
        all_gt.append(gt)
        all_features.append(features)
        all_meta.append(meta)
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{total} test samples")

    all_data = np.array(all_data)
    all_gt = np.array(all_gt)
    all_meta = np.array(all_meta)

    np.savez_compressed(os.path.join(output_dir, 'test_data.npz'), data=all_data)
    np.savez_compressed(os.path.join(output_dir, 'test_gt.npz'), data=all_gt)
    np.savez_compressed(os.path.join(output_dir, 'test_meta.npz'), data=all_meta)
    np.save(os.path.join(output_dir, 'test_features.npy'), _to_ragged_object_array(all_features), allow_pickle=True)
    print(f"Saved test data: {all_data.shape}, GT: {all_gt.shape}")
    return all_data, all_gt


# =====================================================================
# 8. TRAINING SCRIPT  (ResNet and UNet, paper Table I)
# =====================================================================

def train_resnet(output_model_path='models/resnet_trained.h5',
                 epochs=500, batch_size=32, steps_per_epoch=312,
                 val_steps=31, sigma=0.07, M=256, learning_rate=0.003):
    """
    Train the ResNet model exactly as described in Table I of the paper:
        - Optimizer     : RMSprop
        - Learning rate : 0.003
        - Batch size    : 32
        - Epochs        : 500
        - Training size : 10 000  (steps_per_epoch = 10000//32 = 312)
        - Val size      : 1 000   (val_steps = 1000//32 = 31)
        - Loss          : MSE
        - Input shape   : (64, 64, 2)
        - Output shape  : (256, 256, 1)
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import RMSprop

    # -- Build training dataset --
    train_ds = build_tf_training_dataset(batch_size=batch_size, sigma=sigma, M=M)

    # -- Build validation dataset --
    val_conditions = generate_validation_conditions(n_samples=1000)
    val_ds = build_tf_validation_dataset(
        validation_conditions=val_conditions,
        examples_per_condition=1, seed=42,
        batch_size=batch_size, sigma=sigma, M=M
    )
    val_ds_train = val_ds.map(lambda d, g, f, m: (d, g))

    # -- Build model --
    # Inline ResNet definition (from authors' tvt_models.py)
    def res_conv(x, filters=12):
        x_skip = x
        x = tf.keras.layers.Conv2D(filters, (5, 5), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, (5, 5), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    x_in = tf.keras.Input(shape=(64, 64, 2))
    x = tf.keras.layers.Conv2DTranspose(12, (5, 5), strides=(2, 2), padding='same')(x_in)
    for _ in range(64):
        x = res_conv(x)
    x = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same')(x)
    model = tf.keras.Model(inputs=x_in, outputs=x)

    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='mse')
    model.summary()

    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)

    model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds_train,
        validation_steps=val_steps,
    )
    model.save_weights(output_model_path)
    print(f"ResNet model saved to {output_model_path}")
    return model


def train_unet(output_model_path='models/unet_trained.h5',
               epochs=500, batch_size=32, steps_per_epoch=312,
               val_steps=31, sigma=0.07, M=256, learning_rate=0.001):
    """
    Train the U-Net model exactly as described in Table I of the paper:
        - Optimizer     : Adam
        - Learning rate : 0.001
        - Batch size    : 32
        - Epochs        : 500
        - Training size : 10 000  (steps_per_epoch = 10000//32 = 312)
        - Val size      : 1 000   (val_steps = 1000//32 = 31)
        - Loss          : MSE
        - Input shape   : (64, 64, 2)
        - Output shape  : (256, 256, 1)
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam

    # -- Build training dataset --
    train_ds = build_tf_training_dataset(batch_size=batch_size, sigma=sigma, M=M)

    # -- Build validation dataset --
    val_conditions = generate_validation_conditions(n_samples=1000)
    val_ds = build_tf_validation_dataset(
        validation_conditions=val_conditions,
        examples_per_condition=1, seed=42,
        batch_size=batch_size, sigma=sigma, M=M
    )
    val_ds_train = val_ds.map(lambda d, g, f, m: (d, g))

    # -- Build UNet model (from authors' tvt_models.py) --
    def Conv_Block(inputs, model_width, kernel, multiplier):
        x = tf.keras.layers.Conv2D(model_width * multiplier, kernel, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def trans_conv2D(inputs, model_width, multiplier):
        x = tf.keras.layers.Conv2DTranspose(model_width * multiplier, (2, 2), strides=(2, 2), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    length, width, num_channel = 64, 64, 2
    model_depth = 5
    model_width = 32
    kernel_size = 3
    feature_number = 1024

    inputs = tf.keras.Input((length, width, num_channel))
    pool = inputs
    convs = {}

    for i in range(1, model_depth + 1):
        conv = Conv_Block(pool, model_width, kernel_size, 2 ** (i - 1))
        conv = Conv_Block(conv, model_width, kernel_size, 2 ** (i - 1))
        pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
        convs[f"conv{i}"] = conv

    # Autoencoder bottleneck
    shape = pool.shape
    latent = tf.keras.layers.Flatten()(pool)
    latent = tf.keras.layers.Dense(feature_number, name='features')(latent)
    latent = tf.keras.layers.Dense(model_width * shape[1] * shape[2])(latent)
    pool = tf.keras.layers.Reshape((shape[1], shape[2], model_width))(latent)

    conv = Conv_Block(pool, model_width, kernel_size, 2 ** model_depth)
    conv = Conv_Block(conv, model_width, kernel_size, 2 ** model_depth)

    deconv = conv
    convs_list = list(convs.values())
    for j in range(model_depth):
        skip = convs_list[model_depth - j - 1]
        deconv = trans_conv2D(deconv, model_width, 2 ** (model_depth - j - 1))
        deconv = tf.keras.layers.concatenate([deconv, skip], axis=-1)
        deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** (model_depth - j - 1))
        deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** (model_depth - j - 1))

    deconv = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(deconv)
    h = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(inputs)
    deconv = tf.keras.layers.concatenate([deconv, h], axis=-1)
    deconv = Conv_Block(deconv, model_width, kernel_size, 16)

    if M == 256:
        outputs = tf.keras.layers.Conv2DTranspose(1, (2, 2), activation='linear', strides=(2, 2), padding='same')(deconv)
    elif M == 512:
        outputs = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(4, 4), padding='same')(deconv)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.summary()

    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)

    model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds_train,
        validation_steps=val_steps,
    )
    model.save_weights(output_model_path)
    print(f"UNet model saved to {output_model_path}")
    return model


# =====================================================================
# 9. MAIN — DEMONSTRATION & DATASET SAVING
# =====================================================================

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description='DLDOA Dataset Generation — exact reproduction of the '
                    'IEEE TVT 2025 paper by Lloria et al.'
    )
    parser.add_argument(
        '--mode', type=str, default='demo',
        choices=['demo', 'save_train', 'save_val', 'save_test', 'save_all',
                 'train_resnet', 'train_unet'],
        help='Operation mode.'
    )
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help='Output directory for saved datasets.')
    parser.add_argument('--n_train', type=int, default=10000,
                        help='Number of training samples to save.')
    parser.add_argument('--n_test_per_snr', type=int, default=1000,
                        help='Test samples per SNR condition.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs.')
    parser.add_argument('--model_path', type=str, default='models/',
                        help='Directory for saved model weights.')

    args = parser.parse_args()

    # ---- Demo mode: generate a few samples and print shapes ----
    if args.mode == 'demo':
        print("=" * 70)
        print("DLDOA Dataset Generation — Demo")
        print("=" * 70)
        print()

        print("--- Training Data Generator ---")
        print("Paper specs: infinite generator, varies each epoch")
        print("  L:     random in {1,...,9}")
        print("  SNR:   random in [-15, 24] dB")
        print("  P=Q:   random in {16, 32}")
        print("  sigma: 0.07")
        print("  M=N:   256")
        print()

        gen = training_data_generator()
        t0 = time.time()
        for i in range(5):
            data, gt = next(gen)
            print(f"  Sample {i+1}: input={data.shape}, "
                  f"gt={gt.shape}, "
                  f"input range=[{data.min():.4f}, {data.max():.4f}], "
                  f"gt max={gt.max():.4f}")
        dt = time.time() - t0
        print(f"  (5 samples in {dt:.2f}s)")
        print()

        print("--- Validation Data Generator ---")
        print("Paper specs: 1000 fixed samples, seed=42")
        conditions = generate_test_conditions_fig5_fig6()
        gen_val = validation_data_generator(
            validation_conditions=conditions,
            examples_per_condition=2, seed=42,
            sigma=0.07, M=256
        )
        t0 = time.time()
        for i in range(3):
            data, gt, features, meta = next(gen_val)
            L, SNR, P, nt = meta
            print(f"  Sample {i+1}: input={data.shape}, gt={gt.shape}, "
                  f"L={int(L)}, SNR={int(SNR)}dB, P={int(P)}, nt={int(nt)}, "
                  f"angles(psi,phi)={features.shape}")
        dt = time.time() - t0
        print(f"  (3 samples in {dt:.2f}s)")
        print()

        print("--- Test Conditions (Figs. 5-6) ---")
        conds = generate_test_conditions_fig5_fig6()
        print(f"  {len(conds)} conditions: {conds}")
        print()

        print("--- Test Conditions (Fig. 8) ---")
        conds8 = generate_test_conditions_fig8()
        print(f"  {len(conds8)} conditions: {conds8}")
        print()

        print("Dataset generation verified. Use --mode save_all to save to disk.")

    # ---- Save training dataset ----
    elif args.mode == 'save_train':
        print(f"Generating {args.n_train} training samples...")
        save_training_dataset(args.output_dir, n_samples=args.n_train)

    # ---- Save validation dataset ----
    elif args.mode == 'save_val':
        print("Generating 1000 validation samples (seed=42)...")
        save_validation_dataset(args.output_dir)

    # ---- Save test dataset ----
    elif args.mode == 'save_test':
        print(f"Generating test dataset ({args.n_test_per_snr} per SNR)...")
        save_test_dataset(args.output_dir,
                          examples_per_condition=args.n_test_per_snr)

    # ---- Save all ----
    elif args.mode == 'save_all':
        print("=" * 70)
        print("Generating ALL datasets (train + val + test)...")
        print("=" * 70)
        save_training_dataset(args.output_dir, n_samples=args.n_train)
        save_validation_dataset(args.output_dir)
        save_test_dataset(args.output_dir,
                          examples_per_condition=args.n_test_per_snr)
        print("\nAll datasets saved to:", args.output_dir)

    # ---- Train ResNet ----
    elif args.mode == 'train_resnet':
        train_resnet(
            output_model_path=os.path.join(args.model_path, 'resnet_trained.h5'),
            epochs=args.epochs
        )

    # ---- Train UNet ----
    elif args.mode == 'train_unet':
        train_unet(
            output_model_path=os.path.join(args.model_path, 'unet_trained.h5'),
            epochs=args.epochs
        )
