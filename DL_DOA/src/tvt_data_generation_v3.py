import numpy as np
import math
import random
import scipy.ndimage
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def generate_points(M, delta, max_attempts=10000, rng=None):
    """
    Generate M random 2D points within [0, pi] x [0, pi] such that
    no two points are closer than 'delta'. More efficient with capped attempts.
    
    Parameters:
        M (int): Number of points.
        delta (float): Minimum separation between points.
        max_attempts (int): Maximum number of attempts to place points.
        rng (np.random.Generator, optional): Random generator for reproducibility.
    
    Returns:
        list of (x, y) tuples.
    """    
    
    if rng is None:
        rng = np.random.default_rng()

    points = []
    attempts = 0

    while len(points) < M and attempts < max_attempts:
        x = rng.uniform(0, math.pi)
        y = rng.uniform(0, math.pi)
        new_point = (x, y)

        # Check if the new point is far enough from existing points
        too_close = any(math.hypot(new_point[0]-p[0], new_point[1]-p[1]) < delta for p in points)

        if not too_close:
            points.append(new_point)

        attempts += 1

    if len(points) < M:
        raise ValueError(f"Could not generate {M} points with delta={delta} after {max_attempts} attempts.")

    # FINAL global distance check
    points = np.array(points)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < delta:
                raise ValueError(f"Generated points violate minimum delta! Distance between point {i} and {j} is {dist:.4f}")

    return points.tolist()


def wrapTo2Pi(x):
    return np.mod(x, 2*np.pi)


def ev(nt, angle):
    '''Antenna array response under a half-wavelength antenna separation assumption,'''
    k = np.arange(nt)
    vector = (1/np.sqrt(nt)) * np.exp(-1j * np.pi * np.cos(angle) * k)
    vector = np.expand_dims(vector, axis=-1)
    return vector


def beamforming_vector_generation_P(P, nt, error_deg=None):
    ''' Creation of the TX Beamforming matrix with per-antenna static phase errors '''
    
    # This implements Eq.(16) of TSDCE paper
    p = np.arange(P)
    cosp = (1/np.pi) * np.angle(np.exp(1j * (2*np.pi/P) * p))
    phi_p = np.arccos(cosp)

    # Generate phase error once for all beams (static hardware impairment)
    if error_deg is not None:
        d_max = np.deg2rad(error_deg)  # maximum phase error in radians
        phase_error = np.random.uniform(-d_max, d_max, size=nt)
    else:
        phase_error = np.zeros(nt)

    # Create the TX Beamforming matrix (see Eq.(5) of TSDCE)
    F = np.zeros((nt, P), dtype=complex)            
    for idx_p in range(P):
        f_ideal = np.squeeze(ev(nt, phi_p[idx_p]), -1)  # ideal steering vector
        # Apply the same phase error to every beam
        f_perturbed = f_ideal * np.exp(1j * phase_error)
        F[:, idx_p] = f_perturbed

    return F


def beamforming_vector_generation_Q(Q, nr, error_deg=None):
    ''' Creation of the RX Beamforming matrix with per-antenna static phase errors '''
    
    # Implements Eq.(16) of TSDCE paper
    q = np.arange(Q)
    cosq = (1/np.pi) * np.angle(np.exp(-1j * (2*np.pi/Q) * q))
    phi_q = np.arccos(cosq)

    # Generate phase errors once (static impairment across all beams)
    if error_deg is not None:
        d_max = np.deg2rad(error_deg)  # maximum phase error in radians
        phase_error = np.random.uniform(-d_max, d_max, size=nr)  # one error per antenna
    else:
        phase_error = np.zeros(nr)

    # Create the RX Beamforming matrix (see Eq.(5) of TSDCE)
    W = np.zeros((nr, Q), dtype=complex)
    for idx_q in range(Q):
        w_ideal = np.squeeze(ev(nr, phi_q[idx_q]), -1)  # ideal steering vector
        # Apply the same static phase error to every beam
        w_perturbed = w_ideal * np.exp(1j * phase_error)
        W[:, idx_q] = w_perturbed

    return W


def generate_noise(var_alpha, SNR, Q, P, rng=None):
    '''
    Generates (Q x P) complex Gaussian noise matrix with variance scaled 
    according to var_alpha and SNR (in dB). Real and imaginary parts are independent 
    with variance var_alpha * 10^(-SNR/10) / 2 each.

    Parameters:
        var_alpha (float): Scaling factor.
        SNR (float): Signal-to-noise ratio in dB.
        Q (int): Rows (Receive dimension).
        P (int): Columns (Transmit dimension).
        rng (np.random.Generator, optional): Random generator for reproducibility.

    Returns:
        np.ndarray: Complex Gaussian noise matrix (Q, P).
    '''
    if rng is None:
        rng = np.random.default_rng()

    var_noise = var_alpha * 10**(-SNR/10)
    sigma = np.sqrt(var_noise / 2)
    Z = sigma * (rng.standard_normal((Q, P)) + 1j * rng.standard_normal((Q, P)))
    return Z


class myarray(np.ndarray):
    '''Implements Hermitian Transpose'''
    @property
    def H(self):
        return self.conj().transpose()

def generate_channel_v2(nr, nt, angle_v, alpha_l):
    '''
    Generate a MIMO channel matrix with multiple paths, 
    using steering vectors and complex path gains.
    See TSDCE Eq.(1)

    Parameters:
        nr (int): Number of receiver antennas.
        nt (int): Number of transmitter antennas.
        angle_v (array): Angles of departure and arrival (size 2*L).
        alpha_l (array): Complex path gains (size L).

    Returns:
        numpy.ndarray: The (nr x nt) channel matrix.
    '''
    L = len(alpha_l)
    
    Hl = np.zeros((nr, nt, L), dtype=complex)
    
    for l in range(L):
        qq = ev(nt, angle_v[l]).view(myarray).H     # Transmitter steering vector
        ww = ev(nr, angle_v[l+L])                   # Receiver steering vector
        Hl[:, :, l] = alpha_l[l] * (ww * qq)        # Outer product

    Hl = np.sqrt(nt * nr) * Hl                     # Scale
    H = np.sum(Hl, axis=-1)                        # Sum across paths
    return H

def gaus2d(dist_m, dist_n, sigma):
    '''
    Compute the value of a 2D Gaussian function at (dist_m, dist_n) with standard deviation sigma.

    Parameters:
        dist_m (float or array): Distance along the first dimension (x-axis).
        dist_n (float or array): Distance along the second dimension (y-axis).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        float or array: Value(s) of the 2D Gaussian at the given distances.
    '''
    coeff = 1.0 / (2.0 * np.pi * sigma**2)
    exponent = -(dist_m**2 + dist_n**2) / (2.0 * sigma**2)
    return coeff * np.exp(exponent)

def get_real_imag(H):
    '''
    Separate the real and imaginary parts of a complex matrix H
    and stack them along the last dimension.

    Parameters:
        H (numpy.ndarray): Complex input matrix.

    Returns:
        numpy.ndarray: A real-valued array where the last dimension 
                       contains [real, imaginary] parts.
    '''
    mode_real = np.real(H)
    mode_imag = np.imag(H)
    return np.dstack((mode_real, mode_imag))

def euclidean_distance(x, y):
    return distance.euclidean(x, y)


def permute_pairs(A, B):
    '''
    Matches each point in list A to a point in list B by minimizing
    the total Euclidean distance between the pairs.

    Parameters:
        A (list or array-like): List of points (e.g., 2D or 3D vectors).
        B (list or array-like): List of points of the same size as A.

    Returns:
        list of tuples: Matched pairs (A[i], B[j]) minimizing total distance.
    '''
    A = np.asarray(A)
    B = np.asarray(B)

    n = len(A)
    dist_matrix = np.linalg.norm(A[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return [(tuple(A[i]), tuple(B[j])) for i, j in zip(row_ind, col_ind)]


def imshow_with_centered_extent(data, x_axis, y_axis, ax=None, **imshow_kwargs):
    """
    Displays a 2D array using imshow with extent corrected to align axes with pixel centers.

    Parameters:
    - data: 2D numpy array to display (e.g., magnitude or heatmap).
    - x_axis: 1D array of x-coordinates (e.g., spatial or frequency domain).
    - y_axis: 1D array of y-coordinates.
    - ax: Optional matplotlib Axes object to plot on.
    - imshow_kwargs: Any additional arguments to pass to plt.imshow (e.g., cmap, aspect).

    Returns:
    - The Axes object containing the image.
    """
    # Compute pixel spacing (assumes uniform spacing)
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]

    # Adjust extent to align with pixel centers
    extent = [x_axis.min() - dx / 2, x_axis.max() + dx / 2,
              y_axis.min() - dy / 2, y_axis.max() + dy / 2]

    # Create Axes if not provided
    if ax is None:
        ax = plt.gca()

    # Plot with corrected extent
    im = ax.imshow(data, extent=extent, origin='lower', **imshow_kwargs)
    return ax, im

def generate_gt(L, amps, f1, f2, num_points_rx=256, num_points_tx=256, sigma=0.1, return_axes=False, margin_factor=3.0):
    """
    Generate a 2D ground truth map by summing L Gaussian components over an extended spatial grid.

    The grid is extended beyond [0, 2π] by a margin proportional to `sigma`, ensuring that
    Gaussians centered near the edges are fully visible and not clipped.

    Parameters:
        L (int): 
            Number of Gaussian components (paths).
        amps (array-like): 
            Real amplitudes for each Gaussian (length L).
        f1 (array-like): 
            Horizontal (phi) center frequencies (radians in range [0, 2pi]) for each Gaussian (length L).
        f2 (array-like): 
            Vertical (psi) center frequencies (radians in range [0, 2pi]) for each Gaussian (length L).
        num_points_rx (int, optional): 
            Number of points (pixels) along the receive axis (rows). Default is 256.
        num_points_tx (int, optional): 
            Number of points (pixels) along the transmit axis (columns). Default is 256.
        sigma (float, optional): 
            Standard deviation of each Gaussian in radians. Default is 0.025.
        return_axes (bool, optional): 
            If True, also return the spatial frequency grids (p, q). Default is False.
        margin_factor (float, optional): 
            Margin size as a multiple of sigma to extend the grid beyond [0, 2π]. 
            Default is 3.0 (i.e., extend by 3 × sigma).

    Returns:
        numpy.ndarray:
            2D array of shape (num_points_rx, num_points_tx) representing the ground truth map.
        tuple (optional):
            (p, q) axes arrays of spatial frequencies, if `return_axes=True` (p=horizontal, q=vertical).
    """
    # Ensure inputs are arrays
    f1 = np.asarray(f1)  # Horizontal (w_phi)
    f2 = np.asarray(f2)  # Vertical (w_psi)

    # Wrap frequencies
    f1 = wrapTo2Pi(f1)
    f2 = wrapTo2Pi(f2)
    
    # Define margin
    margin = margin_factor * sigma  # in radians

    # Create extended axes
    p = np.linspace(-margin, 2 * np.pi + margin, num_points_tx, endpoint=False)  # horizontal axis (columns)
    q = np.linspace(-margin, 2 * np.pi + margin, num_points_rx, endpoint=False)  # vertical axis (rows)

    Wp, Wq = np.meshgrid(p, q)  # X and Y grids (horizontal first!)

    # Sum Gaussians
    mod = []
    for l in range(L):
        mu = f1[l]  # horizontal center
        nu = f2[l]  # vertical center
        dist_p = Wp - mu  # horizontal distance (columns)
        dist_q = Wq - nu  # vertical distance (rows)
        gauss = amps[l] * gaus2d(dist_p, dist_q, sigma)
        mod.append(gauss)

    J = np.sum(mod, axis=0)

    if return_axes:
        return J, p, q
    else:
        return J

def data_generation(Training=True, sigma=0.1, M = 256, amps_type='ones', normalize_gt=False, condition=None):
    """
    Generator for synthetic MIMO dataset.

    Parameters:
        Training (bool): 
            If True, generate random data; else use fixed condition.
        sigma (float, optional): 
            Standard deviation of the Gaussians in the ground truth map.
        M (int):
            Target size (MxM).            
        amps_type (str, optional): 
            Type of amplitudes for ground truth:
            - 'ones' : all Gaussians have unit amplitude.
            - 'alpha_mag' : amplitude proportional to |α| of channel.
        normalize_gt (bool, optional): 
            If True, normalize ground truth map to unit amplitude.
        condition (tuple, optional): 
            Fixed condition (L, SNR, QP, nt=nr) for testing mode.      

    Yields:
        - Training mode: (data, ground_truth)
        - Testing mode: (data, ground_truth, features)
    """
    while True:
        if Training:
            # Random settings for training
            L = np.random.randint(1, 10)
            SNR = np.random.randint(-15, 25)
            P = random.choice([16, 32])
            Q = P

            # The number of receive/transmit antennas is defined randomly to be
            # less or equal to the codebook size
            if P == 64:
                nt = random.choice([16, 32, 64])
                nr = nt
            elif P == 32:
                nt = random.choice([16, 32])
                nr = nt
            else:
                nt = nr = 16

        else:
            if condition is None:
                raise ValueError("You must provide a condition (L, SNR, QP, ntnr) when Training=False.")

            # Fixed settings for testing
            L, SNR, QP, ntnr = condition
            P = Q = QP
            nt = nr = ntnr

        # --- (rest of the code unchanged) ---

        # Generate beamforming matrices
        F = beamforming_vector_generation_P(P, nt)
        W = beamforming_vector_generation_Q(Q, nr)

        # Generate path coefficients (alpha_l)
        alpha_l = np.sqrt(1 / L) * (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)
        s_idx = np.argsort(np.abs(alpha_l))
        alpha_l = alpha_l[s_idx[::-1]]

        # Generate random AoD and AoA angles
        sep = np.pi / 6
        points = generate_points(L, sep)
        phi_l = [pair[0] for pair in points]
        psi_l = [pair[1] for pair in points]
        angle_v = np.hstack([phi_l, psi_l])

        # Frequency components
        omega_phi_true = np.pi * np.cos(phi_l)
        omega_psi_true = -np.pi * np.cos(psi_l)

        # Generate channel and observations
        H = generate_channel_v2(nr, nt, angle_v, alpha_l)
        G = (W.view(myarray).H @ H) @ F
        Z = generate_noise(1.0, SNR, P, Q)

        Y = G + Z
        Y = get_real_imag(Y)

        # Generate ground truth map
        # Define amplitudes
        if amps_type == 'ones':
            amps = np.ones_like(alpha_l)
        elif amps_type == 'alpha_mag':
            amps = np.abs(alpha_l)
        else:
            raise ValueError(f"Unknown amps_type {amps_type}")

        # Generate ground truth
        gt = generate_gt(L, amps, omega_phi_true, omega_psi_true, num_points_rx=M, num_points_tx=M, sigma=sigma)

        if normalize_gt:
            #norm = np.sqrt(np.sum(gt**2))
            norm = np.max(gt)
            if norm > 0:
                gt = gt / norm

        # Preprocess observation (resize depending on P)
        zoom_factor = 4 if P == 16 else (2 if P == 32 else 1)
        data_preal = scipy.ndimage.zoom(Y[:, :, 0], zoom_factor, order=0)
        data_pimag = scipy.ndimage.zoom(Y[:, :, 1], zoom_factor, order=0)
        data = np.dstack((data_preal, data_pimag))

        # Expand gt
        gt = np.expand_dims(gt, axis=-1)

        features = np.stack([psi_l, phi_l])  # [AoA, AoD]
        
        data = np.real(data)
        gt = np.real(gt)

        if Training:
            yield data.astype(np.float32), gt.astype(np.float32)
        else:
            yield data.astype(np.float32), gt.astype(np.float32), features.astype(np.float32)

def validation_data_generator(validation_conditions, examples_per_condition=1, seed=42, normalize_gt=False, 
                              sigma=0.1, M = 256, amps_type='ones', error_deg=None):
    '''
    Generator that yields fixed validation samples along with generation conditions.

    Args:
        validation_conditions (list): 
            List of (L, SNR, Q=P, nt=nr) tuples.
        examples_per_condition (int): 
            Number of examples to generate per condition.
        seed (int): 
            Random seed for reproducibility.
        normalize_gt (bool): 
            If True, normalize ground truth to unit amplitude.
        sigma (float): 
            Standard deviation of Gaussians for ground truth.
        M (int):
            Target size (MxM).
        amps_type (str): 
            Amplitude generation strategy: 'ones' or 'alpha_mag'.
        error_deg (float):
            Phase angle error (in degrees) at each antenna element.

    Yields:
        (data, ground_truth, features, meta)
        where meta = (L, SNR, Q=P, nt=nr)
    '''
    rng = np.random.default_rng(seed)

    for (L, SNR, P, ntnr) in validation_conditions:
        for _ in range(examples_per_condition):
            Q = P
            nt = nr = ntnr

            # Generate beamforming matrices
            F = beamforming_vector_generation_P(P, nt, error_deg=error_deg)
            W = beamforming_vector_generation_Q(Q, nr, error_deg=error_deg)

            # Generate path coefficients
            alpha_l = np.sqrt(1 / L) * (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
            s_idx = np.argsort(np.abs(alpha_l))
            alpha_l = alpha_l[s_idx[::-1]]

            # Generate random AoD and AoA angles
            sep = np.pi / 6
            points = generate_points(L, sep, rng=rng)
            phi_l = [pair[0] for pair in points]
            psi_l = [pair[1] for pair in points]
            angle_v = np.hstack([phi_l, psi_l])

            # Frequency components
            omega_phi_true = np.pi * np.cos(phi_l)
            omega_psi_true = -np.pi * np.cos(psi_l)

            # Generate channel and observations
            H = generate_channel_v2(nr, nt, angle_v, alpha_l)
            G = (W.view(myarray).H @ H) @ F
            Z = generate_noise(1.0, SNR, P, Q, rng=rng)

            Y = G + Z
            Y = get_real_imag(Y)

            # Define amplitudes
            if amps_type == 'ones':
                amps = np.ones_like(alpha_l)
            elif amps_type == 'alpha_mag':
                amps = np.abs(alpha_l)
            else:
                raise ValueError(f"Unknown amps_type {amps_type}")

            # Generate ground truth
            gt = generate_gt(L, amps, omega_phi_true, omega_psi_true,  num_points_rx=M, num_points_tx=M, sigma=sigma)

            if normalize_gt:
                #norm = np.sqrt(np.sum(gt**2))
                norm = np.max(gt)
                if norm > 0:
                    gt = gt / norm

            # Preprocess observation (resize depending on P)
            zoom_factor = 4 if P == 16 else (2 if P == 32 else 1)
            data_preal = scipy.ndimage.zoom(Y[:, :, 0], zoom_factor, order=0)
            data_pimag = scipy.ndimage.zoom(Y[:, :, 1], zoom_factor, order=0)
            data = np.dstack((data_preal, data_pimag))

            # Expand gt
            gt = np.expand_dims(gt, axis=-1)

            # Ensure real-valued
            data = np.real(data)
            gt = np.real(gt)

            # Features (psi_l, phi_l)
            features = np.stack([psi_l, phi_l])

            # Meta (L, SNR, P=Q, nt=nr) 
            meta = np.array([L, SNR, P, ntnr], dtype=np.float32)

            yield data.astype(np.float32), gt.astype(np.float32), features.astype(np.float32), meta