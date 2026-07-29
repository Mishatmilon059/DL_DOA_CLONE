# ==========================================================================
# TVT Blob Inference Script
# ---------------------------------------------------------------
# This module is designed for modular execution from external main.py 
# scripts. It encapsulates the inference logic using pretrained 
# ResNet and UNet models to detect blob peaks and evaluate angular 
# estimations (AoA/AoD). 
# 
# The script includes:
#
# - RMSE and Probability of Detection (Pd) metric computation
# - Grid-to-angle conversion and permutation matching
# - Peak extraction using OpenCV SimpleBlobDetector
# - Support for multiple SNR/L/QP configurations
# - No training procedures included
#
# The main entry points are:
# - run_inference_and_metrics_resnet()
# - run_inference_and_metrics_unet()
# ---------------------------------------------------------------
# ==========================================================================


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import itertools
from src.tvt_data_generation_v3 import *
from src.tvt_models import *
from tqdm import tqdm
from collections import defaultdict


# ====================================================================
# Blob Prediction Normalization Function
# Normalizes model output to an 8-bit image for blob detection.
# ====================================================================

def prepare_prediction_for_peaks(prediction):
    prediction_map = prediction[:, :, 0].numpy()
    img_norm = cv2.normalize(prediction_map, None, 0, 255, cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)
    return img_norm


# ==========================================================
# Initialize OpenCV Blob Detector with default parameters
# ==========================================================
def get_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 1000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

# ===========================================================================
# Sort detected keypoints by descending intensity value in normalized image
# ===========================================================================
def reorder_keypoints(keypoints, img_norm):
    coords = np.array([kp.pt for kp in keypoints])
    coords_rounded = np.round(coords).astype(int)
    amplitudes = []
    for (x, y) in coords_rounded:
        if 0 <= y < img_norm.shape[0] and 0 <= x < img_norm.shape[1]:
            amplitudes.append(img_norm[y, x])
        else:
            amplitudes.append(0)
    amplitudes = np.array(amplitudes)
    sorted_indices = np.argsort(-amplitudes)
    sorted_keypoints = [keypoints[i] for i in sorted_indices]
    sorted_amplitudes = amplitudes[sorted_indices]
    return sorted_keypoints, sorted_amplitudes

# ==============================================================
# Detect blob peaks using OpenCV and extract peak coordinates
# ==============================================================
def get_blob_peaks(pred, detector, use_threshold=False, threshold_value=50):
    img_norm = prepare_prediction_for_peaks(pred)
    if use_threshold:
        _, img_proc = cv2.threshold(img_norm, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        img_proc = img_norm
    keypoints = detector.detect(img_proc)
    keypoints, amplitudes = reorder_keypoints(keypoints, img_norm)
    peaks_predicted = np.array([kp.pt for kp in keypoints])
    return peaks_predicted, amplitudes

# ==================================================================
# Convert angular coordinates from 2π-based to [-π, π] range
# ==================================================================
def wrap_2pi_to_minus_pi(a):
    a = np.asarray(a)
    b = np.where(a > np.pi, a - 2 * np.pi, a)
    return b

# ====================================================================
# Convert detected blob positions into angular values (psi, phi)
# ====================================================================
def peaks_to_angles(peaks, margin_factor=3.0, sigma=0.1, grid_size=256):
    if peaks.shape[0] == 0:
        return np.array([]), np.array([])
    margin = margin_factor * sigma
    peaks_x_y = peaks.T
    extended_range = 2 * np.pi + 2 * margin
    freqs_ext = -margin + (peaks_x_y / grid_size) * extended_range
    freqs_minus_pi = wrap_2pi_to_minus_pi(freqs_ext)
    psi_est = np.arccos(-freqs_minus_pi[1] / np.pi)
    phi_est = np.arccos(freqs_minus_pi[0] / np.pi)
    return psi_est, phi_est

# ===================================================================
# Match predicted angles with ground truth for metric calculation
# ===================================================================
def prepare_for_metric(angles_est, feat):
    from src.tvt_data_generation_v3 import permute_pairs
    L = feat.shape[-1]
    if len(angles_est[0]) < L:
        psi_est = np.full((L,), np.nan)
        phi_est = np.full((L,), np.nan)
        psi_true = feat[0]
        phi_true = feat[1]
    else:
        angles_est = (angles_est[0][:L], angles_est[1][:L])
        pairs_est = list(zip(angles_est[0], angles_est[1]))
        pairs_true = list(zip(feat[0], feat[1]))
        permuted_pairs = permute_pairs(pairs_true, pairs_est)
        first_pairs = [pair[0] for pair in permuted_pairs]
        second_pairs = [pair[1] for pair in permuted_pairs]
        psi_true, phi_true = zip(*first_pairs)
        psi_est, phi_est = zip(*second_pairs)
    gt_angles = np.array([psi_true, phi_true])
    pred_angles = np.array([psi_est, phi_est])
    return gt_angles, pred_angles

# =========================================
# Compute angular difference in degrees
# =========================================
def get_ang_difference(gt_angles, pred_angles, return_flat=True):
    H = np.angle(np.exp(1j * gt_angles) * np.exp(-1j * pred_angles))
    ang_dif = H * (180 / np.pi)
    ang_dif_flat = ang_dif.flatten()
    return ang_dif_flat if return_flat else ang_dif

# ==========================================================================
# Split predictions into "good" and "bad" based on angular error threshold
# ==========================================================================
def filter_angles(ang_dif_flat, max_deg_error=1.0):
    good_angles = ang_dif_flat[np.abs(ang_dif_flat) <= max_deg_error]
    bad_angles = ang_dif_flat[np.abs(ang_dif_flat) > max_deg_error]
    return good_angles, bad_angles



# ==========================================================================
# RESNET Evaluation Function
# ---------------------------------------------------------------
# This function runs inference using a trained ResNet model 
# and computes the RMSE and Probability of Detection (Pd) 
# for each (L, SNR, QP) test configuration.
# ---------------------------------------------------------------
# - L: Number of signal sources
# - SNR: Signal-to-Noise Ratio (dB)
# - QP: Quantization Parameter
# ---------------------------------------------------------------
# Outputs:
# - final_RMSE: Dictionary with RMSE for each condition
# - final_Pd: Dictionary with Pd for each condition
# ==========================================================================

def run_inference_and_metrics_resnet(model_path="models/inf_model_007_256_resnet.h5"):

    # Define testing conditions
    L_values = [3]
    SNR_values = list(range(-10, 30, 5)) # SNR from -10 to 25 dB
    PQ_values = [16]
    ntnr_values = [16]
    examples_per_condition = 1000   # Number of samples per condition
    sigma = 0.07                    # Noise level (standard deviation)
    M = 256                         # Grid size

    # Cartesian product of all condition values
    validation_conditions = list(itertools.product(L_values, SNR_values, PQ_values, ntnr_values))

    # Generator to provide validation data
    val_generator = lambda: validation_data_generator(
        validation_conditions=validation_conditions,
        examples_per_condition=examples_per_condition,
        seed=42,
        M=M,
        normalize_gt=False,
        sigma=sigma,
        amps_type='ones'
    )

    # TensorFlow dataset from generator
    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(2, None), dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32)
        )
    )

    # Load trained model and blob detector
    model = Resnet(input_shape=(64, 64, 2))
    model.load_weights(model_path)
    detector = get_blob_detector()

    results = {}
    val_iterator = iter(val_dataset)
    total = len(validation_conditions) * examples_per_condition

    # Iterate over all validation examples
    for _ in tqdm(range(total), desc="Evaluating Examples", unit="ej"):
        data, _, feat, cond = next(val_iterator)
        L, SNR, QP = int(cond[0]), int(cond[1]), int(cond[2])
        condition = (L, SNR, QP)

        # Run model inference
        pred = model(tf.expand_dims(data, axis=0), training=False)
        pred = tf.squeeze(pred, axis=0)

        # Detect peaks (blobs)
        peaks, amplitudes = get_blob_peaks(pred, detector)
        idx_order = np.argsort(-amplitudes)
        peaks = peaks[idx_order[:L]]

        # Convert peak positions to estimated angles
        angles_est = peaks_to_angles(peaks, sigma=sigma, grid_size=M)
        gt_angles, pred_angles = prepare_for_metric(angles_est, feat.numpy())

         # Store result for current condition
        if condition not in results:
            results[condition] = []
        results[condition].append({'gt': gt_angles, 'pred': pred_angles})

    # Compute RMSE and Pd per condition
    final_RMSE = {}
    final_Pd = {}
    for condition, examples in results.items():
        good_all = []
        bad_all = []
        for example in examples:
            gt, pred = example['gt'], example['pred']
            if np.isnan(pred).any(): continue
            diffs = get_ang_difference(gt, pred)
            good, bad = filter_angles(diffs, max_deg_error=1.0)
            good_all.append(good)
            bad_all.append(bad)

        good_all = np.concatenate(good_all) if good_all else np.array([])
        bad_all = np.concatenate(bad_all) if bad_all else np.array([])
        total = len(good_all) + len(bad_all)

        Pd = len(good_all) / total if total > 0 else np.nan
        RMSE = np.sqrt(np.mean(good_all ** 2)) if len(good_all) > 0 else np.nan

        final_Pd[condition] = Pd
        final_RMSE[condition] = RMSE

    return final_RMSE, final_Pd


# ==========================================================================
# UNET Evaluation Function
# ---------------------------------------------------------------
# This function runs inference using a trained UNet model 
# and computes the RMSE and Probability of Detection (Pd) 
# for each (L, SNR, QP) test configuration.
# ---------------------------------------------------------------
# - L: Number of signal sources
# - SNR: Signal-to-Noise Ratio (dB)
# - QP: Quantization Parameter
# ---------------------------------------------------------------
# Outputs:
# - final_RMSE: Dictionary with RMSE for each condition
# - final_Pd: Dictionary with Pd for each condition
# ==========================================================================
def run_inference_and_metrics_unet(model_path="models/inf_model_007_256_unet.h5"):

    # Define testing conditions
    L_values = [3]
    SNR_values = list(range(-10, 30, 5))
    PQ_values = [16]
    ntnr_values = [16]
    examples_per_condition = 1000   # Number of samples per condition
    sigma = 0.07                    # Noise level (standard deviation)
    M = 256                         # Grid size

    # Cartesian product of test configurations
    validation_conditions = list(itertools.product(L_values, SNR_values, PQ_values, ntnr_values))

    # Generator function to create validation dataset
    val_generator = lambda: validation_data_generator(
        validation_conditions=validation_conditions,
        examples_per_condition=examples_per_condition,
        seed=42,
        M=M,
        normalize_gt=False,
        sigma=sigma,
        amps_type='ones'
    )

    # Define TF dataset structure
    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(2, None), dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32)
        )
    )

    # Load trained UNet model
    model = UNet(M=M)
    model.load_weights(model_path)
    detector = get_blob_detector()

    results = {}
    val_iterator = iter(val_dataset)
    total = len(validation_conditions) * examples_per_condition

    
    # Perform inference and store results
    for _ in tqdm(range(total), desc="Evaluating Examples", unit="ej"):
        data, _, feat, cond = next(val_iterator)
        L, SNR, QP = int(cond[0]), int(cond[1]), int(cond[2])
        condition = (L, SNR, QP)

        # Predict output and extract peaks
        pred = model(tf.expand_dims(data, axis=0), training=False)
        pred = tf.squeeze(pred, axis=0)

        peaks, amplitudes = get_blob_peaks(pred, detector)
        idx_order = np.argsort(-amplitudes)
        peaks = peaks[idx_order[:L]]

        # Convert peaks to angle estimates
        angles_est = peaks_to_angles(peaks, sigma=sigma, grid_size=M)
        gt_angles, pred_angles = prepare_for_metric(angles_est, feat.numpy())

        # Store per-condition results
        if condition not in results:
            results[condition] = []
        results[condition].append({'gt': gt_angles, 'pred': pred_angles})

    # Compute RMSE and Pd metrics
    final_RMSE = {}
    final_Pd = {}
    for condition, examples in results.items():
        good_all = []
        bad_all = []
        for example in examples:
            gt, pred = example['gt'], example['pred']
            if np.isnan(pred).any(): continue
            diffs = get_ang_difference(gt, pred)
            good, bad = filter_angles(diffs, max_deg_error=1.0)
            good_all.append(good)
            bad_all.append(bad)

        good_all = np.concatenate(good_all) if good_all else np.array([])
        bad_all = np.concatenate(bad_all) if bad_all else np.array([])
        total = len(good_all) + len(bad_all)

        Pd = len(good_all) / total if total > 0 else np.nan
        RMSE = np.sqrt(np.mean(good_all ** 2)) if len(good_all) > 0 else np.nan

        final_Pd[condition] = Pd
        final_RMSE[condition] = RMSE

    return final_RMSE, final_Pd
