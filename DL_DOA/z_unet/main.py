# ==========================================================================
# UNet Memory Script
# ---------------------------------------------------------------
# This script runs the file inference using a trained model(UNet), 
# that computes RMSE and Probability of Detection (Pd) metrics for 
# each (L, SNR, QP) configuration, and generates comparative plots 
# and serialized data for future evaluation.
# ---------------------------------------------------------------
# ==========================================================================

import sys
import os
import matplotlib.pyplot as plt
import pickle


# ===============================================================
# Set up project root and output directory
# ===============================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
figures_dir = os.path.join(project_root, "figures_unet")
os.makedirs(figures_dir, exist_ok=True)

# Suppress TensorFlow logs (only show errors)
# 0 = all logs, 1 = info, 2 = warning, 3 = error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Add project root to system path for module import
sys.path.append(project_root)

# Import UNet inference method
from src.TVT_Blob_Inference import run_inference_and_metrics_unet

# ==================================================================
# Run main that call for Inference and Evaluate UNet Model metrics
# ==================================================================
if __name__ == "__main__":
    
    # Load pretrained ResNet model from path files and compute RMSE and Pd for all SNR conditions
    rmse, pd = run_inference_and_metrics_unet("models/inf_model_007_256_unet.h5")

    # Mostrar resultados para L=3 y QP=16
    target_L = 3
    target_QP = 16

    # Filter metrics for target configuration
    snrs = []
    rmses = []
    pds = []

    for (L, SNR, QP), val in rmse.items():
        if L == target_L and QP == target_QP:
            snrs.append(SNR)
            rmses.append(rmse[(L, SNR, QP)])
            pds.append(pd[(L, SNR, QP)])

    # Sort SNR for proper plotting
    sorted_indices = sorted(range(len(snrs)), key=lambda i: snrs[i])
    snrs_sorted = [snrs[i] for i in sorted_indices]
    rmses_sorted = [rmses[i] for i in sorted_indices]
    pds_sorted = [pds[i] for i in sorted_indices]


    # ===============================================================
    # Generate Plot: RMSE and Pd vs SNR
    # ===============================================================
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # RMSE Plot
    axs[0].plot(snrs_sorted, rmses_sorted, marker='o', label='RMSE')
    axs[0].set_xlabel('SNR (dB)')
    axs[0].set_ylabel('RMSE (degrees)')
    axs[0].set_title(f'RMSE vs SNR\n(L={target_L}, Q=P={target_QP})')
    axs[0].grid(True)
    axs[0].legend()

    # Pd Plot
    axs[1].plot(snrs_sorted, pds_sorted, marker='s', label='Pd', color='green')
    axs[1].set_xlabel('SNR (dB)')
    axs[1].set_ylabel('Probability of Detection (Pd)')
    axs[1].set_title(f'Pd vs SNR\n(L={target_L}, Q=P={target_QP})')
    axs[1].grid(True)
    axs[1].set_ylim(0, 1.05)
    axs[1].legend()

    # Save plot to file
    output_path = os.path.join(figures_dir, "unet_results.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    # ===============================================================
    # Save Metric Dictionaries for External Use
    # ===============================================================
    rmse_path = os.path.join(figures_dir, "unet_rmse.pkl")
    pd_path = os.path.join(figures_dir, "unet_pd.pkl")

    with open(rmse_path, "wb") as f:
        pickle.dump(rmse, f)

    with open(pd_path, "wb") as f:
        pickle.dump(pd, f)

    # Output Confirmation
    print(" ·········· Saved RMSE and Pd dictionaries as unet_rmse.pkl and unet_pd.pkl ··········")
    print(f" ·········· Plot saved Correctly on figures_unet folder ··········")

