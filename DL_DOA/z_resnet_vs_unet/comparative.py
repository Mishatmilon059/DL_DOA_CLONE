# ==========================================================================
# ResNet vs UNet Comparative Script
# ---------------------------------------------------------------
# This script loads precomputed RMSE and Probability of Detection (Pd)
# metrics for both ResNet and UNet models (from serialized .pkl files), 
# and compares their performance under the configuration (L=3, QP=16)
# across different SNR values.
#
# The purpose is to generate a dual subplot figure: 
#   • Left: RMSE vs SNR for ResNet and UNet with diferent color for greater appreciation
#   • Right: Pd vs SNR for ResNet and UNet with diferent color for greater appreciation
#
# The resulting comparison figure is saved in the folder
# "figures_resnet_vs_unet" directory for future analysis.
# ---------------------------------------------------------------
# IMPORTANT
# This script does NOT run inference, it only reads and visualizes metrics.
# ==========================================================================

import os
import pickle
import matplotlib.pyplot as plt

# ===========================
# Path Configuration
# ===========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load directories where metric files are stored
figures_resnet = os.path.join(project_root, 'figures_resnet')
figures_unet = os.path.join(project_root, 'figures_unet')

# Create output directory for comparative plots
figures_dir = os.path.join(project_root, "figures_resnet_vs_unet")
os.makedirs(figures_dir, exist_ok=True)

# =============================
# Load Pickle Data
# =============================
def load_pickle(path):
    # Utility function to load a pickle file.
    with open(path, 'rb') as f:
        return pickle.load(f)

resnet_rmse = load_pickle(os.path.join(figures_resnet, 'resnet_rmse.pkl'))
resnet_pd   = load_pickle(os.path.join(figures_resnet, 'resnet_pd.pkl'))
unet_rmse   = load_pickle(os.path.join(figures_unet,   'unet_rmse.pkl'))
unet_pd     = load_pickle(os.path.join(figures_unet,   'unet_pd.pkl'))

# =============================
# Filtering Parameters
# =============================
# Evaluate only the condition: L=3 and QP=16
target_L = 3
target_QP = 16

def extract_data(metric_dict):
    """
    Extract and sort our metric values for a specific L and QP.
    Returns sorted SNRs with their corresponding metric values.
    """
    snrs = []
    values = []
    for (L, SNR, QP), val in metric_dict.items():
        if L == target_L and QP == target_QP:
            snrs.append(SNR)
            values.append(val)

    # Sort by SNR
    sorted_indices = sorted(range(len(snrs)), key=lambda i: snrs[i])
    snrs_sorted = [snrs[i] for i in sorted_indices]
    values_sorted = [values[i] for i in sorted_indices]
    return snrs_sorted, values_sorted

# Extract values for ResNet and UNet
snr_r, rmse_r = extract_data(resnet_rmse)
_,     pd_r   = extract_data(resnet_pd)
snr_u, rmse_u = extract_data(unet_rmse)
_,     pd_u   = extract_data(unet_pd)

# ===============================
# Plotting ResNet and Unet
# ===============================
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot RMSE comparison
axs[0].plot(snr_r, rmse_r, marker='o', label='ResNet', color='blue')
axs[0].plot(snr_u, rmse_u, marker='o', label='UNet', color='orange')
axs[0].set_xlabel('SNR (dB)')
axs[0].set_ylabel('RMSE (degrees)')
axs[0].set_title(f'RMSE vs SNR (L={target_L}, QP={target_QP})')
axs[0].grid(True)
axs[0].legend()

# Plot Pd comparison
axs[1].plot(snr_r, pd_r, marker='s', label='ResNet', color='blue')
axs[1].plot(snr_u, pd_u, marker='s', label='UNet', color='orange')
axs[1].set_xlabel('SNR (dB)')
axs[1].set_ylabel('Probability of Detection (Pd)')
axs[1].set_title(f'Pd vs SNR (L={target_L}, QP={target_QP})')
axs[1].grid(True)
axs[1].set_ylim(0, 1.05)
axs[1].legend()

# ===============================================================
# Save figure to target path
# ===============================================================
plt.tight_layout()
output_path = os.path.join(figures_dir, 'resnet_vs_unet_comparison.png')
plt.savefig(output_path, dpi=300)

# Log output confirmation
print(f"Comparative figure saved correctly")
