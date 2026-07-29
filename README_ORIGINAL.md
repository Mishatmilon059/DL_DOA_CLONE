# DL_DOA

Deep-learning-based AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems


Authors: 
Diego Lloria, Sandra Roger, Carmen Botella-Mascarell, Maximo Cobos, Saúl Villaescusa (Universitat de València) 


--------------------------------------------------------------------
PROJECT DESCRIPTION
--------------------------------------------------------------------

This project implements a deep learning-based approach for estimating 
Angle-of-Arrival (AoA) and Angle-of-Departure (AoD) parameters in parametric 
mmWave MIMO systems with analog beamforming (ABF).

This implementation reproduces the results and architecture described in:
"Deep-learning-based AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems"
(IEEE Transactions on Vehicular Technology, 2025)

Models:
--------
Download or place pretrained models into:

    - models/inf_model_007_256_resnet.h5
    - models/inf_model_007_256_unet.h5 
    
unet is in diferent parts, check models/readme.md

--------------------------------------------------------------------
SYSTEM REQUIREMENTS
--------------------------------------------------------------------

- Conda (recommended) to create the enviroment (Python 3.10.13)

- TensorFlow >= 2.10
- NumPy
- SciPy
- Matplotlib
- tqdm
- scikit-learn

--------------------------------------------------------------------
INSTALLATION & SETUP
--------------------------------------------------------------------

1. Install Anaconda or Miniconda if not installed.

2. Create and activate your conda environment (example: myenv):
   
   conda create -n demo python=3.10.13
   
   conda activate demo

4. Install required packages:

   conda install tensorflow numpy scipy matplotlib scikit-learn tqdm (PREVIOUS SYSTEM REQUIREMENTS) ...

--------------------------------------------------------------------
HOW TO REPRODUCE THE EXPERIMENTS
--------------------------------------------------------------------

All scripts can be run directly from project root. Each folder contains an independent entry point.

A. Run UNet inference (generates RMSE, Pd and plots):

    python z_unet/main.py

Output:
    → figures_unet/unet_results.png
    → figures_unet/unet_rmse.pkl
    → figures_unet/unet_pd.pkl

B. Run ResNet inference (generates RMSE, Pd and plots):

    python z_resnet/main.py

Output:
    → figures_resnet/resnet_results.png
    → figures_resnet/resnet_rmse.pkl
    → figures_resnet/resnet_pd.pkl

C. Compare ResNet vs UNet (comparative analysis):

    python z_resnet_vs_unet/comparative.py

Output:
    → figures_comparative/comparative_results.png


--------------------------------------------------------------------
NOTES
--------------------------------------------------------------------


- Each main.py script automatically creates output folders if they do not exist.
- The core inference logic is centralized in src/TVT_Blob_Inference.py
- The models were trained following the methodology described in the paper, using synthetic datasets.


- This implementation assumes that you have access to the custom 
  data generators and metric functions described in the paper.

- The experiments simulate different SNRs, number of paths (L), 
  and codebook configurations (P, Q) as described in the paper.

- The models are designed for supervised learning with training data 
  consisting of synthetic channel realizations.

- Extended ResNet achieves better Pd (detection probability) especially 
  at low SNR levels, maintaining competitive RMSE levels.

--------------------------------------------------------------------
REFERENCES
--------------------------------------------------------------------

A ResNet Approach for AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems, 
IEEE PIMRC 2024.

Deep-learning-based AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems,
IEEE Transactions on Vehicular Technology, 2025.

--------------------------------------------------------------------
