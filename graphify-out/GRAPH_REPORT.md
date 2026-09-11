# Graph Report - ai_ml_project  (2026-09-11)

## Corpus Check
- 27 files · ~316,961 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 228 nodes · 375 edges · 11 communities (10 shown, 1 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Physics Channel Model
- DoA Paper Methods
- ResNet Architecture
- Dataset Generation
- PIA-Net Bug Log
- UNet Architecture
- Evaluation Metrics
- SE-ResNet Notebook
- Proposed Architecture
- Result Figures
- Subspace Methods

## God Nodes (most connected - your core abstractions)
1. `BUJHO_SHOHOJ_VABE.md — PIA-Net Explanation (Bengali)` - 16 edges
2. `BeamSeek: Deep Learning-based DOA Estimation for Low-Complexity mmWave Phased Arrays` - 13 edges
3. `validation_data_generator()` - 11 edges
4. `validation_data_generator()` - 11 edges
5. `A ResNet Approach for AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems` - 11 edges
6. `Deep-Learning-Based AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems (IEEE TVT 2026)` - 11 edges
7. `SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation` - 11 edges
8. `run_inference_and_metrics_resnet()` - 10 edges
9. `run_inference_and_metrics_unet()` - 10 edges
10. `training_data_generator()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Ground-Truth Heatmap Formation — spatial-frequency mapping to 256x256 image with path markers` --semantically_similar_to--> `256x256 Heatmap Output — each bright spot encodes (AoD, AoA) for a path`  [INFERRED] [semantically similar]
  notebooks/DLDOA_Visualization.ipynb.txt → BUJHO_SHOHOJ_VABE.md
- `ResNet Model — inf_model_007_256_resnet.h5, 256x256 output, stronger at higher SNR` --conceptually_related_to--> `PIA-Net vs ResNet Comparison — PIA-Net competitive at low SNR (-10 dB), gap at high SNR (+25 dB)`  [INFERRED]
  README_ORIGINAL.md → BUJHO_SHOHOJ_VABE.md
- `TVT_Blob_Inference.py — Centralized inference pipeline computing RMSE and Pd metrics` --conceptually_related_to--> `Pd — Detection Probability metric (fraction of angles correctly detected within 1 degree)`  [INFERRED]
  README_ORIGINAL.md → BUJHO_SHOHOJ_VABE.md
- `TX Uniform Linear Array (ULA) — n_t antennas spaced lambda/2, steering vector defined` --conceptually_related_to--> `Steering-Vector Dictionary A — columns are array steering vectors for discrete angles, used in deep unfolding`  [INFERRED]
  notebooks/DLDOA_Visualization.ipynb.txt → My_Proposed_Architecture.md
- `train_unet()` --calls--> `Conv_Block()`  [INFERRED]
  dldoa_dataset_generation.py → DL_DOA/src/tvt_models.py

## Import Cycles
- None detected.

## Communities (11 total, 1 thin omitted)

### Community 0 - "Physics Channel Model"
Cohesion: 0.07
Nodes (43): beamforming_vector_generation_P(), beamforming_vector_generation_Q(), ev(), gaus2d(), generate_channel_v2(), generate_gt(), generate_noise(), generate_points() (+35 more)

### Community 1 - "DoA Paper Methods"
Cohesion: 0.11
Nodes (30): BUJHO_SHOHOJ_VABE.md — PIA-Net Explanation (Bengali), AoA — Angle of Arrival (signal direction at receiver), AoD — Angle of Departure (signal direction from transmitter), Blob Detection — post-processing to locate bright spots in output heatmap, Bug 1 — Training Data Starvation (only 428 samples used instead of adequate dataset), Bug 2 — Physics Formula Sign Error (steering vector sign flipped, peak 78 degrees off), Bug 3 — ISTA Step Size 725x Too Large (divergence masked by gradient clipping), Bug 4 — Coordinate System Mismatch (physics branch output in wrong projection, 119 pixel / ~57 degree offset) (+22 more)

### Community 2 - "ResNet Architecture"
Cohesion: 0.11
Nodes (27): beamforming_vector_generation_P(), beamforming_vector_generation_Q(), data_generation(), ev(), gaus2d(), generate_channel_v2(), generate_gt(), generate_noise() (+19 more)

### Community 3 - "Dataset Generation"
Cohesion: 0.14
Nodes (25): Agile Beam Switching, Anti-Rectifier Activation (AReLU), Convolutional Neural Network (CNN), COSMOS NSF PAWR Testbed, Covariance Matrix / Empirical Autocorrelation, Data Augmentation for Training, Denoising Autoencoder (DAE), Deep Neural Networks (DNN) (+17 more)

### Community 4 - "PIA-Net Bug Log"
Cohesion: 0.13
Nodes (20): figures_resnet/readme.txt — ResNet Evaluation Outputs, figures_resnet_vs_unet/readme.txt — Comparative Plots ResNet vs UNet, figures_unet/readme.txt — UNet Evaluation Outputs, models/readme.txt — Pre-trained Model Checkpoints, z_resnet/readme.txt — ResNet Entry Point, z_resnet_vs_unet/readme.txt — ResNet vs UNet Comparison Entry Point, z_unet/readme.txt — UNet Entry Point, DLDOA_Visualization.ipynb — Dataset Physics Visualisation Notebook (+12 more)

### Community 5 - "UNet Architecture"
Cohesion: 0.14
Nodes (18): Concat_Block(), Conv_Block(), Feature_Extraction_Block(), Defines a basic residual block with two Conv2D layers. Parameters: x : Input…, Constructs a deep ResNet model for super-resolution inference. Parameters:…, Constructs a flexible UNet architecture. Parameters: length, width : Spatial…, res_conv(), Resnet() (+10 more)

### Community 6 - "Evaluation Metrics"
Cohesion: 0.23
Nodes (16): 6G and Beyond-5G Networks, Analog Beamforming (ABF), Angle of Arrival (AoA) and Angle of Departure (AoD) Estimation, Blob Detection for Peak Search, Cramer-Rao Lower Bound (CRLB), DFT-Based Codebook for Beam Search, Supervised Image-to-Image Translation, Levenberg-Marquardt Algorithm for Model Fitting (+8 more)

### Community 7 - "SE-ResNet Notebook"
Cohesion: 0.28
Nodes (13): filter_angles(), get_ang_difference(), get_blob_detector(), get_blob_peaks(), peaks_to_angles(), prepare_for_metric(), prepare_prediction_for_peaks(), reorder_keypoints() (+5 more)

### Community 8 - "Proposed Architecture"
Cohesion: 0.20
Nodes (15): ResNet DOA Results Figure, UNet DOA Results Figure (RMSE and Pd vs SNR), Deep Learning DOA Estimation using ResNet, DOA estimation performance is SNR-sensitive; low SNR degrades both accuracy and detection, Experiment Configuration: L=3 Sources, Q=P=16, Pd Increases with Increasing SNR (0.2 to ~0.93), Probability of Detection vs SNR (L=3, Q=P=16), RMSE vs SNR (L=3, Q=P=16) (+7 more)

### Community 9 - "Result Figures"
Cohesion: 0.67
Nodes (7): ResNet vs UNet Comparison Figure (RMSE and Pd vs SNR, L=3, QP=16), Experiment configuration: L=3 sources, QP=16 quantization parameter, Probability of Detection (Pd) vs SNR metric for DOA, ResNet model for DOA estimation, RMSE vs SNR metric for DOA (degrees), UNet model for DOA estimation, UNet slightly outperforms ResNet on RMSE and Pd at higher SNR (L=3, QP=16)

## Knowledge Gaps
- **21 isolated node(s):** `z_resnet/readme.txt — ResNet Entry Point`, `z_resnet_vs_unet/readme.txt — ResNet vs UNet Comparison Entry Point`, `z_unet/readme.txt — UNet Entry Point`, `SNR — Signal to Noise Ratio (in dB; -10 dB noisy, +25 dB clean)`, `Image-to-Image Formulation — 16x16 antenna input mapped to 256x256 heatmap output` (+16 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 72 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train_unet()` connect `UNet Architecture` to `Physics Channel Model`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `train_resnet()` connect `UNet Architecture` to `Physics Channel Model`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `validation_data_generator()` (e.g. with `run_inference_and_metrics_resnet()` and `run_inference_and_metrics_unet()`) actually correct?**
  _`validation_data_generator()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `z_resnet/readme.txt — ResNet Entry Point`, `z_resnet_vs_unet/readme.txt — ResNet vs UNet Comparison Entry Point`, `z_unet/readme.txt — UNet Entry Point` to the rest of the system?**
  _21 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Physics Channel Model` be split into smaller, more focused modules?**
  _Cohesion score 0.06868686868686869 - nodes in this community are weakly interconnected._
- **Should `DoA Paper Methods` be split into smaller, more focused modules?**
  _Cohesion score 0.10804597701149425 - nodes in this community are weakly interconnected._
- **Should `ResNet Architecture` be split into smaller, more focused modules?**
  _Cohesion score 0.1103448275862069 - nodes in this community are weakly interconnected._