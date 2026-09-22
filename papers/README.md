# Papers — joint AoA/AoD estimation literature

Reference literature for the DL-DOA project. Downloaded PDFs are open-access (arXiv
preprints or open-access journals) — publisher-paywalled papers with no legitimate free
version are listed below with their DOI/link for manual download via institutional
access (e.g. BUET's IEEE Xplore subscription), not fetched here.

## Downloaded (open access)

| File | Paper | Venue | Status |
|---|---|---|---|
| `Deep_Learning_Based_AoA_and_AoD_Estimation_in_Analog_Millimeter (1).pdf` | Lloria et al., "Deep-Learning-Based AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems" | IEEE TVT 2026 | **This project's base paper.** DOI: 10.1109/TVT.2025.3637908 |
| `A_ResNet_Approach_for_AoA_and_AoD_Estimation_in_Analog_Millimeter_Wave_MIMO_Systems.pdf` | Lloria et al., "A ResNet Approach for AoA and AoD Estimation in Analog Millimeter Wave MIMO Systems" | IEEE PIMRC 2024 | Earlier/prototype version of the base paper. DOI: 10.1109/PIMRC59610.2024.10817260 |
| `Naoumi_2024_ComplexNN_Joint_AoA_AoD_Bistatic_ISAC.pdf` | Naoumi, Bazzi, Bomfin, Chafii, "Complex Neural Network Based Joint AoA and AoD Estimation for Bistatic ISAC" | IEEE JSTSP 2024 | arXiv:2404.00582. **Has public code**: [github.com/salmane-s9/Bistatic_ISAC](https://github.com/salmane-s9/Bistatic_ISAC) |
| `Koh_Lee_2026_DL_Angle_Estimation_Comparative.pdf` | Koh & Lee, "Performance Evaluation of Deep Learning Approaches for Angle Estimation Based on AoA and DoA Estimation" | MDPI Applied Sciences 2026 (open access) | DOI: 10.3390/app16126052. Comparative study (CNN/GRU/A-CRNN), not a single proposed model — supplementary benchmark, indoor-positioning AoA/DoA setting (not mmWave MIMO) |
| `Lim_2021_DL_Beam_Tracking_mmWave_Mobility.pdf` | Lim, Kim, Shim, "Deep Learning-based Beam Tracking for Millimeter-wave Communications under Mobility" | IEEE TCOMM 2021 | arXiv:2102.09785. Relevant if/when doing a tracking-latency follow-up (see chat discussion on inference-time/pilot-rate) |
| `Sub6GHz_mmWave_MIMO_ChannelEst_CNN_UNet_2025.pdf` | "Deep Learning-based mmWave MIMO Channel Estimation using sub-6 GHz Channel Information: CNN and UNet Approaches" | arXiv 2025 | arXiv:2506.11714. Found during this search, not in the original list — CNN/UNet joint AoA-AoD-adjacent, worth a skim |
| `2508.13075v1.pdf` | BeamSeek, "Deep Learning-based DOA Estimation for Low-Complexity mmWave Phased Arrays" | arXiv 2025 | Downloaded earlier this project — real measured inference-time numbers (MLP/CNN/deep-MPDR), used in the inference-latency discussion |
| `Robust_DoA_Estimation_Using_Denoising_Autoencoder_and_Deep_Neural_Networks.pdf` | — | — | Already present, unrelated to this batch |
| `SubspaceNet_Deep_Learning-Aided_Subspace_Methods_for_DoA_Estimation.pdf` | — | — | Already present, unrelated to this batch |

## NOT downloaded — paywalled, no open-access version found

Checked arXiv, the publisher page directly, and ResearchGate for each; none had a
legitimately downloadable free PDF (ResearchGate "Request PDF" pages require the
author to grant access manually, which isn't something to attempt automatically).
Get these via BUET's institutional IEEE Xplore / ACM DL / Springer access, or by
directly messaging the authors (all have ResearchGate profiles, "Request full-text" is
a normal, legitimate ask).

| # | Paper | Venue | DOI | Where to get it |
|---|---|---|---|---|
| 3 | Lloria et al., "Deep Learning Based AoA and AoD Estimation for Millimeter Wave MIMO Systems" | ACM EATIS 2024 | 10.1145/3685243.3685247 | [ACM DL](https://dl.acm.org/doi/10.1145/3685243.3685247) |
| 5 | Meneses-Albalá et al., "Energy and Robustness Trade-offs in Adaptive Neural mmWave Channel Estimation on Edge Devices" | Springer J. Supercomputing 2026 | 10.1007/s11227-026-08246-6 | [Springer](https://link.springer.com/article/10.1007/s11227-026-08246-6) (redirects to login — confirmed not open access) |
| 6 | Naoumi et al., "Deep Learning-Enabled Angle Estimation in Bistatic ISAC Systems" | IEEE GCWkshps 2023 | 10.1109/GCWkshps58843.2023.10464930 | Earlier/shorter version of paper #1 above (which IS downloaded) — lower priority |
| 7 | Huang et al., "MIMO Radar Aided mmWave Time-Varying Channel Estimation in MU-MIMO V2X Communications" | IEEE TWC 2021 | 10.1109/TWC.2021.3085823 | [IEEE Xplore](https://ieeexplore.ieee.org/document/9449980/) |
| 8 | Tong et al., "Deep Learning Compressed Sensing-Based Beamspace Channel Estimation in mmWave Massive MIMO Systems" | — | (not searched yet — lower priority per the original list) | — |
| 9 | Du et al., "Tensor-Based Joint Channel Estimation and Symbol Detection for Time-Varying mmWave Massive MIMO Systems" | IEEE TSP 2021 | 10.1109/TSP.2021.3125607 | [IEEE Xplore](https://ieeexplore.ieee.org/document/9606606/) |
| 10 | Gupta et al., "Bayesian Learning Aided Parameter Estimation and Joint Beamformer Design in mmWave MIMO-OFDM ISAC Systems" | IEEE TCOMM 2025 | 10.1109/TCOMM.2025.3578813 | [IEEE Xplore](https://ieeexplore.ieee.org/iel8/26/11245648/11030617.pdf) (confirmed paywalled, not just geo-blocked) |

## Notes for the literature review

- **Only 2 of the 11 listed papers have confirmed public code**: Naoumi 2024
  ([Bistatic_ISAC](https://github.com/salmane-s9/Bistatic_ISAC)) and this project's own
  base paper, Lloria TVT 2026 ([DLDOA](https://github.com/) — already cloned into this repo).
- Koh & Lee 2026's task setting (indoor AoA/DoA positioning) differs from this project's
  mmWave MIMO joint AoA+AoD setting — cite as a benchmarking-methodology reference, not
  a direct baseline.
- Lim 2021 (beam tracking) becomes directly relevant if/when the "inference time reduces
  pilot-signal period" idea from the chat discussion gets built into an experiment.
