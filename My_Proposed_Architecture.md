# My Proposed Architecture

## Overview

**Physics-Informed Deep-Unfolding Sparse Recovery + Subspace Attention + Differentiable Set Prediction**

```text
Complex Analog-Beamformed Observation
              │
              ▼
       Y ∈ ℂ^(Q × P)
              │
              ▼
┌─────────────────────────────────────┐
│ STAGE 1                             │
│ Physics-Informed Deep Unfolding     │
│ Sparse Recovery                     │
│                                     │
│ Steering-Vector Dictionary A        │
│              ↓                      │
│ Linear Transform                    │
│              ↓                      │
│ Sparse Coding                       │
│              ↓                      │
│ Data Consistency                    │
│              ↓                      │
│ Repeat for K Unfolded Layers        │
└─────────────────────────────────────┘
              │
              ▼
       Sparse Angular Representation
              │
              ▼
┌─────────────────────────────────────┐
│ STAGE 2                             │
│ Subspace Attention / Transformer    │
│                                     │
│ Feature Embedding                   │
│              ↓                      │
│ Positional Encoding                 │
│              ↓                      │
│ Multi-Head Self-Attention           │
│              ↓                      │
│ Feed-Forward Network                │
│              ↓                      │
│ Feature Representation F            │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ STAGE 3                             │
│ Differentiable Set Prediction       │
│                                     │
│ Learnable Angle Queries              │
│              ↓                      │
│ Cross-Attention                     │
│              ↓                      │
│ Prediction Heads                    │
│        ↙          ↓          ↘      │
│      AoD       AoA       Confidence  │
│     (φᵢ)       (ψᵢ)          cᵢ     │
│                                     │
│ Optional: Source-Count Head (L)     │
└─────────────────────────────────────┘
              │
              ▼
     { (φᵢ, ψᵢ, cᵢ) }ᵢ₌₁ᴸ
              │
              ▼
        Estimated AoA / AoD
```

## Stage 1 — Physics-Informed Deep-Unfolding Sparse Recovery

The raw complex observation is processed using the known antenna/steering-vector model rather than first converting it into an upsampled image.

\[
Y \in \mathbb{C}^{Q	imes P}
\]

A steering-vector dictionary is constructed:

\[
A=[a(	heta_1),a(	heta_2),\ldots,a(	heta_G)]
\]

The sparse-recovery optimization is unfolded into **K trainable layers**. Each unfolded layer contains:

1. Linear Transform
2. Sparse Coding
3. Data Consistency

The output is a sparse angular representation.

## Stage 2 — Subspace Attention / Transformer

The sparse angular representation is converted into features and processed with attention to capture relationships between angular components, particularly for low-SNR and closely spaced paths.

Main blocks:

- Feature embedding
- Positional encoding
- Multi-head self-attention
- Feed-forward network
- Add & normalization

Output:

\[
F = 	ext{Transformer}(X_K)
\]

## Stage 3 — Differentiable Set Prediction

Instead of using a non-differentiable OpenCV blob detector, learnable queries directly predict an unordered set of propagation paths.

Each query predicts:

\[
(\phi_i,\psi_i,c_i)
\]

where:

- \(\phi_i\) = AoD
- \(\psi_i\) = AoA
- \(c_i\) = confidence

An optional source-count head can estimate the number of paths \(L\).

Hungarian matching can be used during training to match predicted paths with the ground-truth set.

## Final Output

\[
oxed{
\{(\phi_i,\psi_i,c_i)\}_{i=1}^{L}
}
\]

representing the estimated AoD, AoA, and confidence for the detected propagation paths.
