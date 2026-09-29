# Agent 10 - Worked Examples of the IABR-Net v2 Dataset-Generation Pipeline

**Scope.** Inputs, labels/targets and training/tuning/test banks of IABR-Net v2 (not model accuracy). Every stage of the pipeline is reduced to a hand-checkable example: **4 TX antennas, 4 RX antennas, 2 paths, a 4x4 DFT codebook, concrete angles/gains, a concrete phase/gain impairment vector and a concrete noise draw at SNR = 10.0 dB**.

**Verification statement.** Every number below was produced by `scratch/a10_compute.py` / `scratch/a10_build_report.py` and cross-checked numerically: each example is computed (i) by hand-written loops from the formula, and (ii) by the *unmodified project code* (`iabr2_sim.py`, `dldoa_dataset_generation.py`, `TVT_Blob_Inference.py`). For the 4x4 toy, `iabr2_sim.py` is loaded a second time as a separate module object and only its module globals (`P,Q,NT,NR,F_IDEAL,W_IDEAL`) are re-pointed to 4 antennas, so the *real* `simulate`, `angles_to_cells`, `heat_targets`, `to_ri` run on the toy; no project file was touched. **53 checks, all passed; the largest absolute discrepancy among the exact (tolerance <= 1e-6) checks is 1.1e-07** (float64 round-off, or float32 storage for the stored-bank / float32-target checks); the only statistical check (Monte-Carlo ramp-slope std, relative error 2e-4, tolerance 1e-2) is labelled as such in Section 14. The full check table is in Section 14.

Notation: rows = RX / AoA / index $n,q$; columns = TX / AoD / index $m,p$. $\psi$ = AoA, $\phi$ = AoD, all angles in **radians internally** (degrees appear only in prose). Labels: FACT (read in code / reproduced numerically), DERIVATION (algebra I carried out and verified), INFERENCE, ASSUMPTION, CONCERN. "Authors did" = what the code does; "Problem?" = what may be questionable.

---

## 0. The toy scene (used by Examples 2-10)

| item | value |
|---|---|
| $N_t=N_r=P=Q=N$ | 4 |
| path 1 | $\alpha_1=0.8+0.6j$ ($|\alpha_1|=1$), $\psi_1=60^\circ$ ($\cos\psi_1=0.5$), $\phi_1=120^\circ$ ($\cos\phi_1=-0.5$) |
| path 2 | $\alpha_2=0.4-0.3j$ ($|\alpha_2|=0.5$), $\psi_2=72.542^\circ$ ($\cos\psi_2=0.3$), $\phi_2=45.573^\circ$ ($\cos\phi_2=0.7$) |
| path separation (Euclid. in $(\phi,\psi)$) | $\sqrt{(\phi_1-\phi_2)^2+(\psi_1-\psi_2)^2}=1.3173$ rad $>\pi/6=0.5236$ : legal for the sampler |
| SNR | 10.0 dB, i.e. $\sigma^2=10^{-1}=0.1000$, per-component std $\sqrt{\sigma^2/2}=0.2236$ |
| impairment | $\delta_{max}=5.0^\circ$, $\gamma_{max}=2.0$ dB (bounds); the concrete draw is in Example 5 |
| heatmap grid | $G=8=2N$ (toy analogue of the real $G=32=2\cdot16$) |

The two paths were chosen so that path 1 sits **exactly on a DFT beam** (both $\cos$ values are exact beam cosines) and path 2 sits **between beams** (off-grid), which exhibits the two extreme behaviours (delta vs Dirichlet leakage).

---

## 1. Angle sampler and gains (`sample_paths` -> `DG.generate_points`)

**Authors did (FACT, `iabr2_sim.py:28-38`, `dldoa_dataset_generation.py:46`).** Rejection sampling: draw $(\phi,\psi)\sim U[0,\pi]^2$ (call order: `x` = $\phi$ first, `y` = $\psi$ second), accept if Euclidean distance in the *angle* plane to every accepted point is $\ge\pi/6$. Then draw $\alpha_l=\sqrt{1/L}\,(\mathcal N(0,1)+j\mathcal N(0,1))/\sqrt2$ (all real parts first, then all imag parts), and **sort $\alpha$ by $|\alpha|$ descending independently of the angles**.

**Input.** `default_rng(10)`, $L=2$ (seed chosen by me so that one candidate is rejected; the trace below is the actual stream).

**Formula.** accept iff $\min_{i}\|(x,y)-(x_i,y_i)\|_2\ge\pi/6=0.5236$; $\alpha=\sqrt{1/L}\,(a+jb)/\sqrt2$.

**Calculation (candidate trace).**

| draw | $\phi$ cand | $\psi$ cand | min dist to accepted | accepted? |
|---|---|---|---|---|
| 1 | 3.0034 | 0.6525 | - | yes |
| 2 | 2.6026 | 0.4690 | 0.4407 | NO (0.4407 < 0.5236) |
| 3 | 1.6110 | 0.4270 | 1.4105 | yes |

Gain draws: $a=[0.8430, 0.8579]$, $b=[0.4752, -0.4508]$, so raw $\alpha=[0.4215 + 0.2376j, 0.4290 - 0.2254j]$ (factor $\sqrt{1/2}/\sqrt2=0.5$). $|\alpha|=[0.4839, 0.4846]$ -> sort descending -> $\alpha=[0.4290 - 0.2254j, 0.4215 + 0.2376j]$.

**Output.** $\phi=[3.0034, 1.6110]$ rad $=[172.08, 92.30]^\circ$, $\psi=[0.6525, 0.4270]$ rad $=[37.38, 24.47]^\circ$, sorted $\alpha$ as above. Verified: hand loop = `S4.sample_paths` = `DG.generate_points` bit-exactly (error 0.0).

**Physical interpretation / Problems?**
- FACT: the sort permutes only $\alpha$; path 1 (generated first) always receives the strongest gain. Since $\alpha$ are i.i.d. this carries no angular bias (INFERENCE), but "path index = strength rank" is a property of the *label ordering*, not of geometry. Here the two magnitudes (0.4846 vs 0.4839) are nearly tied, so the ordering is fragile - irrelevant for a permutation-invariant heat map.
- FACT: the separation constraint is in **angle** space, while the network's target/beamspace is in **$\cos$** space. $\cos$ compresses angles near $0,\pi$ (end-fire) - see Example 9 (aliasing) for a legal pair that occupies the same cell.
- CONCERN (Low): $\sum|\alpha|^2$ is random ($\alpha\sim CN(0,1/L)$), so the *realised* SNR differs from the nominal one (Example 6).

---

## 2. Steering vectors and the antenna-domain channel $H$

**Authors did (FACT, `iabr2_sim.py:45-47`).** $a(\theta)_k=e^{-j\pi k\cos\theta}/\sqrt N$, $H=\sqrt{N_tN_r}\sum_l\alpha_l\,a_r(\psi_l)\,a_t^H(\phi_l)$. Element form (DERIVATION, verified): $H[n,m]=\sum_l\alpha_l\,e^{-j\pi n\cos\psi_l}\,e^{+j\pi m\cos\phi_l}$ (the $+j$ on the TX side comes from the Hermitian; the $\sqrt{N_tN_r}=4$ cancels the two $1/\sqrt N$).

**Input.** Toy scene of Section 0.

**Calculation.** $a_r(\psi_1)=[0.5000 + 0.0000j, 0.0000 - 0.5000j, -0.5000 + 0.0000j, 0.0000 + 0.5000j]$ ($\cos\psi_1=0.5$: phases $-\pi k/2$); $a_r(\psi_2)=[0.5000 + 0.0000j, 0.2939 - 0.4045j, -0.1545 - 0.4755j, -0.4755 - 0.1545j]$; $a_t(\phi_1)=[0.5000 + 0.0000j, 0.0000 + 0.5000j, -0.5000 + 0.0000j, 0.0000 - 0.5000j]$; $a_t(\phi_2)=[0.5000 + 0.0000j, -0.2939 - 0.4045j, -0.1545 + 0.4755j, 0.4755 - 0.1545j]$. Unit norm verified.

Worked entry $H[1,2]$: $\alpha_1e^{-j\pi\cdot1\cdot0.5}e^{+j\pi\cdot2\cdot(-0.5)}+\alpha_2e^{-j\pi\cdot0.3}e^{+j\pi\cdot2\cdot0.7}$
$=(0.8+0.6j)\,e^{-j3\pi/2}+(0.4-0.3j)\,e^{+j1.1\pi}$
$=-0.6000 + 0.8000j\;+\;-0.4731 + 0.1617j\;=\;-1.0731 + 0.9617j$.

**Output** ($H$, verified = element loop = outer-product form = `DG.generate_channel_v2` = `simulate()['H']`, error 2e-15):

| n\m | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 1.2000 + 0.3000j | 0.6076 - 0.3001j | -1.2089 - 0.8877j | -0.1269 + 0.6383j |
| **1** | 0.5924 - 1.2999j | -0.3911 - 0.3123j | -1.0731 + 0.9617j | 0.9473 + 0.1222j |
| **2** | -1.2089 - 0.8877j | -0.1269 + 0.6383j | 0.6527 + 1.0778j | 0.3000 - 1.2000j |
| **3** | -1.0731 + 0.9617j | 0.9473 + 0.1222j | 0.9000 - 0.4000j | -1.2999 - 0.5924j |

**Physical interpretation.** $H$ is a sum of 2 rank-one plane-wave patterns. For path 1, $\cos\psi_1=0.5$ makes the row phase advance $-90^\circ$ per RX antenna and $\cos\phi_1=-0.5$ makes the column phase advance $-90^\circ$ per TX antenna. Row index $n$ = RX antenna (AoA), column $m$ = TX antenna (AoD); swapping the two would swap $\psi\leftrightarrow\phi$ and the sign of the phase ramp.

**Problems?** None found in the model equation. FACT: the array phase convention is $e^{-j\pi k\cos\psi}$ at RX and $e^{+j\pi m\cos\phi}$ at TX (opposite signs). The evaluator's inverse mapping (Example 9b) uses exactly this pair (`psi=arccos(-f[1]/pi)`, `phi=arccos(f[0]/pi)`), so the conventions are consistent end to end (verified numerically in 9b).

---

## 3. DFT codebooks

**Authors did (FACT, `dldoa_dataset_generation.py:134-205`).** $F[k,p]=e^{-j2\pi kp/N}/\sqrt N$ (TX), $W[k,q]=e^{+j2\pi kq/N}/\sqrt N=F^*$ (RX). Both are unitary when the number of beams equals $N$.

**Calculation** (verified against closed form, unitarity residual 6e-16):

`F` (TX beams $p$ = columns):

| n\m | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 0.5000 + 0.0000j | 0.5000 + 0.0000j | 0.5000 + 0.0000j | 0.5000 + 0.0000j |
| **1** | 0.5000 + 0.0000j | 0.0000 - 0.5000j | -0.5000 + 0.0000j | 0.0000 + 0.5000j |
| **2** | 0.5000 + 0.0000j | -0.5000 + 0.0000j | 0.5000 + 0.0000j | -0.5000 + 0.0000j |
| **3** | 0.5000 + 0.0000j | 0.0000 + 0.5000j | -0.5000 + 0.0000j | 0.0000 - 0.5000j |

`W` (RX beams $q$ = columns):

| n\m | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 0.5000 + 0.0000j | 0.5000 + 0.0000j | 0.5000 + 0.0000j | 0.5000 + 0.0000j |
| **1** | 0.5000 + 0.0000j | 0.0000 + 0.5000j | -0.5000 + 0.0000j | 0.0000 - 0.5000j |
| **2** | 0.5000 + 0.0000j | -0.5000 + 0.0000j | 0.5000 + 0.0000j | -0.5000 + 0.0000j |
| **3** | 0.5000 + 0.0000j | 0.0000 - 0.5000j | -0.5000 + 0.0000j | 0.0000 + 0.5000j |

Beam cosines (DERIVATION): $\cos\bar\phi_p=\angle e^{j2\pi p/P}/\pi$ and $\cos\bar\psi_q=\angle e^{-j2\pi q/Q}/\pi$ (numerically, $\angle e^{+j\pi}=+\pi$ but $\angle e^{-j\pi}=-\pi$ in float64):

| beam index | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $\cos\bar\phi_p$ (TX) | 0.00 | 0.50 | 1.00 | -0.50 |
| $\bar\phi_p$ (deg) | 90.0 | 60.0 | 0.0 | 120.0 |
| $\cos\bar\psi_q$ (RX) | 0.00 | -0.50 | -1.00 | 0.50 |
| $\bar\psi_q$ (deg) | 90.0 | 120.0 | 180.0 | 60.0 |

**Output / interpretation.** The beam grid is uniform in $\cos$ (spacing $2/N=0.5$), **not** in angle: the four beams sit at 90/60/0/120 deg (TX) and 90/120/180/60 deg (RX). Path 1 ($\cos\psi_1=0.5,\ \cos\phi_1=-0.5$) coincides with RX beam $q=3$ and TX beam $p=3$. RX beams run in the opposite direction of TX beams ($W=F^*$), which is why the RX cell formula carries a minus sign (Example 9).

**Edge case (FACT).** For even $N$, beam $p=N/2$ has $\cos\bar\phi=+1$ and $q=N/2$ has $\cos\bar\psi=-1$ (float64 branch of the angle: $\angle e^{+j\pi}=+\pi$, $\angle e^{-j\pi}=-\pi$), i.e. one beam is exactly end-fire; $a(0)=a(\pi)$ (the two end-fire directions are the same spatial frequency).

---

## 4. Noiseless, unimpaired observation $Y=W^HHF$ (beamspace)

**Authors did (FACT).** $Y[q,p]=w_q^H\,H\,f_p$. Insert the model: $Y[q,p]=\sqrt{N_tN_r}\sum_l\alpha_l\,\rho_r^{(l)}[q]\,\rho_t^{(l)}[p]$ with $\rho_r^{(l)}[q]=w_q^Ha_r(\psi_l)$ and $\rho_t^{(l)}[p]=a_t^H(\phi_l)f_p$ (DERIVATION; both verified against the matrix products, error 6e-16). Each is a Dirichlet kernel: $|\rho|=\left|\dfrac{\sin(Nx/2)}{N\sin(x/2)}\right|$ with $x=\pi(\cos\psi_l-\cos\bar\psi_q)$ (RX) resp. $\pi(\cos\phi_l-\cos\bar\phi_p)$ (TX); verified.

**Calculation.**
- Path 1 (on grid): $\rho_r^{(1)}=[0.000 + 0.000j, 0.000 + 0.000j, 0.000 + 0.000j, 1.000 + 0.000j]$, $\rho_t^{(1)}=[0.000 + 0.000j, 0.000 + 0.000j, 0.000 + 0.000j, 1.000 + 0.000j]$ -> a pure delta at $(q,p)=(3,3)$. Contribution: $4\alpha_1\cdot1\cdot1=3.2000 + 2.4000j$ at $(3,3)$, zero elsewhere.
- Path 2 (off grid): $\rho_r^{(2)}[q]$: $x=\pi(0.3-\cos\bar\psi_q)=[0.3,0.8,1.3,-0.2]\pi$, so $|\rho_r^{(2)}|=[0.5237, 0.2500, 0.2668, 0.7694]$; e.g. $q=0$: $\frac{\sin(0.6\pi)}{4\sin(0.15\pi)}=\frac{0.9511}{4\cdot0.4540}=0.5237$; $q=3$: $\frac{\sin(0.4\pi)}{4\sin(0.1\pi)}=0.7694$. $\rho_t^{(2)}$: $|\rho_t^{(2)}|=[0.2668, 0.7694, 0.5237, 0.2500]$.
  Its largest bin is $(3,1)$: $4\cdot0.5\cdot0.7694\cdot0.7694=1.1840$.
- Bin $(3,3)$ therefore holds path 1 **plus** path-2 leakage: $4\alpha_2\rho_r^{(2)}[3]\rho_t^{(2)}[3]=4(0.4-0.3j)(0.4523 + 0.6225j)(0.2023 - 0.1469j)=0.3640 - 0.1244j$, so $Y[3,3]=3.2000 + 2.4000j+0.3640 - 0.1244j=3.5640 + 2.2756j$.

**Output** ($Y=W^HHF$; equals the *real* `simulate(..., noise=0)['Y']` to 2e-15; Parseval $\|Y\|_F=\|H\|_F$ verified):

| q\p | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | -0.0904 - 0.2645j | 0.3549 - 0.7236j | -0.5191 + 0.1774j | -0.2351 - 0.1153j |
| **1** | 0.0588 - 0.1198j | 0.3640 - 0.1244j | -0.2351 - 0.1153j | -0.0404 - 0.1183j |
| **2** | 0.1348 - 0.0461j | 0.3687 + 0.1808j | -0.0904 - 0.2645j | 0.0588 - 0.1198j |
| **3** | 0.3687 + 0.1808j | 0.3829 + 1.1204j | 0.3549 - 0.7236j | 3.5640 + 2.2756j |

$|Y|$:

| q\p | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 0.2795 | 0.8059 | 0.5486 | 0.2619 |
| **1** | 0.1334 | 0.3847 | 0.2619 | 0.1250 |
| **2** | 0.1424 | 0.4106 | 0.2795 | 0.1334 |
| **3** | 0.4106 | 1.1840 | 0.8059 | 4.2285 |

**Physical interpretation.** Rows $q$ = RX beam (AoA axis), columns $p$ = TX beam (AoD axis). The tall peak at $(3,3)$ is path 1 ($4|\alpha_1|=4$). Path 2 is spread over a whole row and column because its cosine sits between beams (bin $(3,1)$: 1.184 is its centre of mass; sidelobes 0.25-0.81). The observation therefore **is** a 2-D spatial-frequency spectrum of $H$ sampled at $N\times N$ points. A path's location has a *continuous* sub-bin offset that only the Dirichlet pattern encodes.

**Problems?** FACT: cell of path 2 in the 4-grid is $(q,p)=(3.4,1.4)$ (Example 9) - the sub-bin offset is 0.4 bin, and with 16 beams the same sub-bin ambiguity remains; the network must super-resolve. This is the intended task, not a data defect.

---

## 5. Hardware impairment (phase and gain) applied to both codebooks

**Authors did (FACT, `iabr2_sim.py:48-53`).** Draw $\varepsilon_{r,n},\varepsilon_{t,m}\sim U(-1,1)\cdot\delta_{max}$ (rad), $g_{r,n},g_{t,m}=10^{U(-1,1)\gamma_{max}/20}$ (uniform in dB); $D_r=g_re^{j\varepsilon_r}$, $D_t=g_te^{j\varepsilon_t}$; distorted codebooks $W=D_rW_{id}$ (row-scaling), $F=D_tF_{id}$; $Y=W^HHF+Z$.

**Input.** The impairment vector (I chose these values in the "uniform" units so that the *real* `simulate` reproduces them through a stub RNG; verified exactly, error 7e-18):

| index | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $\varepsilon_r$ (deg) | 4.0 | -2.0 | 1.0 | -3.0 |
| $g_r$ (dB) | 1.0 | -0.5 | 0.5 | -1.5 |
| $\varepsilon_t$ (deg) | -3.0 | 4.0 | 1.0 | -4.0 |
| $g_t$ (dB) | -1.0 | 0.5 | 1.5 | -0.5 |

All $|\varepsilon|\le\delta_{max}=5^\circ$, all $|g_{dB}|\le\gamma_{max}=2$ dB.

**Calculation.** $g=10^{g_{dB}/20}$: $g_r=[1.1220, 0.9441, 1.0593, 0.8414]$, $g_t=[0.8913, 1.0593, 1.1885, 0.9441]$. Then $D_r=[1.1193 + 0.0783j, 0.9435 - 0.0329j, 1.0591 + 0.0185j, 0.8402 - 0.0440j]$ and $D_t=[0.8900 - 0.0466j, 1.0567 + 0.0739j, 1.1883 + 0.0207j, 0.9418 - 0.0659j]$.

**Key identity (DERIVATION, verified error 5e-16).**
$$W^HHF=(D_rW_{id})^HH(D_tF_{id})=W_{id}^H\,\underbrace{\big(D_r^*HD_t\big)}_{\tilde H_{sig}}\,F_{id},$$
i.e. the RX impairment enters **conjugated** ($e^{-j\varepsilon_r}$, gain $g_r$) and the TX impairment enters as $e^{+j\varepsilon_t}$ (gain $g_t$). Element form: $\tilde H_{sig}[n,m]=g_{r,n}g_{t,m}e^{j(\varepsilon_{t,m}-\varepsilon_{r,n})}H[n,m]$. Example: $\tilde H_{sig}[0,0]/H[0,0]=g_{r,0}g_{t,0}e^{j(-4^\circ-3^\circ)}=0.9925 - 0.1219j$ (magnitude $1.1220\cdot0.8913=1.0000$, phase $-7^\circ$).

**Output** (noiseless impaired observation, for comparison with Section 4):

| q\p | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | -0.1263 - 0.1861j | 0.5014 - 0.7430j | -0.6128 + 0.1618j | -0.0571 - 0.2187j |
| **1** | 0.0241 - 0.1270j | 0.4597 - 0.0367j | -0.2949 - 0.1961j | 0.4122 - 0.0163j |
| **2** | 0.0654 - 0.0779j | 0.3812 + 0.1174j | -0.1458 - 0.2436j | 0.0553 - 0.0197j |
| **3** | 0.2757 - 0.2237j | 0.5123 + 1.1505j | -0.0739 - 1.0000j | 3.5340 + 2.2651j |

Relative change $\|Y_{imp}-Y_{clean}\|_F/\|Y_{clean}\|_F=0.1917$; e.g. bin $(3,2)$ goes from $0.3549 - 0.7236j$ ($|\cdot|=0.8059$) to $-0.0739 - 1.0000j$ ($|\cdot|=1.0027$): path 1 alone (a pure delta at $(3,3)$ without impairment) now puts magnitude 0.4364 into bin $(3,2)$: the on-grid path **leaks** into neighbouring bins.

**Physical interpretation.** Each antenna branch multiplies the received/transmitted signal by a slightly wrong complex weight. Because the codebook is no longer orthogonal, energy of a path leaks out of its beam and the peak-to-sidelobe pattern is changed - this is exactly the corruption the network's impairment head is trained to undo.

**Problems?**
- FACT: phase is uniform in degrees, gain uniform in **dB** (`10**(u*gdm/20)`); a "±2 dB" bound gives linear amplitude 0.794-1.259.
- CONCERN (Low): in `make_batch` each sample draws its own $\delta_{max}\sim U(0,8^\circ)$, $\gamma_{max}\sim U(0,3\,\mathrm{dB})$, then the per-element draws. The impairment law is a compound (mixture) distribution, heavier-tailed than a fixed-bound one. This is a modelling choice, not an error (FACT: `iabr2_sim.py:111-112`).
- CONCERN (Low): errors are i.i.d. across elements and independent between TX and RX. Correlated hardware drift (e.g. common gain drift) is a different regime: only its mean (unidentifiable) part would remain (Example 8).

---

## 6. Noise at a stated SNR, and the final observation $Y$

**Authors did (FACT, `iabr2_sim.py:54-57`, same as `DG.generate_noise` with `var_alpha=1`).** $\sigma^2=10^{-\mathrm{SNR}/10}$, $Z=\sqrt{\sigma^2/2}\,(N_1+jN_2)$, with $N_1$ ($Q\times P$) drawn **first** and $N_2$ second from the same stream. Added *after* the (distorted) combining.

**SNR table** ($\sigma^2$ and per-component std):

| SNR (dB) | -15 | -10 | 0 | 10 | 24 |
|---|---|---|---|---|---|
| $\sigma^2$ | 31.6228 | 10.0000 | 1.0000 | 0.1000 | 0.0040 |
| std $\sqrt{\sigma^2/2}$ | 3.9764 | 2.2361 | 0.7071 | 0.2236 | 0.0446 |

**Input.** `default_rng(7)`, SNR = 10.0 dB -> $\sigma^2=0.1000$, std $=0.2236$.

**Calculation.** $N_1=$

| q\p | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 0.0012 | 0.2987 | -0.2741 | -0.8906 |
| **1** | -0.4547 | -0.9916 | 0.0601 | 1.3402 |
| **2** | -0.4922 | -0.6205 | 0.4898 | 0.3569 |
| **3** | 0.1054 | -0.9305 | -0.0293 | 0.6953 |

$N_2=$

| q\p | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | -1.3442 | -0.4576 | -1.9012 | -1.2895 |
| **1** | -1.8417 | -0.2351 | -1.2674 | 0.2713 |
| **2** | 0.1568 | -0.1869 | -2.5168 | -0.5387 |
| **3** | -0.0485 | 0.1133 | -1.5301 | -0.4778 |

$Z=0.2236\,(N_1+jN_2)$:

| q\p | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 0.0003 - 0.3006j | 0.0668 - 0.1023j | -0.0613 - 0.4251j | -0.1991 - 0.2883j |
| **1** | -0.1017 - 0.4118j | -0.2217 - 0.0526j | 0.0134 - 0.2834j | 0.2997 + 0.0607j |
| **2** | -0.1101 + 0.0351j | -0.1387 - 0.0418j | 0.1095 - 0.5628j | 0.0798 - 0.1205j |
| **3** | 0.0236 - 0.0108j | -0.2081 + 0.0253j | -0.0065 - 0.3421j | 0.1555 - 0.1068j |

**Output.** $Y=W^HHF+Z$ (equals the real `simulate()['Y']`, error 2e-15). The `Y_clean` companion (ideal codebooks + the *same* noise) also equals the real output (error 2e-15).

| q\p | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | -0.1260 - 0.4867j | 0.5682 - 0.8454j | -0.6741 - 0.2633j | -0.2563 - 0.5070j |
| **1** | -0.0776 - 0.5388j | 0.2380 - 0.0893j | -0.2815 - 0.4795j | 0.7119 + 0.0443j |
| **2** | -0.0447 - 0.0429j | 0.2425 + 0.0756j | -0.0363 - 0.8064j | 0.1351 - 0.1402j |
| **3** | 0.2993 - 0.2345j | 0.3042 + 1.1758j | -0.0804 - 1.3421j | 3.6894 + 2.1583j |

**Realised SNR of this draw (DERIVATION).** Mean $|Z|^2=0.0875$ (nominal 0.1), mean $|Y_{imp,noiseless}|^2=1.4014$ (with $\sum|\alpha|^2=1.25$; nominal design value is $E\sum|\alpha|^2=1$). Realised SNR $=10\log_{10}(1.4014/0.0875)=12.05$ dB versus the label 10 dB.

**Physical interpretation.** Noise is white in the beamspace *and* in the antenna domain (unitary $W,F$). The per-bin SNR of path 1's peak bin is $|Y_{33}|^2/\sigma^2\approx17.6/0.1\to22.5$ dB, so a 10-dB SNR *label* corresponds to a much higher peak-bin SNR - the beamforming gain $N_tN_r$.

**Problems?**
- CONCERN (Low, DERIVATION): the "SNR" is an ensemble quantity, since $\sum_l|\alpha_l|^2\sim\mathrm{Gamma}(L,1/L)$ is random. For $L=1$, $|\alpha|^2\sim\mathrm{Exp}(1)$: the realised SNR relative to nominal has median -1.59 dB, 5th percentile -12.90 dB, 95th percentile 4.77 dB. An "SNR = -10" sample may effectively be -23 dB. Not a bug (matches the base paper's `generate_noise(var_alpha=1, ...)`) but SNR-binned results are blurred.
- CONCERN (Low, FACT): noise is added **after** $D_r$; it is not multiplied by $g_r$, unlike a physical front end where thermal noise enters before the gain-error stage. In the antenna domain this makes the per-antenna SNR proportional to $g_{r,n}^2$ (here $\overline{g_r^2}=0.9950$; up to $\pm3$ dB in the training range). It is a modelling choice - the whitened-noise property simplifies the problem; whether it matches the target hardware is an ASSUMPTION the authors do not state (insufficient evidence in code).

---

## 7. Antenna-domain estimate $\tilde H$ (`to_ant`, the "IABC" front end)

**Authors did (FACT, `notebook_src.py`, `to_ant`).** $\tilde H=W_{id}\,Y\,F_{id}^H$ (`einsum('nq,bqp,mp->bnm', W_IDEAL, Y, conj(F_IDEAL))`), which for $N=4$ equals $N\cdot\mathrm{ifft2}(Y)$ (verified, error 4e-15).

**Formula (DERIVATION).** $\tilde H=W_{id}W_{id}^H(D_r^*HD_t)F_{id}F_{id}^H+W_{id}ZF_{id}^H=D_r^*HD_t+Z'$ because $W_{id}W_{id}^H=I$, $F_{id}F_{id}^H=I$ (unitary).

**Calculation.** Signal part $D_r^*HD_t$ (verified equal to `to_ant` of the noiseless $Y$, error 3e-15):

| n\m | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 1.2276 + 0.1515j | 0.7221 - 0.3566j | -1.6719 - 1.0978j | -0.0390 + 0.6882j |
| **1** | 0.4793 - 1.1023j | -0.3563 - 0.3515j | -1.2589 + 1.0146j | 0.8475 + 0.0794j |
| **2** | -1.1970 - 0.7564j | -0.1796 + 0.7077j | 0.8217 + 1.3569j | 0.1943 - 1.2216j |
| **3** | -0.8047 + 0.7212j | 0.8247 + 0.2110j | 0.9257 - 0.3362j | -1.0406 - 0.4525j |

Noise part $Z'=W_{id}ZF_{id}^H$:

| n\m | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | -0.0747 - 0.7320j | -0.1317 + 0.0219j | 0.0083 - 0.4188j | 0.0101 + 0.4407j |
| **1** | 0.0297 - 0.1003j | 0.1979 + 0.0278j | 0.1112 - 0.0513j | 0.1726 - 0.3371j |
| **2** | -0.0518 - 0.1712j | -0.0797 + 0.3630j | 0.0566 + 0.0686j | 0.0432 - 0.1032j |
| **3** | -0.0966 - 0.1129j | -0.1110 - 0.0222j | -0.1047 + 0.0666j | 0.0217 - 0.1418j |

**Output** $\tilde H=D_r^*HD_t+Z'$ (verified, error 2e-15):

| n\m | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **0** | 1.1530 - 0.5805j | 0.5905 - 0.3347j | -1.6636 - 1.5166j | -0.0288 + 1.1289j |
| **1** | 0.5090 - 1.2026j | -0.1584 - 0.3236j | -1.1477 + 0.9632j | 1.0201 - 0.2577j |
| **2** | -1.2487 - 0.9276j | -0.2593 + 1.0707j | 0.8783 + 1.4255j | 0.2375 - 1.3248j |
| **3** | -0.9014 + 0.6083j | 0.7137 + 0.1887j | 0.8210 - 0.2697j | -1.0190 - 0.5943j |

Check of one entry: $\tilde H[0,0]=(1.2276+0.1515j)+(-0.0747-0.7320j)=1.1530-0.5805j$.

**Physical interpretation.** The IABC step returns the *antenna* domain where the impairment is a plain diagonal scaling from the left and right. Then $Z'$ is again i.i.d. $CN(0,\sigma^2)$: $\|Z'\|_F=\|Z\|_F$ exactly; a 20 000-sample Monte-Carlo gives sample covariance with diagonal $\approx0.999$ and largest off-diagonal magnitude $0.015$ (statistical error $\sim1/\sqrt{20000}$; INFERENCE: white).

**Problems?** None: the projection is exact for unitary codebooks. FACT: this holds only when the number of beams equals the number of antennas (square unitary $W,F$), which is the case in every pipeline variant audited here (16 beams, 16 antennas).

---

## 8. Identifiable subspace of the impairment (`project_impairment`)

**Authors did (FACT, `iabr2_sim.py:76-89`).** The targets for the impairment head are the antenna-domain factors' **phase** $[-\angle D_r;\ +\angle D_t]$ (note the RX sign flip, matching Example 5) after removing the mean and the best-fit linear ramp (orthogonal projector with centred index $n_c=n-(N-1)/2$), and the **log-gain** $[\ln|D_r|;\ln|D_t|]$ after removing only the mean.

**Formula (DERIVATION).** For a vector $x$: $c=\bar x$, $s=\dfrac{\sum_n(x_n-c)\,n_c}{\sum n_c^2}$, $x^{\parallel}=x-c-s\,n_c$. For $N=4$: $n_c=[-1.5, -0.5, 0.5, 1.5]$, $\sum n_c^2=5.0$.

**Input** (from Example 5):
$\theta_r=-\angle D_r=-\varepsilon_r=[-0.0698, 0.0349, -0.0175, 0.0524]$ rad, $\theta_t=+\varepsilon_t=[-0.0524, 0.0698, 0.0175, -0.0698]$ rad; $\ln g_r=[0.1151, -0.0576, 0.0576, -0.1727]$, $\ln g_t=[-0.1151, 0.0576, 0.1727, -0.0576]$.

**Calculation.**

| | mean $c$ | slope $s$ (rad/element) | residual (identifiable) |
|---|---|---|---|
| RX phase | 0.0000 | 0.0314 (= 1.80 deg/el.) | [-0.0227, 0.0506, -0.0332, 0.0052] |
| TX phase | -0.0087 | -0.0105 (= -0.60 deg/el.) | [-0.0593, 0.0733, 0.0314, -0.0454] |
| RX log-gain | -0.0144 | (kept) | [0.1295, -0.0432, 0.0720, -0.1583] |
| TX log-gain | 0.0144 | (kept) | [-0.1295, 0.0432, 0.1583, -0.0720] |

RX-phase arithmetic: $\sum x n_c=(-0.0698)(-1.5)+0.0349(-0.5)+(-0.0175)(0.5)+0.0524(1.5)=0.1571$, $s=0.1571/5=0.0314$ rad $=1.8^\circ$ per element; e.g. residual$_0=-0.0698-0.0314(-1.5)=-0.0227$.

**Output.** The 8-vector of phase targets $=[-0.0227, 0.0506, -0.0332, 0.0052, -0.0593, 0.0733, 0.0314, -0.0454]$; the 8-vector of log-gain targets $=[0.1295, -0.0432, 0.0720, -0.1583, -0.1295, 0.0432, 0.1583, -0.0720]$. Residuals sum to zero and are orthogonal to the ramp (errors 1e-17); the hand residual equals `project_impairment`'s algorithm (which is hard-coded to 16 elements; I validated the general-$N$ copy against the real function on 16-element inputs, error 0.0 - and `impairment_targets` layout error 7e-9 in float32). Only 46.0 % of the RX-phase energy and 93.3 % of the TX-phase energy remain: **the ramp component carries the majority of RX phase energy in this draw.**

**Why the removed parts are unidentifiable (DERIVATION, verified exactly).**
- A constant phase/gain on all RX (or TX) elements multiplies $H$ by a scalar, absorbed into $\alpha$ (indistinguishable from the path gains).
- A linear phase ramp is equivalent to an **angle shift**: $e^{js_rn}e^{-j\pi n\cos\psi}=e^{-j\pi n(\cos\psi-s_r/\pi)}$, so $\cos\psi'=\cos\psi-s_r/\pi$; similarly $\cos\phi'=\cos\phi+s_t/\pi$.
- **Exact equivalence in the toy** (verified, error 1e-15): $D_r^*HD_t$ equals a channel with the *residual-only* impairment on the scene with shifted cosines $\cos\psi'=[0.4900, 0.2900]$, $\cos\phi'=[-0.5033, 0.6967]$ (i.e. $\Delta\psi=[0.659, 0.600]^\circ$, $\Delta\phi=[0.221, 0.267]^\circ$) and rescaled gains $\alpha'=[0.8234 + 0.5674j, 0.3876 - 0.3158j]$ (amplitude factor $e^{\bar\ln g_r+\bar\ln g_t}=1.0000$ here by coincidence of symmetric draws, phase $-0.0401$ rad). Pure-ramp equivalence separately verified (error 8e-16).

**Physical interpretation / Problems?**
- FACT: the network's *impairment* targets are correctly the identifiable part (good design): asking it to regress the ramp would be impossible.
- **CONCERN (Low-Medium, DERIVATION):** the **angle labels** $\psi,\phi$ are the *true* (unshifted) angles, but the input only contains the *shifted* scene: the ramp part of the impairment makes the label **irreducibly** off by $\Delta\cos=\mp s/\pi$ even for a perfect estimator. For the toy this is $0.6^\circ$/$0.25^\circ$. Scaling to the real system: slope std $=\delta_{max}/(\sqrt3\,\|n_c\|)$ = 1.291 deg/el. ($N=4$, $\delta=5^\circ$) vs 0.1566 deg/el. ($N=16$; Monte-Carlo 0.1566), i.e. $\sigma_{\Delta\cos}=0.00087$, an angle error of $\approx0.05^\circ/\sin\psi$ - negligible at broadside but growing near end-fire (matches the prior agent's floor P(>1 deg)=1.4 % at $\delta_{max}=5^\circ$, 2.7 % at 8 deg). Gain ramps do not create this problem (a real exponential ramp is not representable as a real angle shift, so it is identifiable). I did not evaluate any estimator - this is a data-level lower bound only.
- FACT: gain **mean** is removed from both RX and TX log-gain; equivalent to absorbing into $|\alpha|$.

---

## 9. Heat-map target (`angles_to_cells`, `heat_targets`)

**Authors did (FACT, `iabr2_sim.py:71-104`).** Map each path to continuous cell coordinates on a $G\times G$ wrap-around grid, round, and paint a circular Gaussian of std `sigma` cells, take the max over paths (NaN-padded paths are masked):
$$q=\frac{G}{2\pi}\,\mathrm{wrap}_{2\pi}(-\pi\cos\psi),\quad p=\frac{G}{2\pi}\,\mathrm{wrap}_{2\pi}(\pi\cos\phi),\quad h[i,j]=\max_l\exp\!\Big(-\frac{d_G(i,q_l^{r})^2+d_G(j,p_l^{r})^2}{2\sigma^2}\Big),$$
where $q^r=\mathrm{round}(q)\bmod G$ and $d_G(a,b)=\min(|a-b|,G-|a-b|)$.

**Input.** Toy scene, $G=8$, $\sigma=1$ cell.

**Calculation.**

| path | $-\pi\cos\psi$ | $q=8\cdot\mathrm{wrap}/2\pi$ | $q^r$ | $\pi\cos\phi$ | $p=8\cdot\mathrm{wrap}/2\pi$ | $p^r$ |
|---|---|---|---|---|---|---|
| 1 | $-\pi/2\to3\pi/2$ | 6.00 | 6 | $-\pi/2\to3\pi/2$ | 6.00 | 6 |
| 2 | $-0.3\pi\to1.7\pi$ | 6.80 | 7 | $0.7\pi$ | 2.80 | 3 |

(simply $q=G(-\cos\psi \bmod 2)/2$, $p=G(\cos\phi\bmod2)/2$.) Note path 1 lands exactly on a cell, path 2 at fractional offsets $(0.2,0.2)$ cells (from $6.8\to7$ and $2.8\to3$).

Value at cell $(6,5)$: path 1 distance $(0,1)$ -> $e^{-1/2}=0.6065$; path 2 distance $(1,2)$ -> $e^{-5/2}=0.0821$; max = 0.6065.
Value at $(0,3)$: path 1 distance $(\min(6,2),\,3)=(2,3)$ -> $e^{-13/2}=0.0015$; path 2 distance $(\min(7,1),0)=(1,0)$ -> $e^{-1/2}=0.6065$; max = 0.6065 (wrap-around: row 0 is adjacent to row 7).

**Output** ($h$, equals the real `heat_targets` to 1e-8, float32; rows = AoA cell $q$, columns = AoD cell $p$):

| q\p | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **0** | 0.018 | 0.082 | 0.368 | 0.607 | 0.368 | 0.082 | 0.135 | 0.082 |
| **1** | 0.002 | 0.018 | 0.082 | 0.135 | 0.082 | 0.018 | 0.011 | 0.007 |
| **2** | 0.000 | 0.002 | 0.007 | 0.011 | 0.007 | 0.002 | 0.000 | 0.000 |
| **3** | 0.002 | 0.000 | 0.000 | 0.000 | 0.002 | 0.007 | 0.011 | 0.007 |
| **4** | 0.018 | 0.002 | 0.007 | 0.011 | 0.018 | 0.082 | 0.135 | 0.082 |
| **5** | 0.082 | 0.018 | 0.082 | 0.135 | 0.082 | 0.368 | 0.607 | 0.368 |
| **6** | 0.135 | 0.082 | 0.368 | 0.607 | 0.368 | 0.607 | 1.000 | 0.607 |
| **7** | 0.082 | 0.135 | 0.607 | 1.000 | 0.607 | 0.368 | 0.607 | 0.368 |

Peaks: exactly two cells equal 1.000, at $(6,6)$ and $(7,3)$.

**Decoder inverse and quantisation (DERIVATION, verified).** Invert on the rounded cells: $\cos\hat\psi=\mathrm{wrap}(-2\pi q^r/G)/\pi$, $\cos\hat\phi=\mathrm{wrap}(2\pi p^r/G)/\pi$:

| | true | $G=8$ recovered | error | $G=32$ recovered | error |
|---|---|---|---|---|---|
| $\psi_1$ | 60.000 | 60.000 | 0 | 60.000 | 0 |
| $\psi_2$ | 72.542 | 75.522 | 2.98 | 71.790 | -0.75 |
| $\phi_1$ | 120.000 | 120.000 | 0 | 120.000 | 0 |
| $\phi_2$ | 45.573 | 41.410 | -4.16 | 46.567 | 0.99 |

(The $G=32$ column is the *real* grid of IABR-Net v2: same scene -> cells $(24,24)$ and $(27,11)$.) One cell spans $2/G$ in $\cos$ (0.25 for $G=8$, 0.0625 for $G=32$); the half-cell bound is 1.79 deg at broadside for $G=32$ (prior agents' value), but larger toward end-fire since $d\psi=d\cos/\sin\psi$.

**Edge cases (verified with the real function).**
1. **NaN padding**: a third path with $\psi=\phi=$ NaN adds exactly 0 (max error 0.0).
2. **Rounding**: `np.round` is half-to-even: $[0.5,1.5,2.5,3.5]\to[0,2,2,4]$. A path whose continuous cell is exactly $k+0.5$ goes to the *even* neighbour, not consistently up or down (FACT); probability zero for continuous angles.
3. **Wrap $q=G$**: $\psi=90^\circ$ gives $-\pi\cos\psi=-1.9\cdot10^{-16}\Rightarrow q=8.0=G$ exactly in float64; `% G` maps it to cell 0, the correct periodic image (FACT from the code: `np.round(q).astype(int) % G`).
4. **End-fire merge**: $\psi=5^\circ$ and $\psi=175^\circ$ (same $\phi$; a pair the sampler legally allows, distance $170^\circ\gg\pi/6$) give $q=16.06$ and $15.94$ on the 32-grid -> **both round to cell 16, one merged peak** (number of cells equal to 1.0: 1, not 2). Steering-vector distance $\|a_r(5^\circ)-a_r(175^\circ)\|$ is only 0.2098 for 16 elements (0.0447 for 4).

**Physical interpretation.** The heat map is a Gaussian-smoothed indicator on the *beam grid* (uniform in $\cos$), with twice the beam resolution ($G=2N$). Path 1 (on-beam) is represented with zero error; path 2 loses its sub-cell offset (0.2 cell -> $3.0^\circ$ and $4.2^\circ$ for $G=8$, $0.75^\circ$ and $1.0^\circ$ for $G=32$).

**Problems?**
- CONCERN (Low): target quantisation - the label peak is on the integer cell, so any decoder that reads the argmax has a $\le0.5$-cell error floor (unless the decoder interpolates; whether it does is outside this audit's scope: insufficient evidence here).
- CONCERN (Medium-Low, DERIVATION): **end-fire merge**. $a(\psi)$ is periodic in $\cos\psi$, so cells $q$ and $q\pm G$ coincide; ULA angles within $\arccos(1-1/32)=14.36^\circ$ of 0 or $180^\circ$ all round to cell 16. With $\psi\sim U[0,\pi]$ the probability that one path lands in that single cell is $2\arccos(1-1/32)/\pi=16.0$ %. The sampler's angle-space separation $\pi/6$ does not prevent two paths from sharing a cell; prior agents measured max-merge in 1.8 % of training samples. It is a physical ambiguity of the array (identifiability of the label), not a coding error; the target `max` operator hides the merge (fewer peaks than $L$).
- FACT: uniform-in-$\psi$ sampling is *not* uniform in $\cos\psi$ (density $\propto1/\sin\psi$) - it over-represents the sparse-in-angle end-fire region of the beam grid.

### 9b. The original (base paper) target for the same scene: 256x256 Gaussian GT

**Authors did (FACT, `dldoa_dataset_generation.py:313-`, `generate_gt`).** $f_1=\pi\cos\phi$ (columns), $f_2=-\pi\cos\psi$ (rows), each wrapped to $[0,2\pi)$, painted on a $256\times256$ grid with margin: $\omega_k=-3\sigma+k\,(2\pi+6\sigma)/256$, $\sigma=0.07$ rad, peak amplitude $1/(2\pi\sigma^2)=32.4806$.

**Calculation.** Step $=(2\pi+0.42)/256=0.02618$ rad ($=0.48^\circ$ in angle at broadside). Path 1: continuous pixel $(\mathrm{row},\mathrm{col})=(187.990,187.990)$ -> peak pixel $(188,188)$, value $32.4801$; path 2: $(211.986,92.006)\to(212,92)$, value $32.4801$.

**Output / round trip.** The project's own evaluator (`peaks_to_angles`, extracted via AST, `x`=column, `y`=row) returns $\hat\psi=[60.006, 72.549]^\circ$, $\hat\phi=[119.994, 45.577]^\circ$ (true $60,\,72.542;\,120,\,45.573$): the conventions $f_1\leftrightarrow\phi$ (columns), $f_2\leftrightarrow\psi$ (rows) with $\arccos(-f_2/\pi)$, $\arccos(f_1/\pi)$ are consistent with the label-generation convention (verified).

**Interpretation / Problems?** The original target is a fine (256) grid whose pixel step is $\approx$ 1/15 of the 16-beam spacing ($2\pi/16$), with a much narrower blob (std 0.07 rad $\approx0.36$ v2-cells). The residual error of this toy round-trip ($\le0.007^\circ$) is small only because both paths happen to fall within 0.014 pixel of a pixel centre (coincidence); the worst case is half a pixel, i.e. $\pm0.24^\circ$ at broadside. v2 replaces this with $G=32$, $\sigma=1$ cell (0.196 rad, 2.8x wider), coarser by 8x but with an easier-to-learn target - a trade-off, not a defect (see quantisation concern above).

---

## 10. Real/imag stacking and up/down-sampling index maps

**Authors did (FACT).** `to_ri(Y) = np.stack([Y.real, Y.imag], -1).astype(float32)` -> shape $(B,Q,P,2)$ - identical to the base-paper generator's `get_real_imag` (`dstack`; verified equal, error 0.0). `make_batch` then transposes to $(B,2,Q,P)$ = channels-first (`iabr2_sim.py:115`). The base paper additionally upsamples $16\times16\to64\times64$ with `scipy.ndimage.zoom(order=0)` per channel, **after** adding the noise (`dldoa_dataset_generation.py:566,585-587`); v2 stores it as an index gather `UP_SRC` and inverts with `DOWN_IDX`.

**Input.** The final toy observation $Y$ of Section 6.

**Calculation (stacking).** $Y[3,3]=3.6894 + 2.1583j$ -> `ri[3,3]` = $[3.6894,\ 2.1583]$; model tensor `y[0,0,3,3]`$=$Re $=3.6894$, `y[0,1,3,3]`$=$Im $=2.1583$. Shapes: $(1,4,4,2)\to(1,2,4,4)$, dtype float32 (cast loses ~1e-7).

**Calculation (up-sampling map).** `zoom(order=0)` with corner alignment uses source index $\mathrm{src}(o)=\mathrm{round}\big(o\,(N-1)/(M-1)\big)$ (verified against scipy for all pixels, no ties possible for these sizes since $10o\ne42k+21$ and $2o\ne10k+5$).
- Toy $4\to16$ ($M=16$): $\mathrm{src}=[0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]$, block sizes $[3, 5, 5, 3]$ (not uniform: the edge blocks are smaller).
- Real $16\to64$: block sizes $[3, 4, 4, 4, 4, 5, 4, 4, 4, 4, 5, 4, 4, 4, 4, 3]$ (edge blocks 3, some interior blocks 5).
- Example: output pixel $(21,10)$ of the 64x64 image copies source cell $(5,2)$, flat index $82=5\cdot16+2$ (`UP_SRC[21*64+10]`). `DOWN_IDX[82]`$=1223=19\cdot64+7$, the **first (top-left) pixel of that source cell's block**, so `downsample16(upsample64(x)) == x` exactly (verified for random inputs, error 0.0; toy error 0.0).

**Output.** A 64x64 image with only 256 distinct complex values in blocks of 3-5 pixels per side; the noise is replicated with the signal (noise is added at 16x16 *before* zoom in the original generator).

**Physical interpretation / Problems?**
- FACT: the 64x64 representation contains no information beyond the 16x16 one; it exists only to match the base-paper's CNN input size. Because block sizes are non-uniform (3,4,5), a given beam bin is over/under-represented by up to a factor 5/4 and 3/4 in the 64x64 image, and the blocks are not aligned to the 4-pixel stride a naive `repeat(4)` would give.
- FACT: v2's own training path feeds 16x16 directly and uses `downsample16` only when loading the 64x64 "official" bank. The identity of the round trip guarantees no information loss (verified).

---

## 11. One real training batch (16 antennas, `make_batch`)

**Authors did (FACT, `iabr2_sim.py:107-117`).** Draw order from a single generator: $L\sim U\{1..9\}$, SNR $\sim U\{-15..24\}$ (integer dB), clean flag ($20\%$), $\delta_{max}\sim U(0,8^\circ)$, $\gamma_{max}\sim U(0,3\text{ dB})$ (drawn for all samples, then zeroed for clean ones), angles+gains (`sample_paths`, padded with NaN to 9), impairment draws and noise (`simulate`), then targets.

**Input.** `default_rng(2026)`, $B=6$ (the *real* `make_batch`, config of the notebook: $G=32$, $\sigma=1$).

| sample | L | SNR (dB) | clean? | $\delta_{max}$ (deg) | $\gamma_{max}$ (dB) | # cells = 1.0 in heat |
|---|---|---|---|---|---|---|
| 0 | 8 | -12 | no | 5.09 | 0.83 | 8 |
| 1 | 2 | -1 | yes | 0.00 | 0.00 | 2 |
| 2 | 1 | 10 | no | 4.12 | 1.58 | 1 |
| 3 | 6 | -1 | no | 6.61 | 1.29 | 6 |
| 4 | 4 | 18 | no | 3.59 | 1.99 | 4 |
| 5 | 5 | 16 | no | 2.71 | 0.04 | 5 |

The L / clean / $\delta_{max}$ / $\gamma_{max}$ / SNR draws were re-generated independently from the same seed (same call order); they agree with the real batch (unit-cell count equals $L$ in every sample, and the clean sample has an all-zero `imp` target).

**Output (shapes/dtypes).** `y`: ('float32', (6, 2, 16, 16)); `heat`: ('float32', (6, 32, 32)); `imp`: ('float32', (6, 32, 2)) (32 = 16 RX + 16 TX; last axis = [phase, log-gain]).
- Number of unit-valued heat cells equals $L$ in all six samples (no merge occurred here).
- Sample 1 (clean): the whole `imp` target is exactly 0.
- Sample 0 `imp` phase (first 8 entries): [-0.0379, 0.0235, 0.0623, -0.0435, -0.0110, 0.0059, 0.0373, -0.0625]; log-gain: [-0.0524, 0.0670, -0.0546, 0.0565, -0.0685, -0.0042, -0.0787, 0.0435]. The phase entries sum to $\le3.7e-08$ over each half (mean removed).

**Physical interpretation / Problems?** The batch is a mixture of L in 1..9 (with $L=9$ up to the pad size), SNR down to $-15$ dB, and per-sample random impairment severity. FACT: the `imp` target of a clean sample is zero (consistent). CONCERN (Low): with SNR $=-15$ dB (noise std 3.98 per component, versus signal amplitude $\sim N|\alpha|=16|\alpha|$ in the peak bin) low-SNR samples can carry almost no label information; they contribute mostly noise to the loss (INFERENCE; not measured here).

---

## 12. Fixed banks: tuning/test bank anatomy

**Authors did (FACT).** `make_bank(...)` loops `for L: for snr:` calling `sample_paths` then `simulate` once per block of 1000 samples, with a fresh `default_rng(seed)`; `T2_d5` uses seed 9105 with phase $\delta_{max}=5^\circ$, gain $0$ dB; blocks are stored as float32 `Y16 (8000,16,16,2)` with `psi`, `phi` **in radians** `(8000,3)`, `L`, `snr`.

**Regeneration (verified, error 0.0).** Re-running with seed 9105, $L=3$, $\mathrm{SNR}=-10$, $\delta_{max}=5^\circ$, $\gamma_{max}=0$ reproduces the first 1000 rows of `Y16`, `psi`, `phi` **bit-exactly** (float32 comparison). SNR blocks: [(-10, 0), (-5, 1000), (0, 2000), (5, 3000), (10, 4000), (15, 5000), (20, 6000), (25, 7000)] = (SNR dB, first row). All 8000 rows have $L=3$ (unique L: [3]).

**Sample 0 of `T2_d5`.** $L=3$, SNR $=-10$ dB: $\psi=[0.1494, 0.2620, 2.2963]$ rad, $\phi=[1.3882, 2.2481, 2.9902]$ rad. Cells on the 16-grid $(q,p)$: [('8.09', '1.45'), ('8.27', '10.99'), ('5.31', '8.09')]. The three largest $|Y|$ bins are at [[7, 12], [10, 3], [13, 2]] - **none** coincides (within 1 cell) with a path cell: at $-10$ dB the per-entry noise power is 10 versus a per-entry signal power $\sim\sum|\alpha|^2\approx1$, so the bank sample is noise-dominated (as designed). Paths 0 and 1 have $\psi=8.6^\circ$, $15.0^\circ$: near end-fire (cell $q\approx8.1,8.3$ = the wrap edge).

**Sample 7000 of the frozen `eval_bank` (64x64, official generator, seed 42).** `meta`$=[3.0, 25.0, 16.0, 16.0]$ (INFERENCE: $[L,\mathrm{SNR},\cdot,\cdot]$; the meaning of the last two fields (16,16?) is not documented in the files I read - insufficient evidence), `feat`$=$ [ψ;φ] (INFERENCE from the peak match below) $=[1.8242, 1.2870, 0.5483]$ / $[1.1073, 1.3775, 0.2421]$ rad. Down-sampled with `downsample16`, path cells $(q,p)$ on the 16-grid: [('2.01', '3.58'), ('13.76', '1.54'), ('9.17', '7.77')]; the strongest bins by $|Y|$ are [[2, 4], [2, 3], [14, 2], [14, 1], [2, 5], [9, 8]] ($|Y|_{max}=7.689$), attributed to paths [0, 0, 1, 1, 0, 2] (nearest path cell): the top peak lies within 1 cell of a true path (assertion passed), confirming the label order and the down-sampling. SNR 25 dB -> clear peaks (plus their Dirichlet sidelobe rows/columns as the next-strongest bins).

**Physical interpretation / Problems?** Banks are exactly reproducible from their seeds (prior agents found seed collisions among v1 banks; none affects the bank regenerated here). FACT: `psi`/`phi` are stored in radians as `float32` (`0.1494...`), not degrees.

---

## 13. Summary of concerns (all labelled; none modifies data, all derived numerically)

| id | label | claim | example | severity |
|---|---|---|---|---|
| C1 | DERIVATION/CONCERN | Linear phase ramp of the RX/TX impairment is exactly an angle shift; true angle labels are then irreducibly off by $\Delta\cos=\mp s/\pi$ | Ex 8: toy $\Delta\psi=0.66^\circ$; $N=16$, $\delta=5^\circ$: $\sigma_{\Delta\cos}=0.00087\Rightarrow0.05^\circ/\sin\psi$ | Low-Medium |
| C2 | CONCERN | Heat-map peak is on the rounded integer cell: sub-cell offset discarded | Ex 9: toy $\psi_2$ error 3.0 deg ($G=8$), 0.75 deg ($G=32$) | Low |
| C3 | DERIVATION/CONCERN | End-fire: all $\psi\in[0,14.4^\circ]\cup[165.6^\circ,180^\circ]$ share cell 16 (16.0 % of uniform-$\psi$ mass); sampler's angle-space separation does not prevent merges | Ex 9 item 4: $5^\circ$ and $175^\circ$ -> one peak | Medium-Low |
| C4 | CONCERN | Noise is added after the gain-error stage (not shaped by $g_r$) | Ex 6 | Low |
| C5 | DERIVATION | SNR label is ensemble-average: realised SNR fluctuates (L=1: 5th-95th pct = -12.9 to 4.8 dB) | Ex 6: 10 dB label -> 12.05 dB realised | Low |
| C6 | FACT | 16->64 up-sampling uses non-uniform blocks (3,4,5) and replicates noise | Ex 10 | Low |
| C7 | FACT | Round-half-even ties and $q=G$ wrap are handled correctly (`% G`) | Ex 9 items 2-3 | None |
| C8 | FACT | Conventions (RX conj, $e^{\mp j\pi k\cos}$, cell formulas, `get_real_imag` layout) are mutually consistent from channel to evaluator | Ex 2, 5, 9b, 10 | None |
| C9 | ASSUMPTION | `eval_bank` `feat` ordering ([ψ;φ]) and `meta[2:4]` meaning are inferred, not documented | Ex 12 | Low |

---

## 14. Verification table

All checks executed by `scratch/a10_compute.py` (assertions abort the script on failure, so the report exists only if every row passed). Toy checks use the real `simulate`/`sample_paths`/`heat_targets`/`angles_to_cells`/`to_ri` on a 4-antenna re-parameterisation of the *same* source file; 16-antenna checks use the module unchanged.

| # | check | max abs error | tol | pass |
|---|---|---|---|---|
| 1 | Ex1 hand-traced sampler == iabr2_sim.sample_paths (phi) | 0.00e+00 | 1e-12 | yes |
| 2 | Ex1 hand-traced sampler == iabr2_sim.sample_paths (psi) | 0.00e+00 | 1e-12 | yes |
| 3 | Ex1 hand-traced alpha (sorted) == sample_paths alpha | 0.00e+00 | 1e-12 | yes |
| 4 | Ex1 DG.generate_points == hand trace | 0.00e+00 | 1e-12 | yes |
| 5 | Ex2 element form == outer-product form | 5.55e-16 | 1e-12 | yes |
| 6 | Ex2 hand H == DG.generate_channel_v2 | 2.12e-15 | 1e-12 | yes |
| 7 | Ex2 \|a_r\|=1 | 0.00e+00 | 1e-12 | yes |
| 8 | Ex3 F closed form | 8.88e-16 | 1e-12 | yes |
| 9 | Ex3 W closed form | 8.88e-16 | 1e-12 | yes |
| 10 | Ex3 F unitary | 6.05e-16 | 1e-12 | yes |
| 11 | Ex3 W unitary | 5.49e-16 | 1e-12 | yes |
| 12 | Ex3 W = conj(F) | 1.78e-15 | 1e-12 | yes |
| 13 | Ex4 rho_r = w_q^H a_r | 3.89e-16 | 1e-12 | yes |
| 14 | Ex4 rho_t = a_t^H f_p | 6.21e-16 | 1e-12 | yes |
| 15 | Ex4 sum over paths of 4 a rho_r rho_t == W^H H F | 2.51e-15 | 1e-12 | yes |
| 16 | Ex4 Dirichlet magnitude == \|rho_r\| (path 2) | 1.11e-16 | 1e-12 | yes |
| 17 | Ex4 path 1 on grid: rho_r = delta_(q=3) | 5.23e-16 | 1e-12 | yes |
| 18 | Ex4 path 1 on grid: rho_t = delta_(p=3) | 5.23e-16 | 1e-12 | yes |
| 19 | Ex4 iabr2_sim.simulate (clean, zero noise) == hand W^H H F | 1.91e-15 | 1e-12 | yes |
| 20 | Ex4 simulate H == hand H | 2.12e-15 | 1e-12 | yes |
| 21 | Ex4 Parseval \|\|Y\|\|=\|\|H\|\| | 8.88e-16 | 1e-12 | yes |
| 22 | Ex5 D_r from true simulate == hand g e^{j eps} | 6.94e-18 | 1e-12 | yes |
| 23 | Ex5 D_t == hand | 6.94e-18 | 1e-12 | yes |
| 24 | Ex5 W^H H F == W_id^H (D_r^* H D_t) F_id | 4.65e-16 | 1e-12 | yes |
| 25 | Ex5/6 TRUE simulate() Y (impaired + noise) == hand pipeline | 1.97e-15 | 1e-12 | yes |
| 26 | Ex6 simulate Y_clean == clean-codebook + SAME noise | 1.91e-15 | 1e-12 | yes |
| 27 | Ex7 to_ant(noiseless Y) == D_r^* H D_t | 2.59e-15 | 1e-12 | yes |
| 28 | Ex7 to_ant(Y) == D_r^* H D_t + W Z F^H | 2.22e-15 | 1e-12 | yes |
| 29 | Ex7 to_ant == N * ifft2(Y) | 3.98e-15 | 1e-12 | yes |
| 30 | Ex7 \|\|Z'\|\|_F == \|\|Z\|\|_F (unitary) | 0.00e+00 | 1e-12 | yes |
| 31 | Ex8 generalised proj == iabr2_sim.project_impairment (16 el.) | 0.00e+00 | 1e-12 | yes |
| 32 | Ex8 hand residual == proj4 (Rx phase) | 0.00e+00 | 1e-12 | yes |
| 33 | Ex8 hand residual == proj4 (Tx phase) | 0.00e+00 | 1e-12 | yes |
| 34 | Ex8 residual sums to 0 | 1.39e-17 | 1e-12 | yes |
| 35 | Ex8 residual orthogonal to ramp | 3.12e-17 | 1e-12 | yes |
| 36 | Ex8 impairment_targets phase layout [Rx(-angle), Tx(+angle)] | 6.70e-09 | 1e-06 | yes |
| 37 | Ex8 D_r^* H D_t  ==  residual-only impairment on shifted-angle, rescaled scene (EXACT) | 9.69e-16 | 1e-12 | yes |
| 38 | Ex8 pure ramp == pure angle shift | 8.25e-16 | 1e-12 | yes |
| 39 | Ex8 std(ramp slope) formula vs Monte-Carlo (16 el.) | 2.05e-04 | 1e-02 | yes |
| 40 | Ex9 angles_to_cells == hand | 8.88e-16 | 1e-12 | yes |
| 41 | Ex9 heat_targets == hand double loop | 9.15e-09 | 1e-06 | yes |
| 42 | Ex9 NaN-padded third path contributes exactly 0 | 0.00e+00 | 1e-12 | yes |
| 43 | Ex10 to_ri == DG.get_real_imag layout | 0.00e+00 | 1e-06 | yes |
| 44 | Ex10 y[0,0]=Re, y[0,1]=Im | 1.09e-07 | 1e-06 | yes |
| 45 | Ex10 toy zoom map: src = round(o (N-1)/(M-1)) | 0.00e+00 | 1e-12 | yes |
| 46 | Ex10 gather (UP_SRC style) == scipy.ndimage.zoom per channel | 0.00e+00 | 1e-07 | yes |
| 47 | Ex10 toy downsample(upsample(x)) == x | 0.00e+00 | 1e-07 | yes |
| 48 | Ex10 real UP_SRC == round(5o/21) rows x cols | 0.00e+00 | 1e-12 | yes |
| 49 | Ex10 real down(up(x)) == x | 0.00e+00 | 1e-12 | yes |
| 50 | Ex10 real upsample64 == scipy zoom per channel | 0.00e+00 | 1e-07 | yes |
| 51 | Ex12 regenerate T2_d5 block 0 (SNR -10) == stored Y16 | 0.00e+00 | 1e-12 | yes |
| 52 | Ex12 regenerate T2_d5 block 0 == stored psi | 0.00e+00 | 1e-12 | yes |
| 53 | Ex12 regenerate T2_d5 block 0 == stored phi | 0.00e+00 | 1e-12 | yes |

Reproduce: `python D:\ai_ml_project\IABR_v2_DATASET_AUDIT\scratch\a10_build_report.py` (requires `IABR2_SUITE` to be `D:\ai_ml_project`; the script sets it).
