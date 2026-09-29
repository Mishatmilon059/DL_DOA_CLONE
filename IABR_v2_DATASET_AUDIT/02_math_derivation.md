# Agent 02: Mathematical derivation audit of the IABR-Net v2 dataset-generation pipeline

**Scope.** Everything that produces model inputs, targets, training batches and test/tuning banks for IABR-Net v2. Model accuracy is out of scope.
**Method.** Every equation was re-derived by hand and then checked numerically. Four verification scripts are in `D:\ai_ml_project\IABR_v2_DATASET_AUDIT\scratch\`:

| script | what it verifies |
|---|---|
| `a02_v1_codebook_toy.py` | DFT codebooks and their unitarity, the 4x4 two-path toy, the impairment toy, `to_ant`, whiteness of the noise after `to_ant`, simulator == original `generate_channel_v2` |
| `a02_v2_ident_snr_maps.py` | Jacobian null space (identifiability), ramp-equals-angle-shift equivalence, the `project_impairment` projector, the ramp-induced angle floor, SNR bookkeeping, the up/down-sample maps, heat targets, end-fire aliasing |
| `a02_v3_banks.py` | bitwise regeneration of the frozen `eval_bank.npz`, bitwise regeneration of all 7 v2 banks, bank statistics, v1 bank seed collisions |
| `a02_v4_gt_batch.py` | original Gaussian GT grid vs paper Eq. 11, GT-to-angle inverse, heat wrap at the grid edge, sampler feasibility, `make_batch` output contract |
| `a02_v5_independent.py` | code-independent checks (does not import `iabr2_sim`): hand arithmetic of the impaired 4x4 toy, noise colouring under Rx gain error, and a data-only test of the full convention chain (steering signs, codebook, `to_ant`, noise scale, stored angle labels) on the saved banks |

**Re-verification pass.** All four original scripts were re-run in a second pass (outputs in `scratch\rerun\`); every number quoted below reproduced identically (including the 310 s bitwise regeneration of `eval_bank.npz`). The independent script `a02_v5_independent.py` was added in that pass (sections 4, 5 and 13).

Paper text was extracted to `scratch\a02_tvt.txt` (TVT 2026) and `scratch\a02_jsc.txt` (J. Supercomputing 2026). Line numbers below refer to those files.

Labels: **FACT** (read directly from code or paper), **DERIVATION** (worked out analytically), **VERIFIED** (checked numerically, value quoted), **INFERENCE**, **ASSUMPTION**, **CONCERN**, **VERIFIED BUG**.

---

## 0. Notation and conventions (as implemented)

| symbol | meaning | dims / domain | code |
|---|---|---|---|
| $n_t=n_r=P=Q=16$ | Tx/Rx ULA elements and codebook sizes | scalars | `iabr2_sim.py:11` |
| $\psi_l$ | AoA of path $l$ | $[0,\pi]$ rad | `sample_paths` `:37` (`p[1]`) |
| $\phi_l$ | AoD of path $l$ | $[0,\pi]$ rad | `:37` (`p[0]`) |
| $u_l=\pi\cos\psi_l$, $v_l=\pi\cos\phi_l$ | spatial frequencies ("virtual angles") | $[-\pi,\pi]$ | NOMP convention, `notebook_src.py:287,535` |
| $\omega_{\psi_l}=-\pi\cos\psi_l$, $\omega_{\phi_l}=\pi\cos\phi_l$ | paper's frequencies (TVT Eq. 10) | $[-\pi,\pi]$ | `dldoa_dataset_generation.py:454-455` |
| $\alpha_l$ | complex path gain, $\mathcal{CN}(0,1/L)$, sorted by $\lvert\alpha\rvert$ descending | $\mathbb{C}$ | `iabr2_sim.py:35-36` |
| $\mathbf H$ | channel | $\mathbb C^{n_r\times n_t}$ | `:47` |
| $\mathbf F=[\mathbf f_0..\mathbf f_{P-1}]$ | Tx codebook | $\mathbb C^{n_t\times P}$ | `:12` |
| $\mathbf W=[\mathbf w_0..\mathbf w_{Q-1}]$ | Rx codebook | $\mathbb C^{n_r\times Q}$ | `:13` |
| $\mathbf D_r=\mathrm{diag}(g_{r,n}e^{j\epsilon_{r,n}})$, $\mathbf D_t=\mathrm{diag}(g_{t,m}e^{j\epsilon_{t,m}})$ | per-antenna complex distortion | $16\times16$ diagonal | `:49-51` |
| $\mathbf Y$ | beamspace observation, rows = Rx beam $q$, cols = Tx beam $p$ | $\mathbb C^{Q\times P}$ | `:53-57` |
| $\tilde{\mathbf H}$ | antenna-domain observation `to_ant(Y)` | $\mathbb C^{16\times16}$ | `notebook_src.py:274-276` |

The row index is always AoA/Rx and the column index is always AoD/Tx. This holds for `Y`, `to_ant`, `angles_to_cells`, `heat_targets`, the NOMP atoms and the original GT image. **VERIFIED**: an on-grid path for beam (5,11) has $\arg\max|\mathbf Y|=(5,11)$, and `angles_to_cells(16)` gives (5.0, 11.0) (script 2, section K).

---

## 1. Steering vectors

**FACT** (TVT Eqs. 2-3, `a02_tvt.txt:240-249`; code `dldoa_dataset_generation.py:129-131`, `iabr2_sim.py:45-46`):
$$\mathbf a_t(\phi)=\tfrac{1}{\sqrt{n_t}}\big[e^{-j\pi k\cos\phi}\big]_{k=0}^{n_t-1},\qquad \mathbf a_r(\psi)=\tfrac{1}{\sqrt{n_r}}\big[e^{-j\pi k\cos\psi}\big]_{k=0}^{n_r-1}.$$

**DERIVATION.** For a ULA with spacing $d=\lambda/2$, the phase difference between elements for a plane wave at angle $\theta$ from the array axis is $2\pi d\cos\theta/\lambda=\pi\cos\theta$. The sign ($e^{-j\ldots}$) is a convention and is used consistently for Tx and Rx. With the $1/\sqrt{n}$ factor, $\lVert\mathbf a\rVert_2=1$. Angles are in radians throughout; the only degree-to-radian conversion is `np.deg2rad` for the impairment (`iabr2_sim.py:49`).

**DERIVATION (property with consequences).** $\mathbf a$ depends on $\theta$ only through $u=\pi\cos\theta\bmod 2\pi$. On $[0,\pi]$, $\cos$ is injective, so $\theta\mapsto u$ is one-to-one. However, $u=+\pi$ ($\theta=0$) and $u=-\pi$ ($\theta=\pi$) are the same point mod $2\pi$, so $\mathbf a(0)=\mathbf a(\pi)=[(-1)^k]/\sqrt n$. **VERIFIED**: $\max|\mathbf a(0)-\mathbf a(\pi)|=1.1\times10^{-14}$. Near end-fire, $\psi$ and $\pi-\psi$ are almost indistinguishable: 2 deg vs 178 deg differ by only $0.010$ of a 16-beam cell in wrapped $u$, and 5 deg vs 175 deg by $0.061$ of a cell (script 2, section L). See the CONCERN in section 10.

---

## 2. Channel matrix $\mathbf H$

**FACT** (TVT Eq. 1, `a02_tvt.txt:217-223`): $\mathbf H=\sqrt{n_tn_r}\sum_{l=1}^L\alpha_l\,\mathbf a_r(\psi_l)\mathbf a_t^H(\phi_l)$.
Code: `np.sqrt(NT*NR)*einsum('bl,bln,blm->bnm', alpha, a_r, conj(a_t))` (`iabr2_sim.py:47`). The Hermitian on $\mathbf a_t$ is implemented as `conj`, and the outer product has shape $(n_r,n_t)$, so dimensions are correct. Padded paths (NaN angles) get $\alpha=0$ (`:31`, `nan_to_num` `:44`), so they contribute nothing.

**DERIVATION (element form).** $H_{n,m}=\sum_l\alpha_l\,e^{-jnu_l}\,e^{+jmv_l}$, because $\sqrt{n_tn_r}\cdot\frac1{\sqrt{n_r}}\frac1{\sqrt{n_t}}=1$. This matches the decoder atom $a[n,m]=e^{-jnu+jmv}$ (`notebook_src.py:533-535`). The notebook's check "row phase step $=-\pi\cos\psi$, col phase step $=+\pi\cos\phi$" passes (`infer_dldoa_test.log` line 6-7).

**VERIFIED**: `simulate` H equals the original `generate_channel_v2` to $2.2\times10^{-16}$ (script 1, section D). The notebook's own check also passes: 4.78e-15 (log line 5).

**Distribution. FACT**: $\alpha_l=\sqrt{1/L}\,(x+jy)/\sqrt2$ with $x,y\sim\mathcal N(0,1)$, so $\alpha_l\sim\mathcal{CN}(0,1/L)$ and $E\sum_l|\alpha_l|^2=1$. **VERIFIED**: 0.970, 0.989 and 0.999 for L = 1, 3, 9 with 4000 samples each. Sorting by magnitude reorders the paths but does not change the distribution of the set. **FACT**: v2 draws the angles *before* $\alpha$ (`:34-35`), while the original draws $\alpha$ first (`dldoa_dataset_generation.py:549-556`). The distribution is the same, but the RNG streams differ. This only matters for bit-reproduction of original seeds, which v2 does not claim for its own banks.

### Toy example ($n_t=n_r=P=Q=4$, two on-grid paths), with arithmetic

Path 1: $\alpha_1=1$, $\cos\psi_1=0.5$ ($\psi_1=60^\circ$), $\cos\phi_1=-0.5$ ($\phi_1=120^\circ$). Path 2: $\alpha_2=0.5j$, $\cos\psi_2=0$ ($90^\circ$), $\cos\phi_2=1$ ($0^\circ$).
$$H_{n,m}=e^{-j\pi n/2}e^{-j\pi m/2}+0.5j\,e^{0}e^{j\pi m}=(-j)^{n+m}+0.5j(-1)^m.$$
Worked entries: $H_{0,0}=1+0.5j$; $H_{0,1}=-j-0.5j=-1.5j$; $H_{1,2}=(-j)^3+0.5j=j+0.5j=1.5j$; $H_{3,3}=(-j)^6-0.5j=-1-0.5j$.
**VERIFIED**: the full $4\times4$ `generate_channel_v2` output matches this closed form to $1.7\times10^{-15}$ (script 1, section B).

---

## 3. DFT codebooks $\mathbf F$, $\mathbf W$ and their unitarity

**FACT** (`dldoa_dataset_generation.py:155-168, 191-204`, identical to the official `tvt_data_generation_v3.py:74-117`):
$\cos\bar\phi_p=\frac1\pi\angle e^{j2\pi p/P}$, $\mathbf f_p=\mathbf a_t(\bar\phi_p)$; $\cos\bar\psi_q=\frac1\pi\angle e^{-j2\pi q/Q}$, $\mathbf w_q=\mathbf a_r(\bar\psi_q)$.

**DERIVATION.** $\angle(\cdot)\in(-\pi,\pi]$, so $\pi\cos\bar\phi_p\equiv 2\pi p/P \pmod{2\pi}$. Then
$$[\mathbf f_p]_k=\tfrac1{\sqrt{n_t}}e^{-jk\pi\cos\bar\phi_p}=\tfrac1{\sqrt{n_t}}e^{-j2\pi kp/P}\quad(\text{because }e^{-jk\cdot2\pi}=1),$$
$$[\mathbf w_q]_k=\tfrac1{\sqrt{n_r}}e^{-jk\pi\cos\bar\psi_q}=\tfrac1{\sqrt{n_r}}e^{+j2\pi kq/Q}.$$
So $\mathbf F=\mathbf\Phi^*/\sqrt N$ and $\mathbf W=\mathbf\Phi/\sqrt N$, where $\mathbf\Phi_{kp}=e^{j2\pi kp/N}$. Both are unitary when $n=P$ (square), and $\mathbf W=\mathbf F^*$.
**VERIFIED** (16x16): $\max|\mathbf F-e^{-j2\pi kp/16}/4|=2.9\times10^{-15}$, $\max|\mathbf F^H\mathbf F-\mathbf I|=2.4\times10^{-15}$, $\max|\mathbf F\mathbf F^H-\mathbf I|=2.4\times10^{-15}$, $\max|\mathbf W^H\mathbf W-\mathbf I|=2.3\times10^{-15}$, $\max|\mathbf W-\mathbf F^*|=4.4\times10^{-15}$.
Beam grids: $\cos\bar\phi_p=[0,.125,\dots,.875,1,-.875,\dots,-.125]$ and $\cos\bar\psi_q=[0,-.125,\dots,-.875,-1,.875,\dots,.125]$.

**FACT (edge case).** At $p=P/2$, `angle(exp(j*pi))` returns $+\pi$, so $\cos=1$ and $\bar\phi=0$. At $q=Q/2$, `angle(exp(-j*pi))` returns $-\pi$ (the imaginary part is $-1.2\times10^{-16}$), so $\cos=-1$ and $\bar\psi=\pi$. The two are equivalent because $\mathbf a(0)=\mathbf a(\pi)$, so this does not affect unitarity.
**CONCERN (Low).** Unitarity, and hence the exactness of `to_ant`, holds only for $n_t=P$ and $n_r=Q$. That is always true in v2 (16/16). It would *not* hold for the original generator's $P=32,n_t=16$ training case. v2 does not use that case.

---

## 4. Observation $\mathbf Y=\mathbf W^H\mathbf H\mathbf F+\mathbf Z$ (ideal codebooks)

**FACT** (TVT Eqs. 4-5, `a02_tvt.txt:273-313`): $y_{q,p}=\sqrt\rho\,\mathbf w_q^H\mathbf H\mathbf f_p s+\mathbf w_q^H\mathbf n$ with $\rho=1$, $s=1$. Code: `einsum('bnq,bnm,bmp->bqp', conj(W), H, F)` (`iabr2_sim.py:53`), which gives $G_{qp}=\sum_{n,m}W^*_{nq}H_{nm}F_{mp}=[\mathbf W^H\mathbf H\mathbf F]_{qp}$. The dimensions work out as $(Q\times n_r)(n_r\times n_t)(n_t\times P)=Q\times P$.

**DERIVATION (Dirichlet form, check of TVT Eq. 7).**
$\mathbf w_q^H\mathbf a_r(\psi)=\frac1{n_r}\sum_k e^{-j\pi k(\cos\psi-\cos\bar\psi_q)}=\frac1{n_r}\frac{1-e^{-j\pi n_r(\cos\psi-\cos\bar\psi_q)}}{1-e^{-j\pi(\cos\psi-\cos\bar\psi_q)}}$ and
$\mathbf a_t^H(\phi)\mathbf f_p=\frac1{n_t}\frac{1-e^{j\pi n_t(\cos\phi-\cos\bar\phi_p)}}{1-e^{j\pi(\cos\phi-\cos\bar\phi_p)}}$.
Multiplying by $\sqrt{n_tn_r}\alpha_l$ gives $A_l=\alpha_l/\sqrt{n_tn_r}$ times the two ratios. This is exactly TVT Eq. 7 (`a02_tvt.txt:329-338`). With $\omega_q=-\pi\cos\bar\psi_q$ and $\omega_{\psi_l}=-\pi\cos\psi_l$ one gets $\omega_q-\omega_{\psi_l}=\pi(\cos\psi_l-\cos\bar\psi_q)$, which reproduces Eq. 8. The paper's equations are internally consistent with the code.

**DERIVATION (peak location).** $|\mathbf w_q^H\mathbf a_r|$ peaks where $\cos\bar\psi_q=\cos\psi$, i.e. $2\pi q/Q\equiv-\pi\cos\psi$, so $q^\star=Q\,\mathcal W_{2\pi}(-\pi\cos\psi)/2\pi$. Likewise $p^\star=P\,\mathcal W_{2\pi}(\pi\cos\phi)/2\pi$. These are exactly `angles_to_cells` (`iabr2_sim.py:71-73`) and the v1 function.

**Toy (continued, 4x4). DERIVATION + VERIFIED.** Path 1 has $\cos\psi=0.5=\cos\bar\psi_3$ and $\cos\phi=-0.5=\cos\bar\phi_3$, so $Y_{3,3}=\sqrt{16}\cdot1\cdot1\cdot1=4$. Path 2 has $\cos\psi=0=\cos\bar\psi_0$ and $\cos\phi=1=\cos\bar\phi_2$, so $Y_{0,2}=4\cdot0.5j=2j$. All other entries are 0. The computed $\mathbf Y$ has exactly $Y_{3,3}=4$, $Y_{0,2}=2j$ and $|{\rm others}|<10^{-15}$. `angles_to_cells(n=4)` gives $q=[3, 4^-]$, $p=[3,2]$. Here $4^-$ is $4-\varepsilon$, because $-\pi\cos(\pi/2)=-1.9\times10^{-16}$ wraps to just below $2\pi$. **FACT**: the continuous $q$ can therefore equal $G$ rather than 0. `heat_targets` handles this with `% G` (`:96`). Consumers of the continuous value must also wrap.

---

## 5. Per-antenna impairments $\mathbf D_r$, $\mathbf D_t$ and how they enter $\mathbf W^H\mathbf H\mathbf F$

**FACT (v2 model, `iabr2_sim.py:48-53`).** Per sample, independently for Tx and Rx and independently per antenna:
$\epsilon\sim\mathcal U(-\delta_{\max},\delta_{\max})$ (deg converted to rad) and $g=10^{x\gamma_{\max}/20}$ with $x\sim\mathcal U(-1,1)$, so the gain is uniform *in dB* on $[-\gamma_{\max},\gamma_{\max}]$. Then $\mathbf W=\mathbf D_r\mathbf W_{\rm id}$ and $\mathbf F=\mathbf D_t\mathbf F_{\rm id}$ (row scaling; `D[:, :, None]*W_IDEAL`). The same distortion applies to every beam (a static hardware error).

**DERIVATION.**
$$\mathbf Y=(\mathbf D_r\mathbf W_{\rm id})^H\mathbf H(\mathbf D_t\mathbf F_{\rm id})+\mathbf Z=\mathbf W_{\rm id}^H\,\underbrace{\mathbf D_r^{*}\mathbf H\mathbf D_t}_{\tilde{\mathbf H}_{\rm sig}}\,\mathbf F_{\rm id}+\mathbf Z,$$
because $\mathbf D_r$ is diagonal, so $\mathbf D_r^H=\mathbf D_r^*$. In elements, $\tilde H_{n,m}=g_{r,n}e^{-j\epsilon_{r,n}}\,H_{n,m}\,g_{t,m}e^{+j\epsilon_{t,m}}$. **The Rx phase enters with a minus sign and the Tx phase with a plus sign.** This matches the docstrings (`:84`) and the target signs (`:86`: `-angle(D_r)`, `+angle(D_t)`).

**Toy (4 antennas).** $\epsilon_r=[0,2,-1,3]^\circ$ and $\epsilon_t=[1,-2,0,4]^\circ$. The two delta peaks leak: $Y_{3,3}=3.9954-0.0172j$ (was 4) and $Y_{0,2}=0.0093+1.9974j$ (was $2j$), with off-peak entries up to $|{\approx}0.11|$. **VERIFIED**: $\max|\text{to\_ant}(\mathbf Y)-\mathbf D_r^*\mathbf H\mathbf D_t|=2.0\times10^{-15}$ (script 1, section C).

**Hand arithmetic of $Y_{3,3}$ (DERIVATION, then VERIFIED by `a02_v5_independent.py` A).** With $n_t=n_r=4$: $W_{nq}=e^{+j2\pi nq/4}/2$, $F_{mp}=e^{-j2\pi mp/4}/2$, so $Y_{qp}=\tfrac14\sum_{n,m}e^{-j2\pi nq/4}\tilde H_{nm}e^{-j2\pi mp/4}$. Path 1 has $\tilde H^{(1)}_{nm}=\alpha_1r_nt_m(-j)^n(-j)^m$ with $r_n=e^{-j\epsilon_{r,n}}$, $t_m=e^{+j\epsilon_{t,m}}$. For $q=p=3$ the DFT kernel is $e^{-j2\pi3n/4}=j^n$, and $(-j)^nj^n=1$, hence
$$Y^{(1)}_{3,3}=\tfrac{\alpha_1}{4}\Big(\sum_nr_n\Big)\Big(\sum_mt_m\Big).$$
Numbers: $\sum_nr_n=(1+\cos2^\circ+\cos1^\circ+\cos3^\circ)-j(\sin2^\circ-\sin1^\circ+\sin3^\circ)=3.997868-0.069783j$; $\sum_mt_m=(\cos1^\circ+\cos2^\circ+1+\cos4^\circ)+j(\sin1^\circ-\sin2^\circ+\sin4^\circ)=3.996803+0.052309j$. Product $/4=3.995585-0.017446j$. The full simulation gives $3.995400-0.017187j$; the difference $(-1.8+2.6j)\times10^{-4}$ is the leakage of path 2 into beam (3,3). So the mechanism is explicit: $\sum r_n$ and $\sum t_m$ are the "coherent gain" of the distorted arrays. The mean phase of $\mathbf r$ rotates $Y_{3,3}$ (absorbed into $\angle\alpha$, hence unidentifiable), and the ramp moves energy to neighbouring beams (hence identifiable only through angle, which is where the angle offset of section 8 comes from).

**Comparison with the source models.**
- **FACT**: the original generator's `error_deg` path computes `F[:,p] = f_ideal * exp(j*phase_error)` (`tvt_data_generation_v3.py:79-91,105-117`), i.e. $\mathbf F=\mathrm{diag}(e^{j\delta})\mathbf F_{\rm id}$. This is the same structure as v2. **VERIFIED**: max deviation $2.8\times10^{-17}$.
- **FACT**: JSC Sec. 2.2 (`a02_jsc.txt:158-165, 211-217`) specifies $\delta_i\sim\mathcal U[-\delta_{\max},\delta_{\max}]$, independent across elements, and applied to "both the Tx and Rx" arrays. TVT Sec. IV-D says the same. v2's `T2_d{1,2,5}` banks (`notebook_src.py:372`) implement this phase-only model.
- **FACT**: neither paper models gain error. JSC lists amplitude errors as future work (`a02_jsc.txt` ~l.1052). The gain part of v2 (training $\gamma_{\max}\sim\mathcal U(0,3)$ dB, `tune_g3`, `tune_L6`, and the v1 `gain_*` banks) is an extension by the project.
- **FACT**: in the original `error_deg` path the error comes from the global `np.random`, not from the seeded `rng` (`tvt_data_generation_v3.py:81,107`). The original impaired validation sets are therefore not reproducible from `seed`. v2 draws from its seeded `rng`, so its banks are reproducible (VERIFIED in section 12).

**CONCERN (Low; affects the gain banks and gain training only). Noise is not shaped by the Rx gain error.** Physically, $y=\mathbf w^H(\mathbf H\mathbf f+\mathbf n)$, so the noise variance per measurement is $\sigma^2\lVert\mathbf D_r\mathbf w_{\rm id,q}\rVert^2=\sigma^2\frac1{n_r}\sum_n g_{r,n}^2$. The code adds $\mathbf Z$ with a fixed $\sigma^2$ after the distorted combiner (`:55-57`).
- With phase-only errors ($g\equiv1$) this is exact: $\mathbf w_q^H\mathbf w_{q'}=\delta_{qq'}$ still holds.
- With gain errors, the effective SNR is higher than nominal by $E[g_r^2]E[g_t^2]$ in the code, versus $E[g_t^2]$ if the gain error sat before the noise source. **DERIVATION**: $E[g^2]=\frac{\sinh(\gamma\ln10/10)}{\gamma\ln10/10}$, which gives +0.08/+0.30/+0.68/+1.20 dB (both sides) for $\gamma=1/2/3/4$ dB. **VERIFIED**: empirical signal power 1.007/1.092/1.170/1.309.
- Where the gain error sits relative to the dominant noise source is a modeling choice. Insufficient evidence for which is intended. The shift is small (at most 1.2 dB, for the out-of-distribution `gain_g4`).
- **Second-order effect the simulator also omits: noise colouring (DERIVATION + VERIFIED, `a02_v5_independent.py` B).** If the noise physically enters before the combiner, the combiner outputs have covariance $\sigma^2\mathbf W^H\mathbf W=\sigma^2\mathbf W_{\rm id}^H\mathbf D_r^H\mathbf D_r\mathbf W_{\rm id}$, i.e. a circulant matrix generated by $g_{r,n}^2$. It equals $\mathbf I$ only when $g_r\equiv1$. For one draw of Rx gains: $\gamma_{\max}=1$ dB gives max off-diagonal $0.055$, and $\gamma_{\max}=3$ dB gives max off-diagonal $0.155$ with mean diagonal $1.150$. The simulator instead adds white noise $\mathbf Z$ with fixed variance, so its noise is uncorrelated across beams. Effect size is small (off-diagonals of at most 0.16 of the diagonal at the training maximum) and affects only gain-impaired data; phase-only banks (`T2_*`) are exact.

---

## 6. Antenna-domain transform `to_ant`: derivation of $\tilde{\mathbf H}=\mathbf D_r^*\mathbf H\mathbf D_t$

**FACT** (`notebook_src.py:274-276`): `einsum('nq,bqp,mp->bnm', W_IDEAL, Y, conj(F_IDEAL))`, which computes $\tilde{\mathbf H}=\mathbf W_{\rm id}\mathbf Y\mathbf F_{\rm id}^H$. The model's `front()` does the same thing in torch: `Wi @ Y @ Fi.conj().T` (`:499`).

**DERIVATION.** With $\mathbf W_{\rm id}\mathbf W_{\rm id}^H=\mathbf I$ and $\mathbf F_{\rm id}\mathbf F_{\rm id}^H=\mathbf I$ (square unitary, section 3):
$$\mathbf W_{\rm id}\mathbf Y\mathbf F_{\rm id}^H=\mathbf W_{\rm id}\mathbf W_{\rm id}^H\,\mathbf D_r^*\mathbf H\mathbf D_t\,\mathbf F_{\rm id}\mathbf F_{\rm id}^H+\mathbf W_{\rm id}\mathbf Z\mathbf F_{\rm id}^H=\mathbf D_r^*\mathbf H\mathbf D_t+\mathbf Z',$$
$$\mathbf Z'=\mathbf W_{\rm id}\mathbf Z\mathbf F_{\rm id}^H,\quad \mathrm{vec}(\mathbf Z')=(\mathbf F_{\rm id}^*\otimes\mathbf W_{\rm id})\mathrm{vec}(\mathbf Z).$$
A Kronecker product of unitaries is unitary, so $\mathbf Z'$ is i.i.d. $\mathcal{CN}(0,\sigma^2)$, the same distribution as $\mathbf Z$.
**VERIFIED**: $\max|\text{to\_ant}(\mathbf Y)-\mathbf D_r^*\mathbf H\mathbf D_t|=3.6\times10^{-15}$ at 400 dB SNR with $\delta_{\max}=5^\circ$, $\gamma_{\max}=2$ dB. The empirical covariance of $\mathbf Z'$ over 20000 draws has a diagonal of 0.99-1.01 and off-diagonal entries below 0.013 (script 1, section D).
**Interpretation (DERIVATION).** Because $\mathbf W=\mathbf F^*$, `to_ant` is a 2D inverse DFT up to sign and index conventions. The antenna-domain matrix is a sum of 2D complex sinusoids $\alpha_l e^{-jnu_l+jmv_l}$, multiplied element-wise by the outer product $(g_re^{-j\epsilon_r})(g_te^{j\epsilon_t})^T$. This is a rank-1 multiplicative mask, so the impairment is *separable*.

---

## 7. Noise scaling from SNR (dB)

**FACT** (`iabr2_sim.py:55-56`; original `dldoa_dataset_generation.py:236-238`): $\sigma^2=10^{-\mathrm{SNR}/10}$, and $\mathbf Z=\sqrt{\sigma^2/2}\,(\mathbf N_1+j\mathbf N_2)$ with $\mathbf N_i\sim\mathcal N(0,1)^{Q\times P}$. So $E|Z_{qp}|^2=\sigma^2$, and the real and imaginary parts each have variance $\sigma^2/2$. This is the paper's definition "SNR $=1/\sigma_n^2$ with $\rho=1$" (TVT IV-B, `a02_tvt.txt:684-686`). The logarithm is base 10 in power (divide by 10), which is correct for a power ratio.

**DERIVATION (what the SNR actually measures).** $\lVert\mathbf W^H\mathbf H\mathbf F\rVert_F^2=\lVert\mathbf H\rVert_F^2$ (unitary invariance), and
$E\lVert\mathbf H\rVert_F^2=n_tn_r\sum_lE|\alpha_l|^2\lVert\mathbf a_r\mathbf a_t^H\rVert_F^2+(\text{cross terms, zero mean})=n_tn_r\cdot1$. So the average signal power **per beamspace entry** is $E\lVert\mathbf H\rVert_F^2/(QP)=1$, and the nominal SNR equals the average per-entry SNR.
**VERIFIED**: per-entry signal power 0.970/0.990/0.999 (L = 1/3/9) and noise power 1.001 at 0 dB. In the official bank, per-SNR $E|Y|^2$ = 11.01, 4.18, 2.01, 1.33, 1.13, 1.04, 0.995, 0.988 against theory $1+10^{-\mathrm{SNR}/10}$ = 11.00, 4.16, 2.00, 1.32, 1.10, 1.03, 1.01, 1.003.
**FACT (property of the paper's model).** The per-*sample* SNR is random because $\sum_l|\alpha_l|^2\sim\mathrm{Gamma}(L,1/L)$. The std of per-sample signal power is 0.96 at L=1 (exponential), 0.58 at L=3 and 0.34 at L=9 (VERIFIED). The nominal SNR is an ensemble average.
**Toy.** At SNR = 10 dB, $\sigma^2=0.1$ and the real/imag std is $\sqrt{0.05}=0.224$. In the 16-antenna check the residual $\mathbf Y-\mathbf W_{\rm id}^H\mathbf D_r^*\mathbf H\mathbf D_t\mathbf F_{\rm id}$ has RMS 0.300 (theory 0.316).
**FACT (harmless).** The original training generator calls `generate_noise(1.0, SNR, P, Q)` while the signature is `(var_alpha, SNR, Q, P)` (`dldoa_dataset_generation.py:460` vs `:208`). The argument order is swapped, but this is harmless because $P=Q$.

---

## 8. Identifiable subspace: why the mean and the linear phase ramp are unidentifiable

**DERIVATION (invariance group).** Write $\tilde H_{n,m}=r_n\,t_m\sum_l\alpha_le^{-jnu_l+jmv_l}$ with $r_n=g_{r,n}e^{-j\epsilon_{r,n}}$ and $t_m=g_{t,m}e^{j\epsilon_{t,m}}$. For any real $c_1,c_2,a,b$ and $\kappa_1,\kappa_2>0$, the following transformation leaves every $\tilde H_{n,m}$ unchanged:
1. $\epsilon_{r,n}\to\epsilon_{r,n}+c_1$ and $\alpha_l\to\alpha_le^{jc_1}$ (common Rx phase is absorbed in the unknown $\angle\alpha$);
2. $\epsilon_{t,m}\to\epsilon_{t,m}+c_2$ and $\alpha_l\to\alpha_le^{-jc_2}$;
3. $\ln g_{r,n}\to\ln g_{r,n}+\ln\kappa_1$ and $|\alpha_l|\to|\alpha_l|/\kappa_1$ (common gain is absorbed in the unknown $|\alpha|$);
4. same as 3 for Tx with $\kappa_2$;
5. $\epsilon_{r,n}\to\epsilon_{r,n}+an$ and $u_l\to u_l-a$ for **all** $l$, because $e^{-j(\epsilon+an)}e^{-jnu}=e^{-j\epsilon}e^{-jn(u+a)}$ in the notation where $u$ absorbs the ramp;
6. $\epsilon_{t,m}\to\epsilon_{t,m}+bm$ and $v_l\to v_l-b$ for all $l$.

These are 6 real directions. A linear ramp in **log-gain** would require $|e^{\beta n}|\neq1$, i.e. a complex frequency, which the unit-modulus steering model cannot absorb. So a gain ramp *is* identifiable. That is why the code removes only the mean for gain (`iabr2_sim.py:88`) but the mean *and* the ramp for phase (`:76-80, 86`).

**VERIFIED (Jacobian rank).** Take the real parametrization $(\epsilon_r,\epsilon_t,\ln g_r,\ln g_t)\in\mathbb R^{64}$ plus $(u_l,v_l,\Re\alpha_l,\Im\alpha_l)$, and the noiseless $\tilde{\mathbf H}$ as 512 real outputs. The numerical Jacobian has **null-space dimension exactly 6** for L = 1, 2, 3 and 6 (ranks 62/66/70/82 out of 68/72/76/88 parameters). The predicted directions (phase mean, Rx ramp, Tx ramp, gain mean) lie 100% in the null space with $|J d|\sim10^{-9}$. The gain-ramp direction does not ($|Jd|\approx8$-13) (script 2, section E). For L=1 the rank-1 structure does *not* make $\mathbf D_r$ arbitrary: $\mathbf D_r^*\mathbf a_r(\psi)$ determines $\mathbf D_r$ up to exactly the ramp and the scalar.
**VERIFIED (exact equivalence).** A pure Rx ramp $a=0.05$ rad/element and a Tx ramp $b=-0.03$ produce the same $\mathbf Y$ as no impairment with $\cos\psi\to\cos\psi+a/\pi$ and $\cos\phi\to\cos\phi+b/\pi$ (max diff $2.6\times10^{-14}$). For that sample the equivalent shifts were -1.36/-0.94/-0.93 deg (AoA) and +1.94/+0.95/+1.79 deg (AoD).

**FACT/VERIFIED (`project_impairment`).** With $n_c=n-7.5$ (so $\mathbf 1^Tn_c=0$), `x - mean` followed by `- (x·n_c)/(n_c·n_c) n_c` equals $(\mathbf I-\mathbf B\mathbf B^+)\mathbf x$ with $\mathbf B=[\mathbf 1,n_c]$. The maximum deviation from the orthogonal projector is $1.1\times10^{-16}$, the operator is idempotent, and its rank is 14 (= 16 - 2).
**FACT (targets, `:83-89`).** Phase target is $[\mathcal P(-\angle D_r),\mathcal P(+\angle D_t)]$ in radians. Log-gain target is the natural log $\ln|D|$ minus its mean (units: nepers, $\ln g=x\gamma\ln10/20$, max 0.345 at 3 dB). The signs match section 5. The IABC correction $c=\exp(-\hat\ell-j\hat\varphi)$ multiplying $\tilde{\mathbf H}$ (`notebook_src.py:502-503`) cancels $g_re^{-j\epsilon_r}$ on rows and $g_te^{+j\epsilon_t}$ on columns. The sign chain is consistent. `np.angle` wrap is not an issue: $|\epsilon|\le 8^\circ$ in training and at most 15 deg in OOD banks. **VERIFIED** on a 4096 batch: row sums of targets are below $2.4\times10^{-7}$, ramp residual below $10^{-7}$, and 19.3% of samples are clean (config 20%).

**CONCERN (Medium; physics of the dataset, affects every estimator equally). Irreducible angle offset from the unidentifiable phase ramp.** The angle *labels* are the true physical angles, which is correct. But the data are exactly consistent with angles shifted by the ramp component of the phase errors (items 5-6 above). Hence
$$\delta(\cos\psi)=\hat a/\pi,\qquad \hat a=\frac{\sum_n n_c\epsilon_{r,n}}{\sum_n n_c^2},\qquad \sum n_c^2=340,$$
and $\mathrm{std}(\hat a)=\frac{\delta_{\max}}{\sqrt3\sqrt{340}}$ for i.i.d. uniform errors. Numbers:
- Broadside AoA std: 0.010, 0.020, 0.050, 0.080 deg for $\delta_{\max}=1,2,5,8^\circ$.
- Because $d\psi=d(\cos\psi)/\sin\psi$, it grows toward end-fire. Monte Carlo with uniform $\psi$ gives **P(|offset| > 1 deg) = 0.38% (2 deg), 1.44% (5 deg), 2.7% (8 deg)**. An additional 0.7%/1.1%/1.4% of paths are pushed beyond $|\cos|=1$ and alias (script 2, section H).

So for `T2_d5` there is an estimator-independent ceiling of roughly 98.5% on per-angle detection (1 deg threshold), before noise. The reported Pd numbers should be read against this ceiling. INFERENCE (weak): the paper's own d=5 Pd saturates at 0.93-0.94, i.e. about 5-6 points below 100%, while this ceiling accounts for only about 1.5 of them. The rest is not explained by this mechanism (other candidate causes were not analysed here: insufficient evidence). Also note that the ceiling is computed for the v2 phase model (independent uniform errors on both arrays); it applies to any bank built that way, including `T2_d5`.

---

## 9. Heatmap target and `angles_to_cells`

**FACT** (`iabr2_sim.py:71-73, 92-104`; `CFG grid=32, heat_sigma=1.0`, `notebook_src.py:84`):
$$q=G\,\frac{\mathcal W_{2\pi}(-\pi\cos\psi)}{2\pi},\quad p=G\,\frac{\mathcal W_{2\pi}(\pi\cos\phi)}{2\pi},\quad \hat q=\mathrm{round}(q)\bmod G,$$
$$h_{ij}=\max_l\exp\!\Big(-\frac{d_G(i,\hat q_l)^2+d_G(j,\hat p_l)^2}{2\sigma^2}\Big),\qquad d_G(i,k)=\min(|i-k|,G-|i-k|).$$
Here $G=32$, $\sigma=1$ cell, and $d_G$ is the circular distance.
- **DERIVATION.** This is the section 4 peak map on a grid twice as fine as the beams ($G=2Q$). Cell $2k$ is beam $k$ and odd cells are half-beam offsets. This fits `pixel_shuffle(r=2)` (`notebook_src.py:516`): output $(2i+a,2j+b)$ comes from feature $(i,j)$, so $a=b=0$ sits on the beam centre. The decoder inverse is $u=-2\pi\hat q/G$, $v=2\pi\hat p/G$ (`heat_peaks` `:594`). Then $\psi=\arccos(\mathcal W_{(-\pi,\pi]}(u)/\pi)$ (`:623-625`), which is consistent.
- **Why wrap-around is correct (DERIVATION).** $\mathbf Y$ is periodic in beam index (DFT), so a circular Gaussian is the right target geometry. The grid edge $q=0\equiv32$ corresponds to broadside ($u=0$). **VERIFIED**: for $\psi=\pi/2-0.02$, $q=31.68$ rounds to 0, and the row profile over rows 30,31,0,1,2 is 0.135, 0.607, 1, 0.607, 0.135. End-fire ($u=\pm\pi$) sits in the middle row $q=16$.
- **Toy (VERIFIED).** The two paths of the 4x4 toy on the 32-grid: path 1 at $(24,24)$ and path 2 at $q=32\equiv0$, $p=16$. The heatmap has exactly two cells equal to 1, at (0,16) and (24,24). Its sum is 12.566, which equals $2\cdot(\sum_ke^{-k^2/2})^2\approx2\cdot2\pi\sigma^2=4\pi$.
- **NaN padding (FACT).** Padded paths map to $\psi=0$ via `nan_to_num`, but their row factor is multiplied by `~isnan(psi)` (`:100`), so they contribute exactly zero.
- **CONCERN (Low; by design). The target is quantized.** The peak sits on the *rounded* cell, so sub-cell information is discarded. The maximum error is 0.5 cell, i.e. $\Delta\cos=1/32$, which is 1.79 deg at broadside. That is larger than the 1-deg Pd threshold. v2's physics decoder (NOMP) re-estimates off-grid, so this matters only for the "network only (parabolic)" ablation: a target centred on the rounded cell cannot teach sub-cell position.
- **FACT/VERIFIED. `max` over paths, not sum.** When two paths round to the same 32-cell they merge into one positive. This happens in 1.8% of training samples (0.36% at L=3, 5.1% at L=9; 20000 samples). The cause is that the $\pi/6$ separation is enforced in angle space, not in $(u,v)$ (section 10).

---

## 10. Angle sampler and separation

**FACT** (`dldoa_dataset_generation.py:46-103`, identical to official `tvt_data_generation_v3.py:10-55`): sequential rejection sampling of $L$ points $(\phi,\psi)\sim\mathcal U[0,\pi]^2$ with Euclidean distance at least $\pi/6$ in *angle* space, capped at 10000 attempts. **VERIFIED**: 0/3000 failures at L=9. All v2 banks have a minimum pair distance of at least 0.5236 (= $\pi/6$).
**FACT (paper vs. code).** TVT IV-B says AoA/AoD are uniform in $[0,\pi]$ (`a02_tvt.txt:701-703`), while Sec. II says $[0,2\pi]$ (`:233`). The two give the same $\cos$ distribution, so they are equivalent for the data. The paper text never mentions the $\pi/6$ minimum separation; it exists only in the code. The paper also says "SNR between -10 and 25 dB, L = 1 to 10" for training (`:691-692`). The code (`randint(-15,25)`, `randint(1,10)`) gives SNR $\in\{-15..24\}$ and L $\in\{1..9\}$, and v2 follows the *code* (`CFG train_L=(1,9), train_snr=(-15,24)`). This is a paper-text inconsistency, not a v2 bug. **FACT (minor):** training SNR is integer-valued (`rng.integers(-15, 25)` gives $\{-15,\dots,24\}$, `iabr2_sim.py:109`), so the test SNR of 25 dB (official bank and `T2_*`) lies 1 dB above the largest training SNR, exactly as in the original generator (`np.random.randint(-15, 25)`, `dldoa_dataset_generation.py:422`). Similarly the tail banks at -25/-20/30/35 dB are out of the training range by design.

**CONCERN (Medium; inherent to the base paper's dataset and metric, affects every method). End-fire aliasing and non-uniform angular density.**
1. $\psi$ uniform in angle gives $u=\pi\cos\psi$ an arcsine density that piles up at $\pm\pi$. 32.2% of paths fall within one beam cell of end-fire (VERIFIED).
2. $\psi\approx0$ and $\psi\approx\pi$ are adjacent in $u$ (section 1), so a tiny $u$ error across $\pm\pi$ flips the estimate by about 180 deg. **VERIFIED example**: for a true $\psi=3^\circ$, the *perfect* original GT image, decoded at its nearest pixel by the verbatim `peaks_to_angles`, returns $\psi=180.000^\circ$. Pixel 128 is exactly $\omega=\pi$, and $\arccos(-1)=\pi$ (script 4, section P).
3. The $\pi/6$ angle-space separation does not prevent two paths from nearly coinciding in beamspace (for example $\psi_1\approx5^\circ$, $\psi_2\approx175^\circ$ with similar $\phi$). **VERIFIED**: 4.0% of L=3 scenes have a pair within 1 beam cell in both wrapped $u$ and $v$, and 1.1% within 0.5 cell.

These are properties of the published protocol (they are also in the official seed-42 set) and are not errors in v2. They put a floor on achievable Pd and should be disclosed when interpreting Pd.

---

## 11. Gaussian ground truth of the original generator (used for the U-Net and by the evaluator; not used in v2 training)

**FACT, paper (TVT Eq. 11, `a02_tvt.txt:504-534`):** $X_{m,n}=\frac1{2\pi\sigma_1\sigma_2}\sum_l\exp\!\big(-\frac{(\omega_m-\tilde\omega_{\psi_l})^2}{2\sigma_1^2}-\frac{(\omega_n-\tilde\omega_{\phi_l})^2}{2\sigma_2^2}\big)$ with $\omega_m=2\pi m/M$ and $\sigma_1=\sigma_2=0.07$. (The minus sign is lost in the PDF text extraction.)
**FACT, code (`dldoa_dataset_generation.py:313-372` = official `tvt_data_generation_v3.py:269-335`):** grid $\omega_k=-3\sigma+k\frac{2\pi+6\sigma}{256}$, $k=0..255$ (`linspace(..., endpoint=False)`). Rows are AoA ($\tilde\omega_\psi=\mathcal W_{2\pi}(-\pi\cos\psi)$) and columns are AoD ($\tilde\omega_\phi=\mathcal W_{2\pi}(\pi\cos\phi)$) via `meshgrid(p,q)`. The Gaussians are **summed**, with peak $1/(2\pi\sigma^2)=32.48$ per path.
- **VERIFIED.** Grid spacing is 0.026184 rad, versus the paper's $2\pi/256=0.024544$. $\sigma$ is 2.673 pixels. The grid starts at -0.21 and ends at 6.467.
- **CONCERN (Low; paper-vs-code, inherited).** Eq. 11 does not describe the implemented grid (margin, spacing). The evaluator inverts the *implemented* grid exactly: `freqs = -margin + (pix/256)(2π+2·margin)` (`TVT_Blob_Inference.py:108-118`). So code and evaluator agree with each other, and only the paper text is off. **VERIFIED**: nearest-pixel decoding of three GT peaks gives (60.006, 119.994), (94.771, 39.948) and (180.000!, 169.524) against truths (60,120), (95,40), (3,170). The third shows the end-fire flip from section 10.
- **CONCERN (Low; inherited).** The GT is not periodic although $\mathbf Y$ is. A path at $\tilde\omega=0.05$ is drawn at column 10 (value 32.47), but the in-grid alias at $2\pi+0.05$ (column 250) is 0 (VERIFIED). The network has to learn an arbitrary copy selection near $\omega\in[0,3\sigma)$ and $[2\pi-3\sigma,2\pi)$. The evaluator's `wrap_2pi_to_minus_pi` maps either copy to the same angle, so this does not bias the metric.

---

## 12. Up-sample / down-sample index maps

**FACT** (`iabr2_sim.py:14-25`): `UP_SRC = zoom(arange(256).reshape(16,16), 4, order=0).ravel()` is a gather map from 64x64 output positions to 16x16 source indices $s=16q+p$. `DOWN_IDX[s]` is the first output position whose source is $s$. The original generator applies `scipy.ndimage.zoom(Y[:,:,c], 4, order=0)` to each real/imag channel (`dldoa_dataset_generation.py:482-485`).
**DERIVATION.** SciPy's default `grid_mode=False` aligns corners: output coordinate $o$ maps to input $o\cdot\frac{16-1}{64-1}=o\cdot\frac{5}{21}$, and `order=0` rounds to the nearest integer. No ties occur because $5o/21=k+\tfrac12$ would need $o=21(2k+1)/10\notin\mathbb Z$. The map is separable: source row index = round$(5o/21)$.
**VERIFIED.** Row/column block sizes are **[3,4,4,4,4,5,4,4,4,4,5,4,4,4,4,3]**, so the blocks are *not* uniform 4x4. All 256 sources are present, the map is separable, and `upsample64` equals the per-channel `scipy.ndimage.zoom` with max diff 0.0. `downsample16(upsample64(x))==x` exactly. `upsample64(downsample16(d))==d` exactly over **all 8000** official samples. `DOWN_IDX[:16]=[0,3,7,11,15,19,24,...]`.
**FACT/INFERENCE.** TVT describes "nearest-neighbor with $\beta=4$" (`a02_tvt.txt:386-395`), which one could read as uniform 4x4 replication (`np.kron`). The official code uses the non-uniform corner-aligned map. v2 reproduces the *code* bit-exactly, so the U-Net sees the same input distribution it was trained on. Correct.

---

## 13. Bank provenance and bitwise regeneration

- **VERIFIED (provenance of `eval_bank.npz`).** The file contains `data (8000,64,64,2) f32`, `feat (8000,2,3)` = $[\psi;\phi]$, `meta (8000,4)` = [L, SNR, P, nt], sigma=0.07, M=256, seed=42. L=3, P=nt=16, 1000 per SNR, and the SNR blocks start at indices 0, 1000, ..., 7000. **Regenerating with the official `DL_DOA/src/tvt_data_generation_v3.validation_data_generator(conditions L=3 × SNR −10..25 step 5 × P=16 × nt=16, 1000/condition, seed=42)` reproduces `data` and `feat` bit-exactly (max diff 0.0 over all 8000) and `meta` exactly** (script 3, 180 s). The project's `dldoa_dataset_generation.validation_data_generator` with the same arguments also matches (first 60 checked, diff 0). **INFERENCE (strong):** the file was created by `D:\ai_ml_project\scripts\generate_frozen_banks.py::generate_eval_bank`, which calls exactly this and saves exactly these keys (`data, feat, meta, sigma, M, seed`). `PROJECT_STATUS.md:38` states the same. The stored file is 19.2 MB, not the ~256 MB in the script docstring. That is consistent with compression of 4x-replicated data (INFERENCE).
- **VERIFIED (v2 banks).** Re-running the `make_bank` logic (`notebook_src.py:356-368`) with the declared specs (`:372-378`: T2_d{1,2,5} seeds 9101/9102/9105, 1000 per SNR x 8 SNR; tune_clean 9901, tune_d5 9902, tune_g3 9903, tune_L6 9904) reproduces `Y16, psi, phi, L, snr` of all 7 files in `data\banks` with max diff 0.0. The stored `phase_deg/gain_db/seed` attributes match the specs. No first-path angle pair is shared between any v2 bank and the official bank.
- **VERIFIED (independent, data-only end-to-end convention test; `a02_v5_independent.py` C).** Without importing any project code, I took the stored $(\psi,\phi)$ labels and stored $\mathbf Y$ of each bank, formed $\tilde{\mathbf H}=\mathbf W_{\rm id}\mathbf Y\mathbf F_{\rm id}^H$ with my own DFT matrices, fit the complex gains by least squares on atoms $e^{-jn\pi\cos\psi_l+jm\pi\cos\phi_l}$, and compared the residual power per degree of freedom with the nominal noise variance $10^{-\mathrm{SNR}/10}$. Ratio for `tune_clean` (150 samples per SNR): 1.009 / 0.996 / 0.998 / 1.001 at SNR = -5 / 5 / 15 / 25 dB. For `T2_d1` it is 1.00-1.06 (rising to 1.056 at 25 dB, the residual distortion of 1 deg errors). For `T2_d5` it is 1.00 at low SNR and rises to 1.16 / 1.47 / 2.58 at 15 / 20 / 25 dB, which is exactly the unmodelled per-antenna distortion emerging above the noise floor. Negative controls: swapping $\psi\leftrightarrow\phi$ or using $\pi-\phi$ raises the ratio to 3.6 / 30 / 277 (swap) and 3.4 / 27 / 240 (flip) at 5 / 15 / 25 dB. This confirms, from the stored files alone, that (i) the row/column = AoA/AoD assignment, (ii) the $-$ sign on the Rx phase and $+$ on the Tx phase in the steering model, (iii) the codebook/`to_ant` chain, (iv) the noise variance $10^{-\mathrm{SNR}/10}$ per complex entry, and (v) the stored angle labels all agree with each other. The check is sensitive: a wrong convention is detected as soon as SNR $\ge5$ dB.
- **FACT.** The official bank is clean (no impairment). The T2 banks use the phase-only model on both arrays. The v2 authors chose the T2 conditions (L=3, 1000 per SNR, SNR −10..25) (`notebook_src.py:349-350, 372`). **Insufficient evidence**: the JSC paper text does not state L, $n_t$ or the number of test samples for Table 2. It only says validation used "4,800 samples across a grid of conditions (L, SNR, PQ, δmax)" (`a02_jsc.txt` Table 1). The T2 protocol is therefore an ASSUMPTION of the v2 authors, not a replication of a stated protocol.
- **VERIFIED (v1 robustness banks, reused read-only).** The v1 `BANK_SPECS` seed formulas collide: `gain_g0.5_snr15` and `gain_g2_snr0` both use seed 2020 ($2000+5+15=2000+20+0$), and their `psi`/`phi` arrays are identical (500 scenes). Seeds 1000/1015 are shared by `phase_d0_snr{0,15}` and the three `nuis_p*` banks each. The collided gain banks are the same scenes with a rescaled gain pattern and different noise level, so they are paired rather than independent. This does not bias any single bank, but they must not be counted as independent evidence. (The v2 notebook reads only the phase/gain/L/sep/tail banks; the nuisance banks are unused.)
- **INFERENCE (dldoa_test set).** `infer_dldoa_test.py` loads `dldoa_dataset/test_*`, which is not present on this machine. The notebook's check `upsample(downsample(x))==x` passed, so that data used the same zoom-4 map. But the U-Net Pd on it (`outputs/dldoa_test_inference/RESULTS_dldoa_test.md:7`: 0.2244/0.4604/…/0.9545) differs from the U-Net Pd on the official seed-42 bank (`RESULTS_iabr_v2.md:69`: 0.2225/0.4708/…/0.9508). The current `dldoa_dataset_generation.save_test_dataset` (seed 42) would reproduce the official bank bit-exactly (verified above), so the `dldoa_dataset` test set was **not** produced by the current script with default arguments. Its generator, seed and version cannot be established: insufficient evidence.
- **FACT (training stream).** `BatchStream` seeds each worker with `[seed + 17·start, worker_id, os.urandom(4)]` (`iabr2_sim.py:127`, `notebook_src.py:716`). Training data is fresh and non-reproducible by design, like the original. There is no train/test leakage risk because the test banks are fixed-seed and training is urandom-seeded. Exact re-training is not bit-reproducible (CONCERN, Low).

---

## 14. Summary of findings (ranked)

| # | label | severity | finding | evidence |
|---|---|---|---|---|
| 1 | VERIFIED | None | $\mathbf F=\mathbf\Phi^*/4$, $\mathbf W=\mathbf\Phi/4=\mathbf F^*$ are unitary; `to_ant` gives $\mathbf D_r^*\mathbf H\mathbf D_t+\mathbf Z'$ exactly, with $\mathbf Z'$ still i.i.d. $\mathcal{CN}(0,\sigma^2)$ | script 1 A, C, D: errors of about $10^{-15}$ |
| 2 | VERIFIED | None | Simulator equals the original `generate_channel_v2`/`W^H H F`; impairment structure equals the original `error_deg`; signs of the Rx(−)/Tx(+) phase targets and the IABC correction are consistent | script 1 D; `iabr2_sim.py:45-53,83-89`; `notebook_src.py:502-503` |
| 3 | VERIFIED | None | Unidentifiable set = {Rx and Tx phase means, Rx and Tx log-gain means, Rx and Tx phase ramps}: Jacobian null dim = 6 for L = 1, 2, 3, 6; gain ramp is identifiable; `project_impairment` is the exact orthogonal projector | script 2 E, F, G |
| 4 | CONCERN | Medium | The phase ramp is equivalent to a shift of all angles, giving an estimator-independent floor: P(>1 deg offset) = 0.38/1.44/2.7% for $\delta_{\max}$ = 2/5/8 deg (plus 0.7-1.4% aliasing) | script 2 H |
| 5 | CONCERN | Medium | Uniform-angle sampling puts 32% of paths within 1 cell of end-fire, where $\psi\approx0$ and $\psi\approx\pi$ alias: a perfect GT at $\psi$=3 deg decodes to 180 deg; 4% of L=3 scenes have two paths within 1 cell despite π/6 angular separation (base-paper protocol) | script 2 L, script 4 P |
| 6 | VERIFIED | None | `eval_bank.npz` is bit-identical to the official `validation_data_generator(seed=42)` (L=3, SNR −10..25, 1000 each); provenance is `scripts/generate_frozen_banks.py` (INFERENCE) | script 3 M |
| 7 | VERIFIED | None | All 7 v2 banks regenerate bit-exactly from the declared seeds/specs; no shared scenes | script 3 N |
| 8 | VERIFIED | Low | Up-sample map equals SciPy's corner-aligned `zoom(order=0)`, with non-uniform blocks 3/4/5 (not 4x4); faithful to the official code; paper text ambiguous | script 2 J |
| 9 | FACT | Low | Paper text vs code: training SNR −10..25 and L 1..10 in text, but −15..24 and 1..9 in code (v2 follows code); π/6 separation not stated in paper; Eq. 11 grid ≠ implemented margin grid (evaluator matches code) | `a02_tvt.txt:691-692, 529-531`; `tvt_data_generation_v3.py:315-316,364-365` |
| 10 | CONCERN | Low | Gain errors are applied without shaping the noise, so effective SNR is +0.3/+0.7/+1.2 dB above nominal at 2/3/4 dB, and the noise stays white whereas a physical combiner would colour it (off-diagonal up to 0.16 at 3 dB). A modeling choice; papers have no gain model; phase-only banks are exact | script 2 I; script 5 B; `iabr2_sim.py:50-57` |
| 11 | CONCERN | Low | Heat target peaks at the rounded 32-cell (max error 1.79 deg at broadside) and uses max-merge (1.8% of training samples lose a positive) | script 2 K; script 4 Q; heat-positives check |
| 12 | VERIFIED | Low | v1 bank seed collisions: `gain_g0.5_snr15` ≡ `gain_g2_snr0` scenes (seed 2020); `phase_d0_snr{0,15}` share seeds with the unused nuisance banks | script 3 O |
| 13 | INFERENCE | Low | The `dldoa_dataset/test_*` set is not the official seed-42 set (different U-Net Pd), and its generator is unknown: insufficient evidence | `RESULTS_dldoa_test.md:7` vs `RESULTS_iabr_v2.md:69` |
| 14 | FACT | Low | The Table-2 bank protocol (L=3, 1000/SNR) is a v2 assumption; the JSC paper does not state the test conditions | `a02_jsc.txt` Sec. 4, Table 1 |
| 15 | FACT | None | Original `error_deg` impairment uses unseeded global `np.random` (the original impaired sets are not reproducible); v2 uses a seeded rng | `tvt_data_generation_v3.py:81,107` |
| 16 | VERIFIED | None | Code-independent end-to-end check on the stored banks: LS-fit residual on stored angle labels equals the nominal noise variance (ratio 0.996-1.009 on clean data; swapped/flipped conventions give 3-277x), so labels, steering signs, codebook, `to_ant` and noise scale are mutually consistent | `a02_v5_independent.py` C |

No VERIFIED BUG was found in the v2 dataset-generation equations: every transform, sign, conjugation, normalization and index map checked out analytically and numerically. The remaining issues are properties of the physical/sampling model inherited from the base paper (items 4-5), modeling choices of the extension (items 10-11), and provenance/documentation gaps (items 9, 12-14).
