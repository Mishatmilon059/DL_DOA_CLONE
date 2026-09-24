# 03 — Hardware impairment robustness (E4 / E5 / E6) — বিস্তারিত বিশ্লেষণ

> Analyst agent: "Hardware impairment robustness"
> Scope: **E4_phase_error_sweep**, **E5_gain_error_sweep**, **E6_ood_phase_error** (সাথে এদের সব table আর figure)
> Source: `outputs/results/E4_*.json`, `E5_*.json`, `E6_*.json`; `outputs/tables/E4_*`, `E5_*`, `E6_*` (+ `_strict`); `outputs/figures/E4_phase_snr0.png`, `E4_phase_snr15.png`, `E5_gain_snr0.png`, `E5_gain_snr15.png`; notebook cell 5, 6, 9, 12, 14, 15, 17, 18, 22; `outputs/logs/run_log.txt`; `Controls_E1_E2_Abl1_Abl3.json`, `E11_multi_seed.json`, `TRAIN_*.json`।
>
> এই report-এ তিন ধরনের সংখ্যা আছে, আর প্রতিটাকে আলাদা করে চিহ্নিত করা হয়েছে:
> 1. **[FILE]** — JSON বা table থেকে হুবহু নেওয়া (যতগুলো digit ফাইলে আছে, ততগুলোই)।
> 2. **[DERIVED]** — ফাইলের সংখ্যা থেকে আমি যোগ-বিয়োগ বা ভাগ করে বের করেছি (যেমন degradation Δ)।
> 3. **[MY ANALYSIS]** — নতুন হিসাব, যা ফাইলে নেই: (ক) cached prediction (`outputs/cache/pred__*.npz`) থেকে paired bootstrap, (খ) impairment-এর distortion power-এর analytical হিসাব। কোনো training বা নতুন inference চালানো হয়নি। শুধু আগের cached prediction আবার score করা হয়েছে। Script: `C:\Users\Milon\AppData\Local\Temp\claude\D--thesis\a5db9a29-d75e-4561-834b-cd0638a16d91\scratchpad\paired_e456.py`। এই script IABR_s0-এর JSON Pd **হুবহু** মিলিয়েছে (যেমন gain_g4_snr15-এ 0.8033), তাই এর re-scoring বিশ্বাসযোগ্য।

---

## 0. শুরুর আগে: মূল ধারণাগুলো (একদম গোড়া থেকে)

### 0.1 Hardware impairment আসলে কী?
mmWave MIMO-তে Tx-এ `nt = 16` আর Rx-এ `nr = 16` antenna আছে। প্রতিটা antenna-র নিজস্ব RF chain আছে (phase shifter, amplifier, mixer)। বাস্তবে এগুলো নিখুঁত হয় না:
- **Phase error**: antenna *i*-এর phase shifter নির্দিষ্ট phase না দিয়ে একটু বেশি বা কম দেয়: `e^{jε_i}`।
- **Gain error**: antenna *i*-এর amplifier নির্দিষ্ট gain না দিয়ে একটু বেশি বা কম দেয়: `g_i`।

Notebook (cell 5 markdown, cell 6 `simulate()`) এটাকে এভাবে model করে:

```
F = D_t · F_ideal ,   W = D_r · W_ideal ,   D = diag( g · e^{jε} )
ε  ~ U(−δmax, +δmax)            (phase error, প্রতি antenna-য় আলাদা, সব beam-এ একই)
20·log10(g) ~ U(−γmax, +γmax) dB  (gain error — নতুন Phase-0 code, [ASSUMPTION]: symmetric uniform)
Y = Wᴴ H F + Z
```

এখানে গুরুত্বপূর্ণ বিষয়: error-টা **per-antenna**, অর্থাৎ একই antenna যত beam-এ ব্যবহার হয়, সবগুলোতেই একই error থাকে (E0a check: "original phase error is per-antenna (same across beams)" = True)।

### 0.2 IABR-Net-এর "impairment-aware" অংশ (IABC-v2) কী করে
DFT codebook unitary, তাই `W_ideal · Y · F_idealᴴ = D_rᴴ H D_t + noise` (E0a check: "inverse codebook recovers D_r^H H D_t", error 7.0e-15)। মানে inverse codebook প্রয়োগ করলে impairment-টা antenna domain-এ দুটো diagonal matrix হিসেবে আলাদা হয়ে যায়। IABR-Net (cell 12, `FrontEnd` class, mode `'iabc'`) এই ধারণাটাই ব্যবহার করে:
1. **Stage 0**: `Ht = W_ideal · Y · F_idealᴴ` (16×16 antenna-domain image)।
2. **Features** (`element_features()`): 32টা element (16 Rx row + 16 Tx column)। প্রতিটার জন্য দুটো feature নেওয়া হয়: (ক) log-energy ratio, মানে ওই row বা column-এর energy গড়ের তুলনায় কত, আর (খ) dominant path-এর steering phase সরিয়ে এবং linear trend (slope) detrend করে যে residual phase থাকে। Feature-গুলো `stop_gradient` করা।
3. **MLP**: context FC (4→8) আর shared per-element MLP (10→16→2)। এটা প্রতি element-এর জন্য (phase, log-gain) correction predict করে। শেষ layer zero-init করা, তাই শুরুতে এটা identity।
4. Correction `c = exp(−log-gain − j·phase)` প্রয়োগ করে forward codebook দিয়ে আবার beamspace-এ ফেরত নেওয়া হয়। এরপর trunk (10 SE-ResBlock) কাজ করে।
5. **L_pair loss** (cell 15): corrected `y_corr` আর একই channel ও একই noise-এর *clean* (ideal codebook) observation-এর মধ্যে normalized MSE, weight λ_pair = 0.5।

### 0.3 এই তিনটা test-এর তুলনামূলক model-গুলো
| model | কী | impairment দিয়ে train হয়েছে? |
|---|---|---|
| Teacher | base paper ResNet, 469,393 params | না (cell 14 markdown: "trained on clean data by their original pipelines") |
| Student | pruned ResNet r=8, 314,513 params | না |
| FNO | screening model | না |
| DFT-SIC | classical 2D-DFT + SIC, 0 learned params, `dft_ndft=512` | শেখার কিছু নেই |
| **IABR_s0** | মূল model (IABC + pair loss 0.5), seed 0, 20000 step | হ্যাঁ: phase U(0,5°), gain U(0,2 dB) |
| E1_capacity_control | **একই architecture (IABC আছে)**, কিন্তু λ_pair = 0 | হ্যাঁ, একই data |
| Abl1_no_IABC | front-end = `'none'` (identity), IABC পুরো বাদ, λ_pair = 0 | হ্যাঁ, একই data |

**গুরুত্বপূর্ণ সংশোধন (task brief বনাম ফাইল)**: task brief-এ লেখা ছিল "Ablation/sweep models trained 5000 steps"। কিন্তু `TRAIN_E1_capacity_control.json` আর `TRAIN_Abl1_no_IABC.json` দুটোতেই `"steps": 20000` আছে [FILE], আর cell 12-এর `VARIANTS`-এও এদের `'full'` দেওয়া। শুধু Abl6 sweep-এর run-গুলো 5000 step-এর। মানে E4/E5/E6-এ IABR_s0, E1 আর Abl1 **একই data, একই step, একই seed**-এ train হয়েছে। তাই এদের মধ্যে তুলনাটা পুরোপুরি controlled (cell 14 markdown-ও এটাই বলে)। অন্যদিকে Teacher, Student আর FNO-র সাথে তুলনা controlled নয়, শুধু reference।

**Abl1 কেন IABC-এর exact ablation**: `'none'` mode-এ stage 0-ও নেই। কিন্তু correction ছাড়া stage 0 আর forward codebook মিলে identity হয়ে যায় (codebook unitary)। তাই IABR বনাম Abl1-এর মধ্যে পার্থক্য শুধু **IABC-এর learned correction**। IABR বনাম E1-এর পার্থক্য শুধু **pair loss**।

### 0.4 Evaluation protocol (cell 9, 17, 18, 22)
- প্রতিটা condition-এ `n_robust = 500` sample, **L = 3** fixed, SNR ∈ {0, 15} dB।
- প্রতিটা bank-এর seed আলাদা। যেমন `phase_d{d}_snr{s}` → seed = 1000 + 10·d + snr, `gain_g{g}` → 2000 + 10·g + snr, `oodphase_d{d}` → 3000 + d + snr। **মানে প্রতিটা impairment level-এ channel বা scene আলাদা** (paired নয়)। এর ফলটা পরে গুরুত্বপূর্ণ হবে।
- একই bank এক experiment থেকে আরেকটাতে পুনরায় ব্যবহার হয়েছে: E5-এর "0" column = `phase_d0_snr{s}` bank (E4-এর 0° column-এর সাথে হুবহু একই সংখ্যা)। E6-এর "5 (in)" column = `phase_d5_snr{s}` (E4-এর 5° column-এর সাথে হুবহু একই)।
- `pd_paper`: <L detection থাকলে sample **drop** হয় (মূল convention)। `pd_strict`: drop হওয়া sample miss হিসেবে গোনা হয়।
- **IABR-পরিবারের কোনো model কখনো drop করে না** (সব condition-এ `n_dropped = 0` [FILE])। কারণ `predict_iabr` সবসময় top-L local maxima নেয়। তাই IABR, E1, Abl1-এর paper আর strict Pd **হুবহু সমান**। DFT-SIC-ও কখনো drop করে না। Teacher, Student আর FNO drop করে। তাই strict table-এ এদের সংখ্যা কমে যায়।

---

## 0.5 [MY ANALYSIS] Physics: impairment আসলে কতটা "জোরালো"?

এটা পুরো ফলাফল বোঝার চাবিকাঠি। per-element error `x = conj(d_r)·d_t` হলে single path-এর জন্য antenna domain-এ প্রতিটা element-এর magnitude সমান থাকে। তাই distortion-to-signal ratio (DSR) = `E|x − 1|²`। Uniform distribution-এ closed form: `E[e^{jε}] = sin δ/δ`, আর `a = γ·ln10/20` হলে `E[g] = sinh a / a` এবং `E[g²] = sinh 2a / 2a`। Y-তে প্রতি element-এ signal power ≈ 1 আর noise power = 10^(−SNR/10)। তাই DSR-কে সরাসরি noise level-এর সাথে তুলনা করা যায়:

| condition | DSR (raw) | DSR (dB) | SNR 0-তে noise (0 dB)-এর তুলনায় | SNR 15-তে noise (−15 dB)-এর তুলনায় |
|---|---|---|---|---|
| phase 1° | 0.00020 | −36.9 dB | নগণ্য | নগণ্য |
| phase 2° | 0.00081 | −30.9 dB | নগণ্য | নগণ্য |
| phase 5° (training max) | 0.00507 | −22.9 dB | নগণ্য | noise-এর ~8 dB নিচে |
| phase 10° (OOD) | 0.02023 | −16.9 dB | নগণ্য | noise-এর ~2 dB নিচে |
| phase 15° (OOD) | 0.04528 | −13.4 dB | নগণ্য | **noise-এর চেয়ে বেশি** |
| gain 0.5 dB | 0.00222 | −26.5 dB | নগণ্য | নগণ্য |
| gain 1 dB | 0.00895 | −20.5 dB | নগণ্য | noise-এর ~5.5 dB নিচে |
| gain 2 dB (training max) | 0.03712 | −14.3 dB | নগণ্য | **প্রায় noise-এর সমান** |
| gain 4 dB (OOD) | 0.17132 | −7.7 dB | noise-এর ~8 dB নিচে | **noise-এর চেয়ে ~7 dB বেশি** |
| training গড় (δ~U(0,5°), γ~U(0,2 dB)) | 0.0139 | −18.6 dB | — | — |

**এর মানে:**
1. **E4 (≤5°) physically প্রায় "অদৃশ্য" একটা test।** 5° phase error-এর distortion (−22.9 dB) SNR 0 বা SNR 15 দুই জায়গাতেই noise-এর অনেক নিচে। তাই কোনো model-এরই এখানে বড় পতন হওয়ার কথা নয়, আর IABC-এর কাজ দেখানোর সুযোগও প্রায় নেই।
2. **SNR 0 dB-এ সব impairment level noise-এর নিচে ডুবে যায়।** SNR 0-এর curve-গুলো মূলত flat হওয়ার কথা।
3. **আসল "active" condition মাত্র কয়েকটা**: SNR 15 dB-এ gain 2 dB (সীমানায়), gain 4 dB (জোরালো), আর phase 15° (মাঝারি)। IABC কাজ করে কিনা, সেটা আসলে এখানেই যাচাই হয়।
4. **Training data-তে impairment-এর signal দুর্বল ছিল।** Training গড় DSR −18.6 dB, অথচ training SNR −15 থেকে 24 dB পর্যন্ত uniform। ফলে বেশিরভাগ training sample-এ impairment distortion noise-এর নিচে ছিল। IABC-কে শেখানোর মতো gradient signal খুব কম ছিল। এটা "IABC কেন খুব কম শিখেছে" প্রশ্নের একটা বড় ব্যাখ্যা (নিচে §4 দেখো)।

(সতর্কতা: impairment distortion white noise নয়, structured। এটা মূলত sidelobe বাড়ায় আর weak path-কে ঢেকে দেয়। তাই "noise-এর সাথে তুলনা" একটা intuition মাত্র, exact prediction নয়।)

### 0.6 [MY ANALYSIS] Statistical noise floor: কত বড় পার্থক্যকে "আসল" বলা যায়?
- প্রতি condition-এ 500 sample × L=3 × 2 component = 3000 component, কিন্তু sample-গুলো clustered। Pd ≈ 0.87 হলে binomial SE ≈ 0.009, আর Pd ≈ 0.65 হলে ≈ 0.012।
- **আলাদা bank-এর মধ্যে তুলনা** (যেমন 0° বনাম 1°) করলে scene-এর পার্থক্যও ঢুকে পড়ে। প্রমাণ হিসেবে DFT-SIC-কে দেখা যায়, যেটা deterministic আর যার কোনো training নেই: SNR 15-এ 0° → 1°-এ এর Pd 0.9183 → 0.9003 [FILE], অর্থাৎ −1.8 pp। অথচ physics অনুযায়ী 1°-এর distortion −36.9 dB, যা কার্যত শূন্য। **তাই এই −1.8 pp পুরোটাই bank-sampling noise।** মানে E4/E5/E6-এর curve-এ ±2 pp-এর ওঠানামা কোনো অর্থ বহন করে না।
- **একই bank-এ দুই model-এর তুলনা** (যেমন IABR বনাম Abl1) paired, তাই অনেক বেশি নির্ভুল। এর জন্য আমি cached prediction থেকে per-sample paired bootstrap (4000 resample, 95% CI) চালিয়েছি (§4.3)।

---

## 1. E4 — Phase error sweep (in-distribution, δmax ∈ {0, 1, 2, 5}°)

### (1) কী test করা হয়েছে
প্রতি antenna-য় random phase error `ε ~ U(−δmax, δmax)` রেখে, δmax = 0, 1, 2, 5° এবং SNR = 0 ও 15 dB-এ, 7টা model-এর Pd মাপা হয়েছে। প্রতি condition-এ L=3 আর 500 sample। সব level training range-এর (≤5°) ভেতরে।

### (2) Theory
Phase error codebook-এর beam pattern বিকৃত করে। Main lobe সামান্য দুর্বল হয় (array gain loss factor `(sin δ/δ)²`, যা 5°-এ 0.9975, মানে ~0.01 dB), আর sidelobe-এ energy ছড়িয়ে পড়ে। Weak path-এর blob ওই sidelobe floor-এর নিচে চাপা পড়তে পারে। §0.5 অনুযায়ী 5° পর্যন্ত এই floor −22.9 dB, যা দুই SNR-এর noise-এর চেয়েই কম। তাই theory বলে E4-এ পতন খুব ছোট হবে। Lloria এবং অন্য literature-এও phase error-এর এই range (≤5°) ব্যবহার হয়। তাই E4-এর মূল মূল্য হলো **comparability**: base paper-এর setting-এ অন্যদের সাথে সরাসরি তুলনা করা যায়।

### (3) Code
- Bank: cell 9, `BANK_SPECS['phase_d{d}_snr{snr}'] = ('std', dict(n=N_R, L=3, snr=snr, seed=1000+10*d+snr, phase_deg=d))`, তৈরি হয় `make_standard_bank()` → `sample_paths()` + `simulate(..., phase_deg_max=d)` দিয়ে।
- Test: cell 22, `e4()` → `sweep('E4_snr{snr}', ...)`। এটা প্রতিটা `ROBUST_MODELS` = [Teacher, Student, FNO, DFT-SIC, IABR_s0, E1_capacity_control, Abl1_no_IABC]-এর জন্য `predict()` আর `metrics()` চালায়, তারপর `put_table()` দিয়ে paper ও strict table লেখে আর `plot_sweep()` দিয়ে figure আঁকে।
- Metric: cell 18, `metrics()`: Hungarian matching (`prepare_for_metric` → `permute_pairs` = `linear_sum_assignment`) এবং 1° threshold।

### (4) কেন করা হয়েছে
Research report (§21 table) E4-কে "HIGH comparability to Lloria/Meneses" test হিসেবে রেখেছিল। Hypothesis (report §27): "IABC-v2 per-element correction phase error-এ PD degradation কমাবে"। E4 in-range অংশ যাচাই করে, আর E6 যাচাই করে 5°-এর বাইরের অংশ।

### (5) Result [FILE]

**SNR 0 dB, paper-style Pd** (`E4_snr0.md`)

| model | 0° | 1° | 2° | 5° |
|---|---|---|---|---|
| Teacher | 0.6298 | 0.6434 | 0.6466 | 0.6571 |
| Student | 0.6115 | 0.6373 | 0.6248 | 0.6536 |
| FNO | 0.6079 | 0.6034 | 0.6189 | 0.6320 |
| DFT-SIC | 0.6790 | 0.6780 | 0.6920 | 0.7000 |
| IABR_s0 | 0.6307 | 0.6340 | 0.6497 | 0.6573 |
| E1_capacity_control | 0.6297 | 0.6403 | 0.6523 | 0.6587 |
| Abl1_no_IABC | 0.6287 | 0.6307 | 0.6367 | 0.6517 |

**SNR 0 dB, strict Pd** (`E4_snr0_strict.md`): Teacher 0.5643 / 0.5803 / 0.6000 / 0.5967; Student 0.5430 / 0.5633 / 0.5723 / 0.5857; FNO 0.5897 / 0.5853 / 0.6003 / 0.6143। DFT-SIC, IABR_s0, E1 আর Abl1-এর সংখ্যা paper-style-এর হুবহু সমান (drop = 0)।

**SNR 15 dB, paper-style Pd** (`E4_snr15.md`)

| model | 0° | 1° | 2° | 5° |
|---|---|---|---|---|
| Teacher | 0.9127 | 0.9049 | 0.9044 | 0.8859 |
| Student | 0.9013 | 0.8807 | 0.8918 | 0.8724 |
| FNO | 0.8079 | 0.7971 | 0.7835 | 0.7743 |
| DFT-SIC | 0.9183 | 0.9003 | 0.9010 | 0.9007 |
| IABR_s0 | 0.8700 | 0.8480 | 0.8603 | 0.8510 |
| E1_capacity_control | 0.8757 | 0.8503 | 0.8567 | 0.8527 |
| Abl1_no_IABC | 0.8737 | 0.8570 | 0.8533 | 0.8607 |

**SNR 15 dB, strict Pd** (`E4_snr15_strict.md`): Teacher 0.8890 / 0.8597 / 0.8737 / 0.8593; Student 0.8490 / 0.8367 / 0.8543 / 0.8480; FNO 0.7740 / 0.7477 / 0.7553 / 0.7480। বাকিগুলো paper-এর সমান।

**অতিরিক্ত metric (JSON থেকে) [FILE]:**
- Teacher-এর dropped sample (500-এর মধ্যে): SNR 0-এ 52 / 49 / 36 / 46, SNR 15-এ 13 / 25 / 17 / 15। Student: SNR 0-এ 56 / 58 / 42 / 52। FNO: SNR 0-এ 15 / 15 / 15 / 14।
- RMSE (SNR 15, 0→5°): Teacher 0.2852515731836245 → 0.2987660835581366; IABR_s0 0.29069859012401233 → 0.29864858278355877; DFT-SIC 0.2575183100970328 → 0.2778019906343234; Abl1 0.29851984692462563 → 0.319211755063209।
- p95 absolute error (SNR 15): Teacher 1.89986770039252 (0°), 1.8556975680894983 (1°); IABR_s0 5.135436389539285 (0°), **40.609547997729294 (1°)**, 25.937522160145395 (2°), 12.841162557184353 (5°); DFT-SIC 2.094831248945047 (0°)।
- pd_source (AoA আর AoD দুটোই ঠিক), SNR 15, 5°: Teacher 0.7786666666666666, DFT-SIC 0.8266666666666667, IABR_s0 0.7646666666666667, Abl1 0.78।

**Degradation 0° → 5° (paper-style) [DERIVED]:**

| model | SNR 0 Δ | SNR 15 Δ |
|---|---|---|
| Teacher | +0.0273 | −0.0268 |
| Student | +0.0421 | −0.0289 |
| FNO | +0.0241 | −0.0336 |
| DFT-SIC | +0.0210 | −0.0176 |
| IABR_s0 | +0.0266 | −0.0190 |
| E1 | +0.0290 | −0.0230 |
| Abl1_no_IABC | +0.0230 | −0.0130 |

### Figure: `E4_phase_snr0.png`
- **যা দেখা যায়**: x-axis-এ phase error max (0, 1, 2, 5°), y-axis-এ pd_paper (0.60–0.70)। সবার উপরে লাল DFT-SIC line (0.679 → 0.700), আলাদা হয়ে। মাঝখানে Teacher, IABR_s0, E1 আর Abl1-এর একটা গুচ্ছ (0.63 → 0.66)। নিচে FNO (সবুজ, 0.608 → 0.632)। Student-এর line zig-zag (0.611 → 0.637 → 0.625 → 0.654)। **প্রায় সব line phase error বাড়ার সাথে উপরে উঠছে।**
- **মানে**: phase error বাড়লে Pd *বাড়ার* কোনো physical কারণ নেই। Deterministic DFT-SIC-ও +2.1 pp উঠেছে। এর ব্যাখ্যা হলো প্রতিটা x-point-এর bank আলাদা seed-এ তৈরি (§0.4), তাই কিছু bank-এর scene সহজ পড়েছে। এই figure-এর "উঠতি trend" একটা **sampling artifact**, impairment-এর effect নয়। SNR 0-এ impairment noise-এ ডুবে থাকে (§0.5)। Model ranking-টা নির্ভরযোগ্য: DFT-SIC সবার উপরে, IABR/Teacher/E1/Abl1 প্রায় সমান, FNO সবার নিচে।

### Figure: `E4_phase_snr15.png`
- **যা দেখা যায়**: y-axis 0.77–0.92। উপরে DFT-SIC (0.918 → 0.901, 1°-এর পর flat) আর Teacher (0.913 → 0.886, 5°-এ নামছে)। তার নিচে Student (0.901 → 0.872)। আরও নিচে IABR_s0, E1 আর Abl1-এর একটা জট (0.87 → 0.85), যেখানে তিনটা line বারবার একে অপরকে cross করছে। সবার নিচে FNO, যেটা একটানা নামছে (0.808 → 0.774)।
- **মানে**: (ক) ≤5°-এ সব model মাত্র 1.3–3.4 pp হারিয়েছে, যা §0.6-এর noise floor (±2 pp)-এর কাছাকাছি। (খ) IABR, E1 আর Abl1 **পুরোপুরি জট পাকিয়ে আছে**, IABC থাকা বা না থাকায় কোনো দৃশ্যমান পার্থক্য নেই। (গ) IABR-পরিবার Teacher আর DFT-SIC-এর চেয়ে প্রায় 4–5 pp নিচে, এবং এই gap-টা impairment-এর সাথে বদলায় না। অর্থাৎ এটা impairment robustness-এর সমস্যা নয়, **clean-condition accuracy gap** (E3-এ SNR 15-এ IABR_s0 0.8662 বনাম Teacher 0.8994 [FILE, RESULTS.md])। (ঘ) FNO-ই একমাত্র model যার line পরিষ্কারভাবে monotonic ভাবে নামছে।

### (6) Comparison
- **IABR বনাম Abl1 (IABC-এর আসল contribution)**: [MY ANALYSIS] paired diff (strict), SNR 15: 0° −0.0037 [−0.014, +0.006], 1° −0.009 [−0.0193, +0.0017], 2° +0.007 [−0.0037, +0.017], 5° **−0.0097** [−0.0217, +0.002]। SNR 0: 0° +0.002, 1° +0.0033, 2° **+0.013 [+0.0017, +0.024]**, 5° +0.0057 [−0.0063, +0.0177]। মানে ৮টা তুলনার মধ্যে মাত্র ১টায় (SNR 0, 2°) CI শূন্য বাদ দেয়। এতগুলো তুলনা থেকে একটা এমন ফল দৈবক্রমেই আসতে পারে। **SNR 15-এ 5°-এ IABR আসলে Abl1-এর চেয়ে 0.97 pp *খারাপ*** (significant নয়)। সারকথা: E4-এ IABC-এর কোনো প্রমাণযোগ্য লাভ নেই।
- **IABR বনাম Teacher**: SNR 15-এ paper-style Pd-তে IABR প্রতিটা level-এ 2.7–4.3 pp নিচে। Paired strict diff: 0° −0.019 [−0.0347, −0.003] (significant), 1° −0.0117, 2° −0.0133, 5° −0.0083 [−0.026, +0.0103]। 5°-এ gap কমে, কারণ Teacher বেশি নেমেছে, তবে এটা noise-এর ভেতরে। SNR 0-এ strict Pd-তে IABR Teacher-এর চেয়ে +5.0 থেকে +6.6 pp এগিয়ে (CI সবসময় শূন্যের উপরে)। কিন্তু এটা **শুধু drop policy-র কারণে** (Teacher 36–52টা sample drop করে, IABR কোনোটাই করে না)। Paper-style-এ দুটো প্রায় সমান (0.6298 বনাম 0.6307)।
- **IABR বনাম DFT-SIC**: প্রতিটা condition-এ IABR 4.1–5.2 pp নিচে। Paired CI সবসময় পুরোপুরি শূন্যের নিচে। 0-parameter classical baseline-ই phase error-এ সবচেয়ে robust, কারণ তার শেখার কিছু নেই, তাই overfitting-ও নেই।
- **FNO**: সবচেয়ে বেশি পতন (SNR 15-এ −3.36 pp) আর সবচেয়ে কম absolute Pd।

### (7) Verdict — **Mixed / প্রায় inconclusive (IABC-এর claim-এর জন্য: FAIL to demonstrate)**
- ✔ IABR ≤5° phase error-এ স্থিতিশীল (SNR 15-এ −1.90 pp, যা Teacher-এর −2.68 pp-এর চেয়ে কম)।
- ✘ কিন্তু এই স্থিতিশীলতা **IABC-এর কারণে নয়**। Abl1 (IABC ছাড়া) −1.30 pp হারিয়েছে, যা IABR-এর চেয়েও কম।
- ✘ Physics অনুযায়ী E4 range-এ distortion noise-এর নিচে (§0.5), তাই এই test IABC-এর দরকারই তৈরি করে না। এটা test design-এর দুর্বলতা।
- ✘ absolute Pd-তে IABR Teacher আর DFT-SIC দুটোর চেয়েই নিচে (SNR 15)।

---

## 2. E5 — Gain error sweep (γmax ∈ {0, 0.5, 1, 2, 4} dB, 4 dB = OOD)

### (1) কী test করা হয়েছে
প্রতি antenna-য় amplitude error `20·log10(g) ~ U(−γmax, +γmax)` dB (Tx আর Rx দুই দিকেই), γmax = 0, 0.5, 1, 2, 4 dB, SNR 0 আর 15 dB। Training-এ γmax ≤ 2 dB ছিল, তাই 4 dB **out-of-distribution (OOD)**।

### (2) Theory
Phase error-এর তুলনায় gain error-এর প্রভাব বেশি। 2 dB পর্যন্ত per-element amplitude প্রায় ±26% পর্যন্ত ওঠানামা করে, আর 4 dB-এ প্রায় ±58% (10^(4/20) = 1.585)। §0.5 অনুযায়ী 4 dB-এর distortion −7.7 dB, যা SNR 15-এর noise (−15 dB)-এর চেয়ে ~7 dB **বেশি**। তাই SNR 15-এ 4 dB হলো এই পুরো suite-এর সবচেয়ে কঠিন impairment condition। SNR 0-এ তবুও noise বেশি থাকে, তাই সেখানে প্রভাব কম হওয়ার কথা। Design-এর দিক থেকে IABC-এর log-gain output ঠিক এই ধরনের error ধরার জন্যই বানানো (`out[...,1]` = log-gain correction)। Literature (Meneses-Albalá) gain error-কে একটা "named gap" বলেছিল, অর্থাৎ আগে কেউ এটা test করেনি।

### (3) Code
- Bank: cell 9, `BANK_SPECS['gain_g{g}_snr{snr}'] = ('std', dict(n=N_R, L=3, snr=snr, seed=2000+int(10*g)+snr, gain_db=g))`। Gain injection হয় cell 6-এর `simulate()`-এ: `g_r = 10 ** (rng.uniform(-1, 1, (B, NR)) * gdm[:, None] / 20)`। E0a unit test-এ "gain within +-2 dB" = 1.8223193193360356, True [FILE]।
- Test: cell 22, `e5()`, banks = `['phase_d0_snr{snr}'] + ['gain_g{g}_snr{snr}' for g in [0.5, 1, 2, 4]]`, xs = `[0, 0.5, 1, 2, '4 (OOD)']`। মানে 0 dB column হলো E4-এর 0° bank-টাই।

### (4) কেন করা হয়েছে
Research report §21: "E5 — Gain-error sweep ... Closes Meneses-Albalá's own named gap — first in the field"। এটা IABR-Net-এর novelty claim-এর কেন্দ্রীয় test: impairment-aware front-end কি gain error-এ আলাদা কিছু দেয়? 4 dB OOD যাচাই করে training range-এর বাইরে model কতটা ভালো generalize করে।

### (5) Result [FILE]

**SNR 0 dB, paper-style** (`E5_snr0.md`)

| model | 0 | 0.5 | 1 | 2 | 4 (OOD) |
|---|---|---|---|---|---|
| Teacher | 0.6298 | 0.6493 | 0.6605 | 0.6513 | 0.6298 |
| Student | 0.6115 | 0.6393 | 0.6322 | 0.6344 | 0.6055 |
| FNO | 0.6079 | 0.6141 | 0.6228 | 0.6205 | 0.5986 |
| DFT-SIC | 0.6790 | 0.6930 | 0.7007 | 0.7040 | 0.6847 |
| IABR_s0 | 0.6307 | 0.6507 | 0.6520 | 0.6520 | 0.6363 |
| E1_capacity_control | 0.6297 | 0.6470 | 0.6450 | 0.6437 | 0.6397 |
| Abl1_no_IABC | 0.6287 | 0.6490 | 0.6553 | 0.6357 | 0.6257 |

**SNR 0 dB, strict** (`E5_snr0_strict.md`): Teacher 0.5643 / 0.5843 / 0.6090 / 0.5953 / 0.5920; Student 0.5430 / 0.5677 / 0.5690 / 0.5710 / 0.5643; FNO 0.5897 / 0.5993 / 0.6103 / 0.6043 / 0.5890। বাকিগুলো paper-এর সমান।

**SNR 15 dB, paper-style** (`E5_snr15.md`)

| model | 0 | 0.5 | 1 | 2 | 4 (OOD) |
|---|---|---|---|---|---|
| Teacher | 0.9127 | 0.9050 | 0.9018 | 0.8541 | 0.7787 |
| Student | 0.9013 | 0.8921 | 0.8806 | 0.8337 | 0.7611 |
| FNO | 0.8079 | 0.7988 | 0.7834 | 0.7667 | 0.7238 |
| DFT-SIC | 0.9183 | 0.9063 | 0.9017 | 0.8860 | 0.8387 |
| **IABR_s0** | 0.8700 | 0.8570 | 0.8603 | 0.8463 | **0.8033** |
| E1_capacity_control | 0.8757 | 0.8560 | 0.8653 | 0.8477 | 0.8003 |
| Abl1_no_IABC | 0.8737 | 0.8570 | 0.8687 | 0.8437 | 0.7790 |

**SNR 15 dB, strict** (`E5_snr15_strict.md`): Teacher 0.8890 / 0.8633 / 0.8783 / 0.8507 / 0.7740; Student 0.8490 / 0.8403 / 0.8507 / 0.8287 / 0.7550; FNO 0.7740 / 0.7717 / 0.7583 / 0.7560 / 0.7223। বাকিগুলো paper-এর সমান।

**অতিরিক্ত metric, SNR 15, gain 4 dB [FILE]:**

| model | pd_source | rmse | p95 | p95_broadside | pairing_error | n_dropped |
|---|---|---|---|---|---|---|
| Teacher | 0.6826666666666666 | 0.34022045788121535 | 71.21739103230283 | 43.67003377259265 | 0.18376928236083165 | 3 |
| Student | 0.662 | 0.35922629476834017 | 73.25575692331361 | 56.33198956834147 | 0.1875 | 4 |
| FNO | 0.5813333333333334 | 0.4302892860351428 | 63.900926934231364 | 21.185238206333644 | 0.28256513026052105 | 1 |
| DFT-SIC | 0.758 | 0.29542232909358396 | 36.9480360685064 | 1.7305771141136006 | 0.16133333333333333 | 0 |
| IABR_s0 | 0.7073333333333334 | 0.3476101513856931 | 43.39393223463534 | 1.5921287176356456 | 0.192 | 0 |
| E1 | 0.698 | 0.3480749947618269 | 35.14619961553032 | 1.5275863300231658 | 0.20466666666666666 | 0 |
| Abl1 | 0.666 | 0.3597122465356794 | 38.69380053784456 | 2.501035505112515 | 0.226 | 0 |

(উপরের সংখ্যাগুলো `E5_gain_error_sweep.json` থেকে full precision-এ নেওয়া। Report-এর অন্য জায়গায়, যেমন E4 ও E6-এর অতিরিক্ত metric, সংখ্যাগুলো 4 decimal-এ দেখানো হয়েছে। Full precision JSON-এ আছে।)

লক্ষণীয়: Teacher-এর drop সংখ্যা clean-এ 13 থেকে gain 4 dB-এ 3-এ নেমেছে [FILE]। মানে gain error-এ Teacher-এর heatmap-এ বেশি spurious blob তৈরি হয় (সে বেশি peak "দেখে"), কিন্তু সেগুলো ভুল জায়গায়। এর ফলে paper আর strict Pd প্রায় সমান হয়ে যায় (0.7787 বনাম 0.7740)।

**Degradation [DERIVED], paper-style, SNR 15:**

| model | 0→2 dB Δ | 0→4 dB Δ | retention at 4 dB (Pd4/Pd0) |
|---|---|---|---|
| Teacher | −0.0586 | −0.1340 | 0.853 |
| Student | −0.0676 | −0.1402 | 0.844 |
| FNO | −0.0412 | −0.0841 | 0.896 |
| DFT-SIC | −0.0323 | −0.0796 | 0.913 |
| **IABR_s0** | **−0.0237** | **−0.0667** | **0.923** |
| E1 | −0.0280 | −0.0754 | 0.914 |
| Abl1_no_IABC | −0.0300 | −0.0947 | 0.892 |

SNR 0-তে 0→4 dB Δ: Teacher 0.0000, Student −0.0060, FNO −0.0093, DFT-SIC +0.0057, IABR +0.0056, E1 +0.0100, Abl1 −0.0030 [DERIVED]। অর্থাৎ SNR 0-এ gain error কার্যত কোনো প্রভাব ফেলেনি, যা physics-এর সাথে মেলে।

### Figure: `E5_gain_snr0.png`
- **যা দেখা যায়**: y-axis 0.60–0.71। DFT-SIC (লাল) সবার উপরে, 0.679 → 0.704 পর্যন্ত উঠে 4 dB-এ 0.685-এ নামে। মাঝের গুচ্ছে (Teacher, IABR, E1, Abl1) 0.63 থেকে শুরু, 0.5–2 dB-এ ~0.645–0.66-এ উঠে, 4 dB-এ 0.626–0.640-এ নামে। Teacher 1 dB-এ সর্বোচ্চ 0.6605-এ পৌঁছায়। Student আর FNO নিচে থাকে, 4 dB-এ 0.6055 আর 0.5986-এ নামে। মোটামুটি সব line একটা "উল্টো-U" আকার নেয়।
- **মানে**: এই উল্টো-U আকারও bank-sampling-এর ফল। 0.5, 1, 2 dB-এর bank-গুলো আলাদা seed-এর, আর DFT-SIC-ও একই আকার দেখায়। SNR 0-এ gain error ≤4 dB noise-এর নিচে থাকে, তাই কোনো model-এরই অর্থবহ পতন নেই। একটা ছোট লক্ষণ আছে: 4 dB-এ IABC-যুক্ত দুটো model (IABR 0.6363, E1 0.6397) Abl1 (0.6257) আর Teacher (0.6298)-এর চেয়ে সামান্য উপরে। Paired test-এ E1−Abl1 = +0.014 [+0.002, +0.026] significant, আর IABR−Abl1 = +0.0107 [−0.0013, +0.0233] সীমানায়।

### Figure: `E5_gain_snr15.png`
- **যা দেখা যায়**: এটা পুরো group-এর **সবচেয়ে তথ্যবহুল figure**। 0–1 dB-এ সব line প্রায় সমান্তরাল: DFT-SIC ≈ Teacher ~0.90–0.92, Student ~0.88–0.90, IABR/E1/Abl1-এর জট ~0.86–0.87, FNO ~0.78–0.81। **2 dB থেকে Teacher (নীল) আর Student (কমলা) খাড়া ভাবে নামতে থাকে**, আর 4 dB-এ Teacher 0.7787-এ নেমে IABR/E1 (~0.80)-এর **নিচে চলে যায়**। একটা crossover দেখা যায় 2 আর 4 dB-এর মাঝখানে। DFT-SIC সবচেয়ে ধীরে নামে এবং সবসময় সবার উপরে থাকে (0.8387)। 4 dB-এ IABR আর E1 একসাথে ~0.80-এ থাকে, আর Abl1 (গোলাপি) Teacher-এর সাথে ~0.779-এ নেমে যায়। অর্থাৎ **শুধু এই একটা বিন্দুতেই IABC-যুক্ত আর IABC-ছাড়া model আলাদা হয়ে যায়।**
- **মানে**: (ক) clean-data-তে train হওয়া Teacher বা Student gain error-এ ভঙ্গুর: 4 dB-এ −13.4 pp আর −14.0 pp। (খ) impairment-augmented training-এর (IABR, E1, Abl1 তিনটাই এটা পেয়েছে) বড় অংশ লাভ এখানেই দেখা যায়। Abl1 (IABC ছাড়া) −9.47 pp হারিয়েছে, যা Teacher-এর −13.40 pp-এর চেয়ে ভালো। (গ) **IABC-এর নিজস্ব অবদান**: 4 dB-এ IABR (−6.67 pp) আর E1 (−7.54 pp) বনাম Abl1 (−9.47 pp), মানে প্রায় 2–3 pp বাড়তি robustness। (ঘ) তবুও 0-parameter DFT-SIC absolute Pd-তে সবার উপরে।

### (6) Comparison
**[MY ANALYSIS] Paired bootstrap, SNR 15 (strict = paper, কারণ IABR-পরিবার drop করে না):**

| তুলনা | 0.5 dB | 1 dB | 2 dB | **4 dB (OOD)** |
|---|---|---|---|---|
| IABR_s0 − Abl1 | +0.0000 [−0.0113, +0.0117] | −0.0083 [−0.0193, +0.0023] | +0.0027 [−0.0083, +0.014] | **+0.0243 [+0.0107, +0.0387]** |
| E1 − Abl1 | −0.001 | −0.0033 | +0.004 | **+0.0213 [+0.0083, +0.0347]** |
| Abl3_no_SE − Abl1 | — | — | +0.0 [−0.0117, +0.011] | **+0.0257 [+0.0127, +0.039]** |
| IABR_s0 − Teacher (strict) | −0.0063 | −0.018 [−0.0347, −0.0013] | −0.0043 [−0.0193, +0.0103] | **+0.0293 [+0.0117, +0.046]** |
| IABR_s0 − DFT-SIC | −0.0493 | −0.0413 | −0.0397 | **−0.0353 [−0.0503, −0.021]** |

SNR 0, 4 dB: IABR − Abl1 = +0.0107 [−0.0013, +0.0233], E1 − Abl1 = +0.014 [+0.002, +0.026], IABR − Teacher (strict) = +0.0443 [+0.0253, +0.0637] (এটা drop policy-র প্রভাব), IABR − DFT-SIC = −0.0483 [−0.0633, −0.033]। SNR 0, 2 dB: IABR − Abl1 = +0.0163 [+0.0043, +0.0287]।

এর থেকে পাওয়া **সবচেয়ে গুরুত্বপূর্ণ দুটো সিদ্ধান্ত**:
1. **IABC-এর একমাত্র শক্ত, পুনরাবৃত্ত প্রমাণ SNR 15 dB, gain 4 dB (OOD)-এ।** IABC-যুক্ত **তিনটা আলাদা model** (IABR_s0, E1, Abl3_no_SE), যেগুলোর loss বা SE আলাদা, সবাই Abl1-এর চেয়ে +2.1 থেকে +2.6 pp এগিয়ে, এবং তিনটার CI-ই শূন্য বাদ দেয়। তাই এটা দৈবক্রমে পাওয়া ফল হওয়ার সম্ভাবনা কম। (সীমাবদ্ধতা: Abl1-এর একটাই seed। তবে E11 দেখায় IABR seed std ≈ 0.001–0.0025 [FILE], যা 2.4 pp-এর চেয়ে অনেক ছোট।)
2. **এই লাভ pair loss (impairment-specific supervision) থেকে আসছে না।** E1 (pair = 0) IABR-এর প্রায় সমান (+2.13 বনাম +2.43 pp)। মানে লাভটা আসছে IABC-এর **architecture** থেকে, অর্থাৎ per-element log-gain feature আর multiplicative correction-এর structure থেকে, "clean-এর দিকে correction শেখানো" থেকে নয়। (§4-এ আরও বিস্তারিত।)

**Controls table** (`Controls_E1_E2_Abl1_Abl3.json` [FILE]), gain_g4_snr15: IABR_s0 0.8033, E1 0.8003, **E2_dense_frontend 0.6527**, Abl1 0.7790, Abl3_no_SE 0.8047। Dense front-end (457,430 params) সবচেয়ে খারাপ, তাই physics-structured front-end-এর মূল্য পরিষ্কার। gain_g2_snr15: IABR 0.8463, E1 0.8477, E2 0.7467, Abl1 0.8437, Abl3 0.8437। অর্থাৎ 2 dB (training range-এর ভেতরে)-এ IABC-এর কোনো লাভ নেই।

**Multi-seed** (`E11_multi_seed.json` [FILE]): gain_g2_snr15-এ IABR-এর 3 seed = [0.8463, 0.8513, 0.849], mean 0.8488888888888889, std 0.0025018511664883793। মানে IABR-এর ফল seed-এর ওপর স্থিতিশীল। 4 dB-এর multi-seed মাপা হয়নি (কিছুটা gap)।

### (7) Verdict — **Partial success**
- ✔ **IABR-Net-এর সবচেয়ে শক্তিশালী ফল**: SNR 15-এ 4 dB OOD gain error-এ IABR (0.8033) Teacher (0.7787), Student (0.7611), FNO (0.7238) আর Abl1 (0.7790) সবাইকে ছাড়িয়ে যায়। Retention 92.3% (সবার মধ্যে সর্বোচ্চ)। Teacher-এর বিরুদ্ধে paired +2.93 pp, significant।
- ✔ IABC-এর প্রভাব এখানে প্রমাণযোগ্য (+2.43 pp বনাম Abl1, তিনটা IABC model-এ পুনরাবৃত্ত)।
- ✘ কিন্তু 0–2 dB-এ (training range, আর বাস্তবসম্মত range) IABC-এর কোনো পরিমাপযোগ্য লাভ নেই।
- ✘ Absolute Pd-তে 0-parameter DFT-SIC সব level-এ IABR-কে 3.5–5.2 pp ছাড়িয়ে যায়। "Impairment-robust learned estimator" দাবি করতে হলে classical baseline-কে হারাতে হবে, আর সেটা হয়নি।
- ✘ IABR-এর "কম degradation"-এর একটা অংশ **ceiling effect**: IABR clean-এ 0.8700 থেকে শুরু করে (Teacher 0.9127)। নিচ থেকে শুরু করলে কম হারানো সহজ। Teacher-কে IABR ছাড়িয়ে যায় কেবল 4 dB-এ। 2 dB-এও Teacher এগিয়ে (0.8541 বনাম 0.8463)।
- ~ Impairment-augmented training (IABC ছাড়াও) নিজেই Teacher-এর তুলনায় বড় লাভের উৎস: Abl1 −9.47 pp বনাম Teacher −13.40 pp।

---

## 3. E6 — OOD phase error (δmax ∈ {5 (in), 10 (OOD), 15 (OOD)}°)

### (1) কী test করা হয়েছে
Training-এ phase error ≤5° ছিল। E6 সেটা 10° আর 15° পর্যন্ত বাড়িয়ে দেখে, অর্থাৎ distribution shift-এর মধ্যে model কেমন চলে। SNR 0 আর 15, L = 3, 500 sample।

### (2) Theory
§0.5 অনুযায়ী 10°-এ distortion −16.9 dB আর 15°-এ −13.4 dB। SNR 15-এ 15°-এর distortion noise-এর চেয়ে বেশি, তাই পতন দেখা উচিত। Learned model-এর ক্ষেত্রে আরেকটা ঝুঁকি আছে: training-এ কখনো না দেখা impairment level-এ network-এর আচরণ (বিশেষ করে IABC MLP-এর) অনির্দিষ্ট হতে পারে। Research report এখানে "[HYPOTHESIS: degrades]" লিখেছিল, আর বলেছিল IABC "phase error beyond 5°"-এ degradation কমাবে (§27)।

### (3) Code
- Bank: cell 9, `BANK_SPECS['oodphase_d{d}_snr{snr}'] = ('std', dict(n=N_R, L=3, snr=snr, seed=3000+d+snr, phase_deg=d))`, d ∈ {10, 15}।
- Test: cell 22, `e6()`: banks = `['phase_d5_snr{snr}', 'oodphase_d10_snr{snr}', 'oodphase_d15_snr{snr}']`, x-label `['5 (in)', '10 (OOD)', '15 (OOD)']`।
- **লক্ষ্য করার মতো বিষয়: `e6()` `plot_sweep()` call করে না।** তাই **E6-এর কোনো figure নেই।** `outputs/figures/`-এ `E6_*.png` নেই, আর RESULTS.md-এর figure list-এও নেই। এটা একটা code omission, কোনো file হারিয়ে যাওয়া নয়। নিচে table থেকে বর্ণনা দেওয়া হলো।

### (4) কেন করা হয়েছে
Report §21: "E6 — OOD phase tail {10°,15°} — Genuine distribution-shift test, [GENUINE GAP]"। IABC-এর দাবি যদি সত্যি হয় ("impairment-এর physics বোঝে"), তাহলে এর উচিত training range-এর বাইরেও correction extrapolate করা। Abl1-এর তুলনায় এখানে বড় লাভ দেখা উচিত।

### (5) Result [FILE]

**SNR 0 dB, paper-style** (`E6_snr0.md`)

| model | 5 (in) | 10 (OOD) | 15 (OOD) |
|---|---|---|---|
| Teacher | 0.6571 | 0.6409 | 0.6212 |
| Student | 0.6536 | 0.6342 | 0.6069 |
| FNO | 0.6320 | 0.6084 | 0.5883 |
| DFT-SIC | 0.7000 | 0.6850 | 0.6580 |
| IABR_s0 | 0.6573 | 0.6433 | 0.6147 |
| E1_capacity_control | 0.6587 | 0.6427 | 0.6193 |
| Abl1_no_IABC | 0.6517 | 0.6437 | 0.6093 |

**SNR 0 dB, strict** (`E6_snr0_strict.md`): Teacher 0.5967 / 0.5793 / 0.5790; Student 0.5857 / 0.5657 / 0.5450; FNO 0.6143 / 0.5817 / 0.5707। বাকিগুলো paper-এর সমান।

**SNR 15 dB, paper-style** (`E6_snr15.md`)

| model | 5 (in) | 10 (OOD) | 15 (OOD) |
|---|---|---|---|
| Teacher | 0.8859 | 0.8620 | 0.8081 |
| Student | 0.8724 | 0.8456 | 0.7989 |
| FNO | 0.7743 | 0.7760 | 0.7449 |
| DFT-SIC | 0.9007 | 0.8870 | 0.8427 |
| IABR_s0 | 0.8510 | 0.8487 | 0.8043 |
| E1_capacity_control | 0.8527 | 0.8490 | 0.7987 |
| Abl1_no_IABC | 0.8607 | 0.8447 | 0.8017 |

**SNR 15 dB, strict** (`E6_snr15_strict.md`): Teacher 0.8593 / 0.8533 / 0.8000; Student 0.8480 / 0.8287 / 0.7877; FNO 0.7480 / 0.7620 / 0.7360। বাকিগুলো paper-এর সমান।

**অতিরিক্ত metric, SNR 15, 15° [FILE]:** pd_source: Teacher 0.6900, Student 0.6760, FNO 0.5833, DFT-SIC 0.7420, IABR 0.6893, E1 0.6820, Abl1 0.6833। RMSE: Teacher 0.3701, DFT-SIC 0.3420, IABR 0.3702, Abl1 0.3712। p95: Teacher 24.144, IABR 41.157, DFT-SIC 5.746। Teacher-এর drop: 15 → 5 → 5।

**Degradation 5° → 15° [DERIVED], paper-style:**

| model | SNR 0 Δ | SNR 15 Δ (5→10) | SNR 15 Δ (5→15) |
|---|---|---|---|
| Teacher | −0.0359 | −0.0239 | −0.0778 |
| Student | −0.0467 | −0.0268 | −0.0735 |
| FNO | −0.0437 | +0.0017 | −0.0294 |
| DFT-SIC | −0.0420 | −0.0137 | −0.0580 |
| **IABR_s0** | −0.0426 | **−0.0023** | **−0.0467** |
| E1 | −0.0394 | −0.0037 | −0.0540 |
| Abl1_no_IABC | −0.0424 | −0.0160 | −0.0590 |

### E6 figure — **নেই** (উপরে বলা কারণে)। Table থেকে কল্পনা করলে:
SNR 15-এ 5 → 10°-এ IABR প্রায় flat (0.8510 → 0.8487), আর Teacher ও Abl1 1.6–2.4 pp নামে। 10 → 15°-এ সবাই খাড়া ভাবে নামে (IABR −4.44 pp, Teacher −5.39 pp, DFT-SIC −4.43 pp)। 15°-এ IABR (0.8043), Teacher (0.8081), Abl1 (0.8017) আর E1 (0.7987) প্রায় এক বিন্দুতে মিলে যায়। DFT-SIC 0.8427-এ উপরে থাকে। FNO সবচেয়ে কম নামে (−2.94 pp), কিন্তু সবচেয়ে নিচে থাকে (0.7449)।

### (6) Comparison
**[MY ANALYSIS] Paired bootstrap:**

| তুলনা | SNR 0, 10° | SNR 0, 15° | SNR 15, 10° | SNR 15, 15° |
|---|---|---|---|---|
| IABR_s0 − Abl1 | −0.0003 [−0.0123, +0.0113] | +0.0053 [−0.0077, +0.018] | +0.004 [−0.0063, +0.015] | +0.0027 [−0.0087, +0.014] |
| E1 − Abl1 | −0.001 | +0.01 [−0.0017, +0.022] | +0.0043 | −0.003 [−0.015, +0.0093] |
| Abl3 − Abl1 | — | — | −0.0003 [−0.0113, +0.0103] | — |
| IABR_s0 − Teacher (strict) | +0.064 [+0.0463, +0.0827] | +0.0357 [+0.017, +0.055] | −0.0047 [−0.0183, +0.0097] | +0.0043 [−0.0123, +0.0207] |
| IABR_s0 − DFT-SIC | −0.0417 | −0.0433 | −0.0383 [−0.051, −0.0257] | −0.0383 [−0.0523, −0.024] |

- **IABC-এর কোনো পরিমাপযোগ্য OOD-phase লাভ নেই।** চারটা condition-এর কোনোটিতেই IABR − Abl1-এর CI শূন্য বাদ দেয় না। 15°-এ, যেখানে impairment সবচেয়ে বেশি, পার্থক্য মাত্র +0.27 pp।
- SNR 15, 15°-এ IABR আর Teacher কার্যত সমান (+0.43 pp, significant নয়)। SNR 0-এ strict-এ IABR-এর যে এগিয়ে থাকা দেখা যায়, সেটা আবার drop policy-র প্রভাব।
- 5→10°-এ IABR-এর যে "flat" আচরণ (−0.23 pp) দেখা যায়, সেটা আকর্ষণীয় মনে হলেও বিশ্বাসযোগ্য নয়। দুটো bank আলাদা (seed 3010+snr বনাম 1050+snr), আর E1 (−0.37 pp) আর FNO (+0.17 pp)-ও প্রায় flat। একই bank-এ paired করলে IABR − Abl1 = +0.004, significant নয়।
- **Hypothesis §27** ("IABC reduces degradation beyond 5°") এই data দিয়ে **সমর্থিত হয়নি**।

### (7) Verdict — **Failure (IABC-এর core claim-এর জন্য), তবে কোনো catastrophic break নেই**
- ✔ IABR OOD phase-এ ভেঙে পড়ে না। 15°-এ সে Teacher, Student আর FNO-র মতোই gracefully নামে (graceful degradation)।
- ✘ কিন্তু IABC module 5°-এর বাইরে কিছু যোগ করে না (Abl1-এর সাথে পার্থক্য noise-এর ভেতরে)।
- ✘ DFT-SIC 15°-এ IABR-এর চেয়ে ~3.8 pp এগিয়ে।
- ✘ E6-এর কোনো figure তৈরি হয়নি (code omission)।

---

## 4. গভীর বিশ্লেষণ: IABC কেন (প্রায়) কাজ করছে না?

### 4.1 [FILE] Training log-এর `pair` term: IABC আসলে কতটা correct করছে
`L_pair = ||y_corr − y_clean||² / ||y_clean||²` (cell 15)। Training-এর শেষে (step 20000, `TRAIN_*.json` → `final.pair`):

| model | IABC আছে? | λ_pair | final pair (normalized MSE) |
|---|---|---|---|
| Abl1_no_IABC | না (identity) | 0 | **0.008150874727871269** |
| IABR_s0 | হ্যাঁ | 0.5 | **0.0061575557617470624** |
| E1_capacity_control | হ্যাঁ | 0 | **0.06253684965893626** |

এর ব্যাখ্যা:
- Abl1-এর 0.00815 হলো **কোনো correction ছাড়া raw impaired Y আর clean Y-এর দূরত্ব**, অর্থাৎ training distribution-এ impairment-এর গড় "আকার"।
- IABR এটাকে 0.00616-এ নামিয়েছে [DERIVED]: impairment distortion energy-র মাত্র **~24% কমেছে** (≈1.2 dB)। বাকি ~76% থেকেই গেছে। তার ওপর IABR-এর log-এ pair term step 1000-এ 0.0064 থেকে step 20000-এ 0.0062, মানে training-এর শুরুতেই এটা সমতলে পৌঁছে গিয়েছিল আর আর উন্নতি হয়নি (`run_log.txt` [FILE])।
- E1 (pair supervision ছাড়া)-এর IABC এমন একটা transform শিখেছে যেটা clean থেকে **7.7 গুণ দূরে** (0.0625 বনাম raw 0.0082)। Log-এ দেখা যায় step 3000-এর পর E1-এর pair 0.0134 থেকে 0.0405-এ লাফিয়ে ওঠে আর তারপর 0.04–0.06-এ ঘোরাফেরা করে। তবুও E1-এর Pd IABR-এর প্রায় সমান (E3 0.6887 বনাম 0.6909, gain4 0.8003 বনাম 0.8033)। **মানে network-এর detection performance "clean-এর দিকে correction"-এর ওপর নির্ভর করে না।** E1-এর IABC সম্ভবত একটা global gain বা phase rotation, বা feature-dependent re-weighting শিখেছে। Normalized MSE এটাকে বড় error হিসেবে গোনে, কিন্তু trunk-এর (BatchNorm-এর কারণে) এতে কিছু যায় আসে না। (এটা একটা hypothesis। IABC-এর output সরাসরি পরীক্ষা করা হয়নি।)
- **Abl6** (RESULTS.md [FILE], 5000 step): λ_pair = 0, 0.1, 0.5, 1.0 জুড়ে gain2_pd 0.8040–0.8180 আর phase5_pd 0.8150–0.8303। কোনো monotonic trend নেই। Pair loss-এর weight বদলালে impairment robustness বদলায় না।

### 4.2 কেন correction এত দুর্বল — কারণগুলো (code-level)
1. **Feature-গুলো multipath আর noise-এ দূষিত।** `mag_r` হলো antenna-domain row-এর energy ratio। L=3–9 path থাকলে প্রতিটা row-এ path-গুলোর interference আর noise-ই energy-র ওঠানামার প্রধান কারণ। Gain error-এর ±0.46 natural-log amplitude (4 dB) তার তুলনায় ছোট। `res_r` phase residual নেওয়া হয় **শুধু dominant path**-এর steering সরিয়ে (`idx = argmax|Y|`)। L=3 path-এর gain CN(0,1/3) হওয়ায় দ্বিতীয় আর তৃতীয় path প্রায়ই তুলনীয় শক্তির হয়, তখন residual phase মূলত "অন্য path"-এর ছাপ বহন করে, impairment-এর নয়। ফলে per-element (δ, γ) estimation-টা ill-conditioned।
2. **Single snapshot।** Classical array calibration (Paulraj–Kailath 1985, Weiss–Friedlander 1990) অনেক snapshot-এর covariance ব্যবহার করে। এখানে মাত্র একটা Y আছে।
3. **Identifiability।** Notebook নিজেই বলে (cell 10): linear phase slope angle shift-এর সমান, তাই সেটা শনাক্ত করা যায় না। Common gain আর common phase-ও শনাক্ত করা যায় না। মানে 32টা element-এর মধ্যে কয়েকটা degree of freedom আগে থেকেই হারিয়ে যায়।
4. **দুর্বল training signal (§0.5)।** Training গড় DSR −18.6 dB, আর SNR −15 থেকে 24 dB uniform। বেশিরভাগ batch-এ noise impairment-এর চেয়ে বড়, তাই heatmap loss-এর gradient impairment-এর বিষয়ে প্রায় কিছুই "বলে না"।
5. **MLP ছোট আর zero-init** (10→16→2, ~250 param)। এটা identity-র কাছাকাছি থেকে যায়।

### 4.3 তাহলে SNR 15-এ 4 dB-এ লাভটা কোথা থেকে এল?
4 dB-এ gain distortion noise-এর চেয়ে 7 dB বড় (§0.5)। এখানে row বা column energy ratio feature-টা (`mag_r`, `mag_t`) প্রথমবারের মতো multipath বা noise-এর ওঠানামার ওপরে উঠে আসে। তখন ছোট MLP-ও একটা আংশিক amplitude equalization করতে পারে। এটা pair supervision ছাড়াও ঘটে, কারণ heatmap loss নিজেই সেই দিকে ঠেলে দেয়। E1 আর Abl3-এর ফল এর সাথে মেলে। Phase-এর ক্ষেত্রে একই জিনিস ঘটে না, কারণ phase residual feature-টা (dominant-path detrending) বেশি noisy, আর 15°-এ distortion (−13.4 dB) তখনও 4 dB gain-এর (−7.7 dB) চেয়ে 5.7 dB দুর্বল।

---

## 5. সব experiment একসাথে: সারাংশ table

| Test | Condition যেখানে impairment "active" | IABR − Abl1 (IABC লাভ) | IABR বনাম Teacher | IABR বনাম DFT-SIC | Verdict |
|---|---|---|---|---|---|
| E4 phase ≤5° | কার্যত কোথাও না (DSR ≤ −22.9 dB) | noise (−0.97 থেকে +1.3 pp) | SNR 15-এ −0.8 থেকে −1.9 pp (strict paired) | −4.1 থেকে −5.2 pp | Inconclusive / IABC fail |
| E5 gain 0.5–2 dB | SNR 15-এ 2 dB, সীমানায় | noise (−0.83 থেকে +0.27 pp) | 2 dB-এ −0.43 pp (ns) | −4.0 থেকে −4.9 pp | IABC fail |
| **E5 gain 4 dB (OOD)** | **SNR 15-এ জোরালো** | **+2.43 pp, significant, 3টা IABC model-এ পুনরাবৃত্ত** | **+2.93 pp, significant** | −3.53 pp | **Partial success** |
| E6 phase 10–15° (OOD) | SNR 15-এ 15° | noise (+0.27 থেকে +0.4 pp) | ≈ সমান | −3.8 pp | IABC fail, graceful degradation |

---

## Success (কোথায় ভালো)

1. **4 dB OOD gain error, SNR 15 dB — IABR-Net-এর সবচেয়ে ভালো ফল।** Pd 0.8033: Teacher-এর 0.7787, Student-এর 0.7611, FNO-র 0.7238 আর Abl1-এর 0.7790-এর চেয়ে বেশি। Clean থেকে মাত্র −6.67 pp হারিয়েছে (Teacher −13.40 pp), retention 92.3%, যা সব model-এর মধ্যে সর্বোচ্চ। Teacher-এর বিরুদ্ধে paired +2.93 pp [+1.17, +4.60]।
2. **IABC-এর একটা প্রকৃত, পুনরাবৃত্ত প্রভাব আছে, তবে শুধু জোরালো gain error-এ।** IABR, E1 আর Abl3, তিনটা IABC model-ই Abl1-এর চেয়ে +2.1 থেকে +2.6 pp এগিয়ে (CI শূন্য বাদ দেয়)।
3. **Physics-structured front-end dense front-end-এর চেয়ে অনেক ভালো।** gain4 SNR 15-এ E2_dense_frontend 0.6527, যেখানে IABR 0.8033, আর E2-এর params 457,430 বনাম IABR-এর 195,024।
4. **Impairment-augmented training নিজেই বড় লাভ দেয়।** Abl1 (IABC ছাড়া কিন্তু impaired data-য় train করা) 4 dB-এ −9.47 pp হারিয়েছে, যেখানে clean-trained Teacher হারিয়েছে −13.40 pp।
5. **কোনো catastrophic break নেই।** 15° OOD phase-এ IABR gracefully নামে (−4.67 pp at SNR 15), যা Teacher (−7.78 pp) আর DFT-SIC (−5.80 pp)-এর চেয়ে কম।
6. **Zero drop।** IABR সব condition-এ কোনো sample drop করে না। তাই strict Pd-তে (বিশেষ করে SNR 0-এ) Teacher আর Student-এর চেয়ে +3.6 থেকে +6.6 pp এগিয়ে। এটা impairment robustness নয়, detection completeness। তবে strict metric-এ এটা একটা বাস্তব সুবিধা।
7. **Seed-এর ওপর স্থিতিশীল।** gain_g2_snr15-এ std 0.0025, phase_d5_snr15-এ std 0.0010 (E11)।

## Failure / Weakness (কোথায় দুর্বল)

1. **Core claim ("impairment-aware design helps") বেশিরভাগ ক্ষেত্রে প্রমাণিত হয়নি।** E4-এর (≤5° phase) সব level, E5-এর 0.5–2 dB আর E6-এর 10–15° phase, এই সবখানে IABR বনাম Abl1-এর পার্থক্য noise-এর ভেতরে। SNR 15-এ 5°-এ IABR আসলে Abl1-এর চেয়ে 0.97 pp *নিচে* (ns)। Report §27-এর hypothesis ("IABC reduces degradation beyond 5°") সমর্থিত হয়নি।
2. **IABC distortion-এর মাত্র ~24% সরাতে পারে** (pair 0.00815 → 0.00616)। Pair loss (λ = 0.5) ছাড়া IABC "correction"-এর বদলে এমন একটা transform শেখে যেটা clean থেকে আরও দূরে (E1 pair 0.0625), অথচ Pd একই থাকে। অর্থাৎ **"IABC = impairment corrector" ব্যাখ্যাটা data দিয়ে সমর্থিত নয়।** এটা বরং একটা learned per-element re-weighting layer হিসেবে কাজ করছে।
3. **Pair loss-এর কোনো ভূমিকা নেই।** E1 (λ_pair = 0) আর IABR সব condition-এ সমান। Abl6 sweep-এও কোনো trend নেই। Loss-এর একটা term (আর তার জন্য paired clean data তৈরির খরচ) কার্যত অকেজো।
4. **Classical DFT-SIC (0 param) সব impairment condition-এ IABR-এর চেয়ে ভালো।** Paired diff −3.5 থেকে −5.2 pp, আর প্রতিটা CI শূন্যের নিচে। Thesis-এ এটাই সবচেয়ে বড় প্রশ্ন তৈরি করবে।
5. **Clean-accuracy gap-ই মূল সমস্যা।** SNR 15-এ clean অবস্থাতেই IABR Teacher-এর চেয়ে ~4.3 pp নিচে (0.8700 বনাম 0.9127)। IABR-এর "কম degradation"-এর একটা অংশ এই নিচু ceiling-এর ফল। Teacher-কে IABR ছাড়িয়ে যায় শুধু ≥4 dB gain-এ।
6. **Tail error খারাপ।** SNR 15-এ IABR-এর p95 error 5.1°–40.6°, যেখানে Teacher-এর 1.9°–2.2° (E4)। IABR কখনো drop না করায় তার ভুল peak-গুলো outlier হয়ে যায়। (Teacher-এর p95 drop-এর কারণে কিছুটা "পরিষ্কার" দেখায়, তাই এই তুলনাটা পুরোপুরি ন্যায্য নয়।)
7. **Test design-এর দুর্বলতা:**
   - E4-এর range (≤5°) physically noise-এর নিচে (DSR ≤ −22.9 dB), তাই IABC-কে যাচাই করার সুযোগই তৈরি করে না।
   - প্রতিটা impairment level-এর bank **আলাদা seed/scene**-এর, তাই curve-গুলোতে ±2 pp sampling noise থাকে। এর ফলে E4 আর E5-এ SNR 0-এ "impairment বাড়লে Pd বাড়ে"-এর মতো অর্থহীন trend দেখা যায়। প্রমাণ: DFT-SIC 1°-এ −1.8 pp, অথচ physics অনুযায়ী প্রভাব শূন্য।
   - প্রতি condition-এ মাত্র 500 sample। Abl1-এর একটাই seed। E6-এর কোনো figure নেই। 4 dB-এর multi-seed মাপা হয়নি। RMSE-এর কোনো figure নেই।
   - Robustness শুধু L = 3 আর SNR {0, 15}-এ মাপা হয়েছে। High-SNR (20–30 dB), যেখানে impairment সবচেয়ে বেশি ক্ষতি করে, সেটা বাদ পড়েছে।
8. **Impairment model সরল।** শুধু i.i.d. uniform phase/gain আছে। বাস্তবের quantized phase shifter (2–3 bit), mutual coupling, IQ imbalance, phase noise বা PA nonlinearity নেই।

## Improvement ideas (কোথায় উন্নতি দরকার)

**A. Test design ঠিক করা (সবচেয়ে সস্তা, সবচেয়ে জরুরি):**
1. **Paired impairment bank**: একই channel আর একই noise রেখে শুধু D_r আর D_t বদলানো (E10-এর nuisance bank-এর মতো)। এতে degradation curve থেকে scene variance দূর হবে, আর ±2 pp noise প্রায় শূন্যে নেমে আসবে।
2. **High-SNR robustness grid**: SNR 20, 25, 30 dB-এ E4/E5/E6 চালানো। Impairment distortion (−7 থেকে −23 dB) তখন noise-এর ওপরে উঠে আসবে, আর IABC-এর মূল্য (যদি থাকে) দেখা যাবে। বর্তমান SNR {0, 15} IABC-কে পরীক্ষা করার জন্য ভুল জায়গা।
3. **আরও চওড়া impairment range**: phase 20°, 30°, 45° (যা 2–3 bit phase shifter-এর quantization error-এর সমান) আর gain 3, 5, 6 dB। প্রতি condition-এ 2000+ sample, আর Abl1 সহ সব IABR variant-এর 3 seed।
4. **"Impairment penalty" metric**: একই scene-এ paired Pd(clean) − Pd(impaired), আর "equivalent SNR loss" (impairment কত dB SNR হারানোর সমান)। এতে ceiling effect আলাদা হয়ে যাবে।
5. `e6()`-এ `plot_sweep()` যোগ করা, আর RMSE ও pd_source-এর figure-ও বানানো।

**B. IABC-কে সত্যিকারের corrector বানানো:**
6. **Training impairment range বাড়ানো, আর impairment-কে SNR-এর সাথে couple করা**: high-SNR sample-গুলোতে বড় impairment দেওয়া, বা curriculum ব্যবহার করা, যাতে gradient-এ impairment-এর signal থাকে। বর্তমান training গড় DSR −18.6 dB বেশিরভাগ sample-এ noise-এ ডুবে যায়।
7. **Decision-directed বা iterative calibration** (model-based, deep-unfolded): (i) coarse heatmap থেকে L path estimate করা → (ii) `Ĥ = Σ α̂ a_r a_tᴴ` reconstruct করা → (iii) `D_r` আর `D_t`-এর জন্য per-element least-squares করা (rank-L Vandermonde structure ব্যবহার করে) → (iv) correct করে আবার detect করা। এটা 2–3 iteration unroll করা যায়। Dominant-path-only phase residual feature-এর চেয়ে এটা অনেক ভালো conditioned হবে।
8. **SVD-based feature**: `H̃ ≈ D_rᴴ A_r Σ A_tᴴ D_t`। Top-L singular vector আর estimated steering vector-এর element-wise ratio থেকে D-এর estimate পাওয়া যায়, যা IABC-এর input feature হিসেবে দেওয়া যায়।
9. **Pair loss-কে scale/phase-invariant করা** (global gain আর phase বাদ দিয়ে MSE, বা D-এর ওপর সরাসরি supervision, কারণ simulator `D_r` আর `D_t` return করে)। তাহলে "correction" শেখানোর সরাসরি signal থাকবে। বর্তমান E1-এর ফল দেখায় normalized MSE এমন কিছুকে শাস্তি দেয় যেটা Pd-এর জন্য অপ্রাসঙ্গিক।
10. **IABC-এর output পরীক্ষা করা (diagnostic)**: predict করা (δ̂, γ̂) আর আসল (ε, g)-এর মধ্যে correlation, প্রতিটা impairment level-এ। তাহলে সরাসরি জানা যাবে IABC কিছু estimate করছে কিনা।

**C. Architecture বা system level:**
11. **Classical-এর সাথে hybrid**: DFT-SIC সবখানে সবচেয়ে robust। IABC-এর calibration (বা ধারণা 7-এর iterative calibration) DFT-SIC-এর সামনে বসিয়ে দেখা যেতে পারে classical আরও ভালো হয় কিনা। অথবা NN-কে শুধু DFT-SIC-এর output refine করতে ব্যবহার করা যায়।
12. **প্রথমে clean-accuracy gap বন্ধ করা**: SNR 15-এ ~4.3 pp gap না কমালে impairment robustness-এর কোনো দাবিই absolute সংখ্যায় Teacher বা DFT-SIC-কে হারাতে পারবে না। এর মূল কারণ E3/Abl-2/E7/E8-এর বিশ্লেষণে (অন্য agent-এর report) দেখা দরকার। 16×16 coarse grid আর refinement-ই প্রধান সন্দেহ।
13. **বাস্তবসম্মত impairment model**: quantized phase shifter (deterministic, correlated error), mutual coupling (off-diagonal D, IABC-এর diagonal model যেটা ধরতে পারে না), আর array-এর মধ্যে correlated gain drift। এগুলোতেই learned correction-এর আসল মূল্য দেখা যেতে পারে, কারণ classical DFT-SIC তখন মডেল-mismatch-এ পড়বে।
14. **Thesis-এ claim-এর ভাষা**: বর্তমান data যা সমর্থন করে, তা হলো: "impairment-augmented training plus a physics-structured per-element front-end gives the best retention under strong (OOD) gain error at moderate SNR, but does not beat a classical DFT-SIC baseline and gives no measurable benefit for phase errors up to 15° or gain errors up to 2 dB"। এর চেয়ে বড় দাবি বর্তমান data সমর্থন করে না।

---

### Missing বা ambiguous জিনিসের তালিকা
- **E6 figure নেই**: `e6()` `plot_sweep()` call করে না (cell 22)।
- Task brief-এ "Ablation models 5000 steps" লেখা ছিল। ফাইল অনুযায়ী E1, Abl1 আর Abl3 20000 step, শুধু Abl6 5000 step।
- E5-এর "0" column = E4-এর 0° bank, আর E6-এর "5 (in)" = E4-এর 5° bank। তাই সংখ্যাগুলো ইচ্ছাকৃতভাবে একই, এটা কোনো ভুল নয়।
- RESULTS.md-এর table-এ "phase error max" বা "gain error max" হলো distribution-এর সীমা (uniform ±), আসল error-এর গড় নয়।
- 4 dB gain-এ Teacher, Student আর FNO-র OOD অবস্থা আরও খারাপ, কারণ তারা কোনো impairment-ই দেখেনি (clean-trained)। "4 (OOD)" label শুধু IABR-পরিবারের জন্য প্রযোজ্য। Teacher-এর জন্য 0.5 dB থেকেই সব OOD।
