# 04 — Scene difficulty: path count (E7), angular separation (E8), nuisance path (E10)

**Analyst group:** Scene difficulty
**কোন run:** IABR-Net full GPU run (Linux, TF 2.21.0, SMOKE_TEST = False, সব ৩৭টা experiment "done", মোট 3.07 h)
**পড়া source files:**
- Notebook source: `notebooks/IABR_Net_Full_Test_Suite.ipynb`। সংশ্লিষ্ট cells: cell 5, 6 (physics + simulator), cell 8, 9 (banks), cell 12 (IABR-Net), cell 15 (training), cell 17, 18 (evaluation engine + `metrics()`), cell 22 (`e7()`, `e8()`), cell 23 (`e10()`)
- Executed notebook: `notebooks/full_run.ipynb`। এই cells শুধু log line আর figure print করে। E7 চলেছে 176.1 থেকে 179.4 min (3.3 min), E8 চলেছে 179.4 থেকে 180.1 min (0.7 min), E10 চলেছে 180.6 থেকে 181.0 min (0.4 min)। কোনো error নেই।
- JSON: `outputs/results/E7_path_count_sweep.json`, `E8_angular_separation.json`, `E10_nuisance_path.json`। তিনটাতেই `_smoke_test = false`।
- Tables: `E7_snr0.md`, `E7_snr0_strict.md`, `E7_snr15.md`, `E7_snr15_strict.md`, `E7_pairing_snr0.md`, `E7_pairing_snr15.md`, `E8.md`, `E8_strict.md`, `E10.md`
- Figures: `E7_pathcount_snr0.png`, `E7_pathcount_snr15.png`, `E8_separation.png`। **E10-এর কোনো figure নেই**, শুধু table আছে।
- তুলনার context: `RESULTS.md`, `PROJECT_STATUS.md` (Notebook 4-এর nuisance result), `ResearchState/FINAL_RESEARCH_REPORT.md`, `RESEARCH_DESIGN.md`, `CANDIDATE_ARCHITECTURES.md`, আর `dldoa_dataset_generation.py`

সংখ্যা নিয়ে নিয়ম: যে সংখ্যাগুলো উদ্ধৃত করেছি সেগুলো table-এ যেমন আছে তেমনই দিয়েছি (4 decimal)। কিছু সংখ্যা table-এ নেই, যেমন `n_dropped`, `pd_source`, `rmse`, `p95`, `pairing_error`। সেগুলো JSON থেকে নিয়ে 4 decimal-এ দেখিয়েছি। "Δ" লেখা পার্থক্যগুলো আমি নিজে ফাইলের সংখ্যা থেকে বিয়োগ করে বের করেছি।

---

## 0. প্রথমে কিছু ভিত্তি (এই তিনটা test বোঝার জন্য দরকার)

### 0.1 "Scene difficulty" বলতে কী বোঝায়
এর আগের group-গুলো (E4/E5/E6) hardware impairment নিয়ে প্রশ্ন করেছে: antenna-র phase বা gain নষ্ট হলে model টেকে কিনা। এই group-এর প্রশ্ন আলাদা। **Hardware একদম নিখুঁত (phase = 0, gain = 0), কিন্তু scene-টাই কঠিন।** Scene তিনভাবে কঠিন হতে পারে:
1. **E7:** একসাথে অনেকগুলো path (L = 1 থেকে 10)।
2. **E8:** দুটো path খুব কাছাকাছি (30° থেকে 1° পর্যন্ত)।
3. **E10:** ৩টা "আসল" path, আর তার সাথে একটা অপ্রত্যাশিত interferer বা "nuisance" path, যার power −20, −10 বা 0 dB।

একটা গুরুত্বপূর্ণ ফল: এই তিন test-এর কোনো bank-এ impairment নেই (`make_standard_bank`-এ `phase_deg = 0`, `gain_db = 0`; `make_separation_bank` আর `make_nuisance_bank` `simulate()`-কে impairment ছাড়াই ডাকে)। তাই IABR-Net-এর মূল নতুনত্ব, মানে **IABC-v2 impairment correction, এখানে কিছুই করার সুযোগ পায় না।** এই test-গুলো আসলে IABR-এর **trunk + 16×16 coarse heatmap + 3×3 local-max decoder + 7×7 crop offset head**-কে পরীক্ষা করে। নিচে দেখা যাবে IABR_s0, E1_capacity_control (pair loss = 0) আর Abl1_no_IABC (IABC নেই) প্রায় হুবহু এক সংখ্যা দিচ্ছে। এটা এই যুক্তির সরাসরি প্রমাণ।

### 0.2 Metric-গুলো (cell 17 আর cell 18-এর `metrics()`)
- **`pd_paper`:** component-level Pd। AoA আর AoD আলাদা আলাদা component, 1° threshold, Hungarian matching দিয়ে গোনা হয়। কোনো sample-এ model যদি L-এর চেয়ে কম detection দেয়, তাহলে পুরো sample-টা **বাদ (dropped)** যায়। এটা original paper-এর convention।
- **`pd_strict`:** একই হিসাব, কিন্তু dropped sample-কে miss হিসেবে গোনা হয়। `strict_bad += 2*len(sel)` লাইনটা এটা করে।
- **`pd_source`:** একটা path তখনই সঠিক, যখন তার AoA আর AoD দুটোই 1°-এর মধ্যে। Dropped path-ও মোট সংখ্যায় ধরা হয়, অর্থাৎ এটাও strict ধরনের metric।
- **`pairing_error`:** matched path-গুলোর মধ্যে কত ভাগে **ঠিক একটা** component (AoA বা AoD) 1°-এর মধ্যে আছে, অন্যটা নেই। Dropped sample এখানে গোনা হয় না।
- **`n_dropped`:** কতগুলো sample বাদ গেছে।

**খুব জরুরি পর্যবেক্ষণ।** IABR-Net কখনো কোনো sample drop করে না। E7, E8, E10-এর সব condition-এ `n_dropped = 0`। কারণ `predict_iabr()` সবসময় `local_maxima(heat, K = 10)` থেকে top-L peak নেয়, ফলে ঠিক L-টা উত্তর দেয়। অন্যদিকে Teacher, Student আর FNO-এর blob detector কম blob পেলে sample drop হয়। তাই:
- IABR আর DFT-SIC-এর ক্ষেত্রে `pd_paper = pd_strict`।
- Teacher, Student আর FNO-এর `pd_paper` **ফোলানো**, কারণ কঠিন sample-গুলো বাদ দিয়ে শুধু সহজগুলোর উপর গড় করা হয়েছে।

তাই এই report-এ দুটো metric-ই পাশাপাশি দেখাব। IABR-এর সাথে ন্যায্য তুলনা হলো strict metric।

### 0.3 Beamspace resolution: কতটা কাছাকাছি হলে দুটো path আলাদা করা যায়
16-element half-wavelength ULA-তে DFT beamspace-এর একটা cell হলো u = cos θ domain-এ Δu = 2/16 = 0.125। মোটামুটি এটাই Rayleigh resolution। Angle domain-এ এটা কত ডিগ্রি হবে, তা θ-র উপর নির্ভর করে। Δθ ≈ Δu / sin θ, তাই আনুমানিক:
- broadside-এ (θ = 90°) 1 cell ≈ 7.2°
- θ = 40° বা 140°-এ ≈ 11°
- θ = 20° বা 160°-এ ≈ 21°

এগুলো আমার নিজের physics হিসাব, notebook-এ print করা নেই।

IABR-Net-এর heatmap **16×16**, মানে প্রতি cell-এ একটা pixel। Training target-এ Gaussian-এর σ = 0.8 cell (`heat_sigma_cells`)। Inference-এ peak খোঁজা হয় 3×3 `maximum_filter` দিয়ে। ফলে দুটো path-এর মধ্যে Chebyshev দূরত্ব মোটামুটি 2 cell-এর কম হলে দুটো মিলে একটা blob হয়ে যায়, আর দ্বিতীয় peak-টা হারিয়ে যায়। অন্য দিকে Teacher-এর output 256×256 (16 গুণ সূক্ষ্ম grid), আর DFT-SIC 512-point zero-padded DFT চালিয়ে successive interference cancellation (SIC) করে: শক্তিশালী path-টা বিয়োগ করে তারপর পরেরটা খোঁজে। **এই resolution-এর পার্থক্যটাই E7 (বড় L) আর E8 (ছোট separation)-এর ফল বোঝার মূল চাবি।**

### 0.4 Training distribution-এ path-গুলো কতটা কাছে আসতে পারে
`sample_paths()` মূল generator-এর `DG.generate_points(L, sep = π/6)` ব্যবহার করে। `dldoa_dataset_generation.py`-র line 46–103 অনুযায়ী এটা (AoD, AoA) plane-এ যেকোনো দুটো path-এর মধ্যে **কমপক্ষে 30° Euclidean দূরত্ব** রাখে। তাই training-এ দুটো path কখনো 30°-এর চেয়ে কাছে আসেনি। Teacher-ও একই generator থেকে train হয়েছে। এটা E8 বোঝার জন্য গুরুত্বপূর্ণ (নিচে দেখুন)।

---

## E7 — Path count sweep (L = 1, 2, 3, 4, 5, 6, 7, 8, 10)

### (1) কী test করা হয়েছে
Path-এর সংখ্যা L-কে 1 থেকে 10 পর্যন্ত বাড়ানো হয়েছে। প্রতিটা L-এর জন্য 500টা sample, SNR 0 dB (কঠিন) আর 15 dB (মাঝারি), impairment নেই। প্রতিটা model-এর Pd (paper আর strict) আর AoA/AoD pairing-error rate মাপা হয়েছে। Model-গুলো হলো Teacher, Student, FNO, DFT-SIC, IABR_s0, E1_capacity_control আর Abl1_no_IABC (cell 22-এর `ROBUST_MODELS`)।

### (2) Theory
- Total power স্থির থাকে, কারণ `alpha ~ CN(0, 1/L)`। তাই L বাড়লে **প্রতিটা path দুর্বল হয়**: প্রতি path-এর SNR মোটামুটি 10·log10(L) dB কমে যায়, L = 10-এ প্রায় 10 dB কম।
- Path বাড়লে beamspace-এ path-গুলো ঘন হয়। বিশেষ করে end-fire-এর কাছে u = cos θ সংকুচিত হয়ে যায়, তাই 30° angle-দূরত্বও মাত্র 1–2 DFT cell হতে পারে। দুটো blob একে অপরের sidelobe-এর উপর পড়ে, আর 16×16 grid-এ merge হয়ে যায়।
- **Pairing problem:** L-টা AoA আর L-টা AoD জানা থাকলেও কোন AoA কোন AoD-এর সাথে যায়, সেটা ঠিক করতে হয়। Joint 2D heatmap-এর সুবিধা হলো একটা peak মানেই একটা (AoA, AoD) জোড়া। তাই সঠিক pairing-এর নিশ্চয়তা আসে heatmap-এর গঠন থেকেই। IABR-এর design report (CANDIDATE_ARCHITECTURES §(b), "path-token coupling bottleneck") দাবি করে যে একই token map থেকে coarse heatmap, AoA offset আর AoD offset পড়া হয় বলে pairing রক্ষা পায়।
- **OOD:** training-এ L ∈ {1..9} ছিল (`CFG['train_L'] = (1, 9)`)। তাই শুধু L = 10 out-of-distribution। Cell 8-এর markdown স্পষ্টভাবে research report-এর একটা ভুল সংশোধন করে: report-এ L = 7, 8-কে OOD বলা ছিল, আসলে ওগুলো in-distribution।
- **Structural limit:** IABR-এর `k_slots = 10`। L = 10-এ সবগুলো slot ব্যবহার হয়। L > 10 হলে বর্তমান design-এ চলবে না।

### (3) Code (cell 22-এর `e7()`, cell 9-এর bank specs)
```python
Ls = [1, 2, 3, 4, 5, 6, 7, 8, 10]
banks = [f'phase_d0_snr{snr}' if L == 3 else f'L{L}_snr{snr}' for L in Ls]
```
- L = 3-এর জন্য আলাদা bank বানানো হয়নি। E4-এর `phase_d0_snr{snr}` bank (L = 3, phase = 0, seed = 1000 + snr) পুনরায় ব্যবহার হয়েছে। তাই JSON-এ L = 3-এর key-র নাম `phase_d0_snr0` আর `phase_d0_snr15`, `L3_...` নয়। বাকি L-গুলোর bank হলো `L{L}_snr{snr}`, seed = 4000 + 10·L + snr, n = 500।
- `sweep()` প্রতিটা model আর bank-এর জন্য `metrics(predict(m, bname), b)` চালায়, তারপর paper আর strict দুটো table লেখে। `e7()` এর উপর আলাদা করে `pairing_error` table লেখে এবং `plot_sweep(... key = 'pd_paper')` দিয়ে figure আঁকে।
- IABR-এর decoding (`predict_iabr`) এভাবে হয়: `sigmoid(heat)`, তারপর `local_maxima(heat, K = 10)` (3×3 `maximum_filter`, `mode = 'wrap'`), তারপর top-L cell নেওয়া, তারপর learned offset `dq, dp` যোগ করা, তারপর `cells_to_angles`। **L-টা oracle হিসেবে দেওয়া থাকে**, সব model-কেই, যেটা paper-এর convention।

### (4) কেন করা হয়েছে
Research report-এর test plan (FINAL_RESEARCH_REPORT line 300) বলে: *"E7 — Path-count sweep ... Tests path-token pairing at/near capacity"*। সেখানে প্রশ্ন ছিল দুটো:
- (ক) L বাড়লে IABR-এর 16×16 coarse grid আর 10-slot top-K design ভেঙে পড়ে কিনা।
- (খ) pairing, মানে AoA-AoD জোড়া মেলানো, Teacher-এর চেয়ে ভালো থাকে কিনা।

RESEARCH_DESIGN line 101-এ আগে থেকেই সতর্ক করা ছিল: *"If the coarse grid is too coarse, closely-spaced sources ... could collapse into the same coarse bin"*। এই ঝুঁকি multipath বা L-sweep axis-এই প্রথম ধরা পড়ার কথা।

### (5) Result: হুবহু সংখ্যা

**E7, SNR 0 dB, paper-style Pd (`tables/E7_snr0.md`)**

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 (OOD) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher | 0.7650 | 0.7040 | 0.6298 | 0.6029 | 0.5582 | 0.5398 | 0.5055 | 0.4614 | 0.4507 |
| Student | 0.7380 | 0.6761 | 0.6115 | 0.5825 | 0.5620 | 0.5253 | 0.4896 | 0.4547 | 0.4198 |
| FNO | 0.7500 | 0.6744 | 0.6079 | 0.5588 | 0.5088 | 0.4733 | 0.4401 | 0.4260 | 0.3860 |
| DFT-SIC | 0.8260 | 0.7470 | 0.6790 | 0.6358 | 0.5946 | 0.5495 | 0.5329 | 0.4931 | 0.4470 |
| IABR_s0 | 0.7940 | 0.7000 | 0.6307 | 0.5940 | 0.5396 | 0.5030 | 0.4723 | 0.4338 | 0.3839 |
| E1_capacity_control | 0.7960 | 0.6930 | 0.6297 | 0.5877 | 0.5398 | 0.5070 | 0.4710 | 0.4340 | 0.3854 |
| Abl1_no_IABC | 0.7970 | 0.6965 | 0.6287 | 0.5860 | 0.5382 | 0.5065 | 0.4670 | 0.4339 | 0.3870 |

**E7, SNR 0 dB, strict Pd (`tables/E7_snr0_strict.md`)**

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 (OOD) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher | 0.7650 | 0.6730 | 0.5643 | 0.4968 | 0.4198 | 0.3368 | 0.2477 | 0.1716 | 0.0676 |
| Student | 0.7380 | 0.6450 | 0.5430 | 0.4660 | 0.3788 | 0.2773 | 0.2076 | 0.1319 | 0.0487 |
| FNO | 0.7500 | 0.6650 | 0.5897 | 0.5130 | 0.4406 | 0.3607 | 0.2896 | 0.2087 | 0.1104 |
| DFT-SIC | 0.8260 | 0.7470 | 0.6790 | 0.6358 | 0.5946 | 0.5495 | 0.5329 | 0.4931 | 0.4470 |
| IABR_s0 | 0.7940 | 0.7000 | 0.6307 | 0.5940 | 0.5396 | 0.5030 | 0.4723 | 0.4338 | 0.3839 |
| E1_capacity_control | 0.7960 | 0.6930 | 0.6297 | 0.5877 | 0.5398 | 0.5070 | 0.4710 | 0.4340 | 0.3854 |
| Abl1_no_IABC | 0.7970 | 0.6965 | 0.6287 | 0.5860 | 0.5382 | 0.5065 | 0.4670 | 0.4339 | 0.3870 |

**E7, SNR 15 dB, paper-style Pd (`tables/E7_snr15.md`)**

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 (OOD) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher | 0.9430 | 0.9383 | 0.9127 | 0.8860 | 0.8634 | 0.8331 | 0.8155 | 0.7867 | 0.7494 |
| Student | 0.9240 | 0.9122 | 0.9013 | 0.8653 | 0.8460 | 0.8158 | 0.7920 | 0.7689 | 0.7318 |
| FNO | 0.8180 | 0.8278 | 0.8079 | 0.7793 | 0.7493 | 0.7189 | 0.6887 | 0.6604 | 0.6188 |
| DFT-SIC | 0.9450 | 0.9250 | 0.9183 | 0.8865 | 0.8626 | 0.8372 | 0.8199 | 0.7951 | 0.7549 |
| IABR_s0 | 0.9280 | 0.9015 | 0.8700 | 0.8407 | 0.7988 | 0.7660 | 0.7381 | 0.7113 | 0.6694 |
| E1_capacity_control | 0.9180 | 0.9065 | 0.8757 | 0.8353 | 0.7976 | 0.7567 | 0.7353 | 0.7095 | 0.6699 |
| Abl1_no_IABC | 0.9270 | 0.9115 | 0.8737 | 0.8367 | 0.7968 | 0.7575 | 0.7270 | 0.7061 | 0.6639 |

**E7, SNR 15 dB, strict Pd (`tables/E7_snr15_strict.md`)**

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 (OOD) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher | 0.9430 | 0.9270 | 0.8890 | 0.8240 | 0.7978 | 0.7498 | 0.6801 | 0.6215 | 0.5426 |
| Student | 0.9240 | 0.8940 | 0.8490 | 0.7788 | 0.7496 | 0.6885 | 0.6146 | 0.5429 | 0.4303 |
| FNO | 0.8180 | 0.8245 | 0.7740 | 0.7170 | 0.6474 | 0.5737 | 0.4931 | 0.4333 | 0.3465 |
| DFT-SIC | 0.9450 | 0.9250 | 0.9183 | 0.8865 | 0.8626 | 0.8372 | 0.8199 | 0.7951 | 0.7549 |
| IABR_s0 | 0.9280 | 0.9015 | 0.8700 | 0.8407 | 0.7988 | 0.7660 | 0.7381 | 0.7113 | 0.6694 |
| E1_capacity_control | 0.9180 | 0.9065 | 0.8757 | 0.8353 | 0.7976 | 0.7567 | 0.7353 | 0.7095 | 0.6699 |
| Abl1_no_IABC | 0.9270 | 0.9115 | 0.8737 | 0.8367 | 0.7968 | 0.7575 | 0.7270 | 0.7061 | 0.6639 |

**E7 pairing-error rate, SNR 0 dB (`tables/E7_pairing_snr0.md`)**

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 (OOD) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher | 0.2820 | 0.3138 | 0.3445 | 0.3368 | 0.3293 | 0.3072 | 0.3149 | 0.3098 | 0.3440 |
| Student | 0.2840 | 0.3208 | 0.3356 | 0.3425 | 0.3217 | 0.3119 | 0.3133 | 0.3319 | 0.3431 |
| FNO | 0.3040 | 0.3408 | 0.3306 | 0.3301 | 0.3284 | 0.3456 | 0.3113 | 0.3041 | 0.2951 |
| DFT-SIC | 0.2280 | 0.2600 | 0.3153 | 0.3165 | 0.3044 | 0.3123 | 0.2891 | 0.2888 | 0.2824 |
| IABR_s0 | 0.2600 | 0.2960 | 0.3333 | 0.3060 | 0.3032 | 0.2913 | 0.2846 | 0.2835 | 0.2838 |
| E1_capacity_control | 0.2760 | 0.3000 | 0.3273 | 0.3095 | 0.3028 | 0.2973 | 0.2820 | 0.2830 | 0.2816 |
| Abl1_no_IABC | 0.2620 | 0.3050 | 0.3373 | 0.3100 | 0.2972 | 0.3057 | 0.2809 | 0.2812 | 0.2736 |

**E7 pairing-error rate, SNR 15 dB (`tables/E7_pairing_snr15.md`)**

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 (OOD) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher | 0.0820 | 0.1093 | 0.1198 | 0.1538 | 0.1719 | 0.1811 | 0.1771 | 0.1962 | 0.1967 |
| Student | 0.1080 | 0.1449 | 0.1536 | 0.1750 | 0.1869 | 0.1931 | 0.1892 | 0.2072 | 0.2099 |
| FNO | 0.3000 | 0.2540 | 0.2575 | 0.2576 | 0.2690 | 0.2698 | 0.2713 | 0.2691 | 0.2568 |
| DFT-SIC | 0.0980 | 0.1140 | 0.1233 | 0.1480 | 0.1436 | 0.1730 | 0.1649 | 0.1718 | 0.1894 |
| IABR_s0 | 0.1240 | 0.1370 | 0.1560 | 0.1575 | 0.1592 | 0.1707 | 0.1729 | 0.1690 | 0.1784 |
| E1_capacity_control | 0.1360 | 0.1310 | 0.1367 | 0.1655 | 0.1640 | 0.1847 | 0.1729 | 0.1690 | 0.1758 |
| Abl1_no_IABC | 0.1300 | 0.1270 | 0.1460 | 0.1625 | 0.1648 | 0.1830 | 0.1809 | 0.1713 | 0.1782 |

**JSON থেকে অতিরিক্ত সংখ্যা (table-এ নেই)**

Teacher-এর `n_dropped` (প্রতি condition 500 sample-এর মধ্যে):

| SNR | L=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| 0 dB | 0 | 22 | 52 | 88 | 124 | 188 | 255 | 314 | **425** |
| 15 dB | 0 | 6 | 13 | 35 | 38 | 50 | 83 | 105 | **138** |

- অন্যদের `n_dropped` at L = 10: Student 442 (SNR 0) আর 206 (SNR 15); FNO 357 আর 220। DFT-SIC, IABR_s0, E1 আর Abl1-এর সব condition-এ 0।

SNR 15 dB-এ IABR_s0, Teacher আর DFT-SIC-এর `pd_source` (দুটো component-ই ঠিক), `rmse` আর `p95` (ডিগ্রি):

| L | IABR pd_source | Teacher pd_source | DFT-SIC pd_source | IABR rmse | Teacher rmse | DFT-SIC rmse | IABR p95 | Teacher p95 | DFT-SIC p95 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.866 | 0.902 | 0.896 | 0.2737 | 0.2283 | 0.2168 | 1.5110 | 1.1367 | 1.1258 |
| 2 | 0.833 | 0.873 | 0.868 | 0.2695 | 0.2591 | 0.2423 | 2.1820 | 1.2543 | 1.6107 |
| 3 | 0.792 | 0.8307 | 0.8567 | 0.2907 | 0.2853 | 0.2575 | 5.1354 | 1.8999 | 2.0948 |
| 4 | 0.762 | 0.7525 | 0.8125 | 0.3085 | 0.3040 | 0.2749 | 42.7337 | 2.5736 | 3.0935 |
| 5 | 0.7192 | 0.7184 | 0.7908 | 0.3301 | 0.3148 | 0.2937 | 66.2579 | 3.8271 | 7.1468 |
| 6 | 0.6807 | 0.6683 | 0.7507 | 0.3361 | 0.3245 | 0.2981 | 70.5232 | 6.6016 | 9.1758 |
| 7 | 0.6517 | 0.6063 | 0.7374 | 0.3352 | 0.3375 | 0.3161 | 73.1261 | 15.3959 | 25.9911 |
| 8 | 0.6268 | 0.544 | 0.7093 | 0.3449 | 0.3440 | 0.3272 | 79.0200 | 42.7650 | 40.4698 |
| 10 | 0.5802 | 0.4714 | 0.6602 | 0.3548 | 0.3594 | 0.3403 | 76.0282 | 52.5240 | 47.1792 |

(Teacher-এর `rmse` আর `p95` শুধু non-dropped sample থেকে হিসাব করা, তাই তার দিকে একটু পক্ষপাত আছে।)

### (6) Comparison

**IABR বনাম Teacher (paper-style, SNR 15 dB):** Δ হলো IABR − Teacher।

| L=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| −0.0150 | −0.0368 | −0.0427 | −0.0453 | −0.0646 | −0.0671 | −0.0774 | −0.0754 | −0.0800 |

L বাড়ার সাথে পিছিয়ে থাকার পরিমাণ ধারাবাহিকভাবে বাড়ছে, 1.5 pp থেকে 8 pp পর্যন্ত। DFT-SIC-এর তুলনায়ও একই pattern: L1-এ −0.0170, L10-এ −0.0855। L = 1-এ ফাঁক মাত্র 1.5 pp; E3-এ SNR 15 dB-এ দেখা ফাঁকটাও (0.8662 বনাম 0.8994) একই ধরনের। অর্থাৎ বড় L-এ ফাঁক বেড়ে যাওয়ার পেছনে আছে multi-path resolution সমস্যা।

**IABR বনাম Teacher (paper-style, SNR 0 dB):**
- L = 1-এ IABR আগে: 0.7940 বনাম 0.7650, **+0.0290**।
- L = 2 থেকে 4 পর্যন্ত প্রায় সমান: −0.0040, +0.0009, −0.0089।
- L ≥ 5 থেকে পিছিয়ে: L5 −0.0186, L6 −0.0368, L10 −0.0668।
- কম SNR আর একটা path-এ IABR-এর SNR head আর focal-loss heatmap noise-এ কিছুটা ভালো কাজ করছে বলে মনে হয়।

**Strict metric-এ চিত্র উল্টে যায়:**
- SNR 0 dB-এ IABR সব L-এ Teacher-এর চেয়ে ভালো। L10-এ 0.3839 বনাম 0.0676, কারণ Teacher 425/500 sample drop করেছে। L8-এ 0.4338 বনাম 0.1716।
- SNR 15 dB-এ L ≥ 4 থেকে IABR আগে। L4-এ 0.8407 বনাম 0.8240, L7-এ 0.7381 বনাম 0.6801, L10-এ 0.6694 বনাম 0.5426। L ≤ 3-এ Teacher আগে (L3-এ 0.8890 বনাম 0.8700)।
- `pd_source`-এও (SNR 15) L ≥ 4-এ IABR ≥ Teacher, যেমন L10-এ 0.5802 বনাম 0.4714।
- **কিন্তু DFT-SIC strict metric-এ প্রতিটা L-এ, দুই SNR-এই, সবার উপরে।** L10-এ SNR 15 dB-এ 0.7549, SNR 0 dB-এ 0.4470।

**OOD ধাপ (L = 8 থেকে L = 10), paper-style:**

| | IABR | Teacher | DFT-SIC | Student |
|---|---|---|---|---|
| SNR 15 dB | −0.0419 | −0.0373 | −0.0402 | −0.0371 |
| SNR 0 dB | −0.0499 | −0.0107 | −0.0461 | — |

- IABR-এর কোনো "cliff" নেই। ঢাল DFT-SIC-এর মতোই মসৃণ, আর DFT-SIC-এর কোনো training distribution নেই। তাই L = 10-এর পতন প্রায় পুরোটাই physics থেকে আসে: প্রতি path-এর power কম আর path-গুলো ঘন। OOD বলে বিশেষ কোনো ক্ষতি দেখা যাচ্ছে না।
- SNR 0-এ Teacher-এর ছোট পতন (−0.0107) বিশ্বাসযোগ্য নয়। L10-এ তার মাত্র 75টা sample টিকে ছিল, আর সেগুলোই সবচেয়ে সহজ।

**Pairing error:**
- SNR 0 dB-এ IABR (0.2600 থেকে 0.3333) সব L-এ Teacher-এর (0.2820 থেকে 0.3445) চেয়ে কম। L10-এ 0.2838 বনাম 0.3440। DFT-SIC কম L-এ সবচেয়ে ভালো (L1 0.2280, L2 0.2600); L ≥ 7-এ IABR আর DFT-SIC প্রায় সমান (L10 0.2838 বনাম 0.2824)।
- SNR 15 dB-এ L ≤ 3-এ IABR Teacher-এর চেয়ে খারাপ (L1 0.1240 বনাম 0.0820)। L ≥ 5-এ ভালো (L5 0.1592 বনাম 0.1719, L8 0.1690 বনাম 0.1962, L10 0.1784 বনাম 0.1967)। L = 8 আর 10-এ IABR পরিবার DFT-SIC-এর চেয়েও কম (L10-এ DFT-SIC 0.1894)।
- **সতর্কতা ১:** L = 1-এ "pairing error" আসলে কোনো pairing ভুল হতে পারে না, কারণ জোড়া মাত্র একটা। এটা শুধু দেখায় যে একটা component 1°-এর বাইরে গেছে, মানে sub-cell precision-এর দুর্বলতা। তাই এই metric-এ precision আর pairing মিশে আছে।
- **সতর্কতা ২:** Teacher-এর pairing error শুধু non-dropped sample-এর উপর হিসাব করা।

**IABR পরিবারের ভেতরের তুলনা:** IABR_s0, E1_capacity_control আর Abl1_no_IABC-এর পার্থক্য প্রায় সবখানে ≤ 0.01। উদাহরণ: SNR 15, L10-এ 0.6694, 0.6699, 0.6639। অর্থাৎ **IABC-v2 আর pairing loss path-count behaviour-এ কোনো প্রভাব ফেলে না।** এটা প্রত্যাশিত, কারণ bank-এ impairment নেই।

**FNO:** প্রায় সবখানে সবার নিচে। SNR 15 L1-এ 0.8180, L10-এ 0.6188। শুধু strict SNR 0-এ বড় L-এ Teacher আর Student-এর উপরে, কারণ FNO কম drop করে।

**Statistical note:** প্রতিটা condition-এ 500টা sample আছে। L = 1-এ 1000টা component থেকে Pd-এর standard error মোটামুটি √(0.8·0.2/1000) ≈ 0.013। তাই L ≤ 3-এ 0.01–0.02 মাপের পার্থক্য (যেমন SNR 0-এ IABR বনাম Teacher) পরিসংখ্যানগতভাবে দুর্বল প্রমাণ। বড় L-এ 5–8 pp-এর ফাঁক স্পষ্টভাবে বাস্তব। এটা আমার হিসাব; notebook কোনো confidence interval দেয় না।

**p95 tail:** SNR 15-এ L ≥ 4 থেকে IABR-এর p95 লাফিয়ে 42.7337° থেকে 79.0200°-এ ওঠে। DFT-SIC একই L-এ 3.0935° থেকে 40.4698°, আর DFT-SIC-ও কোনো sample drop করে না। এর মানে IABR বড় L-এ প্রায়ই পুরোপুরি **ভুল জায়গায় peak** দেয়। কারণ merge হওয়া blob-এর বদলে দ্বিতীয় peak হিসেবে আসে noise-এর একটা local maximum। IABR "উত্তর দেব না" বলতে পারে না, সবসময় L-টা উত্তর দেয়।

### (7) Verdict: **Mixed**
- **Success:**
  - কম SNR আর কম L-এ Teacher-এর সমান বা ভালো (SNR 0, L1: +0.0290)।
  - Strict আর `pd_source` metric-এ, মাঝারি থেকে বড় L-এ Teacher-এর চেয়ে ভালো, কারণ IABR কখনো drop করে না।
  - OOD L = 10-এ কোনো cliff নেই।
  - কম SNR আর বড় L-এ pairing error Teacher-এর চেয়ে কম।
- **Failure:**
  - SNR 15 dB paper-style metric-এ L বাড়ার সাথে Teacher আর DFT-SIC-এর চেয়ে ধারাবাহিকভাবে পিছিয়ে পড়ে, L10-এ 8 pp পর্যন্ত।
  - L ≥ 4-এ বড় বড় ভুল (gross error) বাড়ে।
  - DFT-SIC, যার 0টা parameter, প্রতিটা L-এ strict metric-এ IABR-এর উপরে।
  - L = 1-এ high-SNR precision-ও Teacher আর DFT-SIC-এর চেয়ে খারাপ (rmse 0.2737 বনাম 0.2283 আর 0.2168)।

### Figure: `E7_pathcount_snr0.png`
- **কী দেখায়:** x-axis-এ L = 1..8, 10 (OOD); y-axis-এ `pd_paper`; সাতটা model-এর line।
- সব line বাম থেকে ডানে প্রায় সরলরেখায় নামছে, 0.74–0.83 থেকে 0.38–0.45-এ।
- লাল DFT-SIC line পুরো পথ সবার উপরে। শুধু L = 10-এ নীল Teacher line (0.4507) তাকে সামান্য ছুঁয়ে ফেলে।
- IABR পরিবারের তিনটা line (বেগুনি IABR_s0, বাদামি E1, গোলাপি Abl1) একে অপরের উপর পড়ে আছে, আলাদা করে প্রায় দেখা যায় না।
- L = 1-এ IABR পরিবার Teacher-এর উপরে (প্রায় 0.795 বনাম 0.765)। L = 2–4-এ Teacher-এর গায়ে গায়ে চলে। L ≥ 5 থেকে Teacher-এর নিচে নামে। L = 10-এ FNO-এর সাথে একেবারে নিচে (প্রায় 0.384)।
- **মানে:** কম SNR-এ IABR মাঝারি-সহজ scene-এ ভালো, কিন্তু path বাড়লে paper-style metric-এ Teacher আর DFT-SIC-এর চেয়ে দ্রুত পড়ে যায়। এই figure-টা paper-style, তাই Teacher-এর drop সুবিধা এর ভেতরে লুকানো আছে। Strict table দেখলে উল্টো চিত্র পাওয়া যায়।

### Figure: `E7_pathcount_snr15.png`
- **কী দেখায়:** একই axes, SNR 15 dB।
- Teacher (নীল) আর DFT-SIC (লাল) প্রায় এক line হয়ে উপরে থাকে, 0.94 থেকে 0.75-এ নামে। Student (কমলা) তার সামান্য নিচে।
- IABR পরিবারের তিনটা line প্রায় একসাথে L = 1-এ 0.92–0.93 থেকে শুরু হয়, কিন্তু Teacher আর DFT-SIC-এর চেয়ে **খাড়া ঢালে** নেমে L = 10-এ প্রায় 0.665–0.67-এ পৌঁছায়। ফলে দুই গোষ্ঠীর মধ্যে একটা ক্রমশ চওড়া "wedge" তৈরি হয়।
- FNO (সবুজ) সবার নিচে, আর তার L = 1-এর বিন্দু L = 2-এর চেয়ে নিচে (0.8180 বনাম 0.8278)। এটা একটা অস্বাভাবিক বিন্দু।
- **মানে:** high SNR-এ noise আর সমস্যা নয়, তাই যা বাকি থাকে তা হলো **resolution আর sub-cell precision**। এখানেই IABR-এর 16×16 grid-এর সীমা স্পষ্ট হয়ে যায়।

---

## E8 — Angular separation sweep (30°, 20°, 10°, 5°, 3°, 2°, 1°)

### (1) কী test করা হয়েছে
প্রতিটা scene-এ ঠিক 2টা path, SNR 15 dB, impairment নেই। দ্বিতীয় path-এর **AoA আর AoD দুটোই** প্রথম path থেকে s ডিগ্রি সরানো, যেখানে s = 30, 20, 10, 5, 3, 2, 1। প্রতিটা s-এ 500টা sample। প্রশ্ন একটাই: দুটো path কতটা কাছাকাছি এলে model আর তাদের আলাদা করতে পারে না।

### (2) Theory
- **Rayleigh limit:** array aperture সীমিত (N = 16), তাই classical beamforming দুটো উৎসকে তখনই আলাদা করতে পারে যখন তারা u-domain-এ মোটামুটি ≥ 2/N = 0.125 দূরে থাকে, মানে প্রায় 1 DFT cell। Angle domain-এ এটা broadside-এ প্রায় 7°, 40°-এ প্রায় 11°, 20°-এ প্রায় 21° (§0.3)। তাই 10° separation অনেক sample-এর জন্য Rayleigh limit-এর আশেপাশে বা তার নিচে, আর 5° বা তার কম প্রায় সবসময় limit-এর নিচে।
- **Super-resolution:** SIC-এর মতো model-based পদ্ধতি এই limit কিছুটা ভাঙতে পারে। শক্তিশালী path-টা খুঁজে তার reconstructed contribution বিয়োগ করলে residual-এ দুর্বল path-টা বের হয়ে আসে।
- **Grid resolution:** একটা heatmap network যতই ভালো হোক, দুটো আলাদা peak আঁকার মতো pixel না থাকলে দুটো path বের করতে পারবে না। IABR-এর grid 16×16, Teacher-এর 256×256।
- **Training distribution:** training-এ Euclidean min separation ছিল 30° (§0.4)। E8-এ দুটো axis-ই s পরিমাণ সরানো, তাই Euclidean দূরত্ব s·√2। s = 30-এ 42.4° (in-distribution); s = 20-এ 28.3° (সীমার ঠিক নিচে, সামান্য OOD); s ≤ 10-এ ≤ 14.1° (পরিষ্কার OOD)। **Learned model-গুলো, মানে Teacher, Student, FNO আর IABR সবাই, s ≤ 20°-এ এমন scene দেখছে যা training-এ কখনো আসেনি।** DFT-SIC-এর training নেই, তাই এই সমস্যা তার নেই। এই কারণটা আমি code আর generator পড়ে বের করেছি; notebook নিজে এটা বলে না।
- **1° threshold-এর artefact:** s = 1°-এ দুটো path এত কাছে যে দুটোর মাঝখানে একটা মাত্র estimate দিলেও সেটা প্রতিটা path-এর প্রতিটা component-এর 0.5°-এর মধ্যে থাকে। Hungarian matching ওই একটা estimate দিয়ে একটা path-কে match করে। দ্বিতীয় estimate যদি অন্য কোথাও পড়ে, তাহলে Pd প্রায় 0.5 হয়। তাই **s ≤ 3°-এ Pd আবার বাড়ে। এটা resolution-এর উন্নতি নয়, metric-এর artefact।** Research design (RESEARCH_DESIGN line 441) আগে থেকেই বলেছিল যে 5–10°-এর নিচের ফল "diagnostic-only", মানে কোনো defect হিসেবে দেখানো যাবে না।

### (3) Code
- **Bank:** `make_separation_bank(name, n, sep_deg, snr, seed)` (cell 9)। `psi1` আর `phi1` আসে U(20°, 160° − s) থেকে; `psi = [psi1, psi1 + d]`, `phi = [phi1, phi1 + d]`; `alpha ~ CN(0, 1/2)` দুটো path-এর জন্য, শক্তিশালীটা আগে। Seed = 5000 + s, SNR = 15, n = 500, আর `simulate()` চলে impairment ছাড়া। দুটো path-এর power-এর অনুপাত random, তাই কোনো কোনো sample-এ দ্বিতীয় path অনেক দুর্বল। এটাও কঠিনতা বাড়ায়।
- **Test:** `e8()` (cell 22):
  ```python
  seps = [30, 20, 10, 5, 3, 2, 1]
  r, ms = sweep('E8', ..., banks, 'separation (deg)', seps)
  plot_sweep(r, ms, banks, seps, 'separation (deg)', 'E8_separation.png', key='pd_source')
  ```
  **খেয়াল করুন:** table-এ আছে `pd_paper` আর `pd_strict`, কিন্তু **figure আঁকা হয়েছে `pd_source` দিয়ে**, মানে দুটো component-ই ঠিক হতে হবে আর dropped path miss হিসেবে গোনা হবে। তাই figure-এর সংখ্যা table-এর সাথে মিলবে না।

### (4) কেন করা হয়েছে
- FINAL_RESEARCH_REPORT line 301 বলে: *"E8 — Angular separation Δθ down to 1° ... [HYPOTHESIS: collapses <5–10°] ... resolution-limit caveat applies"*।
- CANDIDATE_ARCHITECTURES §9 failure mode 3 (line 612) আগেই অনুমান করেছিল যে দুটো peak একই 16×16 coarse bin-এ পড়তে পারে, আর তখন crop-refinement head মাত্র একটা anchor দেখবে বলে দুটো angle ফিরিয়ে আনতে পারবে না।
- E8 সরাসরি এই failure mode যাচাই করে। Naoumi-র paper-এ নাম দেওয়া "closely-spaced-target stress test" gap-টাও এটা পূরণ করে।

### (5) Result: হুবহু সংখ্যা

**E8, paper-style Pd (`tables/E8.md`)** (L = 2, SNR 15 dB)

| model | 30 | 20 | 10 | 5 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| Teacher | 0.9888 | 0.9879 | 0.9322 | 0.6811 | 0.3640 | 0.3615 | 0.4778 |
| Student | 0.9818 | 0.9814 | 0.8996 | 0.5679 | 0.3372 | 0.3500 | 0.4803 |
| FNO | 0.9370 | 0.9049 | 0.6200 | 0.2673 | 0.2853 | 0.3799 | 0.4794 |
| DFT-SIC | 0.9950 | 0.9950 | 0.9765 | 0.5705 | 0.3395 | 0.3950 | 0.4825 |
| IABR_s0 | 0.9935 | 0.9760 | 0.6145 | 0.2880 | 0.2660 | 0.3335 | 0.4645 |
| E1_capacity_control | 0.9935 | 0.9765 | 0.6110 | 0.2830 | 0.2685 | 0.3430 | 0.4615 |
| Abl1_no_IABC | 0.9930 | 0.9760 | 0.6120 | 0.3000 | 0.2800 | 0.3335 | 0.4610 |

**E8, strict Pd (`tables/E8_strict.md`)**

| model | 30 | 20 | 10 | 5 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| Teacher | 0.9690 | 0.9780 | 0.8875 | 0.4005 | 0.0910 | 0.0535 | 0.0430 |
| Student | 0.9425 | 0.9520 | 0.8330 | 0.2760 | 0.0870 | 0.0700 | 0.0730 |
| FNO | 0.9295 | 0.8995 | 0.5580 | 0.1160 | 0.1050 | 0.1360 | 0.1860 |
| DFT-SIC | 0.9950 | 0.9950 | 0.9765 | 0.5705 | 0.3395 | 0.3950 | 0.4825 |
| IABR_s0 | 0.9935 | 0.9760 | 0.6145 | 0.2880 | 0.2660 | 0.3335 | 0.4645 |
| E1_capacity_control | 0.9935 | 0.9765 | 0.6110 | 0.2830 | 0.2685 | 0.3430 | 0.4615 |
| Abl1_no_IABC | 0.9930 | 0.9760 | 0.6120 | 0.3000 | 0.2800 | 0.3335 | 0.4610 |

**JSON থেকে অতিরিক্ত সংখ্যা**

| separation | IABR pd_source | Teacher pd_source | DFT-SIC pd_source | Teacher n_dropped | IABR p50 | IABR p95 | IABR pairing_err | Teacher pairing_err |
|---|---|---|---|---|---|---|---|---|
| 30° | 0.989 | 0.96 | 0.993 | 10 | 0.1101 | 0.4407 | 0.009 | 0.0184 |
| 20° | 0.97 | 0.968 | 0.993 | 5 | 0.1246 | 0.6318 | 0.012 | 0.0202 |
| 10° | 0.595 | 0.852 | 0.968 | 24 | 0.4515 | 83.3133 | 0.039 | 0.0746 |
| 5° | 0.246 | 0.338 | 0.508 | 206 | 4.8611 | 94.7295 | 0.084 | 0.2126 |
| 3° | 0.227 | 0.059 | 0.305 | 375 | 3.2303 | 97.7169 | 0.078 | 0.256 |
| 2° | 0.289 | 0.038 | 0.365 | 426 | 3.2198 | 95.7908 | 0.089 | 0.2095 |
| 1° | 0.448 | 0.035 | 0.455 | 455 | 2.5669 | 93.6813 | 0.033 | 0.1778 |

অন্যান্য: 20°-এ IABR-এর `p95_endfire = 69.9098`, যদিও broadside-এ p95 মাত্র 0.4733। মানে end-fire-এর কাছে, যেখানে u-domain সংকুচিত, 20°-এও collapse শুরু হয়ে গেছে।

### (6) Comparison
- **30°:** IABR (0.9935) learned model-গুলোর মধ্যে সেরা, DFT-SIC-এর (0.9950) প্রায় সমান, Teacher-এর (0.9888 paper, 0.9690 strict) উপরে। Pairing error মাত্র 0.009, Teacher-এর 0.0184-এর অর্ধেক।
- **20°:** IABR 0.9760, Teacher 0.9879 (Δ −0.0119), DFT-SIC 0.9950। Strict metric-এ IABR (0.9760) প্রায় Teacher-এর (0.9780) সমান।
- **10°, মূল দুর্বলতা:** IABR নেমে যায় 0.6145-এ। Teacher 0.9322 (Δ −0.3177), Student 0.8996, DFT-SIC 0.9765 (Δ −0.3620)। এখানে IABR FNO-এর (0.6200) সমান স্তরে নেমে গেছে। Strict metric-এও Teacher 0.8875 বনাম IABR 0.6145। IABR-এর p50 0.4515° কিন্তু p95 83.3133°। এই bimodal ভাঙন বলছে প্রথম path প্রায় নিখুঁত, আর দ্বিতীয়টা প্রায়ই পুরো ভুল জায়গায় (noise-এর local max)। এটা ঠিক "coarse-bin collapse", যা design report আগেই অনুমান করেছিল। 10°-এ দুটো path-এর দূরত্ব u-domain-এ মাত্র প্রায় 0.9–1.4 cell, যা 3×3 local-max filter-এর চোখে একটাই peak।
- **5°:** IABR 0.2880। Teacher paper-style 0.6811, কিন্তু strict 0.4005, কারণ 206টা drop। DFT-SIC 0.5705। `pd_source`-এ Teacher 0.338 আর IABR 0.246।
- **3° থেকে 1°:** Teacher-এর strict ধসে পড়ে (0.0910, 0.0535, 0.0430), কারণ 375 থেকে 455টা sample drop। কিন্তু IABR-এর strict (0.2660, 0.3335, 0.4645) Teacher-এর চেয়ে অনেক বেশি, আর 1°-এ DFT-SIC-এর (0.4825) কাছাকাছি। **কিন্তু এটা "resolution সাফল্য" নয়।** §2-এর 1° threshold artefact-এর কারণে সব model-এর paper-style Pd 1°-এ প্রায় 0.46–0.48 (Teacher 0.4778, DFT-SIC 0.4825, IABR 0.4645)। মানে সবাই কার্যত একটা path খুঁজে পাচ্ছে। Strict metric-এ IABR-এর সুবিধা আসে কেবল এখান থেকে যে সে drop করে না।
- **Pairing error:** প্রতিটা separation-এ IABR পরিবারের pairing error Teacher, Student আর FNO-এর চেয়ে কম। 5°-এ 0.084 বনাম Teacher 0.2126। DFT-SIC 3° থেকে 1°-এ IABR-এর কাছাকাছি বা ভালো (0.069, 0.06, 0.055)। Joint heatmap আর shared token-এর pairing রক্ষার দাবি এখানে অন্তত আংশিকভাবে সত্য: IABR যখন একটা path খুঁজে পায়, তার AoA আর AoD সাধারণত একসাথেই ঠিক থাকে।
- **IABR পরিবার:** IABR_s0, E1 আর Abl1-এর পার্থক্য ≤ 0.012 (5°-এ Abl1 0.3000 বনাম IABR 0.2880)। Resolution সীমাটা আসে trunk আর head থেকে, IABC-এর এতে কোনো ভূমিকা নেই।

### (7) Verdict: **Mixed, কিন্তু 10°-এ স্পষ্ট failure**
- **Success:**
  - ≥ 20° separation-এ (training distribution-এর কাছাকাছি) IABR প্রায় নিখুঁত (0.9935, 0.9760), আর pairing error সবচেয়ে কম।
  - কখনো drop না করায় strict metric-এ খুব কাছের path-এ Teacher-এর চেয়ে বেশি টিকে থাকে।
- **Failure:**
  - Design report অনুমান করেছিল collapse হবে "< 5–10°"-এ। বাস্তবে IABR **10°-এই ধসে যায়** (0.6145), যেখানে Teacher (0.9322) আর DFT-SIC (0.9765) তখনও ভালো। অর্থাৎ IABR-এর effective resolution মোটামুটি **2 DFT cell**, যা Rayleigh limit (প্রায় 1 cell)-এর দ্বিগুণ খারাপ, আর 0 parameter-এর classical DFT-SIC-এর চেয়েও খারাপ।
  - 10°-এ IABR FNO-এর স্তরে নেমে যায়।
  - Design-এর caveat মানলে 10° এখনো "diagnostic-only" অঞ্চলের নিচে পড়ে না, কারণ DFT-SIC এখানে 97.65% পায়। তাই **এটাকে resolution limit বলে পাশ কাটানো যায় না। এটা architecture-এর একটা দুর্বলতা।**

### Figure: `E8_separation.png`
- **কী দেখায়:** x-axis-এ separation 30 → 1 (বাম থেকে ডানে ক্রমশ কাছে); y-axis-এ **`pd_source`**, table-এর `pd_paper` নয়।
- **লাল DFT-SIC:** 30° আর 20°-এ প্রায় 0.99, 10°-এ 0.968, তারপর 5°-এ 0.508 আর 3°-এ 0.305-এ নামে, তারপর 2° আর 1°-এ আবার 0.365 আর 0.455-এ ওঠে।
- **নীল Teacher:** 0.96, 0.968, 10°-এ 0.852, 5°-এ 0.338, তারপর 3° থেকে 1°-এ প্রায় 0.04–0.06-এ মাটিতে পড়ে থাকে। Drop-এর কারণে strict ধরনের metric-এ এটা হয়।
- **বেগুনি, বাদামি, গোলাপি IABR পরিবার:** তিনটা line প্রায় একটাই। 30°-এ 0.99 আর 20°-এ 0.97, তারপর **10°-এ হঠাৎ প্রায় 0.59-এ ধস** (Teacher আর Student-এর অনেক নিচে)। 5° আর 3°-এ প্রায় 0.23–0.26, তারপর DFT-SIC-এর মতো আবার উঠে 1°-এ প্রায় 0.45।
- **সবুজ FNO:** 10°-এ 0.425, আর 5° থেকে নিচে প্রায় 0.06–0.15।
- **কমলা Student:** Teacher-এর মতো, 10°-এ 0.778।
- **মানে:**
  1. IABR-এর ধস শুরু হয় 20° আর 10°-এর মাঝখানে, অন্য সবার আগে। FNO-ও আগে ধসে, তবে সে বরাবরই দুর্বল।
  2. 3° থেকে 1°-এ ডানদিকে যে "U-shape" উঠে আসা, সেটা 1° threshold-এর artefact (§2), আসল উন্নতি নয়।
  3. Teacher-এর ডানদিকের মেঝে (প্রায় 0.04) আসে তার drop থেকে, কারণ সে একটা blob পেলেই পুরো sample বাদ দেয়।

---

## E10 — Nuisance-path stress test (3 principal + 1 interferer)

### (1) কী test করা হয়েছে
প্রতিটা scene-এ ৩টা "principal" path আর ১টা "nuisance" (interferer) path। Nuisance-এর power principal-দের মোট power-এর তুলনায় **−20, −10 বা 0 dB**। SNR 0 আর 15 dB, 200টা scene। প্রশ্ন দুটো:
- (ক) একটা বাড়তি শক্তিশালী interferer এলে model কি ৩টা আসল path এখনো ঠিকভাবে খুঁজে পায়? এটা `pd_principal`।
- (খ) Model interferer-টাকেও detect করে কি? এটা `pd_nuisance`।

এটা PROJECT_STATUS-এ বর্ণিত Notebook 4 ("supervisor's idea")-এর protocol-এর পুনরাবৃত্তি, এবার IABR-Net আর DFT-SIC যোগ করে। E10-এ E1_capacity_control নেই (`e10()`-এর model list-এ তাকে রাখা হয়নি)।

### (2) Theory
- **Masking:** 0 dB মানে nuisance-এর power principal-দের **মোট** power-এর সমান, অর্থাৎ গড়ে প্রতিটা principal path-এর প্রায় 3 গুণ। তাই এটা scene-এর সবচেয়ে শক্তিশালী path হয়ে যায়। এর sidelobe দুর্বল principal path-গুলোকে ঢেকে দিতে পারে। শক্তিশালী peak-এর কাছে থাকা দুর্বল peak হারিয়ে যায়।
- **−20 dB:** nuisance-এর power 1%, প্রায় noise floor-এর কাছে, তাই তাকে detect করা প্রায় অসম্ভব। কিন্তু L = 4 দেওয়া থাকে, তাই model-কে 4টা peak বের করতেই হয়। চতুর্থ peak-টা হয় একটা noise spike। Blob-detector-নির্ভর model কম blob পেলে পুরো sample drop করে।
- **Robust estimator-এর আদর্শ আচরণ:** nuisance power যাই হোক, `pd_principal` প্রায় স্থির থাকা উচিত। `pd_nuisance` power বাড়ার সাথে বাড়বে, এটা শুধু protocol ঠিকমতো কাজ করছে কিনা তার sanity check।
- **Paired design:** একই seed (1000 + snr) থেকে একই scene, একই principal alpha আর একই noise আসে, শুধু nuisance-এর magnitude বদলায়। তাই power-এর মধ্যে তুলনা করলে scene-এর ভিন্নতার noise বাদ পড়ে।

### (3) Code
- **Bank:** `make_nuisance_bank(name, n_scenes, snr, power_db, seed)` (cell 9)।
  - `DG.generate_points(4, π/6)` থেকে চারটা point আসে (min separation 30°, training-এর মতোই)।
  - Principal-দের জন্য `a_p ~ CN(0, 1/3)`, মানে L = 3-এর gain law।
  - `mag = sqrt(10^(power_db/10) * sum|a_p|^2)` দিয়ে nuisance-এর magnitude, আর phase random।
  - Noise আলাদা `noise_seed` থেকে আসে, তাই power বদলালেও একই থাকে।
  - `L = 4` store করা হয়, feat index 3 হলো nuisance।
- **Test:** `e10()` (cell 23):
  ```python
  principal = metrics(pr, b, sub=slice(0, 3)); nuis = metrics(pr, b, sub=slice(3, 4))
  ```
  Hungarian matching চারটা path-এর উপরেই হয় (`prepare_for_metric`), তারপর `sub` দিয়ে principal বা nuisance column আলাদা করা হয়। Table-এ থাকে `pd_principal` (paper), `pd_principal_strict` আর `pd_nuisance` (paper)।

### (4) কেন করা হয়েছে
- FINAL_RESEARCH_REPORT line 303 বলে: *"E10 — Notebook-4-style nuisance-path stress test ... Reuses the project's one existing beat-the-teacher robustness result"*।
- Notebook 4-এ SNR 0 dB আর 0 dB nuisance-এ Student-এর Pd_p (0.6174) Teacher-এর (0.6065) চেয়ে বেশি ছিল (Δ +0.0109)। পুরো compression project-এ এটাই ছিল একমাত্র "teacher-কে হারানো" data point।
- E10 পরীক্ষা করে: (ক) IABR-Net unexpected interference-এর বিরুদ্ধে কতটা টেকে; (খ) Notebook 4-এর ফলটা নতুন scene-এ reproduce হয় কিনা।

### (5) Result: হুবহু সংখ্যা (`tables/E10.md`)

| model | snr | nuisance_db | pd_principal | pd_principal_strict | pd_nuisance |
|---|---|---|---|---|---|
| Teacher | 0 | -20 | 0.6246 | 0.3717 | 0.0714 |
| Student | 0 | -20 | 0.6381 | 0.3542 | 0.0450 |
| FNO | 0 | -20 | 0.5843 | 0.4792 | 0.0549 |
| DFT-SIC | 0 | -20 | 0.6967 | 0.6967 | 0.0650 |
| IABR_s0 | 0 | -20 | 0.6425 | 0.6425 | 0.0750 |
| Abl1_no_IABC | 0 | -20 | 0.6392 | 0.6392 | 0.0825 |
| Teacher | 0 | -10 | 0.6449 | 0.5675 | 0.5540 |
| Student | 0 | -10 | 0.6440 | 0.5442 | 0.5296 |
| FNO | 0 | -10 | 0.5979 | 0.5800 | 0.4459 |
| DFT-SIC | 0 | -10 | 0.6992 | 0.6992 | 0.5525 |
| IABR_s0 | 0 | -10 | 0.6458 | 0.6458 | 0.4925 |
| Abl1_no_IABC | 0 | -10 | 0.6450 | 0.6450 | 0.4900 |
| Teacher | 0 | 0 | 0.6545 | 0.5825 | 0.7978 |
| Student | 0 | 0 | 0.6229 | 0.5450 | 0.7571 |
| FNO | 0 | 0 | 0.5745 | 0.5400 | 0.7447 |
| DFT-SIC | 0 | 0 | 0.6958 | 0.6958 | 0.7975 |
| IABR_s0 | 0 | 0 | 0.6292 | 0.6292 | 0.7975 |
| Abl1_no_IABC | 0 | 0 | 0.6333 | 0.6333 | 0.7700 |
| Teacher | 15 | -20 | 0.9044 | 0.8275 | 0.6940 |
| Student | 15 | -20 | 0.8883 | 0.7817 | 0.6562 |
| FNO | 15 | -20 | 0.7861 | 0.6800 | 0.5116 |
| DFT-SIC | 15 | -20 | 0.8892 | 0.8892 | 0.7325 |
| IABR_s0 | 15 | -20 | 0.8433 | 0.8433 | 0.6650 |
| Abl1_no_IABC | 15 | -20 | 0.8417 | 0.8417 | 0.6400 |
| Teacher | 15 | -10 | 0.8998 | 0.8683 | 0.8420 |
| Student | 15 | -10 | 0.8839 | 0.8442 | 0.8377 |
| FNO | 15 | -10 | 0.7733 | 0.7192 | 0.7581 |
| DFT-SIC | 15 | -10 | 0.8700 | 0.8700 | 0.8825 |
| IABR_s0 | 15 | -10 | 0.8283 | 0.8283 | 0.8175 |
| Abl1_no_IABC | 15 | -10 | 0.8400 | 0.8400 | 0.7900 |
| Teacher | 15 | 0 | 0.8795 | 0.8575 | 0.9385 |
| Student | 15 | 0 | 0.8674 | 0.8283 | 0.9031 |
| FNO | 15 | 0 | 0.7240 | 0.6842 | 0.8386 |
| DFT-SIC | 15 | 0 | 0.8817 | 0.8817 | 0.8900 |
| IABR_s0 | 15 | 0 | 0.8150 | 0.8150 | 0.8600 |
| Abl1_no_IABC | 15 | 0 | 0.8183 | 0.8183 | 0.8700 |

**JSON থেকে অতিরিক্ত সংখ্যা**
- Teacher-এর `n_dropped` (200 scene-এর মধ্যে): SNR 0-এ 81 (−20 dB), 24 (−10), 22 (0); SNR 15-এ 17, 7, 5। Student SNR 0 −20-এ 89; FNO 36। DFT-SIC, IABR আর Abl1-এর সব condition-এ 0।
- IABR-এর principal p95 (SNR 15): 23.5531° (−20), 24.4925° (−10), 52.6810° (0)। Teacher-এর: 1.9372°, 1.8540°, 2.1564° (drop-এর পরে)। DFT-SIC-এর: 2.5905°, 3.7247°, 3.6300°।
- IABR-এর principal `pd_source` (SNR 15): 0.7617, 0.7367, 0.7283। DFT-SIC-এর: 0.8117, 0.7917, 0.8083। Teacher-এর: 0.76, 0.7933, 0.7783।

### (6) Comparison
- **Principal Pd-এর স্থিরতা (−20 থেকে 0 dB-এর মধ্যে max − min):**
  - IABR: SNR 0-এ 0.0166 (0.6425, 0.6458, 0.6292); SNR 15-এ 0.0283 (0.8433 → 0.8283 → 0.8150, একটানা নামছে)।
  - DFT-SIC: SNR 0-এ 0.0034; SNR 15-এ 0.0192।
  - Teacher paper-style: SNR 0-এ 0.0299; SNR 15-এ 0.0249। Teacher strict SNR 0-এ 0.3717 থেকে 0.5825, মানে 0.2108-এর দোলা।
  - সবাই মোটামুটি স্থির। কেউই "ভেঙে" পড়ে না। IABR-এর স্থিরতা Teacher-এর paper-style-এর মতো, আর Teacher-এর strict-এর চেয়ে অনেক ভালো।
- **Absolute level:**
  - SNR 0-এ IABR paper-style-এ Teacher-এর প্রায় সমান (0.6425 বনাম 0.6246 at −20; 0.6458 বনাম 0.6449 at −10; 0.6292 বনাম 0.6545 at 0)।
  - SNR 0-এ **strict metric-এ IABR অনেক এগিয়ে:** −20 dB-এ 0.6425 বনাম 0.3717 (Δ +0.2708), কারণ Teacher 81/200 scene drop করেছে।
  - SNR 15-এ IABR পিছিয়ে। Paper-style-এ Teacher-এর চেয়ে −0.0611, −0.0715, −0.0645; DFT-SIC-এর চেয়ে −0.0459, −0.0417, −0.0667। Strict-এও Teacher (0.8275, 0.8683, 0.8575) IABR-এর (0.8433, 0.8283, 0.8150) উপরে, শুধু −20 dB-এ IABR এগিয়ে (+0.0158)।
  - **DFT-SIC সব SNR-এ principal Pd-এ সবার উপরে, বা Teacher-এর paper-style-এর সমান।**
- **Nuisance detection:**
  - IABR SNR 0 dB-এ 0 dB nuisance-এ 0.7975, যা DFT-SIC (0.7975) আর Teacher-এর (0.7978) সমান।
  - SNR 15-এ IABR 0.8600, Teacher 0.9385 আর DFT-SIC 0.8900-এর চেয়ে কম।
  - −20 dB-এ সবাই প্রায় 0.05–0.08। এটা physics-এর দিক থেকে প্রত্যাশিত।
- **Gross error:** SNR 15-এ nuisance শক্তিশালী হলে IABR-এর principal p95 23.5531° থেকে 52.6810°-এ লাফায়। DFT-SIC-এর p95 3.7247°-এর নিচেই থাকে। শক্তিশালী interferer-এর পাশে থাকা দুর্বল principal path-কে IABR মাঝে মাঝে পুরোপুরি হারিয়ে ফেলে আর তার বদলে একটা spurious peak দেয়। এটা E7 আর E8-এর মতো একই "coarse grid আর 3×3 NMS masking" সমস্যা।
- **IABR বনাম Abl1_no_IABC:** পার্থক্য ≤ 0.0117 (principal)। SNR 15 −10-এ Abl1 একটু ভালো (0.8400 বনাম 0.8283)। IABC-এর এখানে কোনো ভূমিকা নেই।
- **Notebook 4 reproduce হয়নি:**
  - আগে SNR 0 dB, 0 dB nuisance-এ Student (0.6174) Teacher-এর (0.6065) উপরে ছিল।
  - এই run-এ Student 0.6229, Teacher 0.6545, Δ **−0.0316**। Strict-এ 0.5450 বনাম 0.5825।
  - তাই PROJECT_STATUS-এর "only outperforming data point" দাবিটা এই নতুন 200-scene bank-এ টেকে না। আগের +0.0109 সম্ভবত sampling noise ছিল। 600টা principal path-এ Pd-এর standard error প্রায় 0.02 (আমার হিসাব), আর +0.0109 তার চেয়ে ছোট।
  - Teacher-এর নিজের সংখ্যাও Notebook 4 থেকে আলাদা (0.6065 বনাম 0.6545), যা একই ইঙ্গিত দেয়: scene-set বদলালে এই মাপের পার্থক্য এমনিই আসে।

### (7) Verdict: **Mixed (robustness-এ success, accuracy-তে পিছিয়ে)**
- **Success:**
  - Interferer এলেও IABR ভাঙে না। Principal Pd-এর দোলা ≤ 0.0283, আর কোনো drop নেই।
  - কম SNR-এ strict metric-এ Teacher আর Student-এর চেয়ে অনেক ভালো।
  - SNR 0-এ শক্তিশালী nuisance detect করায় Teacher আর DFT-SIC-এর সমান।
- **Weakness:**
  - SNR 15-এ Teacher আর DFT-SIC-এর চেয়ে 4–7 pp পিছিয়ে।
  - Nuisance শক্তিশালী হলে principal Pd একটানা নামে (0.8433 → 0.8150), আর gross error দ্বিগুণ হয়।
  - Parameter ছাড়া DFT-SIC এখানেও সেরা।

---

## Success (কোথায় ভালো)

1. **কখনো drop করে না, strict metric-এ শক্তিশালী।** E7, E8, E10-এর সব condition-এ IABR-এর `n_dropped = 0`। ফলে কঠিন scene-এ strict Pd-তে Teacher আর Student-কে বড় ব্যবধানে হারায়:
   - E7 SNR 0, L10: 0.3839 বনাম Teacher 0.0676
   - E7 SNR 15, L10: 0.6694 বনাম 0.5426
   - E10 SNR 0, −20 dB: 0.6425 বনাম 0.3717

   Real system-এ একটা "উত্তর নেই" মানে পুরো link-এর জন্য beam নেই। সেই দৃষ্টিতে IABR সবসময় একটা estimate দেয়।
2. **কম SNR আর সহজ scene-এ Teacher-এর সমান বা ভালো।** E7 SNR 0, L1: 0.7940 বনাম 0.7650 (+0.0290)। L2–L4-এ ±0.01-এর মধ্যে। E10 SNR 0-এ paper-style-এ Teacher-এর প্রায় সমান।
3. **ভালোভাবে আলাদা path-এ প্রায় নিখুঁত।** E8-এ 30° separation-এ 0.9935 (learned model-দের মধ্যে সেরা), 20°-এ 0.9760।
4. **Pairing রক্ষা।** E8-এর প্রতিটা separation-এ, আর E7-এ কম SNR অথবা বড় L (SNR 15-এ L ≥ 5)-এ IABR-এর pairing error Teacher-এর চেয়ে কম। E7 SNR 15-এ L8 আর L10-এ DFT-SIC-এর চেয়েও কম। Shared path-token থেকে coarse heatmap আর দুই offset head পড়ার design এখানে কাজ করছে।
5. **OOD L = 10-এ কোনো cliff নেই।** L8 থেকে L10-এর পতন (SNR 15-এ −0.0419) DFT-SIC-এর (−0.0402) মতো। এটা physics-এর কারণে, generalization-এর ব্যর্থতা নয়।
6. **Interference-এর বিরুদ্ধে স্থিতিশীল।** E10-এ principal Pd-এর দোলা ≤ 0.0283।
7. এসব কিছু IABR করছে **195,024 parameter** (Teacher-এর প্রায় 41%) আর **97.17 MFLOPs** (Teacher-এর 15.203 GFLOPs-এর তুলনায় প্রায় 156 গুণ কম) দিয়ে (E9 table)।

## Failure / Weakness (কোথায় দুর্বল)

1. **Angular resolution, সবচেয়ে বড় দুর্বলতা।** E8-এ 10° separation-এ IABR 0.6145, যেখানে Teacher 0.9322 আর DFT-SIC 0.9765। IABR FNO-এর (0.6200) স্তরে নেমে যায়। Design report collapse অনুমান করেছিল "< 5–10°"-এ; বাস্তবে সেটা হয় আরও আগে, 10°-এই। 20°-এও end-fire-এর কাছে p95 69.9098°। মূল কারণ: **16×16 coarse heatmap (প্রতি DFT cell-এ এক pixel) + σ = 0.8 cell target + 3×3 local-max NMS**। এর ফলে effective resolution দাঁড়ায় প্রায় 2 cell, Rayleigh limit-এর দ্বিগুণ। 7×7 crop-offset head একটা anchor থেকে দুটো angle বের করতে পারে না (CANDIDATE_ARCHITECTURES §9 failure mode 3 সত্য প্রমাণিত)।
2. **বড় L-এ, high SNR-এ, পিছিয়ে পড়া।** E7 SNR 15 paper-style-এ Teacher-এর তুলনায় ফাঁক L1-এ −0.0150 থেকে L10-এ −0.0800 পর্যন্ত একটানা বাড়ে। একই কারণ: ঘন path-এর blob merge হয়ে যায়।
3. **Spurious peak বা gross error।** IABR সবসময় L-টা peak দেয়, আর কোনো confidence বা objectness দিয়ে "এটা আসল নয়" বলতে পারে না। ফলে merge হলে দ্বিতীয় peak হয় noise-এর একটা local max। প্রমাণ:
   - E7 SNR 15-এ p95: L4 42.7337°, L5 66.2579° (DFT-SIC 3.0935°, 7.1468°)
   - E8 10°-এ p95 83.3133° কিন্তু p50 0.4515°
   - E10 SNR 15-এ 0 dB nuisance-এ principal p95 52.6810°

   Design critique-এ "false-positive coarse peaks / hallucinated refinement" নামে এই ঝুঁকির কথা বলা ছিল (CANDIDATE_ARCHITECTURES line 615)। সেটা সত্যি হয়েছে।
4. **High SNR-এ sub-cell precision কম।** E7 SNR 15, L1-এ rmse 0.2737, যেখানে Teacher 0.2283 আর DFT-SIC 0.2168। Pairing error-ও L1-এ বেশি (0.1240 বনাম 0.0820)। এটা E3-এ 20–25 dB-এ IABR-এর Pd প্রায় 0.87-এ আটকে থাকার সাথে মেলে।
5. **DFT-SIC সবখানে উপরে।** Strict metric-এ E7-এর প্রতিটা L আর দুই SNR, E8-এর প্রতিটা separation, আর E10-এর প্রতিটা condition-এ 0-parameter-এর classical DFT-SIC IABR-এর সমান বা ভালো। Thesis-এ এটা সৎভাবে লিখতে হবে। একটা learned model-এর উচিত অন্তত classical baseline-কে ছাড়িয়ে যাওয়া।
6. **এই তিন test-এ IABC-v2-এর অবদান শূন্য।** Impairment নেই, তাই IABR_s0, E1 আর Abl1-এর সংখ্যা প্রায় হুবহু এক। এটা নিজে কোনো failure নয়। কিন্তু এর মানে scene difficulty axis-এ IABR-এর সুবিধা বা অসুবিধা পুরোটাই trunk আর head design-এর।
7. **Structural cap।** `k_slots = 10`, তাই L > 10 path সামলানো সম্ভব নয়।
8. **Notebook 4-এর "Student beats Teacher" ফল reproduce হয়নি।** E10-এ SNR 0 dB, 0 dB nuisance-এ Student − Teacher = −0.0316। IABR-এর দুর্বলতা নয়, কিন্তু thesis-এর দাবি সংশোধন করা দরকার।

## Improvement ideas (কোথায় উন্নতি দরকার)

1. **Heatmap resolution বাড়ানো (সবচেয়ে বেশি প্রভাব ফেলবে বলে মনে হয়)।**
   - (ক) **Oversampled beamspace:** stage 0-এর পরে antenna-domain H̃ থেকে 2× বা 4× zero-padded DFT নিয়ে 32×32 বা 64×64 input বানানো। Trunk-এর cost বাড়বে (32×32-এ প্রায় 4 গুণ), তবু Teacher-এর চেয়ে অনেক কম থাকবে।
   - (খ) Trunk 16×16-এ রেখে শেষে pixel-shuffle বা transposed-conv দিয়ে heatmap-কে 32×32 বা 64×64-এ upsample করা।
   - (গ) `heat_sigma_cells` 0.8 থেকে কমানো, আর 3×3-এর বদলে sub-cell peak-finding বা ছোট NMS ব্যবহার করা।
   - লক্ষ্য: E8-এর 10°-এ Pd ≥ 0.93, মানে Teacher-এর স্তর।
2. **প্রতি cell-এ একাধিক path (multi-anchor)।** CenterNet-এর মতো প্রতি cell-এ শুধু একটা offset না নিয়ে 2টা slot বা anchor আর প্রতিটার আলাদা objectness রাখা, যাতে একই coarse bin-এ দুটো path ধরা যায়। এতে design-এর failure mode 3 সরাসরি সমাধান হয়।
3. **Learned SIC বা iterative cancellation।** DFT-SIC-এর দাপট দেখায় যে "শক্তিশালী path খুঁজে তার reconstructed contribution বিয়োগ করে আবার খোঁজা" কৌশলটা resolution আর masking (E10) দুটোর বিরুদ্ধেই কাজ করে। IABR-এ এর তিনটা রূপ হতে পারে:
   - প্রথম pass-এর detected path-গুলোর steering vector `a_r a_tᴴ` (physics জানা) দিয়ে residual `Y − Σ α̂ W^H a_r a_tᴴ F` বানিয়ে দ্বিতীয় pass চালানো। এটা deep unfolding বা model-based DL।
   - অথবা DFT-SIC-এর output-কে prior হিসেবে দিয়ে IABR শুধু correction শিখবে (residual-on-classical)।
4. **Objectness বা confidence head আর path-count estimation।**
   - প্রতিটা peak-এর জন্য একটা "আসল কিনা" score শেখানো, যাতে spurious peak (p95-এর tail) বাদ দেওয়া বা আলাদা করে চিহ্নিত করা যায়।
   - L oracle হিসেবে না দিয়ে model-কে নিজে L estimate করতে শেখানো। বাস্তবে L জানা থাকে না।
   - এটা করলে "never drops" সুবিধা থাকবে, আবার gross error-ও কমবে।
5. **Training distribution বাড়ানো।** Training-এর min separation 30° (`generate_points(L, π/6)`), তাই E8-এর ≤ 20° আসলে OOD। Training-এ কাছাকাছি path-এর scene যোগ করা দরকার: min separation কমিয়ে ধরুন 2–5° করা, বা curriculum, বা `make_separation_bank`-এর মতো explicit close-pair augmentation। সাথে শক্তিশালী-দুর্বল power অনুপাতের scene-ও (E10-এর মতো masking)। Resolution বাড়ানোর (1, 2) সাথে একসাথে করতে হবে। শুধু data দিয়ে 16×16 grid-এর সীমা পার হওয়া যাবে না।
6. **Set-prediction head (Phase 2 plan)।** FINAL_RESEARCH_REPORT line 409-এর Phase 2 প্রস্তাব করে IABR-এর efficient front-end আর trunk-এর উপর Candidate A-এর shared-query set-prediction (DETR-ধরনের) head বসাতে। Query-ভিত্তিক head grid-এ আটকে থাকে না, তাই E7 আর E8-এর resolution সমস্যার একটা সম্ভাব্য সমাধান। তবে সেখানে query capacity (N_q) সীমা আর query collapse ঝুঁকি আছে (CANDIDATE_ARCHITECTURES line 171)।
7. **Subspace (gridless) branch।** Stage 0 already antenna-domain H̃ (16×16) দেয়। এর উপর 2D-ESPRIT বা unitary-ESPRIT ধরনের subspace estimate একটা আলাদা branch বা prior হিসেবে যোগ করা যায়। Subspace পদ্ধতি Rayleigh limit-এর নিচে resolution দিতে পারে, আর IABC-corrected H̃ তার জন্য ভালো input।
8. **High-SNR precision।** Offset head-এর jitter কমানো দরকার। কয়েকটা উপায়: SNR-conditioned offset head (SE-এর SNR estimate দিয়ে), বড় crop, বা high-SNR sample-এ offset loss-এর ওজন বাড়ানো। লক্ষ্য L1 SNR 15-এ rmse ≤ Teacher (0.2283)।
9. **Evaluation-এর উন্নতি (thesis reporting-এর জন্য):**
   - প্রতিটা table-এ paper আর strict দুটোই রাখা, আর Teacher-এর `n_dropped` পাশে দেখানো। না হলে paper-style Pd Teacher-কে ভুলভাবে বড় দেখায়।
   - E8 figure-এর y-axis যে `pd_source`, তা caption-এ লেখা।
   - E8-এ ≤ 3°-এর U-shape-কে threshold artefact হিসেবে চিহ্নিত করা।
   - E7 আর E10-এ bootstrap CI যোগ করা, কারণ L ≤ 3-এ 0.01–0.02-এর পার্থক্য noise-এর মধ্যে।
   - E8 একাধিক SNR-এ চালানো (এখন শুধু 15 dB)। E10-এ E1_capacity_control নেই; দরকার হলে যোগ করা।
10. **Thesis claim সংশোধন।** Notebook 4-এর "compressed student beats teacher under strong interference" ফল E10-এ reproduce হয়নি (Δ −0.0316)। এটা নতুন সংখ্যাসহ সংশোধন করা উচিত।

---

### সংক্ষিপ্ত চূড়ান্ত রায় (এই group)
| Experiment | Verdict | এক লাইনে কারণ |
|---|---|---|
| E7 path count | **Mixed** | Strict metric আর কম SNR-এ Teacher-এর সমান বা ভালো, কোনো drop নেই, OOD cliff নেই। কিন্তু SNR 15 paper-style-এ L-এর সাথে ফাঁক বাড়ে (L10-এ −0.0800), আর DFT-SIC সবখানে উপরে। |
| E8 angular separation | **Mixed → 10°-এ Failure** | ≥ 20°-এ প্রায় নিখুঁত, কিন্তু 10°-এ 0.6145 (Teacher 0.9322, DFT-SIC 0.9765)। কারণ 16×16 grid + 3×3 NMS resolution bottleneck। |
| E10 nuisance path | **Mixed (robust কিন্তু কম accurate)** | Principal Pd স্থির (≤ 0.0283 দোলা), কম SNR-এ strict-এ Teacher-এর চেয়ে অনেক ভালো। কিন্তু SNR 15-এ 4–7 pp পিছিয়ে, আর শক্তিশালী nuisance-এ gross error বাড়ে। |
