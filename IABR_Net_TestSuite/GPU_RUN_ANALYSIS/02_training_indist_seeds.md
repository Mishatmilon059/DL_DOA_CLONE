# Report 02: Training, In-distribution SNR sweep (E3), Multi-seed (E11), SNR tails

**Analyst group:** TRAIN_IABR_s0/s1/s2 + `training_loss.png`, E3_in_distribution_snr_sweep + `E3_snr_sweep.png`, E11_multi_seed, SNR_tails_extrapolation (paper + strict)
**Source files:** `outputs/results/*.json`, `outputs/tables/{E3,E11,SNR_tails,SNR_tails_strict}.md`, `outputs/figures/{training_loss,E3_snr_sweep}.png`, `outputs/logs/run_log.txt`, `outputs/checkpoints/IABR_s*/history.json`, notebook `IABR_Net_Full_Test_Suite.ipynb` (cell 14–16, 17–18, 20, 22, 25), executed `full_run.ipynb`, আর `outputs/cache/pred__*.npz` (cache করা prediction, extra diagnosis-এর জন্য)।

> **নিয়ম:** যেখানে সংখ্যা সরাসরি JSON/table থেকে নেওয়া, সেটা file-এ যেমন আছে তেমনই দেওয়া হয়েছে (table-এ 4 decimal)। যে সংখ্যা আমি নিজে হিসাব করেছি (difference, seed mean, CI, cache থেকে error breakdown), সেগুলোকে **[derived]** লেবেল দেওয়া আছে। Notebook নিজে এই হিসাবগুলো print করেনি।

---

## 0. TL;DR: এক নজরে

| প্রশ্ন | উত্তর |
|---|---|
| Training ঠিকমতো converge করেছে? | **হ্যাঁ।** তিনটা seed-ই ~0.41 total loss-এ এসে থেমেছে (s0 0.41664, s1 0.41244, s2 0.41293)। Divergence হয়নি। s2-তে step 2000-এর পরে resume-জনিত একটা spike আছে, কিন্তু সেটা ঠিক হয়ে গেছে। |
| E3 mean Pd (paper-style) | DFT-SIC **0.7326** > Teacher **0.7103** > Student **0.6956** > IABR (s2 0.6916, s0 0.6909, s1 0.6897) > FNO 0.6235 |
| E3 mean Pd (**strict**, dropped sample-কে miss ধরে) | DFT-SIC **0.7326** > **IABR ~0.691** > Teacher 0.6727 > Student 0.6469 > FNO 0.6018. **Strict metric-এ IABR-Net Teacher-কে হারায়।** |
| কোথায় IABR জেতে | Low SNR (−10, −5 dB)-এ Teacher/Student/FNO-কে হারায়। Strict metric-এ −10 থেকে 10 dB পর্যন্ত Teacher-কে হারায়। Strict-এ Student-কে সব SNR-এ হারায় বা সমান থাকে। FNO-কে সব জায়গায় হারায়। IABR কোনো sample drop করে না (DFT-SIC ছাড়া আর কোনো model এটা পারে না)। |
| কোথায় IABR হারে | **DFT-SIC-এর কাছে প্রতিটা SNR-এ, দুই metric-এই** হারে। ≥5 dB (paper) বা ≥15–20 dB (strict)-এ Teacher-এর কাছে হারে। **High-SNR plateau ~0.87:** 20→25 dB-এ IABR প্রায় বাড়েই না (0.8733→0.8745), যেখানে Teacher 0.9376-এ পৌঁছায়। |
| Seed stability (E11) | খুব stable: E3 std **0.0010**, phase_d5 std 0.0010, gain_g2 std 0.0025। শুধু 25 dB-এ per-SNR spread বেশি (s1 0.8513 বনাম s0/s2 ~0.875)। |
| SNR tails | −25/−20 dB-এ সব model chance floor-এ (~0.02–0.03)। 30/35 dB-এ **IABR-এর Pd কমে যায়** (0.8554 → 0.8411), অথচ Teacher বাড়তে থাকে (0.9390 → 0.9465)। Training range-এর বাইরে IABR ভালোভাবে **extrapolate করে না**। |
| High-SNR ceiling-এর মূল কারণ **[derived, cache থেকে]** | (1) কাছাকাছি থাকা দুটো path coarse 16×16 grid-এ merge হয়ে যায়: SNR ≥15-এ IABR-এর "পুরো path ভুল" ঘটনাগুলোর **82%** এমন path, যা অন্য একটা path-এর 1.5 cell-এর মধ্যে আছে। (2) End-fire sub-cell precision: 25 dB-এ IABR-এর miss component-এর 85% end-fire-এ। End-fire miss rate IABR 0.324, Teacher 0.162, DFT-SIC 0.210। |

---

## 1. পটভূমি: metric আর data আগে বুঝে নাও

### 1.1 Frozen evaluation bank (E3-এর data)
- `data/frozen_banks/eval_bank.npz`: 8000 sample। **সব sample-এ L = 3 path।** SNR ∈ {−10, −5, 0, 5, 10, 15, 20, 25} dB, প্রতিটা SNR-এ 1000 sample। **[derived, `meta` column পড়ে যাচাই করা]**
- এই bank project-এর original `validation_data_generator` দিয়ে তৈরি, আর project-এর আগের সব registry number এই bank থেকেই এসেছে। V0 test দেখিয়েছে Teacher এখানে 0.7102635574 পায়, আর registry-তে আছে 0.7103495885, মানে PASS।
- প্রতি SNR-এ component সংখ্যা 1000 × 3 path × 2 (AoA, AoD) = 6000। তাই per-SNR Pd-এর sampling SE মোটামুটি 0.004–0.006 **[derived]**।
- **খেয়াল রাখো:** training SNR range −15..24 dB, তাই **25 dB নিজেই training range-এর একটু বাইরে** (Teacher আর IABR দুজনের জন্যই, কারণ notebook comment অনুযায়ী দুজনই একই original generator range-এ train হয়েছে)।

### 1.2 Metric (cell 17–18, `metrics()` function)
| Metric | মানে | কেন গুরুত্বপূর্ণ |
|---|---|---|
| `pd_paper` | Component-level (AoA আর AoD আলাদাভাবে)। 1° threshold, Hungarian matching (`permute_pairs` → `linear_sum_assignment`)। যে sample-এ model L-এর কম peak দেয়, সেই sample পুরোটাই **denominator থেকে বাদ** যায়। | Paper Table II-এর সাথে সরাসরি তুলনা করা যায়। কিন্তু যে model বেশি drop করে, তার জন্য এই metric **পক্ষপাতদুষ্ট**। |
| `pd_strict` | একই হিসাব, কিন্তু dropped sample-এর সব component miss ধরা হয়। | **ন্যায্য তুলনা।** IABR আর DFT-SIC সবসময় ঠিক L-টা estimate দেয়, তাই তাদের paper = strict। |
| `pd_source` | একটা path তখনই ঠিক ধরা হয়, যখন **AoA আর AoD দুটোই** 1°-এর মধ্যে থাকে। | Joint পেয়ারিং ঠিক আছে কিনা বোঝায়। |
| `rmse` | শুধু **hit (≤1°) component**-এর RMSE, degree-তে। | এটা bounded (≤1°)। Model-দের hit set আলাদা, তাই তুলনা সাবধানে করতে হবে। |
| `p50` / `p95` | সব matched component-এর absolute error-এর median আর 95th percentile। | Tail বা বড় error ধরে। |
| `p95_endfire` / `p95_broadside` | End-fire মানে true angle 0° বা 180°-এর 30°-এর মধ্যে। | Gap-5 (Jacobian singularity) probe করে। |
| `pairing_error` | Matched path-দের মধ্যে কত অংশে ঠিক একটা component ঠিক। | AoA–AoD pairing সমস্যা বোঝায়। |

---

## 2. TRAIN_IABR_s0 / s1 / s2 (main model training, 3 seed)

### (1) কী test করা হয়েছে
IABR-Net (195,024 params) তিনটা আলাদা random seed (0, 1, 2) দিয়ে পুরো 20,000 step train করা হয়েছে। Hyperparameter সব একই, আলাদা শুধু seed। প্রতিটা run-এর loss history, final loss, train time রাখা হয়েছে।

### (2) Theory
- **Synthetic on-the-fly training:** প্রতিটা batch নতুন করে generate হয়, কোনো fixed dataset নেই। 20,000 × 256 = **5.12M আলাদা sample** দেখা হয় **[derived]**। একই sample দুবার আসে না, তাই "overfitting" ধারণাটা এখানে প্রায় খাটে না। Training loss নিজেই generalization-এর ভালো proxy।
- **Sample distribution:** L ~ U{1..9}, SNR ~ U{−15..24} dB (integer)। Per-sample impairment: phase δmax ~ U(0, 5°), gain γmax ~ U(0, 2 dB)। এরপর প্রতিটা antenna-র error U(−δmax, δmax) এবং U(−γmax, γmax) dB। প্রতিটা sample-এর সাথে একই channel আর একই noise দিয়ে কিন্তু ideal codebook-এ একটা **paired clean observation**-ও তৈরি হয় (`L_pair`-এর target)।
- **Loss** (`train_step`, cell 15):
  - `L_heat` হলো CenterNet-style **focal loss**, 16×16 coarse heatmap-এর উপর। Target Gaussian, σ = 0.8 cell। Positive শুধু exact GT cell, negative-দের (1−target)^4 দিয়ে down-weight করা হয়।
  - `L_off` হলো **smooth-L1**, sub-cell offset (dq, dp)-এর উপর। শুধু GT peak cell-এ হিসাব হয়, cell unit-এ।
  - `L_pair` হলো IABC-corrected Y আর clean Y-এর মধ্যে NMSE।
  - `L_snr` হলো SE gate-এর গড় থেকে SNR/10 regress করার MSE।
  - মোট loss = 1·heat + 1·off + 0.5·pair + 0.1·snr
- **Optimizer:** Adam, global clipnorm 5.0, 500 step linear warm-up থেকে 1e-3, তারপর cosine decay দিয়ে 1e-5।
- **Seed কী নিয়ন্ত্রণ করে:** `tf.keras.utils.set_random_seed(1000+seed)` weight init ঠিক করে, আর `np.random.default_rng([seed, worker])` data stream ঠিক করে। তাই seed variance-এর মধ্যে **init আর data order দুটোই** আছে।

### (3) Code (কোথায় কী হয়)
- Cell 12 `VARIANTS`: `'IABR_s0'/'IABR_s1'/'IABR_s2': ('iabc', True, {}, 'full')`
- Cell 15 `make_batch()` data বানায়, `coarse_targets()` heatmap/offset target বানায়, `focal_loss()`, `lr_at()`, আর `train_variant()` training loop চালায়। Checkpoint প্রতি 2000 step-এ হয়, history log প্রতি 200 step-এ (সেই 200 step window-এর গড়)।
- Cell 16 সব variant train করে, তারপর `training_loss.png` আঁকে।

### (4) কেন করা হয়েছে
- Main model-টা আদৌ শেখে কিনা, stable কিনা, loss কোথায় plateau করে, সেটা দেখতে।
- তিনটা seed রাখা হয়েছে E11-এর জন্য, যাতে research report-এর "single-run number = point estimate only" (Gap 10) caveat বন্ধ করা যায়।
- Step সংখ্যা 20,000 রাখা হয়েছে project-এর আগের 4-way screening-এর সাথে step-count মেলাতে।

### (5) Result (JSON `TRAIN_IABR_s*.json` + `history.json`)

| seed | steps | params | train_min | final loss | heat | off | pair | snr | final lr |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 20000 | 195024 | 21.60 | 0.41663518 | 0.39241009 | 0.02085853 | 0.00615756 | 0.00287786 | 1.0000006e-05 |
| s1 | 20000 | 195024 | 21.45 | 0.41243508 | 0.38858596 | 0.02040715 | 0.00613187 | 0.00376042 | 1.0000006e-05 |
| s2 | 20000 | 195024 | 19.39* | 0.41292508 | 0.38903030 | 0.02052453 | 0.00614646 | 0.00297024 | 1.0000006e-05 |

\* s2-এর `train_min` শুধু resume-এর পরের সময় ধরেছে (নিচে দেখো)।

Loss trajectory (history.json থেকে, প্রতিটা মান 200-step window-এর গড়):

| step | s0 loss | s1 loss | s2 loss | মন্তব্য |
|---|---|---|---|---|
| 200 | 1.6238 | 1.6075 | 1.4012 | warm-up চলছে। s1-এর snr term 4.4661 (বাকি দুটোর ~1.3–1.5) |
| 1000 | 0.4705 | 0.4871 | 0.4743 | দ্রুত নেমে গেছে |
| 2000 | 0.4453 | 0.4460 | 0.4464 | |
| 2200 | 0.4441 | 0.4468 | **0.5841** | **s2 resume spike** (snr 0.4574, heat 0.4997) |
| 3000 | 0.4382 | 0.4370 | 0.4468 | s2 আবার স্বাভাবিক |
| 5000 | 0.4308 | 0.4315 | 0.4281 | |
| 10000 | 0.4175 | 0.4200 | 0.4219 | |
| 15000 | 0.4116 | 0.4130 | 0.4146 | |
| 20000 | 0.4166 | 0.4124 | 0.4129 | |

**[derived]** window-গড়ের গড় (noise বোঝার জন্য):
- steps 10k–15k: s0 0.4172 (std 0.0026), s1 0.4159 (0.0026), s2 0.4174 (0.0023)
- steps 15k–20k: s0 0.4143 (0.0020), s1 0.4128 (0.0018), s2 0.4138 (0.0024)

তাই step 10k থেকে 20k-এ loss মাত্র ~0.003 কমে, যা **window noise (~0.002)-এর সমান মাপের**। দ্বিতীয় অর্ধেকে model প্রায় কিছুই শিখছে না, অর্থাৎ plateau।

**s2 interruption (run_log.txt):** প্রথম session শুরু হয়েছিল 2026-09-23T23:31। s0 আর s1 শেষ হয়, s2 step 2000 পর্যন্ত যায় (log line 83), তারপর process বন্ধ হয়ে যায়। দ্বিতীয় session শুরু হয় 2026-09-24T04:06, আর log বলছে `[IABR_s2] resumed at step 2000`। RESULTS.md-এর "3.07 h" শুধু দ্বিতীয় session-এর সময়। প্রথম session-এর ~45.5 min এর মধ্যে ধরা নেই, তাই মোট compute ≈ 3.8 h **[derived]**।

### (6) Comparison
- **Seed-দের মধ্যে:** final loss-এর পার্থক্য সর্বোচ্চ 0.0042 (s0 বনাম s1), যা window noise-এর মধ্যেই। Offset loss 0.0204–0.0209, pair loss 0.00613–0.00616, দুটোই প্রায় একই।
- **অন্য variant-দের সাথে (figure থেকে):** E1, Abl1, Abl3 একই ~0.41–0.42-এ থামে। শুধু **E2_dense_frontend** ধীরে নামে (step 2500-এ ~0.6) আর উপরে থেমে যায় (~0.44)। Dense front-end যে optimization-এও খারাপ, এটা তার চিহ্ন।
- Teacher/Student/FNO এই suite-এ train হয়নি (pretrained weight load হয়েছে), তাই তাদের সাথে loss তুলনা সম্ভব নয়।

### (7) Verdict: **Success (training pipeline), সাথে দুটো সতর্কতা**
- Training stable, reproducible, আর তিনটা seed একই জায়গায় converge করে।
- **সতর্কতা 1, heat loss plateau:** মোট loss-এর প্রায় 94% হলো `heat` (0.389–0.392)। 10k step-এর পরে এটা প্রায় নড়ে না। Focal loss-এর absolute floor নেই বলে সংখ্যাটা নিজে ব্যাখ্যা করা যায় না, কিন্তু flat curve বলছে **আরও step দিলে বেশি লাভ হবে না। সীমাবদ্ধতাটা representation বা architecture-এ** (নিচে E3 diagnosis দেখো)।
- **সতর্কতা 2, offset precision একেবারে threshold-এর কিনারে [derived, rough]:** `off` ≈ 0.0205 হলো দুই component-এর smooth-L1 যোগফল। Per component ≈ 0.0103। Quadratic regime ধরে নিলে RMS offset error ≈ √(2×0.0103) ≈ 0.14 cell। এক cell = cos-space-এ 1/8 = 0.125, তাই Δu ≈ 0.018, যা broadside-এ ≈ 1.0° angle error। মানে training-SNR range জুড়ে গড় offset error নিজেই প্রায় 1° threshold-এর সমান। End-fire-এ (sin ψ ছোট) এটা আরও অনেক বড় হয়। এটা training SNR range-এর গড় (low SNR-ও ধরা আছে), তাই high SNR-এ আসল error কম। কিন্তু দিকটা স্পষ্ট: **offset head-এর precision-ই মূল bottleneck-গুলোর একটা।**
- **Pair loss থেকে পাওয়া পর্যবেক্ষণ:** `pair` 0.0082 (step 200, IABC-এর শেষ layer zero-init, তাই তখন প্রায় identity) থেকে নেমে 0.0061–0.0062। অর্থাৎ IABC-v2 clean observation-এর দিকে distortion মাত্র **~25% কমাতে পেরেছে [derived]**। (Context: E1-এ λpair = 0 থাকায় সেখানে pair term 0.04–0.05-এ ঘোরে, তবু E3 Pd প্রায় একই। IABC clean-matching করে কিনা, Pd-তে সেটার প্রভাব সামান্য। এটা Controls agent-এর বিষয়।)
- **SNR head:** `snr` 0.0029–0.0038, মানে (SNR/10)² unit-এ, তাই RMS ≈ 0.054 → **~0.5–0.6 dB SNR estimation error [derived]**। SE gate-গুলো SNR খুব ভালোভাবে encode করে। এটা ভালো diagnostic, কিন্তু সম্ভবত SNR tail-এ খারাপ extrapolation-এর একটা কারণ (নিচে §5)।
- **s2 resume artifact:** step 2000→2200-এ loss 0.4464 থেকে 0.5841, snr term 0.0122 থেকে 0.4574। ~step 3000-এ ঠিক হয়ে যায়। সম্ভাব্য কারণ (hypothesis, যাচাই করা হয়নি): Keras-3 optimizer আর `tf.train.Checkpoint`-এর deferred restore-এ Adam-এর moment ঠিকমতো restore হয়নি। তার উপর data iterator একই seed থেকে আবার শুরু হওয়ায় step 1–2000-এর batch-গুলো আবার এসেছে। Final metric-এ কোনো ক্ষতি দেখা যায় না (s2-ই E3-এ সবচেয়ে ভালো, 0.6916)। তবু resume code ঠিক করা উচিত।

### Figure: `training_loss.png`
- **বাম panel ("Main runs + controls"):** x-axis step 0–20000, y-axis total loss। সব curve প্রথম ~1000 step-এ ~1.4–2.45 থেকে খাড়াভাবে ~0.5-এ নামে, তারপর প্রায় flat হয়ে ~0.41–0.42-এ থাকে। **বেগুনি E2_dense_frontend** আলাদা: সবচেয়ে ধীরে নামে (~2500 step-এ ~0.6), আর শেষে উপরে থাকে (~0.44)। **সবুজ IABR_s2**-তে ~2200 step-এ ~0.59-এর একটা স্পষ্ট spike আছে। এটাই resume artifact। বাকি IABR/E1/Abl1/Abl3 curve-গুলো একে অপরের উপর এমনভাবে পড়েছে যে আলাদা করা যায় না।
- **ডান panel ("Abl-6 loss-weight sweep"):** 12টা 5000-step run। শুরু ~1.5–1.9, আর সবগুলো ~0.43-এ মিশে যায়। Loss weight (λpair, λsnr) বদলালে training dynamics-এ কোনো দৃশ্যমান পার্থক্য হয় না।
- **মানে:** training health ভালো। Architecture-এর পার্থক্য (E2 ছাড়া) loss-এ দেখা যায় না। কোনো instability নেই। শুধু s2-এর resume spike একটা engineering bug-এর লক্ষণ।

---

## 3. E3: In-distribution SNR sweep (headline comparison)

### (1) কী test করা হয়েছে
Frozen 8000-sample bank-এ (L = 3, SNR −10..25 dB, impairment ছাড়া) প্রতিটা model-এর per-SNR Pd (paper, strict, source), RMSE, p50/p95, end-fire/broadside p95, pairing error। Model: Teacher, Student, FNO, DFT-SIC, IABR_s0/s1/s2, E1_capacity_control, E2_dense_frontend, Abl1_no_IABC, Abl3_no_SE।

### (2) Theory
- **SNR আর Pd:** SNR বাড়লে noise কমে, তাই peak আরও পরিষ্কার হয় আর Pd বাড়ে। কিন্তু high SNR-এ Pd **ceiling**-এ আটকে যায়, কারণ তখন error noise থেকে আসে না, আসে **model-এর নিজের resolution বা precision limit** থেকে (project-এর Gap 5: 256×256 grid-এ perfect model-এর Pd ceiling ≈ 0.972, আর end-fire-এ Jacobian `dψ = du / sin ψ` quantization error-কে বড় করে দেয়)।
- **IABR-Net-এর বাজি:** 16×16 native beamspace grid (256×256-এর বদলে), sub-cell offset regression, circular padding। এতে compute অনেক কমে (97.17 MFLOPs, Teacher-এর 15.203 GFLOPs-এর তুলনায় ~157× কম)। কিন্তু research report নিজেই সতর্ক করেছিল: "a coarser grid is the same design choice pushing the wrong way on [Gap 5]"।
- **DFT-SIC:** oversampled 2D-DFT (NDFT = 512) peak খোঁজে, তারপর successive interference cancellation। প্রতিবার সবচেয়ে শক্তিশালী path বিয়োগ করে বাকিদের খোঁজে। Learned parameter শূন্য। কাছাকাছি path আলাদা করার ক্ষেত্রে SIC সুবিধা পায়।
- **Hypothesis (FINAL_RESEARCH_REPORT §27):** "IABR-Net's PD/RMSE at matched SNR will fall **between** the classical DFT-SIC baseline … and the full teacher"।

### (3) Code
- Cell 20, `e3()`: `load_bank('E3_frozen')` (cell 9) frozen bank-কে 64×64 থেকে 16×16-এ downsample করে (`downsample16`, bit-exact, E0a-তে যাচাই করা)। প্রতিটা model-এর জন্য `predict()` → `by_snr()` → `metrics()`।
- Heatmap baseline-গুলোর জন্য `predict_heatmap()`: `upsample64` → model → original `get_blob_peaks` → amplitude অনুযায়ী top-L → `peaks_to_angles`।
- DFT-SIC-এর জন্য `predict_dft_sic()` → `classical_dft_sic(Y, L, 512)`।
- IABR-এর জন্য `predict_iabr(decode='learned')`: backbone → sigmoid heatmap → `local_maxima(heat, K=10)` (3×3 maximum filter, wrap mode) → top-L cell → crop-refine head থেকে (dq, dp) → `cells_to_angles`। **Top-L maxima সবসময় ≥ L থাকে, তাই IABR প্রায় কখনো drop করে না।**
- `mean_over_snr` হলো 8টা SNR-এর **unweighted গড়** (`np.nanmean`)।
- Figure: 3টা panel (pd_paper, pd_strict, rmse), সব model।

### (4) কেন করা হয়েছে
Paper-এর 8-point SNR grid-এর সাথে সরাসরি, high-comparability তুলনা। Project-এর registry number (Teacher 0.7103) এই bank থেকেই এসেছে। "IABR-Net baseline-দের তুলনায় কোথায় দাঁড়ায়", এই মূল প্রশ্নের উত্তর এখান থেকেই আসে।

### (5) Result: exact সংখ্যা

**(a) paper-style Pd (tables/E3.md)**

| model | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 25 | mean Pd | mean strict | mean source | RMSE AoA | RMSE AoD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Teacher | 0.2040 | 0.4360 | 0.6393 | 0.7789 | 0.8618 | 0.8994 | 0.9251 | 0.9376 | 0.7103 | 0.6727 | 0.5775 | 0.3751 | 0.3767 |
| Student | 0.1963 | 0.4331 | 0.6308 | 0.7592 | 0.8422 | 0.8831 | 0.9044 | 0.9155 | 0.6956 | 0.6469 | 0.5458 | 0.4041 | 0.3998 |
| FNO | 0.2030 | 0.4133 | 0.6104 | 0.7216 | 0.7965 | 0.7967 | 0.7467 | 0.7001 | 0.6235 | 0.6018 | 0.4575 | 0.4730 | 0.4487 |
| DFT-SIC | 0.2640 | 0.4988 | 0.6857 | 0.7975 | 0.8655 | 0.9042 | 0.9208 | 0.9245 | 0.7326 | 0.7326 | 0.6321 | 0.3564 | 0.3590 |
| IABR_s0 | 0.2288 | 0.4548 | 0.6392 | 0.7597 | 0.8310 | 0.8662 | 0.8733 | 0.8745 | 0.6909 | 0.6909 | 0.5830 | 0.3880 | 0.3889 |
| IABR_s1 | 0.2290 | 0.4602 | 0.6440 | 0.7617 | 0.8288 | 0.8707 | 0.8718 | 0.8513 | 0.6897 | 0.6897 | 0.5802 | 0.3888 | 0.3975 |
| IABR_s2 | 0.2267 | 0.4565 | 0.6453 | 0.7602 | 0.8243 | 0.8680 | 0.8760 | 0.8755 | 0.6916 | 0.6916 | 0.5836 | 0.3902 | 0.3902 |
| E1_capacity_control | 0.2303 | 0.4492 | 0.6343 | 0.7547 | 0.8312 | 0.8670 | 0.8690 | 0.8737 | 0.6887 | 0.6887 | 0.5795 | 0.3899 | 0.3874 |
| E2_dense_frontend | 0.1255 | 0.2807 | 0.4770 | 0.6292 | 0.7218 | 0.7627 | 0.7808 | 0.7857 | 0.5704 | 0.5704 | 0.4257 | 0.4711 | 0.4686 |
| Abl1_no_IABC | 0.2233 | 0.4458 | 0.6375 | 0.7582 | 0.8293 | 0.8672 | 0.8713 | 0.8640 | 0.6871 | 0.6871 | 0.5785 | 0.3917 | 0.3919 |
| Abl3_no_SE | 0.2208 | 0.4447 | 0.6293 | 0.7545 | 0.8230 | 0.8698 | 0.8728 | 0.8678 | 0.6854 | 0.6854 | 0.5748 | 0.3892 | 0.3895 |

**(b) strict Pd per SNR (JSON `pd_strict`, table-এ নেই)**

| model | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| Teacher | 0.1867 | 0.3977 | 0.5830 | 0.7150 | 0.8170 | 0.8688 | 0.9020 | 0.9113 |
| Student | 0.1802 | 0.3885 | 0.5627 | 0.6848 | 0.7883 | 0.8398 | 0.8637 | 0.8670 |
| FNO | 0.1928 | 0.3935 | 0.5982 | 0.7028 | 0.7638 | 0.7688 | 0.7198 | 0.6742 |
| DFT-SIC | 0.2640 | 0.4988 | 0.6857 | 0.7975 | 0.8655 | 0.9042 | 0.9208 | 0.9245 |
| IABR_s0 (= paper) | 0.2288 | 0.4548 | 0.6392 | 0.7597 | 0.8310 | 0.8662 | 0.8733 | 0.8745 |

**(c) Dropped sample (1000-এর মধ্যে, JSON `n_dropped`)**

| model | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| Teacher | 85 | 88 | 88 | 82 | 52 | 34 | 25 | 28 |
| Student | 82 | 103 | 108 | 98 | 64 | 49 | 45 | 53 |
| FNO | 50 | 48 | 20 | 26 | 41 | 35 | 36 | 37 |
| DFT-SIC, সব IABR variant | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**(d) Source Pd, p50, p95, end-fire, pairing (JSON থেকে বেছে নেওয়া)**

| metric | model | −10 | 0 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|
| pd_source | Teacher | 0.0860 | 0.4380 | 0.7333 | 0.7957 | 0.8460 | 0.8687 |
| | DFT-SIC | 0.1447 | 0.5313 | 0.7840 | 0.8380 | 0.8613 | 0.8717 |
| | IABR_s0 | 0.1093 | 0.4873 | 0.7413 | 0.7850 | 0.8007 | 0.7990 |
| p50 (deg) | Teacher | 9.0578 | 0.6013 | 0.2135 | 0.1453 | 0.1118 | 0.0959 |
| | DFT-SIC | 5.8444 | 0.5040 | 0.1878 | 0.1329 | 0.1099 | 0.1016 |
| | IABR_s0 | 6.6573 | 0.5968 | 0.2284 | 0.1788 | 0.1618 | 0.1791 |
| p95 (deg) | Teacher | 120.8724 | 86.6414 | 3.1405 | 1.8904 | 1.4592 | 1.2434 |
| | DFT-SIC | 104.5167 | 75.1242 | 4.1146 | 2.4553 | 1.9199 | 1.7826 |
| | IABR_s0 | 101.3094 | 85.9223 | 44.1100 | 6.0935 | 12.2915 | 5.7229 |
| p95_broadside | Teacher | 101.4751 | 48.8313 | 0.9946 | 0.6187 | 0.4524 | 0.3978 |
| | DFT-SIC | 79.4916 | 27.8615 | 0.7613 | 0.4401 | 0.3064 | 0.2673 |
| | IABR_s0 | 79.7456 | 38.1010 | 1.2043 | 0.6526 | 0.5441 | 0.5610 |
| p95_endfire | Teacher | 160.5547 | 169.5277 | 170.8260 | 56.1623 | 87.8786 | 4.3777 |
| | DFT-SIC | 137.9201 | 169.2602 | 170.0568 | 173.1061 | 173.7434 | 173.4393 |
| | IABR_s0 | 134.2851 | 168.6341 | 171.3084 | 165.5715 | 160.9066 | 161.9926 |
| pairing_error | Teacher | 0.2200 | 0.3180 | 0.1765 | 0.1515 | 0.1149 | 0.0878 |
| | DFT-SIC | 0.2387 | 0.3087 | 0.1630 | 0.1323 | 0.1190 | 0.1057 |
| | IABR_s0 | 0.2390 | 0.3037 | 0.1793 | 0.1623 | 0.1453 | 0.1510 |

Mean over SNR (JSON `mean_over_snr`): p95 হলো Teacher 44.98, DFT-SIC 40.25, IABR_s0 53.21। p95_endfire হলো Teacher 123.46, DFT-SIC 166.55, IABR_s0 161.97।

### (6) Comparison: প্রতিটা SNR-এ জয়/পরাজয়

**[derived]** তিন seed-এর mean থেকে baseline বাদ দিয়ে (Δ > 0 মানে IABR ভালো)। Per-SNR seed std 0.0010–0.0034, তবে 25 dB-এ 0.0137।

**paper-style Pd, Δ = IABR(mean of 3 seeds) − baseline**

| baseline | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| Teacher | **+0.0242** | **+0.0211** | +0.0036 (≈tie) | −0.0184 | −0.0338 | −0.0311 | −0.0514 | −0.0705 |
| Student | **+0.0319** | **+0.0241** | **+0.0120** | +0.0013 (tie) | −0.0142 | −0.0148 | −0.0306 | −0.0484 |
| FNO | **+0.0252** | **+0.0438** | **+0.0325** | **+0.0389** | **+0.0316** | **+0.0716** | **+0.1270** | **+0.1670** |
| DFT-SIC | −0.0358 | −0.0417 | −0.0428 | −0.0370 | −0.0374 | −0.0359 | −0.0471 | −0.0574 |

**strict Pd, Δ = IABR(mean) − baseline**

| baseline | −10 | −5 | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| Teacher | **+0.0415** | **+0.0595** | **+0.0598** | **+0.0455** | **+0.0111** | −0.0006 (tie) | −0.0283 | −0.0442 |
| Student | **+0.0480** | **+0.0687** | **+0.0802** | **+0.0757** | **+0.0397** | **+0.0284** | **+0.0101** | +0.0001 (tie) |
| FNO | **+0.0353** | **+0.0637** | **+0.0447** | **+0.0577** | **+0.0642** | **+0.0994** | **+0.1539** | **+0.1929** |
| DFT-SIC | −0.0358 | −0.0417 | −0.0428 | −0.0370 | −0.0374 | −0.0359 | −0.0471 | −0.0574 |

**Mean-level সারাংশ [derived]:**
- paper: IABR (seed mean 0.6907) বনাম Teacher 0.7103 হলো −0.0196 (Teacher-এর 97.2%)। বনাম Student 0.6956 হলো −0.0049। বনাম FNO +0.0672। বনাম DFT-SIC −0.0419।
- strict: IABR 0.6907 বনাম Teacher 0.6727 হলো **+0.0180**। বনাম Student 0.6469 হলো **+0.0438**। বনাম FNO **+0.0889**। বনাম DFT-SIC −0.0419।
- source: IABR mean (0.5830, 0.5802, 0.5836 থেকে ≈ 0.5823) বনাম Teacher 0.5775 হলো +0.0048। বনাম DFT-SIC 0.6321 হলো −0.0498।
- দুটো gap-ই (Teacher-এর সাথে −0.0196 paper আর +0.0180 strict) seed std 0.0010-এর চেয়ে প্রায় 20 গুণ বড়। তাই **ordering নির্ভরযোগ্য**।

**RMSE (শুধু hit-এর উপর):** IABR_s0 Teacher-এর চেয়ে ভালো −10 (0.5420 বনাম 0.5506), 0 (0.4529 বনাম 0.4576), আর 5 dB-এ (0.3855 বনাম 0.3918)। 10 dB থেকে খারাপ হয়ে যায়: 25 dB-এ 0.2943 বনাম 0.2381। DFT-SIC সব SNR-এ সবচেয়ে কম RMSE পায়। IABR_s0-এর p50 20→25 dB-এ **বেড়ে যায়** (0.1618 থেকে 0.1791), আর s1-এর 0.2280, অথচ Teacher-এর কমে (0.1118 থেকে 0.0959)। এটা precision floor আর training-range edge effect-এর লক্ষণ।

**Mid-SNR gross outlier:** 10 dB-এ IABR_s0-এর p95 = 44.1100°, Teacher-এর 3.1405°। অর্থাৎ 10 dB-এ IABR-এর 5%-এর বেশি component বড় ভুল। একটা অংশ ব্যাখ্যা করা যায় এভাবে: Teacher কঠিন sample drop করে (52টা) আর সেগুলো তার p95-এ ঢোকে না, IABR কিছুই drop করে না। কিন্তু নিচের diagnosis দেখায় IABR-এর নিজস্ব "wrong-peak" failure-ও আছে।

**Controls-এর সাথে (context):** E1 0.6887, Abl1 0.6871, Abl3 0.6854, আর IABR mean 0.6907। পার্থক্য মাত্র 0.0020 / 0.0036 / 0.0053। E1 (capacity control) IABR থেকে আলাদা করা যায় না। মানে E3-এর clean data-তে IABC-v2 বা pair loss কার্যত কোনো লাভ দেয় না। এটা স্বাভাবিকও, কারণ E3-এ কোনো impairment নেই। E2 (dense front-end) 0.5704-এ, স্পষ্টভাবে অনেক খারাপ।

### (6b) Extra diagnosis: high-SNR ceiling কোথা থেকে আসে **[derived, cache থেকে]**
আমি `outputs/cache/pred__{model}__E3_frozen.npz` (cache করা prediction) আর frozen bank-এর ground truth নিয়ে notebook-এর matching logic আবার চালিয়েছি। একই Hungarian matching, wrapped angle diff, 1° threshold। আমার হিসাব official Pd হুবহু মিলিয়ে দেয়: IABR_s0 25 dB = 1 − 753/6000 = 0.8745 ✓, Teacher 25 dB = 1 − 364/5832 = 0.9376 ✓।

**Miss breakdown (component-level):**

| model | SNR | miss rate | end-fire miss rate | broadside miss rate | miss size: 1–2° / 2–5° / 5–20° / 20–90° / 90–150° / >150° |
|---|---|---|---|---|---|
| Teacher | 20 | 0.0749 | 0.190 | 0.016 | 218 / 102 / 6 / 11 / 4 / 97 |
| DFT-SIC | 20 | 0.0792 | 0.216 | 0.008 | 180 / 121 / 27 / 12 / 14 / 121 |
| IABR_s0 | 20 | 0.1267 | **0.317** | **0.028** | 275 / 153 / 42 / **118** / **66** / 106 |
| Teacher | 25 | 0.0624 | 0.162 | 0.014 | 190 / 68 / 13 / 16 / 8 / 69 |
| DFT-SIC | 25 | 0.0755 | 0.210 | 0.009 | 162 / 120 / 32 / 15 / 12 / 112 |
| IABR_s0 | 25 | 0.1255 | **0.324** | **0.028** | 264 / 177 / 51 / **95** / **64** / 102 |

**"পুরো path ভুল" (AoA আর AoD দুটোই >10° off), 1000 sample-এর মধ্যে কতগুলো sample-এ এমন অন্তত একটা path আছে:**

| model | −10 | 0 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| Teacher | 758 | 187 | 19 | 9 | 2 | 10 |
| DFT-SIC | 770 | 185 | 30 | 12 | 11 | 7 |
| IABR_s0 | 783 | 233 | **107** | **69** | **80** | **61** |
| IABR_s1 | 784 | 236 | 98 | 70 | 75 | 79 |
| IABR_s2 | 783 | 233 | 103 | 71 | 80 | 70 |

**এই wrong-peak path-গুলো কোথায় (SNR ≥ 15, সব path একসাথে):**
- IABR_s0: 229টা wrong-peak path। এর মধ্যে **188টা (82%)** অন্য একটা true path-এর 1.5 coarse cell (wrapped Chebyshev distance)-এর মধ্যে আছে। অথচ সব path-এর মধ্যে এমন কাছাকাছি path মাত্র 5.2%।
- Teacher: 25টা wrong-peak path, এর মধ্যে 10টা (40%) কাছাকাছি। DFT-SIC: 33টা, এর মধ্যে 14টা (42%)।

**ব্যাখ্যা:**
1. **Coarse-grid merge (নতুন আর সবচেয়ে বড় finding):** দুটো path 16×16 beamspace-এ ≤1.5 cell দূরে থাকলে IABR-এর 3×3 `local_maxima` NMS এদের একটা peak-এ মিলিয়ে ফেলে। তখন top-L-এর তৃতীয় peak হিসেবে একটা spurious বা ভুল local maximum আসে, আর crop-refine head সেখানে "confident" angle বানিয়ে দেয়। কাছাকাছি path-দের প্রায় 40% এভাবে হারিয়ে যায় (188 / ~468 **[derived, approximate]**)। Teacher-এর 256×256 grid আর DFT-SIC-এর SIC (subtract, তারপর re-detect) এই সমস্যা এড়ায়। এটা E8-এর 10° separation collapse-এর (IABR 0.6145 বনাম Teacher 0.9322) সাথেও মেলে। এই failure 3টা seed-এই একই রকম, তাই এটা **systematic, random নয়**।
2. **End-fire sub-cell precision:** 25 dB-এ IABR-এর 753টা miss-এর মধ্যে 642টা (85%) end-fire-এ। End-fire-এ `dψ = du / sin ψ` হওয়ায় ছোট offset error বড় angle error হয়ে যায়। তার উপর half-wavelength ULA-তে u = ±1 (0° আর 180°) beamspace-এ periodic neighbour, তাই wrap-এর কাছে ভুল হলে প্রায় 180° flip হয় (>150° bin)। এই ">150°" bin সব model-এরই আছে (Teacher 69, DFT-SIC 112, IABR 102 at 25 dB), তাই এটা physics-level ambiguity। কিন্তু **1–5° miss** (IABR 441 বনাম Teacher 258, 25 dB) IABR-এর precision limit।
3. **Broadside-এও IABR দুর্বল:** miss rate 0.028, যা Teacher-এর (0.014) প্রায় দ্বিগুণ আর DFT-SIC-এর (0.009) তিনগুণ। এর মধ্যে merge-জনিত মিস-ও আছে।

### (7) Verdict: **Mixed**
- **Success:**
  - Low SNR-এ (−10, −5 dB) paper metric-এও IABR সব learned baseline-কে হারায়।
  - Strict metric-এ IABR mean 0.6907 > Teacher 0.6727। কোনো sample drop হয় না, যা deployment-এর জন্য বাস্তব সুবিধা।
  - Teacher-এর 2.4× কম param আর ~157× কম FLOPs নিয়ে Teacher-এর paper Pd-এর 97.2% পায়।
  - Student-এর (314K) চেয়ে কম param-এ paper mean-এ মাত্র −0.0049 পিছিয়ে, আর strict-এ +0.0438 এগিয়ে।
  - FNO-কে সব SNR-এ অনেকটা ব্যবধানে হারায়।
- **Failure:**
  - **DFT-SIC (0 param)-এর কাছে সব 8টা SNR-এ, দুই metric-এই হারে** (−0.036 থেকে −0.057)। Research report নিজেই এটাকে "critical baseline" বলেছিল ("IABR-Net's whole motivation rests on beating this 0-param … reference")। এই শর্ত **পূরণ হয়নি**।
  - §27 hypothesis "DFT-SIC আর Teacher-এর মাঝখানে" paper-style-এ **ভুল প্রমাণিত** (IABR দুজনের নিচে)। Strict-এ **সত্য** (DFT-SIC > IABR > Teacher)।
  - High-SNR plateau ~0.87: coarse-grid merge আর end-fire precision এর কারণ।

### Figure: `E3_snr_sweep.png`
- **Panel 1, "Pd (paper-style)":** সব curve −10 dB-এ ~0.2 থেকে উপরে ওঠে। **লাল DFT-SIC** −10 থেকে 15 dB পর্যন্ত সবার উপরে। 20–25 dB-এ **নীল Teacher** তাকে ছাড়িয়ে ~0.94-এ যায়। **কমলা Student** Teacher-এর একটু নিচে। IABR-এর তিনটা seed (বেগুনি s0, বাদামি s1, গোলাপি s2), E1 (ধূসর), Abl1 (cyan), আর Abl3 (নীল) প্রায় একটা "গুচ্ছ" হয়ে ~0.87-এ **flat হয়ে যায়**। বাদামি s1 25 dB-এ একটু নিচে নামে (0.8513)। **সবুজ FNO** 10–15 dB-এ ~0.80-এ চূড়ায় পৌঁছায়, তারপর নামতে থাকে (25 dB-এ 0.70)। **হলদে-সবুজ E2** সবার নিচে (0.13 → 0.79)।
- **Panel 2, "Pd (strict)":** এখানে ছবি বদলে যায়। low থেকে mid SNR-এ IABR গুচ্ছ Teacher আর Student-এর **উপরে** থাকে (Teacher আর Student drop-এর জন্য নিচে নামে)। ~15 dB-এ IABR আর Teacher কাটাকাটি করে, 20–25 dB-এ Teacher উপরে চলে যায়। DFT-SIC সব জায়গায় সবার উপরে।
- **Panel 3, "RMSE of detected (deg)":** সবাই ~0.53–0.56 থেকে নামে। **লাল DFT-SIC** আর **নীল Teacher** 25 dB-এ ~0.23–0.24-এ যায়। IABR গুচ্ছ ~0.285–0.29-এ flat হয়ে থাকে, আর বাদামি s1 25 dB-এ **উপরে ওঠে** (0.3243)। FNO 15 dB-এর পরে উপরে ওঠে। E2 ~0.41-এ থাকে।
- **Figure defect:** matplotlib-এর 10-রঙের cycle ঘুরে আসায় **Teacher আর Abl3_no_SE দুটোই একই নীল রঙে** আঁকা হয়েছে। Legend দেখে আলাদা করা যায় না। JSON অনুযায়ী উপরের নীল line (paper 25 dB-এ 0.9376, RMSE 0.2381) হলো Teacher, আর ~0.868-এর নীল line হলো Abl3। Thesis-এ ব্যবহারের আগে color বা linestyle ঠিক করতে হবে।
- **মানে:** figure পরিষ্কার দেখায় যে IABR-এর সমস্যা **high-SNR saturation**। Low-SNR-এ সে প্রতিযোগিতামূলক বা এগিয়ে। Strict-এ তার অবস্থান অনেক শক্তিশালী।

---

## 4. E11: Multi-seed variance

### (1) কী test করা হয়েছে
IABR_s0/s1/s2-এর paper-style Pd তিনটা condition-এ: E3_frozen (mean over SNR), phase_d5_snr15 (phase ≤5°, 15 dB, 500 sample, L = 3), gain_g2_snr15 (gain ≤2 dB, 15 dB, 500 sample)। প্রতিটার mean আর sample std।

### (2) Theory
Deep network-এর result init আর data order-এর উপর নির্ভর করে। Single-seed number দিয়ে দুটো model-এর পার্থক্য দাবি করা যায় না, যতক্ষণ না পার্থক্যটা seed-to-seed variance-এর চেয়ে বড় হয়। এখানে তিনটা seed একই evaluation bank-এ চালানো হয়েছে (paired), তাই std শুধু **training randomness** মাপে, evaluation sampling noise নয়।

### (3) Code
Cell 25 `e11()`: `available_iabr(['IABR_s0','IABR_s1','IABR_s2'])`। E3-এর জন্য `by_snr(...)['mean_over_snr']['pd_paper']`, বাকিগুলোর জন্য `metrics(...)['pd_paper']`। `np.std(v, ddof=1)` দিয়ে sample std। Table হলো `tables/E11.md`।

### (4) কেন করা হয়েছে
Research report-এর Gap 10 ("No paper … reports cross-seed training variance") আর §27-এর statistical validity threat। Rigor bundle-এর অংশ (strict Pd + multi-seed + real latency), যেটাকে report "cheapest, most defensible novelty" বলেছে।

### (5) Result (tables/E11.md, JSON)

| condition | n_seeds | mean | std | values |
|---|---|---|---|---|
| E3_frozen | 3 | 0.6907 (0.6907291667) | 0.0010 (0.0009547) | [0.6909, 0.6897, 0.6916] |
| phase_d5_snr15 | 3 | 0.8501 (0.8501111111) | 0.0010 (0.0010184) | [0.851, 0.8503, 0.849] |
| gain_g2_snr15 | 3 | 0.8489 (0.8488888889) | 0.0025 (0.0025019) | [0.8463, 0.8513, 0.849] |

**[derived]**
- **95% CI of mean** (t₀.₉₇₅,₂ = 4.303): E3 0.6907 ± 0.0024, phase_d5 0.8501 ± 0.0025, gain_g2 0.8489 ± 0.0062।
- n = 3 দিয়ে std-এর নিজস্ব অনিশ্চয়তা বড়। χ² দিয়ে σ-এর 95% interval ≈ [0.52σ̂, 6.29σ̂], তাই E3-এর আসল seed σ 0.006-এর কাছাকাছি পর্যন্তও হতে পারে।
- **Per-SNR seed spread (E3, paper):** std −10: 0.0013, −5: 0.0027, 0: 0.0032, 5: 0.0010, 10: 0.0034, 15: 0.0023, 20: 0.0021, **25: 0.0137** (range 0.0242, কারণ s1 = 0.8513)।
- **Evaluation noise-এর সাথে তুলনা:** 500-sample robustness bank-এ 3000 component। Pd ≈ 0.85-এ binomial SE ≈ 0.0065, যা seed std (0.0010–0.0025)-এর চেয়ে বেশি। E3-এর mean-এ (48,000 component) SE ≈ 0.0021।

### (6) Comparison
- IABR বনাম Teacher (E3 paper −0.0196, strict +0.0180), দুটোই 95% CI-এর অনেক বাইরে। **Ordering robust।**
- IABR বনাম Student (paper −0.0049): CI ±0.0024-এর বাইরে, তাই "Student সামান্য ভালো (paper)" কথাটা পরিসংখ্যানগতভাবে টেকে। Strict-এ IABR স্পষ্টভাবে ভালো।
- Ablation-এর সাথে: E1 (0.6887), Abl1 (0.6871), Abl3 (0.6854) single-seed। IABR mean থেকে তাদের পার্থক্য 0.0020 (CI-এর ভেতরে, **significant নয়**), 0.0036, আর 0.0053 (CI-এর বাইরে, কিন্তু ablation-গুলো নিজে একটা seed মাত্র, তাই দুর্বল evidence)।
- Robustness bank-এ (phase_d5, gain_g2) model-দের মধ্যে পার্থক্য প্রায় ≤0.013 হলে সেটা evaluation noise-এর মধ্যে পড়ে। Controls table-এর Abl1 phase_d5 0.8607 বনাম IABR 0.8510 পার্থক্য (+0.0097) **significant নয়**।
- Teacher, Student, FNO-র কোনো seed variance নেই (একটাই pretrained checkpoint)। তাই তুলনাটা একপাক্ষিক।

### (7) Verdict: **Success**
IABR-Net খুব reproducible। Training seed থেকে result-এর uncertainty ±0.001–0.0025। এটা thesis-এর জন্য একটা শক্ত rigor point। সতর্কতা: 25 dB-এ (training range-এর edge) per-SNR variance বেশি, আর n = 3 ছোট।

---

## 5. SNR_tails_extrapolation (training range-এর বাইরে SNR)

### (1) কী test করা হয়েছে
L = 3, impairment ছাড়া, প্রতিটা SNR-এ 500 sample (bank seed 6000 + snr), SNR ∈ {−25, −20, 30, 35} dB। Training range −15..24 dB, তাই এগুলো দুই দিকেই OOD। Model: Teacher, Student, FNO, DFT-SIC, **IABR_s0 (শুধু একটা seed)**, E1_capacity_control, Abl1_no_IABC। paper আর strict দুই table।

### (2) Theory
- **Low tail:** noise এত বেশি যে signal প্রায় দেখা যায় না। Back-of-envelope **[derived]**: ‖H‖²_F ≈ NT·NR·Σ|α|² ≈ 256, তাই L = 3-এ প্রতিটা path-এর beamspace peak cell-এ power ≈ 85 (≈ 19.3 dB)। Per-cell noise variance = 10^(−SNR/10), তাই −25 dB-এ peak-cell SNR ≈ −5.7 dB, আর −20 dB-এ ≈ −0.7 dB (leakage ধরলে আরও কম)। এখানে কোনো estimator-ই নির্ভরযোগ্য হবে বলে আশা করা যায় না।
- **High tail:** noise আরও কমে। Ideal estimator-এর Pd বাড়া বা অন্তত ধরে রাখা উচিত। যদি Pd **কমে**, তাহলে model "SNR = x" দেখে শিখেছে, physics শেখেনি। মানে distribution-shift fragility।
- Design doc-এ (CANDIDATE_ARCHITECTURES §9.7) আলাদা একটা risk বলা হয়েছিল: low SNR-এ IABC-এর input statistics অবিশ্বাস্য হয়ে যায়, আর সেটা SNR-extrapolation gap-এর সাথে জড়িয়ে যায়।

### (3) Code
- Cell 9 `BANK_SPECS['tail_snr{s}'] = ('std', dict(n=500, L=3, snr=s, seed=6000+s))` → `make_standard_bank`।
- Cell 22 `tails()` → `sweep('SNR_tails', ...)` → প্রতিটা `ROBUST_MODELS`-এর জন্য `metrics(predict(m, bank), bank)`। তারপর `put_table('SNR_tails')` আর `'SNR_tails_strict'`। কোনো figure নেই।

### (4) কেন করা হয়েছে
RESEARCH_DESIGN-এর Tier-2 "unseen-SNR extrapolation tails … first genuine OOD test in the whole evidence base"। কোনো paper −25 বা 35 dB test করে না। বাস্তবে SNR training range-এর বাইরে গেলে model ভেঙে পড়ে কিনা, সেটা দেখতে।

### (5) Result

**paper-style (tables/SNR_tails.md)**

| model | −25 | −20 | 30 | 35 |
|---|---|---|---|---|
| Teacher | 0.0184 | 0.0257 | 0.9390 | 0.9465 |
| Student | 0.0204 | 0.0257 | 0.9191 | 0.9148 |
| FNO | 0.0207 | 0.0190 | 0.6622 | 0.6387 |
| DFT-SIC | 0.0213 | 0.0303 | 0.9240 | 0.9370 |
| IABR_s0 | 0.0190 | 0.0273 | 0.8554 | 0.8411 |
| E1_capacity_control | 0.0177 | 0.0233 | 0.8713 | 0.8631 |
| Abl1_no_IABC | 0.0227 | 0.0280 | 0.8677 | 0.8563 |

**strict (tables/SNR_tails_strict.md)**

| model | −25 | −20 | 30 | 35 |
|---|---|---|---|---|
| Teacher | 0.0183 | 0.0257 | 0.9127 | 0.9257 |
| Student | 0.0203 | 0.0250 | 0.8787 | 0.8800 |
| FNO | 0.0207 | 0.0190 | 0.6477 | 0.6157 |
| DFT-SIC | 0.0213 | 0.0303 | 0.9240 | 0.9370 |
| IABR_s0 | 0.0190 | 0.0273 | 0.8537 | 0.8343 |
| E1_capacity_control | 0.0177 | 0.0233 | 0.8713 | 0.8613 |
| Abl1_no_IABC | 0.0227 | 0.0280 | 0.8677 | 0.8563 |

**আরও কিছু JSON field (500 sample-এর মধ্যে):**

| model | SNR | pd_source | p50 | p95 | pairing_error | n_dropped |
|---|---|---|---|---|---|---|
| Teacher | 30 / 35 | 0.868 / 0.8827 | 0.0903 / 0.089 | 1.1971 / 1.1026 | 0.0919 / 0.0879 | 14 / 11 |
| DFT-SIC | 30 / 35 | 0.8753 / 0.8947 | 0.0964 / 0.0982 | 2.064 / 1.4921 | 0.0973 / 0.0847 | 0 / 0 |
| Student | 30 / 35 | 0.8193 / 0.8213 | 0.1586 / 0.1564 | 1.625 / 1.5815 | 0.1241 / 0.122 | 22 / 19 |
| IABR_s0 | 30 / 35 | 0.7607 / 0.736 | 0.2436 / 0.2539 | 7.4998 / 5.8769 | 0.1864 / 0.1983 | **1 / 4** |
| E1 | 30 / 35 | 0.7887 / 0.7827 | 0.1983 / 0.2075 | 4.0447 / 12.1858 | 0.1653 / 0.1576 | 0 / 1 |
| Abl1 | 30 / 35 | 0.7787 / 0.7733 | 0.2077 / 0.2104 | 4.7549 / 6.3085 | 0.178 / 0.166 | 0 / 0 |
| (low tail) IABR_s0 | −25 / −20 | 0.0 / 0.0053 | 33.0435 / 31.2208 | 103.5705 / 100.8348 | 0.038 / 0.044 | 0 / 0 |

**Extra diagnosis [derived, cache `pred__*__tail_snr30/35.npz` থেকে]:**

| model | SNR | miss / comps | end-fire miss rate | 1–2° miss | wrong-peak paths (তার মধ্যে কাছাকাছি path) |
|---|---|---|---|---|---|
| Teacher | 30 | 178/2916 | 0.160 | 89 | 6 (0) |
| DFT-SIC | 30 | 228/3000 | 0.197 | 74 | 10 (1) |
| IABR_s0 | 30 | 433/2994 | **0.380** | 163 | 36 (26) |
| IABR_s0 | 35 | 473/2976 | **0.404** | 197 | 31 (16) |
| E1 | 30 / 35 | 386/3000, 410/2994 | 0.340 / 0.330 | 146 / 153 | 27 (25) / 40 (22) |
| Abl1 | 30 / 35 | 397/3000, 431/3000 | 0.348 / 0.360 | 159 / 160 | 31 (26) / 36 (24) |

IABR-এর end-fire miss rate E3-তে 25 dB-এ 0.324 ছিল, 30 dB-এ 0.380, 35 dB-এ 0.404, অর্থাৎ SNR বাড়ার সাথে **বাড়ছে**। 1–2° near-miss-ও বাড়ছে। Wrong-peak path-এর ভাগ প্রায় একই থাকে (~2.1–2.4% path)।

### (6) Comparison
- **Low tail (−25, −20):** সব model 0.0177–0.0303, মানে chance floor। SE ≈ 0.0029 **[derived]**, তাই এখানে model-দের মধ্যে পার্থক্য (যেমন DFT-SIC −20 dB-এ 0.0303 বনাম IABR 0.0273) **significant নয়**। pd_source প্রায় 0। p50 ~31–39°, মানে প্রায় random। IABC-এর low-SNR misbehavior-এর কোনো আলাদা প্রমাণ নেই (Abl1 বনাম IABR পার্থক্য noise-এর মধ্যে)।
- **High tail (30, 35), আসল finding:**
  - **Teacher বাড়ে:** 0.9376 (25, E3) → 0.9390 → 0.9465। DFT-SIC 0.9245 → 0.9240 → 0.9370। Student মোটামুটি flat (0.9155 → 0.9191 → 0.9148)।
  - **IABR_s0 নামে:** 0.8745 (25, E3) → 0.8554 → 0.8411। p50 বাড়ে: 0.1791 → 0.2436 → 0.2539। FNO-ও নামে (0.7001 → 0.6622 → 0.6387), কিন্তু FNO E3-তেই আগে থেকে নামছিল।
  - 30 dB-এ IABR − Teacher = −0.0836 (paper), −0.0590 (strict) **[derived]**। 35 dB-এ −0.1054 (paper), −0.0914 (strict)। Strict-এ IABR শুধু FNO-কে হারায়। Student-এর কাছেও হারে (0.8537 বনাম 0.8787 at 30)।
  - **E1 আর Abl1 IABR_s0-এর চেয়ে একটু ভালো** (E1 +0.0159 / +0.0220, Abl1 +0.0123 / +0.0152, paper)। দুটোরই λpair = 0। পার্থক্য ≈ 2–3.4 SE (unpaired), আর single seed। তাই এটা শুধু **ইঙ্গিত**: pair loss বা clean-matching IABC হয়তো high-SNR extrapolation সামান্য ক্ষতিগ্রস্ত করে। Abl1-এ IABC নেই, তবু decline আছে, তাই **decline-এর মূল কারণ IABC নয়, trunk বা head**।
  - IABR এখানে প্রথমবার sample drop করে (30 dB-এ 1টা, 35 dB-এ 4টা)। মানে heatmap-এ 3টার কম local maximum ছিল (hypothesis: খুব "পরিষ্কার" heatmap যেখানে NMS-এর পরে শুধু 2টা peak টিকে থাকে)।

### (7) Verdict: **Failure (high-SNR extrapolation), আর low tail inconclusive**
- Low tail-এ কেউই কাজ করে না। এটা model-এর ব্যর্থতা নয়, physics-এর সীমা।
- **High tail-এ IABR-Net-এর একটা স্পষ্ট দুর্বলতা:** SNR বাড়লে performance কমে। Teacher একই training range পেয়েও বাড়তে থাকে। তাই দুর্বলতাটা IABR design-এর নিজস্ব।
- **সম্ভাব্য কারণ (hypothesis, যাচাই করা হয়নি):**
  - (a) SE gate-গুলো SNR-কে খুব শক্তভাবে encode করে (SNR head RMS ~0.5 dB)। Training range-এর বাইরে gate অদেখা মানে চলে যায়, আর trunk feature shift করে।
  - (b) Offset head শুধু training SNR-এর noise-level-এ calibrated। Noise কমলে heatmap বা token-এর statistics বদলায়, আর end-fire offset precision খারাপ হয় (diagnosis: end-fire miss rate 0.324 → 0.404)।
  - (c) Input normalization নেই।
  - **Abl3_no_SE-কে tail-এ চালানো হয়নি।** তাই hypothesis (a) সরাসরি test করা যায়নি। এটা একটা **missing experiment**।

---

## 6. সব মিলিয়ে: cross-experiment ছবি
1. **Training ঠিক আছে।** সমস্যা optimization-এ নয়, representation-এ: 16×16 coarse grid, 3×3 NMS, আর offset precision।
2. **"Low SNR-এ ভালো, high SNR-এ plateau"** প্যাটার্নটা E3, SNR tails, E11-এর 25 dB variance, আর training-এর offset loss floor, সবকিছুতে একই। কারণ হিসেবে মনে হয় দুটো জিনিস কাজ করে: (i) কাছাকাছি path merge হয়ে যাওয়া (E8-এর 10° collapse-এর সাথে মেলে), (ii) end-fire sub-cell precision।
3. **Strict metric-এ অবস্থান অনেক ভালো:** IABR কোনো sample drop করে না। Teacher 0 dB-এ 8.8% sample drop করে (88/1000)। Thesis-এ strict Pd-কে headline হিসেবে রাখা **ন্যায্য আর বাড়তি সুবিধাজনক**। তবে paper-style-ও পাশে দিতে হবে।
4. **DFT-SIC হলো আসল প্রতিদ্বন্দ্বী।** 0 param, সব SNR-এ সবার উপরে বা সমান। Research report-এর মূল motivation ("beat DFT-SIC by more than its params are worth") এই run-এ **পূরণ হয়নি**। (Note: আগে sanity notebook-এ DFT-SIC 20 dB-এ "0.884" এসেছিল per-source metric দিয়ে। এখানে একই evaluator দিয়ে paper-style 0.9208, আর pd_source 0.8613। PROJECT_STATUS আর notebook দুটোতেই এই correction লেখা আছে।)
5. **Seed variance খুব কম,** তাই উপরের সব বড় পার্থক্য (≥0.01, E3-এ) নির্ভরযোগ্য। 500-sample bank-এ ছোট পার্থক্য (<0.013) নয়।

---

## Success (কোথায় ভালো)
- **Stable, reproducible training:** 3 seed-এর final loss 0.4124–0.4166, E3 Pd std 0.0010 (95% CI ±0.0024)। Divergence নেই, NaN নেই।
- **Low-SNR শ্রেষ্ঠত্ব (learned model-দের মধ্যে):**
  - −10 dB-এ IABR 0.2288 / 0.2290 / 0.2267, যেখানে Teacher 0.2040, Student 0.1963, FNO 0.2030।
  - −5 dB-এ 0.4548 / 0.4602 / 0.4565, যেখানে Teacher 0.4360।
  - Low SNR-এ p50 আর p95-ও Teacher-এর চেয়ে ভালো (−10 dB p50 6.6573 বনাম 9.0578)।
- **Strict Pd-এ Teacher-কে হারায়:** mean 0.6907 বনাম 0.6727। −10 থেকে 10 dB পর্যন্ত প্রতিটা SNR-এ এগিয়ে (+0.0111 থেকে +0.0598)। Student-কে strict-এ 7টা SNR-এ হারায় আর 25 dB-এ সমান থাকে।
- **শূন্য dropped sample:** DFT-SIC-এর মতো সবসময় ঠিক L-টা estimate দেয়। Teacher, Student, FNO 20–108 sample drop করে।
- **Efficiency-accuracy trade-off:** 195K param আর 0.097 GFLOPs নিয়ে Teacher-এর (469K, 15.203 GFLOPs) paper Pd-এর 97.2% পায়। Student-এর (314K) প্রায় সমান (−0.0049 paper, +0.0438 strict)। FNO-কে সব জায়গায় হারায়।
- **Source Pd (mean) Teacher-এর চেয়ে সামান্য বেশি:** 0.5830 বনাম 0.5775 (s0)।

## Failure / Weakness (কোথায় দুর্বল)
- **DFT-SIC-এর কাছে সব SNR-এ হার** (−0.0358 থেকে −0.0574, দুই metric-এই)। 0-param classical method-কে না হারাতে পারা thesis-এর claim-এর জন্য সবচেয়ে বড় সমস্যা।
- **High-SNR plateau ~0.87:** 15→25 dB-এ IABR_s0 মাত্র 0.8662 → 0.8745, যেখানে Teacher 0.8994 → 0.9376। 25 dB-এ Teacher থেকে −0.0705 (paper, seed mean)।
- **Coarse-grid merge failure [derived]:** SNR ≥15-এ 6–8% sample-এ একটা পুরো path ভুল জায়গায় (Teacher-এ 0.2–1%)। এর 82% কাছাকাছি path-দের merge থেকে আসে।
- **End-fire precision:** IABR end-fire miss rate 0.317–0.324 (20–25 dB), যেখানে Teacher 0.162–0.190। p95_endfire সব SNR-এ >127° (Teacher 25 dB-এ 4.3777)।
- **Broadside precision-ও দুর্বল:** p50 25 dB-এ 0.1791° বনাম Teacher 0.0959°। p95_broadside 0.5610 বনাম 0.3978।
- **Mid-SNR gross outlier:** 10 dB-এ p95 44.1100° বনাম Teacher 3.1405°।
- **SNR extrapolation failure:** 30/35 dB-এ Pd কমে (0.8554 / 0.8411), যখন Teacher বাড়ে (0.9390 / 0.9465)। Strict-এ IABR Student-এর কাছেও হারে। 25 dB-এই এর শুরু দেখা যায় (s1 0.8513, p50 বেড়ে যাওয়া)।
- **Pairing error high SNR-এ বেশি:** 25 dB-এ 0.1510 বনাম Teacher 0.0878। 30 dB-এ 0.1864 বনাম 0.0919।
- **Training loss 10k step-এর পরে flat।** আরও step দিলে এই সমস্যা মিটবে না।
- **Engineering issue:**
  - s2-তে resume spike (optimizer বা data-iterator state restore হয়নি বলে মনে হয়)।
  - E3 figure-এ Teacher আর Abl3 একই রঙে।
  - Tail test-এ শুধু s0 আছে, Abl3 নেই।
  - RESULTS.md-এর "3.07 h" প্রথম session বাদ দিয়ে হিসাব করা।

## Improvement ideas (কোথায় উন্নতি দরকার)
Research agent-এর জন্য অগ্রাধিকার অনুযায়ী সাজানো। প্রতিটা উপরের কোনো নির্দিষ্ট দুর্বলতার সাথে যুক্ত।

1. **Coarse grid-এর resolution বাড়াও (merge failure-এর জন্য, সবচেয়ে বেশি লাভের সম্ভাবনা):**
   - Antenna-domain H̃-কে zero-pad করে 2D-FFT নাও, 2× বা 4× oversampled beamspace (32×32 বা 64×64) পেতে। DFT-SIC যেভাবে NDFT = 512 ব্যবহার করে, সেভাবে। এটা physics-consistent, আর FLOPs এখনও Teacher-এর তুলনায় অনেক কম থাকবে।
   - অথবা শেষে sub-pixel conv বা PixelShuffle দিয়ে heatmap upsample করো।
   - NMS kernel ছোট করো, অথবা per-cell একাধিক slot বা objectness রাখো।
2. **SIC-style iterative detection বা hybrid:** DFT-SIC 0 param নিয়েও জেতে, কারণ সে strongest path বিয়োগ করে বাকিদের খোঁজে।
   - (a) Network দিয়ে প্রথম peak ধরো, reconstructed path subtract করো, তারপর আবার চালাও (deep-unfolded SIC)।
   - (b) DFT-SIC estimate-কে initialization ধরে network-কে শুধু residual refiner বানাও।
   - (c) Report §31 Phase-2-এর DETR-style set-prediction head (Candidate A fusion)।
3. **End-fire-aware offset learning:**
   - Offset loss angle domain-এ হিসাব করো, অথবা u-space loss-কে 1/sin ψ (Jacobian) দিয়ে weight করো।
   - Wrap-aware (sin/cos) parameterization ব্যবহার করো।
   - Crop head-কে predicted peak-এর উপরেও train করো (GT index ±1 cell jitter আর hard negative দিয়ে), যাতে train আর test-এর mismatch কমে। এখন head শুধু GT cell-এ train হয়, কিন্তু inference-এ predicted peak পায়।
4. **SNR extrapolation ঠিক করো:**
   - Training SNR range বাড়াও (যেমন −20..40 dB), অথবা high-SNR oversampling করো।
   - Input-কে per-sample power দিয়ে normalize করো।
   - SE-SNR coupling কমাও (λsnr ছোট করো, বা gate-এ SNR-invariance regularizer দাও)।
   - **আগে Abl3_no_SE আর s1/s2 দিয়ে tail test চালাও,** যাতে কারণ আলাদা করা যায়। Tail bank n = 500 থেকে ≥2000 করো।
5. **Capacity বা receptive-field test (সস্তা, 5000-step sweep):** C = 48/64 বা 14–16 block দিয়ে দেখো plateau সরে কিনা। E1 দেখিয়েছে একই capacity-তে front-end বদলালে কিছু হয় না। তাই trunk বা head-এর capacity বা resolution-ই পরের প্রশ্ন।
6. **Statistics আর reporting:**
   - IABR আর প্রতিটা ablation-এর জন্য ≥5 seed।
   - Per-sample bootstrap CI আর paired test (McNemar) বনাম Teacher বা DFT-SIC।
   - Strict Pd-কে headline হিসেবে রাখো (paper-style পাশে)।
   - Per-SNR ranking table thesis-এ দাও।
7. **Engineering fix:**
   - Resume-এর সময় Adam state যাচাই করো (restore-এর পরে `opt.iterations` আর moments চেক করো)। Data iterator-এর position বা seed step অনুযায়ী offset করো।
   - Figure-এ ≥11টা আলাদা color বা linestyle ব্যবহার করো।
   - RESULTS.md-তে সব session-এর মোট wall-clock লেখো।

---

### Missing বা ambiguous item (স্পষ্টভাবে)
- E3 bank-এ −15 dB নেই (RESEARCH_DESIGN-এর 13-point grid পুরো cover হয়নি)। Frozen bank-এর 8 point-ই ব্যবহার করা হয়েছে।
- SNR tail-এর জন্য কোনো figure তৈরি হয়নি (notebook design অনুযায়ী, file missing নয়)।
- SNR tail-এ শুধু IABR_s0। E11-এ tail condition নেই।
- Frozen bank-এর impairment setting file-এ স্পষ্ট লেখা নেই। Original `validation_data_generator` থেকে এসেছে। E5-এর gain = 0 row-এর সাথে সামঞ্জস্য দেখে clean বলেই মনে হয়, কিন্তু আমি সরাসরি যাচাই করিনি।
- s2-এর resume spike-এর কারণ অনুমান মাত্র। Checkpoint-এর ভেতরে optimizer state আমি inspect করিনি।
- "Cache থেকে diagnosis" অংশের সব সংখ্যা আমার নিজের script থেকে (scratchpad `diag.py`, `diag2.py`)। Official Pd-এর সাথে মিলিয়ে যাচাই করা, কিন্তু notebook-এর output নয়।
