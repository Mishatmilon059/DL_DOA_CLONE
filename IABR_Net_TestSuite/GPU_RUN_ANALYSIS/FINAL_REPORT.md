# IABR-Net GPU Run — চূড়ান্ত বিশ্লেষণ রিপোর্ট

> **এই ডকুমেন্ট কী:** IABR-Net-এর সম্পূর্ণ GPU test-suite রান (৩৭টা experiment, `_smoke_test: false`, মোট wall-clock ≈ ৩.৮ ঘণ্টা দুই session মিলিয়ে) থেকে পাওয়া প্রতিটা সংখ্যার একটা একক, verified, সংশোধিত সমন্বয়। এটা একটা multi-agent pipeline-এর চূড়ান্ত পণ্য: ছয়জন analyst (group report 01–06) → একজন MOTHER agent (07, সমন্বয়) → একজন adversarial VERIFIER (08, প্রতিটা সংখ্যা raw source-এর বিপরীতে হাতে-চেক) → একজন RESEARCH agent (09, literature-backed next-step)। **এই ফাইনাল রিপোর্টে verifier-এর ধরা তিনটা ভুল আর একটা unverifiable claim সংশোধন/চিহ্নিত করে দেওয়া হয়েছে** (বিস্তারিত §৬-এ)।

---

## ০. কীভাবে পড়বে

এই রিপোর্ট একটা সমন্বিত সারসংক্ষেপ — পুরো detail (প্রতিটা JSON field, প্রতিটা code snippet, প্রতিটা derived হিসাবের method) ধরে না। নিচের group report-গুলোতে সেই আসল detail আছে, একই ফোল্ডারে (`GPU_RUN_ANALYSIS/`):

| ফাইল | কী আছে |
|---|---|
| `01_sanity_verification.md` | E0a, BANKS_build, E0b, V0, V1 — পুরো pipeline বিশ্বাসযোগ্য কিনা তার প্রমাণ |
| `02_training_indist_seeds.md` | Training (3 seed), E3 headline SNR sweep, E11 multi-seed, SNR_tails extrapolation |
| `03_impairment_robustness.md` | E4 (phase), E5 (gain), E6 (OOD phase) — IABC-v2-এর মূল দাবির পরীক্ষা |
| `04_scene_difficulty.md` | E7 (path count), E8 (angular separation), E10 (nuisance path) |
| `05_ablations_controls.md` | Controls (E1/E2/Abl1/Abl3), Abl-2 (decode ablation), Abl-6 (loss-weight sweep) |
| `06_efficiency.md` | E9 — params/FLOPs/latency/memory |
| `07_mother_synthesis.md` | ছয়টা group report-এর সমন্বয় (cross-experiment pattern সহ) |
| `08_verification.md` | ৩৩টা claim-এর numeric audit — ২৯ CORRECT, ৩ WRONG, ১ UNVERIFIABLE |
| `09_next_steps_research.md` | Literature-backed fix প্রস্তাব, real citation সহ |

এই ফাইনাল রিপোর্ট `07` থেকে headline নিয়ে, `08`-এর সংশোধন প্রয়োগ করে, আর `01`–`06`-এর detail দিয়ে প্রতিটা experiment-কে সম্পূর্ণ করে বানানো। কোনো নতুন সংখ্যা এখানে আবিষ্কার করা হয়নি — সব কিছু উপরের ৯টা ফাইল থেকে হুবহু নেওয়া বা তাদের রিপোর্ট করা derived হিসাব।

---

## ১. সংক্ষেপে (Executive Summary)

**IABR-Net কি সফল?** সংক্ষেপে: **আংশিকভাবে সফল — parameter/FLOP efficiency-তে সিদ্ধান্তমূলক জয়, কিন্তু thesis-এর মূল novelty claim (impairment-aware IABC-v2 correction একটা 0-parameter classical baseline, DFT-SIC-কে হারাবে) data দিয়ে সমর্থিত নয়।**

- **Efficiency-তে নিঃসন্দেহে সফল।** IABR-Net মাত্র **195,024 params** (Teacher 469,393-এর **41.5%**, Student 314,513-এর 62.0%, FNO 334,321-এর 58.3%) আর **97.17 MFLOPs** (Teacher 15.203 GFLOPs-এর **~156× কম**, Student 10.157 GFLOPs-এর ~104.5× কম, FNO 0.172 GFLOPs-এর ~1.77× কম) দিয়ে বানানো (E0b, E9)। GPU memory মাত্র 66 MB (বাকি সব ~2.6–2.9× বেশি), weights ফাইল মাত্র 1.01 MB। **batch=128-এ** throughput Teacher-এর 36.6× — তবে এই বিশাল অনুপাত শুধু বড় batch-এ সত্যি: batch=32-এ অনুপাত মাত্র ~7.6×, আর batch=8-এ ~3.0× (IABR-Net-এর latency batch-নির্বিশেষে প্রায় flat থাকায়, batch যত বাড়ে অনুপাতও তত বাড়ে)।
- **In-distribution accuracy-তে "মাঝারি" — Teacher আর Student-এর কাছাকাছি, কিন্তু DFT-SIC-এর নিচে সবসময়।** E3 frozen bank (L=3, SNR −10..25 dB)-এ mean paper-style Pd: **DFT-SIC 0.7326 > Teacher 0.7103 > Student 0.6956 > IABR (3-seed mean ≈ 0.6907) > FNO 0.6235**। (তিন seed-এর paper Pd হলো s0=0.6909, s1=0.6897, s2=0.6916 — তাদের সঠিক গড় **0.690729 ≈ 0.6907**, headline-এ শুধু s0-এর 0.6909 ব্যবহার করলে সেটা "3-seed mean" নয়, একটা seed-এর একার সংখ্যা।) **Strict Pd-তে** (dropped sample-কে miss ধরলে, যেটা IABR-এর কখনো sample drop না-করার সুবিধা তুলে ধরে) IABR (0.6907) আসলে **Teacher-কে হারায়** (0.6727) এবং Student-কেও (0.6469), তবে DFT-SIC-এর (0.7326) নিচেই থাকে। 15 dB-এ IABR_s0 0.8662 (paper), Teacher 0.8994, DFT-SIC 0.9042 — clean condition-এই IABR প্রায় 3.3–3.8 pp পিছিয়ে।
- **যে দাবিটা ব্যর্থ হয়েছে: "IABC-v2 impairment-aware correction DFT-SIC-কে হারানোর মূল অস্ত্র।"** E4 (phase ≤5°), E5 (gain 0.5–2 dB), E6 (phase 10–15° OOD) — এই তিনটার প্রায় সব condition-এ IABR বনাম Abl1_no_IABC (IABC ছাড়া) পার্থক্য statistically noise-এর মধ্যে (paired bootstrap CI শূন্য বাদ দেয় না)। শুধু **একটা condition-এ** (SNR 15 dB, gain error 4 dB OOD) IABC-এর প্রকৃত, পুনরাবৃত্ত (তিনটা IABC variant-এ) সুবিধা দেখা গেছে — IABR (0.8033) Abl1-এর (0.7790) চেয়ে +2.43 pp এবং Teacher-এর (0.7787) চেয়ে +2.46 pp এগিয়ে। কিন্তু এই একই condition-এ IABR **Student-কে (0.7611) +4.22 pp** ব্যবধানে হারায় — অর্থাৎ IABC-এর লাভ Teacher/Abl1-এর বিরুদ্ধে ~+2.4 pp হলেও Student-এর বিরুদ্ধে অনেক বড়, তাই এই তিনটা তুলনার জন্য একটাই সংখ্যা ("+2.4 থেকে +2.9 pp") ব্যবহার করা ভুল। আর সেখানেও **0-parameter DFT-SIC** (0.8387) IABR-এর (0.8033) চেয়ে এগিয়ে থাকে। প্রতিটা impairment condition-এ (E4/E5/E6, দুটো SNR, সব level) DFT-SIC IABR-কে হারায়।
- **যেখানে সবচেয়ে বড় architectural দুর্বলতা ধরা পড়েছে: angular resolution।** E8-এ দুটো path 10° কাছাকাছি হলে IABR ধসে পড়ে (0.6145, Teacher 0.9322, DFT-SIC 0.9765) — FNO-এর স্তরে নেমে যায়। কারণ (Abl-2 দিয়ে প্রমাণিত): coarse heatmap মাত্র 16×16, আর 3×3 NMS দুটো কাছের path-কে একই cell-এ merge করে ফেলে। Coarse decode একা Pd মাত্র 0.1396 দেয় — **প্রায় পুরো accuracy আসে learned crop-refinement head থেকে** (0.1396 → 0.6909, +55.1 pp)।
- **SNR extrapolation-এ ব্যর্থতা:** training range (−15..24 dB)-এর বাইরে, high SNR (30, 35 dB)-এ IABR-এর Pd **কমে যায়** (0.8554 → 0.8411), যেখানে Teacher বাড়তে থাকে (0.9390 → 0.9465)। High-SNR-এ IABR-এর plateau (~0.87) মূলত দুটো কারণে: কাছাকাছি path merge (SNR≥15-এ wrong-peak path-এর 82% অন্য একটা true path-এর 1.5 cell-এর মধ্যে) আর end-fire sub-cell precision (25 dB-এ miss-এর 85% end-fire-এ)।
- **Training pipeline নিজে সম্পূর্ণ সুস্থ ও reproducible:** 3 seed একই জায়গায় converge করে (E3 std মাত্র 0.0010, 95% CI ±0.0024), কোনো divergence নেই।

**এক লাইনে:** IABR-Net একটা **parameter/FLOP-efficient, robust-to-drop (কখনো sample drop করে না), moderate-accuracy** architecture — কিন্তু "impairment-aware" নামের মূল novelty claim প্রায় সব robustness test-এ অপ্রমাণিত থেকে গেছে, এবং এর angular resolution ও high-SNR extrapolation-এর সীমাবদ্ধতা এর accuracy-কে DFT-SIC-এর মতো একটা 0-parameter classical method-এর নিচে রেখে দিয়েছে।

### Parameter/FLOP headline table

| Model | Params | Teacher-এর % | GFLOPs | Teacher-এর তুলনায় | GPU mem (batch32) | Weights file |
|---|---|---|---|---|---|---|
| Teacher | 469,393 | — | 15.203 | — | 185.85 MB | 2.80 MB |
| Student (r=8) | 314,513 | 67.0% | 10.157 | 1.5× কম | 169.84 MB | 2.33 MB |
| FNO | 334,321 | 71.2% | 0.172 | 88.4× কম | 187.67 MB | 1.40 MB |
| DFT-SIC | 0 | — | measure করা হয়নি (CPU) | — | — | — |
| **IABR-Net** | **195,024** | **41.5%** | **0.097** | **~156× কম** | **65.74 MB** | **1.01 MB** |

---

## ২. প্রতিটা experiment ধাপে ধাপে

### ২.০ Sanity ও Verification chain — E0a, BANKS_build, E0b, V0, V1

এই পাঁচটা gate পুরো run-এর "ভিত্তি"। এরা pass না করলে বাকি কোনো সংখ্যা বিশ্বাসযোগ্য হতো না।

| Test | কী verify করে | মূল সংখ্যা | Verdict |
|---|---|---|---|
| **E0a** (Phase-0 unit tests) | Physics: unitary codebook, simulator == original generator, gain/phase injection, 16↔64 conversion, angle↔cell mapping | 11/11 check pass, error ~1e-15 বা 0.0 | ✅ Success |
| **BANKS_build** | 56টা fixed robustness bank আছে ও hash অক্ষত | 56 bank, sha256 সব মিলেছে (নিজস্ব যাচাই: 0 mismatch) | ✅ Success (এই GPU-তে regenerate হয়নি, `build_min ≈ 0`) |
| **E0b** (model instantiation) | IABR-Net-এর আসল params/FLOPs | 195,024 params, 97,169,941 FLOPs | ✅ Success — estimate (193,104 / 95.3M)-এর 1–2% কাছে, পার্থক্য BatchNorm bookkeeping দিয়ে ব্যাখ্যাযোগ্য |
| **V0** (Teacher reproduces registry) | Weights + evaluator + frozen bank ঠিক আছে কিনা | Teacher mean Pd 0.7102636, registry 0.7103496, diff −8.6e-5 | ✅ Success — tolerance (2e-3)-এর 23 গুণ ভেতরে, কিন্তু আর "bit-exact" (13+ sig fig) নয় |
| **V1** (generator fidelity) | নতুন vectorized simulator বনাম frozen bank distribution | সব SNR-এ diff < 3σ (0.42σ, 0.32σ, 0.11σ) | ✅ Success — coverage সীমিত (শুধু L=3, clean) |

**Theory/Code সংক্ষেপে:** E0a checks করে (ক) `W_ideal`/`F_ideal` unitary কিনা, (খ) vectorized `simulate()` original `DG.generate_channel_v2`-এর সাথে মেলে কিনা, (গ) gain/phase injection সঠিক range-এ আছে কিনা, (ঘ) inverse codebook impairment-কে antenna-domain-এ diagonal matrix হিসেবে ফেরত আনে কিনা (IABC-v2-এর গাণিতিক ভিত্তি — error 7.03e-15), (ঙ) 16×16↔64×64 conversion bit-exact কিনা। V0 frozen 8000-sample bank-এ Teacher চালিয়ে registry-র মান reproduce করে (evaluator/weights/data সব ঠিক প্রমাণ করে)। V1 নতুন generator দিয়ে বানানো clean L=3 bank-এ Teacher-এর Pd frozen bank-এর সমান কিনা (Bernoulli SE-ভিত্তিক 3σ test) দেখে।

**কেন গুরুত্বপূর্ণ:** এই পাঁচটা gate pass করায়, পরের সব experiment-এ (E3–E11, Abl-*) IABR-Net যেখানে পিছিয়ে (E3 mean Pd, high-SNR plateau, E8-এর 10° collapse) বা এগিয়ে (strict Pd, FLOPs), সেই পার্থক্য সিমুলেটর/evaluator/data bug থেকে নয় — **model-এর নিজের**।

**দুর্বলতা:** (১) V0 এবার bit-exact নয় (আগে "13+ sig fig exact" claim ছিল, এখন −8.6e-5 diff) — thesis-এ "reproduces to within 8.6e-5" লিখতে হবে, "bit-exact" নয়। (২) V1-এর coverage সরু (শুধু L=3, clean, ৩টা SNR)। (৩) BANKS এই GPU machine-এ regenerate হয়নি, শুধু hash-check হয়েছে। (৪) E0b-এর FLOP breakdown-এ Conv2D ছাড়া বাকি 1.83M FLOP কোন op থেকে এসেছে তা স্পষ্ট নয় (total সংখ্যা ঠিক)।

---

### ২.১ Training ও In-distribution headline — TRAIN_s0/s1/s2, E3, E11, SNR_tails

**কী test করা হয়েছে:** IABR-Net তিনটা seed (0,1,2) দিয়ে পুরো 20,000-step synthetic on-the-fly training; frozen 8000-sample bank (L=3, SNR −10..25 dB)-এ headline comparison; multi-seed variance (E11); training range-এর বাইরে (SNR −25,−20,30,35 dB) extrapolation।

**Theory:** Loss = 1·heat + 1·off + 0.5·pair + 0.1·snr। IABR-Net-এর বাজি হলো 16×16 native beamspace grid + sub-cell offset regression, যা Teacher-এর 256×256 grid-এর তুলনায় FLOPs অনেক কমায়। Hypothesis ছিল (FINAL_RESEARCH_REPORT §27): "IABR-Net-এর PD/RMSE matched SNR-এ DFT-SIC আর Teacher-এর **মাঝখানে** পড়বে।"

**Code সংক্ষেপে:** Adam optimizer, cosine LR decay, per-sample L~U{1..9}, SNR~U{−15..24}dB, per-sample impairment (phase δ~U(0,5°), gain γ~U(0,2dB))। E3-এ প্রতিটা model frozen bank-এর 16×16 downsample-এ চালানো হয়েছে; IABR-এর `predict_iabr`-এ heatmap sigmoid → `local_maxima(K=10)` → top-L cell → crop-refine offset → angle।

**Result — training:**

| seed | final loss | heat | off | pair | snr |
|---|---|---|---|---|---|
| s0 | 0.41664 | 0.39241 | 0.02086 | 0.00616 | 0.00288 |
| s1 | 0.41244 | 0.38859 | 0.02041 | 0.00613 | 0.00376 |
| s2 | 0.41293 | 0.38903 | 0.02052 | 0.00615 | 0.00297 |

তিনটা seed একই জায়গায় (~0.41–0.42) converge করে। s2-তে step 2000→2200-এ একটা resume-artifact spike আছে (loss 0.4464→0.5841, কারণ দ্বিতীয় session-এ checkpoint resume-এর সময় optimizer state ঠিকমতো restore হয়নি বলে মনে হয়) — কিন্তু final metric-এ ক্ষতি হয়নি (s2-ই আসলে সেরা seed, E3 Pd 0.6916)। Heat loss মোট loss-এর ~94%, আর 10k step পরে প্রায় flat — architecture/representation-এর সীমা, আরও step দিলে লাভ হবে না।

**Result — E3 headline (paper-style, mean over SNR):**

| model | mean Pd (paper) | mean Pd (strict) | mean Pd (source) |
|---|---|---|---|
| DFT-SIC | **0.7326** | 0.7326 | 0.6321 |
| Teacher | 0.7103 | 0.6727 | 0.5775 |
| Student | 0.6956 | 0.6469 | 0.5458 |
| **IABR (3-seed mean)** | **≈0.6907** (s0 0.6909, s1 0.6897, s2 0.6916) | **≈0.6907** | ≈0.5823 |
| FNO | 0.6235 | 0.6018 | 0.4575 |

15 dB-এ: IABR_s0 0.8662, Teacher 0.8994, DFT-SIC 0.9042। High-SNR plateau: IABR 20→25 dB-এ প্রায় flat (0.8733→0.8745), Teacher 0.9251→0.9376। **[derived, cache re-scoring, report 02]** SNR≥15-এ IABR-এর "পুরো path ভুল" ঘটনার 82% এমন path যা অন্য true path-এর 1.5 coarse cell-এর মধ্যে (coarse-grid merge) — Teacher/DFT-SIC-এর এই হার মাত্র 40–42%। 25 dB-এ IABR-এর miss-এর 85% end-fire-এ (end-fire miss rate 0.324 বনাম Teacher 0.162)।

**Result — E11 (multi-seed variance):**

| condition | mean | std | values |
|---|---|---|---|
| E3_frozen | 0.6907 | 0.0010 | [0.6909, 0.6897, 0.6916] |
| phase_d5_snr15 | 0.8501 | 0.0010 | [0.851, 0.8503, 0.849] |
| gain_g2_snr15 | 0.8489 | 0.0025 | [0.8463, 0.8513, 0.849] |

খুব stable — 95% CI ±0.0024 (E3)। শুধু 25 dB-এ per-seed spread বেশি (std 0.0137, বাকি SNR-এ ~0.001–0.003) — training-range-এর কিনারায় model-এর আচরণ অস্থির হওয়ার একটা সংকেত।

**Result — SNR_tails (extrapolation, single seed s0):**

| model | −25 dB | −20 dB | 30 dB | 35 dB |
|---|---|---|---|---|
| Teacher | 0.0184 | 0.0257 | 0.9390 | **0.9465** (বাড়ছে) |
| DFT-SIC | 0.0213 | 0.0303 | 0.9240 | 0.9370 |
| **IABR_s0** | 0.0190 | 0.0273 | 0.8554 | **0.8411** (কমছে) |

Low tail-এ সবাই chance floor-এ (inconclusive)। High tail-এ IABR-এর একমাত্র স্পষ্ট দুর্বলতা: SNR বাড়লে performance **কমছে**, Teacher একই training range পেয়েও বাড়তে থাকে। **Abl3_no_SE-কে এই tail bank-এ চালানো হয়নি — এটা একটা missing experiment** (mother synthesis নিজেই এটা চিহ্নিত করেছে, ৩ নম্বর research fix হিসেবে §৫-এ প্রস্তাবিত)।

![E3 SNR sweep](../_gpu_run_extracted/IABR_Net_TestSuite/outputs/figures/E3_snr_sweep.png)

**Figure ব্যাখ্যা:** তিনটা panel (Pd paper-style, Pd strict, RMSE), সব 11টা model, SNR −10 থেকে 25 dB। Panel 1-এ নিম্ন SNR-এ লাল DFT-SIC সবার উপরে, উচ্চ SNR-এ নীল Teacher তাকে ছাড়িয়ে যায়; IABR-এর তিনটা seed + E1 + Abl1 + Abl3 একটা গুচ্ছ হয়ে ~0.87-এ flat হয়ে যায়। Panel 2 (strict)-এ ছবি বদলে যায় — low-mid SNR-এ IABR গুচ্ছ Teacher/Student-এর উপরে (drop policy-র প্রভাব)। **Figure-এর একটা bug:** matplotlib-এর 10-রঙা cycle ঘুরে আসায় Teacher আর Abl3_no_SE দুটোই একই নীল রঙে আঁকা হয়েছে — legend দেখে আলাদা করা যায় না, thesis-এ ব্যবহারের আগে ঠিক করতে হবে (দুই independent analyst — report 01 ও 02 — এটা আলাদাভাবে ধরেছে)।

**Verdict:** Training — ✅ Success (stable, reproducible)। E3 — **Mixed** (low-SNR-এ ও strict metric-এ ভালো, কিন্তু DFT-SIC-কে সব SNR-এ হারায়, high-SNR plateau আছে)। E11 — ✅ Success (ordering robust)। SNR_tails — **Failure** (high-tail extrapolation ব্যর্থ)।

---

### ২.২ Hardware Impairment Robustness — E4 (phase), E5 (gain), E6 (OOD phase)

**কী test করা হয়েছে:** প্রতি antenna-য় random phase error (E4: δ∈{0,1,2,5}°, E6 OOD: δ∈{10,15}°) বা gain error (E5: γ∈{0,0.5,1,2,4dB}, 4dB = OOD), SNR {0,15} dB, L=3, 500 sample প্রতি condition। Model: Teacher, Student, FNO, DFT-SIC, IABR_s0, E1_capacity_control (IABC আছে, pair loss=0), Abl1_no_IABC (IABC নেই)।

**Theory — impairment আসলে কতটা "জোরালো" [MY ANALYSIS, report 03]:** Distortion-to-signal ratio (DSR) হিসাব করে দেখা যায়:

| condition | DSR (dB) | SNR 15-তে noise (−15dB)-এর তুলনায় |
|---|---|---|
| phase 5° (training max) | −22.9 | নগণ্য (noise-এর ~8dB নিচে) |
| phase 15° (OOD) | −13.4 | noise-এর চেয়ে বেশি |
| gain 2 dB (training max) | −14.3 | প্রায় সমান |
| **gain 4 dB (OOD)** | **−7.7** | **noise-এর ~7dB বেশি** |

অর্থাৎ E4-এর পুরো range (≤5°) physically noise-এর নিচে ডুবে থাকে — IABC-কে প্রমাণ করার সুযোগ প্রায় নেই। একমাত্র "active" condition হলো SNR 15-এ gain 4dB (OOD)। **আরেকটা গুরুত্বপূর্ণ finding:** প্রতিটা impairment level আলাদা seed/bank-এর, তাই curve-এ ±2pp sampling noise থাকে — প্রমাণ: deterministic, training-হীন DFT-SIC-ও SNR 15, 0°→1°-এ −1.8pp নামে, যদিও physics অনুযায়ী প্রভাব শূন্য হওয়ার কথা।

**Result (SNR 15 dB, paper-style Pd):**

| model | E4: 0° | E4: 5° | E5: 2 dB | **E5: 4 dB (OOD)** | E6: 10° | E6: 15° |
|---|---|---|---|---|---|---|
| Teacher | 0.9127 | 0.8859 | 0.8541 | 0.7787 | 0.8620 | 0.8081 |
| DFT-SIC | 0.9183 | 0.9007 | 0.8860 | **0.8387** | 0.8870 | 0.8427 |
| **IABR_s0** | 0.8700 | 0.8510 | 0.8463 | **0.8033** | 0.8487 | 0.8043 |
| E1 (pair=0) | 0.8757 | 0.8527 | 0.8477 | 0.8003 | 0.8490 | 0.7987 |
| Abl1 (no IABC) | 0.8737 | 0.8607 | 0.8437 | **0.7790** | 0.8447 | 0.8017 |

**Comparison — IABC-এর প্রকৃত অবদান [MY ANALYSIS, paired bootstrap, report 03]:**

| তুলনা | E4 (5°) | E5 (2 dB) | **E5 (4 dB OOD)** | E6 (15°) |
|---|---|---|---|---|
| IABR − Abl1 | −0.97 pp (ns) | +0.27 pp (ns) | **+2.43 pp [+1.07, +3.87]** | +0.27 pp (ns) |
| IABR − Teacher (strict, paired) | −0.83 pp (ns) | −0.43 pp (ns) | **+2.93 pp [+1.17, +4.60]** | +0.43 pp (ns) |
| IABR − DFT-SIC | −4.1 থেকে −5.2 pp (সব significant, শূন্যের নিচে) | −3.97 pp | −3.53 pp | −3.8 pp |

**একমাত্র জায়গা যেখানে IABC-এর লাভ প্রমাণযোগ্য:** SNR 15, gain 4 dB (OOD)। তিনটা আলাদা IABC-variant (IABR_s0, E1, Abl3_no_SE) সবাই Abl1-এর চেয়ে +2.1 থেকে +2.6 pp এগিয়ে, CI শূন্য বাদ দেয়। কিন্তু এই একই condition-এ IABR **Student-কে (0.7611) +4.22 pp** ব্যবধানে হারায় — যা Teacher/Abl1-এর বিরুদ্ধে দেখা +2.4–2.6 pp পরিসরের বাইরে, তাই "Teacher/Student/Abl1-কে +2.4 থেকে +2.9 pp ব্যবধানে হারায়" জাতীয় একক সংখ্যা ভুল — প্রতিটা baseline-এর বিরুদ্ধে margin আলাদা। বাকি সব condition-এ (E4-এর সব level, E5-এর 0–2dB, E6-এর সব level) IABR বনাম Abl1 পার্থক্য noise-এর ভেতরে — IABC-v2 কার্যত কোনো প্রমাণযোগ্য লাভ দেয় না।

**গভীর কারণ [report 03 §4]:** Training log-এর `pair` metric দেখায় IABC distortion-এর মাত্র **~24%** কমাতে পেরেছে (Abl1-এর raw 0.00815 → IABR-এর 0.00616)। E1 (pair supervision ছাড়া) IABC এমন একটা transform শিখেছে যা clean থেকে ৭.৭ গুণ দূরে (0.0625), তবুও Pd প্রায় সমান — মানে detection performance "clean-এর দিকে correction"-এর উপর নির্ভর করে না। IABC সম্ভবত একটা global re-weighting শিখেছে, সত্যিকারের impairment corrector নয় (hypothesis, সরাসরি inspect করা হয়নি)।

![E4 phase sweep SNR 0](../_gpu_run_extracted/IABR_Net_TestSuite/outputs/figures/E4_phase_snr0.png)
![E4 phase sweep SNR 15](../_gpu_run_extracted/IABR_Net_TestSuite/outputs/figures/E4_phase_snr15.png)

**Figure ব্যাখ্যা (E4):** SNR 0-তে (উপরের ছবি) সব line-ই phase বাড়ার সাথে সামান্য *উপরে* উঠছে — এটা sampling artifact, impairment-এর real effect নয় (§0.5)। SNR 15-এ (নিচের ছবি) IABR/E1/Abl1 একটা জট পাকিয়ে আছে, কোনো দৃশ্যমান পার্থক্য নেই; পুরো IABR-family Teacher আর DFT-SIC-এর ~4–5pp নিচে, আর এই ফাঁক impairment বাড়ার সাথে বদলায় না — এটা impairment-robustness-এর সমস্যা নয়, **clean-condition accuracy gap**।

![E5 gain sweep SNR 0](../_gpu_run_extracted/IABR_Net_TestSuite/outputs/figures/E5_gain_snr0.png)
![E5 gain sweep SNR 15](../_gpu_run_extracted/IABR_Net_TestSuite/outputs/figures/E5_gain_snr15.png)

**Figure ব্যাখ্যা (E5, সবচেয়ে তথ্যবহুল figure):** SNR 15 panel-এ 0–1dB-এ সব line প্রায় সমান্তরাল, কিন্তু **2dB থেকে Teacher আর Student খাড়া ভাবে নামতে থাকে**, আর 4dB-এ Teacher (0.7787) IABR/E1 (~0.80)-এর নিচে চলে যায় — একটা crossover। DFT-SIC সবচেয়ে ধীরে নামে, সবসময় সবার উপরে। এটাই পুরো experiment-group-এর একমাত্র জায়গা যেখানে IABC-যুক্ত ও IABC-ছাড়া model দৃশ্যমানভাবে আলাদা হয়।

**E6-এর কোনো figure নেই** (`e6()` `plot_sweep()` call করে না — code omission, file হারিয়ে যায়নি)।

**Verdict:** E4 — Mixed/Inconclusive (IABC claim fail to demonstrate)। E5 — **Partial success** (একমাত্র 4dB OOD-তে IABC-এর প্রমাণযোগ্য লাভ, কিন্তু DFT-SIC-এর নিচেই)। E6 — Failure for IABC claim, তবে কোনো catastrophic break নেই (graceful degradation)।

---

### ২.৩ Scene Difficulty — E7 (path count), E8 (angular separation), E10 (nuisance path)

এই তিনটা test-এ **কোনো hardware impairment নেই** (phase=0, gain=0) — তাই IABC-v2-এর এখানে কিছু করার সুযোগ নেই। এই test-গুলো আসলে IABR-এর trunk + 16×16 coarse heatmap + 3×3 NMS decoder + crop-refinement head পরীক্ষা করে।

**E7 — Path count sweep (L = 1..8, 10):** SNR {0,15}, প্রতি L-এ 500 sample। Total power স্থির থাকায় (α~CN(0,1/L)) L বাড়লে প্রতিটা path দুর্বল হয়, আর beamspace-এ path-গুলো ঘন হয়।

| SNR 15 paper-style | L=1 | L=3 | L=5 | L=8 | L=10 |
|---|---|---|---|---|---|
| Teacher | 0.9430 | 0.9127 | 0.8634 | 0.7867 | 0.7494 |
| DFT-SIC | 0.9450 | 0.9183 | 0.8626 | 0.7951 | 0.7549 |
| IABR_s0 | 0.9280 | 0.8700 | 0.7988 | 0.7113 | 0.6694 |

paper-style-এ Teacher-এর তুলনায় ফাঁক L1-এ −1.5pp থেকে L10-এ −8.0pp পর্যন্ত ধারাবাহিকভাবে বাড়ে। **কিন্তু strict metric-এ চিত্র উল্টে যায়:** L10 SNR15-এ IABR 0.6694 বনাম Teacher 0.5426 (Teacher 138/500 sample drop করে, IABR কখনো করে না)। L10 SNR0-এ IABR strict 0.3839 বনাম Teacher 0.0676 (Teacher 425/500 drop করে)। **DFT-SIC strict metric-এও প্রতিটা L-এ সবার উপরে।** OOD ধাপ (L8→L10)-এ IABR-এর কোনো "cliff" নেই — ঢাল DFT-SIC-এর মতোই মসৃণ, তাই এই পতন physics-driven, OOD generalization failure নয়।

**E8 — Angular separation sweep (30°→1°, L=2, SNR 15):** Training-এ minimum Euclidean separation ছিল 30° — তাই s≤20° সব learned model-এর জন্যই কার্যত OOD।

| separation | Teacher | DFT-SIC | **IABR_s0** |
|---|---|---|---|
| 30° | 0.9888 | 0.9950 | 0.9935 |
| 20° | 0.9879 | 0.9950 | 0.9760 |
| **10°** | **0.9322** | **0.9765** | **0.6145** |
| 5° | 0.6811 | 0.5705 | 0.2880 |

**10° separation-এ IABR ধসে পড়ে** (0.6145, Teacher-এর তুলনায় −0.3177, DFT-SIC-এর তুলনায় −0.3620) — FNO-এর (0.6200) স্তরে নেমে যায়। p50=0.4515° কিন্তু p95=83.31° — bimodal: প্রথম path প্রায় নিখুঁত, দ্বিতীয়টা প্রায়ই সম্পূর্ণ ভুল জায়গায় (noise-এর local max)। কারণ: 10°-এ দুটো path-এর দূরত্ব u-domain-এ মাত্র ~0.9–1.4 cell, যা 3×3 local-max filter-এর চোখে একটাই peak। (3° থেকে 1°-এর মধ্যে Pd আবার বাড়া দেখা যায় — এটা resolution-এর উন্নতি নয়, 1° threshold-এর একটা metric artefact, design doc-এ আগে থেকেই "diagnostic-only" বলে চিহ্নিত।)

**E10 — Nuisance path stress test (3 principal + 1 interferer, power −20/−10/0 dB, 200 scene):**

| SNR 15, principal Pd | −20 dB | −10 dB | 0 dB |
|---|---|---|---|
| Teacher | 0.9044 | 0.8998 | 0.8795 |
| DFT-SIC | 0.8892 | 0.8700 | 0.8817 |
| IABR_s0 | 0.8433 | 0.8283 | 0.8150 |

IABR-এর principal Pd nuisance power বাড়ার সাথে একটানা নামে (dolation ≤0.0283), Teacher/DFT-SIC-এর তুলনায় SNR 15-এ 4–7pp পিছিয়ে। **[FILE, JSON]** IABR-এর principal p95 (SNR 15): −20 dB-এ 23.55°, −10 dB-এ 24.49°, 0 dB-এ 52.68° — শক্তিশালী nuisance-এ gross error বেড়ে যায়। **(এই সংখ্যাগুলো `E10_nuisance_path.json`/`tables/E10.md`-এ সরাসরি pd_principal/pd_nuisance column-এ আছে, তবে p95 field রাখা হয়নি — এই p95 বিশ্লেষণটা report 04-এর নিজস্ব cached-prediction re-scoring script থেকে এসেছে, তাই raw file দিয়ে সরাসরি cross-verify করা যায়নি — verifier একে "UNVERIFIABLE" চিহ্নিত করেছে, বিস্তারিত §৬-এ।)** কম SNR-এ strict metric-এ IABR Teacher-এর চেয়ে অনেক ভালো (−20dB-এ 0.6425 বনাম 0.3717, Teacher 81/200 scene drop করে)। **PROJECT_STATUS-এর আগের "Student beats Teacher under nuisance" ফল এই নতুন bank-এ reproduce হয়নি** (Δ = −0.0316, উল্টো দিকে) — সম্ভবত সেটা sampling noise ছিল।

![E7 path count SNR 0](../_gpu_run_extracted/IABR_Net_TestSuite/outputs/figures/E7_pathcount_snr0.png)
![E7 path count SNR 15](../_gpu_run_extracted/IABR_Net_TestSuite/outputs/figures/E7_pathcount_snr15.png)

**Figure ব্যাখ্যা:** SNR 15 (নিচের ছবি)-তে Teacher ও DFT-SIC প্রায় এক line হয়ে উপরে থাকে (0.94→0.75); IABR-family খাড়া ঢালে নেমে L10-এ ~0.67-এ পৌঁছায় — দুই গোষ্ঠীর মধ্যে ক্রমশ চওড়া হওয়া একটা "wedge" তৈরি হয়। এটা high-SNR-এ noise আর সমস্যা না হওয়ায় যা বাকি থাকে তা হলো resolution আর sub-cell precision।

![E8 angular separation](../_gpu_run_extracted/IABR_Net_TestSuite/outputs/figures/E8_separation.png)

**Figure ব্যাখ্যা (y-axis এখানে `pd_source`, `pd_paper` নয়):** IABR-family (বেগুনি/বাদামি/গোলাপি) 20° পর্যন্ত DFT-SIC/Teacher-এর কাছাকাছি (~0.97–0.99), কিন্তু **10°-এ হঠাৎ ~0.59-এ ধস** — Teacher/Student-এর অনেক নিচে। এটাই এই পুরো test-suite-এর সবচেয়ে স্পষ্ট, একক architectural failure।

**E10-এর কোনো figure নেই** (শুধু table)।

**Verdict:** E7 — Mixed (strict/কম-SNR-এ ভালো, paper-style high-SNR-এ ফাঁক বাড়ে)। E8 — Mixed, কিন্তু 10°-এ **স্পষ্ট Failure**। E10 — Mixed (robust কিন্তু কম accurate)।

---

### ২.৪ Ablations ও Controls — E1, E2, Abl1, Abl3, Abl-2, Abl-6

**Controls টেবিল (E3_frozen mean Pd + robustness condition, paper-style):**

| model | params | E3_frozen | phase_d5_snr15 | gain_g2_snr15 | **gain_g4_snr15** |
|---|---|---|---|---|---|
| IABR_s0 (main) | 195,024 | 0.6909 | 0.8510 | 0.8463 | 0.8033 |
| E1_capacity_control (pair=0) | 195,024 | 0.6887 | 0.8527 | 0.8477 | 0.8003 |
| **E2_dense_frontend** | **457,430** | **0.5704** | 0.7473 | 0.7467 | **0.6527** |
| Abl1_no_IABC (front=none) | 194,774 | 0.6871 | 0.8607 | 0.8437 | 0.7790 |
| Abl3_no_SE | 192,093 | 0.6854 | 0.8537 | 0.8437 | 0.8047 |

**E1_capacity_control** — মূল architecture হুবহু একই, শুধু `λ3(pair)=0`। সব condition-এ IABR_s0-এর ≤0.3pp-এর মধ্যে থাকে → **`L_pair` loss কার্যত অকার্যকর প্রমাণিত** (Abl-6 sweep-এও একই সিদ্ধান্ত, নিচে দেখুন)। **methodology caveat:** design doc-এ E1-কে "same-capacity, no-adaptivity" module বলার কথা ছিল, বাস্তবায়িত E1 আসলে IABC আছে এমন architecture, শুধু pair loss=0 — এই ভূমিকাটা বরং **Abl1** পালন করে (params প্রায় identical, 194,774 বনাম 195,024, IABC সম্পূর্ণ নেই)।

**E2_dense_frontend** — IABC-v2-এর বদলে একটা সাধারণ `Dense(512→512)` front-end, 457,430 params (~2.34× বেশি)। প্রতিটা condition-এ E1-এর চেয়ে 10–15pp খারাপ (gain_g4-এ −14.8pp), যদিও params অনেক বেশি — এমনকি Abl1 (front='none', কোনো correction-ই নেই)-এর চেয়েও খারাপ। **Physics-informed inductive bias > raw capacity — এই সিদ্ধান্তের সবচেয়ে জোরালো প্রমাণ।**

**Abl1_no_IABC** — IABC-এর "pure" ablation (params প্রায় identical to main)। E1-এর তুলনায়: E3_frozen প্রায় সমান, phase_d5-এ Abl1 বরং একটু ভালো (+0.80pp), কিন্তু **gain_g4 (OOD)-এ IABR-family স্পষ্টভাবে এগিয়ে** (E1 0.8003 বনাম Abl1 0.7790, +2.13pp)। এটাই IABC-এর একমাত্র condition-specific, প্রমাণযোগ্য সুবিধা।

**Abl3_no_SE** — SE attention সম্পূর্ণ সরিয়ে দিলে (192,093 params, 2,931 কম) সব condition-এ IABR_s0-এর ≤0.6pp-এর মধ্যে, দিক মিশ্র (দুই condition-এ SE ছাড়া বরং একটু ভালো)। **SE attention কার্যত অকার্যকর।**

**Abl-2 — Decode ablation (headline finding of this group):**

| decode mode | pd_paper | p50 | p95_broadside | p95_endfire |
|---|---|---|---|---|
| coarse (16×16 grid center শুধু) | **0.1396** | 3.6746° | 71.29° | 173.80° |
| parabolic interpolation | 0.1786 | 2.7907° | 57.31° | 170.77° |
| **learned refine (আসল IABR)** | **0.6909** | **0.3711°** | **43.37°** | **164.29°** |

Coarse-only decode-এ Pd মাত্র **0.1396** (86% component ভুল)! **Learned crop-refinement head একলাফে Pd 0.6909-এ নিয়ে যায় — coarse-এর তুলনায় +55.1 pp।** IABR-Net-এর প্রায় পুরো accuracy এই refinement head থেকে আসে, শুধুমাত্র 41.5% params দিয়ে Teacher-এর (0.7158, pooled) 96.5% আর DFT-SIC-এর (0.7326) 94.3% accuracy পৌঁছায়। তবে **end-fire tail এখনও অমীমাংসিত**: p95_endfire সব model-এই বিশাল (161°–174°, physics-driven, model-independent Jacobian singularity), আর learned-refine সামান্য সবচেয়ে ভালো (164.29°) হলেও কোনো বড় উন্নতি নেই।

**Abl-6 — Loss-weight sweep (λ3∈{0,0.1,0.5,1.0} × λ4∈{0,0.1,0.3}, 12 run, 5000 step single-seed):** E3_subset_pd রেঞ্জ 0.6555–0.6601 (0.46pp spread), phase5_pd 0.8150–0.8303 (1.53pp), gain2_pd 0.8040–0.8180 (1.40pp)। Main model-এর নিজের weight (λ3=0.5, λ4=0.1) কোনো metric-এই সেরা না। **এই architecture loss-weight choice-এর প্রতি খুবই robust — কিন্তু এটাও নিশ্চিত করে যে main-এর জন্য বেছে নেওয়া weight-এর কোনো বিশেষ যৌক্তিকতা এই sweep থেকে পাওয়া যায় না।**

`training_loss.png`-এর ডান panel এই 12-run sweep দেখায় — সব curve প্রায় identical trajectory-তে ~0.42–0.43-এ converge করে, একে অপরের থেকে আলাদা করা প্রায় অসম্ভব — visually এই robustness নিশ্চিত করে। বাম panel-এ E2_dense_frontend আলাদাভাবে (উঁচুতে, ধীরে) নামে, আর IABR_s2-তে resume-artifact spike দেখা যায়।

**Verdict:** E1 — Neutral (pair loss কার্যত অকার্যকর)। E2 — **Strong confirmatory failure** (দিয়ে বোঝা যায় architecture-ই কারণ, capacity নয়)। Abl1 — Neutral bulk, শুধু gain4-এ +2.13pp। Abl3 — Failure (SE অকার্যকর)। Abl-2 — **✅ Headline success** (refinement head-ই মূল ইঞ্জিন)। Abl-6 — ✅ Robust, কিন্তু tuning-এর ন্যায্যতা দুর্বল।

---

### ২.৫ Efficiency — E9 (params/FLOPs/latency/memory)

**Result — মূল সারণি:**

| model | params | GFLOPs | weights_MB | gpu_peak_MB | nn_ms (b=1) | e2e_ms | mean_pd |
|---|---|---|---|---|---|---|---|
| Teacher | 469,393 | 15.203 | 2.80 | 186 | 7.850 | 9.48 | 0.7103 |
| Student | 314,513 | 10.157 | 2.33 | 170 | 7.092 | 8.76 | 0.6956 |
| FNO | 334,321 | 0.172 | 1.40 | 188 | **1.251** | **2.64** | 0.6235 |
| **IABR-Net** | **195,024** | **0.097** | **1.01** | **66** | 3.650 | 4.25 | 0.6909 |
| DFT-SIC | 0 | — | — | — | — | 8.03 | 0.7326 |

**Batch-sweep latency (throughput/s):**

| model | b=1 | b=8 | b=32 | b=128 (throughput/s) |
|---|---|---|---|---|
| Teacher | 7.85 ms | 10.35 ms | 28.68 ms | 141.95 ms (901.75/s) |
| Student | 7.09 ms | 10.68 ms | 24.41 ms | 116.38 ms (1099.82/s) |
| FNO | 1.25 ms | 1.41 ms | 4.74 ms | 24.34 ms (5259.62/s) |
| IABR-Net | 3.65 ms | 3.50 ms | 3.77 ms | 3.88 ms (**32992.61/s**) |

**IABR-Net-এর latency batch 1→128 জুড়ে প্রায় flat** (3.65→3.50→3.77→3.88 ms), মানে GPU compute-saturate হচ্ছে না — kernel-launch/overhead-bound। Teacher/Student/FNO-এর latency batch-এর সাথে প্রায় linear বাড়ে। ফলে batch-128-এ IABR-Net-এর throughput Teacher-এর **36.6×**, Student-এর 30.0×, FNO-এর 6.3× — কিন্তু এই বিশাল multiplier ছোট batch-এ অনেক কমে যায়: **batch=32-এ Teacher-এর তুলনায় মাত্র ~7.6× (8480.0 বনাম 1115.9 sample/s), batch=8-এ মাত্র ~3.0× (2287.7 বনাম 773.3)।** তাই "batch≥32-এ 36.6×" জাতীয় সাধারণীকরণ ভুল — এই ratio শুধু batch=128-এ সত্যি।

**একটা genuinely negative finding:** batch=1 real-time inference-এ IABR-Net *সবচেয়ে দ্রুত না* — **FNO NN-only-তে ~2.92× এবং end-to-end-এ ~1.61× দ্রুত**, যদিও FNO-এর FLOPs (0.172G) IABR-এর (0.097G) চেয়ে বেশি এবং accuracy অনেক খারাপ (Pd 0.6235 বনাম 0.6909)। কারণ: IABR-এর pipeline-এ per-element IABC correction, top-K gather, crop-based refine head — এই ছোট, sequential, gather/index-heavy op batch=1-এ kernel-launch overhead-dominated; FNO বড় কিন্তু fusable conv/FFT op চালায়। "কম FLOPs/params = কম latency" ধারণা সবসময় সত্য নয়।

**Decode-cost বিভাজন [derived]:** IABR-Net-এর NN-e2e gap (≈0.60 ms) সব model-এর মধ্যে সবচেয়ে সস্তা (Teacher ≈1.63ms, FNO ≈1.39ms) — 16×16 coarse heatmap-এর peak-search 256×256 blob-detector-এর চেয়ে ~2.3–2.8× সস্তা।

**Accuracy-per-efficiency:** Pd per 100K params: IABR-Net 0.3543 (সব NN baseline-এর সেরা)। Pd per GFLOP: IABR-Net ≈7.11–7.12 (FNO-এর প্রায় দ্বিগুণ; **note:** rounded 0.097 GFLOPs দিয়ে হিসাব করলে 7.123–7.124 আসে, precise 0.09717 দিয়ে 7.11 — সিদ্ধান্ত একই থাকে, শুধু rounding artifact)।

**Verdict:** **Mixed, কিন্তু সার্বিকভাবে ইতিবাচক দিকে ঝুঁকে থাকা একটা সফলতা।** Params/FLOPs/memory/high-batch-throughput-এ decisively জিতেছে, pre-run estimate প্রায় হুবহু মিলেছে। কিন্তু batch=1 real-time inference-এ FNO দ্রুততর, আর hypothesis "e2e latency may narrow the gap" ভুল প্রমাণিত হয়েছে (gap প্রায় একই থেকে গেছে)। এই group-এর জন্য কোনো figure/visualization তৈরি হয়নি।

---

## ৩. Master Comparison Table — সব experiment এক নজরে

| # | Experiment | কী test করে | IABR-Net result | Best baseline | Verdict |
|---|---|---|---|---|---|
| E0a | Phase-0 unit tests | Physics/simulator সঠিকতা | 11/11 pass, error ~1e-15 | — | ✅ Success |
| BANKS_build | 56 fixed eval bank | Data integrity | 56/56 hash match | — | ✅ Success (এই GPU-তে regenerate হয়নি) |
| E0b | Model instantiation | Real params/FLOPs | 195,024 params, 97.17M FLOPs | Teacher 469,393/15.20G | ✅ Success |
| V0 | Teacher registry reproduce | Evaluator/weights/bank সঠিকতা | Teacher mean Pd 0.71026 (registry 0.71035, diff −8.6e-5) | — | ✅ Success (আর bit-exact নয়) |
| V1 | নতুন generator fidelity | Simulator distribution match | সব SNR-এ diff < 3σ | — | ✅ Success (coverage সীমিত) |
| TRAIN_s0/s1/s2 | Training convergence | Loss ও seed stability | final loss 0.4124–0.4166 | — | ✅ Success (s2 resume spike) |
| **E3** | In-dist SNR sweep, headline | Paper Pd 3-seed mean **0.6907**, strict 0.6907 | DFT-SIC 0.7326, Teacher strict 0.6727 | Mixed — strict-এ Teacher-কে হারায়, DFT-SIC-এর কাছে সব metric-এ হারে |
| E11 | Multi-seed variance | Reproducibility | std 0.0010–0.0025 | — | ✅ Success |
| SNR_tails | OOD SNR extrapolation | Generalization | 30/35dB: 0.8554→0.8411 (কমছে) | Teacher: 0.9390→0.9465 (বাড়ছে) | ❌ Failure (high tail) |
| E4 | Phase error ≤5° | IABC benefit, low impairment | SNR15,5°: 0.8510 (Abl1 0.8607, খারাপ) | Teacher 0.8859, DFT-SIC 0.9007 | ❌ IABC-এর প্রমাণযোগ্য লাভ নেই |
| E5 | Gain error, incl. OOD 4dB | IABC benefit | SNR15,4dB(OOD): **0.8033**, vs Teacher +2.46pp, vs Abl1 +2.43pp, vs Student +4.22pp | Teacher 0.7787, DFT-SIC 0.8387 | ⚠️ Partial success — DFT-SIC-এর নিচে |
| E6 | OOD phase 10–15° | Distribution-shift | SNR15,15°: 0.8043 (Abl1 0.8017, ns) | Teacher 0.8081, DFT-SIC 0.8427 | ❌ IABC claim fail, graceful degradation |
| E7 | Path count L=1..10 | Multipath capacity | SNR15 paper L1 0.928→L10 0.6694; strict L10 0.6694 vs Teacher 0.5426 | Teacher (paper, high-L), DFT-SIC (strict) | Mixed |
| E8 | Angular separation 30°→1° | Resolution limit | **10°-এ ধস: 0.6145** | Teacher 0.9322, DFT-SIC 0.9765 | ❌ **Clear failure** |
| E10 | Nuisance path | Interference robustness | Principal Pd দোলা ≤0.0283, SNR15@0dB 0.8150 | Teacher 0.8795, DFT-SIC 0.8817 | Mixed |
| E1 (pair=0) | পেয়ার-লস দরকার কিনা | সব condition <0.3pp পার্থক্য | — | Neutral — pair loss কার্যত অকার্যকর |
| E2 (dense frontend) | Capacity বনাম architecture | E3: 0.5704 (main-এর চেয়ে 12.0pp খারাপ, 2.34× বেশি params) | — | ✅ Strong confirmatory failure |
| Abl1 (no IABC) | IABC pure ablation | E3: 0.6871 (main 0.6907, Δ −0.0038, ns) | — | Neutral bulk; শুধু gain_g4-এ +2.13pp |
| Abl3 (no SE) | SE attention দরকার কিনা | সব condition <0.6pp পার্থক্য | — | ❌ Failure — SE কার্যত অকার্যকর |
| Abl-2 | Decode ablation | Accuracy কোথা থেকে আসে | coarse 0.1396 → learned-refine **0.6909** (+55.1pp) | Teacher 0.7158, DFT-SIC 0.7326 (pooled) | ✅ **Headline success** |
| Abl-6 | Loss-weight sweep | λ3×λ4 sensitivity | Spread 0.46–1.53pp সব 12 combo-তে | — | ✅ Robust, tuning ন্যায্যতা দুর্বল |
| **E9** | Efficiency | Compute efficiency | 195,024 params, 0.097 GFLOPs, 66MB, b128 throughput 32,993/s (Teacher-এর 36.6×, শুধু b128-এ) | Teacher (params/FLOPs), FNO (b=1 latency 2.9× দ্রুত) | Mixed-positive |

---

## ৪. Success / Failure / Improvement

### ৪.১ Success (গুরুত্ব অনুযায়ী)

1. **Efficiency claim পুরোপুরি নিশ্চিত ও measurement-ভিত্তিক (E0b, E9):** 195,024 params (est. 193,104-এর 1% কাছে), 97,169,941 FLOPs (~156× কম Teacher-এর চেয়ে)। GPU memory 66MB (~35% Teacher-এর)। batch-128 throughput Teacher-এর 36.6× (batch=32-এ ~7.6×, batch=8-এ ~3.0×)।
2. **Learned crop-refinement head IABR-Net-এর accuracy-র প্রায় পুরো উৎস (Abl-2):** coarse-only 0.1396 → learned-refine 0.6909 (+55.1pp), Teacher-এর 96.5% ও DFT-SIC-এর 94.3% অর্জন মাত্র 41.5% params দিয়ে।
3. **Physics-informed architecture > raw capacity, সরাসরি প্রমাণিত (E2 vs E1/Abl1):** 2.34× বেশি params দিয়েও dense front-end 10–15pp খারাপ করে।
4. **কখনো sample drop করে না:** সব robustness experiment-এ n_dropped=0, তাই strict metric-এ কঠিন scene-এ Teacher/Student-কে বড় ব্যবধানে হারায় (E7 L10 strict: 0.3839 vs 0.0676; E10 SNR0/−20dB: 0.6425 vs 0.3717)।
5. **Training সম্পূর্ণ reproducible ও stable:** 3-seed E3 std 0.0010, gain_g2 std 0.0025, কোনো divergence নেই।
6. **Sanity/verification chain সম্পূর্ণ pass:** simulator, evaluator, weights, frozen bank — সব bug-free প্রমাণিত, তাই বাকি সব দুর্বলতা model-এর নিজের, infrastructure-এর নয়।
7. **একমাত্র OOD condition-এ IABC-এর প্রকৃত সুবিধা:** SNR15, gain 4dB — retention 92.3% (সর্বোচ্চ সব model-এ), Teacher/Abl1-কে হারায় (+2.4–2.9pp, তিনটা IABC variant-এ পুনরাবৃত্ত)।

### ৪.২ Failure/Weakness (গুরুত্ব অনুযায়ী)

1. **[সবচেয়ে গুরুত্বপূর্ণ] DFT-SIC (0 params)-কে কোনো test-এ হারাতে পারেনি।** E3, E4, E5 (এমনকি IABC-এর সেরা condition-এও), E6, E7 (strict), E8, E10 (SNR15) — সব জায়গায় হারে। Research report-এর central hypothesis "IABC-v2 এই critical 0-param baseline-কে হারাবে" **ব্যর্থ**।
2. **Angular resolution — সবচেয়ে বড় architectural gap।** E8: 10° separation-এ 0.6145 (Teacher 0.9322, DFT-SIC 0.9765)। কারণ 16×16 grid + 3×3 NMS-এর effective resolution Rayleigh limit-এর দ্বিগুণ খারাপ (Abl-2 root-cause confirm করে)।
3. **SNR extrapolation ব্যর্থতা।** 30dB→35dB-এ Pd 0.8554→0.8411 (কমছে), Teacher 0.9390→0.9465 (বাড়ছে)। Abl3_no_SE-কে tail-এ test করা হয়নি — missing experiment।
4. **IABC-v2-এর core claim বেশিরভাগ ক্ষেত্রে অপ্রমাণিত।** E4-এর সব level, E5-এর 0–2dB, E6-এর সব level-এ IABR বনাম Abl1 পার্থক্য noise-এর ভেতরে। IABC distortion-এর মাত্র ~24% সরাতে পারে।
5. **High-SNR plateau ~0.87, coarse-grid merge-এর কারণে।** SNR≥15-এ wrong-peak path-এর 82% অন্য true path-এর 1.5 cell-এর মধ্যে (merge failure)।
6. **L বাড়লে high-SNR-এ ধারাবাহিক পতন (E7):** L1 (−1.5pp) থেকে L10 (−8.0pp), Teacher-এর তুলনায়।
7. **End-fire precision অমীমাংসিত (Abl-2, V0, E3):** learned-refine-এও p95_endfire ~164°, physics-level Jacobian singularity।
8. **SE attention কার্যত অকার্যকর (Abl3):** সব condition-এ <0.6pp পার্থক্য, দিক মিশ্র।
9. **batch=1 real-time inference-এ IABR সবচেয়ে দ্রুত না (E9):** FNO NN-only-তে ~2.92×, e2e-তে ~1.61× দ্রুত, যদিও accuracy অনেক খারাপ।
10. **E10 nuisance-এ SNR15-তে 4–7pp পিছিয়ে,** শক্তিশালী interferer-এ gross error বাড়ে (নোট: exact magnitude derived cache-analysis থেকে, §৬ দেখুন)।

### ৪.৩ Improvement Ideas (গুরুত্ব অনুযায়ী)

1. **Heatmap resolution বাড়ানো (সর্বোচ্চ priority — E8/E7/E3/E10 সব দুর্বলতার common root cause):** oversampled beamspace (32×32/64×64), অথবা trunk 16×16 রেখে pixel-shuffle upsample, অথবা multi-anchor per cell। লক্ষ্য: E8-এর 10°-এ Pd ≥0.93।
2. **SNR extrapolation ঠিক করা:** training SNR range বাড়ানো, input normalization, SE-SNR coupling কমানো, এবং **Abl3_no_SE-কে tail bank-এ চালানো** (missing diagnostic)।
3. **IABC-কে সত্যিকারের impairment corrector বানানো:** iterative/deep-unfolded calibration, scale/phase-invariant pair loss, training impairment range/strength বাড়ানো।
4. **SIC-style iterative detection বা DFT-SIC-এর সাথে hybrid** — DFT-SIC-এর "subtract strongest, re-detect" কৌশল resolution (E8) ও masking (E10) দুই সমস্যাতেই জেতে।
5. **Objectness/confidence head + learned L estimation** — spurious peak কমাতে, "never drops" সুবিধা রেখেও false-positive কমাতে।
6. **λ3 (pair loss) সরিয়ে দেওয়া বিবেচনা করা** — দুই independent test (E1, Abl-6) একই সিদ্ধান্তে।
7. **End-fire-aware loss/parameterization** — sin/cos (direction-cosine) offset representation।
8. **Statistical rigor বাড়ানো:** সব ablation-এ ≥3 seed, robustness bank-এ paired design (একই scene, শুধু impairment বদলানো)।
9. **Reporting fix:** E3/E8 figure-এর color bug ঠিক করা, E6-এর missing figure যোগ করা, "bit-exact" claim সংশোধন, প্রতিটা table-এ paper ও strict Pd পাশাপাশি রাখা।

---

## ৫. এখন কী করা উচিত — Research agent-এর ranked next-step plan

RESEARCH agent (`09_next_steps_research.md`) mother synthesis-এর প্রতিটা top weakness-এর জন্য real web research (WebSearch + WebFetch, প্রতিটা citation-এর URL সরাসরি খোলা হয়েছে) করে literature-backed fix খুঁজেছে। কোনো citation বানানো হয়নি।

### দ্রুত মানচিত্র

| Weakness | মূল প্রস্তাবিত fix | মূল citation |
|---|---|---|
| DFT-SIC-কে কখনো হারায়নি | Resolution বাড়ানো + confidence head + deep-unfolded SIC | Huang+23, Klioui'25, Shmuel+23/24, Schieler+24 |
| 10° resolution collapse | Finer heatmap grid | Off-grid DOA framework (Huang+23) |
| SNR extrapolation ব্যর্থতা | Designed training-SNR distribution | Luan & Thompson, ICC 2023 |
| IABC-v2 core claim প্রায় অপ্রমাণিত | Physics-grounded impairment loss | Mateos-Ramos+24, SDOA-Net (Chen+24) |
| High-SNR plateau, L বাড়লে খারাপ | (resolution fix + confidence head) | (same as above) + Schieler+24 |
| SE attention অকার্যকর | SE→ECA বা remove | ECA-Net, Wang+ CVPR 2020 |
| End-fire precision অমীমাংসিত | sin/cos direction-cosine parameterization | Biternion Nets, Beyer+ DAGM 2015 |

### Ranking — Impact vs Effort

| Rank | Fix | Impact | Effort | কারণ |
|---|---|---|---|---|
| 1 | **A — Resolution বাড়ানো** | সবচেয়ে বেশি (E3/E7/E8-এর common root cause) | মাঝারি | একটাই architecture change, তিনটা independent experiment-এ প্রভাব ফেলবে |
| 2 | **F — sin/cos end-fire parameterization** | মাঝারি (দীর্ঘদিনের unsolved issue) | খুবই কম | শুধু loss/target বদল, দ্রুততম ROI |
| 3 | **C — Designed SNR training distribution** | মাঝারি | খুবই কম | data/training recipe change, নতুন param না |
| 4 | **E — SE→ECA/remove** | কম-মাঝারি | খুবই কম | Abl3 framework already আছে |
| 5 | **G — Confidence/objectness head** | মাঝারি | মাঝারি | নতুন head + loss term |
| 6 | **D — Physics-grounded IABC loss** | মাঝারি (thesis core claim বাঁচানো/খারিজ করা) | মাঝারি | নতুন loss design+tuning দরকার |
| 7 | **B — Deep-unfolded SIC-style refine** | সবচেয়ে বেশি সম্ভাব্য (thesis-এর central hypothesis বাঁচাতে পারে) | উচ্চ | নতুন iterative block, বেশি compute/debug সময় |

### মূল citation-গুলো (URL সরাসরি WebFetch দিয়ে verify করা)

- **Off-Grid DOA Estimation via Deep Learning Framework** — Huang, Zhang, Tao et al., *Science China Info. Sci.*, 2023. https://link.springer.com/article/10.1007/s11432-022-3750-5
- **SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation** — Shmuel, Merkofer, Revach, van Sloun, Shlezinger, arXiv:2306.02271 (2023/2024)
- **Circulant ADMM-Net for Fast High-resolution DoA Estimation** — Klioui, arXiv:2502.19076 (2025)
- **Achieving Robust Generalization for Wireless Channel Estimation Neural Networks by Designed Training Data** — Luan, Thompson, IEEE ICC 2023, arXiv:2302.02302
- **Unsupervised Learning for Gain-Phase Impairment Calibration in ISAC Systems** — Mateos-Ramos, Häger, Keskin, Le Magoarou, Wymeersch, arXiv:2410.04176 (2024)
- **SDOA-Net: An Efficient Deep Learning-Based DOA Estimation Network for Imperfect Array** — Chen, Chen, Liu, Chen, Wang, IEEE TIM, 2024, arXiv:2203.10231
- **Grid-free Harmonic Retrieval and Model Order Selection using Deep CNNs** — Schieler, Semper, Faramarzahangari, Döbereiner, Schneider, Thomä, EuCAP 2024, arXiv:2211.04846
- **ECA-Net: Efficient Channel Attention for Deep CNNs** — Wang, Wu, Zhu, Li, Zuo, Hu, CVPR 2020, arXiv:1910.03151
- **Biternion Nets: Continuous Head Pose Regression from Discrete Training Labels** — Beyer, Hermans, Leibe, DAGM 2015

### Thesis-timeline plan

- **ধাপ ১ (১–২ সপ্তাহ, কম effort, দ্রুত ফলাফল):** Fix F (sin/cos end-fire parameterization), Fix C (SNR training distribution + missing Abl3_no_SE tail run), Fix E (SE→ECA variant, Abl3_eca)। তিনটাই independent, parallel-এ চালানো সম্ভব, fail করলেও thesis-এর ক্ষতি নেই।
- **ধাপ ২ (২–৪ সপ্তাহ, মূল architecture change):** Fix A (heatmap resolution/pixel-shuffle upsample), Fix G (confidence/objectness head) সাথে বান্ডেল করা।
- **ধাপ ৩ (২–৩ সপ্তাহ, thesis narrative-নির্ধারক, সময় থাকলে):** Fix D (physics-grounded IABC loss) — "IABC কি সত্যিই impairment-aware, নাকি শুধু efficiency architecture?" প্রশ্নের চূড়ান্ত উত্তর।
- **ধাপ ৪ (future work/stretch):** Fix B (deep-unfolded SIC-style iterative refinement) — সবচেয়ে বড় সম্ভাবনা DFT-SIC-কে সত্যিই হারানোর, কিন্তু কার্যত একটা নতুন architecture paper-level কাজ; deadline না মিললে "Future Work" section-এ concrete proposal হিসেবে রাখা।

**সততা ও সীমাবদ্ধতা:** এই research সম্পূর্ণ literature-নির্দেশিত hypothesis — কোনো fix এখানে বাস্তবায়ন/পরীক্ষা করা হয়নি। বেশ কিছু paper শুধু abstract/landing-page থেকে verify করা হয়েছে (full PDF text সবসময় পড়া যায়নি)। Cost estimate সব আনুমানিক।

---

## ৬. Caveats ও Verifier Correction Summary

`08_verification.md`-এ ৩৩টা numeric claim হাতে যাচাই করা হয়েছে raw `outputs/results/*.json`, `outputs/tables/*.md`, `outputs/RESULTS.md`-এর বিপরীতে। ফলাফল: **২৯টা CORRECT, ৩টা WRONG, ১টা UNVERIFIABLE।** এই ফাইনাল রিপোর্টে চারটাই নিচের মতো সংশোধন/চিহ্নিত করা হয়েছে (আর কোনোটা mother synthesis-এর অসংশোধিত ভাষা ব্যবহার করছে না):

| # | মূল ভুল দাবি (07-এ যেভাবে ছিল) | সংশোধিত/সঠিক মান | এই রিপোর্টে কোথায় ঠিক করা হয়েছে |
|---|---|---|---|
| 1 | "IABR (0.6909, 3-seed mean)" | প্রকৃত 3-seed mean **0.690729 ≈ 0.6907** — 0.6909 শুধু IABR_s0-এর একার মান, "3-seed mean" নয় | §১ Executive Summary, §২.১ |
| 2 | "batch≥32-এ throughput Teacher-এর 36.6×" | 36.6× শুধু **batch=128**-এ সত্যি। batch=32-এ ~7.6×, batch=8-এ ~3.0× | §১, §২.৫, §৩, §৪.১ |
| 3 | Master table E5 row: "Teacher/Student/Abl1-কে হারায় (+2.4 থেকে +2.9pp)" | vs Teacher +2.46pp, vs Abl1 +2.43pp (এই দুটো range-এ), কিন্তু **vs Student +4.22pp** (range-এর বাইরে) — একটা combined range ভুল | §২.২, §৩ |
| 4 | "মাত্র 41.6% params দিয়ে" | **41.5%** (195,024/469,393), মূল ডকুমেন্টের নিজের §1-এর সাথেই সামঞ্জস্যহীন ছিল | §১, §২.৪ |

**একটা UNVERIFIABLE claim চিহ্নিত করা হয়েছে (সংখ্যা রাখা হয়েছে কিন্তু provenance স্পষ্ট করে দেওয়া হয়েছে):** "E10: nuisance শক্তিশালী হলে principal p95 23.55°→52.68°-এ লাফায় (SNR15)"। রাজ `E10_nuisance_path.json`/`tables/E10.md`-এ শুধু pd_principal/pd_principal_strict/pd_nuisance column আছে — কোনো p95 field নেই। এই সংখ্যাটা report 04-এর নিজস্ব cached-prediction re-scoring script (`04_scene_difficulty.md`) থেকে এসেছে — নিজের ভেতরে consistent এবং official Pd-এর সাথে মিলিয়ে যাচাই করা, কিন্তু raw JSON/table থেকে স্বাধীনভাবে cross-check করা যায়নি। §২.৩-এ এটা explicit note সহ রাখা হয়েছে।

### অন্যান্য গুরুত্বপূর্ণ caveat (verifier-এর নিশ্চিত করা)

- **কোনো omitted experiment নেই:** `status.json`-এর ৩৭টা key-ের প্রতিটাই কোনো না কোনো group report (01–06) বা mother synthesis (07)-এ নাম ধরে আলোচিত হয়েছে।
- **কোনো undiscussed figure নেই:** `outputs/figures/`-এর ৯টা PNG-ই কোনো না কোনো report-এ reference/discuss করা হয়েছে। E6, E9, E10, Abl-2, Abl-6-এর জন্য কোনো figure file-ই ship হয়নি (notebook design choice, missing file নয়) — mother synthesis নিজেই এটা সঠিকভাবে flag করেছিল।
- **Sample size ছোট, robustness experiment-এ:** E4/E5/E6/E7/E8 প্রতি condition-এ মাত্র 500 sample (E10-এ 200), binomial SE ~0.009–0.013। বেশিরভাগ ablation single-seed (শুধু main IABR-এর 3 seed আছে, E11)।
- **V0 আর "bit-exact" claim:** এই run-এ diff −8.6e-5, আগের "13+ sig fig exact" দাবির বিপরীতে। ব্যাখ্যাযোগ্য (GPU kernel/oneDNN পার্থক্য) কিন্তু thesis-এ ভাষা সংশোধন দরকার: "reproduces to within 8.6e-5", "bit-exact" নয়।
- **E1_capacity_control নামের সাথে বাস্তবায়নের গরমিল:** design doc-এ "same-capacity, no-adaptivity" module চাওয়া হয়েছিল, বাস্তবায়িত E1 আসলে main-এর হুবহু architecture (শুধু pair loss=0)। Abl1_no_IABC-ই আসলে এই ভূমিকা পালন করে — thesis-এ স্পষ্ট করে লিখতে হবে।
- **PROJECT_STATUS-এর "Student beats Teacher under interference" claim E10-এ reproduce হয়নি** (Δ −0.0316, বিপরীত দিকে) — আগের ফল সম্ভবত sampling noise ছিল।
- **"IABC কী শিখেছে" সরাসরি inspect করা হয়নি** — predicted (δ̂, γ̂) বনাম আসল (ε, g)-এর correlation কখনো measure করা হয়নি; "IABC একটা global re-weighting শিখেছে, impairment corrector নয়"-এই ব্যাখ্যা একটা hypothesis, direct প্রমাণ নয়।
- **Research agent (09)-এর নিজস্ব caveat:** সব fix literature-নির্দেশিত hypothesis, কিছুই বাস্তবায়ন/পরীক্ষা করা হয়নি।

---

*এই রিপোর্টের প্রতিটা সংখ্যা group report 01–09 থেকে হুবহু নেওয়া বা তাদের রিপোর্ট করা derived হিসাব — কোনো নতুন সংখ্যা এখানে আবিষ্কার করা হয়নি। Verifier (08)-এর ধরা ৩টা ভুল সংশোধন করা হয়েছে, ১টা unverifiable claim provenance-সহ চিহ্নিত করা হয়েছে।*
