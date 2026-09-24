# 07 — Mother Synthesis: IABR-Net GPU Run, সব experiment একসাথে

> **Role:** MOTHER agent। ছয়জন analyst-এর ছয়টা group report (01–06) সম্পূর্ণ পড়ে, তার উপর ভিত্তি করে এই সমন্বিত বিশ্লেষণ লেখা হয়েছে। এখানে কোনো নতুন সংখ্যা হিসাব করা হয়নি — সবকিছু analyst report ছয়টা (`GPU_RUN_ANALYSIS/01_*.md` … `06_*.md`) থেকে নেওয়া, যেগুলো নিজেরা `outputs/results/*.json`, `outputs/tables/*.md` আর `outputs/figures/*.png` থেকে numbers তুলেছিল। কোনো সংখ্যা আবিষ্কার করা হয়নি।
> **Source reports:** `01_sanity_verification.md`, `02_training_indist_seeds.md`, `03_impairment_robustness.md`, `04_scene_difficulty.md`, `05_ablations_controls.md`, `06_efficiency.md` (সবগুলোর path: `D:\ai_ml_project\IABR_Net_TestSuite\GPU_RUN_ANALYSIS\`)।

---

## ১. সংক্ষেপে (Executive Summary)

**IABR-Net কি সফল?** সংক্ষেপে: **আংশিকভাবে সফল, কিন্তু thesis-এর মূল দাবি (impairment-aware correction একটা 0-parameter classical baseline-কে হারাবে) data দিয়ে সমর্থিত নয়।**

- **Efficiency-তে নিঃসন্দেহে সফল।** IABR-Net **195,024 params** (Teacher 469,393-এর **41.5%**, Student 314,513-এর 62.0%, FNO 334,321-এর 58.3%) আর **97.17 MFLOPs** (Teacher 15.203 GFLOPs-এর **~156× কম**, Student 10.157 GFLOPs-এর ~104.5× কম, FNO 0.172 GFLOPs-এর ~1.77× কম) দিয়ে বানানো (E0b, E9)। GPU memory 66 MB (বাকি সব ~2.6–2.9× বেশি), weights মাত্র 1.01 MB। batch≥32-এ throughput Teacher-এর 36.6×।
- **In-distribution accuracy-তে "মাঝারি" — Teacher আর Student-এর কাছাকাছি, কিন্তু DFT-SIC-এর নিচে সবসময়।** E3 frozen bank (L=3, SNR −10..25 dB)-এ mean paper-style Pd: **DFT-SIC 0.7326 > Teacher 0.7103 > Student 0.6956 > IABR (0.6909, 3-seed mean) > FNO 0.6235**। **Strict Pd-তে** (dropped sample-কে miss ধরলে, যেটা IABR-এর কখনো sample drop না-করার সুবিধা তুলে ধরে) IABR (0.6907) আসলে **Teacher-কে হারায়** (0.6727) এবং Student-কেও (0.6469), তবে DFT-SIC-এর (0.7326) নিচেই থাকে। 15 dB-এ IABR 0.8662 (paper), Teacher 0.8994, DFT-SIC 0.9042 — clean condition-এই IABR প্রায় 3.3–3.8 pp পিছিয়ে।
- **যে দাবিটা ব্যর্থ হয়েছে: "IABC-v2 impairment-aware correction DFT-SIC-কে হারানোর মূল অস্ত্র।"** E4 (phase ≤5°), E5 (gain 0.5–2 dB), E6 (phase 10–15° OOD) — এই তিনটার প্রায় সব condition-এ IABR বনাম Abl1_no_IABC (IABC ছাড়া) পার্থক্য statistically noise-এর মধ্যে (paired bootstrap CI শূন্য বাদ দেয় না)। শুধু **একটা condition-এ** (SNR 15 dB, gain error 4 dB OOD) IABC-এর প্রকৃত, পুনরাবৃত্ত (তিনটা IABC variant-এ) সুবিধা দেখা গেছে (+2.1 থেকে +2.6 pp বনাম Abl1)। কিন্তু সেখানেও **0-parameter DFT-SIC** (0.8387) IABR-এর (0.8033) চেয়ে এগিয়ে। প্রতিটা impairment condition-এ (E4/E5/E6, দুটো SNR, সব level) DFT-SIC IABR-কে হারায়।
- **যেখানে সবচেয়ে বড় architectural দুর্বলতা ধরা পড়েছে: angular resolution।** E8-এ দুটো path 10° কাছাকাছি হলে IABR ধসে পড়ে (0.6145, Teacher 0.9322, DFT-SIC 0.9765) — FNO-এর স্তরে নেমে যায়। কারণ (Abl-2 দিয়ে প্রমাণিত): coarse heatmap মাত্র 16×16, আর 3×3 NMS দুটো কাছের path-কে একই cell-এ merge করে ফেলে। Coarse decode একা Pd মাত্র 0.14 দেয় — **প্রায় পুরো accuracy আসে learned crop-refinement head থেকে** (0.14 → 0.69, +55 pp)।
- **SNR extrapolation-এ ব্যর্থতা:** training range (−15..24 dB)-এর বাইরে, high SNR (30, 35 dB)-এ IABR-এর Pd **কমে যায়** (0.8554 → 0.8411), যেখানে Teacher বাড়তে থাকে (0.9390 → 0.9465)। High-SNR-এ IABR-এর plateau (~0.87) মূলত দুটো কারণে: কাছাকাছি path merge (82% wrong-peak path একটা true path-এর 1.5 cell-এর মধ্যে) আর end-fire sub-cell precision (25 dB-এ miss-এর 85% end-fire-এ)।
- **Training pipeline নিজে সম্পূর্ণ সুস্থ ও reproducible:** 3 seed একই জায়গায় converge করে (E3 std মাত্র 0.0010, 95% CI ±0.0024), কোনো divergence নেই।

**এক লাইনে:** IABR-Net একটা **parameter/FLOP-efficient, robust-to-drop (never drops a sample), moderate-accuracy** architecture — কিন্তু "impairment-aware" নামের মূল novelty claim প্রায় সব robustness test-এ অপ্রমাণিত থেকে গেছে, এবং এর angular resolution ও high-SNR extrapolation-এর সীমাবদ্ধতা এর accuracy-কে DFT-SIC-এর মতো একটা 0-parameter classical method-এর নিচে রেখে দিয়েছে।

---

## ২. Master Comparison Table — সব experiment এক নজরে

| # | Experiment | কী test করে | IABR-Net result | Best baseline (model) | Verdict |
|---|---|---|---|---|---|
| E0a | Phase-0 unit tests | Physics/simulator সঠিকতা | 11/11 pass, error ~1e-15 | — (কোনো model-test না) | ✅ Success |
| BANKS_build | 56 fixed eval bank | Data integrity/reproducibility | 56/56 hash match | — | ✅ Success (caveat: এই GPU-তে regenerate হয়নি) |
| E0b | Model instantiation | Real params/FLOPs measurement | 195,024 params, 97.17M FLOPs | Teacher 469,393 params / 15.20 GFLOPs | ✅ Success (estimate থেকে 1–2% পার্থক্য, ব্যাখ্যাযোগ্য) |
| V0 | Teacher reproduces registry | Evaluator/weights/bank সঠিকতা | Teacher mean Pd 0.71026 (registry 0.71035, diff −8.6e-5) | — | ✅ Success (আর bit-exact নয়, কিন্তু tolerance-এর ২৩ গুণ ভেতরে) |
| V1 | New generator fidelity | নতুন simulator বনাম frozen bank | diff < 3σ সব SNR-এ (0.42σ, 0.32σ, 0.11σ) | — | ✅ Success (coverage সীমিত: শুধু L=3, clean) |
| TRAIN_s0/s1/s2 | Training convergence, 3 seed | Loss ও seed stability | final loss 0.4124–0.4166, খুব stable | — | ✅ Success (2টা caveat: heat-loss plateau, s2 resume spike) |
| **E3** | In-dist SNR sweep (L=3, −10..25 dB), headline | Paper Pd mean **0.6909** (s0), strict **0.6907** | **DFT-SIC 0.7326** (paper), Teacher 0.6727 (strict) | **Mixed**: strict-এ Teacher-কে হারায় (+0.018), paper-এ Teacher-এর 97.2% পায়, কিন্তু DFT-SIC-এর কাছে সব metric-এ হারে (−0.042) |
| E11 | Multi-seed variance | Seed-to-seed reproducibility | E3 std 0.0010, phase_d5 std 0.0010, gain_g2 std 0.0025 | — | ✅ Success — খুব stable, ordering robust |
| SNR_tails | OOD SNR (−25,−20,30,35 dB) | Extrapolation | 30/35 dB: 0.8554→0.8411 (**কমছে**) | Teacher 0.9390→0.9465 (**বাড়ছে**) | ❌ Failure (high tail); low tail সবাই chance level, inconclusive |
| E4 | Phase error ≤5° (in-dist) | IABC benefit, low impairment | SNR15, 5°: 0.8510 (Abl1: 0.8607, **IABR খারাপ**) | Teacher 0.8859, DFT-SIC 0.9007 | ❌/Inconclusive — IABC-এর কোনো প্রমাণযোগ্য লাভ নেই, absolute Pd Teacher/DFT-SIC-এর নিচে |
| E5 | Gain error, incl. OOD 4dB | IABC benefit, বিশেষত OOD gain | SNR15, 4dB(OOD): **0.8033** (retention 92.3%, সর্বোচ্চ) | Teacher 0.7787, DFT-SIC 0.8387 | ⚠️ Partial success — Teacher/Student/Abl1-কে হারায় (+2.4 থেকে +2.9pp), কিন্তু DFT-SIC-এর নিচে; 0–2dB-এ কোনো লাভ নেই |
| E6 | OOD phase 10–15° | Distribution-shift robustness | SNR15, 15°: 0.8043 (Abl1: 0.8017, তফাত ns) | Teacher 0.8081, DFT-SIC 0.8427 | ❌ Failure (IABC claim) — graceful degradation তবে DFT-SIC-এর ~3.8pp নিচে |
| E7 | Path count L=1..10 | Multipath/pairing capacity | SNR15 paper: L1 0.928→L10 0.6694; strict L10 0.6694 vs Teacher 0.5426 | Teacher (paper, উচ্চ L), DFT-SIC (strict, সব L) | Mixed — strict/কম-SNR-এ ভালো, paper-style high-SNR-এ L বাড়ার সাথে ফাঁক বাড়ে (L10: −8pp) |
| E8 | Angular separation 30°→1° | Resolution limit | **10°-এ ধস: 0.6145** | Teacher 0.9322, DFT-SIC 0.9765 | ❌ **Clear failure** — effective resolution ~2 DFT cell, Rayleigh limit-এর দ্বিগুণ খারাপ |
| E10 | Nuisance path (3+1, power −20..0dB) | Interference robustness | Principal Pd দোলা ≤0.0283 (স্থির), SNR15@0dB: 0.8150 | Teacher 0.8795, DFT-SIC 0.8817 (SNR15); DFT-SIC সবচেয়ে স্থির (strict) | Mixed — স্থিতিশীল/drop-free, কিন্তু SNR15-এ 4–7pp পিছিয়ে |
| Controls: E1 (pair-loss=0) | λ3 (pair loss) দরকার কিনা | সব condition <0.3pp পার্থক্য বনাম main | — | Neutral — pair loss কার্যত অকার্যকর |
| Controls: E2 (dense frontend, 2.3× params) | Capacity বনাম architecture | E3: 0.5704 (main 0.6909-এর চেয়ে **11.8pp খারাপ**, যদিও 2.34× বেশি params) | — | ✅ Strong confirmatory failure — architecture/inductive-bias-ই আসল কারণ, capacity নয় |
| Controls: Abl1 (no IABC) | IABC-এর pure ablation | E3: 0.6871 (main 0.6909, Δ −0.0038, ns) | — | Neutral bulk; শুধু gain_g4-এ IABR +2.13pp এগিয়ে |
| Controls: Abl3 (no SE) | SE attention দরকার কিনা | সব condition <0.6pp পার্থক্য, দিক মিশ্র | — | ❌ Failure — SE attention কার্যত অকার্যকর |
| Abl-2 | Decode ablation (coarse/parabolic/learned) | Accuracy কোথা থেকে আসে | coarse 0.1396 → learned-refine **0.6909** (+55.1pp) | Teacher 0.7158, DFT-SIC 0.7326 (pooled) | ✅ **Headline success** — refinement head-ই মূল ইঞ্জিন; কিন্তু end-fire p95 এখনো ~164° (unsolved) |
| Abl-6 | Loss-weight sweep (12 runs) | λ3×λ4 sensitivity | Spread মাত্র 0.46–1.53pp সব 12 combo-তে | — | ✅ Robust — কিন্তু main-এর নিজস্ব weight কোনো metric-এই সেরা না |
| **E9** | Efficiency (params/FLOPs/latency/memory) | Compute efficiency | 195,024 params, 0.097 GFLOPs, 66MB GPU mem, throughput b128: 32,993/s | Teacher (params/FLOPs), FNO (batch-1 latency: 1.25ms vs IABR 3.65ms) | Mixed-positive — params/FLOPs/memory/high-batch-throughput-এ জয়ী, কিন্তু batch=1 real-time-এ FNO ~2.9× দ্রুত |

---

## ৩. Cross-experiment patterns — কোথায় ফলাফল একে অপরকে শক্তিশালী করে বা বিরোধ করে

### ৩.১ পরস্পর-শক্তিশালীকারী প্যাটার্ন (reinforcing)

1. **"IABC-v2 কার্যত কাজ করে না" — চারটা independent test একই সিদ্ধান্তে পৌঁছেছে:**
   - E4/E5(0–2dB)/E6: IABR বনাম Abl1-এর paired CI প্রায় সবসময় শূন্য বাদ দেয় না।
   - Controls §E1: pair loss (λ3) সরালে সব condition-এ <0.3pp পার্থক্য।
   - Abl-6: 12-run loss-weight sweep-এও কোনো trend নেই (spread <1.5pp)।
   - Training log-এর `pair` metric নিজেই দেখায় IABC distortion-এর মাত্র ~24% সরাতে পারে (0.00815 → 0.00616), আর E1 (pair supervision ছাড়া, pair metric 7.7× বেশি খারাপ) তবুও same Pd দেয় — অর্থাৎ IABC "correction" আসলে detection-এর জন্য গুরুত্বপূর্ণ না।
   - **একমাত্র ব্যতিক্রম:** SNR15, gain 4dB OOD-এ IABC তিনটা variant-এ (IABR, E1, Abl3) পুনরাবৃত্তভাবে +2.1–2.6pp দেয়। এটা physics-consistent: শুধু এই condition-এ impairment distortion (−7.7dB) noise (−15dB)-এর চেয়ে জোরালো (§0.5, report 03), তাই এখানেই gradient signal ছিল।

2. **"IABR কখনো sample drop করে না" — একটা single design choice-এর প্রভাব বারবার দেখা যায়:** E3, E4–E6, E7, E8, E10, Abl-2 — সব জায়গায় IABR-এর paper Pd = strict Pd, আর কঠিন scene-এ (উচ্চ L, ছোট separation, শক্তিশালী nuisance) এই কারণেই strict metric-এ Teacher/Student-এর চেয়ে অনেক এগিয়ে দেখায় (Teacher-এর drop rate L10-এ 42.5%, E8 10°-এ 4.8%)। এটা একটা বাস্তব robustness সুবিধা (deployment-এ সবসময় একটা answer দেওয়া), কিন্তু এটা impairment-robustness নয়।

3. **"Coarse-grid merge failure" একটা mechanism, তিনটা experiment-এ একই আকারে ধরা পড়ে:**
   - E8: 10° separation-এ ধস (0.6145)।
   - E3-এর extra diagnosis: SNR≥15-এ wrong-peak path-এর 82% অন্য true path-এর 1.5 cell-এর মধ্যে।
   - E10: nuisance শক্তিশালী হলে principal p95 23.55°→52.68°-এ লাফায় (interferer-এর পাশে থাকা দুর্বল path হারিয়ে যায়)।
   - সবগুলোর মূল কারণ Abl-2 দিয়ে নিশ্চিত: 16×16 coarse heatmap + 3×3 NMS-এর effective resolution ~2 DFT cell।

4. **"E2 dense-frontend failure" আর "E9 efficiency success" — একই সিদ্ধান্তের দুই দিক:** E2 প্রমাণ করে 2.34× বেশি params দিয়েও physics-informed structure ছাড়া front-end 10–15pp খারাপ করে — মানে architecture/inductive bias আসল কারণ। এটা E9-এর "IABR-Net-এর accuracy-per-parameter (0.3543) ও accuracy-per-FLOP (7.124) সব NN baseline-এর সেরা" ফলাফলের সাথে সরাসরি সামঞ্জস্যপূর্ণ — দুটোই বলছে efficiency শুধু "ছোট মডেল" থেকে আসেনি, গঠন থেকে এসেছে।

5. **High-SNR plateau (E3) ↔ SNR extrapolation failure (SNR_tails) ↔ E11-এর 25dB variance — একই সীমান্তে তিনটা signal:** E3-তে 20→25dB-এ IABR প্রায় flat (0.8733→0.8745), E11-এ ঠিক 25dB-তেই per-seed spread সবচেয়ে বেশি (std 0.0137, বাকি SNR-এ ~0.001–0.003), আর SNR_tails-এ 25dB-এর পরে (30, 35dB) Pd সরাসরি কমতে শুরু করে। তিনটাই training-range-এর কিনারায় model-এর আচরণ অস্থির হয়ে ওঠার একই সংকেত।

### ৩.২ পরস্পর-বিরোধী বা বিভ্রান্তিকর প্যাটার্ন (contradicting / needs caveat)

1. **E3-এর "strict Pd-তে Teacher-কে হারায়" বনাম E7/E8/E10-এর high-SNR/hard-scene ফলাফল একে অপরকে qualify করে, বিরোধ করে না — কিন্তু headline-এ বিভ্রান্তি হতে পারে।** E3 (L=3 fixed, easy scene)-এ IABR strict-এ Teacher-কে হারায় (+0.018), কিন্তু L বাড়লে (E7) বা path কাছাকাছি এলে (E8) বা nuisance এলে (E10), high-SNR-এ IABR-এর absolute accuracy Teacher-এর চেয়ে ক্রমশ খারাপ হতে থাকে। তাই "strict Pd-তে Teacher-কে হারায়" claim-টা **শুধু L=3, easy-scene condition-এ সত্য**, general claim নয়।

2. **E5-এর "IABC gain-এ সাহায্য করে" বনাম E4/E6-এর "phase-এ সাহায্য করে না" — এটা contradiction নয়, physics দিয়ে ব্যাখ্যাযোগ্য (report 03, §0.5), কিন্তু thesis narrative-এ এটা স্পষ্টভাবে বলতে হবে:** gain error-এর distortion phase error-এর চেয়ে বেশি জোরালো (4dB gain = −7.7dB distortion, বনাম 15° phase = −13.4dB), তাই শুধু gain-এই noise-এর উপরে ওঠার মতো signal পাওয়া গেছে। **এই asymmetry design-এর ব্যর্থতা নয়, test-condition-এর সীমার ফল** — কিন্তু thesis-এ "IABC works for gain, not phase" না লিখে "IABC works only when distortion exceeds noise (which in this test suite happened only for the OOD-gain condition)" লেখা বেশি সৎ হবে।

3. **PROJECT_STATUS-এর আগের "Student beats Teacher under nuisance interference" ফল E10-এ reproduce হয়নি (Δ −0.0316, উল্টো দিকে)।** এটা IABR-Net-এর কোনো দুর্বলতা নয়, কিন্তু thesis-এ আগের এই দাবিটা ব্যবহার করা থাকলে সংশোধন প্রয়োজন — sampling noise-এর একটা সতর্ক উদাহরণ (±0.02 SE একটা 200-scene bank-এ)।

4. **V0-এর "আর bit-exact নয়" (diff −8.6e-5) বনাম আগের CPU/GPU run-এর "13+ significant figures exact" দাবি — সরাসরি বিরোধ, thesis-language সংশোধন দরকার।** Sanity group নিজেই এটা flag করেছে: এখন থেকে "reproduces within 8.6e-5" লিখতে হবে, "bit-exact" নয়।

5. **E3-এর figure legend bug (Teacher ও Abl3_no_SE একই নীল রঙে) দুইটা আলাদা analyst group (01 আর 02) স্বাধীনভাবে ধরেছে** — এটা প্যাটার্ন না হলেও দুই দিক থেকে confirm হওয়া একটা নির্ভরযোগ্য বাগ রিপোর্ট।

---

## ৪. Success / Failure / Improvement — গুরুত্ব অনুযায়ী সাজানো

### ৪.১ Success (ranked)

1. **Efficiency claim পুরোপুরি নিশ্চিত ও measurement-ভিত্তিক (E0b, E9):** 195,024 params (est. 193,104-এর 1% কাছে, পার্থক্য পুরোপুরি BatchNorm bookkeeping দিয়ে ব্যাখ্যাযোগ্য), 97,169,941 FLOPs (est. ~160×-এর কাছাকাছি ~156×)। GPU memory 66MB (Teacher-এর ~35%)। batch-128 throughput Teacher-এর 36.6×।
2. **Learned crop-refinement head-ই IABR-Net-এর accuracy-র প্রায় পুরো উৎস (Abl-2):** coarse-only 0.1396 → learned-refine 0.6909 (+55.1pp), Teacher-এর 96.5% আর DFT-SIC-এর 94.3% পৌঁছায় মাত্র 41.6% params দিয়ে।
3. **Physics-informed architecture > raw capacity, সরাসরি প্রমাণিত (E2 vs E1/Abl1):** 2.34× বেশি params দিয়েও dense front-end 10–15pp খারাপ করে।
4. **Never drops a sample:** সব robustness experiment-এ n_dropped=0, তাই strict metric-এ কঠিন scene-এ Teacher/Student-কে বড় ব্যবধানে হারায় (E7 L10: 0.3839 vs 0.0676; E10 SNR0/−20dB: 0.6425 vs 0.3717)।
5. **Training সম্পূর্ণ reproducible ও stable:** 3-seed E3 std 0.0010, gain_g2 std 0.0025, কোনো divergence নেই (E11)।
6. **Sanity/verification chain সম্পূর্ণ pass:** simulator, evaluator, weights, frozen bank — সব bug-free প্রমাণিত (E0a, V0, V1, BANKS_build), তাই বাকি সব দুর্বলতা model-এর নিজের, infrastructure-এর নয়।
7. **একমাত্র OOD condition-এ IABC-এর প্রকৃত সুবিধা:** SNR15, gain 4dB — retention 92.3% (সর্বোচ্চ সব model-এ), Teacher/Student/Abl1-কে হারায় (+2.1–2.9pp, তিনটা IABC variant-এ পুনরাবৃত্ত)।

### ৪.২ Failure / Weakness (ranked, প্রতিটার সাথে নির্দিষ্ট প্রমাণ)

1. **[সবচেয়ে গুরুত্বপূর্ণ] DFT-SIC (0 params)-কে কোনো test-এ হারাতে পারেনি।** E3 (সব 8 SNR, দুই metric), E4 (সব phase level), E5 (সব gain level, এমনকি IABC-এর সেরা condition 4dB OOD-তেও, 0.8033 vs 0.8387), E6 (সব phase level), E7 (strict metric, সব L), E8 (সব separation), E10 (SNR15, সব nuisance level)। Research report-এর নিজের central hypothesis ("IABC-v2 beats this critical 0-param baseline") **ব্যর্থ**।
2. **Angular resolution — সবচেয়ে বড় architectural gap।** E8: 10° separation-এ 0.6145 (Teacher 0.9322, DFT-SIC 0.9765), FNO-এর স্তরে নেমে যায়। কারণ 16×16 grid + 3×3 NMS-এর effective resolution Rayleigh limit-এর দ্বিগুণ খারাপ (Abl-2 root-cause confirm করে)।
3. **SNR extrapolation ব্যর্থতা।** SNR_tails: 30dB→35dB-এ Pd 0.8554→0.8411 (কমছে), Teacher 0.9390→0.9465 (বাড়ছে)। কারণ এখনো hypothesis-স্তরে (SE-SNR coupling বা offset calibration), Abl3_no_SE-কে tail-এ test করা হয়নি — এটা একটা missing experiment।
4. **IABC-v2-এর core claim বেশিরভাগ ক্ষেত্রে অপ্রমাণিত।** E4-এর সব level, E5-এর 0–2dB, E6-এর সব level-এ IABR বনাম Abl1 পার্থক্য noise-এর ভেতরে (paired CI শূন্য বাদ দেয় না)। IABC distortion-এর মাত্র ~24% সরাতে পারে।
5. **High-SNR plateau ~0.87, coarse-grid merge-এর কারণে।** E3: 20→25dB-এ প্রায় flat (Teacher 0.9376-এর বিপরীতে)। SNR≥15-এ wrong-peak path-এর 82% অন্য true path-এর 1.5 cell-এর মধ্যে (merge failure)।
6. **L বাড়লে high-SNR-এ ধারাবাহিক পতন (E7):** SNR15 paper-style-এ L1 (−1.5pp) থেকে L10 (−8.0pp), Teacher-এর তুলনায়।
7. **End-fire precision অমীমাংসিত (Abl-2, V0, E3):** learned-refine-এও p95_endfire ~164° (physics-level Jacobian singularity, refinement head দিয়ে সমাধান হয়নি)।
8. **SE attention কার্যত অকার্যকর (Abl3):** সব condition-এ <0.6pp পার্থক্য, দিক মিশ্র। 2,931 params (1.5%) বাস্তবে কিছু যোগ করছে না।
9. **batch=1 real-time inference-এ IABR সবচেয়ে দ্রুত না (E9):** FNO NN-only-তে ~2.92× এবং e2e-তে ~1.61× দ্রুত, যদিও FNO-এর accuracy অনেক খারাপ (Pd 0.6235 vs 0.6909)।
10. **E10 nuisance-এ SNR15-তে 4–7pp পিছিয়ে**, আর শক্তিশালী interferer-এ gross error দ্বিগুণ হয় (p95 23.55°→52.68°)।

### ৪.৩ Improvement Ideas (ranked)

1. **Heatmap resolution বাড়ানো (সর্বোচ্চ priority — E8/E7/E3/E10 সব দুর্বলতার common root cause):** oversampled beamspace (32×32/64×64), অথবা trunk 16×16 রেখে pixel-shuffle upsample, অথবা multi-anchor per cell। লক্ষ্য: E8-এর 10°-এ Pd ≥0.93।
2. **SNR extrapolation ঠিক করা:** training SNR range বাড়ানো, input normalization, SE-SNR coupling কমানো, এবং সবচেয়ে জরুরি — **Abl3_no_SE-কে tail bank-এ চালানো** (এখন এই diagnostic experiment missing)।
3. **IABC-কে সত্যিকারের impairment corrector বানানো:** iterative/deep-unfolded calibration, scale/phase-invariant pair loss, অথবা training impairment range/strength বাড়ানো যাতে gradient signal noise-এর উপরে ওঠে (বর্তমান training গড় DSR −18.6dB, বেশিরভাগ sample-এ noise-এর নিচে)।
4. **SIC-style iterative detection বা DFT-SIC-এর সাথে hybrid** — যেহেতু DFT-SIC-এর "subtract strongest, re-detect" কৌশলই resolution (E8) ও masking (E10) দুই সমস্যাতেই জেতে।
5. **Objectness/confidence head + learned L estimation** — spurious peak (gross error-এর প্রধান উৎস) কমাতে এবং "never drops" সুবিধা বজায় রেখেও false-positive কমাতে।
6. **λ3 (pair loss) সরিয়ে দেওয়া বিবেচনা করা** — দুই independent test (E1, Abl-6) একই সিদ্ধান্তে, architecture সরল হবে কোনো ক্ষতি ছাড়াই।
7. **End-fire-aware loss/parameterization** — Jacobian-weighted offset loss অথবা wrap-aware (sin/cos) representation।
8. **Statistical rigor বাড়ানো:** সব ablation-এ ≥3 seed (এখন বেশিরভাগ single-seed), robustness bank-এ paired design (একই scene, শুধু impairment বদলানো) যাতে sampling noise কমে।
9. **Reporting fix:** E3/E8 figure-এর color bug ঠিক করা, E6-এর missing figure যোগ করা, "bit-exact" claim সংশোধন করে "reproduces within 8.6e-5" লেখা, প্রতিটা table-এ paper ও strict Pd পাশাপাশি রাখা।

---

## ৫. Honest Caveats (sample sizes, seed variance, সন্দেহজনক বা যাচাই-অযোগ্য বিষয়)

- **Sample sizes ছোট, robustness experiment-এ:** E4/E5/E6/E7/E8 প্রতি condition-এ মাত্র 500 sample (E10-এ 200), তাই binomial SE ~0.009–0.013। E4/E5-এর মতো জায়গায় ±2pp ওঠানামা প্রায়ই pure sampling noise (DFT-SIC-এর ক্ষেত্রে দেখানো হয়েছে, যার কোনো training নেই তবু curve ওঠানামা করে) — কারণ প্রতিটা impairment level আলাদা seed/scene-এর bank থেকে আসে, paired নয়।
- **Seed variance ছোট কিন্তু n=3 নিজেই ছোট:** E11-এ IABR-এর তিন seed std 0.0010–0.0025, কিন্তু n=3 দিয়ে এই std-এর নিজস্ব 95% CI বড় (χ² হিসাবে σ প্রকৃত মান 6× পর্যন্ত হতে পারে)। বেশিরভাগ ablation (E1, E2, Abl1, Abl3, সব E4–E10-এর IABR-family রেফারেন্স) **single seed**।
- **Abl-6 sweep মাত্র 5000 step, subset eval (SNR-প্রতি 200 sample) আর single-seed** — 1–1.5pp spread real trade-off নাকি noise তা নিশ্চিত না।
- **V0 আর "bit-exact" দাবি:** এই run-এ diff −8.6e-5, আগের run-গুলোর "13+ significant figures exact" দাবির বিপরীতে। ব্যাখ্যাযোগ্য (GPU kernel/oneDNN পার্থক্য) কিন্তু thesis-এ ভাষা সংশোধন দরকার।
- **BANKS_build এই GPU machine-এ regenerate হয়নি** (`build_min` ≈ 0) — আগে থেকে ship করা 56টা bank ব্যবহার হয়েছে, hash মিলেছে কিন্তু এই environment-এ generator নিজে deterministic কিনা তা প্রমাণিত হয়নি।
- **E1_capacity_control নামের সাথে বাস্তবায়নের গরমিল:** design doc-এ "same-capacity, no-adaptivity" module চাওয়া হয়েছিল, কিন্তু বাস্তবায়িত E1 আসলে main-এর হুবহু architecture (শুধু pair loss=0) — Abl1_no_IABC-ই আসলে এই ভূমিকা পালন করে। এটা একটা discrepancy, যা thesis-এ স্পষ্ট করে লিখতে হবে।
- **PROJECT_STATUS-এর "Student beats Teacher under interference" দাবি E10-এ reproduce হয়নি** (Δ −0.0316, বিপরীত দিকে) — আগের ফলটা সম্ভবত sampling noise ছিল।
- **s2 training resume spike** (step 2000→2200-এ loss 0.4464→0.5841): কারণ শুধু hypothesis (optimizer state restore বা data iterator re-seed), কেউ checkpoint-এর ভেতর inspect করেনি। Final metric-এ ক্ষতি হয়নি (s2-ই সেরা seed) কিন্তু root cause অনিশ্চিত।
- **E0b-এর FLOP breakdown অসম্পূর্ণ:** total সংখ্যা (97,169,941) নির্ভরযোগ্য, কিন্তু profiler-এর per-op breakdown-এ শুধু Conv2D আছে; বাকি ~1.83M FLOP কোন op থেকে তা স্পষ্ট নয়। FNO-এর FFT FLOPs analytic estimate, measured নয়। DFT-SIC-এর কোনো FLOPs/GPU-memory measure করা হয়নি (শুধু latency)।
- **"IABC কী শিখেছে" সরাসরি inspect করা হয়নি** — predicted (δ̂, γ̂) বনাম আসল (ε, g)-এর correlation কখনো measure করা হয়নি; §4 (report 03)-এর ব্যাখ্যা (IABC একটা global re-weighting শিখেছে, impairment corrector নয়) একটা hypothesis, direct প্রমাণ নয়।
- **Figure gaps:** E6, E9, E10, Abl-2/6-এর জন্য কোনো dedicated figure নেই (code omission/design choice, missing file নয়)। E3/E8-এর existing figure-এ color-legend bug আছে (দুই independent analyst confirm করেছে)।
- **প্রতিটা analyst নিজের "derived"/"my analysis" সংখ্যা (paired bootstrap, cache re-scoring, physics DSR হিসাব) নিজেদের script দিয়ে তৈরি করেছে, যেগুলো official JSON Pd-এর সাথে cross-verify করা হয়েছে (হুবহু মিলেছে) কিন্তু এগুলো notebook-এর নিজস্ব output নয় — thesis-এ ব্যবহারের আগে independently re-verify করা ভালো।

---

*এই synthesis-এর সব সংখ্যা ছয়টা group report (01–06) থেকে হুবহু নেওয়া। কোনো নতুন সংখ্যা এখানে হিসাব করা হয়নি। প্রতিটা claim-এর পেছনে source report ও experiment নাম বন্ধনীতে দেওয়া আছে।*
