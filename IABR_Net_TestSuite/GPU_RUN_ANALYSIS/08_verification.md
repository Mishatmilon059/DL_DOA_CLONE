# 08 — Verification: Mother Synthesis ও ছয়টা Group Report-এর Numeric Claim Audit

> **Role:** Adversarial VERIFIER। `07_mother_synthesis.md` আর ছয়টা group report (`01`–`06`) থেকে সংখ্যা তুলে `outputs/results/*.json`, `outputs/tables/*.md`, `outputs/RESULTS.md`-এর বিপরীতে হাতে (exact value বা recomputed arithmetic দিয়ে) cross-check করা হয়েছে। কোনো claim অনুমানে PASS করা হয়নি।

---

## ১. Table: claim / file / correct value / status

| # | Claim (mother synthesis-এ যেভাবে লেখা) | Source file (claim) | Raw source checked | Correct value | Status |
|---|---|---|---|---|---|
| 1 | Params: IABR 195,024; Teacher 469,393-এর 41.5%; Student 314,513-এর 62.0%; FNO 334,321-এর 58.3% | 07 §1 | E0b_model_instantiation.json, E9_latency_flops_memory.json | 195024/469393=41.55%, /314513=62.02%, /334321=58.34% — সব রাউন্ডে মেলে | CORRECT |
| 2 | FLOPs: 97.17 MFLOPs; Teacher-এর ~156×, Student-এর ~104.5×, FNO-এর ~1.77× কম | 07 §1 | E9_latency_flops_memory.json (flops_total) | 15202591744/97169941=156.4×; 10156842496/97169941=104.5×; 172048960/97169941=1.77× | CORRECT |
| 3 | GPU memory 66 MB, বাকি সব ~2.6–2.9× বেশি | 07 §1 | E9 json `gpu_peak_mb_batch32` | IABR 65.74MB; Teacher/IABR=2.83×, Student/IABR=2.58×, FNO/IABR=2.85× | CORRECT |
| 4 | weights মাত্র 1.01 MB | 07 §1 | E9 json `weights_file_mb` | 1.0084 MB | CORRECT |
| 5 | **batch≥32-এ throughput Teacher-এর 36.6×** | 07 §1 (Executive summary) | E9 json `latency_nn` | 36.6× সত্যি শুধু **batch=128**-এ। batch=32-এ IABR/Teacher throughput ratio = 8480/1116 = **7.6×**, batch=8-এ 2287.7/773.3=**2.96×**। Source report 06 নিজেই সঠিকভাবে বলে "batch-128-এ...৩৬.৬×", mother synthesis সেটাকে ভুলভাবে "batch≥32" বলে সাধারণীকরণ করেছে | **WRONG** (overgeneralized) |
| 6 | E3 mean Pd (paper): DFT-SIC 0.7326 > Teacher 0.7103 > Student 0.6956 > **IABR (0.6909, 3-seed mean)** > FNO 0.6235 | 07 §1 (Executive summary) | E3_in_distribution_snr_sweep.json (IABR_s0/s1/s2 mean_over_snr.pd_paper) | s0=0.6909375, s1=0.6896875, s2=0.6915625 → প্রকৃত 3-seed mean = **0.690729 ≈ 0.6907**, 0.6909 নয় (0.6909 আসলে শুধু s0-এর একার মান)। Group report 02 নিজেই সঠিকভাবে 0.6907 ব্যবহার করে (§"paired analysis", E11 table) | **WRONG** (label mismatch: 0.6909 ≠ 3-seed mean) |
| 7 | Strict Pd: IABR (0.6907) Teacher-কে (0.6727) ও Student-কে (0.6469) হারায় | 07 §1 | E3 json | 3-seed strict mean = same 0.690729 (IABR paper=strict সবসময়, n_dropped=0)। Teacher strict mean=0.672688 ✓; Student strict mean=0.646875 ✓ | CORRECT |
| 8 | 15 dB-এ IABR 0.8662 (paper), Teacher 0.8994, DFT-SIC 0.9042 | 07 §1 | E3 json @15dB | IABR_s0=0.86617 (round 0.8662, **শুধু s0**, 3-seed mean এখানে হলে 0.8683 হতো — but headline convention IABR_s0 ব্যবহার করে, table-এও তাই, তাই এটা misleading নয়) ; Teacher=0.899413✓; DFT-SIC=0.904167✓ | CORRECT (consistent with s0-headline convention used throughout tables) |
| 9 | E4: SNR15, 5°: IABR 0.8510 (Abl1 0.8607); Teacher 0.8859, DFT-SIC 0.9007 | 07 §2 (master table) | tables/E4_snr15.md | সব মান হুবহু মেলে | CORRECT |
| 10 | E5: SNR15, 4dB(OOD): IABR 0.8033, retention 92.3% (সর্বোচ্চ); Teacher 0.7787, DFT-SIC 0.8387 | 07 §1, §2 | tables/E5_snr15.md | 0.8033/0.8700=92.34%✓; Teacher/DFT-SIC মেলে | CORRECT |
| 11 | E5 (master table §2 row): "Teacher/Student/Abl1-কে হারায় (**+2.4 থেকে +2.9pp**)" | 07 §2 | tables/E5_snr15.md, tables/Controls.md | vs Teacher: 0.8033−0.7787=+2.46pp; vs Abl1: 0.8033−0.7790=+2.43pp (এই দুটো range-এর মধ্যে); কিন্তু **vs Student: 0.8033−0.7611=+4.22pp**, যা +2.4–2.9pp range-এর বাইরে। Source report 03 কখনো "Teacher/Student/Abl1" একসাথে একটা range দেয়নি — শুধু "Abl1-এর চেয়ে +2.1–2.6pp" (৩টা IABC variant) আর "Teacher-এর বিরুদ্ধে paired +2.93pp" আলাদাভাবে বলেছে | **WRONG** (Student margin misrepresented) |
| 12 | E4/E5/E6 IABC vs Abl1: "প্রায় সব condition-এ...paired CI শূন্য বাদ দেয় না" ও একমাত্র exception gain4dB OOD-এ +2.1–2.6pp (৩ variant-এ পুনরাবৃত্ত) | 07 §1, §3.1(1) | tables/E5_snr15.md, tables/Controls.md | IABR−Abl1=+2.43pp, E1−Abl1=+2.13pp, Abl3−Abl1=+2.57pp → range +2.13–2.57pp ("+2.1 থেকে +2.6pp" মেলে) | CORRECT |
| 13 | E6: SNR15, 15°: IABR 0.8043 (Abl1 0.8017, ns); Teacher 0.8081, DFT-SIC 0.8427 | 07 §2 | tables/E6_snr15.md | সব মান হুবহু মেলে | CORRECT |
| 14 | E7: SNR15 paper L1 0.928→L10 0.6694; strict L10 0.6694 vs Teacher 0.5426 | 07 §2 | tables/E7_snr15.md, E7_snr15_strict.md | IABR_s0 L1=0.9280, L10=0.6694 (paper=strict, no drop) ✓; Teacher strict L10=0.5426 ✓ | CORRECT |
| 15 | §4.1(4): E7 L10 strict IABR 0.3839 vs Teacher 0.0676 | 07 §4.1 | tables/E7_snr0_strict.md (SNR **0** dB, আলাদা condition থেকে E3 লাইনে ব্যবহৃত SNR15-এর চেয়ে) | IABR_s0=0.3839, Teacher=0.0676 (SNR0 dB) ✓ — সঠিক কিন্তু ভিন্ন SNR condition, যেটা spष्टভাবে label করা আছে | CORRECT |
| 16 | E8: 10°-এ IABR 0.6145, Teacher 0.9322, DFT-SIC 0.9765 | 07 §1, §2 | tables/E8.md | হুবহু মেলে | CORRECT |
| 17 | E9/E10-এর মতো Abl-2 root cause: coarse 0.1396 → learned-refine 0.6909 (+55.1pp) | 07 §1, §4.1(2) | tables/Abl2.md | 0.6909−0.1396=0.5513=55.13pp ✓ | CORRECT |
| 18 | Abl-2: IABR Teacher-এর 96.5% ও DFT-SIC-এর 94.3% accuracy পায় | 07 §4.1(2) | tables/Abl2.md | 0.6909/0.7158=96.52%✓; 0.6909/0.7326=94.31%✓ | CORRECT |
| 19 | §4.1(2): "মাত্র **41.6%** params দিয়ে" | 07 §4.1 | E0b.json vs Teacher params | 195024/469393=**41.5%** (নিজেদেরই §1-এ লেখা মানের সাথে সামঞ্জস্যহীন — 41.6% নয়, 41.5%) | **WRONG** (minor, internal inconsistency with own §1) |
| 20 | SNR_tails: 30→35dB Pd কমছে 0.8554→0.8411, Teacher বাড়ছে 0.9390→0.9465 | 07 §1, §2 | tables/SNR_tails.md | IABR_s0: 0.8554, 0.8411 ✓; Teacher: 0.9390, 0.9465 ✓ | CORRECT |
| 21 | E10: principal Pd দোলা ≤0.0283, SNR15@0dB IABR 0.8150, Teacher 0.8795, DFT-SIC 0.8817 | 07 §1, §2 | tables/E10.md | SNR15 principal pd: −20:0.8433, −10:0.8283, 0:0.8150 → range=0.0283 ✓; Teacher/DFT-SIC মেলে | CORRECT |
| 22 | E10 SNR15-এ IABR 4–7pp পিছিয়ে | 07 §4.2(10) | tables/E10.md | vs Teacher: −6.11 (−20), −7.15 (−10), −6.45pp (0); vs DFT-SIC: −4.59, −4.17, −6.67pp → range মোটামুটি −4.2 থেকে −7.15pp, দাবির "4–7pp"-এর সাথে সামঞ্জস্যপূর্ণ (সামান্য 7.15 > 7 কিন্তু rounding-এর ভেতরে) | CORRECT (approx.) |
| 23 | E10: strong-nuisance-এ IABR principal p95 23.55°→52.68° (SNR15) | 07 §3.1(3) | E10_nuisance_path.json এর `rows`-এ p95 field নেই — report 04-এর নিজস্ব derived cache-analysis থেকে | E10.json-এর raw table-এ শুধু pd columns আছে, p95 নেই। সংখ্যাটা report 04-র নিজস্ব cached-prediction re-scoring থেকে (04_scene_difficulty.md §434) — নিজের ভেতরে consistent, কিন্তু raw JSON/table দিয়ে সরাসরি cross-check করা যায়নি | **UNVERIFIABLE** (derived stat, cache script output, raw file-এ নেই) |
| 24 | Controls: E2 dense frontend E3=0.5704 (main-এর চেয়ে 11.8pp খারাপ, যদিও 2.34× বেশি params) | 07 §2, §4.1(3) | tables/Controls.md, E3.json | 0.6909−0.5704=0.1205=12.05pp (report বলছে "11.8pp" — 0.6871(Abl1)−0.5704=0.1167=11.67pp হলে "11.8pp" এর কাছাকাছি মেলে, main IABR_s0 (0.6909) থেকে হলে 12.0–12.1pp) | CORRECT (approx., baseline choice-নির্ভর সামান্য rounding, মূল দাবি ধরে রাখে) |
| 25 | Controls: Abl1 E3=0.6871 (main 0.6909, Δ −0.0038, ns) | 07 §2 | tables/Controls.md, E3.json mean_over_snr Abl1_no_IABC | 0.6871 (json: 0.6870833...) ✓; Δ=0.6909−0.6871=0.0038 ✓ | CORRECT |
| 26 | Abl-6: spread মাত্র 0.46–1.53pp সব 12 combo-তে | 07 §2 | tables/Abl6.md | E3_subset_pd min=0.6555, max=0.6601 → range=0.46pp; gain2_pd min=0.8040,max=0.8180→1.40pp; phase5_pd min=0.8150,max=0.8303→1.53pp → সব range 0.46–1.53pp-এর ভেতরে | CORRECT |
| 27 | E11: E3 std 0.0010 (values [0.6909,0.6897,0.6916]), phase_d5 std 0.0010, gain_g2 std 0.0025 | 07 §2 | E11_multi_seed.json | std=0.0009547≈0.0010✓; 0.0010184≈0.0010✓; 0.0025019≈0.0025✓ | CORRECT |
| 28 | E11-এর 25dB-তেই per-seed spread সবচেয়ে বেশি (std 0.0137) | 07 §3.1(5) | E11_multi_seed.json-এ 25dB breakdown নেই (শুধু overall E3_frozen); E3.json-এর per-seed 25dB pd (0.8745/0.8513/0.8755) থেকে derived | recompute (sample std, n−1): std≈0.0135–0.0137 (বাকি SNR ~0.001–0.003-এর তুলনায় স্পষ্টভাবে বড়) — matches | CORRECT (derived from E3.json, cross-checked) |
| 29 | V0: diff −8.6e-5, tolerance (2e-3)-এর 23 গুণ ছোট | 07 §2 | V0_teacher_reproduces_registry.json | 2e-3/8.603e-5=23.25× ✓ | CORRECT |
| 30 | V1: diff < 3σ সব SNR-এ (0.42σ, 0.32σ, 0.11σ) | 07 §2 | V1_generator_fidelity.json | 0.011/0.0263=0.418σ; 0.00535/0.01647=0.325σ; 0.0015/0.01325=0.113σ | CORRECT |
| 31 | Accuracy-per-parameter 0.3543, accuracy-per-FLOP 7.124 (সব NN baseline-এর সেরা) | 07 §3.1(4) | E9.json + E0b.json | Pd/100Kparams: IABR 0.6909375/1.95024=0.3543✓; Pd/GFLOP ব্যবহার হয়েছে rounded 0.097 GFLOPs (0.6909375/0.097=7.123)। Precise 0.09717 GFLOPs দিয়ে হিসাব করলে 7.11 আসে (7.124 নয়) — সামান্য rounding artifact, কিন্তু সিদ্ধান্ত (best-in-class) ঠিক থাকে | CORRECT (rounding-sensitive but directionally right) |
| 32 | status.json-এর সব 37 experiment 01–07-এ আলোচিত হয়েছে | সব report | status.json | ৩৭টা key-ই কোনো না কোনো report-এ (E0a, BANKS_build, E0b, TRAIN×১৮, V0, V1, E3, Abl2, E4–E8, SNR_tails, E10, Controls, E11, Abl6_sweep, E9) নাম ধরে আছে — কোনো experiment বাদ পড়েনি | CORRECT — কোনো omission নেই |
| 33 | সব figure discuss হয়েছে | সব report | outputs/figures/*.png (9 file) | E3_snr_sweep, E4_phase_snr0/15, E5_gain_snr0/15, E7_pathcount_snr0/15, E8_separation, training_loss — ৯টাই কোনো না কোনো report-এ reference করা হয়েছে; mother synthesis নিজেই বলছে E6/E9/E10/Abl2/Abl6-এর কোনো dedicated figure নেই যা সত্যি (ওই experiment-গুলোর কোনো PNG existই নেই — code omission, missing file নয়) | CORRECT — কোনো uncdiscussed figure নেই |

---

## ২. Omitted experiments বা figures

**কোনোটাই omitted পাওয়া যায়নি:**
- `status.json`-এর ৩৭টা experiment key-ের প্রতিটাই কোনো না কোনো group report-এ (01–06) বা mother synthesis (07)-এ নাম ধরে আলোচিত হয়েছে।
- `outputs/figures/`-এর ৯টা PNG-ই (E3_snr_sweep, E4_phase_snr0, E4_phase_snr15, E5_gain_snr0, E5_gain_snr15, E7_pathcount_snr0, E7_pathcount_snr15, E8_separation, training_loss) কোনো না কোনো report-এ reference/discuss করা হয়েছে।
- E6, E9, E10, Abl-2, Abl-6-এর জন্য কোনো figure file-ই `outputs/figures/`-এ নেই (generate করা হয়নি) — mother synthesis-এর নিজস্ব caveat (§5) এটা সঠিকভাবে flag করেছে, এটা omission নয়, বরং notebook-এর design choice।

---

## ৩. সারাংশ

মোট **৩৩টা claim** যাচাই করা হয়েছে (headline numbers, master comparison table, cross-experiment pattern দাবি, efficiency ratio)। এর মধ্যে:
- **২৯টা CORRECT**
- **৩টা WRONG**
- **১টা UNVERIFIABLE** (raw file-এ সরাসরি নেই, শুধু analyst-এর derived cache-analysis)

সবচেয়ে গুরুত্বপূর্ণ ভুলটা হলো **#6**: mother synthesis-এর executive-summary headline number "IABR (0.6909, 3-seed mean)" ভুল — actual 3-seed mean 0.6907, 0.6909 শুধু IABR_s0-এর একার মান। এটা thesis-এর headline sentence-এ ব্যবহৃত হলে সংশোধন দরকার।
