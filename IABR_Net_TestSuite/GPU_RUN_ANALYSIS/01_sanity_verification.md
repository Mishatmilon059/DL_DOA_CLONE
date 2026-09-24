# 01 — Sanity ও Verification group: E0a, BANKS_build, E0b, V0, V1

> **Analyst:** "Sanity & verification" agent
> **যেসব file পড়া হয়েছে:** notebook source (`IABR_Net_Full_Test_Suite.ipynb`, cell 0–13, 17–19, 22), executed notebook (`full_run.ipynb`), `outputs/RESULTS.md`, `outputs/results/{E0a_phase0_unit_tests, E0b_model_instantiation, BANKS_build, V0_teacher_reproduces_registry, V1_generator_fidelity, status}.json`, `outputs/tables/V1.md`, `outputs/logs/run_log.txt`, `outputs/logs/config.json`, `data/generated_banks/MANIFEST.json`, `ResearchState/FINAL_RESEARCH_REPORT.md` (§17), `PROJECT_STATUS.md`।
> **নিজে আলাদা করে যা যাচাই করেছি (read-only, কোনো training চালাইনি):** 56টা bank file-এর sha256 বনাম MANIFEST, কয়েকটা bank-এর shape/spec, frozen bank ও নতুন bank-এর signal+noise power, parameter আর FLOP-এর হাতে-করা হিসাব। এগুলো নিচে **"[নিজস্ব যাচাই]"** লেবেল দিয়ে আলাদা করে দেখানো আছে, যাতে notebook-এর নিজের output-এর সাথে গুলিয়ে না যায়।

---

## 0. এক নজরে (TL;DR)

| Experiment | কী যাচাই করে | মূল সংখ্যা | Verdict |
|---|---|---|---|
| **E0a** Phase-0 unit tests | physics, simulator, impairment injection, 16↔64 conversion, angle↔cell mapping | 11টার মধ্যে 11টা check PASS; numerical error সব ~1e-15 বা 0.0 | ✅ **Success** |
| **BANKS_build** | 56টা fixed test bank আছে এবং hash করা আছে | `n_banks` 56, `build_min` 0.00039453903834025064 (মানে নতুন করে কিছু generate হয়নি, আগে থেকে ship করা bank ব্যবহার হয়েছে) | ✅ **Success** (একটা সতর্কতা আছে, §3 দেখো) |
| **E0b** model instantiation | IABR-Net-এর আসল parameter, FLOPs, memory | params 195,024 (trainable 193,744), FLOPs 97,169,941, memory 0.74395751953125 MB | ✅ **Success** — estimate-এর চেয়ে সামান্য বেশি, আর পার্থক্যটা পুরোপুরি ব্যাখ্যা করা যায় |
| **V0** Teacher registry-র সংখ্যা reproduce করে কিনা | weights + evaluator + frozen bank ঠিক আছে কিনা | 0.7102635574471814, registry 0.7103495885388549, diff −8.603109167348855e-05 (tolerance 2e-3) | ✅ **Success** — তবে আগের মতো bit-exact না (§4 দেখো) |
| **V1** নতুন generator কতটা বিশ্বস্ত | নতুন bank-গুলো পুরনো data distribution-এর সাথে মেলে কিনা | তিনটা SNR-এই \|diff\| < 3σ (0.0110 < 0.0263, 0.0053 < 0.0165, 0.0015 < 0.0132) | ✅ **Success** — তবে test-এর coverage সীমিত |

**মূল কথা:** এই পাঁচটা gate-এর সবগুলো pass করেছে। তাই পরের সব experiment-এ (E3–E11, Abl-*) IABR-Net বনাম baseline-এর যে পার্থক্য দেখা যাচ্ছে, সেটা simulator, evaluator, weights বা data-র কোনো bug থেকে আসেনি। পার্থক্যটা সত্যিই model-এর নিজের। এটাই এই group-এর সবচেয়ে বড় "success"। মনে রাখা দরকার, এই group-এর কাজ IABR-Net ভালো কিনা বলা নয়। এর কাজ হলো বাকি সব সংখ্যাকে বিশ্বাসযোগ্য বানানো।

---

## 1. কেন এই পাঁচটা test বাকি সবকিছুর আগে pass করতে হবে

পুরো notebook-টা একটা dependency chain-এর উপর দাঁড়িয়ে আছে:

```
E0a (physics + simulator ঠিক?) ──► BANKS_build (নতুন test data ওই simulator দিয়ে বানানো)
       │                                   │
       │                                   └──► V1 (নতুন data পুরনো data-র মতো আচরণ করে?)
       │                                              │
       ▼                                              ▼
E0b (model যা বলা হয়েছে তাই?)      E4/E5/E6/E7/E8/E10/SNR-tails (সব নতুন bank-এর উপর চলে)
       │
       ▼
TRAIN_* (training-ও একই simulate() function দিয়ে on-the-fly data বানায়)

V0 (Teacher + evaluator + frozen bank ঠিক?) ──► E3, Abl-2, E9-এর mean_pd, এবং সব table-এর "Teacher" row
```

- **E0a ফেল করলে:** training data এবং সব robustness bank ভুল physics দিয়ে বানানো হতো। তখন model যত ভালো বা খারাপই দেখাক, সেটার কোনো মানে থাকত না। এজন্য code-এ `assert all(passed.values()), 'E0a physics checks failed -- do not trust anything downstream'` রাখা আছে (cell 6), আর `BANKS_build` চালানো হয় `needs=['E0a_phase0_unit_tests']` দিয়ে।
- **BANKS_build ফেল করলে বা bank বদলে গেলে:** E4–E10-এর সংখ্যা reproducible থাকত না। একেক run-এ একেক data হতো।
- **E0b না থাকলে:** thesis-এ "195K params, ~41% of teacher" বা "157× fewer FLOPs" দাবি শুধু estimate-এর উপর দাঁড়িয়ে থাকত। Research report নিজেই §17-এ লিখেছে: "the total has **not** been measured"।
- **V0 ফেল করলে:** এর মানে হতো Teacher weights, evaluator (`prepare_for_metric → get_ang_difference → filter_angles`), frozen bank বা 64×64 input reconstruction-এর কোনো একটায় গোলমাল। তখন এই run-এর কোনো সংখ্যাকেই project-এর আগের সংখ্যার (registry 0.7103) সাথে তুলনা করা যেত না।
- **V1 ফেল করলে:** নতুন bank-গুলো (E4–E10) frozen bank-এর চেয়ে "সহজ" বা "কঠিন" distribution থেকে আসত। তখন robustness curve-এ যে ঢাল দেখা যায়, সেটা impairment থেকে এসেছে নাকি data distribution-এর পার্থক্য থেকে, আলাদা করা যেত না।

---

## 2. Run-এর provenance (কোন session-এ কী চলেছে)

এটা বোঝা দরকার, কারণ এই group-এর তিনটা result (E0a, BANKS, E0b) **প্রথম session**-এ হিসাব হয়ে cache হয়েছিল:

- `run_log.txt` line 1–83: **Session 1** শুরু `2026-09-23T23:31:28`। এতে E0a, BANKS_build, E0b চলেছে, তারপর IABR_s0, IABR_s1 পুরো train হয়েছে আর IABR_s2 step 2000 পর্যন্ত গেছে (line 83: `[   45.5 min]   [IABR_s2] step 2000/20000 ...`)। এরপর log থেমে গেছে, অর্থাৎ session-টা মাঝপথে বন্ধ হয়েছিল।
- line 84 থেকে: **Session 2** শুরু `2026-09-24T04:06:50`। এখানে `[E0a_phase0_unit_tests] cached`, `[BANKS_build] cached`, `[E0b_model_instantiation] cached` দেখা যায়, আর `[IABR_s2] resumed at step 2000`।
- `RESULTS.md`-এর "Total wall-clock this session: 3.07 h" শুধু Session 2-এর সময়। দুই session মিলিয়ে মোট সময় আনুমানিক 3.07 h + ~45.5 min।
- V0 আর V1 চলেছে Session 2-তে (`[170.2 min]` থেকে `[170.8 min]`)।
- দুই session-ই একই machine ও config-এ চলেছে: Linux (`/home/ubuntu/...`), `TF 2.21.0 | numpy 2.5.3 | device: GPU`, `SMOKE_TEST = False`। প্রতিটা JSON-এ `"_smoke_test": false`।

**মূল্যায়ন:** cache-এ রাখা result আর নতুন result একই code, config আর environment থেকে এসেছে, তাই মিশে যাওয়ার সমস্যা নেই। একটা ছোট বিষয় আছে (এটা এই group-এর বাইরে, কিন্তু sanity-র অংশ): resume-এর সময় model, optimizer আর step checkpoint থেকে restore হয় (`tf.train.Checkpoint(model=model, optimizer=opt, step=step_var)`)। কিন্তু data iterator `make_dataset(seed)` আবার একই seed থেকে শুরু হয়। ফলে IABR_s2 প্রথম 2000 step-এর batch-গুলো দ্বিতীয়বার দেখেছে। data অসীম ও synthetic, তাই প্রভাব নগণ্য হওয়ার কথা। IABR_s2-এর E3 mean Pd 0.6916, যা তিন seed-এর মধ্যে সবচেয়ে বেশি। তাই resume-এ কোনো ক্ষতি হয়নি।

---

## 3. E0a — Phase-0 unit tests (physics + gain injection)

### (1) কী test করা হয়েছে
Simulator, codebook, impairment injection, data storage format আর angle↔beam-cell mapping-এর মোট **11টা** নির্দিষ্ট গাণিতিক বৈশিষ্ট্য। এর মধ্যে কয়েকটা check নতুন লেখা code-কে project-এর **original** generator (`dldoa_dataset_generation.py`) আর frozen bank-এর সাথে মিলিয়ে দেখে।

### (2) Theory
- Observation model: `Y = Wᴴ H F + Z`। এখানে `F` (nt×P) আর `W` (nr×Q) হলো 16×16 DFT codebook, আর `H = √(nt·nr) Σ_l α_l a_r(ψ_l) a_t(φ_l)ᴴ`।
- Codebook square আর **unitary** হলে (`WᴴW = I`, `FᴴF = I`), `W · Y · Fᴴ` দিয়ে সরাসরি antenna domain-এ ফেরত যাওয়া যায়। IABR-Net-এর stage 0 (inverse-codebook transform) এই একটা গাণিতিক সত্যের উপর দাঁড়িয়ে আছে।
- Hardware impairment: per-antenna complex error, `F = D_t F_ideal`, `W = D_r W_ideal`, `D = diag(g·e^{jε})`। Phase `ε ~ U(−δ, δ)`, আর gain `20log10(g) ~ U(−γ, γ)` dB। Gain-এর distribution-টা **[ASSUMPTION]**, কারণ research report এটা নির্দিষ্ট করেনি।
- তাহলে `W_ideal · Y_noiseless · F_idealᴴ = D_rᴴ H D_t`। মানে antenna domain-এ error-টা শুধু দুইটা diagonal matrix। IABC-v2 ঠিক এই জায়গাতেই per-element correction করে। এই identity ভুল হলে IABC-v2-এর পুরো ধারণাটাই ভেঙে পড়ে।
- DFT beamspace-এ একটা single path থাকলে |Y|-এর peak সবসময় সবচেয়ে কাছের DFT bin-এ পড়ে। তাই continuous cell position থেকে peak-এর দূরত্ব ≤ 0.5 cell হওয়ার কথা।

### (3) Code (cell 6, `def e0a()`)
| # | Check (JSON key) | Code-এ আসলে কী করে | Tolerance |
|---|---|---|---|
| 1 | `W_ideal unitary`, `F_ideal unitary` | `max|WᴴW − I|`, `max|FᴴF − I|`। `W_IDEAL`, `F_IDEAL` আসে original `DG.beamforming_vector_generation_Q/P` থেকে | 1e-10 |
| 2 | `simulator == original generate_channel_v2 path` | L=3 আর L=5-এর দুইটা sample। নতুন vectorized `simulate()` বনাম original `DG.generate_channel_v2` + `W_IDEALᴴ H F_IDEAL + noise`, একই angle, α আর noise দিয়ে | 1e-10 |
| 3a | `gain within +-2 dB` | `gain_db_max=2`-এ একটা sample-এর 32টা element-এর `max|20log10|D||` | ≤ 2.0 |
| 3b | `phase within +-5 deg` | ওই sample-এর `max|angle(D)|` (deg) | ≤ 5.0 |
| 3c | `inverse codebook recovers D_r^H H D_t` | impaired, noiseless Y থেকে `W_IDEAL·Y·F_IDEALᴴ` বনাম `conj(D_r)·H·D_t` | 1e-9 |
| 4 | `original phase error is per-antenna (same across beams)` | `DG.beamforming_vector_generation_P(P, NT, error_deg=5)` / `F_IDEAL`। অনুপাতটা প্রতিটা column (beam)-এ একই কিনা, অর্থাৎ original generator-এর error আসলেই per-antenna diagonal কিনা | 1e-12 |
| 5a | `frozen bank: upsample(downsample(x)) == x` | frozen bank-এর প্রথম 64টা 64×64 input → 16×16 → আবার 64×64 | ঠিক 0.0 |
| 5b | `frozen bank P values` | `fb['meta'][:,2]`-এর unique মান | `[16]` হতে হবে |
| 6a | `|Y| argmax within 0.5 cell of predicted cell (200 single paths)` | 200টা single-path, SNR 300 dB (কার্যত noiseless)। `argmax|Y|` বনাম `angles_to_cells(ψ, φ)`, circular distance ধরে | ≤ 0.5 |
| 6b | `cells->angles round trip (max |cos err|)` | `cells_to_angles(angles_to_cells(ψ, φ))`-এর cos error | 1e-9 |

### (4) কেন করা হয়েছে
- এই notebook-এ নতুন লেখা সব data-path code (vectorized simulator, gain injection, 16×16 compact storage, IABR-Net-এর নিজস্ব cell↔angle decoding) project-এর আগের যাচাই করা code-এর সাথে হুবহু মেলে কিনা, সেটা প্রমাণ করা।
- Hypothesis: "নতুন code কোনো নতুন physics ঢোকায়নি। শুধু gain error যোগ হয়েছে, আর সেটা ঠিকমতো per-antenna diagonal হিসেবে ঢুকেছে।"
- Research report নিজেই বলেছিল, gain injection আর FLOP counter-এর মতো Phase-0 code তখনো লেখা হয়নি ("blocking unbuilt-code dependency")। E0a সেই নতুন code-এর প্রথম পরীক্ষা।

### (5) Result (`E0a_phase0_unit_tests.json`, হুবহু)
| Check | Value | Passed |
|---|---|---|
| W_ideal unitary | 2.2514774938931273e-15 | true |
| F_ideal unitary | 2.4422301218707346e-15 | true |
| simulator == original generate_channel_v2 path | 3.1264164235997256e-15 | true |
| gain within +-2 dB | 1.8223193193360356 | true |
| phase within +-5 deg | 4.910482083241738 | true |
| inverse codebook recovers D_r^H H D_t | 7.027150473277823e-15 | true |
| original phase error is per-antenna (same across beams) | 2.2887833992611187e-16 | true |
| frozen bank: upsample(downsample(x)) == x | 0.0 | true |
| frozen bank P values | [16] | true |
| \|Y\| argmax within 0.5 cell of predicted cell (200 single paths) | 0.49933410819027557 | true |
| cells->angles round trip (max \|cos err\|) | 3.3306690738754696e-16 | true |

`"all_passed": true`, `"_elapsed_min": 0.0034407774607340493`। run_log line 11–21-এ প্রতিটার পাশে `PASS` লেখা।

**প্রতিটা সংখ্যার মানে:**
- `~1e-15` মাত্রার মানগুলো float64-এর machine precision (≈2.2e-16 × কয়েকটা operation)। অর্থাৎ এগুলো গাণিতিকভাবে হুবহু সমান। কোনো approximation error নেই।
- `0.0` (upsample/downsample): frozen bank-এর 64×64 input আসলেই 16×16-এর zoom-4 nearest-neighbour copy। তাই compact 16×16 storage থেকে baseline-দের জন্য **bit-identical** input আবার বানানো যায়। এটা V0-এর পূর্বশর্ত।
- `1.8223…` dB আর `4.9104…` deg: একটা sample-এর 32টা element-এর মধ্যে সর্বোচ্চ মান, যা সীমা (2 dB, 5°)-এর ভেতরে এবং কাছাকাছি। অর্থাৎ injection আসলেই পুরো range জুড়ে হচ্ছে।
- `0.49933…`: 200টা path-এর মধ্যে সবচেয়ে খারাপটায় peak আর আসল continuous cell-এর দূরত্ব 0.4993 cell। Theory অনুযায়ী এটা কখনো 0.5-এর বেশি হতে পারে না, আর 200 sample-এ 0.5-এর খুব কাছে যাওয়াই স্বাভাবিক। মানটা tolerance-এর গা ঘেঁষে দেখালেও এটা physics-এর প্রত্যাশিত আচরণ, borderline fail নয়। এই check প্রমাণ করে যে AoA→row (−cos convention) আর AoD→column (+cos convention)-এর sign ও wrap convention ঠিক আছে।

### (6) Comparison
E0a কোনো model-এর test নয়, তাই এখানে Teacher, Student, FNO বা DFT-SIC-এর সাথে সরাসরি তুলনা নেই। তবে দুইটা পরোক্ষ তুলনা প্রাসঙ্গিক:
- `PROJECT_STATUS.md`-এ লেখা আছে, local CPU-তে (TF 2.21) আগের যাচাইয়ে "inverse codebook recovers `D_rᴴHD_t` to 8e-15" আর simulator "to 3e-15" এসেছিল। GPU run-এর মান 7.027e-15 আর 3.126e-15, অর্থাৎ একই মাত্রা। Machine বদলালেও physics check স্থির থেকেছে।
- আগের classical sanity test (DFT-SIC, 20 dB-এ Pd 0.884 vs paper 0.891) project-এর physics আগেই যাচাই করেছিল। E0a check 2 দেখায় যে নতুন vectorized simulator সেই যাচাই করা original generator-এর সাথেই ~1e-15-এ মেলে। (খেয়াল রাখো: `PROJECT_STATUS.md` line 200 অনুযায়ী 0.884 সংখ্যাটা ছিল per-source metric। এই run-এ DFT-SIC-এর frozen bank 20 dB `pd_source` = 0.8613333333333333, `NDFT=512`-এ। আগের sanity notebook `NDFT=1024` ব্যবহার করেছিল, তাই দুইটা হুবহু তুলনীয় নয়।)

### (7) Verdict: ✅ **Success**
11টার মধ্যে 11টা pass, আর numerical মানগুলো machine precision-এ। Simulator, codebook, impairment model আর storage format নিয়ে সন্দেহের কোনো জায়গা নেই।

**সীমাবদ্ধতা (Weakness of the test itself, model-এর নয়):**
1. Gain/phase-এর range check হয়েছে মাত্র **একটা sample**-এর 32টা element-এ। Distribution সত্যিই uniform কিনা (যেমন histogram বা KS test) যাচাই করা হয়নি।
2. Simulator == original check শুধু **impairment ছাড়া** করা হয়েছে, দুইটা sample-এ (L=3, 5)। Impaired path-এর সাথে original generator-এর তুলনা আলাদাভাবে হয়নি। Check 4 শুধু দেখায় original error per-antenna। Check 3 দেখায় নতুন injection-ও per-antenna। দুইটা মিলিয়ে convention একই, এটা যুক্তিসঙ্গত অনুমান, কিন্তু সরাসরি প্রমাণ নয়।
3. Check 2-এ noise দুই পথেই একই array দেওয়া হয়। তাই **SNR/noise variance definition** এখানে test হয়নি। (V1 আর আমার নিজস্ব power check (§8) এটা পরোক্ষভাবে cover করে।)
4. Path amplitude-এর নিয়ম (`α ~ CN(0, 1/L)`, strongest first) original generator-এর সাথে সরাসরি মেলানো হয়নি। V1 এটাও পরোক্ষভাবে cover করে।
5. `cells_to_angles` শুধু নিজের inverse-এর সাথে round-trip করা হয়েছে, original `peaks_to_angles`-এর সাথে নয়। তবে IABR-Net-এর E3 Pd 0.69, তাই mapping ভুল হলে Pd প্রায় শূন্য হতো। কার্যত mapping ঠিক আছে।

---

## 4. BANKS_build — 56টা fixed evaluation bank

### (1) কী test করা হয়েছে
Robustness experiment-গুলোর জন্য 56টা deterministic, fixed-seed test bank `data/generated_banks/`-এ আছে কিনা। প্রতিটার sha256 hash `MANIFEST.json`-এ লেখা হয়।

### (2) Theory
Reproducibility-র নিয়ম হলো, সব model-কে **একই sample**-এ মূল্যায়ন করতে হবে (paired comparison)। তাহলে model-দের মধ্যে পার্থক্য sampling noise-এর কারণে হবে না। নির্দিষ্ট seed আর hash দুইটা জিনিস নিশ্চিত করে: (ক) আবার চালালে হুবহু একই data আসবে, (খ) data কেউ বদলে দিলে ধরা পড়বে।

### (3) Code (cell 7, `build_banks()`, `BANK_SPECS`)
- `make_standard_bank`: L paths, নির্দিষ্ট SNR, phase বা gain error, `n = CFG['n_robust'] = 500` sample।
- `make_separation_bank`: L=2, দ্বিতীয় path-এর AoA ও AoD দুইটাই প্রথমটার থেকে `sep_deg` দূরে, angle [20°, 160°]-এর মধ্যে।
- `make_nuisance_bank`: 3টা principal path + 1টা nuisance path (−20/−10/0 dB), 200 scene। একই seed ব্যবহার করে power level-গুলোর মধ্যে paired।
- Bank-এর হিসাব: SNR {0, 15} × (phase {0,1,2,5} = 4, gain {0.5,1,2,4} = 4, OOD phase {10,15} = 2, L {1,2,4,5,6,7,8,10} = 8, nuisance {−20,−10,0} = 3) = 2 × 21 = 42। সাথে separation {30,20,10,5,3,2,1}° = 7, SNR tails {−25,−20,30,35} = 4, V1 clean L=3 {0,15,25} = 3। মোট **56**।
- **গুরুত্বপূর্ণ:** `make_*` function-গুলো file আগে থেকে থাকলে `return` করে। Markdown cell 8-এ লেখা আছে "The package ships them pre-generated"।
- Code পড়ে দেখা গেছে, E5-এর gain = 0 column আর E7-এর L = 3 column **আলাদা bank নয়**। দুইটাই `phase_d0_snr{snr}` bank ব্যবহার করে (E5-এর code-এ `banks = [f'phase_d0_snr{snr}'] + ...`)। তাই E4-এর d=0, E5-এর g=0 আর E7-এর L=3-এর মান একই। (যেমন Teacher 15 dB-এ তিন জায়গাতেই 0.9127।)

### (4) কেন করা হয়েছে
E4–E10-এর প্রতিটা table-এ 7টা model একই 500টা sample-এ তুলনা হয়। Bank fixed না থাকলে দুইটা model-এর মধ্যে 0.01 Pd-এর পার্থক্য sampling noise নাকি আসল, বলা যেত না। Frozen `eval_bank.npz` (E3-এর জন্য) ছোঁয়া হয়নি, আর নতুন সব condition নতুন bank-এ রাখা হয়েছে।

### (5) Result (`BANKS_build.json`, হুবহু)
- `"n_banks": 56`
- `"dir": "data/generated_banks"`
- `"build_min": 0.00039453903834025064` (≈ 0.024 সেকেন্ড)
- `"_elapsed_min": 0.0003946105639139811`

**[নিজস্ব যাচাই]** আমি নিজে 56টা `.npz` file-এর sha256 হিসাব করে `MANIFEST.json`-এর সাথে মিলিয়েছি: **56 entries, 0 mismatch**। নমুনা হিসেবে কয়েকটা bank-এর ভেতরের তথ্য spec-এর সাথে মিলেছে:
- `v1_clean_snr15`: `Y16` (500,16,16,2), `psi`/`phi` (500,3), `phase_deg` 0.0, `gain_db` 0.0, `seed` 7015 (spec: n 500, L 3, snr 15, seed 7015)
- `gain_g2_snr15`: `gain_db` 2.0, `seed` 2035
- `sep_5deg_snr15`: `psi` (500,2), `sep_deg` 5.0, `seed` 5005
- `nuis_p0_snr0`: `Y16` (200,16,16,2), `psi` (200,4), `power_db` 0.0, `seed` 1000

### (6) Comparison
Model comparison এখানে প্রযোজ্য নয়। Frozen bank-এর সাথে তুলনা: frozen bank-এ 8000 sample (প্রতি SNR-এ 1000, SNR −10…25 dB, step 5), **সব L = 3** ([নিজস্ব যাচাই]: `meta[:,0]`-এর Counter {3: 8000}), P=Q=16, `sigma` 0.07, `M` 256, `seed` 42। নতুন bank প্রতি condition-এ 500 sample (nuisance-এ 200)। অর্থাৎ robustness table-এর প্রতিটা ঘরের statistical uncertainty frozen bank-এর চেয়ে প্রায় √2 গুণ বেশি।

### (7) Verdict: ✅ **Success** (সতর্কতা সহ)
56টা bank আছে, spec মেনে বানানো, আর hash intact। **সতর্কতা:** `build_min` প্রায় শূন্য, মানে এই GPU run-এ bank **নতুন করে generate হয়নি**। আগে (local CPU-তে, package বানানোর সময়) generate করা file ব্যবহার হয়েছে। এটা reproducibility-র দিক থেকে ভালো (একই data)। কিন্তু এর অর্থ, এই GPU machine-এ `make_*` function আবার চালালে হুবহু একই bytes আসবে কিনা, সেটা এই run-এ প্রমাণ হয়নি। Hash শুধু প্রমাণ করে যে file পথে বদলায়নি। V1 দেখায় যে এই bank-এর distribution ঠিক আছে, তাই practical ঝুঁকি কম।

---

## 5. E0b — IABR-Net-এর আসল size (params, FLOPs, memory)

### (1) কী test করা হয়েছে
Research report-এ IABR-Net-এর যে size estimate করা হয়েছিল (≈193,104 params, ≈95.3M FLOPs), সেটার জায়গায় আসল instantiate করা model-এর মাপা সংখ্যা বসানো।

### (2) Theory
- Parameter count হলো সব weight tensor-এর মোট element সংখ্যা। BatchNorm-এর moving mean/variance **non-trainable**, কিন্তু `count_params()` এগুলোও গোনে।
- FLOPs গোনার convention: 1 MAC = 2 FLOPs। একটা 3×3 conv, Cin=Cout=32, 16×16 map-এ: `2·9·32·32·256 = 4,718,592` FLOPs।
- TF profiler complex einsum আর FFT গোনে না। তাই IABR-Net-এর stage-0 codebook transform (4টা complex 16×16×16 matmul, প্রতি complex MAC = 8 real FLOPs) আলাদাভাবে analytically যোগ করা হয়েছে।

### (3) Code (cell 10, `e0b()`, `profiler_flops()`, `complex_einsum_flops_iabr()`)
- `build_iabr()` দিয়ে model বানায়, তারপর প্রতিটা layer-এর `count_params()` নেয়।
- `iabr_infer_fn` = backbone → `top_k` (K=10) → refine। এটাকে `convert_variables_to_constants_v2` দিয়ে freeze করে `tf.compat.v1.profiler`-এর `float_operation()` চালানো হয়, batch 1-এ।
- `complex_einsum_flops_iabr()` = `4 * 16**3 * 8` = 131,072।

### (4) কেন করা হয়েছে
Thesis-এর efficiency দাবি (Teacher-এর তুলনায় কম params, অনেক কম FLOPs) estimate থেকে measurement-এ নিয়ে যাওয়ার জন্য। Report নিজেই এটাকে Reproducibility Checklist-এর item 1 বলেছিল। `graph_op_types` দেখায় যে পুরো pipeline (IABC-v2-এর `Angle`, `Cumsum`, `ComplexAbs`, `TopKV2`, `GatherNd`) একটা TF graph-এর ভেতরে চলে। মানে inference-এ বাইরের numpy step নেই (blob detection বাদে, যা E9-এর e2e latency-তে ধরা হয়)।

### (5) Result (`E0b_model_instantiation.json`, হুবহু)
| Field | Value |
|---|---|
| params_total | 195024 |
| params_trainable | 193744 |
| report_estimate_params | 193104 |
| flops_profiler | 97038869 |
| flops_complex_supplement | 131072 |
| flops_total | 97169941 |
| report_estimate_flops | 95300000.0 |
| profiler_counted_ops | {"Conv2D": 95207424} |
| param_memory_MB | 0.74395751953125 |

Per-layer: `front_end` 250, `conv2d_264` (stem) 608, `se_res_block` … `se_res_block_9` প্রতিটা 19044, `conv2d_285` (token) 1056, `conv2d_286` (heat) 33, `dense_20` 528, `dense_21` 785, `dense_22` 528, `dense_23` 785, `dense_24` 11, `circular_pad*` 0।

**[নিজস্ব যাচাই] হাতে হিসাব:**
- `front_end` 250 = ctx Dense(4→8) 40 + mlp1 Dense(10→16) 176 + mlp2 Dense(16→2) 34 ✔
- `se_res_block` 19,044 = 2 × conv(3·3·32·32+32 = 9,248) + 2 × BN(4·32 = 128) + SE(32→4: 132, 4→32: 160) = 18,496 + 256 + 292 ✔
- মোট: 250 + 608 + 10×19,044 + 1,056 + 33 + 528 + 785 + 528 + 785 + 11 = **195,024** ✔
- Non-trainable = 195,024 − 193,744 = 1,280 = 10 block × 2 BN × 32 channel × 2 (moving mean + moving var) ✔
- **Report-এর estimate-এর সাথে পার্থক্য, 1,920:** report-এর §17-এ 10টা residual block মিলিয়ে ধরা হয়েছিল 185,600, মানে block-প্রতি 18,560 = 18,496 + 64। অর্থাৎ report block-প্রতি মাত্র একটা BN-এর γ/β (64) গুনেছিল। Keras-এ আসলে 2টা BN × (γ, β, moving mean, moving var) = 256। পার্থক্য 10 × (256 − 64) = **1,920**, যা ঠিক মাপা পার্থক্য (195,024 − 193,104)। বাকি সব component (IABC 250, SE 2,920, heads 1,313×2, SNR head 11) report-এর সাথে হুবহু মেলে।
- Conv FLOPs: 20 × 4,718,592 (trunk) + 294,912 (stem, Cin=2) + 524,288 (token 1×1) + 16,384 (heat 1×1) = **95,207,424** ✔। এটা profiler-এর `Conv2D` সংখ্যার সাথে হুবহু মেলে।
- `flops_profiler` (97,038,869) − Conv2D (95,207,424) = 1,831,445 FLOPs অন্য op থেকে এসেছে (মূলত crop-refine Dense/MatMul)। **একটা অস্পষ্টতা:** `profiler_counted_ops` dictionary-তে শুধু `Conv2D` দেখা যায়, অথচ total-এ বাকি 1.83M-ও আছে। অর্থাৎ per-op breakdown অসম্পূর্ণ। Total সংখ্যাটা ঠিক আছে, কিন্তু breakdown-এর উপর ভরসা করে কোনো দাবি করা উচিত নয়।

### (6) Comparison
| Model | Params | IABR-Net-এর params ওই model-এর কত % | GFLOPs (E9) |
|---|---|---|---|
| Teacher | 469,393 | 41.5% (195,024/469,393) | 15.203 |
| Student r=8 | 314,513 | 62.0% | 10.157 |
| FNO | 334,321 | 58.3% | 0.172 (FFT-এর analytic supplement সহ) |
| DFT-SIC | 0 | — | — |
| **IABR-Net** | **195,024** | — | **0.097** |

- FLOPs-এ Teacher ÷ IABR-Net ≈ 15.203e9 / 97.17e6 ≈ **156.5×** কম। Student-এর তুলনায় ≈104.5×, FNO-এর তুলনায় ≈1.8×।
- Params-এ IABR-Net সবচেয়ে ছোট learned model। Memory 0.744 MB (float32)। E9-এর `weights_MB` 1.01 হলো file overhead সহ।
- Ablation variant-গুলোর param সংখ্যাও এই breakdown-এর সাথে মেলে ([নিজস্ব যাচাই], Controls table থেকে): `Abl1_no_IABC` 194,774 = 195,024 − 250 ✔; `Abl3_no_SE` 192,093 = 195,024 − 2,920 − 11 ✔; `E1_capacity_control` 195,024 (একই architecture, শুধু pairing loss বন্ধ) ✔।

### (7) Verdict: ✅ **Success**
মাপা size estimate-এর 1% (params) আর 2% (FLOPs)-এর মধ্যে, আর params-এর পার্থক্যটা পুরোপুরি BN bookkeeping দিয়ে ব্যাখ্যা করা যায়। Efficiency দাবি এখন measurement-এর উপর দাঁড়িয়ে। **Thesis-এ ব্যবহারের পরামর্শ:** "~41.5% of teacher params, ~157× fewer FLOPs" লিখো। "~193K" নয়, **195,024** লেখো।

---

## 6. V0 — Teacher registry-র সংখ্যা reproduce করে কিনা

### (1) কী test করা হয়েছে
Base paper-এর pretrained Teacher (64-block ResNet, `inf_model_007_256_resnet.h5`, 469,393 params)-কে frozen 8000-sample bank-এ project-এর original evaluator দিয়ে চালালে mean paper-style Pd আগে record করা registry মানের (0.7103495885388549) সমান আসে কিনা।

### (2) Theory
একটা deterministic pipeline (fixed weights + fixed data + fixed evaluator) যেকোনো machine-এ একই সংখ্যা দেওয়ার কথা। Floating-point kernel-এর ক্রম বদলালে (cuDNN, oneDNN, batch size) heatmap-এ ~1e-6 মাত্রার পার্থক্য আসতে পারে। Blob detector-এর threshold-এর একদম কাছে থাকা কয়েকটা peak তাতে এদিক-ওদিক হতে পারে। তাই সামান্য পার্থক্য প্রত্যাশিত, কিন্তু বড় পার্থক্য মানে আসল mismatch।
- `pd_paper` (component-level, original convention): L-এর কম detection থাকলে পুরো sample বাদ ("dropped")।
- `pd_strict`: dropped sample-এর সব component miss হিসেবে গোনা হয়।
- `pd_source`: একটা path তখনই সঠিক যখন AoA আর AoD দুইটাই 1°-এর মধ্যে।

### (3) Code (cell 14, `def v0()`)
- `predict('Teacher', 'E3_frozen')`: frozen bank-এর `data` (64×64×2)-কে `downsample16` করে রাখা হয়, আবার `upsample64` করে Teacher-কে দেওয়া হয় (E0a check 5a-এর কারণে এটা bit-identical)। তারপর `get_blob_peaks` → top-L → `peaks_to_angles(sigma=0.07, grid_size=256)`।
- `by_snr()` → প্রতি SNR-এ `metrics()` → `prepare_for_metric` → `get_ang_difference` → 1° threshold।
- `ok = abs(mean_pd - ref) < 2e-3`, সাথে `assert ok`। Code-এর comment বলে: "a different batch size / cuDNN kernel may flip a handful of the 48,000 components, so allow 2e-3"।
- এই prediction `outputs/cache/`-এ cache হয়, আর E3, Abl-2, V1 একই cache ব্যবহার করে।

### (4) কেন করা হয়েছে
এটা পুরো evaluation stack-এর end-to-end test: Teacher weights ঠিকমতো load হয়েছে কিনা, frozen bank বদলায়নি কিনা, 16→64 reconstruction ঠিক কিনা, আর matching/metric code আগের মতো আচরণ করে কিনা। V0 pass করলে এই run-এর সব Pd সংখ্যা project-এর আগের সব সংখ্যা (baseline registry, FNO screening ইত্যাদি)-র সাথে সরাসরি তুলনীয়।

### (5) Result (`V0_teacher_reproduces_registry.json`, হুবহু)
- `teacher_mean_pd`: 0.7102635574471814
- `registry`: 0.7103495885388549
- `diff`: −8.603109167348855e-05
- `passed`: true

Per-SNR (Teacher, frozen bank, প্রতি SNR-এ 1000 sample):

| SNR | pd_paper | pd_strict | pd_source | rmse (deg) | p50 | p95 | p95_endfire | p95_broadside | pairing_error | n_dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| −10 | 0.2040072859744991 | 0.18666666666666668 | 0.086 | 0.5505716580799557 | 9.057772121122351 | 120.87238839250945 | 160.55471032190724 | 101.4750507573503 | 0.22003642987249544 | 85 |
| −5 | 0.43603801169590645 | 0.39766666666666667 | 0.25433333333333336 | 0.5114723463090313 | 1.283368711846481 | 110.70238266069617 | 166.6662636440757 | 79.10017094311549 | 0.31432748538011696 | 88 |
| 0 | 0.6392543859649122 | 0.583 | 0.438 | 0.4576029225703634 | 0.6013402381952855 | 86.64140660923954 | 169.5276953389372 | 48.83134650395048 | 0.31798245614035087 | 88 |
| 5 | 0.7788671023965141 | 0.715 | 0.598 | 0.3918291273719822 | 0.34304162506775226 | 33.882431436876125 | 171.7136085883185 | 1.8611760473804895 | 0.2549019607843137 | 82 |
| 10 | 0.8618143459915611 | 0.817 | 0.7333333333333333 | 0.3247391261404888 | 0.21352157767170943 | 3.140490848227453 | 170.8259622345174 | 0.9946395717618856 | 0.17651195499296765 | 52 |
| 15 | 0.8994133885438234 | 0.8688333333333333 | 0.7956666666666666 | 0.28001770206699045 | 0.14525807491859666 | 1.890385123384765 | 56.16225978218166 | 0.6187287752139532 | 0.15148378191856451 | 34 |
| 20 | 0.9251282051282051 | 0.902 | 0.846 | 0.25302258094549807 | 0.1118297445327248 | 1.459202289829643 | 87.87860526056356 | 0.4524371138684803 | 0.11487179487179487 | 25 |
| 25 | 0.9375857338820301 | 0.9113333333333333 | 0.8686666666666667 | 0.2381248907495086 | 0.09587175148823165 | 1.243369888924269 | 4.377656742143212 | 0.3977629472439415 | 0.0877914951989026 | 28 |
| **mean** | **0.7102635574471814** | **0.6726875** | **0.5775** | 0.37592254427922733 | — | 44.979007156210926 | 123.46334523908055 | — | — | — |

(Mean RMSE AoA 0.37511362229768364, AoD 0.3766945701758291।)

### (6) Comparison
- **আগের record-এর সাথে:** `PROJECT_STATUS.md` (line 169–171) অনুযায়ী আগের CPU run আর আগের GPU run দুইটাতেই "exact match to 13+ significant figures (0.7103495885388549 both times)" এসেছিল। এই নতুন Linux GPU-তে (TF 2.21, oneDNN on) মান 0.7102635574…। অর্থাৎ **এবার bit-exact নয়**, 4র্থ দশমিক ঘরে −0.0000860 পার্থক্য। [নিজস্ব অনুমান] 8টা SNR-এর mean-এ −8.6e-5 মানে per-SNR Pd-এর যোগফলে ≈ −6.9e-4। প্রতি SNR-এ ~5,500–5,900 matched component থাকে, তাই এটা মোটামুটি ~4টা component-এর সমান flip (বা dropped হওয়া বদলে যাওয়ার সমান)। Tolerance (2e-3) এর প্রায় 23 গুণ বড়। তাই এটা kernel-level floating-point পার্থক্য, আসল mismatch নয়। (কোন SNR-এ পার্থক্য হয়েছে বলা যাচ্ছে না, কারণ registry-র per-SNR মান এই package-এ নেই।)
- **E3-এর সাথে consistency:** E3 table-এর Teacher row (0.2040 … 0.9376, mean 0.7103, strict 0.6727, source 0.5775) হুবহু V0-এর per-SNR মান। এটা প্রত্যাশিত, কারণ একই cache।
- **অন্য model-দের সাথে (E3 থেকে, context-এর জন্য):** mean paper Pd: Teacher 0.7103, Student 0.6956, FNO 0.6235, DFT-SIC 0.7326, IABR_s0 0.6909। Student আর FNO-এর মান registry-র (`PROJECT_STATUS.md`: Student 0.6956, FNO 0.6235) সাথে 4 দশমিক ঘর পর্যন্ত মেলে। অর্থাৎ বাকি দুই baseline-ও reproduce হয়েছে। এটা V0-এর বাড়তি প্রমাণ, যদিও notebook এটা আলাদা gate হিসেবে assert করে না।
- **Strict বনাম paper-এর ফাঁক:** Teacher-এর mean paper 0.7103 কিন্তু strict 0.6727, কারণ প্রতি SNR-এ 25–88টা sample dropped (L-এর কম blob)। IABR-Net কখনো sample drop করে না (E3-এ IABR_s0-এর paper = strict = 0.6909)। তাই strict metric-এ IABR_s0 (0.6909) Teacher (0.6727)-এর চেয়ে এগিয়ে, কিন্তু paper metric-এ পিছিয়ে। V0 নিশ্চিত করে যে Teacher-এর এই drop আচরণ আসল, bug নয়। Thesis-এ কোন metric headline করা হবে, এই বিষয়টা সেখানে কেন্দ্রীয় হবে।
- **End-fire-এর লক্ষণ (Teacher-এর নিজের মধ্যেই):** `p95_endfire` 5 dB-এ 171.7°, 10 dB-এ 170.8°, অথচ `p95_broadside` মাত্র 1.86° আর 0.99°। অর্থাৎ angle 0° বা 180°-এর কাছে (যেখানে cos-domain resolution খারাপ) বড় error-এর tail **বেস Teacher-এরও আছে**। এটা IABR-Net-এর নতুন কোনো bug নয়, বরং physics আর evaluator-এর ধর্ম। Abl-2-এর end-fire বিশ্লেষণে এটা মনে রাখা দরকার।

### (7) Verdict: ✅ **Success**
Teacher, evaluator আর frozen bank ঠিক আছে। এই run-এর Pd সংখ্যা project-এর আগের সংখ্যার সাথে তুলনীয়। **ছোট দুর্বলতা:** আগের "13+ significant figures exact" দাবি এই GPU-তে টেকেনি। Thesis-এ লিখতে হবে "reproduces to within 8.6e-5 (≈0.01%)"। "bit-exact" লেখা যাবে না।

---

## 7. V1 — নতুন generator-এর fidelity

### (1) কী test করা হয়েছে
নতুন vectorized `simulate()` দিয়ে বানানো clean (impairment ছাড়া) L=3 bank-এ Teacher-এর Pd, original generator দিয়ে বানানো frozen bank-এ Teacher-এর Pd-এর সাথে statistically সমান কিনা। SNR 0, 15 আর 25 dB-এ দেখা হয়েছে।

### (2) Theory
দুইটা bank যদি একই distribution থেকে আসে, তাহলে একই fixed model-এর Pd-এর পার্থক্য শুধু sampling noise। Pd-কে Bernoulli proportion ধরলে standard error `se = √(p(1−p)/n_comp)`, যেখানে `n_comp = 6 × n_samples` (L=3 × 2 component)। `|diff| ≤ 3σ` হলে "একই distribution" hypothesis প্রত্যাখ্যান করা যায় না।
Teacher এখানে "measuring instrument" হিসেবে কাজ করে। এটা frozen bank-এর distribution-এ train করা, তাই distribution সামান্য বদলালেও (angle sampler, α-র নিয়ম, SNR definition, zoom map) এর Pd-তে সেটা ধরা পড়ার কথা।

### (3) Code (cell 14, `def v1()`)
- `fb = by_snr(predict('Teacher', 'E3_frozen'), ...)`: frozen bank-এর প্রতি SNR-এ Pd (V0-এর মতো একই cache)।
- `load_bank(f'v1_clean_snr{s}')`: 500 sample, L=3, phase 0, gain 0, seed 7000+s।
- `se = math.sqrt(max(ref*(1-ref), 1e-9) / (6*len(b['L'])))`
- `ok = abs(mine - ref) <= max(3*se, 0.02)`। অর্থাৎ tolerance হলো **3σ অথবা 0.02, দুইটার মধ্যে যেটা বড়**।
- Table লেখা হয় `outputs/tables/V1.md`-এ।

### (4) কেন করা হয়েছে
E4–E10 আর SNR-tails-এর সব bank এই নতুন generator দিয়ে বানানো। Generator যদি frozen bank-এর চেয়ে সহজ বা কঠিন data বানাত, তাহলে "phase error 5°-এ Pd কমেছে" জাতীয় সব সিদ্ধান্ত বিভ্রান্তিকর হতো। V1 দেখায় impairment = 0 অবস্থায় নতুন generator পুরনো generator-এর সমান।

### (5) Result (`V1_generator_fidelity.json` + `tables/V1.md`, হুবহু)
| snr | frozen_bank_pd | new_generator_pd | diff | three_sigma | ok |
|---|---|---|---|---|---|
| 0 | 0.6392543859649122 | 0.650255288110868 | 0.011000902145955727 | 0.026302559722745728 | true |
| 15 | 0.8994133885438234 | 0.9047619047619048 | 0.005348516218081412 | 0.01647442973689294 | true |
| 25 | 0.9375857338820301 | 0.9390828199863107 | 0.001497086104280565 | 0.013249761375540892 | true |

`"all_ok": true`। (`V1.md`-এ একই মান 4 দশমিক ঘরে: 0.6393/0.6503/0.0110/0.0263, 0.8994/0.9048/0.0053/0.0165, 0.9376/0.9391/0.0015/0.0132।) `V1.md`-এর শিরোনামে "V1 â€”" দেখা যায়। এটা শুধু em-dash (—)-এর encoding দেখানোর সমস্যা (file UTF-8, কিন্তু Windows console cp1252-এ পড়েছে), data-র কোনো সমস্যা নয়।

### (6) Comparison ও ব্যাখ্যা
- তিনটা SNR-এ diff-কে 3σ দিয়ে ভাগ করলে: 0.011/0.0263 ≈ 0.42, 0.0053/0.0165 ≈ 0.32, 0.0015/0.0132 ≈ 0.11। অর্থাৎ **সব পার্থক্য ~1.3σ-এর নিচে**, পরিষ্কার pass। 0.02-এর floor ব্যবহার করতে হয়নি।
- তিনটা diff-ই ধনাত্মক (নতুন bank-এ Teacher সামান্য ভালো)। এটা কি systematic bias? **[নিজস্ব cross-check]** আরেকটা স্বাধীন clean L=3 bank আছে: `phase_d0_snr{0,15}` (E4-এর d=0, অন্য seed)। সেখানে Teacher-এর Pd: SNR 0-এ 0.6298 (frozen 0.6393-এর **চেয়ে কম**), SNR 15-এ 0.9127 (বেশি)। অর্থাৎ নতুন generator-এর দুইটা আলাদা bank frozen bank-এর দুই পাশেই পড়ছে। এটা sampling noise-এর লক্ষণ, systematic bias নয়। SNR 0-এ দুইটা clean bank-এর মধ্যে (0.6503 বনাম 0.6298) পার্থক্য 0.0205। দুইটা স্বাধীন 500-sample estimate-এর পার্থক্যের SE ≈ √2 × 0.00877 ≈ 0.0124, তাই এটা ≈ 1.65σ, স্বাভাবিক।
- **[নিজস্ব যাচাই] SNR definition:** প্রতি entry-র গড় power `E|Y|²` = signal (≈1) + noise (10^(−SNR/10)) হওয়ার কথা। Frozen bank: SNR −10-এ 11.02423, 0-এ 2.019174, 25-এ 1.0042527। নতুন bank: `v1_clean_snr0` 1.9828509, `phase_d0_snr0` 1.9768391, `v1_clean_snr15` 1.0079864, `v1_clean_snr25` 0.98110825। দুই generator-এরই noise level আর signal normalization তত্ত্বের সাথে মেলে। L=3-এ per-sample signal power-এর বড় fluctuation (sum of 3 exponential) আছে, তাই sample mean ±0.02–0.03 এদিক-ওদিক হওয়া স্বাভাবিক। এটা E0a-এর সীমাবদ্ধতা 3 (SNR definition test হয়নি) পূরণ করে।
- **Baseline-দের context:** একই ধরনের clean L=3 bank (`phase_d0`, 15 dB)-এ: Teacher 0.9127, Student 0.9013, FNO 0.8079, DFT-SIC 0.9183, IABR_s0 0.8700 (E4 table)। V1 এই তুলনাগুলোকে বৈধ করে।

### (7) Verdict: ✅ **Success** (coverage সীমিত)
নতুন generator-এর data Teacher-এর চোখে frozen bank থেকে আলাদা করা যায় না।
**Test-এর সীমাবদ্ধতা:**
1. শুধু **L=3, impairment = 0**, আর 3টা SNR। L-sweep bank (L=1…10), separation bank, nuisance bank আর tail SNR (−25, −20, 30, 35)-এর আলাদা generator path (`make_separation_bank`, `make_nuisance_bank`)-এর fidelity V1 যাচাই করে না। Frozen bank-এ শুধু L=3 আছে, তাই অন্য L-এর জন্য কোনো reference-ই নেই।
2. 3σ হিসাবে ধরা হয়েছে যে 6টা component স্বাধীন। আসলে একই sample-এর component-গুলো correlated (dropped sample একসাথে 6টা হারায়, একই noise)। তাই আসল σ বড়। এর মানে test **বেশি কড়া** (conservative) হয়েছে, আর তারপরও pass করেছে। এটা ভালো, তবে উল্লেখ করা দরকার।
3. একটাই instrument (Teacher)। Distribution পার্থক্য Teacher-এর কাছে অদৃশ্য কিন্তু IABR-Net-এর কাছে দৃশ্যমান, এমন পরিস্থিতি সম্ভব, যদিও অসম্ভাব্য।
4. SNR 15 আর 25-এ 0.02-এর floor 3σ-এর চেয়ে বড়। তাই "ok" শর্ত আসলে ±0.02। এবার সব diff 3σ-এর মধ্যে ছিল বলে প্রভাব পড়েনি, কিন্তু অন্য run-এ এই floor একটা বড় পার্থক্যকেও লুকিয়ে ফেলতে পারে।

---

## 8. Figures

**এই group-এর নিজস্ব কোনো figure নেই।** `outputs/figures/`-এর 9টা PNG (E3, E4×2, E5×2, E7×2, E8, training_loss) সব অন্য group-এর। E0a, BANKS, E0b, V0 বা V1-এর জন্য notebook কোনো plot বানায় না। এটা missing file নয়, design অনুযায়ী এমনই। তবে V0 যাচাইয়ের জন্য প্রাসঙ্গিক একটা figure আমি খুলে দেখেছি:

**`E3_snr_sweep.png` (V0 cross-check হিসেবে):** তিনটা panel: Pd (paper-style), Pd (strict), আর RMSE of detected (deg), SNR −10 থেকে 25 dB পর্যন্ত, 11টা model।
- Paper-style panel-এ Teacher-এর (নীল) curve 25 dB-এ ≈0.94-এ শেষ হয়, আর উচ্চ SNR-এ সবার উপরে থাকে। এটা V0-এর 0.9375857… মানের সাথে মেলে। নিম্ন SNR-এ DFT-SIC (লাল) সবার উপরে (−10 dB-এ ≈0.26)।
- RMSE panel-এ Teacher (নীল) আর DFT-SIC (লাল) 25 dB-এ সবচেয়ে নিচে (≈0.24, ≈0.23)। এটা V0-এর rmse 0.2381248907495086-এর সাথে মেলে।
- **একটা figure-এর সমস্যা পেয়েছি:** legend-এ **Teacher আর Abl3_no_SE দুইটারই রং একই নীল**। Matplotlib-এর 10-রঙা default cycle 11তম series-এ আবার শুরু হয়, তাই এমন হয়েছে। ফলে plot দেখে দুইটা নীল line কোনটা কার, আলাদা করা যায় না। (উচ্চ SNR-এ উপরে উঠে যাওয়া নীল line-টা Teacher, আর ≈0.87-এ চ্যাপ্টা হয়ে যাওয়া নীল line-টা Abl3_no_SE, table থেকে মিলিয়ে বলা যায়।) Thesis-এ এই figure ব্যবহারের আগে আলাদা রং বা linestyle দিয়ে আবার plot করা উচিত।

---

## 9. সব মিলিয়ে: এই gate-গুলো পরের সিদ্ধান্তে কী প্রভাব ফেলে

1. **E3–E11-এ IABR-Net যেখানে পিছিয়ে** (যেমন E3 mean paper Pd 0.6909 বনাম Teacher 0.7103 আর DFT-SIC 0.7326, উচ্চ SNR-এ ≈0.87-এ saturation, E8-এ 10° separation-এ 0.6145 বনাম Teacher 0.9322), সেই দুর্বলতা **model-এর নিজের**। Simulator, evaluator বা data-র bug দিয়ে এটা ব্যাখ্যা করা যাবে না। কারণ E0a, V0 আর V1 এই সম্ভাবনাগুলো বাদ দিয়ে দেয়।
2. **IABR-Net যেখানে এগিয়ে** (strict Pd-এ, কারণ কখনো sample drop করে না; FLOPs 157× কম; params 41.5%), সেটাও বাস্তব। E0b দিয়ে মাপা, আর V0 দিয়ে Teacher-এর drop আচরণ নিশ্চিত।
3. Gain/phase robustness (E4–E6) নিয়ে যেকোনো সিদ্ধান্তের ভিত্তি হলো E0a check 3 (injection ঠিক) + V1 (d=0 অবস্থায় distribution ঠিক)।

---

## Success (কোথায় ভালো)
- **E0a:** 11টার মধ্যে 11টা physics check machine precision-এ pass করেছে (~1e-15 বা 0.0)। Vectorized simulator original `generate_channel_v2`-এর সাথে 3.1264164235997256e-15-এ মেলে। Inverse codebook `D_rᴴHD_t` ফেরত দেয় 7.027150473277823e-15-এ, যা IABC-v2-এর গাণিতিক ভিত্তি। 16↔64 conversion bit-exact।
- **BANKS_build:** 56টা bank-এর সবগুলোর sha256 MANIFEST-এর সাথে মেলে (0 mismatch, [নিজস্ব যাচাই])। Spec (n, L, seed, impairment) file-এর ভেতরের তথ্যের সাথে মেলে।
- **E0b:** 195,024 params আর 97,169,941 FLOPs মাপা হয়েছে। Estimate থেকে পার্থক্য (1,920 params) হুবহু BN bookkeeping দিয়ে ব্যাখ্যা করা যায়। Conv FLOPs হাতে করা হিসাবের সাথে হুবহু মেলে (95,207,424)। Ablation-দের param সংখ্যাও consistent।
- **V0:** Teacher mean Pd 0.7102635574471814, registry 0.7103495885388549, diff −8.6e-05 (tolerance-এর 23 গুণ ছোট)। Student (0.6956) আর FNO (0.6235)-ও registry-র সাথে 4 দশমিক ঘরে মেলে।
- **V1:** তিনটা SNR-এই নতুন generator বনাম frozen bank-এর পার্থক্য 3σ-এর অনেক নিচে (0.42, 0.32 আর 0.11 × 3σ)। অন্য স্বাধীন clean bank frozen-এর দুই পাশেই পড়ে, তাই কোনো systematic bias নেই।
- Pipeline robust: session interrupt হওয়ার পর cache আর checkpoint থেকে ঠিকমতো resume হয়েছে। 37টা experiment-এর সবগুলো `done`।

## Failure / Weakness (কোথায় দুর্বল)
এই group-এ কোনো test **fail করেনি**। নিচের দুর্বলতাগুলো test design বা reporting-এর, model-এর নয়:
1. **V0 এবার bit-exact নয়:** আগে "13+ significant figures exact" ছিল, এবার −8.6e-05। ব্যাখ্যা করা যায় (GPU kernel বা oneDNN পার্থক্য), কিন্তু thesis-এ "deterministic/bit-exact" দাবি আর করা যাবে না।
2. **V1-এর coverage সরু:** শুধু L=3, clean, আর 3টা SNR। L≠3, separation আর nuisance generator-এর জন্য কোনো fidelity reference নেই।
3. **E0a-এর কিছু check ছোট sample-এ:** gain/phase range মাত্র একটা sample-এ দেখা হয়েছে। Distribution-এর আকৃতি (uniform কিনা) যাচাই হয়নি। Impaired path-এর original generator-এর সাথে সরাসরি তুলনা নেই। SNR/noise definition আর α-র নিয়ম E0a-তে test হয়নি (V1 আর আমার power check পরোক্ষভাবে cover করে)।
4. **BANKS এই machine-এ regenerate হয়নি** (`build_min` ≈ 0)। Hash শুধু প্রমাণ করে file intact। এই environment-এ generator deterministic কিনা, সেটা প্রমাণ হয়নি।
5. **E0b-এর FLOP breakdown অসম্পূর্ণ:** `profiler_counted_ops`-এ শুধু Conv2D আছে, অথচ total-এ আরও 1,831,445 FLOPs আছে। Total ঠিক, কিন্তু breakdown অস্পষ্ট। E9-এ FNO-এর FFT FLOPs analytic estimate, মাপা নয়।
6. **Figure-এর সমস্যা:** `E3_snr_sweep.png`-এ Teacher আর Abl3_no_SE একই নীল রঙে আঁকা। Reader-এর বিভ্রান্ত হওয়ার ঝুঁকি আছে।
7. **Resume-এর ছোট সমস্যা:** IABR_s2 resume-এর পর data iterator আবার seed থেকে শুরু হয়েছে (প্রথম 2000 step-এর batch আবার দেখেছে)। প্রভাব নগণ্য (s2-ই সেরা seed, 0.6916)।
8. `V1.md`-এর শিরোনামে encoding-এর গোলমাল ("â€”") দেখা যায়। এটা শুধু display-এর সমস্যা (UTF-8 file cp1252-এ পড়া)।

## Improvement ideas (কোথায় উন্নতি দরকার)
1. **V1 বাড়ানো:** original `dldoa_dataset_generation` generator দিয়ে L ∈ {1, 5, 8} আর phase error 5°-এর ছোট reference bank বানাও। তারপর নতুন generator-এর সমতুল্য bank-এর সাথে Teacher আর DFT-SIC দুই instrument দিয়ে তুলনা করো। Per-sample bootstrap দিয়ে σ হিসাব করো (component-independence assumption বাদ দিয়ে), আর 0.02 floor সরিয়ে দাও।
2. **E0a-তে নতুন check যোগ করা:** (ক) 10⁴ sample-এ gain আর phase-এর histogram বা KS test (uniform কিনা); (খ) impaired path: `DG.beamforming_vector_generation_P(error_deg=5)` দিয়ে বানানো F বনাম `simulate(phase_deg_max=5)`, একই RNG-draw সাজিয়ে; (গ) noise variance = 10^(−SNR/10) আর `E|α|² = 1/L`-এর সরাসরি assert; (ঘ) `cells_to_angles` বনাম original `peaks_to_angles`, একই peak-এ।
3. **Determinism report:** V0-এ per-SNR registry মান রাখো আর কোন SNR-এ কতটা component flip হয়েছে সেটা log করো। চাইলে `TF_ENABLE_ONEDNN_OPTS=0` আর `tf.config.experimental.enable_op_determinism()` দিয়ে একবার চালিয়ে দেখো bit-exact ফিরে আসে কিনা। Thesis-এ "reproduces to 8.6e-5" লেখো।
4. **BANKS:** GPU machine-এ একটা bank মুছে আবার generate করে hash মিলিয়ে দেখো (cross-machine determinism test)। MANIFEST-এ generator code-এর hash-ও রাখো।
5. **E0b:** profiler-এর per-op breakdown ঠিক করো (`cmd='op'` output-এ সব op-এর `float_ops` গোনো)। FNO-এর FFT FLOPs একটা empirical counter দিয়ে যাচাই করো। Thesis-এ "195,024 params (41.5% of Teacher), 97.17 MFLOPs (~157× fewer than Teacher)" লেখো।
6. **Figure:** সব multi-model plot-এ 11টার বেশি series থাকলে `tab20` colormap বা linestyle/marker variation ব্যবহার করো। Teacher আর DFT-SIC-কে মোটা বা কালো line দিয়ে reference হিসেবে আলাদা করো।
7. **Resume:** checkpoint-এ data-generator-এর RNG state-ও রাখো (অথবা `seed + start_step` দিয়ে rng বানাও), যাতে resume-এর পর একই batch আবার না আসে।
8. **Metric reporting:** V0 দেখায় Teacher প্রতি SNR-এ 25–88টা sample drop করে, আর IABR-Net কখনো করে না। তাই thesis-এর প্রতিটা table-এ `pd_paper` আর `pd_strict` দুইটাই পাশাপাশি রাখো (notebook এখন তাই করে)। কোনটা headline হবে, সেটা method section-এ স্পষ্ট করে justify করো।
