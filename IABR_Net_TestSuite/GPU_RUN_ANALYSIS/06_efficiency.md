# E9 — Efficiency (Latency, FLOPs, Memory) বিশ্লেষণ রিপোর্ট

**Analyst scope:** শুধু `E9_latency_flops_memory` (params, FLOPs, memory, batch 1/8/32/128 latency, end-to-end latency, vs Teacher/Student/FNO/DFT-SIC)। Context-এর জন্য `E0b_model_instantiation` এবং `E3_in_distribution_snr_sweep`-এর কিছু সংখ্যা ব্যবহার করা হয়েছে, কিন্তু সেগুলো নিজের group নয় — শুধু accuracy-vs-efficiency ব্যাখ্যার জন্য।

---

## ১. এই experiment আসলে কী (সংক্ষেপে)

E9 হলো পুরো test-suite-এর **একমাত্র efficiency experiment**। এখানে ৫টা model — **Teacher** (469,393-param base-paper ResNet, 64 residual block), **Student** (314,513-param magnitude-pruned r=8 ResNet), **FNO** (334,321-param spectral-conv screening model), **IABR-Net** (195,024-param প্রস্তাবিত architecture) এবং **DFT-SIC** (classical, 0 learned param) — এই সবগুলোর জন্য একই GPU-তে, একই harness দিয়ে measure করা হয়েছে:

1. Parameter count
2. FLOPs (forward pass, batch 1)
3. Model-এর on-disk আকার (weights MB) এবং param memory (MB)
4. GPU peak memory (batch 32-তে)
5. NN-only latency — batch 1, 8, 32, 128-এ p50/p99/mean/throughput
6. End-to-end (e2e) latency — NN forward pass + CPU-side peak-decoding (blob detection / local-maxima + angle conversion) একসাথে, p50/p99
7. Accuracy (mean Pd, mean RMSE, E3 sweep থেকে ধার করা) — শুধু efficiency numbers-কে context দেওয়ার জন্য পাশে বসানো হয়েছে।

এটা `outputs/results/E9_latency_flops_memory.json`-এ raw ফর্মে এবং `outputs/tables/E9.md` / `outputs/RESULTS.md`-এ summarized table আকারে আছে।

---

## ২. Theory — এই measurement-এর পেছনের যুক্তি

**(ক) FLOPs কেন params-এর সমানুপাতিক নয়:** IABR-Net-এর trunk পুরোটাই 16×16 resolution-এ চলে (কোনো up/down-sampling নেই, শুধু stage-0-তে ফিক্সড inverse-codebook transform দিয়ে 64×64 observation-কে 16×16 antenna-domain-এ আনা হয়), যেখানে Teacher/Student একটা `Conv2DTranspose(stride=2)` দিয়ে 64×64 → 128×128-এ upsample করে তারপর ৬৪টা residual block চালায়। Convolution-এর FLOPs স্প্যাশিয়াল resolution-এর বর্গের সমানুপাতিক, তাই ৮× কম spatial area (16×16 vs 128×128) মানে conv layer-প্রতি FLOPs-ও বহুগুণ কমে যায় — এটাই `FINAL_RESEARCH_REPORT.md`-তে (§ "Efficiency arithmetic re-derivation") lines 471-473-এ হাতে-করা derivation: "Trunk: 20 convs ... ≈94.4M FLOPs. Matches the report's ≈95.3M" এবং "Teacher ... ≈15.1 GFLOPs ... IABR-Net ≈160× cheaper"।

**(খ) Latency ≠ FLOPs:** GPU-তে wall-clock latency নির্ভর করে শুধু raw FLOP count-এর ওপর না — kernel launch overhead, op-এর সংখ্যা (fusion সম্ভব কিনা), memory-bandwidth-bound vs compute-bound হওয়া, batch size-এ parallelism কাজে লাগানো যাচ্ছে কিনা — এসবের ওপরও নির্ভর করে। তাই এই experiment ইচ্ছা করেই শুধু FLOPs না, real wall-clock (`time.perf_counter`) percentile (p50/p99) latency-ও মাপে batch 1/8/32/128 জুড়ে — যাতে "কম FLOPs মানেই দ্রুত" ধারণাটা empirically verify/falsify করা যায় (নিচে §৭-এ দেখা যাবে এই ধারণাটা আংশিক ভুল প্রমাণিত হয়েছে)।

**(গ) End-to-end vs NN-only latency আলাদা রাখার কারণ:** আউটপুট heatmap থেকে actual angle বের করতে peak-detection দরকার — Teacher/Student/FNO-এর জন্য 256×256 heatmap-এর ওপর blob-detector (OpenCV-শ্রেণির CPU op) চালাতে হয়, কিন্তু IABR-Net-এর coarse heatmap মাত্র 16×16 (Q×P), তাই তার peak-search অনেক সস্তা। `RESEARCH_DESIGN.md` (lines 696-697) আগেই নোট করেছিল যে এই CPU-side decode cost literature-এ কখনো আলাদা করে রিপোর্ট করা হয় না, আর ছোট model-এর ক্ষেত্রে সেটাই dominate করতে পারে — তাই NN-only আর e2e দুটোই আলাদাভাবে মাপা হয়েছে, কখনো merge করা হয়নি।

**(ঘ) Complex-valued FLOPs profiler মিস করে:** TensorFlow-র built-in `tf.compat.v1.profiler`-এ `Einsum`/`Complex`/`Conj` op-এর FLOPs count করা হয় না (শুধু real-valued Conv2D/MatMul ধরে)। IABR-Net-এর stage-0 inverse+forward DFT-codebook transform (16×16×16 complex matmul, ৪টা) এবং FNO-এর `SpectralConv2D`-এর rfft2d/irfft2d + complex mode-mixing — এই দুটোই profiler-এর চোখ এড়িয়ে যায়, তাই একটা manual "supplement" হিসাব যোগ করা হয়েছে (নিচে code অংশে)।

---

## ৩. Code — নোটবুক ঠিক কী করে

সোর্স: `notebooks/IABR_Net_Full_Test_Suite.ipynb`, cell #26 (মার্কার কমেন্ট `# ---------------------------------------------------------------- 21. E9`), function `e9()`, run হয় `R_E9 = run_exp('E9_latency_flops_memory', e9, needs=['E0a_phase0_unit_tests'])` দিয়ে।

- **`time_it(fn, x, iters)`** (cell 26): প্রথমে ৩টা warm-up call (graph tracing/XLA compile ইত্যাদির খরচ বাদ দিতে), তারপর `CFG['latency_iters']`-বার `time.perf_counter()`-দিয়ে টাইম নিয়ে p50, p99, mean রিটার্ন করে।
- **`gpu_peak_mb(fn, x)`**: `tf.config.experimental.reset_memory_stats('GPU:0')` দিয়ে counter রিসেট করে একবার forward pass চালিয়ে `get_memory_info('GPU:0')['peak']` (bytes → MiB) রিপোর্ট করে; `GPUS` না থাকলে `None`।
- **`profiler_flops(fn, specs)`** (cell 13, E0b-তে সংজ্ঞায়িত, E9-এ পুনর্ব্যবহৃত): `convert_variables_to_constants_v2` দিয়ে graph freeze করে `tf.compat.v1.profiler.profile(...cmd='op'...)` চালায়, শুধু op-level counted FLOPs রিটার্ন করে (`Conv2D` mainly)।
- **`complex_einsum_flops_iabr()`**: হার্ডকোড করা supplement — "4 complex 16×16×16 matmuls; complex MAC = 8 real FLOPs" → `4 * 16**3 * 8 = 131,072` FLOPs। এটাই JSON-এর `flops_supplement: 131072`।
- **`fno_fft_flops()`**: FNO-র জন্য rfft2d/irfft2d + mode-mixing supplement, `≈8 blocks × 12 channels × (rFFT+irFFT ≈2.5·N·log2N) + mode-mixing`, JSON-এ `flops_supplement: 111427584` (≈111.4M)।
- **মূল loop**: `BASE` dict (Teacher/Student/FNO) + trained হলে `IABR-Net` (checkpoint `MAIN` থেকে) — প্রতিটার জন্য `specs` বানিয়ে `profiler_flops` চালানো হয়, তারপর `CFG['latency_batches']` (= 1, 8, 32, 128) প্রতিটাতে random input (`tf.random.normal`) দিয়ে `time_it` চালানো হয়, `gpu_peak_mb` batch=32-এ মাপা হয়, weights ফাইলের disk size (`os.path.getsize`) আর param-count×4 bytes (`param_memory_mb`) হিসাব হয়।
- **e2e latency**: প্রথমে একটা `run_one` warm-up call (batch-1 graph trace করানোর জন্য), তারপর `ns = min(CFG['latency_e2e_samples'], len(bank['L']))` স্যাম্পলের ওপর loop করে প্রতিটার জন্য পুরো `predict_heatmap`/`predict_iabr` কল (forward pass + peak-decode + angle-conversion) টাইম করে p50/p99 নেয়।
- **DFT-SIC**: এটা কোনো `tf.function`-এ wrap করা নেই (pure numpy/scipy loop, `classical_dft_sic`), তাই `profiler_flops` চালানো যায়নি — `flops_total: null` — শুধু e2e wall-clock (`classical_dft_sic(Yc[i], L, ndft)` per sample) মাপা হয়েছে; `gpu_peak_mb`/`weights_file_mb`/`nn_ms_b1` — এসব key-ই JSON-এ নেই কারণ এগুলো একটা GPU-graph বা checkpoint-নির্ভর model না।
- **Table generation**: শেষে `E3_in_distribution_snr_sweep.json`-এর `mean_over_snr.pd_paper`/`rmse` টেনে এনে `mean_pd`/`mean_rmse` কলাম বসানো হয় (IABR-Net-এর জন্য `MAIN` seed ব্যবহৃত হয়েছে — সংখ্যাগুলো মিলিয়ে দেখা গেছে এটা `IABR_s0`)। `put_table('E9', ...)` দিয়ে `outputs/tables/E9.md`-এ লেখা হয়।

---

## ৪. কেন এই test করা হয়েছে (hypothesis)

`FINAL_RESEARCH_REPORT.md` অনুযায়ী IABR-Net-এর central efficiency claim ছিল: "≈193,104 parameters (≈41.1% of teacher)" এবং "≈95.3M FLOPs (estimate, unvalidated — no in-repo FLOP tool exists)" — এগুলো ছিল হাতে-করা estimate, কখনো real model instantiate করে measure করা হয়নি। `RESEARCH_DESIGN.md`-এ (lines 293, 302, 337, 346-347, 407-408) এই ফাঁকটাকে explicit করে চিহ্নিত করা হয়েছিল "Gap 1" (কোনো paper params+FLOPs+latency+memory একসাথে রিপোর্ট করে না) এবং "Gap 8" (joint 2D AoA/AoD model-এর standalone inference latency কেউ রিপোর্ট করেনি) হিসেবে, এবং একে "the single cheapest, most defensible, near-zero-risk novelty contribution" বলা হয়েছে (line 618)। তিনটা নির্দিষ্ট hypothesis টেস্ট করার কথা ছিল:

1. **[TO MEASURE]** IABR-Net-এর real param/FLOP measured number কি estimate-এর কাছাকাছি? (§ E0b, কিন্তু E9-ও একই measured params/FLOPs পুনর্ব্যবহার করে)
2. **[HYPOTHESIS, line 361]** "NN-only latency will be substantially lower than the teacher's ... but end-to-end latency ... may narrow that gap considerably — untested।"
3. Gap 8 বন্ধ করা — Teacher/Student/FNO-এর কখনো latency মাপা হয়নি, এই baseline registry-র জন্যও প্রথমবার measure করা।

---

## ৫. Result — আসল সংখ্যা (JSON/table থেকে হুবহু)

### ৫.১ মূল সারণি (`outputs/tables/E9.md` / `outputs/RESULTS.md` lines 512-520)

| model | params | GFLOPs | weights_MB | gpu_peak_MB | nn_ms_b1 | e2e_ms | e2e_p99_ms | mean_pd | mean_rmse |
|---|---|---|---|---|---|---|---|---|---|
| Teacher | 469,393 | 15.203 | 2.80 | 186 | 7.850 | 9.48 | 11.76 | 0.7103 | 0.3759 |
| Student | 314,513 | 10.157 | 2.33 | 170 | 7.092 | 8.76 | 10.51 | 0.6956 | 0.4020 |
| FNO | 334,321 | 0.172 | 1.40 | 188 | 1.251 | 2.64 | 3.72 | 0.6235 | 0.4606 |
| **IABR-Net** | **195,024** | **0.097** | **1.01** | **66** | **3.650** | **4.25** | **4.99** | **0.6909** | **0.3885** |
| DFT-SIC | 0 | — | — | — | — | 8.03 | 8.99 | 0.7326 | 0.3577 |

### ৫.২ Batch-sweep latency (raw JSON, `latency_nn`, ms, p50/p99/mean, throughput/s)

| model | b=1 p50 | b=8 p50 | b=32 p50 | b=128 p50 | b=128 throughput/s |
|---|---|---|---|---|---|
| Teacher | 7.8499 | 10.3452 | 28.6755 | 141.9467 | 901.75 |
| Student | 7.0919 | 10.6752 | 24.4097 | 116.3823 | 1099.82 |
| FNO | 1.2511 | 1.4059 | 4.7403 | 24.3364 | 5259.62 |
| IABR-Net | 3.6498 | 3.4969 | 3.7736 | 3.8797 | 32992.61 |

লক্ষণীয়: IABR-Net-এর batch-1-থেকে-batch-128 latency প্রায় **flat** (3.65 → 3.50 → 3.77 → 3.88 ms) — অর্থাৎ এটা GPU-কে compute-saturate করছে না, latency মূলত kernel-launch/overhead-bound, compute-bound না। বিপরীতে Teacher/Student/FNO-এর latency batch-এর সাথে প্রায় রৈখিকভাবে বাড়ে (Teacher b32→b128: 28.68→141.95ms, ৪.৯৫× — batch ৪× বাড়ায় latency-ও প্রায় ৪×)। ফলে IABR-Net-এর batch-128 throughput (32,992.6/s) Teacher-এর চেয়ে **৩৬.৬×**, Student-এর চেয়ে **৩০.০×**, FNO-এর চেয়ে **৬.৩×** বেশি।

### ৫.৩ GPU peak memory ও model size

| model | gpu_peak_mb (batch32) | weights_file_mb | param_memory_mb |
|---|---|---|---|
| Teacher | 185.85 | 2.797 | 1.791 |
| Student | 169.84 | 2.333 | 1.200 |
| FNO | 187.67 | 1.401 | 1.275 |
| IABR-Net | **65.74** | **1.008** | **0.744** |

IABR-Net-এর GPU peak memory Teacher-এর ~২.৮৩×, Student-এর ~২.৫৮×, FNO-এর ~২.৮৬× কম — এখানে FNO-র কথা লক্ষণীয়: FNO-এর FLOPs (0.172 GFLOPs) খুবই কম হলেও এর peak memory (187.67 MB) Teacher-এর (185.85 MB) সমান — কারণ FNO-ও `Conv2DTranspose(stride=2)` দিয়ে 64×64→128×128-এ upsample করে (`build_fno`, cell 11), তাই activation-memory resolution-নির্ভর, FLOPs-নির্ভর না। IABR-Net সারাক্ষণ 16×16-তেই থাকে বলে এখানে স্পষ্ট সুবিধা পায়।

### ৫.৪ FLOPs breakdown

| model | flops_profiler (counted Conv2D/MatMul) | flops_supplement (complex/FFT, হাতে-হিসাব) | flops_total |
|---|---|---|---|
| Teacher | 15,099,494,400 | 0 | 15,202,591,744* |
| Student | 10,066,329,600 | 0 | 10,156,842,496* |
| FNO | 60,621,376 | 111,427,584 | 172,048,960 |
| IABR-Net | 97,038,869 | 131,072 | 97,169,941 |
| DFT-SIC | — (not a TF graph) | — | null |

(*Teacher/Student-এর `flops_profiler` আর `flops_total` টেবিলে সামান্য ভিন্ন দেখাচ্ছে — raw JSON-এ `flops_profiler: 15202591744`-ই সরাসরি `flops_total`-এর সমান, `counted_ops.Conv2D: 15099494400` হলো শুধু op-breakdown-এর একটা sub-entry, বাকিটা BiasAdd/BatchNorm ইত্যাদি অন্য op থেকে আসছে — JSON-এ এই differences স্পষ্টভাবে আলাদা করে দেখানো নেই, তাই এখানে ambiguity আছে বলে উল্লেখ করা হলো।)

**E0b-এর সাথে cross-check (supporting context, নিজের group না):** `outputs/results/E0b_model_instantiation.json`-এ IABR-Net-এর measured params = 195,024 (trainable 193,744), measured FLOPs = 97,169,941 — এগুলো `FINAL_RESEARCH_REPORT.md`-এর pre-run হাতে-করা estimate ≈193,104 params, ≈95.3M FLOPs-এর সাথে খুব কাছাকাছি মেলে (params ~1.0% বেশি, FLOPs ~2.0% বেশি) — অর্থাৎ design doc-এর manual arithmetic ভালোভাবে verify হয়েছে।

---

## ৬. Comparison

- **Params:** IABR-Net (195,024) সবচেয়ে ছোট — Teacher-এর ৪১.৫% (report-estimate ৪১.১%-এর কাছাকাছি), Student-এর ৬২.০%, FNO-এর ৫৮.৩%।
- **FLOPs:** IABR-Net সবচেয়ে কম (0.097 GFLOPs) — Teacher-এর চেয়ে **~156×** কম (report-এর হাতে-করা estimate ছিল "~160× cheaper", খুবই কাছাকাছি মিলেছে), Student-এর চেয়ে **~104.5×** কম, FNO-এর চেয়েও **~1.77×** কম।
- **GPU memory:** IABR-Net সবচেয়ে কম (66 MB), বাকি সব ~২.৫-২.৯× বেশি।
- **NN-only latency (batch 1):** এখানে IABR-Net *সবচেয়ে দ্রুত না* — FNO (1.251 ms) IABR-Net (3.650 ms)-এর চেয়ে **~২.৯২×** দ্রুত, যদিও FNO-এর FLOPs (0.172G) IABR-Net-এর (0.097G) চেয়ে বেশি। ব্যাখ্যা: IABR-Net pipeline-এ per-element IABC-v2 correction, top-K gather, crop-based refine head — এই ছোট ছোট, sequential, gather/index-heavy op বেশি, যেগুলো batch=1-এ kernel-launch overhead-dominated; FNO তুলনায় বড় কিন্তু কম-সংখ্যক, বেশি fusable conv/FFT op চালায়। batch বাড়ালে (32, 128) IABR-Net-এর flat-latency সুবিধা কাজে লাগে আর FNO-কে ছাড়িয়ে যায়।
- **End-to-end latency:** এখানেও IABR-Net সবচেয়ে দ্রুত *না* — **FNO-ই সার্বিকভাবে সবচেয়ে কম latency-র model (e2e p50 = 2.64 ms)**, IABR-Net (4.25 ms)-এর চেয়ে ~১.৬১× দ্রুত। IABR-Net তবু Teacher (9.48 ms, ~২.২৩× ধীর) এবং Student (8.76 ms, ~২.০৬× ধীর)-এর চেয়ে অনেক দ্রুত, এবং classical DFT-SIC (8.03 ms)-এর চেয়েও প্রায় ~১.৮৯× দ্রুত।
- **DFT-SIC latency বিস্ময়কর দিক:** ০-parameter classical baseline হলেও DFT-SIC-এর e2e latency (8.03 ms) Teacher-এর কাছাকাছি এবং IABR-Net-এর প্রায় দ্বিগুণ — কারণ successive-interference-cancellation একটা sequential, path-by-path CPU loop (`classical_dft_sic`), যা GPU-accelerated neural net-এর চেয়ে ধীর হতে পারে, বিশেষত path count (L) বেশি হলে। "০ parameter = দ্রুত" ধারণাটা এখানে ভুল প্রমাণিত।
- **Decode-cost বিভাজন (NN vs e2e-এর পার্থক্য থেকে আনুমানিক):** Teacher decode ≈ 9.48−7.85 = 1.63 ms; Student ≈ 8.76−7.09 = 1.67 ms; FNO ≈ 2.64−1.25 = 1.39 ms; IABR-Net ≈ 4.25−3.65 = **0.60 ms**। IABR-Net-এর coarse heatmap ১৬×১৬ (256 cell) হওয়ায় তার peak-search/local-maxima ধাপ ২৫৬×২৫৬-heatmap-ভিত্তিক blob-detector-এর চেয়ে ~২.৩-২.৮× সস্তা — এটা design doc-এ explicit hypothesis হিসেবে ছিল না, কিন্তু ফলাফলে স্পষ্ট একটা বাড়তি সুবিধা হিসেবে দেখা গেছে।
- **Accuracy-per-efficiency (E3 mean_over_snr, pd_paper থেকে, context-এর জন্য):**
  - Pd per 100K params: Teacher 0.1513, Student 0.2212, FNO 0.1865, **IABR-Net 0.3543** (সবচেয়ে বেশি accuracy-per-parameter নেটওয়ার্কের মধ্যে)।
  - Pd per GFLOP: Teacher 0.047, Student 0.068, FNO 3.625, **IABR-Net 7.124** (FNO-এর প্রায় দ্বিগুণ accuracy-per-FLOP)।
  - RMSE-এও IABR-Net (0.3885) FNO (0.4606)-এর চেয়ে ভালো, প্রায় Teacher (0.3759)-এর সমান।

---

## ৭. Figures

`outputs/figures/` ডিরেক্টরিতে **E9-এর জন্য কোনো figure তৈরি হয়নি** — সেখানে যা আছে তা হলো `E3_snr_sweep.png`, `E4_phase_snr0.png`, `E4_phase_snr15.png`, `E5_gain_snr0.png`, `E5_gain_snr15.png`, `E7_pathcount_snr0.png`, `E7_pathcount_snr15.png`, `E8_separation.png`, `training_loss.png` — এই তালিকাটা `outputs/RESULTS.md`-এর "## Figures" section-এও (lines 524-532) নিশ্চিত করা হয়েছে। অর্থাৎ efficiency/latency/FLOPs/memory-এর কোনো visual (bar chart, scatter plot ইত্যাদি) এই run-এ তৈরি হয়নি — এটা একটা স্পষ্ট gap, নিচে "Improvement ideas"-এ উল্লেখ করা হলো।

---

## ৮. Verdict

**মিশ্র (Mixed) — কিন্তু সার্বিকভাবে ইতিবাচক দিকে ঝুঁকে থাকা একটা সফলতা।**

- **যেখানে hypothesis পুরোপুরি নিশ্চিত হয়েছে:** params (~২.৪×) ও FLOPs (~১৫৬×) কমানোর claim বাস্তব measurement-এ প্রায় হুবহু মিলেছে design-doc-এর হাতে-করা estimate-এর সাথে। GPU memory ও storage-এও IABR-Net স্পষ্টভাবে সবচেয়ে efficient। batch বাড়ানো হলে (≥32) IABR-Net latency ও throughput-এ decisively জিতে যায় (b128-এ Teacher-এর ৩৬.৬× throughput)।
- **যেখানে hypothesis (line 361, "e2e latency may narrow the gap considerably") *ভুল* প্রমাণিত হয়েছে:** e2e gap আসলে *কমেনি*, বরং Teacher-এর তুলনায় প্রায় একই অনুপাতে (NN-only ratio ২.১৫× vs e2e ratio ২.২৩×) থেকে গেছে — কারণ IABR-Net-এর decode ধাপও (ছোট heatmap-এর কারণে) সস্তা।
- **যেখানে একটা genuinely নেতিবাচক/অপ্রত্যাশিত ফলাফল পাওয়া গেছে:** batch=1 বাস্তব-সময় inference-এর জন্য (যা এই ধরনের pilot-based single-symbol channel-estimation-এর বাস্তব deployment scenario) IABR-Net *সবচেয়ে দ্রুত model না* — FNO NN-only-তে ~২.৯২× এবং end-to-end-এ ~১.৬১× দ্রুত, যদিও FNO-এর FLOPs বেশি এবং accuracy (Pd 0.6235 vs 0.6909, RMSE 0.4606 vs 0.3885) উল্লেখযোগ্যভাবে খারাপ। এটা প্রমাণ করে যে "কম FLOPs/params = কম latency" ধারণা সবসময় সত্য না — architecture-এর op-composition (sequential gather/top-k/crop-refine বনাম বড় কিন্তু simple conv/FFT) latency নির্ধারণে বড় ভূমিকা রাখে।

---

## Success (কোথায় ভালো)

1. Params (~৪১.৫% of Teacher), FLOPs (~156× কম Teacher-এর চেয়ে, ~১.৭৭× কম FNO-এর চেয়েও), GPU memory (66 MB, বাকি সব ~২.৬-২.৯× বেশি), এবং on-disk weights size (1.01 MB, সবচেয়ে ছোট) — সব efficiency axis-এ IABR-Net decisively জিতেছে, এবং pre-run হাতে-করা estimate (§ FINAL_RESEARCH_REPORT.md) প্রায় হুবহু বাস্তব measurement-এর সাথে মিলেছে (params ১%, FLOPs ২% error-এর মধ্যে)।
2. batch≥32-তে IABR-Net-এর latency প্রায় flat থাকে (compute-headroom আছে), ফলে batch-128-এ throughput Teacher-এর ৩৬.৬×, Student-এর ৩০.০×, FNO-এর ৬.৩×।
3. accuracy-per-parameter (0.3543 Pd/100K params) ও accuracy-per-FLOP (7.124 Pd/GFLOP) — দুটোতেই IABR-Net সব neural-net baseline-এর চেয়ে ভালো, অর্থাৎ efficiency শুধু "ছোট মডেল" না, প্রতি-unit-compute বেশি accuracy-ও দিচ্ছে।
4. IABR-Net-এর decode (peak-search) ধাপ (~0.60 ms) অন্য সব model-এর decode (~1.39-1.67 ms)-এর চেয়ে সস্তা — এটা design-এর একটা বাড়তি, আগে explicit-ভাবে hypothesize করা হয়নি এমন সুবিধা।

## Failure / Weakness (কোথায় দুর্বল)

1. **batch=1 real-time inference-এ IABR-Net সবচেয়ে দ্রুত না** — FNO NN-only-তে ~২.৯২× এবং end-to-end-এ ~১.৬১× দ্রুত, যদিও FNO-এর FLOPs বেশি। যেহেতু বাস্তব wireless deployment-এ প্রায়ই single-pilot, batch=1-জাতীয় real-time constraint থাকে, এই ফলাফল IABR-Net-এর "efficiency" claim-কে batch-size-নির্ভর একটা qualification দেয়।
2. **[HYPOTHESIS] "e2e latency may narrow the gap" (RESEARCH_DESIGN.md line 361) নিশ্চিত হয়নি** — বাস্তবে গ্যাপ প্রায় একই থাকে, কখনো সামান্য বাড়েও।
3. Teacher/Student-এর `flops_total` বনাম `flops_profiler`/`counted_ops.Conv2D`-এর মধ্যে JSON-এ সামান্য সংখ্যাগত অসামঞ্জস্য আছে (§৫.৪-এ উল্লেখিত) — কোন op-গুলো বাকি FLOPs-এর জন্য দায়ী তা explicit-ভাবে breakdown করা নেই, তাই পুরোপুরি audit করা যায়নি।
4. DFT-SIC-এর জন্য FLOPs/GPU-memory/weights measure করা হয়নি (শুধু e2e latency) — তাই params-efficiency ছাড়া DFT-SIC-এর সাথে পুরো compute-efficiency তুলনা সম্ভব না; এটা একটা explicit gap, file-এ null/absent হিসেবে দেখা গেছে।
5. **কোনো figure/visualization এই experiment-group-এর জন্য তৈরি হয়নি** — params/FLOPs/latency/memory কোনো bar-chart বা scatter-plot নেই।

## Improvement ideas (কোথায় উন্নতি দরকার)

1. IABR-Net-এর batch=1 latency কেন FNO-এর চেয়ে বেশি তা profile করা দরকার (TF profiler-এর op-level timeline দিয়ে) — সম্ভবত top-K gather/crop-refine head-এর ছোট, sequential op-গুলো fuse/batch করলে batch-1 latency কমতে পারে (যেমন `tf.function(jit_compile=True)`/XLA দিয়ে)।
2. একটা efficiency-vs-accuracy Pareto-front scatter plot (x=GFLOPs বা latency, y=Pd, বাবল-সাইজ=params) তৈরি করা উচিত — এখন এই তুলনা শুধু টেবিল আকারে আছে, visual নেই।
3. Teacher/Student-এর `flops_total` breakdown (Conv2D ছাড়া বাকি op কোথা থেকে আসছে) স্পষ্ট করে দেখানো দরকার, যাতে number-টা পুরোপুরি auditable হয়।
4. DFT-SIC-এর জন্যও একটা comparable compute-cost metric (যেমন CPU FLOPs বা wall-clock CPU-time breakdown per SIC-iteration) যোগ করলে "0 params" claim-টা compute-efficiency দিক থেকেও পুরোপুরি ন্যায্যতা পাবে।
5. deployment-relevant batch-size (batch=1, real-time) আর throughput-relevant batch-size (batch≥32) — এই দুই regime-কে রিপোর্টে স্পষ্টভাবে আলাদা করে present করা উচিত, যাতে "IABR-Net সবচেয়ে efficient" claim-টা ভুলভাবে সব scenario-তে generalize না হয়।
