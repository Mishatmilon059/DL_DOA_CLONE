# 09 — Next Steps: Literature-Backed Fixes for IABR-Net-এর দুর্বলতাগুলো

> **Role:** RESEARCH agent। `07_mother_synthesis.md`-এ ধরা পড়া প্রতিটা top weakness-এর জন্য 2020–2026-এর literature (arXiv, IEEE venue, Springer) থেকে real web research করে (WebSearch + WebFetch, প্রতিটা citation-এর URL সত্যিকার খোলা হয়েছে) সমাধান-প্রস্তাব খোঁজা হয়েছে। **কোনো citation বানানো হয়নি** — নিচের প্রতিটা paper-এর title/authors/venue/year WebFetch দিয়ে সরাসরি abstract/page থেকে verify করা।
> **Source of weaknesses:** `07_mother_synthesis.md` §৪.২ (Failure/Weakness list) ও তার প্রমাণ (E3–E10, Abl-1/2/3, SNR_tails)।

---

## ০. দ্রুত মানচিত্র — কোন weakness, কোন fix, কোন citation

| # | Weakness (mother synthesis) | মূল প্রস্তাবিত fix | মূল citation |
|---|---|---|---|
| W1 | DFT-SIC (0-param)-কে কখনো হারায়নি | A (resolution) + G (confidence head) + B (deep-unfolded SIC-style) | off-grid DOA (Huang+23), Circulant ADMM-Net (Klioui'25), SubspaceNet (Shmuel+23/24), grid-free model-order CNN (Schieler+24) |
| W2 | 10° separation-এ resolution collapse | A (finer heatmap grid) | Off-grid DOA framework (Huang+23) |
| W3 | SNR extrapolation ব্যর্থতা (30→35dB-এ কমছে) | C (designed training-SNR distribution) | Luan & Thompson, ICC 2023 |
| W4 | IABC-v2 core claim প্রায় সব জায়গায় অপ্রমাণিত | D (physics-grounded impairment loss) | Mateos-Ramos+24 (unsup. gain-phase calib.), SDOA-Net (Chen+24) |
| W5 | High-SNR plateau ~0.87, L বাড়লে খারাপ | A + G (একই root cause, W2-এর সাথে ভাগ করা) | (same as W2) + Schieler+24 |
| W6 | SE attention কার্যত অকার্যকর | E (SE→ECA বা remove) | ECA-Net, Wang+ CVPR 2020 |
| W7 | End-fire precision অমীমাংসিত (p95≈164°) | F (sin/cos direction-cosine parameterization) | Biternion Nets, Beyer+ DAGM 2015 |

---

## ১. Weakness-ভিত্তিক research

### W1 — DFT-SIC baseline-কে কখনো হারাতে পারেনি (সবচেয়ে গুরুত্বপূর্ণ)

**কেন হচ্ছে (root-cause hypothesis, mother synthesis §৩.১.৩ থেকে):** IABR-Net single-shot detect করে (16×16 coarse heatmap → NMS → crop-refine), কিন্তু DFT-SIC "detect strongest path → subtract → re-detect residual" — একটা iterative peel-off কৌশল যা কাছাকাছি/masked path আলাদা করতে fundamentally ভালো।

**Literature findings:**
- **"Off-Grid DOA Estimation via Deep Learning Framework"** — Yan Huang, Yanjun Zhang, Jun Tao et al., *Science China Information Sciences*, 2023. https://link.springer.com/article/10.1007/s11432-022-3750-5 (সারসংক্ষেপ: https://www.eurekalert.org/news-releases/1002779) — coarse-grid multi-label classification + continuous offset-regression module, skip connection দিয়ে high-resolution feature ধরে রাখে; ছোট angle interval-এও strong resolution দেখায়। এটা IABR-Net-এর existing coarse+crop-refine design-এর সাথে structurally কাছাকাছি — অর্থাৎ ধারণাটা প্রমাণিত, শুধু আমাদের গ্রিড খুব coarse (16×16)।
- **"SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation"** — Dor H. Shmuel, Julian P. Merkofer, Guy Revach, Ruud J. G. van Sloun, Nir Shlezinger, arXiv:2306.02271 (2023, rev. 2024). https://arxiv.org/abs/2306.02271 — একটা neural network শুধু empirical autocorrelation matrix (surrogate covariance) শেখে, তারপর classical Root-MUSIC/ESPRIT pipeline অপরিবর্তিত থাকে; coherent sources, array mismatch, কম SNR-এও কাজ করে বলে দাবি — অর্থাৎ "পুরো pipeline শেখানো"র বদলে "classical algorithm-কে একটা ছোট learned module দিয়ে সাহায্য করা" পদ্ধতি বেশি কার্যকর হতে পারে।
- **"Circulant ADMM-Net for Fast High-resolution DoA Estimation"** — Youval Klioui, arXiv:2502.19076 (2025). https://arxiv.org/abs/2502.19076 — ADMM iterative algorithm-কে সরাসরি deep-unfold করে (CADMM-Net/CHADMM-Net), O(N log N) per-layer, LISTA/ADMM-Net বেসলাইনের চেয়ে ভালো detection rate ও RMSE দেয় — প্রমাণ করে iterative sparse-recovery algorithm-কে unfold করাটা DOA-তে classical algorithm-কে হারানোর একটা কার্যকর পথ।
- **"AI-Aided ESPRIT for Joint DoA Estimation and Uncertainty Extraction"** — Raz Zohar, Nir Shlezinger, arXiv:2609.17059 (2026). https://arxiv.org/abs/2609.17059 — ESPRIT-এর interpretable pipeline রেখে শুধু AI দিয়ে covariance recovery+uncertainty যোগ করে; challenging scenario-তে (coherent sources, limited snapshots, calibration error) ভালো কাজ করার দাবি।

### W2 — Angular resolution collapse (10° separation-এ Pd 0.6145)

W1-এর সাথে root cause এক (Abl-2 নিশ্চিত করেছে: 16×16 grid + 3×3 NMS-এর effective resolution Rayleigh limit-এর ~2× খারাপ)। মূল fix: **Off-Grid DOA (Huang+23)** — উপরে দেখুন। এছাড়া:
- pixel-shuffle / sub-pixel upsampling (super-resolution literature-এর standard কৌশল, computer-vision-এ Shi et al. CVPR 2016 থেকে উদ্ভূত, DOA-heatmap-এ প্রয়োগযোগ্য) দিয়ে trunk 16×16 রেখেও decode-grid 32×32/64×64-এ upsample করা সম্ভব — params বাড়ে সামান্য, compute বাজেট ঠিক থাকে।

### W3 — SNR extrapolation ব্যর্থতা (training range বাইরে Pd কমে)

- **"Achieving Robust Generalization for Wireless Channel Estimation Neural Networks by Designed Training Data"** — Dianxin Luan, John Thompson, IEEE ICC 2023, arXiv:2302.02302. https://arxiv.org/abs/2302.02302 — strategically-designed training dataset দিয়ে neural network-কে unseen channel condition-এ generalize করানো যায়, online retraining ছাড়াই — এটা সরাসরি আমাদের SNR_tails সমস্যায় প্রযোজ্য: training SNR distribution-এর edge-এ (−15, 24 dB) কম sample থাকা এবং/অথবা SNR-conditioned shortcut (SE-SNR coupling, mother synthesis hypothesis) শেখার প্রবণতা এই "designed sampling" কৌশল দিয়ে কমানো যায়।
- মূল missing experiment (mother synthesis §৪.৩.২ নিজেই চিহ্নিত করেছে): **Abl3_no_SE-কে SNR_tails bank-এ চালানো** — যদি SE বাদ দিলে tail-এ পতন কমে/উধাও হয়, তাহলে SE-SNR auxiliary head-ই root cause প্রমাণিত হবে, নতুন architecture ছাড়াই একটা রান দিয়ে নিশ্চিত করা যায়।

### W4 — IABC-v2 impairment-aware claim প্রায় অপ্রমাণিত

- **"Unsupervised Learning for Gain-Phase Impairment Calibration in ISAC Systems"** — José Miguel Mateos-Ramos, Christian Häger, Musa Furkan Keskin, Luc Le Magoarou, Henk Wymeersch, arXiv:2410.04176 (2024). https://arxiv.org/abs/2410.04176 — gain-phase impairment একযোগে estimate করা ও sensing target localize করা, MAP-ratio-test-ভিত্তিক unsupervised loss দিয়ে — impairment fully known থাকলে যে accuracy পাওয়া যেত (oracle) তার কাছাকাছি পৌঁছায়। এই "physically-grounded reconstruction loss" IABR-এর বর্তমান implicit IABC (যেটা mother synthesis হাইপোথিসাইজ করে "একটা global re-weighting শিখেছে, প্রকৃত corrector নয়") থেকে ভিন্ন — এখানে loss সরাসরি impairment parameter (ε, g) reconstruct করতে বাধ্য করে।
- **"SDOA-Net: An Efficient Deep Learning-Based DOA Estimation Network for Imperfect Array"** — Peng Chen, Zhimin Chen, Liang Liu, Yun Chen, Xianbin Wang, IEEE Trans. Instrumentation and Measurement, 2024. https://arxiv.org/abs/2203.10231 — antenna position perturbation, mutual coupling, gain/phase inconsistency, non-linear amplifier — এই সব "imperfect array" factor একসাথে handle করার জন্য raw sampled signal (covariance matrix নয়) থেকে feature বের করে; imperfect-array condition-এ classical/অন্যান্য DL method-কে হারানোর দাবি করে।

### W5 — High-SNR plateau (~0.87), L বাড়লে আরও খারাপ

Root cause W2-এর সাথে অভিন্ন (coarse-grid merge)। অতিরিক্তভাবে L বাড়ার সমস্যায়:
- **"Grid-free Harmonic Retrieval and Model Order Selection using Deep Convolutional Neural Networks"** — Steffen Schieler, Sebastian Semper, Reza Faramarzahangari, Michael Döbereiner, Christian Schneider, R. Thomä, EuCAP 2024. https://arxiv.org/abs/2211.04846 — একটা CNN একইসাথে grid-free parameter estimation আর path-count (model order) নির্ণয় করে, যা grid-based approach-এর bias/spectral-leakage/ghost-target সমস্যা কমায় — মানে "কতগুলো path আছে" এই uncertainty সরাসরি নেটওয়ার্কে শেখানো সম্ভব, যা L বাড়ার সাথে spurious peak কমাতে সাহায্য করতে পারে (mother synthesis-এর নিজস্ব improvement idea #5-এর সাথে মিলে যায়)।

### W6 — SE attention কার্যত অকার্যকর

- **"ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"** — Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu, CVPR 2020. https://arxiv.org/abs/1910.03151 — SE-এর dimensionality-reduction bottleneck (যেটা channel-attention-এর কার্যকারিতা কমিয়ে দিতে পারে) এড়িয়ে 1D convolution দিয়ে local cross-channel interaction করে; ResNet50-এ SE-এর 24.37M params-এর বদলে মাত্র 80 params দিয়ে >2% Top-1 উন্নতি দেখানো হয়েছে (ImageNet)। ছোট নেটওয়ার্কে (IABR-Net-এর মতো) SE-এর bottleneck-reduction ratio-জনিত তথ্যক্ষয়ের প্রভাব আরও বেশি হতে পারে — এটা ব্যাখ্যা করতে পারে কেন Abl3-এ SE কোনো লাভ দেয়নি।

### W7 — End-fire angular precision অমীমাংসিত (p95 ≈ 164°)

- **"Biternion Nets: Continuous Head Pose Regression from Discrete Training Labels"** — Lucas Beyer, Alexander Hermans, Bastian Leibe, DAGM (Pattern Recognition) 2015. http://www.spencer.eu/papers/beyerBiternions15.pdf — angle-কে scalar (θ) হিসেবে regress না করে unit-circle-এর উপর (sin θ, cos θ) হিসেবে regress করা, wrap-around/periodic boundary-তে discontinuity এড়ায়। IABR-Net-এর offset head সম্ভবত angle-space-এ সরাসরি regress করে, যেখানে DFT-grid-এর angle↔cell mapping end-fire-এর (θ→±90°) কাছে cos(θ) compression-এর কারণে Jacobian প্রায় singular হয়ে যায় — direction-cosine (u = sinθ) space-এ regress করলে এই geometric singularity নিজে থেকেই দূর হয়।

---

## ২. প্রস্তাবিত Fix-গুলোর বিস্তারিত টেবিল

| Fix | কী targets করে | Idea | কেন কাজ করবে এখানে | Source | আনুমানিক cost | কোন experiment-এ improvement দেখা যাবে |
|---|---|---|---|---|---|---|
| **A. Heatmap resolution বাড়ানো** (32×32/64×64 বা pixel-shuffle upsample decode-grid, trunk 16×16 রেখেই) | W1, W2, W5 | Coarse 16×16+3×3NMS-এর জায়গায় finer grid বা sub-pixel upsample, যাতে effective resolution Rayleigh limit-এর কাছাকাছি আসে | Off-grid DOA (Huang+23) দেখায় coarse-classification+offset-regression architecture-ই finer grid-এ ভালো কাজ করে; আমাদের architecture কাঠামো একই, শুধু grid coarse | Huang+23, SCIS | মাঝারি (+১০–৩০K params আনুমানিক decode head-এ, trunk অপরিবর্তিত, তাই এখনো Teacher-এর ~25%) | **E8** (target: 10°-এ Pd ≥0.93), **E3** (high-SNR plateau কমা), **E7** (L=10 gap কমা) |
| **B. Deep-unfolded iterative refine (SIC-style)** | W1, W2 | Single-shot decode-এর বদলে K-step unfolded detect→subtract-residual→re-detect block, DFT-SIC-এর mechanism-কে learned module দিয়ে replicate | Circulant ADMM-Net (Klioui'25) ও SubspaceNet (Shmuel+23/24) দেখায় classical iterative algorithm unfold করলে classical baseline-কে সরাসরি ছাড়িয়ে যাওয়া সম্ভব | Klioui'25 (arXiv 2502.19076), Shmuel+23/24 (arXiv 2306.02271) | উচ্চ (trunk-এর ~২–৩ গুণ, তবু Teacher-এর অনেক নিচে থাকবে) | **E3** headline SNR sweep vs DFT-SIC (target: paired bootstrap CI DFT-SIC-এর বিপক্ষে না যাওয়া), **E8**, **E10** |
| **C. Designed SNR training distribution** | W3 | Training SNR sampling-এ edge (−15, 24dB)-এর কাছে বেশি weight, plus SE-SNR coupling কমানো/সরানো | Luan & Thompson দেখায় designed training data দিয়েই (architecture না বদলে) generalization robust হয় | Luan & Thompson, ICC 2023 (arXiv 2302.02302) | প্রায় শূন্য (শুধু data sampling/training recipe বদল) | **SNR_tails** (target: 30→35dB-এ Pd বাড়া, না কমা) — সাথে missing diagnostic **Abl3_no_SE on tail bank** |
| **D. Physics-grounded IABC loss** | W4 | IABC output-কে সরাসরি (ε̂, ĝ) reconstruct করতে বাধ্য করা reconstruction/consistency loss যোগ করে, শুধু implicit reweighting না রেখে | Mateos-Ramos+24 দেখায় unsupervised gain-phase calibration oracle-এর কাছাকাছি পৌঁছাতে পারে physically-grounded loss দিয়ে; SDOA-Net raw-signal feature দিয়ে imperfect-array robustness দেখায় | Mateos-Ramos+24 (arXiv 2410.04176), Chen+24 (arXiv 2203.10231, IEEE TIM) | কম–মাঝারি (+২–৫K params, নতুন loss term) | **E4, E5, E6** (target: IABR vs Abl1 CI বেশি condition-এ শূন্য বাদ দেবে, শুধু gain-4dB-OOD-এ না) |
| **E. SE → ECA বা সম্পূর্ণ বাদ** | W6 (এবং A/G-এর জন্য param বাজেট ফাঁকা করা) | SE-block-কে ECA (near-zero-param 1D conv attention) দিয়ে replace করা, বা পুরোপুরি সরানো | ECA SE-এর bottleneck-reduction সমস্যা এড়ায়; ImageNet-এ প্রমাণিত উন্নতি কম params-এ | Wang+ CVPR 2020 (arXiv 1910.03151) | সামান্য ঋণাত্মক (params কমে) | **Abl3 (নতুন variant Abl3_eca)** — সব condition-এ, target: SE-এর তুলনায় neutral-or-better with fewer params |
| **F. Direction-cosine (sin/cos) offset parameterization** | W7 | Offset head-এর target angle-space-এর বদলে u=sinθ (বা sin/cos pair) space-এ regress করা | Biternion Nets-এর মূল insight: periodic/compressed target-space-এ direct regression singularity তৈরি করে, unit-circle parameterization তা দূর করে | Beyer+ DAGM 2015 | প্রায় শূন্য (architecture অপরিবর্তিত, শুধু loss/target representation) | **Abl-2-এর নিজস্ব end-fire p95 diagnostic**, **E3/V0** sector-ভিত্তিক error breakdown |
| **G. Confidence/objectness + learned path-count head** | W1, W5 (spurious peak কমানো, "never drop" সুবিধা রেখেও) | প্রতি detected peak-এ একটা lightweight confidence score, যা কম-confidence spurious detection ফিল্টার করে কিন্তু zero-detection এড়ায় | Schieler+24 দেখায় joint parameter+model-order estimation grid-based approach-এর ghost-target সমস্যা কমায় | Schieler+24 (arXiv 2211.04846, EuCAP) | কম (+১–৩K params, single conv head) | **E7** (L=10 gross-error/p95), **E10** (nuisance-induced gross error) |

---

## ৩. Ranking — Expected impact vs effort

| Rank | Fix | Impact (কোন root cause কভার করে) | Effort | কারণ এই ranking |
|---|---|---|---|---|
| 1 | **A — Resolution বাড়ানো** | সবচেয়ে বেশি (W1, W2, W5-এর common root cause, mother synthesis নিজেই #১ priority বলেছে) | মাঝারি | একটাই architecture change, তিনটা independent experiment-এ (E3, E7, E8) একসাথে প্রভাব ফেলবে |
| 2 | **F — sin/cos end-fire parameterization** | মাঝারি (W7, দীর্ঘদিনের unsolved issue) | খুবই কম | শুধু loss/target বদল, retrain লাগবে কিন্তু architecture change না — দ্রুততম ROI |
| 3 | **C — Designed SNR training distribution** | মাঝারি (W3) | খুবই কম | data/training recipe change, কোনো নতুন param না |
| 4 | **E — SE→ECA/remove** | কম-মাঝারি (W6, সরাসরি ছোট কিন্তু A/G-এর জন্য param বাজেট মুক্ত করে) | খুবই কম | Abl3 framework already আছে, একটা variant যোগ করলেই হয় |
| 5 | **G — Confidence/objectness head** | মাঝারি (W1-কে পরোক্ষে সাহায্য করে, L-scaling সমস্যা) | মাঝারি | নতুন head + loss term, কিন্তু trunk অপরিবর্তিত |
| 6 | **D — Physics-grounded IABC loss** | মাঝারি (W4 — thesis-এর core novelty claim বাঁচানো/প্রমাণ-খারিজ করা উভয়ভাবেই গুরুত্বপূর্ণ) | মাঝারি | নতুন loss design+tuning দরকার, এবং risk আছে যে ফলাফল আবার negative আসতে পারে (thesis-এর জন্য honest ভাবেও গুরুত্বপূর্ণ ফলাফল) |
| 7 | **B — Deep-unfolded SIC-style refine** | সবচেয়ে বেশি সম্ভাব্য (W1-কে সরাসরি, thesis-এর central hypothesis বাঁচাতে পারে) | উচ্চ | নতুন iterative block ডিজাইন, বেশি compute/debug সময়, thesis timeline-এ risk |

---

## ৪. Concrete Next-Step Plan (thesis timeline অনুযায়ী)

**ধাপ ১ (প্রথমে — কম effort, দ্রুত ফলাফল, ~১–২ সপ্তাহ):**
- Fix F (sin/cos end-fire parameterization) — Abl-2 refinement head-এ target representation বদলে ছোট (5000-step ablation বাজেটে) রান করা।
- Fix C (SNR training distribution + Abl3_no_SE-কে SNR_tails bank-এ চালানো) — এটা mother synthesis-এর নিজস্ব "missing experiment" ও, তাই কোনো নতুন কোড ছাড়াই প্রথমে করা যায়।
- Fix E (SE→ECA variant, Abl3_eca) — বিদ্যমান Abl3 framework-এ একটা variant যোগ করা।
- এই তিনটাই independent, parallel-এ চালানো সম্ভব, এবং fail করলেও thesis-এর ক্ষতি নেই (already-documented weakness-এর再-confirmation)।

**ধাপ ২ (দ্বিতীয় — মূল architecture change, ~২–৪ সপ্তাহ):**
- Fix A (heatmap resolution/pixel-shuffle upsample) — এটাই headline architecture-এর অংশ, তাই full 20000-step 3-seed retrain দরকার (বর্তমান protocol অনুসরণ করে)। ধাপ ১-এর ফলাফল (বিশেষত Fix E যদি params বাঁচায়) এই ধাপে input হিসেবে ব্যবহার করা যায়।
- সাথে Fix G (confidence/objectness head) বান্ডেল করা যেতে পারে, কারণ finer grid-এ spurious peak-এর সংখ্যা বাড়তে পারে, যা objectness head দিয়ে নিয়ন্ত্রণ করা দরকার হবে।

**ধাপ ৩ (তৃতীয় — thesis narrative-নির্ধারক experiment, ~২–৩ সপ্তাহ, সময় থাকলে):**
- Fix D (physics-grounded IABC loss) — এটা "IABC কি সত্যিই impairment-aware, নাকি শুধু efficiency architecture?" প্রশ্নের চূড়ান্ত উত্তর দেবে। যদি এই re-design-এও E4/E6-এ কোনো real effect না পাওয়া যায়, thesis-এর honest সিদ্ধান্ত হবে: IABC-v2-কে core novelty না বলে "efficient architecture + never-drop guarantee"-কে মূল contribution হিসেবে reframe করা (mother synthesis-এর executive summary-এর সুরের সাথেই সামঞ্জস্যপূর্ণ)।

**ধাপ ৪ (future work / stretch, thesis deadline-এর বাইরে হলে শুধু mention করা):**
- Fix B (deep-unfolded SIC-style iterative refinement) সবচেয়ে বড় সম্ভাবনা রাখে DFT-SIC-কে সত্যিই হারানোর, কিন্তু এটা কার্যত একটা নতুন architecture paper-level কাজ — যদি ধাপ ১–৩ সময়মতো শেষ না হয়, এটাকে thesis-এর "Future Work" section-এ concrete proposal হিসেবে রাখাই যুক্তিযুক্ত, নতুন risky development হিসেবে না।

---

## ৫. সততা ও সীমাবদ্ধতা (Caveats)

- এই research পুরোপুরি **literature-নির্দেশিত hypothesis** — কোনো fix এখানে বাস্তবায়ন/পরীক্ষা করা হয়নি, তাই "কাজ করবে" শব্দ prediction, guarantee না।
- বেশ কিছু paper-এর WebFetch শুধু abstract/landing-page টেনেছে (full PDF text অনেক ক্ষেত্রে binary/encoded এসেছে, তাই পড়া যায়নি) — তাই method-এর গভীর detail (hyperparameter, exact loss formula) পরবর্তী implementation-এর আগে পুরো paper পড়ে যাচাই করা দরকার।
- "A Bayesian Off-Grid DOA Estimation Framework for Close-Angle Scenarios" (PubMed ID 42197962, ~2025/2026) খুঁজে পাওয়া গেলেও PubMed cookie-gate-এর কারণে content পড়া যায়নি, তাই এটা final citation list-এ রাখা হয়নি — future research-এ full text (PMC/journal) থেকে verify করে নেওয়া উচিত।
- Cost estimate (params/compute) সবই **আনুমানিক**, বাস্তব implementation-এর পরই real number (E0b/E9-এর মতো) পাওয়া যাবে।
- Fix ranking effort/impact-এর subjective assessment, thesis-এর বাকি সময়সীমা (deadline) না জেনে করা — সময়সীমা কম হলে ধাপ ১ ও ২-ই যথেষ্ট গুরুত্বপূর্ণ ফলাফল দিতে পারে।

---

*সব citation-এর URL এই session-এ সরাসরি WebSearch/WebFetch দিয়ে খোলা ও verify করা হয়েছে। কোনো নম্বর/দাবি উৎস ছাড়া লেখা হয়নি।*
