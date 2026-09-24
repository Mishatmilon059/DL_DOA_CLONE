# IABR-Net GPU Run Analysis — Group 5: Ablations and Controls
### (Controls_E1_E2_Abl1_Abl3, Abl2_refinement_endfire, Abl6_loss_weight_sweep)

**Analyst এজেন্ট:** Ablations and Controls
**সোর্স:** `IABR_Net_Full_Test_Suite.ipynb` (মার্কডাউন+কোড), `full_run.ipynb` (executed output), `outputs/results/*.json`, `outputs/tables/Controls.md`, `outputs/tables/Abl2.md`, `outputs/tables/Abl6.md`, `outputs/figures/training_loss.png`, `outputs/logs/run_log.txt`, `outputs/logs/config.json`, এবং `ResearchState/FINAL_RESEARCH_REPORT.md` / `RESEARCH_DESIGN.md`।

---

## ০. আগে একটু background (এই রিপোর্ট পড়ার আগে জানা দরকার)

IABR-Net-এর ভেতর তিনটা প্রধান অংশ আছে:

1. **Stage 0 — inverse-codebook transform:** observation `Y = Wᴴ H F + Z`-কে ফিক্সড (non-learned) transform দিয়ে ফেরত antenna-domain-এ আনা হয় (`H̃ = W_ideal⁻¹ Y F_ideal⁻ᴴ` টাইপ), কারণ hardware impairment (per-antenna phase/gain error) antenna-domain-এ একদম diagonal ম্যাট্রিক্স আকারে থাকে — তাই ওখানেই correct করা সবচেয়ে সহজ (physics justification, `RESEARCH_DESIGN.md` লাইন ৪৬৬-৪৬৮)।
2. **IABC-v2 (Impairment-Aware Beamspace Correction, v2):** প্রতিটা antenna-element-এর জন্য একটা শেয়ার্ড-ওয়েট ছোট MLP (context FC `4→8`, per-element MLP `10→16→2`) যেটা প্রতি এলিমেন্টের phase/gain distortion অনুমান করে সংশোধন করে দেয়, তারপর আবার forward codebook দিয়ে beamspace-এ ফিরিয়ে আনা হয়। এই মডিউলটা মাত্র ~২৫০ params-এর।
3. **10-block SE-ResNet trunk** (C=32, circular padding, 16×16 রেজোলিউশনে) + **coarse heatmap** + **7×7 crop refinement heads** (AoA/AoD sub-cell offset) + **SNR auxiliary head** (শুধু SE থাকলে চালু হয়)।

Loss: `L = λ1·L_heat + λ2·L_off + λ3·L_pair + λ4·L_snr`। ডিজাইন ডকুমেন্ট অনুযায়ী (`FINAL_RESEARCH_REPORT.md` §19-২৪):
- `λ3 (pair)` = `L_pair_consist` — IABC-v2-corrected beamspace ম্যাপ আর একই চ্যানেলের **ক্লিন (impairment-ছাড়া) paired observation**-এর মধ্যে MSE। এটা IABC-কে শেখায় ঠিক কোথায় correction করতে হবে।
- `λ4 (snr)` = `L_SE_snr` — SE gate-এর pooled ভ্যালু থেকে true per-sample SNR প্রেডিক্ট করার auxiliary regression loss। এটা টেস্ট করে "attention activation SNR-এর সাথে correlate করে" — এই hypothesis (Candidate A3)।
- Main model-এর ডিফল্ট (`config.json`, `RESEARCH_DESIGN.md`-এর recommended starting weight): `λ1=1.0, λ2=1.0, λ3=0.5, λ4=0.1`।

এই group-এর সব experiment এই তিনটা কম্পোনেন্ট (front-end architecture, capacity, loss weight) নিয়েই — তাই context-টা মনে রাখা জরুরি, কারণ নিচে দেখা যাবে **Controls (E1/E2/Abl1/Abl3) গ্রুপের প্রতিটা variant-এর loss-weight নিজেই আলাদা** (এটা bug না, ইচ্ছাকৃত, কিন্তু গুরুত্বপূর্ণ সূক্ষ্মতা — নিচে বিস্তারিত)।

---

## ১. Controls: E1_capacity_control, E2_dense_frontend, Abl1_no_IABC, Abl3_no_SE

সবগুলো একই `controlled()` ফাংশন দিয়ে ইভ্যালুয়েট হয়েছে (নোটবুক সেল ~২৪, "Controlled comparisons: E1 / E2 / Abl-1 / Abl-3"), একই ৫টা কন্ডিশনে: `E3_frozen` (mean over SNR, ৮০০০-sample frozen bank), `phase_d5_snr15`, `gain_g2_snr15`, `gain_g4_snr15`, `oodphase_d10_snr15` (প্রতিটায় n=৫০০ samples, SNR=15dB)। মেট্রিক = `pd_paper` (paper-style Pd, ১° threshold, Hungarian matching)। `IABR_s0` (মূল মডেল) রেফারেন্স হিসেবে একই টেবিলে আছে।

### VARIANTS ডেফিনিশন (নোটবুক কোড, সেল ১২, লাইন যেখানে `VARIANTS = {...}` আছে):

```python
VARIANTS = {
    'IABR_s0': ('iabc', True, {}, 'full'),
    'E1_capacity_control': ('iabc', True, dict(pair=0.0), 'full'),
    'E2_dense_frontend':   ('dense', True, dict(pair=0.0), 'full'),
    'Abl1_no_IABC':        ('none', True, dict(pair=0.0), 'full'),
    'Abl3_no_SE':          ('iabc', False, dict(snr=0.0), 'full'),
}
```

**একটা জরুরি methodology note (রিপোর্ট শুরুতেই ফ্ল্যাগ করা দরকার):** `RESEARCH_DESIGN.md` (লাইন ২৮২, ৩৫৯) অনুযায়ী মূল ডিজাইনে **E1 (capacity-matched control)** এর কথা ছিল "same-parameter-count module, **no impairment-specific loss**" — অর্থাৎ একটা *ভিন্ন* module যেটা adaptivity ছাড়াই একই সাইজের। কিন্তু বাস্তবে execute হওয়া কোডে E1 আসলে **হুবহু IABR_s0-এর মতোই আর্কিটেকচার** (`front='iabc', use_se=True`, params ১৯৫,০২৪ — main-এর সাথে identical), পার্থক্য শুধু `λ3 (pair loss)=0` বনাম main-এর `0.5`। তাই E1 প্রকৃতপক্ষে "capacity vs adaptivity" প্রশ্নের সরাসরি উত্তর দেয় না (কারণ IABC-v2 module পুরোপুরি ওখানে আছে, adaptivity intact) — বরং এটা টেস্ট করে **`L_pair` loss আদৌ দরকার কিনা**, আর্কিটেকচার একই রেখে। "capacity vs architecture" প্রশ্নের আসল উত্তর আসে **Abl1 বনাম E2**-এর তুলনা থেকে (নিচে দেখুন)। এটা design-doc আর executed-notebook-এর মধ্যে একটা discrepancy — analyst হিসেবে এটা না বলে গেলে ভুল হবে।

### Controls টেবিল (`outputs/tables/Controls.md`, হুবহু):

| model | params | E3_frozen | phase_d5_snr15 | gain_g2_snr15 | gain_g4_snr15 | oodphase_d10_snr15 |
|---|---|---|---|---|---|---|
| IABR_s0 | 195,024 | 0.6909 | 0.8510 | 0.8463 | 0.8033 | 0.8487 |
| E1_capacity_control | 195,024 | 0.6887 | 0.8527 | 0.8477 | 0.8003 | 0.8490 |
| E2_dense_frontend | 457,430 | 0.5704 | 0.7473 | 0.7467 | 0.6527 | 0.7457 |
| Abl1_no_IABC | 194,774 | 0.6871 | 0.8607 | 0.8437 | 0.7790 | 0.8447 |
| Abl3_no_SE | 192,093 | 0.6854 | 0.8537 | 0.8437 | 0.8047 | 0.8443 |

---

### ১.১ E1_capacity_control

**(১) কী টেস্ট করা হয়েছে:** IABR_s0-এর হুবহু একই আর্কিটেকচার (`front='iabc'`, `use_se=True`, params ১৯৫,০২৪ — বিট-বাই-বিট identical) কিন্তু training loss-এ `λ3 (pair)=0.0` (main-এ `0.5`)। সব বাকি hyperparameter, ডেটা, seed(=০) একই।

**(২) Theory:** `L_pair` হলো IABC-v2-corrected output-কে "ক্লিন" (impairment-ছাড়া) paired observation-এর দিকে টেনে নিয়ে যাওয়ার auxiliary supervision। থিওরিটিক্যালি এটা IABC-এর correction-কে দ্রুত ও সঠিকভাবে converge করাতে সাহায্য করার কথা (একটা explicit target দিয়ে), কিন্তু IABC module নিজেই মূল heatmap loss (`L_heat`)-এর মাধ্যমে indirect ভাবে শিখতে পারে (যেহেতু ভুল correction হলে heatmap ভুল হবে, gradient ব্যাক-প্রোপ্যাগেট হবে)। তাই প্রশ্ন হলো: `L_pair` কি সত্যিই দরকারি, নাকি heatmap loss একাই যথেষ্ট?

**(৩) Code:** `IABRNet.call()` রিটার্ন করে `y_corr` (IABC-এর আউটপুট), আর training loop-এ (নোটবুক-এর `train_variant()` ফাংশন, যেটার সোর্স সরাসরি এই এক্সট্র্যাক্টে নেই কিন্তু `TRAIN_*.json`-এর `final` ব্লকে `pair` কম্পোনেন্ট লগ হয়েছে) `pair` loss কম্পোনেন্ট `y_corr`-কে ক্লিন paired sample-এর সাথে MSE করে, তারপর `λ3` দিয়ে weight করে। `VARIANTS` ডিকশনারিতে `dict(pair=0.0)` override দিয়ে এই টার্মটা শূন্য করে দেওয়া হয়েছে (gradient-এ কোনো প্রভাব নেই, কিন্তু loss ভ্যালু হিসেবে এখনো লগ হয়)।

**(৪) কেন করা হয়েছে:** যেহেতু Abl1/E2 (নিচে) স্বাভাবিকভাবেই `λ3=0` নিয়ে ট্রেইন হয়েছে (কারণ ওদের front-end IABC না, তাই `L_pair`-এর জন্য একটা "correct" target তৈরি করার মানে থাকে না ওই architecture-এ), তাই IABR_s0 (main, `λ3=0.5`)-এর সাথে সরাসরি তুলনা করলে "IABC vs no-IABC" পার্থক্যের সাথে "λ3=0.5 vs λ3=0" পার্থক্যও মিশে যেত (confound)। E1 এই confound সরানোর জন্য — main-এর মতোই architecture রেখে শুধু `λ3=0` করে একটা fair "zero-pair-loss" baseline বানানো হয়েছে, যাতে Abl1/E2-এর সাথে apples-to-apples তুলনা করা যায়।

**(৫) Result (JSON, `Controls_E1_E2_Abl1_Abl3.json`):**
- `E1_capacity_control|E3_frozen`: pd_paper = **0.6886666666666668**, rmse = 0.38864019126456767, p95 = 54.025800636615955, p95_endfire = 161.32840713062183
- `phase_d5_snr15`: pd_paper = **0.8526666666666667**, rmse=0.3106860882268247, pairing_error=0.16933333333333334
- `gain_g2_snr15`: pd_paper = **0.8476666666666667**
- `gain_g4_snr15`: pd_paper = **0.8003333333333333**
- `oodphase_d10_snr15`: pd_paper = **0.849**
- Training log (`TRAIN_E1_capacity_control.json`): ২০,০০০ স্টেপ, ২১.৪ মিনিট, final total loss = 0.4134, final `pair` কম্পোনেন্ট (log হয়েছে কিন্তু gradient-এ প্রভাব ফেলেনি) = 0.0625।

**(৬) Comparison:** IABR_s0 (main, `λ3=0.5`)-এর সাথে E1-এর পার্থক্য সব কন্ডিশনে ০.২-০.৩ পার্সেন্টেজ পয়েন্টের মধ্যে (E3: 0.6909→0.6887, −0.22pp; phase_d5: 0.851→0.8527, **+0.17pp**; gain_g2: 0.8463→0.8477, +0.14pp; gain_g4: 0.8033→0.8003, −0.30pp; oodphase: 0.8487→0.849, +0.03pp)। দিকনির্দেশ mixed — কোথাও সামান্য ভালো, কোথাও সামান্য খারাপ, কোনোটাই single-seed noise-এর চেয়ে বড় না।

**(৭) Verdict — **Mixed/Neutral, তবে গুরুত্বপূর্ণ negative finding**: `L_pair` লস (λ3=0.5→0) সরিয়ে দিলে প্র্যাকটিক্যালি কোনো পার্থক্য পড়ে না (সব শর্তে <0.3pp)। অর্থাৎ এই auxiliary loss টার্মটা, যেটা ডিজাইন ডকুমেন্টে গুরুত্বপূর্ণ বলে বিবেচিত হয়েছিল, বাস্তবে IABC-এর কার্যকারিতায় কোনো measurable যোগ করছে না — heatmap loss একাই IABC-কে ঠিকভাবে train করার জন্য যথেষ্ট। এটা একটা simplification opportunity (নিচে "Improvement ideas"-এ)। তবে মনে রাখা দরকার — এটা "capacity control" হিসেবে originally যা টেস্ট করার কথা ছিল তা টেস্ট করেনি (উপরের methodology note দেখুন)।

---

### ১.২ E2_dense_frontend

**(১) কী টেস্ট করা হয়েছে:** IABC-v2-এর বদলে একটা সাধারণ, physics-informed নয় এমন `Dense(512→512)` লেয়ার (২৬২,৬৫৬ params) front-end হিসেবে (`front='dense'`), বাকি trunk অপরিবর্তিত। `λ3(pair)=0` (কারণ dense front-এর আউটপুট কোনো "correction" না, তাই clean-pair target-এর সাথে মেলানোর মানে থাকলেও architecture-ভিত্তিক নয়)। মোট params ৪৫৭,৪৩০ — main/E1/Abl1-এর তুলনায় **প্রায় ২.৩ গুণ বেশি**।

**(২) Theory:** এটা project-এর "parameter-efficiency claim"-কে সরাসরি টেস্ট করে (`RESEARCH_DESIGN.md` লাইন ২৮৩, ২৯৫: "Front-end-swap control... empirically tests the parameter-efficiency claim")। Hypothesis ছিল: physics-informed IABC front (মাত্র ~২৫০ params, কারণ এটা fixed DFT transform + tiny per-element MLP ব্যবহার করে, DFT নিজে যেহেতু cost-free ফিক্সড ট্রান্সফর্ম) একটা generic learned dense layer-এর (যেটা raw input-এর ওপর সরাসরি ১০০০ গুণ বেশি free parameter দিয়ে ঘুরিয়ে-প্যাচিয়ে যা খুশি শিখতে পারে) সমান বা কাছাকাছি Pd দেবে — যদি সেটা প্রমাণ হয়, তাহলে বলা যায় IABC-এর সুবিধা "বেশি capacity"-র জন্য না, বরং সঠিক physics-inductive-bias-এর (antenna-domain-এ diagonal impairment structure বোঝা) জন্য।

**(৩) Code:** `FrontEnd.call()`-এ `mode=='dense'` হলে: `tf.reshape(self.dense(tf.reshape(y_ri, [-1, 2*Q*P])), [-1, Q, P, 2])` — অর্থাৎ পুরো ১৬×১৬×২ ইনপুটকে ফ্ল্যাট করে ৫১২→৫১২ ডেন্স লেয়ারে পাস করে আবার রিশেপ করা হয়, কোনো physics কাঠামো (DFT, antenna-domain, diagonal correction) ছাড়াই।

**(৪) কেন করা হয়েছে:** "capacity vs architecture" প্রশ্নের সবচেয়ে সরাসরি উত্তর এই experiment-ই দেয় — যদি বেশি params-এর dense front ভালো বা সমান করত, তাহলে বলা যেত gain মূলত capacity থেকে আসছে; যদি খারাপ করে (কম-params IABC-এর তুলনায়), তাহলে architecture/inductive-bias-ই আসল কারণ।

**(৫) Result:** `E2_dense_frontend|E3_frozen`: pd_paper = **0.5704166666666667**, rmse=0.469846012123939, p95=74.25887935111895। `phase_d5_snr15`: pd_paper=**0.7473333333333333**, rmse=0.42790487539802297, pairing_error=**0.288** (main-এর তুলনায় ~৭০% বেশি pairing error)। `gain_g2_snr15`: **0.7466666666666667**। `gain_g4_snr15`: **0.6526666666666666** (সবচেয়ে বড় gap দেখা যায় এখানে)। `oodphase_d10_snr15`: **0.7456666666666667**। Training log (`TRAIN_E2_dense_frontend.json`): final loss=0.4388 (বাকি সব variant-এর ~0.41-0.42-এর চেয়ে স্পষ্টভাবে বেশি), আর final `pair` কম্পোনেন্ট **5.0861** — যেটা main/E1/Abl1/Abl3-এর pair-লগ ভ্যালু (0.006-0.06)-এর তুলনায় ৮০-৮০০ গুণ বড়! (এই টার্মের weight যদিও ট্রেনিং-এ 0, কিন্তু raw loss হিসেবে দেখাচ্ছে dense front কতটা "clean paired signal"-এর কাছাকাছি যেতে ব্যর্থ হচ্ছে — একটা সাইড-ইন্ডিকেটর যে এই front-end আসলে impairment বোঝেই না।)

**(৬) Comparison:** E1 (একই params-অর্ডারের competitor, ১৯৫,০২৪ params, `λ3=0` — অর্থাৎ E2-এর apples-to-apples baseline)-এর সাথে তুলনায় E2 **প্রতিটা কন্ডিশনে ১০-১৫ পার্সেন্টেজ পয়েন্ট খারাপ** (E3: −11.8pp, phase_d5: −10.5pp, gain_g2: −10.1pp, gain_g4: **−14.8pp**, oodphase: −10.3pp), যদিও params **২.৩৪ গুণ বেশি**। এটা Abl1 (front='none', ১৯৪,৭৭৪ params, কোনো front-end correction-ই নেই)-এর চেয়েও খারাপ — অর্থাৎ IABC না রেখেও raw input সরাসরি trunk-এ দেওয়া (Abl1) এই dense-front (E2)-এর চেয়ে ভালো ফল দেয়! Teacher (469,393 params, Pd 0.7158 pooled, Abl2.json থেকে) এবং DFT-SIC (0 params, Pd 0.7326)-এর তুলনায়ও E2 বহু পিছিয়ে, যদিও Teacher-এর params-এর সাথে E2 তুলনীয় (457K vs 469K)।

**(৭) Verdict — **Failure (as designed to be, confirms the hypothesis strongly)**: এই ফলাফল parameter-efficiency claim-কে জোরালোভাবে সমর্থন করে — বেশি params (২.৩৪×) দিয়ে physics-structure ছাড়া front-end বানালে **উল্টো খারাপ ফল** হয়, বিশেষ করে বড় impairment-এ (gain_g4: −14.8pp)। এটা স্পষ্ট প্রমাণ যে IABR-Net-এর সুবিধা **capacity নয়, architecture/inductive-bias (DFT + antenna-domain diagonal correction)**-এর কারণে।

---

### ১.৩ Abl1_no_IABC

**(১) কী টেস্ট করা হয়েছে:** Front-end সম্পূর্ণ identity (`front='none'` — `FrontEnd.call()`-এ শুধু `return y_ri`, কোনো correction নেই), বাকি সব (SE trunk, heads) অপরিবর্তিত। Params ১৯৪,৭৭৪ (main-এর চেয়ে মাত্র ২৫০ কম — এটাই IABC-v2 module-এর সাইজ)। `λ3(pair)=0` (যৌক্তিক, কারণ কোনো correction output নেই যা clean pair-এর সাথে মেলানো যায়)।

**(২) Theory:** এটা IABC-v2-এর "pure" ablation — trunk raw impairment-corrupted beamspace ম্যাপ সরাসরি দেখে, কোনো explicit per-element correction ছাড়াই। Hypothesis (`RESEARCH_DESIGN.md` লাইন ৩৫৮): IABC-v2 phase error-এ সাহায্য করবে, কিন্তু "will not generalize to gain error until Phase-0 code exists" — যেহেতু এই রানে Phase-0 gain-error injection কোড ব্যবহৃত হয়েছে (training-এই gain impairment আছে), তাই এখানে gain error-এও IABC-এর উপকার দেখা যাওয়ার কথা যদি hypothesis সত্যি হয়।

**(৩) Code:** `FrontEnd.__init__`-এ `mode='none'` হলে কোনো layer তৈরিই হয় না (শুধু ফিক্সড কনস্ট্যান্ট রাখা হয়), আর `call()`-এ `if self.mode == 'none': return y_ri` — একদম raw input trunk-এ চলে যায়।

**(৪) কেন করা হয়েছে:** এই experiment-ই আসলে (E1-এর নামের বিপরীতে) সবচেয়ে কাছের জিনিস "capacity-matched, no-adaptivity control"-এর — কারণ params প্রায় identical (194,774 ≈ 195,024, ফারাক মাত্র ০.১৩%) কিন্তু IABC-এর adaptive correction নেই। তাই Abl1 vs E1/main তুলনাই "impairment-adaptivity বনাম শুধু capacity" প্রশ্নের সবচেয়ে সরাসরি উত্তর দেয় (design doc-এর original intent অনুযায়ী), E1-এর তুলনায় বেশি সঠিকভাবে।

**(৫) Result:** `Abl1_no_IABC|E3_frozen`: pd_paper=**0.6870833333333333**, rmse=0.3917987670371492, p95_endfire=157.2927321399392 (এই গ্রুপের সব variant-এর মধ্যে সবচেয়ে কম p95_endfire!)। `phase_d5_snr15`: pd_paper=**0.8606666666666667** (এই কন্ডিশনে সবচেয়ে বেশি — এমনকি main-এর চেয়েও বেশি!), p95=**9.585160650834931** (main-এর ১২.৮৪-এর চেয়ে ভালো)। `gain_g2_snr15`: pd_paper=**0.8436666666666667**। `gain_g4_snr15`: pd_paper=**0.779** (এই গ্রুপে সবচেয়ে কম — main/E1-এর চেয়ে ২-২.২pp কম)। `oodphase_d10_snr15`: pd_paper=**0.8446666666666667**।

**(৬) Comparison:** E1 (params-matched, `λ3` একই =0)-এর সাথে তুলনায়: E3_frozen প্রায় সমান (0.6871 vs 0.6887, −0.16pp), **phase_d5-এ Abl1 বরং ভালো** (0.8607 vs 0.8527, +0.80pp), gain_g2-ও প্রায় সমান (0.8437 vs 0.8477, −0.40pp), **gain_g4-এ IABC স্পষ্ট সুবিধা দেয়** (E1 0.8003 vs Abl1 0.779, IABC থাকলে +2.13pp), oodphase প্রায় সমান (0.8447 vs 0.849, −0.43pp)।

**(৭) Verdict — **Mixed, condition-specific success**: IABC-v2 সব জায়গায় সাহায্য করছে না — বরং শুধু সবচেয়ে বড় impairment কন্ডিশনে (gain_g4 = 4dB gain error, যেটা training range-এর 2dB max-এর চেয়ে বাইরে, অর্থাৎ OOD gain) স্পষ্ট (+2.1pp) সুবিধা দেখায়। মাঝারি/in-distribution কন্ডিশনে (phase_d5, gain_g2, oodphase, E3) IABC-এর প্রভাব নগণ্য বা এমনকি সামান্য নেতিবাচক। এটা design doc-এর hypothesis-কে **আংশিকভাবে** সমর্থন করে (লাইন ৩৫৯: "some, not all, of IABR-Net's advantage... is attributable to impairment-adaptivity")। IABC মাত্র ২৫০ params-এ (০.১৩% ওভারহেড) এই targeted OOD-gain সুবিধা দেয় — সস্তা কিন্তু narrow-scope উপকারিতা।

---

### ১.৪ Abl3_no_SE

**(১) কী টেস্ট করা হয়েছে:** ১০-ব্লক trunk-এ Squeeze-and-Excitation (SE, channel attention) গেট সম্পূর্ণ সরিয়ে দেওয়া (`use_se=False`)। যেহেতু SNR auxiliary head SE-gate-এর pooled output (`gms`) ইনপুট হিসেবে নেয়, `use_se=False` হলে সেই ইনপুট architecture-গতভাবেই zero হয়ে যায় (`gm = tf.zeros_like(...)`), তাই `λ4(snr)=0` **বাধ্যতামূলকভাবে** সেট করা হয়েছে (এটা E1/E2/Abl1-এর `pair=0`-এর মতো "choice" না, বরং architecture-এর সরাসরি ফলাফল)। Params ১৯২,০৯৩ (main-এর চেয়ে ২,৯৩১ কম — ১০টা SE ব্লকের squeeze/excite dense লেয়ারের মোট সাইজ)। `λ3(pair)=0.5` (main-এর মতোই অপরিবর্তিত, যেহেতু front এখনো `iabc`)।

**(২) Theory:** SE block প্রতিটা residual block-এর আউটপুটকে channel-wise gate করে (global-average-pooled feature থেকে একটা sigmoid gate শিখে), যা network-কে কোন feature channel গুরুত্বপূর্ণ তা dynamically বাছাই করতে সাহায্য করে। ডিজাইন হাইপোথিসিস (Candidate A3, `RESEARCH_DESIGN.md` লাইন ৪৩, ৪৯): SE attention-এর activation SNR-এর সাথে correlate করবে (তাই SNR aux head-টা কাজে লাগবে), আর SE "প্রায় বিনামূল্যে" (<১% overhead) কিছু robustness যোগ করবে।

**(৩) Code:** `SEResBlock.call()`-এ `if self.use_se:` ব্লকটা স্কিপ হয়ে যায় (কোনো gating হয় না, `gm=zeros`), আর `IABRNet.__init__`-এ `self.snr_head = KL.Dense(1) if use_se else None` — অর্থাৎ SNR head-ই তৈরি হয় না।

**(৪) কেন করা হয়েছে:** SE হলো "almost free" attention — টেস্ট করা দরকার এটা সত্যিই accuracy-তে অবদান রাখছে কিনা, নাকি শুধু params যোগ করছে।

**(৫) Result:** `Abl3_no_SE|E3_frozen`: pd_paper=**0.6853541666666667**। `phase_d5_snr15`: pd_paper=**0.8536666666666667**। `gain_g2_snr15`: pd_paper=**0.8436666666666667**। `gain_g4_snr15`: pd_paper=**0.8046666666666666**। `oodphase_d10_snr15`: pd_paper=**0.8443333333333334**। Training log (`TRAIN_Abl3_no_SE.json`): final loss=0.4192, final logged `snr` কম্পোনেন্ট=**1.5274** (স্থির-ই থাকে পুরো ট্রেনিং জুড়ে ~1.51-1.55, কখনো কমে না — কারণ SNR head prediction সবসময় ০ (SE না থাকায়), তাই এই ভ্যালুটা আসলে target SNR label-এর নিজস্ব variance/scale, শেখার কোনো সাইন নেই। এটা confirm করে যে `λ4=0` বাধ্যতামূলক ছিল, না দিলে dead গ্রেডিয়েন্ট টার্ম থাকত)।

**(৬) Comparison:** IABR_s0 (main, একমাত্র পার্থক্য SE)-এর সাথে তুলনায়: E3_frozen −0.56pp (0.6909→0.6854), phase_d5 **+0.27pp** (0.851→0.8537), gain_g2 −0.26pp, gain_g4 **+0.14pp** (0.8033→0.8047), oodphase −0.43pp। সবগুলো পার্থক্যই ০.৬pp-এর নিচে, দিক মিশ্র (দুইটা শর্তে SE-ছাড়া আসলে সামান্য ভালো!)।

**(৭) Verdict — **Failure (SE কার্যত অকার্যকর, negative-but-useful finding)**: SE attention এই architecture-এ measurable কোনো সুবিধা দিচ্ছে না — সব পার্থক্যই noise-level (<০.৬pp, mixed direction)। ২,৯৩১ params (~১.৫%) এবং SNR auxiliary head-সহ পুরো মেকানিজমটা কার্যত অকেজো এই সেটআপে। "attention activations correlate with SNR" hypothesis-টাও পরোক্ষভাবে untested থেকে যায় (যেহেতু SE ছাড়া SNR head চালানোই যায় না, direct A/B টেস্ট সম্ভব হয়নি — শুধু "SE থাকলে কি সামগ্রিক Pd বাড়ে" টেস্ট হয়েছে, "SE-gate কি সত্যিই SNR predict করে" সেটা আলাদা প্রশ্ন যা এই experiment answer করে না)।

---

## ২. Abl-2 — Refinement decode ablation + end-fire tail (Gap-5 probe)

**(১) কী টেস্ট করা হয়েছে:** IABR-Net-এর angle-decoding pipeline-এর তিনটা ধাপকে আলাদা করে টেস্ট করা: (a) **coarse** — শুধু ১৬×১৬ heatmap গ্রিডের peak cell-এর center নেওয়া, কোনো sub-cell refinement ছাড়া; (b) **parabolic** — classical quadratic peak-interpolation (৩ নেবার cell দিয়ে parabola fit করে sub-pixel peak বের করা); (c) **learned refine** — IABR-Net-এর নিজস্ব ৭×৭ crop-refinement head (AoA/AoD sub-cell offset রিগ্রেশন)। Baseline হিসেবে Teacher, Student, FNO, DFT-SIC-ও একই ৮০০০-sample frozen bank-এ (pooled, সব SNR মিলিয়ে) দেখানো হয়েছে।

**(২) Theory (Gap-5 hypothesis, `RESEARCH_DESIGN.md`):** Project-এর নিজস্ব diagnosed root-cause finding হলো — angle-to-pixel mapping-এর Jacobian `d(π·cosψ)/dψ = −π·sinψ`, যেটা end-fire angle-এর (ψ→0° বা ১৮০°) কাছে শূন্যের দিকে যায়। মানে সেখানে ছোট pixel quantization error-ও বিশাল angular error-এ পরিণত হয় (singularity)। এটাই "Pd ceiling ≈ 0.972" সমস্যার মূল কারণ বলে project-এর আগের ceiling-test ও diffusion-sharpening ablation-এ শনাক্ত হয়েছিল। IABR-Net-এর coarse+crop-refine dual-head ডিজাইনটা এই সমস্যার একটা direct architectural response — coarse grid দিয়ে মোটামুটি লোকেশন বের করে, তারপর একটা ছোট রিজিওনে (৭×৭ crop) continuous regression দিয়ে sub-cell precision আদায় করা, যাতে raw ১৬×১৬ (বা ২৫৬×২৫৬ heatmap-এর) গ্রিড resolution-এর সীমাবদ্ধতা পার হওয়া যায়।

**(৩) Code:** `abl2()` ফাংশন (নোটবুক সেল ২১) — `predict(m, 'E3_frozen', dec)`-এ `dec ∈ {'coarse','parabolic','learned'}` argument দিয়ে decode mode বদলানো হয়, শুধু `IABR_s0` মডেলের জন্য (trunk/weights একই, শুধু post-processing আলাদা)। `metrics()` ফাংশন (Hungarian matching, ১° threshold, original evaluator) প্রতিটার ওপর চালানো হয়।

**(৪) কেন করা হয়েছে:** এটা টেস্ট করে IABR-Net-এর accuracy আসলে **কোথা থেকে আসছে** — heatmap trunk থেকে, নাকি refinement head থেকে। যদি coarse-ই ভালো ফল দেয়, তাহলে refinement head অপ্রয়োজনীয়। যদি coarse খারাপ হয় কিন্তু learned-refine ভালো হয়, তাহলে বোঝা যায় refinement head-ই আসল কাজ করছে এবং Gap-5-এর grid-quantization সমস্যা সমাধানে সেটা কার্যকর।

**(৫) Result (`outputs/tables/Abl2.md`, হুবহু):**

| model | pd_paper | pd_strict | rmse | p50 | p95 | p95_broadside | p95_endfire |
|---|---|---|---|---|---|---|---|
| Teacher | 0.7158 | 0.6727 | 0.3492 | 0.3083 | 77.6013 | 52.4771 | 166.4830 |
| Student | 0.6995 | 0.6469 | 0.3768 | 0.3762 | 82.1367 | 57.9059 | 167.4402 |
| FNO | 0.6246 | 0.6018 | 0.4469 | 0.6059 | 86.5447 | 52.4148 | 167.4574 |
| DFT-SIC | 0.7326 | 0.7326 | 0.3398 | 0.2759 | 71.0252 | 38.0856 | 166.7511 |
| IABR_s0 [coarse] | **0.1396** | 0.1396 | 0.5743 | **3.6746** | 161.6392 | 71.2878 | 173.7988 |
| IABR_s0 [parabolic] | **0.1786** | 0.1786 | 0.5815 | **2.7907** | 121.8036 | 57.3103 | 170.7734 |
| IABR_s0 [learned refine] | **0.6909** | 0.6909 | 0.3680 | 0.3711 | 79.7813 | **43.3719** | **164.2901** |

(JSON precision, `n_dropped`: Teacher=482, Student=602, FNO=293, DFT-SIC=0, সব IABR_s0 variant=0।)

**(৬) Comparison:** coarse-decode Pd মাত্র **0.1396** — অর্থাৎ ৮৬% sample-এ কম্পোনেন্ট-লেভেল angle ভুল! parabolic interpolation সামান্য উন্নতি দেয় (0.1786, +৪.৯pp) কিন্তু তাও ব্যবহারযোগ্য নয়। কিন্তু **learned refine head একলাফে Pd 0.6909-এ নিয়ে যায়** — coarse-এর তুলনায় **+৫৫.১ পার্সেন্টেজ পয়েন্ট**! এটা Teacher (0.7158)-এর ৯৬.৫% এবং DFT-SIC (0.7326)-এর ৯৪.৩%, Student (0.6995)-এর প্রায় সমান, FNO (0.6246)-এর চেয়ে ভালো — সবই মাত্র ১৯৫K params দিয়ে (Teacher-এর ৪১.৬%)। p50 (median absolute error)-ও coarse-এর ৩.৬৭° থেকে learned-refine-এ ০.৩৭°-এ নেমে আসে (~১০× উন্নতি)। p95_broadside-এও learned-refine (৪৩.৩৭°) DFT-SIC (৩৮.০৯°)-এর পরেই সবচেয়ে ভালো, এমনকি Teacher (৫২.৪৮°)-এর চেয়েও ভালো।

**End-fire tail (Gap-5 probe নিজেই):** p95_endfire **সব মডেলেই** বিশালাকার (১৬১°-১৭৪°, প্রায় maximum possible error)! এটা confirm করে end-fire সমস্যাটা model-independent, physics-driven সীমাবদ্ধতা — এমনকি ০-params DFT-SIC-ও (১৬৬.৭৫°) এই সমস্যা থেকে মুক্ত না। তবে আকর্ষণীয়ভাবে, **IABR_s0 [learned refine]-এর p95_endfire (164.29°) এই টেবিলের সবচেয়ে কম** — Teacher, Student, FNO, DFT-SIC সবার চেয়ে সামান্য ভালো, যদিও coarse (173.80°) সবচেয়ে খারাপ।

**(৭) Verdict — **Success (headline finding of this whole group)**: এই ablation প্রমাণ করে IABR-Net-এর প্রায় **পুরো accuracy আসে learned crop-refinement head থেকে**, raw ১৬×১৬ coarse heatmap গ্রিড একা প্রায় অকেজো (Pd ০.১৪)। এটা Gap-5 hypothesis-এর একটা partial confirmation-ও: refinement head grid-quantization সমস্যার বড় অংশ কাটিয়ে উঠতে পারে (broadside-এ চমৎকার), কিন্তু **end-fire-এর Jacobian-singularity সমস্যা কাটাতে পারেনি** (p95_endfire এখনো ~১৬৪°, DFT-SIC-এর কাছাকাছিই, বড় কোনো উন্নতি নেই)। এটাই design ডকুমেন্টের ভবিষ্যদ্বাণীর সাথে মেলে (RESEARCH_DESIGN.md লাইন ৫০১: end-fire probe "Stage-3 differentiable set-prediction" দিয়েই আসলে সমাধান হবে, crop-refine দিয়ে না) — অর্থাৎ IABR-Net-এর বর্তমান refinement approach একটা **আংশিক** সমাধান, root-cause (Jacobian singularity) এখনো অমীমাংসিত।

---

## ৩. Abl-6 — Loss-weight sensitivity sweep (λ3 × λ4 গ্রিড, ১২ রান)

**(১) কী টেস্ট করা হয়েছে:** `λ3 (pair) ∈ {0.0, 0.1, 0.5, 1.0}` × `λ4 (snr) ∈ {0.0, 0.1, 0.3}` — মোট ১২টা কম্বিনেশন, প্রতিটা `front='iabc', use_se=True` (main architecture অপরিবর্তিত), কিন্তু শুধু ৫,০০০ স্টেপ (main-এর ২০,০০০-এর ¼, single seed=০)। Eval: `E3_subset_pd` (frozen bank-এর subset, প্রতি SNR-এ ২০০ sample — main-এর ১০০০-এর ২০%), `phase5_pd` (phase_d5_snr15, n=৫০০), `gain2_pd` (gain_g2_snr15, n=৫০০)।

**(২) Theory:** `FINAL_RESEARCH_REPORT.md` (লাইন ২৫৮, ৩১৫) নিজেই বলে λ3/λ4-এর starting weight ছিল "unvalidated, requires the sensitivity sweep", আর adversarial review-এ "currently to be tuned with zero characterization" হিসেবে flag করা হয়েছিল — অর্থাৎ এই sweep-টা একটা open gap বন্ধ করার জন্যই ডিজাইন করা।

**(৩) Code:** `VARIANTS` ডিকশনারিতে nested loop দিয়ে জেনারেট (`for l3 in CFG['sweep_l3']: for l4 in CFG['sweep_l4']: VARIANTS[f'Abl6_l3_{l3}_l4_{l4}'] = ('iabc', True, dict(pair=l3, snr=l4), 'sweep')`)। `abl6()` ফাংশন (সেল ২৫) প্রতিটা রান ইভ্যালুয়েট করে টেবিল বানায়।

**(৪) কেন করা হয়েছে:** main model-এর ডিফল্ট `λ3=0.5, λ4=0.1` কতটা optimal, বা loss-weight choice আদৌ কতটা sensitive — সেটা যাচাই করার জন্য, যাতে থিসিসে বলা যায় এই ওয়েটগুলো নির্বিচারে বাছা হয়নি।

**(৫) Result (`outputs/tables/Abl6.md`, হুবহু):**

| run | lambda3 | lambda4 | E3_subset_pd | phase5_pd | gain2_pd |
|---|---|---|---|---|---|
| Abl6_l3_0.0_l4_0.0 | 0.0000 | 0.0000 | 0.6580 | 0.8267 | 0.8040 |
| Abl6_l3_0.0_l4_0.1 | 0.0000 | 0.1000 | **0.6601** | 0.8273 | 0.8113 |
| Abl6_l3_0.0_l4_0.3 | 0.0000 | 0.3000 | 0.6576 | 0.8220 | 0.8160 |
| Abl6_l3_0.1_l4_0.0 | 0.1000 | 0.0000 | 0.6559 | **0.8150** | 0.8167 |
| Abl6_l3_0.1_l4_0.1 | 0.1000 | 0.1000 | 0.6581 | 0.8213 | 0.8140 |
| Abl6_l3_0.1_l4_0.3 | 0.1000 | 0.3000 | 0.6579 | **0.8303** | 0.8090 |
| Abl6_l3_0.5_l4_0.0 | 0.5000 | 0.0000 | 0.6572 | 0.8157 | 0.8080 |
| Abl6_l3_0.5_l4_0.1 (= main-এর ওয়েট) | 0.5000 | 0.1000 | 0.6570 | 0.8287 | 0.8060 |
| Abl6_l3_0.5_l4_0.3 | 0.5000 | 0.3000 | 0.6563 | 0.8207 | 0.8097 |
| Abl6_l3_1.0_l4_0.0 | 1.0000 | 0.0000 | 0.6563 | 0.8187 | 0.8080 |
| Abl6_l3_1.0_l4_0.1 | 1.0000 | 0.1000 | **0.6555** (সর্বনিম্ন) | 0.8240 | **0.8180** |
| Abl6_l3_1.0_l4_0.3 | 1.0000 | 0.3000 | 0.6557 | 0.8297 | 0.8110 |

**(৬) Comparison:** ১২টা রানের মধ্যে spread খুবই সরু — `E3_subset_pd` রেঞ্জ ০.৬৫৫৫–০.৬৬০১ (মাত্র **০.৪৬pp** স্প্রেড), `phase5_pd` রেঞ্জ ০.৮১৫০–০.৮৩০৩ (**১.৫৩pp**), `gain2_pd` রেঞ্জ ০.৮০৪০–০.৮১৮০ (**১.৪০pp**)। main-এর নিজস্ব ওয়েট (λ3=0.5, λ4=0.1) কোনো metric-এই সেরা না (E3_subset: ১২-এর মধ্যে ৮ম; phase5: ৪র্থ; gain2: ৭ম) — বরং λ3=0.0, λ4=0.1 (কোনো pair loss নেই!) E3_subset-এ সেরা, আর λ3=1.0, λ4=0.1 gain2-এ সেরা। কোনো single combination তিনটাতেই সেরা না — trade-off আছে, কিন্তু trade-off-এর মাত্রা (~১-১.৫pp) খুবই ছোট, সম্ভবত single-seed/৫০০০-স্টেপ noise-এর সীমার মধ্যেই।

**(৭) Verdict — **Robustness confirmed, কিন্তু optimal-tuning claim দুর্বল**: এই sweep দেখায় loss-weight (λ3, λ4) নির্বাচন এই architecture-এ **খুবই robust** — পুরো ৪×৩ গ্রিড জুড়ে ফলাফল প্রায় অপরিবর্তিত (<১.৫pp)। এটা ভালো খবর (hyperparameter-sensitive না, deployment/reproduction সহজ), কিন্তু এটাও দেখায় main model-এর জন্য বেছে নেওয়া `λ3=0.5, λ4=0.1`-এর কোনো বিশেষ ন্যায্যতা এই sweep-এর ফলাফল থেকে পাওয়া যায় না (কোনো metric-এ এটা সেরা না) — Section ১.১-এর ফলাফলের সাথে মিলিয়ে (λ3=0 করলেও IABR_s0-তে প্রভাব নগণ্য) বলা যায় **λ3 (pair loss) সামগ্রিকভাবেই একটা কম-প্রভাবশালী hyperparameter**, শুধু ২০ পার্সেন্ট-ডেটা/¼-স্টেপ subset-sweep না, পুরো ২০,০০০-স্টেপ full-run-এও (Section ১.১) একই সিদ্ধান্তে পৌঁছানো যায়।

---

## ৪. Figures

এই group-এর (Controls, Abl-2, Abl-6) নিজস্ব কোনো ডেডিকেটেড figure PNG নেই — `outputs/figures/` ডিরেক্টরিতে যা আছে (`E3_snr_sweep.png`, `E4_phase_snr*.png`, `E5_gain_snr*.png`, `E7_pathcount_snr*.png`, `E8_separation.png`) সবই অন্য analyst-group-এর (E3, E4, E5, E7, E8) — সেগুলোয় এই group-এর variant-গুলো নেই (ROBUST_MODELS list-এ শুধু `E1_capacity_control` ও `Abl1_no_IABC` আছে, কিন্তু সেই figure-গুলো নিজেই E4/E5/E7/E8-group-এর অধীনে, এই group-এর নয়)।

তবে **`training_loss.png`** এই group-এর সরাসরি সম্পর্কিত (ট্রেনিং কার্ভ E1/E2/Abl1/Abl3/Abl6-সহ সব variant-এর) — এটা খুলে দেখা হয়েছে:

**যা দেখা যাচ্ছে:**
- **বাম প্যানেল ("Main runs + controls"):** IABR_s0, IABR_s1, IABR_s2 (৩ seed), E1_capacity_control, Abl1_no_IABC, Abl3_no_SE — এই ৬টা কার্ভ প্রায় সম্পূর্ণভাবে একে অপরের ওপর ওভারল্যাপ করে (একসাথে দ্রুত নেমে ~২৫০০ স্টেপের মধ্যে total loss ≈০.৪৫-এ পৌঁছায়, তারপর ধীরে ধীরে ~০.৪১-এ কনভার্জ করে ২০,০০০ স্টেপে)। এর মধ্যে **শুধু E2_dense_frontend (বেগুনি রঙ) স্পষ্টভাবে আলাদা** — অনেক ধীরে নামে (শুরুতে ২.৪৪ থেকে) এবং **final loss ~০.৪৪-০.৪৬-এ প্লাটু করে**, বাকি সব curve-এর চেয়ে স্পষ্টভাবে উঁচুতে, পুরো ট্রেনিং জুড়েই। এটা visually Section ১.২-এর numerical finding-কে নিশ্চিত করে — E2 শুধু final-eval-এ না, পুরো training dynamics-এই পিছিয়ে ছিল। IABR_s2 (সবুজ)-তে ~২০০০ স্টেপের কাছে একটা ছোট স্পাইক দেখা যায় (loss হঠাৎ বেড়ে আবার কমে যায়) — সম্ভবত একটা ক্ষণস্থায়ী training instability, স্ব-সংশোধিত, চূড়ান্ত ফলাফলে প্রভাব ফেলেনি বলেই মনে হয়।
- **ডান প্যানেল ("Abl-6 loss-weight sweep"):** ১২টা কার্ভই একসাথে প্রায় identical trajectory অনুসরণ করে — ৫০০ স্টেপের মধ্যে ~০.৫৫-এ নেমে, ৫০০০ স্টেপে সবাই ~০.৪২-০.৪৩-এ কনভার্জ করে, একে অপরের থেকে আলাদা করা প্রায় অসম্ভব এই স্কেলে। এটা Section ৩-এর "robust to loss-weight" সিদ্ধান্তকে visually সমর্থন করে।

**মানে:** ফিগারটা Section ১.২ (E2 দুর্বল) ও Section ৩ (Abl-6 insensitive) উভয় সিদ্ধান্তের একটা স্বাধীন visual নিশ্চয়তা দেয়।

---

## Success (কোথায় ভালো)

1. **Learned crop-refinement head (Abl-2)** — এই group-এর সবচেয়ে বড় positive finding। Coarse heatmap decode একা Pd ০.১৪, কিন্তু learned refine head যোগ করলে ০.৬৯-এ পৌঁছায় (+৫৫pp) — Teacher-এর ৯৬.৫% accuracy অর্জন করে মাত্র ৪১.৬% params দিয়ে। p95_broadside-এও DFT-SIC-এর পরে সেরা।
2. **E2 (dense-frontend) ব্যর্থতা প্রকারান্তরে সাফল্য** — এটা IABR-Net-এর মূল দাবিকে (parameter-efficient, physics-informed architecture > raw capacity) জোরালোভাবে প্রমাণ করে: ২.৩৪× বেশি params দিয়েও dense front সব শর্তে ১০-১৫pp খারাপ।
3. **Loss-weight robustness (Abl-6)** — পুরো ১২-রান গ্রিডে ফলাফল <১.৫pp-এর মধ্যে থাকা মানে architecture hyperparameter-sensitive না, deployment/reproduction সহজ।
4. **IABC-v2-এর targeted benefit (Abl-1 vs E1)** — সবচেয়ে বড় (OOD) gain-impairment কন্ডিশনে (gain_g4) স্পষ্ট +2.1pp সুবিধা, মাত্র ২৫০ params-এ (০.১৩% ওভারহেড)।

## Failure / Weakness (কোথায় দুর্বল)

1. **E1_capacity_control ডিজাইন-ইনটেন্ট থেকে বিচ্যুত** — মূল ডিজাইনে "same-capacity, no-adaptivity" module-এর কথা ছিল, বাস্তবায়নে এটা "same architecture, শুধু λ3=0" হয়ে গেছে, যা আসল "capacity vs adaptivity" প্রশ্নের সরাসরি উত্তর দেয় না (যদিও Abl1 এই কাজ ভালোভাবে করেছে, তাই সামগ্রিক conclusion এখনো valid, কিন্তু E1-এর নামকরণ ও উদ্দেশ্য নোটবুকে বিভ্রান্তিকর)।
2. **L_pair loss (λ3) কার্যত অকার্যকর** — Section ১.১ ও Section ৩ দুই জায়গাতেই দেখা গেছে λ3 বাড়ানো/কমানো ফলাফলে প্রায় কোনো প্রভাব ফেলে না। এই auxiliary loss-এর জন্য design doc-এ যে গুরুত্ব দেওয়া হয়েছিল, বাস্তব ডেটায় তার সমর্থন মেলেনি।
3. **SE attention (Abl-3) কার্যত অকার্যকর** — সব শর্তে পার্থক্য <০.৬pp, দিক মিশ্র। ২,৯৩১ params ও পুরো SNR-auxiliary-head মেকানিজম কার্যত অব্যবহৃত মূল্য যোগ করছে।
4. **End-fire সমস্যা অমীমাংসিত (Abl-2)** — learned refinement head broadside-এ চমৎকার কাজ করলেও p95_endfire (১৬৪°) এখনো বিশাল, DFT-SIC-এর কাছাকাছিই — Gap-5-এর মূল root-cause (Jacobian singularity) refinement head দিয়ে সমাধান হয়নি।
5. **Abl-6-এর noise floor অস্পষ্ট** — মাত্র single-seed, ৫০০০-স্টেপ, subset-eval (২০০/SNR) হওয়ায় ১-১.৫pp স্প্রেড আসলে "real trade-off" নাকি নিছক run-to-run noise তা নিশ্চিতভাবে বলা যায় না (কোনো error bar/repeat নেই)।

## Improvement ideas (কোথায় উন্নতি দরকার)

1. **প্রকৃত capacity-matched control বানানো:** একটা variant বানানো উচিত যেখানে IABC-v2-এর params-সংখ্যার সমান একটা non-adaptive/random বা linear module (physics-informed না) বসিয়ে টেস্ট করা — তাহলে "capacity vs adaptivity" প্রশ্নের সত্যিকারের নিয়ন্ত্রিত উত্তর মিলবে (design doc-এর original intent অনুযায়ী), Abl1 (params ≈ matched কিন্তু ০ params correction) আর E2 (মিসম্যাচড params, ভিন্ন mechanism)-এর মাঝামাঝি gap পূরণ হবে।
2. **λ3 (pair loss) সরিয়ে দেওয়া বিবেচনা করা** — যেহেতু দুই জায়গাতেই (E1 ও Abl-6) এর প্রভাব নগণ্য প্রমাণিত, থিসিসে এটা simplify করে λ3=0 রেখে main results রিপোর্ট করলে architecture সরল হবে, training কিছুটা দ্রুত হবে, ফলাফলে কোনো ক্ষতি নেই।
3. **SE ব্লক ড্রপ করে param-বাজেট আরও কমানো বিবেচনা করা** — Abl-3-এর ফলাফল অনুযায়ী SE ছাড়া প্রায় একই accuracy পাওয়া যায় ২,৯৩১ কম params-এ (১৯৫,০২৪ → ১৯২,০৯৩, ~১.৫% ছোট) — IABR-Net-কে আরও efficient বানানোর একটা সস্তা সুযোগ।
4. **Abl-6-এ multi-seed বা longer-step repeat যোগ করা** যাতে ১-১.৫pp স্প্রেড real signal নাকি noise তা confidence interval দিয়ে নিশ্চিত করা যায় — বর্তমানে single-seed/৫০০০-স্টেপ ফলাফলের ওপর ভিত্তি করে "robust to loss weight" দাবি করা হচ্ছে, যা সম্ভবত সত্যি কিন্তু statistically আরও শক্তভাবে প্রমাণযোগ্য করা দরকার।
5. **End-fire-নির্দিষ্ট refinement/loss** বিবেচনা করা (যেমন design doc-এ প্রস্তাবিত Stage-3 differentiable set-prediction head, বা end-fire-weighted loss) যেহেতু বর্তমান crop-refine head সেই সমস্যা সমাধান করেনি — এটা Gap-5-এর সবচেয়ে গুরুত্বপূর্ণ open প্রশ্ন যা এই group-এর ডেটা দিয়ে নিশ্চিত হয়েছে।
6. **Abl1_no_IABC-কে "capacity control"-এর ভূমিকায় থিসিসে re-label/re-frame করা** যেহেতু এটা actually E1-এর নাম যা suggests করে তার চেয়ে বেশি সেই ভূমিকা পালন করে (params almost identical, adaptivity ব্যতীত) — এতে থিসিসের bengali/english narrative আরও স্পষ্ট হবে।

---

*এই রিপোর্টে উদ্ধৃত সব সংখ্যা সরাসরি `outputs/results/Controls_E1_E2_Abl1_Abl3.json`, `outputs/results/Abl2_refinement_endfire.json`, `outputs/results/Abl6_loss_weight_sweep.json`, `outputs/results/TRAIN_*.json`, `outputs/tables/Controls.md`, `outputs/tables/Abl2.md`, `outputs/tables/Abl6.md`, এবং `outputs/logs/run_log.txt` / `config.json` থেকে নেওয়া — কোনো সংখ্যা অনুমান/আবিষ্কার করা হয়নি। গ্রুপের বাইরের কোনো experiment/table/figure স্কিপ করা হয়নি বলে যাচাই করা হয়েছে (`outputs/figures/` ডিরেক্টরি লিস্টিং যাচাই করে দেখা গেছে এই group-এর জন্য `training_loss.png` ছাড়া অন্য কোনো ডেডিকেটেড figure নেই)।*
