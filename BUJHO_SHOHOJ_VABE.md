# PIA-Net — shob kichu shohoj vabe (architecture + result-er mane)

---

## 1. Ashol problem-ta ki?

Ekta transmitter theke signal ashche receiver-e. Kintu signal shoja ek pothe ashe na --
deyal/building-e dhakka kheye **koyekta alada poth** (path) diye ashe. Amader ber korte hobe:

- **AoD** (Angle of Departure) — signal-ta transmitter theke **kon dike** rowna dilo
- **AoA** (Angle of Arrival) — receiver-e **kon dik theke** eshe porlo

Test-e L = 3 path, tai proti sample-e ber korte hobe **3 x 2 = 6 ta kon (angle)**.

Amader hate ja ache: antenna array 16 x 16 = 256 ta mapa songkha. Ei songkha guло theke
6-ta kon ber kora-i puro kaj.

---

## 2. Paper-er cholakoushol: kon-ke CHOBI banano

Shorashori songkha theke kon ber kora kothin. Tai paper ekta shundor trick kore --
problem-take **image-to-image** kore fela hoy:

```
INPUT  : 16 x 16 chobi   (antenna-r mapa data, real + imaginary = 2 channel)
             |
          [ MODEL ]
             |
OUTPUT : 256 x 256 chobi (kalo background, L-ta ujjwol fôta)
```

Protita **ujjwol fôta = ekta path**. Fôta-r **obosthan (x, y) = shei path-er (AoD, AoA)**.

Tarpor:
1. Chobi-te fôta khujo (blob detection)
2. Fôta-r pixel obosthan -> kon-e convert koro (ekta formula ache)
3. Shesh!

Tomar posted chobi-te eta-i dekhecho: majher chobi = ground truth (3-ta poriskar fôta),
dan pasher-ta = model-er prediction.

---

## 3. Metric duita — ki mape?

### Pd (Detection Probability) — "koyta thik dhorlam?"

Proti sample-e 6-ta kon ber korte hobe. Ekta kon "thik" dhora hoy jodi asholer theke
**1 degree-r modhdhe** pore.

```
Pd = (thik dhora kon-er shonkha) / (moth kon-er shonkha)
```

- Pd = 1.00 -> shob kon thik (nikhut)
- Pd = 0.60 -> 100-tar modhdhe 60-ta thik
- Pd = 0.00 -> ekta-o thik na

**Tomar model: Pd = 0.60** mane 6-ta kon-er modhdhe gore prai 3.6-ta thik dhorche.

> **Ekta guruttopurno niyom:** blob detector jodi 3-tar kom fôta khuje pay, tahole ঐ
> sample-er **6-ta kon-i vul** dhora hoy (4/6 na, 0/6). Kothin niyom, kintu paper-er
> niyom-i eta.

### RMSE — "je gulo thik dhorlam, shegulo koto nikhut?"

Ja **already 1 degree-r modhdhe ache**, shudhu shei gulo niye gôr error (degree-te).

- RMSE = 0.20 -> thik-dhora kon gulo gore 0.2 degree vul (khub nikhut)
- RMSE = 0.50 -> gore 0.5 degree vul

> **Khub joruri:** RMSE **shudhu shofol** gulo niye hishab hoy. Tai eta kokhono 1.0-er
> beshi hote pare na. Ami nije-i ekbar bhul kore ei songkha-take "shomoshto error"
> vebechilam -- oita vul chilo.

### SNR — "signal koto poriskar?"

Signal-er tulonay noise koto kom. dB-te:
- **-10 dB** = onek noise, khub kothin obostha
- **+25 dB** = prai poriskar signal, shohoj obostha

Shob model-i noise beshi hole kharap kore. Tai proti SNR-e alada kore mapa hoy.

---

## 4. "Chance floor 0.5774" -- eta ki?

Dhoro tumi kichu-i na bujhe **eloмelo vabe** kon guess korle. Tobu vagyokrome kichu guess
1 degree-r modhdhe pore jabe. Shei "vagyer" guess gulo [0, 1] degree-r modhdhe **shoman
vabe chorano** thakbe, ar tader RMS hoy:

```
1 / sqrt(3) = 0.5774
```

Tai:
- **RMSE ~ 0.577 -> model ashole kichu shekheni**, shudhu vagyer hit
- **RMSE spashto 0.577-er niche -> model shotti shikhche**

Ei ekta songkha diye-i ami dhorte perechilam je tomar prothom result-ta chilo **puropuri
vagyer**: shob SNR-e RMSE prai flat 0.579 chilo -- hubohu 0.5774!

---

## 5. Architecture (PIA-Net) — bhitore ki hocche?

Duita **alada rasta ek shathe** kaj kore, tarpor mishe jay:

```
        16x16 input (antenna data)
           /              \
    PATH A              PATH B
  (physics)            (learned)
      |                    |
 Antenna-r JANA         Kono formula na,
 formula (steering      shudhu data theke
 vector) diye ekta      pattern shekhe --
 mota-muti onuman       CNN
      |                    |
      \____ mishe jay ____/
                |
        Attention (kon jayga-r shathe
        kon jayga-r shomporko dekhe)
                |
        Upsample: 32x32 -> 256x256
                |
        256x256 heatmap output
```

**Keno duita rasta?**
- **Path A (physics)** interpretable -- amra jani keno eta ja dey ta dey. Kintu eta ekai
  onek dhire shekhe.
- **Path B (learned CNN)** kono physics jane na, shudhu udaharon theke shekhe. Druto
  converge kore.
- **Duita ek shathe** thakle Path A vul korleo model attke jay na.

Ei "physics + learning ek shathe" idea-ta-i tomar contribution.

---

## 6. Tomar ashol result -- shohoj vabe

| SNR | Tomar Pd | ResNet Pd | mane ki |
|---|---|---|---|
| -10 dB (khub noisy) | 0.19 | 0.20 | **prai shoman!** |
| 0 dB | 0.46 | 0.64 | ResNet egiye |
| +25 dB (poriskar) | 0.60 | 0.94 | **ResNet onek egiye** |

**Ek line-e:** tomar model **noise-er modhdhe ResNet-er shoman valo**, kintu **signal
poriskar hole shei shubidha-ta nite pare na**.

Eta ekta **saturation** problem: SNR 15 theke 25 porjonto tomar Pd 0.61 -> 0.60 -- prai
naড়e-i na. ResNet oikhane 0.90 -> 0.94 hoy. Mane noise ar tomar shotru na, onno kichu
attkacche.

---

## 7. Char-ta bug (shohoj upoma diye)

**Bug 1 — training data 428-ta**
> Porikkha-r jonno 5 lakh flashcard porar bodle matro 428-ta pore giyechile. Baki shob
> mukhosto hoye gelo, notun proshne fail.

**Bug 2 — physics formula-r sign ulta**
> Manchitro thik chilo, kintu uttor-dokkhin ulta chapano. Tai "peak" 78 degree vul
> jaygay dekhachhilo.

**Bug 3 — ISTA-r step size 725x boro**
> Ekta chair-er dike hatchile, kintu protita pa 725 gun boro. Chair-e pouchano-r bodle
> onek dure chole jachhile. Ami age ekta "clip" boshiye eta **dheke** felechilam -- rog
> shari hoyni, uposhorgo lukiyechilo.

**Bug 4 — coordinate system alada (shobcheye boro)**
> Physics branch **thik uttor** ber korchilo, kintu shei uttor-take **vul manchitre**
> boshacchilo. GPS coordinate thik, kintu vul projection-e plot kora. Fole tothyo 119
> pixel (~57 degree) dure gie porto -- 20 pixel-er modhdhe portо **0%**.
> Ekhon: 7.8 pixel, 98% thik jaygay.

---

## 8. Ekhon obostha ki

Char-tai bug fix kora, protita-r jonno notebook-e **nijer check cell** ache:

| Check | ki dekhe |
|---|---|
| Part 2.1 | dictionary thik angle-e peak dey kina |
| Part 4.1 | ISTA branch shotti kaj korche kina |
| Part 4.2 | physics-er uttor thik pixel-e pouchacche kina |
| Part 12 (Test A/B) | model shotti shikhche na vagyer hit |

**Bug 4 fix-er por model-take abar train kora hoyni.** Tai ekhonkar 0.60 songkha-ta
**purano** -- oita emon ekta model-er, jar physics branch 57 degree vul chilo.

---

## 9. Supervisor-ke ki bolbe (shot o rokkhoniyo)

> "Physics-informed deep-unfolded architecture implement korechi. Char-ta implementation
> bug (training data starvation, dictionary sign error, ISTA step divergence, coordinate
> mismatch) khuje ber kore fix korechi -- protita-r jonno alada verification test ache.
>
> Ekhon model **jachaijogyo vabe shikhche**: SNR-er shathe Pd-r correlation 0.92, RMSE
> chance floor (0.577)-er niche.
>
> **Kom SNR-e ResNet-er 93% porjonto pouchachi ar RMSE prai shoman (1.03x)** -- physics
> prior oikhane-i shobcheye beshi kaje lagche. Beshi SNR-e ekhono gap ache; sheshtom bug
> (coordinate mismatch) fix korar por-er result ekhono ashe ni."

**Ja bolbe NA:** "amar model baseline-ke hariye dey" -- eta ekhono proman hoyni.
