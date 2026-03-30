# Tréninkový pipeline

Popis celého procesu od WAV banky po profil pro IthacaCoreResonator syntezér.

---

## Přehled

```
WAV banka  (m060-vel3-f44.wav, ...)
    │
    ▼  extract_params.py
params-<banka>.json          fyzikální parametry (B, τ, A₀, f₀, šum, ...)
    │
    ▼  find_outliers.py      (--drop)
params-<banka>.json          bez chybně extrahovaných samplov
    │
    ▼  compute_spectral_eq.py
params-<banka>.json          + LTASE EQ křivka (tělová rezonance)
    │
    ▼  train_instrument_profile.py
params-nn-profile-<banka>.json   NN-smoothed profil pro 88 × 8 pozic
profile-<banka>.pt               váhy surrogate modelu
    │
    ▼  closed_loop_finetune.py   (volitelný)
profile-<banka>.pt               doladěný model (MRSTFT fine-tuning)
```

---

## Rychlý start — wrapper skript

`analysis/train_pipeline.py` spustí celý pipeline jediným příkazem:

```bash
# Celý pipeline s výchozími nastaveními (1800 epoch)
python analysis/train_pipeline.py --bank soundbanks/ks-grand

# Přidat MRSTFT fine-tuning po NN tréninku
python analysis/train_pipeline.py --bank soundbanks/ks-grand --finetune

# Jen výpis příkazů (nic nespustí)
python analysis/train_pipeline.py --bank soundbanks/ks-grand --dry-run

# Od kroku 4 (params JSON existuje, přeskočit extrakci)
python analysis/train_pipeline.py --bank soundbanks/ks-grand --start-at 4

# S jiným out-dir a explicitním pojmenováním banky
python analysis/train_pipeline.py \
    --bank      C:/SoundBanks/IthacaPlayer/ks-grand \
    --bank-name ks-grand \
    --out-dir   analysis/
```

**Výstup** (v `analysis/` pokud nezměněno `--out-dir`):
| Soubor | Obsah |
|---|---|
| `params-ks-grand.json` | naměřené fyzikální parametry |
| `params-nn-profile-ks-grand.json` | NN-smoothed profil pro syntetizér |
| `profile-ks-grand.pt` | váhy surrogate modelu |

---

## Krok 1 — Extrakce fyzikálních parametrů

**Skript:** `analysis/extract_params.py`

```bash
python -u analysis/extract_params.py \
    --bank    soundbanks/ks-grand \
    --out     analysis/params-ks-grand.json \
    --workers 4
```

**Vstup:** adresář se soubory pojmenovanými `m{midi:03d}-vel{vel}-f44.wav`
(např. `m060-vel3-f44.wav`). Velocity index vel = 0–7 (8 vrstev).

**Extrahované parametry** (per nota, per parciál):
- `B` — inharmonicita: $f_k = k \cdot f_0 \cdot \sqrt{1 + B k^2}$
- `tau1`, `tau2`, `a1` — bi-exponenciální útlum: $a_1 e^{-t/\tau_1} + (1-a_1) e^{-t/\tau_2}$
  - Podmínka: $\tau_2 \geq 3 \cdot \tau_1$ (threshold = 3.0×, musí souhlasit s generací)
- `beat_hz` — detune frekvence strun (za vzniku beatingu)
- `A0` — počáteční amplituda parciálu
- `noise` — útok: `attack_tau_s`, `centroid_hz`, `A_noise`
  - `A_noise` je **bezrozměrný poměr** noise_rms / signal_rms (ne absolutní amplituda)
- `duration_s` — délka samplů (používá se pro detekci "kicked" souborů)

**Ochrana před kicked soubory:** pokud délka souboru je < 70 % délky z předchozí extrakce
téhož klíče, soubor se přeskočí. Chrání před náhodně zkrácenými WAV záznamy.

**Log:** `analysis/runtime-logs/extract-params-log.txt`

---

## Krok 2 — Filtrace outlierů

**Skript:** `analysis/find_outliers.py`

```bash
# Vizuální inspekce (bez modifikace)
python analysis/find_outliers.py \
    --params analysis/params-ks-grand.json \
    --z 10 --plot --feature B

# Aplikovat drop in-place (doporučeno po inspekci)
python analysis/find_outliers.py \
    --params analysis/params-ks-grand.json \
    --z 10 --drop
```

**Metoda:** pro každou velocity vrstvu: lokální mediánový smoother (±5 not) → MAD-sigma →
z-score. Outlier = residuál > `--z` sigma.

**Parametry:** `B`, `tau1_mean` (průměr 6 parciálů), `A0_mean`, `f0_ratio`

**Doporučené prahové hodnoty:**
- `--z 10` — grubý filter (jen zjevné chyby jako $\tau_1 > 60$ s)
- `--z 3.5` — jemná inspekce (může odstranit i lehce odlehlé hodnoty)

Po `--drop` jsou chybějící pozice v NN tréninku automaticky interpolovány.

---

## Krok 3 — Spektrální EQ

**Skript:** `analysis/compute_spectral_eq.py`

```bash
python -u analysis/compute_spectral_eq.py \
    --params  analysis/params-ks-grand.json \
    --bank    soundbanks/ks-grand \
    --workers 4
```

Modifikuje `params-ks-grand.json` **in-place** — přidá pole `spectral_eq` ke každému samplu.

**Metoda (LTASE):**
$$H(f) = \frac{\text{LTASE}_\text{orig}(f)}{\text{LTASE}_\text{synth}(f)}$$

Zachycuje tělovou rezonanci nástroje: EQ křivka koriguje rozdíl mezi originálem
a fyzikálním syntezátorem. Používá se v enginu parametrem `eq_strength` (0=bypass, 1=plné EQ).

**Log:** `analysis/runtime-logs/spectral-eq-log.txt`

---

## Krok 4 — Surrogate NN trénink

**Skript:** `analysis/train_instrument_profile.py`

```bash
python -u analysis/train_instrument_profile.py \
    --in         analysis/params-ks-grand.json \
    --out        analysis/params-nn-profile-ks-grand.json \
    --model      analysis/profile.pt \
    --epochs     1800 \
    --eval-every 10
```

### Architektura (faktorizovaná NN)

Každý výstup má dedikovanou pod-síť — fyzikálně motivovaná faktorizace:

| Síť | Vstup | Výstup | Závislost |
|---|---|---|---|
| `B_net` | (midi) | log(B) | vel-nezávislá |
| `tau1_k1_net` | (midi, vel) | log(τ₁) pro k=1 | vel-závislá |
| `tau_ratio_net` | (midi, k) | log(τ_k/τ_k1) | vel-nezávislá |
| `A0_net` | (midi, k, vel) | log(A₀\_ratio) | vel-závislá |
| `df_net` | (midi, k) | log(beat\_hz) | vel-nezávislá |
| `eq_net` | (midi, freq) | gain\_db | vel-nezávislá |
| `noise_net` | (midi, vel) | log(attack\_τ), log(centroid), log(A\_noise) | vel-závislá |
| `biexp_net` | (midi, k, vel) | logit(a₁), log(τ₂/τ₁) | vel-závislá |
| `dur_net` | (midi) | log(duration\_s) | vel-nezávislá |

Všechny pozitivní výstupy trénované v **log-space** (MSE na log hodnotách = geometrická chyba).

### Featurizace vstupů

```python
midi_feat(midi) → [m, sin(π·m), sin(2π·m), sin(4π·m), cos(π·m), cos(2π·m)]
vel_feat(vel)   → [v, v^0.5, v^2]          kde v = vel/7.0
k_feat(k)       → [kn, log(k)/log(90), 1/k]  kde kn = (k-1)/89
```

### Výstup

- Původní naměřené hodnoty jsou zachovány (`_interpolated: false`)
- NN generuje chybějící pozice a dropped noty (`_interpolated: true`)
- Plná tabulka: 88 MIDI × 8 velocity = 704 pozic

### Typická konvergence (ks-grand, 1800 epoch, CPU)

```
epoch  100/1800  loss=0.914  eval=1.617  lr=2.98e-03
epoch  300/1800  loss=0.860  eval=1.259
epoch  600/1800  loss=0.837  eval=1.234
epoch  900/1800  loss=0.794  eval=1.159
epoch 1200/1800  loss=0.775  eval=1.107
epoch 1500/1800  loss=0.766  eval=1.102
epoch 1800/1800  loss=0.764  eval=1.101
```

**Log:** `analysis/runtime-logs/train-profile-log.txt`

---

## Krok 5 — Closed-loop MRSTFT fine-tuning (volitelný)

**Skript:** `analysis/closed_loop_finetune.py`

Dolaďuje váhy NN přímo minimalizací Multi-Resolution STFT Loss (MRSTFT)
oproti originálním WAV nahrávkám. Gradient teče přes diferenciabilní proxy synth.

```bash
# Evaluace (bez update)
python analysis/closed_loop_finetune.py \
    --mode eval \
    --model analysis/profile.pt \
    --bank  soundbanks/ks-grand

# Fine-tuning
python analysis/closed_loop_finetune.py \
    --mode      finetune \
    --model     analysis/profile.pt \
    --bank      soundbanks/ks-grand \
    --epochs    200 \
    --lr        3e-4 \
    --batch-size 8
```

### MRSTFT Loss

Kombinuje 3 FFT velikosti pro pokrytí různých časových škál:

| FFT | Rozlišení | Zachycuje |
|---|---|---|
| 256 | ≈ 6 ms | transient / útok, šumová obálka |
| 1024 | ≈ 23 ms | časové obálky harmonik |
| 4096 | ≈ 93 ms | frekvence parciálů, sustain tvar |

Pro každou škálu: **spectral convergence** + **log-magnitude** chyba.

### Gradient path

```
NN váhy
  └→ B_net → frekvence parciálů (f_k)
  └→ tau1/ratio_net → útlumové obálky
  └→ A0_net → amplitudy parciálů
  └→ biexp_net → bi-exponenciální obálka
  └→ noise_net → šumová obálka
       └→ proxy synth (torch) → MRSTFT vs originál → loss.backward()
```

Gradienty tečou přes 48/80 parametrů (df_net, eq_net, wf_net, dur_net jsou mimo proxy).

### Proxy synth vs C++ render

| | Proxy (torch) | C++ (RenderServer) |
|---|---|---|
| Použití | gradient computation | evaluace, výsledné WAV |
| Výstup | mono | stereo |
| Striny | 1 oscilátor/parciál | 2–3 striny + beating |
| EQ | ne | ano |
| Stereo decorr | ne | ano |
| Rychlost | ~0.5 s/nota (CPU) | ~0.1 s/nota |

**Log:** `analysis/runtime-logs/finetune.log`

---

## Integrace se syntezérem

Po tréninku použij výsledný JSON jako vstup pro IthacaRenderServer:

```bash
# Spuštění render serveru s novým profilem
build/bin/IthacaRenderServer.exe analysis/params-nn-profile-ks-grand.json

# Nebo jako default v server_main.cpp:
# std::string params_json = "analysis/params-nn-profile-ks-grand.json";
```

Ověření level calibrace:

```python
from analysis.render_client import RenderClient
import soundfile as sf, numpy as np, math

with RenderClient("build/bin/IthacaRenderServer.exe",
                  "analysis/params-nn-profile-ks-grand.json") as rc:
    rc.render(midi=60, vel=3, output="exports/test.wav", duration=3.0)

audio, sr = sf.read("exports/test.wav", dtype='float32', always_2d=True)
rms = math.sqrt((np.mean(audio[:,0]**2) + np.mean(audio[:,1]**2)) / 2)
# vel=3: vel_gain = (4/8)^0.7 ≈ 0.616
expected = 0.06 * (4/8)**0.7
print(f"rms={rms:.5f}  expected={expected:.5f}  ratio={rms/expected:.4f}")
# ratio ≈ 1.0000 — garantováno post-render normalizací
```

---

## Výstupní soubory

| Soubor | Role | Generuje |
|---|---|---|
| `analysis/params-<banka>.json` | naměřené fizikální parametry | extract_params.py |
| `analysis/params-nn-profile-<banka>.json` | NN-smoothed profil (88×8) | train_instrument_profile.py |
| `analysis/profile-<banka>.pt` | váhy surrogate modelu | train_instrument_profile.py |
| `analysis/runtime-logs/extract-params-log.txt` | log extrakce | extract_params.py |
| `analysis/runtime-logs/train-profile-log.txt` | log NN tréninku | train_instrument_profile.py |
| `analysis/runtime-logs/finetune.log` | log fine-tuningu | closed_loop_finetune.py |

---

## Časté problémy

### Extrakce selhává na vel=7 treble

Sampley pianissima (vel=7) u vysokých not jsou příliš krátké pro peak detection.
Přeskočí se automaticky — NN pozici interpoluje ze sousedních velocity vrstev.

### tau1 exploduje u forte treble (m100+, vel 5–7)

Záporný exponenciální fit konverguje k τ₁ >> 10 s. Odstraní se v kroku 2 (`--z 10 --drop`).

### B je falešně vysoké u treble (MIDI > 95)

Přesahuje fyzikálně smysluplný rozsah (> 5×10⁻³). Odstraní se outlier filtrem.

### Kicked soubory (WAV zkrácen re-nahráváním)

Extrakce automaticky přeskočí soubory s délkou < 70 % délky z předchozí extrakce
(ochrana `kick_threshold=0.70`). Log obsahuje seznam přeskočených klíčů.

### MRSTFT fine-tuning: OOM

Proxy synth alokuje (K, N) matice na nota. Pro K=60 parciálů a 3 s délku:
~128 MB/nota (bez gradient tape), ~384 MB při backprop.

Řešení: `--k-max 30` (půlí paměť) nebo `--duration 1.5`.

### compute_spectral_eq: jiný `--bank` argument

`extract_params.py` používá `--bank`, `compute_spectral_eq.py` také `--bank`.
`find_outliers.py` bere `--params` (ne `--bank`). Wrapper to řeší automaticky.
