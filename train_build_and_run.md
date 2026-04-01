# IthacaCoreResonator — trénink, build a spuštění

Kompletní průvodce od WAV banky po zvuk ze syntetizátoru.

---

## Obsah

1. [Předpoklady](#1-předpoklady)
2. [Pipeline trénování — přehled](#2-pipeline-trénování--přehled)
3. [Krok 1 — Extrakce fyzikálních parametrů](#3-krok-1--extrakce-fyzikálních-parametrů)
4. [Krok 2 — Filtrace outlierů](#4-krok-2--filtrace-outlierů)
5. [Krok 3 — Spektrální EQ](#5-krok-3--spektrální-eq)
6. [Krok 4 — Surrogate NN trénink](#6-krok-4--surrogate-nn-trénink)
7. [Krok 5a — MRSTFT fine-tuning](#7-krok-5a--mrstft-fine-tuning)
8. [Krok 5b — Globální SynthConfig optimalizace](#8-krok-5b--globální-synthconfig-optimalizace)
9. [Krok 6 — Regenerace JSON z fine-tuned modelu](#9-krok-6--regenerace-json-z-fine-tuned-modelu)
10. [Quickstart — train_pipeline.py](#10-quickstart--train_pipelinepy)
11. [Export do ICR JSON formátu](#11-export-do-icr-json-formátu)
12. [Parametrické soubory ICR — struktura a klíče](#12-parametrické-soubory-icr--struktura-a-klíče)
13. [Build přes CMake](#13-build-přes-cmake)
14. [Spuštění](#14-spuštění)
15. [Časté problémy](#15-časté-problémy)

---

## 1. Předpoklady

### Python prostředí

```bash
pip install -r analysis/requirements.txt
```

Hlavní závislosti: `torch`, `numpy`, `scipy >= 1.10`, `soundfile`, `matplotlib`.

### WAV banka

Soubory pojmenované `m{midi:03d}-vel{vel}-f44.wav` (vel = 0–7, MIDI 21–108).  
Příklad: `m060-vel3-f44.wav` = nota C4, velocity band 3.

Standardní umístění: `C:/SoundBanks/IthacaPlayer/ks-grand/`

---

## 2. Pipeline trénování — přehled

```
WAV banka  (m060-vel3-f44.wav, ...)
    │
    ▼  analysis/extract_params.py
params-<banka>.json              fyzikální parametry per nota × parcial
    │
    ▼  analysis/find_outliers.py --drop
params-<banka>.json              bez chybně extrahovaných vzorků
    │
    ▼  analysis/compute_spectral_eq.py
params-<banka>.json              + LTASE spektrální EQ křivka per nota
    │
    ▼  analysis/train_instrument_profile.py
profile-<banka>-nn.pt            váhy surrogate NN modelu (99 214 params)
params-<banka>-nn.json           NN-smoothed profil 88×8 pozic
    │
    ▼  analysis/closed_loop_finetune.py --mode finetune   (volitelné)
profile-<banka>-ft.pt            doladěný model po MRSTFT fine-tuningu
    │
    ▼  analysis/train_instrument_profile.py --epochs 0
soundbanks/params-<banka>-ft.json   ICR-ready JSON pro GUI/CLI/RenderServer
    │
    ▼  analysis/closed_loop_finetune.py --mode global     (volitelné)
params-<banka>-ft.synth_config.json   beat_scale + noise_level optimalizace
    │
    ▼  analysis/export_soundbank_params.py                (alternativní cesta)
analysis/params-piano-soundbank.json  přímý export soundbank → ICR s EQ fitováním
```

**Konvence souborů:**

| Fáze | Model `.pt` | Params `.json` pro ICR |
|---|---|---|
| NN trénink | `profile-<banka>-nn.pt` | `params-<banka>-nn.json` |
| MRSTFT fine-tuning | `profile-<banka>-ft.pt` | `soundbanks/params-<banka>-ft.json` |
| Globální opt. | — | `params-<banka>-ft.synth_config.json` |

---

## 3. Krok 1 — Extrakce fyzikálních parametrů

**Skript:** `analysis/extract_params.py`

```bash
python -u analysis/extract_params.py \
    --bank    "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --out     analysis/params-ks-grand.json \
    --workers 4
```

| Argument | Výchozí | Popis |
|---|---|---|
| `--bank` | — | Adresář s WAV soubory |
| `--out` | — | Výstupní JSON |
| `--workers` | `4` | Počet paralelních procesů |

**Co se extrahuje** (per nota, per parciál):

| Klíč | Jednotka | Popis |
|---|---|---|
| `B` | — | Inharmonicita: `f_k = k·f₀·√(1+B·k²)` |
| `f0_hz` | Hz | Základní frekvence |
| `tau1` | s | Rychlý exponenciální pokles |
| `tau2` | s | Pomalý exponenciální pokles (≥ tau1) |
| `a1` | [0,1] | Váha rychlé složky bi-exp obalky |
| `beat_hz` | Hz | Deladění strun (příčina beatingu) |
| `A0` | — | Počáteční amplituda parciálu |
| `noise.attack_tau_s` | s | Decay time šumové obálky |
| `noise.A_noise` | — | Amplituda šumu (poměr k signálu) |
| `duration_s` | s | Délka vzorku |

**Log:** `analysis/runtime-logs/extract-params-log.txt`

---

## 4. Krok 2 — Filtrace outlierů

**Skript:** `analysis/find_outliers.py`

```bash
# Vizuální inspekce — zobrazí histogram s outlieri
python analysis/find_outliers.py \
    --params analysis/params-ks-grand.json \
    --z 10 --plot --feature B

# Drop outlierů in-place
python analysis/find_outliers.py \
    --params analysis/params-ks-grand.json \
    --z 10 --drop
```

| Argument | Popis |
|---|---|
| `--params` | Vstupní JSON (upravuje in-place) |
| `--z` | Z-score práh (10 = hrubá filtrace, 3.5 = jemná) |
| `--drop` | Odstraní outlieri z JSON |
| `--plot` | Zobrazí histogram |
| `--feature` | Parametr k vizualizaci (`B`, `tau1`, `A0`, …) |

Typické outlieri: `tau1 > 60 s` (chybná extrakce), `B > 5×10⁻³` (nerealistická inharmonicita).

---

## 5. Krok 3 — Spektrální EQ

**Skript:** `analysis/compute_spectral_eq.py`

```bash
python -u analysis/compute_spectral_eq.py \
    --params  analysis/params-ks-grand.json \
    --bank    "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --workers 4
```

| Argument | Popis |
|---|---|
| `--params` | JSON k rozšíření (in-place) |
| `--bank` | Adresář WAV (potřebné pro výpočet LTASE) |
| `--workers` | Počet paralelních procesů |

Přidá pole `spectral_eq` per vzorek: `{"freqs_hz": [...], "gains_db": [...]}` — LTASE (Long-Term Average Spectrum) poměr syntetizátoru vůči originálu. Tato křivka je vstupem pro EQ fitting v exportním skriptu.

**Log:** `analysis/runtime-logs/spectral-eq-log.txt`

---

## 6. Krok 4 — Surrogate NN trénink

**Skript:** `analysis/train_instrument_profile.py`

```bash
python -u analysis/train_instrument_profile.py \
    --in         analysis/params-ks-grand.json \
    --out        analysis/params-ks-grand-nn.json \
    --model      analysis/profile.pt \
    --epochs     1800 \
    --eval-every 10
```

| Argument | Výchozí | Popis |
|---|---|---|
| `--in` | — | Vstupní JSON s extrahovanými + EQ parametry |
| `--out` | — | Výstupní JSON (NN-smoothed profil 88×8) |
| `--model` | — | Výstupní `.pt` soubor s vahami |
| `--epochs` | `1800` | Počet epoch gradient descent |
| `--eval-every` | `10` | Frekvence evaluačního logu |
| `--epochs 0` | — | Jen inference (pro regeneraci JSON z existujícího `.pt`) |

**Architektura NN** — faktorizovaná síť, každá sub-síť predikuje jinou fyzikální veličinu:

| Sub-síť | Vstup | Výstup | Trénovaná přes proxy |
|---|---|---|:---:|
| `B_net` | (midi) | log(B) — inharmonicita | ano |
| `tau1_k1_net` | (midi, vel) | log(τ₁) pro k=1 | ano |
| `tau_ratio_net` | (midi, k) | log(τ_k / τ_{k=1}) | ano |
| `A0_net` | (midi, k, vel) | log(A0_ratio) | ano |
| `df_net` | (midi, k) | log(beat_hz) per parciál | ano |
| `phi_net` | (midi, vel) | φ_diff [rad] mezi strunami | ano |
| `biexp_net` | (midi, k, vel) | logit(a1), log(τ₂/τ₁) | ano |
| `noise_net` | (midi, vel) | log(attack_tau, A_noise) | ano |
| `eq_net` | (midi, freq) | gain_dB spektrálního EQ | ne (proxy je mono) |
| `wf_net` | (midi) | log(width_factor) | ne |
| `dur_net` | (midi) | log(duration_s) | ne |

Kladné výstupy trénovány v log-prostoru (MSE na log hodnotách = geometrická chyba).

**Typická konvergence** (ks-grand, 1800 epoch, CPU i5-12th gen, ~25 min):

```
epoch  300/1800  loss=1.071  eval=1.213
epoch  900/1800  loss=0.989  eval=1.286
epoch 1800/1800  loss=0.967  eval=1.289
```

**Výstupy:**
- `analysis/profile.pt` — `{"state_dict": ..., "hidden": 64, "eq_freqs": [...]}`
- `analysis/params-ks-grand-nn.json` — 704 vzorků, originály zachovány

---

## 7. Krok 5a — MRSTFT fine-tuning

**Skript:** `analysis/closed_loop_finetune.py --mode finetune`

Doladí váhy NN minimalizací Multi-Resolution STFT Loss (MRSTFT) oproti originálním WAV nahrávkám. Gradient teče přes diferenciabilní proxy synth (`torch_synth.py`).

```bash
python analysis/closed_loop_finetune.py \
    --mode         finetune \
    --model        analysis/profile.pt \
    --out          analysis/profile-finetuned.pt \
    --bank         "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --epochs       200 \
    --lr           3e-4 \
    --batch-size   8 \
    --eval-every   20 \
    --k-max        40 \
    --duration     3.0 \
    --render-dir   exports/finetune-samples \
    --sample-notes "21:3,48:3,60:3,84:5,96:5,108:7" \
    --log          analysis/runtime-logs/finetune.log
```

| Argument | Výchozí | Popis |
|---|---|---|
| `--mode` | — | `finetune` / `global` / `eval` |
| `--model` | — | Vstupní `.pt` (výstup kroku 4) |
| `--out` | — | Výstupní `.pt` po fine-tuningu |
| `--bank` | — | Adresář WAV |
| `--epochs` | `200` | Počet epoch |
| `--lr` | `3e-4` | Learning rate |
| `--batch-size` | `8` | Vzorků per gradient krok |
| `--eval-every` | `20` | Frekvence evaluace |
| `--k-max` | `60` | Max počet parciálů (snižte pro rychlost) |
| `--duration` | `3.0` | Délka renderovaných vzorků [s] |
| `--render-dir` | — | Adresář pro ukládání WAV checkpointů |
| `--sample-notes` | — | `"midi:vel,..."` — noty k renderování při eval |
| `--log` | — | Cesta k log souboru |
| `--opt-params` | — | Pro `--mode global`: `beat_scale,noise_level` |

**MRSTFT Loss** — kombinuje 3 FFT velikosti:

| FFT | Rozlišení | Zachycuje |
|---|---|---|
| 256 | ~6 ms | transient, šumová obálka |
| 1024 | ~23 ms | časové obálky harmonik |
| 4096 | ~93 ms | frekvence, sustain tvar |

**Rychlost** (CPU i5-12th gen, k-max=40, duration=3.0, batch=8):
- 1 gradient krok: ~15–20 s
- 1 epocha (88 kroků): ~2–3 min
- Celý run 200 epoch: ~7–10 hodin

**Rychlejší varianta** (~3–4× rychlejší, mírně nižší kvalita gradientu):

```bash
python analysis/closed_loop_finetune.py \
    --mode finetune --model analysis/profile.pt \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --duration 1.5 --k-max 20 --epochs 100
# 1 epocha: ~45 s, celý run: ~2 hod
```

---

## 8. Krok 5b — Globální SynthConfig optimalizace

Optimalizuje skalární parametry `beat_scale` a `noise_level` při zmrazených vahách NN.

```bash
python analysis/closed_loop_finetune.py \
    --mode       global \
    --model      analysis/profile-finetuned.pt \
    --bank       "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --opt-params beat_scale,noise_level \
    --epochs     100 \
    --lr         0.05 \
    --log        analysis/runtime-logs/finetune-global.log
```

**Výstup:** `analysis/profile-finetuned.synth_config.json`

```json
{
  "beat_scale": 1.001,
  "noise_level": 0.961
}
```

Hodnoty se zadají jako CLI parametry při spuštění GUI nebo se nastaví přes `RenderClient.set_config()`.

---

## 9. Krok 6 — Regenerace JSON z fine-tuned modelu

Syntetizátor načítá vždy **JSON**, ne `.pt`. Po fine-tuningu je třeba JSON přegenerovat inferencí (bez dalšího tréninku):

```bash
python -u analysis/train_instrument_profile.py \
    --in     analysis/params-ks-grand.json \
    --out    soundbanks/params-ks-grand-ft.json \
    --model  analysis/profile-finetuned.pt \
    --epochs 0
```

- `--in`: původní surová extrakce (nezměněná)
- `--out`: cílový JSON do `soundbanks/` (cílové umístění pro ICR)
- `--model`: fine-tuned checkpoint
- `--epochs 0`: přeskočí gradient update, jen inference pro všech 88×8 pozic

Trvá 10–30 sekund. Výstup je kompatibilní s GUI, CLI i RenderServerem.

---

## 10. Quickstart — train_pipeline.py

`analysis/train_pipeline.py` spustí kroky 1–6 jediným příkazem:

```bash
# Celý pipeline
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand"

# Explicitní parametry
python analysis/train_pipeline.py \
    --bank      "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --bank-name ks-grand \
    --out-dir   analysis/ \
    --epochs    1800

# Od kroku 4 (extrakce a EQ už hotové)
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --start-at 4

# Přidat MRSTFT fine-tuning
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --finetune --ft-epochs 200

# Dry-run — jen výpis příkazů
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --dry-run
```

---

## 11. Export do ICR JSON formátu

Existují dvě cesty jak dostat parametry do ICR JSON:

### Cesta A — z NN modelu (přes train_instrument_profile.py)

Používá se po kroku 4 nebo 6. JSON obsahuje NN-smoothed parametry interpolované pro všech 88×8 pozic. Viz krok 6 výše.

**Vstup:** `.pt` model  
**Výstup:** `soundbanks/params-<banka>-ft.json`

### Cesta B — přímo ze soundbank JSON (export_soundbank_params.py)

Exportuje parametry přímo z fyzikálně extrahované banky bez NN. Výhodou je věrnost originálním naměřeným hodnotám. Zahrnuje fitování spektrálního EQ jako IIR biquad kaskádu.

```bash
python analysis/export_soundbank_params.py \
    --soundbank soundbanks/params-ks-grand-ft.json \
    --out       analysis/params-piano-soundbank.json \
    --sr        44100 \
    --duration  3.0 \
    --target-rms 0.06 \
    --rng-seed  0
```

| Argument | Výchozí | Popis |
|---|---|---|
| `--soundbank` | `soundbanks/params-ks-grand-ft.json` | Vstupní soundbank JSON |
| `--out` | `analysis/params-piano-soundbank.json` | Výstupní ICR JSON |
| `--sr` | `44100` | Vzorkovací frekvence [Hz] |
| `--duration` | `3.0` | Délka pro výpočet rms_gain [s] |
| `--target-rms` | `0.06` | Cílová RMS úroveň (vel_band=7) |
| `--rng-seed` | `0` | Seed pro výpočet phi (musí odpovídat C++) |

**Co skript dělá:**
1. Pro každou z 704 not (MIDI 21–108 × vel 0–7) přečte fyzikální parametry
2. Vygeneruje počáteční fáze `phi` ze seedovaného NumPy RNG (stejný seed jako C++)
3. Normalizuje `attack_tau` — omezí na `tau1` prvního parciálu (zabrání dlouhému šumovému chvostu)
4. Renderuje notu (bez šumu) a spočte `rms_gain` zahrnující analytický výkon šumu
5. Fituje spektrální EQ křivku jako 5 biquad sekcí metodou minimální fáze + least-squares IIR
6. Zapíše JSON s kompletní strukturou

---

## 12. Parametrické soubory ICR — struktura a klíče

### Formát souboru

Soubory mají příponu `.json`, formát `piano_core_v1`. Jsou načítány C++ třídou `PianoCore::load()` (přes `nlohmann/json`). Výchozí umístění: `soundbanks/` nebo `analysis/`.

### Metadata (kořenová úroveň)

```json
{
  "format":     "piano_core_v1",
  "source":     "soundbank:params-ks-grand-ft.json",
  "sr":         44100,
  "target_rms": 0.06,
  "vel_gamma":  0.7,
  "k_max":      60,
  "rng_seed":   0,
  "duration_s": 3.0,
  "n_notes":    704,
  "notes":      { ... }
}
```

| Klíč | Typ | Popis |
|---|---|---|
| `format` | string | Verze formátu — vždy `"piano_core_v1"` |
| `source` | string | Odkud byla data exportována |
| `sr` | int | Vzorkovací frekvence při exportu [Hz] |
| `target_rms` | float | Cílová RMS úroveň pro vel_band=7 |
| `vel_gamma` | float | Exponent pro velocity škálování: `vel_gain = ((vel+1)/8)^γ` |
| `k_max` | int | Maximální počet parciálů na notu |
| `rng_seed` | int | Seed pro reprodukovatelné generování fází |
| `duration_s` | float | Délka renderované noty pro rms_gain výpočet |
| `n_notes` | int | Celkový počet not v souboru |
| `notes` | object | Slovník not (klíč: `"m021_vel0"`, …) |

### Klíče noty (`notes["m{midi:03d}_vel{vel}"]`)

```json
{
  "midi":       60,
  "vel":        3,
  "f0_hz":      261.63,
  "K_valid":    60,
  "phi_diff":   2.718,
  "attack_tau": 0.052,
  "A_noise":    0.031,
  "rms_gain":   0.0847,
  "partials":   [ ... ],
  "eq_biquads": [ ... ]
}
```

| Klíč | Typ | Popis |
|---|---|---|
| `midi` | int | MIDI nota (21–108) |
| `vel` | int | Velocity band (0–7) |
| `f0_hz` | float | Základní frekvence noty [Hz] |
| `K_valid` | int | Počet platných parciálů v `partials[]` |
| `phi_diff` | float | Fázový offset mezi strunami [rad] — generovaný RNG se seedem `rng_seed + midi*256 + vel` |
| `attack_tau` | float | Decay time šumové obálky [s] — omezeno na `tau1` k=1 |
| `A_noise` | float | Bezrozměrná amplituda šumu (poměr k signálu) |
| `rms_gain` | float | Normalizační faktor: `target_rms × vel_gain / total_rms` |
| `partials` | array | Pole parciálů (max `k_max` položek) |
| `eq_biquads` | array | 5 biquad sekcí spektrálního EQ (může být prázdné) |

### Klíče parciálu (`partials[k]`)

```json
{
  "f_hz":    261.63,
  "A0":      1.406,
  "tau1":    14.36,
  "tau2":    14.36,
  "a1":      1.0,
  "beat_hz": 0.0,
  "phi":     4.113
}
```

| Klíč | Typ | Popis |
|---|---|---|
| `f_hz` | float | Frekvence parciálu [Hz] (zahrnuje inharmonicitu) |
| `A0` | float | Počáteční amplituda |
| `tau1` | float | Rychlá časová konstanta bi-exp obalky [s] |
| `tau2` | float | Pomalá časová konstanta bi-exp obalky [s] (≥ tau1) |
| `a1` | float | Váha rychlé složky: `env = a1·e^(-t/τ₁) + (1-a1)·e^(-t/τ₂)` |
| `beat_hz` | float | Frekvence beatingu [Hz] — deladění dvou strun parciálu |
| `phi` | float | Počáteční fáze [rad] — generovaná deterministicky z `rng_seed` |

**Synthézní model** (C++ PianoCore, na základě `phi` a `beat_hz`):

```
f+  = f_hz + beat_hz * beat_scale / 2     (struna 1)
f-  = f_hz - beat_hz * beat_scale / 2     (struna 2)

s1  = cos(2π · f+ · t + phi)
s2  = cos(2π · f- · t + phi + phi_diff_k)   ← phi_diff_k: náhodné per parciál
env = a1 · exp(-t/τ₁) + (1-a1) · exp(-t/τ₂)

parciál = A0 · rms_gain · env · (s1 + s2) / 2
```

### Klíče EQ biquadu (`eq_biquads[i]`, Direct Form II, i = 0–4)

```json
{
  "b": [0.9996, -0.5990, 0.1301],
  "a": [-0.5971,  0.1295]
}
```

| Klíč | Typ | Popis |
|---|---|---|
| `b` | [b0, b1, b2] | Čitatel přenosové funkce (normalizovaný, a0 = 1) |
| `a` | [a1, a2] | Jmenovatel přenosové funkce (bez a0) |

**Diferenční rovnice** (Direct Form II, aplikuje se v C++ processBlock):

```
w[n] = x[n] - a1·w[n-1] - a2·w[n-2]
y[n] = b0·w[n] + b1·w[n-1] + b2·w[n-2]
```

5 sekcí v sérii tvoří celkový spektrální EQ filtr. Stav je nezávislý pro L a R kanál.

---

## 13. Build přes CMake

Projekt používá CMake ≥ 3.16 s MSVC toolchainem (primárně). Žádný Makefile — build řídí `cmake --build`.

### Prerekvizity

| Nástroj | Verze | Poznámka |
|---|---|---|
| Visual Studio 2022 | 17.x | MSVC toolchain + x64 |
| CMake | ≥ 3.16 | `winget install Kitware.CMake` |
| Git | libovolná | FetchContent stahuje GLFW + ImGui |
| OpenGL | systémový | součást Windows, není třeba instalovat |

### Konfigurace (první spuštění)

```bat
cmake -B build -G "Visual Studio 17 2022" -A x64
```

CMake automaticky stáhne při prvním configure:
- **GLFW 3.4** — okno, vstup (OpenGL)
- **Dear ImGui v1.91.9** — GUI framework

Vyžaduje internet při prvním spuštění. Další buildy jsou offline.

### Build targety

```bat
# Všechny targety najednou
cmake --build build --config Release

# Jen GUI
cmake --build build --config Release --target IthacaCoreResonatorGUI

# Jen render server (pro trénink / headless)
cmake --build build --config Release --target IthacaRenderServer

# Debug build
cmake --build build --config Debug
```

**Výstupní binárky:**

| Target | Cesta | Popis |
|---|---|---|
| `IthacaCoreResonatorGUI` | `build/bin/Release/IthacaCoreResonatorGUI.exe` | GUI s Dear ImGui, MIDI vstup, real-time audio |
| `IthacaCoreResonator` | `build/bin/Release/IthacaCoreResonator.exe` | Headless CLI, real-time audio |
| `IthacaRenderServer` | `build/bin/Release/IthacaRenderServer.exe` | Offline renderer, TCP server pro Python klienta |

### Kompilační volby (automatické)

| Volba | Platforma | Efekt |
|---|---|---|
| `/arch:AVX2` | MSVC x86_64 | AVX2 + FMA vektorizace |
| `-mavx2 -mfma` | GCC/Clang x86_64 | totéž |
| `/O2 /DNDEBUG` | MSVC Release | optimalizace |
| `/Od /Zi` | MSVC Debug | bez optimalizací + debug info |
| `__WINDOWS_MM__` | Win32 | WinMM MIDI backend (RtMidi) |
| `_CRT_SECURE_NO_WARNINGS` | Win32 | potlačí `scanf_s` varování |

### Custom CMake targety

```bat
# Spuštění SineCore (quick test)
cmake --build build --target run

# Smazání exportovaných WAV
cmake --build build --target clean-exports
```

### Post-build akce (automatické)

Po každém buildu CMake zkopíruje `soundbanks/` vedle každého exe:

```
build/bin/Release/
    IthacaCoreResonatorGUI.exe
    soundbanks/           ← kopie ze zdrojového soundbanks/
IthacaRenderServer/
    soundbanks/
```

### Čistý build od nuly

```bat
rmdir /S /Q build
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Alternativní toolchainy

**Developer Command Prompt:**
```bat
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

**MinGW-w64 / GCC:**
```bash
cmake -B build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-mingw
```

---

## 14. Spuštění

### GUI (doporučeno)

```bat
build\bin\Release\IthacaCoreResonatorGUI.exe --core PianoCore --params analysis/params-piano-soundbank.json
```

**CLI argumenty:**

| Argument | Výchozí | Popis |
|---|---|---|
| `--core` | `PianoCore` | Synth core: `PianoCore` nebo `SineCore` |
| `--params` | — | Cesta k ICR JSON souboru s parametry |

**GUI parametry (real-time, bez restartu):**

| Parametr | Skupina | Rozsah | Výchozí | Popis |
|---|---|---|---|---|
| `beat_scale` | Timbre | 0–4 × | 1.0 | Škáluje beat_hz všech parciálů |
| `noise_level` | Timbre | 0–4 × | 1.0 | Škáluje amplitudu šumu |
| `eq_strength` | Timbre | 0–1 × | 1.0 | Blend spektrálního EQ (0 = bypass, 1 = plný EQ) |
| `pan_spread` | Stereo | 0–π rad | 0.55 | Úhel rozevření strun 1 a 2 v konstantní-výkonové panoramě |
| `keyboard_spread` | Stereo | 0–π rad | 0.60 | Rozptyl středu panoramy přes celou klávesnici (bas vlevo, výšky vpravo) |
| `stereo_decorr` | Stereo | 0–2 × | 1.0 | Síla Schroederova first-order all-pass dekorélátoru |
| `rng_seed` | Debug | 0–9999 | 0 | Seed pro šumový PRNG |

**GUI — pravý panel (vizualizace poslední noty):**

Při každém stisknutí klávesy GUI zobrazí detail pro danou (midi, velocity) kombinaci:

| Sloupec | Obsah |
|---|---|
| STRUCTURE | počet parciálů, width_factor |
| NOISE | centroid_hz, floor_rms, tau_s šumové obálky |
| SPECTRAL EQ | frekvenční odezva biquad kaskády (min / max / mean dB) |

Sloupec **SPECTRAL EQ** zobrazuje odezvu **specifickou pro konkrétní notu a velocity** — koeficienty jsou načteny z `note_params_[midi][vel]`, takže každá nota může mít jinou EQ křivku (fitovanou z LTASE měření). Odezva se vypočítá ze 5 biquad sekcí na 32 logaritmicky rozložených frekvencích (30 Hz – 18 kHz) a aktualizuje se při každém noteOn.

### Headless CLI

```bat
build\bin\Release\IthacaCoreResonator.exe --core PianoCore --params analysis/params-piano-soundbank.json [midi_port]
```

`midi_port` — index MIDI vstupu (výchozí: 0). Dostupné porty jsou vypsány při startu.

### Render server (pro Python klienta)

```bat
build\bin\Release\IthacaRenderServer.exe soundbanks/params-ks-grand-ft.json --port 9876 --log analysis/runtime-logs/render-server.log
```

Python klient:

```python
from analysis.render_client import RenderClient

with RenderClient("build/bin/Release/IthacaRenderServer.exe",
                  "soundbanks/params-ks-grand-ft.json") as rc:
    rc.render(midi=60, vel=3, output="exports/C4.wav", duration=0.0, sr=44100)
```

---

## 15. Časté problémy

### NaN loss po epochách 80–130 (MRSTFT fine-tuning)

`B_net` zdrift → velké `log_B` → `B = exp(∞)` → `f_hz = ∞` → `cos(∞) = NaN` → permanentní NaN.  
Oprava je v `torch_synth.py` (`B` clampnuté na max=0.0, `isfinite` guard ve valid masce).  
Pokud NaN nastalo: začni znovu od checkpointu před explozí nebo od základního `profile.pt`.

### CMake ukazuje na původní adresář po kopírování projektu

```bat
del build\CMakeCache.txt
del build\_deps\glfw-subbuild\CMakeCache.txt
del build\_deps\imgui-subbuild\CMakeCache.txt
cmake -B build -G "Visual Studio 17 2022" -A x64
```

### Linker zamkne .exe (GUI spuštěné při buildu)

```bat
taskkill /F /IM IthacaCoreResonatorGUI.exe
cmake --build build --config Release
```

### OOM nebo příliš pomalý MRSTFT trénink

Snižte `--k-max` a `--duration`:
```bash
--k-max 20 --duration 1.5   # ~64 MB/nota místo ~384 MB, 3–4× rychlejší
```

### UnicodeEncodeError na Windows (cp1252)

Všechny skripty volají `sys.stdout.reconfigure(encoding='utf-8')` automaticky.  
Pokud chybí, přidej na začátek skriptu:
```python
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
```

### Načítání staršího profile.pt (bez phi_net)

`load_model()` používá `strict=False` — phi_net se inicializuje náhodně, ostatní váhy se zachovají.  
Doporučeno: krátký fine-tuning pro inicializaci phi_net:
```bash
python analysis/closed_loop_finetune.py --mode finetune --model analysis/profile.pt \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" --epochs 50 --lr 5e-5
```
