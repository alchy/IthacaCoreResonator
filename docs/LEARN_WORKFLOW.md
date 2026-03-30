# Training / Learning Workflow

Popis celého pipeline od WAV banky po natrénovaný model.

## Přehled

```
WAV banka
    |
    v
extract_params.py        -- fyzikální parametry (B, tau, A0, f0, ...)
    |
    v
find_outliers.py         -- odstranění chybně extrahovaných samplov
    |
    v
compute_spectral_eq.py   -- LTASE EQ křivka (per note, per vel)
    |
    v
train_instrument_profile.py  -- (budoucí) surrogate model / param optimizer
    |
    v
params-<banka>.json      -- čistá banka pro IthacaRenderServer
```

---

## Krok 1 — Extrakce parametrů

```bash
python analysis/extract_params.py \
    --bank_dir "C:/SoundBanks/.../ks-grand" \
    --out      analysis/params-ks-grand.json \
    --workers  8
```

Vstup: adresář se samples pojmenovanými `m{midi}_vel{vel}.wav` (nebo podobně).
Výstup: `params-ks-grand.json` se strukturou:

```json
{
  "bank_dir": "...",
  "n_samples": 704,
  "summary":  { "B_by_midi": {...}, ... },
  "samples": {
    "m060_vel3": {
      "midi": 60, "vel": 3,
      "B": 0.000412,
      "f0_fitted_hz": 261.8,
      "n_partials": 24,
      "partials": [ {"k":1, "f_hz":..., "A0":..., "tau1":..., "tau2":..., "a1":...}, ... ],
      "noise": {...}
    }, ...
  }
}
```

**Typické chyby extrakce:**
- `vel7` (pianissimo) u treble: příliš krátký sample → zero-size array při peak detection → sample se přeskočí (OK)
- Konvergence decay fitu: tau1 >> 10 s u treble forte → chyba (viz krok 2)
- Falešný B: inharmonicita 0.003 u not nad MIDI 95 → chyba

---

## Krok 2 — Filtrace outlierů

```bash
# Přehled (bez modifikace JSON)
python analysis/find_outliers.py --params analysis/params-ks-grand.json --z 10

# Vizuální kontrola
python analysis/find_outliers.py --params analysis/params-ks-grand.json --z 10 --plot --feature B
python analysis/find_outliers.py --params analysis/params-ks-grand.json --z 10 --plot --feature tau1_mean

# Brutální drop (in-place)
python analysis/find_outliers.py --params analysis/params-ks-grand.json --z 10 --drop
```

**Parametry checked:** `B`, `tau1_mean` (prumer prvnich 6 parcialu), `A0_mean`, `f0_ratio`
**Metoda:** lokální median smoother (±5 not) → MAD-based sigma → z-score
**Práh:** `--z 10` pro brutální filter; `--z 3.5` pro jemnou inspekci

Chybějící samples (po drop) se ve VoiceManager LUT interpolují ze sousedních velocity vrstev.

**Erfahrungen s ks-grand bankou (2025-03):**
- 704 extrahováno → 699 OK, 5 skip (vel7 treble, příliš krátké)
- 699 → 603 po `--z 10` drop (96 outlierů)
- Hlavní problemy: tau1 explodoval u forte treble (m100–m108, vel 5–7), B falešný u treble

---

## Krok 3 — Spektrální EQ

```bash
python analysis/compute_spectral_eq.py \
    --params analysis/params-ks-grand.json \
    --bank_dir "C:/SoundBanks/.../ks-grand"
```

Počítá LTASE (Long-Term Average Spectrum Envelope) z reálných WAV samplov a generuje EQ křivku
per note per velocity do params JSON. Používá se v enginu jako `eq_strength` blend.

---

## Krok 4 — Render server

Headless C++ render server pro training loop IPC.

### Build

```bash
cmake --build build --target IthacaRenderServer --config Release
```

**Důležité:** Render server sdílí source files s hlavním executable (`RENDER_SYNTH_SOURCES` ⊂ `SYNTH_SOURCES`).
Po změně engine kódu stačí rebuild — žádná sdílená knihovna, žádný extra krok.

```
synth/note_lut.cpp          <-- shared
synth/resonator_voice.cpp   <-- shared
synth/voice_manager.cpp     <-- shared
synth/biquad_eq.cpp         <-- shared
synth/offline_renderer.cpp  <-- render server only
synth/render_server.cpp     <-- render server only
```

### Level calibrace

IthacaRenderServer normalizuje každý render post-hoc na `target_rms * vel_gain` — stejně jako
Python `synthesize_note()`. Důvod: náhodné počáteční fáze stringů způsobují rozptyl ±50 % u not
s málo parciály; post-normalizace to eliminuje a zaručuje konzistentní amplitudu trénovacích dat.

Pro real-time syntézu (ICR/GUI) se používá analytická level_scale formule s `render_ref_duration_s`.

### Protokol (stdin/stdout JSON)

Jeden JSON objekt per řádek. Server píše log na **stderr**, protokol na **stdout**.

```bash
# Ping
echo '{"cmd":"ping"}' | ./IthacaRenderServer params-ks-grand.json

# Render noty
echo '{"cmd":"render","midi":60,"vel":80,"sr":44100,"duration":3.0,"output":"exports/m060_vel80.wav"}' | ...

# Nastavení synth parametrů
echo '{"cmd":"set_config","params":{"beat_scale":1.5,"eq_strength":0.8}}' | ...

# Reload params JSON za běhu (po update extrakce)
echo '{"cmd":"reload","params":"analysis/params-ks-grand.json"}' | ...
```

### Python client

```python
from analysis.render_client import RenderClient

with RenderClient("build/bin/Release/IthacaRenderServer.exe",
                  "analysis/params-ks-grand.json") as rc:
    rc.set_config(beat_scale=1.5, eq_strength=0.8)
    n_frames = rc.render(midi=60, vel=80, output="exports/note.wav",
                         duration=3.0, sr=44100)
    # duration=0 → auto-detect silence tail
```

---

## Krok 5 — Instrument profile (surrogate smooth model)

Cíl: natrénovat fyzikálně-motivovanou NN, která vyhladí extrahované parametry přes celý
MIDI rozsah — vyplní chybějící/dropped not a opraví extrakční šum.

### Architektura (faktorizovaná NN)

```
Vstup: (midi, vel, k_partial)
           ↓
  B_net        MLP(midi)           → B          (inharmonicita, vel-nezávislá)
  tau1_k1_net  MLP(midi, vel)      → tau1 pro k=1
  tau_ratio_net MLP(midi, k)       → log(tau_k / tau_k1)
  A0_net       MLP(midi, k, vel)   → log(A0_ratio)
  df_net       MLP(midi, k)        → beating detune
  eq_net       MLP(midi, freq)     → EQ gain_db
  noise_net    MLP(midi, vel)      → attack_tau, centroid_hz, A_noise
  biexp_net    MLP(midi, k, vel)   → a1, tau2/tau1
  dur_net      MLP(midi)           → duration_s
```

Všechny pozitivní výstupy trénované v log-space (MSE na log hodnotách = geometrická chyba).
Výstupy: NN-smoothed profil zachovává původní naměřené hodnoty (`--no-preserve-orig` pro override).

### Spuštění

```bash
python -u analysis/train_instrument_profile.py \
    --in    analysis/params-ks-grand.json \
    --out   analysis/params-nn-profile-ks-grand.json \
    --model analysis/profile.pt \
    --epochs 800 --hidden 64 --lr 0.003
```

### Výsledky (ks-grand, 2026-03-30)

```
Available measured samples: 603
Dataset: B=576  tau=4735  tau1_k1=603  A0=31911  df=24272  eq=9648  noise=603  biexp=5433
Model parameters: 90,189

epoch  100/800  loss=1.019822  lr=2.89e-03
epoch  200/800  loss=0.904562  lr=2.57e-03
epoch  300/800  loss=0.866582  lr=2.08e-03
epoch  400/800  loss=0.848845  lr=1.51e-03
epoch  500/800  loss=0.836595  lr=9.47e-04
epoch  600/800  loss=0.828756  lr=4.65e-04
epoch  700/800  loss=0.825320  lr=1.43e-04
epoch  800/800  loss=0.824400  lr=3.00e-05

Final loss: 0.824 (log-space MSE)

Output: 704 entries (603 originals preserved + 101 NN-generated)
NN B range: 1.37e-4 – 1.70e-3 (fyzikálně správný rozsah)
Model:  analysis/profile.pt
Output: analysis/params-nn-profile-ks-grand.json
```

### Poznámky

- 23 originálních samplov má B ≈ 0 (extrakce selhala, přežily outlier filter) — NN je nahradila správnými hodnotami pro NN-generated pozice; pro zachované originály zůstávají (jsou označeny `_from_profile=false`)
- Chybějící pozice (dropped vel=7 treble, mezery v LUT) jsou doplněny NN a označeny `_from_profile=true`
- Model lze re-použít: `torch.load("analysis/profile.pt")` vrátí `{"model": ..., "args": ...}`

---

## Workflow pro novou banku

1. Nahraj WAV soubory do `C:/SoundBanks/<banka>/`
2. `python analysis/extract_params.py --bank_dir ... --out analysis/params-<banka>.json`
3. `python analysis/find_outliers.py --params analysis/params-<banka>.json --z 10 --plot --feature B` — vizuální kontrola
4. `python analysis/find_outliers.py --params analysis/params-<banka>.json --z 10 --drop`
5. `python analysis/compute_spectral_eq.py --params analysis/params-<banka>.json --bank_dir ...`
6. Surrogate trénink:
   ```bash
   python -u analysis/train_instrument_profile.py \
       --in  analysis/params-<banka>.json \
       --out analysis/params-nn-profile-<banka>.json \
       --model analysis/profile-<banka>.pt \
       --epochs 800 --hidden 64 --lr 0.003
   ```
   Typická konvergenční křivka (ks-grand, 800 epoch, ~2 min na CPU):
   ```
   epoch  100: loss ≈ 1.02  epoch  400: loss ≈ 0.85
   epoch  200: loss ≈ 0.90  epoch  800: loss ≈ 0.82 (final)
   ```
7. Test render (ověří level calibraci):
   ```python
   from analysis.render_client import RenderClient
   import scipy.io.wavfile as wf, numpy as np, math
   with RenderClient("build/bin/Release/IthacaRenderServer.exe",
                     "analysis/params-nn-profile-<banka>.json") as rc:
       rc.render(midi=60, vel=80, output="/tmp/test.wav", duration=3.0)
   sr, d = wf.read("/tmp/test.wav")
   rms = math.sqrt((np.mean(d[:,0]**2) + np.mean(d[:,1]**2)) / 2)
   expected = 0.06 * (80/127)**0.7
   print(f"rms={rms:.5f}  expected={expected:.5f}  ratio={rms/expected:.4f}")
   # ratio by měl být 1.0000 (post-render normalizace garantuje)
   ```
8. Swapni params v `server_main.cpp` default nebo předej jako argv

---

## Soubory

| Soubor | Role |
|---|---|
| `analysis/extract_params.py` | WAV → physics params JSON |
| `analysis/find_outliers.py` | Outlier detection + drop |
| `analysis/compute_spectral_eq.py` | LTASE EQ křivka |
| `analysis/render_client.py` | Python wrapper pro render server |
| `analysis/physics_synth.py` | Python reference implementace fyziky |
| `synth/offline_renderer.h/.cpp` | Headless C++ renderer (s post-render RMS normalizací) |
| `synth/render_server.h/.cpp` | stdin/stdout JSON server |
| `server_main.cpp` | Render server entry point |
| `analysis/train_instrument_profile.py` | Surrogate NN trénink |
| `analysis/params-ks-grand.json` | Naměřená banka (603 samplov po filtru) |
| `analysis/params-nn-profile-ks-grand.json` | NN-smoothed profil (704 pozic) |
| `analysis/profile.pt` | Natrénovaný surrogate model |
