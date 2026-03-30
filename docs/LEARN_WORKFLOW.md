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

## Krok 5 — Surrogate model (TODO)

Cíl: natrénovat diferenciální NN approximující C++ engine, pak přes NN optimalizovat synth parametry.

### Architektura (navržená)

```
Vstup: (midi, vel) → fyzikální features (B, f0_nominal, n_strings, ...)
                    ↓
          Factorisovaná NN (physics-motivated):
            sub-net B(midi)        → inharmonicita
            sub-net tau(midi, vel) → decay
            sub-net A0(midi, vel)  → amplitudy
            sub-net df(midi, vel)  → beating detune
            sub-net EQ(midi, vel)  → spektrální korekce
                    ↓
          SynthConfig params → IthacaRenderServer → WAV
```

### Training loop

```python
# 1. Generování (params → render → WAV) párů
# 2. Feature extrakce z WAV (mel-spektrogram nebo per-partial features)
# 3. Loss: MSE v mel-spektrogram nebo perceptual (STFT loss)
# 4. Backprop přes NN surrogate (ne přes C++ — ten je black box oracle)
# 5. Optimalizace SynthConfig params přes surrogate

# Render server jako oracle pro ground truth:
rc.set_config(**current_params)
rc.render(midi=m, vel=v, output=wav_path)
target_features = extract_features(wav_path)
```

### Surrogate vs. přímá optimalizace

- C++ je black box — chain rule se přeruší na hranici subprocess
- NN surrogate nahrazuje C++ pro gradient computation
- C++ slouží jako ground truth oracle pro generování training dat
- Surrogate se trénuje na (SynthConfig params → audio features) mapování
- Výsledek: optimalizované SynthConfig, ne natrénovaná NN pro přímé použití

---

## Workflow pro novou banku

1. Nahraj WAV soubory do `C:/SoundBanks/<banka>/`
2. `python analysis/extract_params.py --bank_dir ... --out analysis/params-<banka>.json`
3. `python analysis/find_outliers.py --params analysis/params-<banka>.json --z 10 --plot --feature B` — vizuální kontrola
4. `python analysis/find_outliers.py --params analysis/params-<banka>.json --z 10 --drop`
5. `python analysis/compute_spectral_eq.py --params analysis/params-<banka>.json --bank_dir ...`
6. Test render: `python -c "from analysis.render_client import RenderClient; ..."`
7. Swapni params v `server_main.cpp` default nebo předej jako argv

---

## Soubory

| Soubor | Role |
|---|---|
| `analysis/extract_params.py` | WAV → physics params JSON |
| `analysis/find_outliers.py` | Outlier detection + drop |
| `analysis/compute_spectral_eq.py` | LTASE EQ křivka |
| `analysis/render_client.py` | Python wrapper pro render server |
| `analysis/physics_synth.py` | Python reference implementace fyziky |
| `synth/offline_renderer.h/.cpp` | Headless C++ renderer |
| `synth/render_server.h/.cpp` | stdin/stdout JSON server |
| `server_main.cpp` | Render server entry point |
| `analysis/params-ks-grand.json` | Aktuální banka (603 samplov po filtru) |
