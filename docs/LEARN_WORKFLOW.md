# Tréninkový workflow — quickstart

Stručný přehled. Kompletní dokumentace v `docs/TRAIN.md`.

---

## Přehled pipeline

```
WAV banka  (m060-vel3-f44.wav, vel = 0-7)
    │
    ▼  extract_params.py        fyzikální parametry per nota
    │
    ▼  find_outliers.py --drop  filtrace chybně extrahovaných hodnot
    │
    ▼  compute_spectral_eq.py   LTASE spektrální EQ křivka
    │
    ▼  train_instrument_profile.py    surrogate NN (99 214 params)
    │
    ▼  closed_loop_finetune.py        MRSTFT fine-tuning (volitelný)
    │
    ▼  IthacaRenderServer.exe  +  params-nn-profile-<banka>.json
```

---

## Jednoduché spuštění (wrapper)

```bash
# Kompletní pipeline od extrakce po NN trénink
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand"

# Jen NN trénink (params JSON existuje)
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --start-at 4 \
    --epochs 1800

# S MRSTFT fine-tuningem
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --start-at 4 \
    --finetune --ft-epochs 200
```

---

## Ruční kroky

### 1. Extrakce parametrů

```bash
python -u analysis/extract_params.py \
    --bank    "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --out     analysis/params-ks-grand.json \
    --workers 4
```

Vstup: `m{midi:03d}-vel{vel}-f44.wav` (vel = 0–7).
Výstup: JSON s parametry B, tau1/tau2/a1, A0, beat_hz, noise per nota.

### 2. Filtrace outlierů

```bash
# Vizuální kontrola
python analysis/find_outliers.py \
    --params analysis/params-ks-grand.json --z 10 --plot --feature B

# Drop in-place
python analysis/find_outliers.py \
    --params analysis/params-ks-grand.json --z 10 --drop
```

### 3. Spektrální EQ

```bash
python -u analysis/compute_spectral_eq.py \
    --params  analysis/params-ks-grand.json \
    --bank    "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --workers 4
```

### 4. NN trénink

```bash
python -u analysis/train_instrument_profile.py \
    --in         analysis/params-ks-grand.json \
    --out        analysis/params-nn-profile-ks-grand.json \
    --model      analysis/profile.pt \
    --epochs     1800 \
    --eval-every 10
```

Log: `analysis/runtime-logs/train-profile-log.txt`

### 5. MRSTFT fine-tuning (volitelný)

```bash
# Fine-tuning NN vah (doporučené parametry — k-max=40 snižuje paměť a čas)
python analysis/closed_loop_finetune.py \
    --mode         finetune \
    --model        analysis/profile.pt \
    --out          analysis/profile-finetuned.pt \
    --bank         "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --epochs       200 --lr 3e-4 --batch-size 8 \
    --eval-every   20 \
    --k-max        40 --duration 3.0 \
    --render-dir   exports/finetune-samples \
    --sample-notes "21:3,48:3,60:3,84:5,96:5,108:7" \
    --log          analysis/runtime-logs/finetune.log

# Globální SynthConfig optimalizace (beat_scale, noise_level) — spustit po finetuning
python analysis/closed_loop_finetune.py \
    --mode global \
    --model analysis/profile-finetuned.pt \
    --bank  "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --opt-params beat_scale,noise_level \
    --epochs 100 --lr 0.05 \
    --log    analysis/runtime-logs/finetune-global.log
```

Výstup `--render-dir` ukládá WAV vzorky per checkpoint:
```
exports/finetune-samples/
    epoch-0000/   m021_vel3.wav  m048_vel3.wav  m060_vel3.wav
                  m084_vel5.wav  m096_vel5.wav  m108_vel7.wav
    epoch-0020/   ...
    epoch-0200/   ...
```

**Rychlost** (k-max=40, CPU i5-12th gen): eval 704 not ~1 min, epocha ~2–3 min, cely run ~7–10h.

---

## Build a spuštění render serveru

```bash
# Build
cmake -S . -B build
cmake --build build --config Release

# Spuštění
build/bin/Release/IthacaRenderServer.exe \
    analysis/params-nn-profile-ks-grand.json \
    --port 9876
```

```python
# Python klient
from analysis.render_client import RenderClient

with RenderClient("build/bin/Release/IthacaRenderServer.exe",
                  "analysis/params-nn-profile-ks-grand.json") as rc:
    # vel = velocity band 0-7 (odpovídá trénovacím datům)
    n = rc.render(midi=60, vel=3, output="exports/c4_vel3.wav", duration=3.0)
```

**Velocity konvence:** render server a trénovací data používají vel = band 0-7.
MIDI klaviatura (0–127) se mapuje automaticky ve VoiceManageru.

---

## Klíčové soubory

| Soubor | Role |
|---|---|
| `analysis/extract_params.py` | WAV → physics params JSON |
| `analysis/find_outliers.py` | Outlier detection + drop |
| `analysis/compute_spectral_eq.py` | LTASE EQ křivka |
| `analysis/train_instrument_profile.py` | Surrogate NN trénink (99 214 params) |
| `analysis/train_pipeline.py` | Wrapper — kroky 1-5 jedním příkazem |
| `analysis/mrstft_loss.py` | Multi-Resolution STFT Loss |
| `analysis/torch_synth.py` | Diferenciabilní proxy synth (2-string per parcial) |
| `analysis/closed_loop_finetune.py` | MRSTFT fine-tuning + global SynthConfig opt |
| `analysis/render_client.py` | Python wrapper pro render server |
| `analysis/physics_synth.py` | Python reference implementace fyziky |
| `synth/offline_renderer.h/.cpp` | Headless C++ renderer |
| `synth/render_server.h/.cpp` | TCP JSON render server |
| `server_main.cpp` | Render server entry point |
| `analysis/params-ks-grand.json` | Naměřená banka (704 samplov) |
| `analysis/params-nn-profile-ks-grand.json` | NN-smoothed profil (88×8) |
| `analysis/profile.pt` | Natrénovaný model — EQ-supervised (krok 4) |
| `analysis/profile-finetuned.pt` | Doladěný model po MRSTFT fine-tuningu (krok 5) |

---

Detailní dokumentace: **[docs/TRAIN.md](TRAIN.md)**
