# IthacaCoreResonator — workflow quickstart

Stručný přehled od WAV banky po živé hraní. Kompletní dokumentace v `docs/TRAIN.md`.

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
    │                                 → profile.pt
    │                                 → params-nn-profile-<banka>.json
    │
    ▼  closed_loop_finetune.py --mode finetune   (volitelný)
    │                                 → profile-finetuned.pt
    │
    ▼  closed_loop_finetune.py --mode global     (volitelný)
    │                                 → profile.synth_config.json
    │
    ▼  IthacaCoreResonatorGUI.exe  params-nn-profile-<banka>.json
       IthacaCoreResonator.exe     (headless)
       IthacaRenderServer.exe      (offline rendering)
```

---

## Spuštění syntezátoru

### GUI (primární)

```bat
build\bin\Release\IthacaCoreResonatorGUI.exe analysis\params-nn-profile-ks-grand.json
```

GUI obsahuje:
- Piano vizualizaci a voice matrix
- Panel SynthConfig s live sliders (beat_scale, eq_strength, noise_level, ...)
- Peak metering, SysEx LED indikátor
- Klávesová fallback: viz CLI níže

### CLI (headless — bez GUI)

```bat
build\bin\Release\IthacaCoreResonator.exe [params.json] [midi_port]
```

| Argument | Výchozí | Popis |
|---|---|---|
| `params.json` | `soundbanks/salamander.json` | Physics parameter tabulka |
| `midi_port` | `0` | Index MIDI vstupu (0 = první dostupný) |

Dostupné MIDI porty jsou vypsány při startu:
```
[INF][MIDI] port [0] loopMIDI Port
[INF][MIDI] port [1] Arturia KeyLab
```

**Klávesová fallback** (bez MIDI hardware):
```
a s d f g h j k   →   C4 D4 E4 F4 G4 A4 B4 C5
z                 →   sustain pedal (toggle)
q                 →   quit
```

### Parametry při spuštění s NN profilem

Po tréninku a fine-tuning použij `params-nn-profile-ks-grand.json` místo výchozího `soundbanks/salamander.json`:

```bat
:: GUI
build\bin\Release\IthacaCoreResonatorGUI.exe analysis\params-nn-profile-ks-grand.json

:: CLI s druhým MIDI portem
build\bin\Release\IthacaCoreResonator.exe analysis\params-nn-profile-ks-grand.json 1
```

### Aplikace optimalizovaného SynthConfig

Po `--mode global` vznikne `profile.synth_config.json` s hodnotami `beat_scale` a `noise_level`.
Tyto hodnoty lze aplikovat přes SysEx (Python helper) nebo ručně v GUI:

```python
import json
from analysis.render_client import RenderClient

cfg = json.load(open("analysis/profile.synth_config.json"))
# cfg = {"beat_scale": 1.23, "noise_level": 0.88}

# Via RenderServer (offline / testování):
with RenderClient("build/bin/Release/IthacaRenderServer.exe",
                  "analysis/params-nn-profile-ks-grand.json") as rc:
    rc.set_config(**cfg)

# Via SysEx na živý ICR (loopMIDI):
from analysis.sysex_test import build_set_param, PARAMS, open_port
midi = open_port(0)
for name, value in cfg.items():
    pid = PARAMS[name][0]
    midi.send_message(build_set_param(pid, float(value)))
midi.close_port()
```

Parametry lze také nastavit přímo v GUI pomocí sliderů (panel vpravo).

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

# S MRSTFT fine-tuningem a WAV checkpointy
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --start-at 4 \
    --finetune --ft-epochs 200 \
    --ft-render-dir exports/finetune-samples
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

Výstup `--render-dir` ukládá WAV vzorky per checkpoint (timestamp přidán automaticky):
```
exports/finetune-samples-20260331-0810/
    epoch-0000/   m021_vel3.wav  m048_vel3.wav  m060_vel3.wav
                  m084_vel5.wav  m096_vel5.wav  m108_vel7.wav
    epoch-0020/   ...
    epoch-0200/   ...
```

**Rychlost** (k-max=40, CPU i5-12th gen): eval 704 not ~1 min, epocha ~2–3 min, celý run ~7–10h.

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
| `analysis/sysex_test.py` | SysEx test helper + codec verifikace |
| `analysis/physics_synth.py` | Python reference implementace fyziky |
| `synth/offline_renderer.h/.cpp` | Headless C++ renderer |
| `synth/render_server.h/.cpp` | TCP JSON render server |
| `server_main.cpp` | Render server entry point |
| `analysis/params-ks-grand.json` | Naměřená banka (704 samplov) |
| `analysis/params-nn-profile-ks-grand.json` | NN-smoothed profil (88×8) — vstup pro ICR |
| `analysis/profile.pt` | Natrénovaný model — EQ-supervised (krok 4) |
| `analysis/profile-finetuned.pt` | Doladěný model po MRSTFT fine-tuningu (krok 5) |
| `analysis/profile.synth_config.json` | Optimalizovaný SynthConfig (beat_scale, noise_level) |

---

Detailní dokumentace: **[docs/TRAIN.md](TRAIN.md)** | **[docs/SYSEX.md](SYSEX.md)** | **[docs/RENDER_SERVER.md](RENDER_SERVER.md)**
