# Tréninkový pipeline

Popis celého procesu od WAV banky po profil pro IthacaCoreResonator syntezér.

---

## Přehled

```
WAV banka  (m060-vel3-f44.wav, ...)
    |
    v  extract_params.py
params-<banka>.json          fyzikalni parametry (B, tau, A0, f0, sum, ...)
    |
    v  find_outliers.py      (--drop)
params-<banka>.json          bez chybne extrahovanych samplov
    |
    v  compute_spectral_eq.py
params-<banka>.json          + LTASE EQ krivka (telova rezonance)
    |
    v  train_instrument_profile.py
params-nn-profile-<banka>.json   NN-smoothed profil pro 88 x 8 pozic
profile-<banka>.pt               vahy surrogate modelu
    |
    v  closed_loop_finetune.py   (volitelny)
profile-<banka>.pt               doladeny model (MRSTFT fine-tuning)
```

---

## Build C++ kodu

### Prerekvizity

- CMake >= 3.16
- Visual Studio 2022 Community (MSVC) nebo GCC/Clang
- Internet pripojeni (FetchContent stahuje GLFW + ImGui pri prvnim buildu)

### Prvni konfigurace

```bash
# Z korene projektu
cmake -S . -B build

# Vystup:
# -- Configuring done (43.1s)
# -- Generating done (0.9s)
# -- Build files have been written to: .../build
```

### Build IthacaRenderServer (headless render server)

```bash
cmake --build build --target IthacaRenderServer --config Release

# Vystup:
#   server_main.cpp
#   offline_renderer.cpp
#   render_server.cpp
#   resonator_voice.cpp
#   ...
#   IthacaRenderServer.vcxproj -> build/bin/Release/IthacaRenderServer.exe
```

Binary: `build/bin/Release/IthacaRenderServer.exe`

### Build GUI aplikace

```bash
cmake --build build --target IthacaCoreResonatorGUI --config Release
# Binary: build/bin/Release/IthacaCoreResonatorGUI.exe
```

### Poznamka k CMakeCache

Po prekopirovani adresare muze CMakeCache.txt ukazovat na puvodni cestu.
Reseni: smaz cache a rekonfiguruj:

```bash
rm build/CMakeCache.txt
rm build/_deps/glfw-subbuild/CMakeCache.txt
rm build/_deps/imgui-subbuild/CMakeCache.txt
cmake -S . -B build
```

---

## Spusteni render serveru

### Ruční start

```bash
build/bin/Release/IthacaRenderServer.exe \
    analysis/params-nn-profile-ks-grand.json \
    --port 9876 \
    --log  analysis/runtime-logs/render-server.log
```

**Vystup (server je pripraven):** server nic nevypisuje na stdout — log jde do souboru.
Klient se pripoji na `127.0.0.1:9876` a server posle `{"status":"ready"}`.

**Log soubor:** `analysis/runtime-logs/render-server.log`

```
[INF][RenderServer] params: analysis/params-nn-profile-ks-grand.json
[INF][RenderServer] port:   9876
[INF][RenderServer] stopped
```

### Python klient — ping

```python
from analysis.render_client import RenderClient

with RenderClient("build/bin/Release/IthacaRenderServer.exe",
                  "analysis/params-nn-profile-ks-grand.json") as rc:
    rc.ping()
    cfg = rc.get_config()
    print(cfg)
    # {'beat_scale': 1.0, 'eq_strength': 1.0, 'noise_level': 1.0,
    #  'target_rms': 0.06, 'vel_gamma': 0.7, ...}
```

### Renderovani vzorku

```bash
python -c "
import sys; sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from analysis.render_client import RenderClient
import time

notes = [(21,3,'A0'),(48,3,'C3'),(60,3,'C4'),(84,5,'C6'),(96,5,'C7'),(108,7,'C8')]

with RenderClient('build/bin/Release/IthacaRenderServer.exe',
                  'analysis/params-nn-profile-ks-grand.json') as rc:
    for midi, vel, name in notes:
        out = f'exports/sample_{name}_vel{vel}.wav'
        t0 = time.time()
        n = rc.render(midi=midi, vel=vel, output=out, duration=0.0, sr=44100)
        print(f'{name} m{midi:03d} vel{vel}: {n/44100:.1f}s -> {out}  [{time.time()-t0:.2f}s]')
"
```

Vystup:
```
A0  m021 vel3: 15.0s -> exports/sample_A0_vel3.wav   [1.41s]
C3  m048 vel3: 11.5s -> exports/sample_C3_vel3.wav   [2.28s]
C4  m060 vel3: 12.1s -> exports/sample_C4_vel3.wav   [2.75s]
C6  m084 vel5:  1.0s -> exports/sample_C6_vel5.wav   [0.07s]
C7  m096 vel5:  2.5s -> exports/sample_C7_vel5.wav   [0.09s]
C8  m108 vel7:  7.9s -> exports/sample_C8_vel7.wav   [0.27s]
```

**Rychlost (Release, AVX2, i5-12th gen):** bass noty ~1.4 s/nota, treble < 0.3 s/nota.

### Overeni level kalibrace

```python
import soundfile as sf, numpy as np, math

notes = [(21,3,'A0'),(60,3,'C4'),(108,7,'C8')]
target_rms, vel_gamma = 0.06, 0.7

for midi, vel, name in notes:
    audio, sr = sf.read(f'exports/sample_{name}_vel{vel}.wav',
                         dtype='float32', always_2d=True)
    rms      = math.sqrt((np.mean(audio[:,0]**2) + np.mean(audio[:,1]**2)) / 2)
    vel_gain = ((vel + 1) / 8.0) ** vel_gamma
    expected = target_rms * vel_gain
    print(f'{name}: rms={rms:.5f}  expected={expected:.5f}  ratio={rms/expected:.4f}')

# A0: rms=0.03693  expected=0.03693  ratio=1.0000
# C4: rms=0.03693  expected=0.03693  ratio=1.0000
# C8: rms=0.06000  expected=0.06000  ratio=1.0000
```

Ratio = 1.0000 zarucena post-render normalizaci v `offline_renderer.cpp`.

---

## Krok 1 — Extrakce fyzikalnich parametru

**Skript:** `analysis/extract_params.py`

```bash
python -u analysis/extract_params.py \
    --bank    "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --out     analysis/params-ks-grand.json \
    --workers 4
```

**Vstup:** adresar se soubory `m{midi:03d}-vel{vel}-f44.wav` (vel = 0-7).

**Extrahovane parametry** (per nota, per parcial):
- `B` — inharmonicita: f_k = k * f0 * sqrt(1 + B*k^2)
- `tau1`, `tau2`, `a1` — bi-exponencialni utlum (tau2 >= 3 * tau1)
- `beat_hz` — detune strun (vznik beatingu)
- `A0` — pocatecni amplituda parcials
- `noise` — `attack_tau_s`, `centroid_hz`, `A_noise` (bezrozmerny pomer noise/signal)
- `duration_s` — delka sampluu

**Ochrana pred kicked soubory:** soubor s delkou < 70 % predchozi extrakce se preskoci.

**Log:** `analysis/runtime-logs/extract-params-log.txt`

---

## Krok 2 — Filtrace outlieru

**Skript:** `analysis/find_outliers.py`

```bash
# Vizualni inspekce
python analysis/find_outliers.py \
    --params analysis/params-ks-grand.json \
    --z 10 --plot --feature B

# Drop outlieru in-place
python analysis/find_outliers.py \
    --params analysis/params-ks-grand.json \
    --z 10 --drop
```

**Prahove hodnoty:**
- `--z 10` — gruba filtrace (jen zjevne chyby: tau1 > 60 s, B > 5e-3)
- `--z 3.5` — jemna inspekce

---

## Krok 3 — Spektralni EQ

**Skript:** `analysis/compute_spectral_eq.py`

```bash
python -u analysis/compute_spectral_eq.py \
    --params  analysis/params-ks-grand.json \
    --bank    "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --workers 4
```

Modifikuje JSON in-place — prida pole `spectral_eq` per samplu (LTASE ratio).

**Log:** `analysis/runtime-logs/spectral-eq-log.txt`

---

## Krok 4 — Surrogate NN trenink

**Skript:** `analysis/train_instrument_profile.py`

```bash
python -u analysis/train_instrument_profile.py \
    --in         analysis/params-ks-grand.json \
    --out        analysis/params-nn-profile-ks-grand.json \
    --model      analysis/profile.pt \
    --epochs     1800 \
    --eval-every 10
```

**Log:** `analysis/runtime-logs/train-profile-log.txt`

### Architektura (faktorizovana NN)

| Sit | Vstup | Vystup | Zavislost |
|---|---|---|---|
| `B_net` | (midi) | log(B) | vel-nezavisla |
| `tau1_k1_net` | (midi, vel) | log(tau1) pro k=1 | vel-zavisla |
| `tau_ratio_net` | (midi, k) | log(tau_k/tau_k1) | vel-nezavisla |
| `A0_net` | (midi, k, vel) | log(A0_ratio) | vel-zavisla |
| `df_net` | (midi, k) | log(beat_hz) | vel-nezavisla |
| `eq_net` | (midi, freq) | gain_db | vel-nezavisla |
| `noise_net` | (midi, vel) | log(attack_tau), log(centroid), log(A_noise) | vel-zavisla |
| `biexp_net` | (midi, k, vel) | logit(a1), log(tau2/tau1) | vel-zavisla |
| `dur_net` | (midi) | log(duration_s) | vel-nezavisla |

Vsechny kladne vystupy trenovane v log-space (MSE na log hodnotach = geometricka chyba).

### Typicka konvergence (ks-grand, 1800 epoch, CPU i5-12th gen, ~25 min)

```
Available measured samples: 704
Train/eval split: 632 train, 72 eval
Model parameters: 90,189

epoch  100/1800  loss=0.914945  lr=2.98e-03  eval=1.617853
epoch  300/1800  loss=0.860287  lr=2.80e-03  eval=1.259550
epoch  600/1800  loss=0.837567  lr=2.26e-03  eval=1.234982
epoch  900/1800  loss=0.794354  lr=1.51e-03  eval=1.159825
epoch 1200/1800  loss=0.775356  lr=7.73e-04  eval=1.107646
epoch 1500/1800  loss=0.766831  lr=2.29e-04  eval=1.102404
epoch 1800/1800  loss=0.764882  lr=3.00e-05  eval=1.101438

Model saved -> analysis/profile.pt
Written -> analysis/params-nn-profile-ks-grand.json
  NN-interpolated: 0  |  Measured originals: 704
```

### Vystupy

- `analysis/profile.pt` — format: `{"state_dict": ..., "hidden": 64, "eq_freqs": [...]}`
- `analysis/params-nn-profile-ks-grand.json` — 704 samplov, originaly zachovany (`_interpolated: false`)

---

## Krok 5 — Closed-loop MRSTFT fine-tuning (volitelny)

**Skript:** `analysis/closed_loop_finetune.py`

Doladi vahu NN minimalizaci Multi-Resolution STFT Loss (MRSTFT) oproti originalnim WAV nahravkam.
Gradient tece pres diferencovatelny proxy synth (torch_synth.py).

### MRSTFT Loss

Kombinuje 3 FFT velikosti pro pokryti ruznych casovych skal:

| FFT | Rozliseni | Zachycuje |
|---|---|---|
| 256 | ~6 ms | transient / utok, sumova obalka |
| 1024 | ~23 ms | casove obalky harmonik |
| 4096 | ~93 ms | frekvence parcialu, sustain tvar |

Pro kazdou skalu: **spectral convergence** + **log-magnitude** chyba → prumer.

### Evaluace (bez update)

```bash
python analysis/closed_loop_finetune.py \
    --mode eval \
    --model analysis/profile.pt \
    --bank  "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --log   analysis/runtime-logs/finetune.log
```

Vystup (vybrane not, ks-grand, proxy synth vs original, mono, 3s crop):
```
m021 vel3  MRSTFT=3.6584
m036 vel3  MRSTFT=3.2573
m048 vel3  MRSTFT=3.8912
m060 vel3  MRSTFT=3.4201
m072 vel5  MRSTFT=2.6781
m084 vel5  MRSTFT=4.0683
m096 vel5  MRSTFT=7.9246
m108 vel7  MRSTFT=6.8498
-- mean MRSTFT = 6.8891  (704/704 notes) --
```

Vyssi MRSTFT = vetsi rozdil proxy vs original. Bass noty (MIDI 21-48) maji nizssi ztatu
diky vice parcialu v audibilnim pasu. Treble noty (MIDI 96+) maji jen 2-5 parcialu — NN
extrapoluje a MRSTFT je vyssi.

**Log:** `analysis/runtime-logs/finetune.log`

### Fine-tuning s prubeznymi WAV vzorky

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
    --render-dir   exports/finetune-samples \
    --sample-notes "21:3,48:3,60:3,84:5,96:5" \
    --log          analysis/runtime-logs/finetune.log
```

**`--render-dir`**: pri kazdem eval checkpointu ulozi WAV vzorky via proxy synth:
```
exports/finetune-samples/
    epoch-0000/
        m021_vel3.wav   # A0 stredni velocity
        m048_vel3.wav   # C3
        m060_vel3.wav   # stredni C
        m084_vel5.wav   # C6 forte
        m096_vel5.wav   # C7 forte
    epoch-0020/
        m021_vel3.wav
        ...
    epoch-0200/
        ...
```

Tyto soubory jsou mono (proxy synth), zakladni kvalita — pro finalni stereo vzorky pouzij IthacaRenderServer.

**Log:** `analysis/runtime-logs/finetune.log`

```
-- Fine-tune: 704 reference notes, batch=8, epochs=200, lr=0.0003 --

[epoch 0 / 200] initial evaluation:
  m021 vel0  MRSTFT=5.9474
  m021 vel3  MRSTFT=3.6584
  ...
  -- mean MRSTFT = 6.8891  (704/704 notes) --
  sample -> m021_vel3.wav
  sample -> m060_vel3.wav
  ...

[epoch   10/200] loss=6.7123  lr=3.00e-04  t=347.2s
[epoch   20/200] loss=6.5891  lr=2.99e-04  t=344.1s

[epoch 20 eval]
  -- mean MRSTFT = 6.6120  (704/704 notes) --
  new best: 6.6120
  sample -> m021_vel3.wav
  ...
```

### Rychlost (CPU, i5-12th gen)

- 1 gradient krok (1 nota, k_max=40, duration=3s): ~0.5 s
- 1 epocha (704 not, batch=8): ~6 min (88 kroku x ~4s)
- Plny eval (704 not, no grad): ~2.5 min
- Cely run (200 epoch + 10 evalu): ~23 hodin

**Doporuceni pro rychlejsi run:**
```bash
# Kratsi vzorky + mene parcialu = ~3x rychlejsi
python analysis/closed_loop_finetune.py \
    --mode finetune \
    --model analysis/profile.pt \
    --bank  "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --duration 1.5 \
    --k-max    20 \
    --epochs   100
# 1 epocha: ~1.5 min, cely run: ~4 hod
```

### Gradient path (co se uci)

```
NN vahy
  +--> B_net       --> frekvence parcialu (f_k = k*f0*sqrt(1+B*k^2))
  +--> tau1_k1_net --> utlumove obalky (tau1)
  +--> tau_ratio   --> utlumove obalky (tau_k / tau_k1)
  +--> A0_net      --> amplitudy parcialu
  +--> biexp_net   --> bi-exponencialni obalka (a1, tau2/tau1)
  +--> noise_net   --> sumova obalka (A_noise, attack_tau)
       |
       v  proxy synth (torch_synth.py, mono)
       |
       v  MRSTFT loss vs original WAV (mrstft_loss.py)
       |
       v  loss.backward() --> optimizer.step()
```

df_net, eq_net, wf_net, dur_net nemaji gradient skrze proxy (proxy je nezahrnuje).

---

## Pouziti quickstart wrapperem

`analysis/train_pipeline.py` spusti kroky 1-5 jednim prikazem:

```bash
# Cely pipeline (doporucene)
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand"

# S explicitni bankou a vystupem
python analysis/train_pipeline.py \
    --bank      "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --bank-name ks-grand \
    --out-dir   analysis/ \
    --epochs    1800

# Od kroku 4 (extract + EQ uz hotove)
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --start-at 4

# Pridat MRSTFT fine-tuning
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --finetune --ft-epochs 200

# Dry-run (jen vypis prikazu)
python analysis/train_pipeline.py \
    --bank "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --dry-run
```

---

## Integrace se syntezatorem

Po treninku pouzij JSON profil jako vstup pro IthacaRenderServer:

```bash
# Server s novym profilem
build/bin/Release/IthacaRenderServer.exe \
    analysis/params-nn-profile-ks-grand.json \
    --port 9876

# Nebo default v server_main.cpp:
# std::string params_json = "analysis/params-nn-profile-ks-grand.json";
```

---

## Soubory a logy

| Soubor | Obsah | Generuje |
|---|---|---|
| `analysis/params-<banka>.json` | namere fizikalni parametry | extract_params.py |
| `analysis/params-nn-profile-<banka>.json` | NN-smoothed profil (88x8) | train_instrument_profile.py |
| `analysis/profile-<banka>.pt` | vahy surrogate modelu | train_instrument_profile.py |
| `build/bin/Release/IthacaRenderServer.exe` | headless render server | cmake --build |
| `exports/*.wav` | renderovane vzorky | IthacaRenderServer / finetune |
| `exports/finetune-samples/epoch-NNNN/*.wav` | prubehu vzorky z fine-tuningu | closed_loop_finetune.py |
| `analysis/runtime-logs/extract-params-log.txt` | log extrakce | extract_params.py |
| `analysis/runtime-logs/spectral-eq-log.txt` | log EQ vypoctu | compute_spectral_eq.py |
| `analysis/runtime-logs/train-profile-log.txt` | log NN treninku | train_instrument_profile.py |
| `analysis/runtime-logs/render-server.log` | log render serveru | IthacaRenderServer |
| `analysis/runtime-logs/finetune.log` | log MRSTFT fine-tuningu | closed_loop_finetune.py |

---

## Caste problemy

### CMake konfiguruje spatny adresar

Kdyz je projekt zkopirovan, CMakeCache.txt ukazuje na puvodni cestu:
```
CMake Error: The current CMakeCache.txt directory ... is different
```
Reseni:
```bash
rm build/CMakeCache.txt
rm build/_deps/glfw-subbuild/CMakeCache.txt
rm build/_deps/imgui-subbuild/CMakeCache.txt
cmake -S . -B build
```

### Extrakce selhava na vel=7 treble

Kratke sampley — preskocuji se automaticky, NN interpoluje.

### tau1 exploduje u forte treble (MIDI 100+, vel 5-7)

Odstranim outlier filtrem (`--z 10 --drop`).

### MRSTFT fine-tuning: OOM nebo prilis pomaly

Proxy synth alokuje (K, N) matice:
- K=60, duration=3s: ~384 MB/nota pri backprop
- K=40, duration=3s: ~256 MB/nota
- K=20, duration=1.5s: ~64 MB/nota (~3x rychlejsi)

```bash
# Rychlejsi varianta (ztrata neco na kvalite gradientu)
python analysis/closed_loop_finetune.py \
    --mode finetune \
    ... \
    --k-max 20 --duration 1.5
```

### UnicodeEncodeError na Windows (cp1252)

Pridej na zacatek skriptu:
```python
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
```
Vsechny skripty v tomto projektu to delaji automaticky.
