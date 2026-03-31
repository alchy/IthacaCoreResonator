# Tréninkový pipeline

Popis celého procesu od WAV banky po profil pro IthacaCoreResonator syntezér.

---

## Přehled

```
WAV banka  (m060-vel3-f44.wav, ...)
    |
    v  extract_params.py
params-<banka>.json          fyzikalni parametry (B, tau, A0, f0, beat_hz, ...)
    |
    v  find_outliers.py      (--drop)
params-<banka>.json          bez chybne extrahovanych samplov
    |
    v  compute_spectral_eq.py
params-<banka>.json          + LTASE EQ krivka (telova rezonance)
    |
    v  train_instrument_profile.py
params-nn-profile-<banka>.json   NN-smoothed profil pro 88 x 8 pozic
profile-<banka>.pt               vahy surrogate modelu (99 214 params)
    |
    v  closed_loop_finetune.py --mode finetune   (volitelny)
profile-<banka>.pt               doladeny model (MRSTFT fine-tuning NN vah)
    |
    v  closed_loop_finetune.py --mode global     (volitelny)
profile.synth_config.json        optimalizovane globalni SynthConfig parametry
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

### Rucni start

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

# vel = velocity band 0-7 (odpovidaji trenovacim datum)
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
A0  m021 vel3: 15.0s -> exports/sample_A0_vel3.wav   [~1.4s]
C3  m048 vel3: 13.2s -> exports/sample_C3_vel3.wav   [~2.3s]
C4  m060 vel3:  6.9s -> exports/sample_C4_vel3.wav   [~1.5s]
C6  m084 vel5:  1.0s -> exports/sample_C6_vel5.wav   [~0.1s]
C7  m096 vel5:  0.8s -> exports/sample_C7_vel5.wav   [~0.1s]
C8  m108 vel7:  0.6s -> exports/sample_C8_vel7.wav   [~0.1s]
```

**Velocity konvence:** render server pouziva velocity BAND 0-7 (odpovidaji trenovacim datum:
`m060-vel3-f44.wav` → vel=3). Pro priehrávání z MIDI klaviatury (0-127) VoiceManager automaticky
mapuje: `vel_band = round(midi_vel * 7 / 127)`.

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
    vel_gain = ((vel + 1) / 8.0) ** vel_gamma   # vel = band 0-7
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

| Sit | Vstup | Vystup | Zavislost | Gradient skrze proxy |
|---|---|---|---|---|
| `B_net` | (midi) | log(B) | vel-nezavisla | ano — frekvence parcialu |
| `tau1_k1_net` | (midi, vel) | log(tau1) pro k=1 | vel-zavisla | ano — utlumove obalky |
| `tau_ratio_net` | (midi, k) | log(tau_k/tau_k1) | vel-nezavisla | ano — utlumove obalky |
| `A0_net` | (midi, k, vel) | log(A0_ratio) | vel-zavisla | ano — amplitudy |
| `df_net` | (midi, k) | log(beat_hz) | vel-nezavisla | ano — beating frekvence |
| `phi_net` | (midi, vel) | phi_diff (rad) | vel-zavisla | ano — pocatecni faze strun |
| `biexp_net` | (midi, k, vel) | logit(a1), log(tau2/tau1) | vel-zavisla | ano — bi-exp obalka |
| `noise_net` | (midi, vel) | log(attack_tau), log(centroid), log(A_noise) | vel-zavisla | ano — sumova obalka |
| `eq_net` | (midi, freq) | gain_db | vel-nezavisla | ne (proxy EQ nema) |
| `wf_net` | (midi) | log(width_factor) | vel-nezavisla | ne (proxy je mono) |
| `dur_net` | (midi) | log(duration_s) | vel-nezavisla | ne |

Vsechny kladne vystupy trenovane v log-space (MSE na log hodnotach = geometricka chyba).

**Poznamka k phi_net:** pocitaci faze phi_diff = faze2 - faze1 mezí dvema strunami parcials.
- phi_diff = 0: konstruktivni interference → maximalni utok
- phi_diff = π: destruktivni interference → minimalni utok, maximalni beating
- Init bias = 0 (konstruktivni), site se uci optimalni fazi per (midi, vel)

### Typicka konvergence (ks-grand, 1800 epoch, CPU i5-12th gen, ~25 min)

```
Available measured samples: 704
Train/eval split: 632 train, 72 eval
Model parameters: 99,214

epoch  100/1800  loss=1.133206  lr=2.98e-03  eval=1.507818
epoch  300/1800  loss=1.070663  lr=2.80e-03  eval=1.213169
epoch  600/1800  loss=1.030450  lr=2.26e-03  eval=1.258982
epoch  900/1800  loss=0.989487  lr=1.51e-03  eval=1.286126
epoch 1200/1800  loss=0.973444  lr=7.73e-04  eval=1.285231
epoch 1500/1800  loss=0.968038  lr=2.29e-04  eval=1.288656
epoch 1800/1800  loss=0.966700  lr=3.00e-05  eval=1.288875

Model saved -> analysis/profile.pt
Written -> analysis/params-nn-profile-ks-grand.json
  NN-interpolated: 0  |  Measured originals: 704
```

Poznamka: `eval` obsahuje jak physics loss tak EQ MSE loss (`eval = physics + 0.1 * eq_mse`).
Vysledny eval ~1.29 odpovida physics eval ~1.14 + 0.1 × eq_mse ~0.15.
Bez EQ termu (stary trening) byl eval=1.101 — to bylo nize, protoze eq_net nebyl supervizovan.

### Vystupy

- `analysis/profile.pt` — format: `{"state_dict": ..., "hidden": 64, "eq_freqs": [...]}`
- `analysis/params-nn-profile-ks-grand.json` — 704 samplov, originaly zachovany (`_interpolated: false`)

**Zpetna kompatibilita:** `closed_loop_finetune.load_model()` pouziva `strict=False` → existujici
`profile.pt` (bez phi_net) se nacte a phi_net se inicializuje nahodne (pote uci se pres MRSTFT).

---

## Krok 5 — Closed-loop MRSTFT fine-tuning

**Skript:** `analysis/closed_loop_finetune.py`

Doladi vahy NN (nebo globalni SynthConfig) minimalizaci Multi-Resolution STFT Loss (MRSTFT)
oproti originalnim WAV nahravkam. Gradient tece pres diferencovatelny proxy synth (`torch_synth.py`).

### MRSTFT Loss

Kombinuje 3 FFT velikosti pro pokryti ruznych casovych skal:

| FFT | Rozliseni | Zachycuje |
|---|---|---|
| 256 | ~6 ms | transient / utok, sumova obalka |
| 1024 | ~23 ms | casove obalky harmonik |
| 4096 | ~93 ms | frekvence parcialu, sustain tvar |

Pro kazdou skalu: **spectral convergence** + **log-magnitude** chyba → prumer.

### Diferencovatelny proxy synth (torch_synth.py)

Proxy pouziva 2-string model per parcial (oproti puvodnimu 1-string):

```
Kazdy parcial k:
  f+ = f_k + beat_hz_k / 2     (struna 1)
  f- = f_k - beat_hz_k / 2     (struna 2)

  s1 = cos(2π * f+ * t + phi1)
  s2 = cos(2π * f- * t + phi1 + phi_diff)
  parcial = A0_k * env_k(t) * (s1 + s2) / 2

  phi_diff = phi_net(midi, vel) ... skalarna hodnota per nota
  beat_hz_k = exp(df_net(midi, k)) * beat_scale
```

Gradient tece pres: `B_net` (frekvence), `tau nets` (obalky), `A0_net` (amplitudy),
`biexp_net` (bi-exp obalka), `df_net` (beating), `phi_net` (pocatecni faze strun),
`noise_net` (sum). `beat_scale` a `noise_level` mohou byt tenzory → gradient skrze ne.

### Evaluace (bez update)

```bash
python analysis/closed_loop_finetune.py \
    --mode eval \
    --model analysis/profile.pt \
    --bank  "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --log   analysis/runtime-logs/finetune.log
```

### Fine-tuning NN vah

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
        ...
    epoch-0200/
        ...
```

### Globalni SynthConfig optimalizace

Optimalizuje skalarne parametry SynthConfig (beat_scale, noise_level) s NN vahy zmrazenymi.
Gradienty tecou pres `torch_synth.render_note_differentiable()` → MRSTFT.

```bash
python analysis/closed_loop_finetune.py \
    --mode global \
    --model analysis/profile.pt \
    --bank  "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --opt-params beat_scale,noise_level \
    --epochs 100 \
    --lr     0.05 \
    --log    analysis/runtime-logs/finetune-global.log
```

Vystup JSON: `analysis/profile.synth_config.json` (nebo `--config-out <path>`)
```json
{
  "beat_scale": 1.234,
  "noise_level": 0.876
}
```

Pouziti ve serveru:
```python
with RenderClient(...) as rc:
    import json
    cfg = json.load(open("analysis/profile.synth_config.json"))
    rc.set_config(cfg)
```

### Rychlost (CPU, i5-12th gen)

Merenou na run s `--k-max 40 --duration 3.0 --batch-size 8`:

- 1 gradient krok (batch 8 not, k_max=40, duration=3s): ~15–20 s
- 1 epocha (704 not, batch=8): ~2–3 min (88 kroku)
- Plny eval (704 not, no grad): ~1 min
- Cely run (200 epoch + 10 evalu): ~7–10 hodin

S vychozim `--k-max 60` (bez explicitniho omezeni) je cas ~2× vyssi (vice parcialu).

**Doporuceni pro rychlejsi run:**
```bash
# Kratsi vzorky + mene parcialu = ~3-4x rychlejsi
python analysis/closed_loop_finetune.py \
    --mode finetune \
    --model analysis/profile.pt \
    --bank  "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --duration 1.5 \
    --k-max    20 \
    --epochs   100
# 1 epocha: ~45 s, cely run: ~2 hod
```

### Gradient path (co se uci)

```
NN vahy
  +--> B_net         --> frekvence parcialu (f_k = k*f0*sqrt(1+B*k^2))
  +--> tau1_k1_net   --> utlumove obalky (tau1)
  +--> tau_ratio_net --> utlumove obalky (tau_k / tau_k1)
  +--> A0_net        --> amplitudy parcialu
  +--> biexp_net     --> bi-exponencialni obalka (a1, tau2/tau1)
  +--> df_net        --> beating frekvence per parcial
  +--> phi_net       --> pocatecni relativni faze strun
  +--> noise_net     --> sumova obalka (A_noise, attack_tau)
       |
       v  proxy synth 2-string (torch_synth.py, mono)
          s1 = cos((f+df/2)*t + phi1)
          s2 = cos((f-df/2)*t + phi1 + phi_diff)
          parcial = A0 * env * (s1+s2)/2
       |
       v  MRSTFT loss vs original WAV (mrstft_loss.py, 3 skaly)
       |
       v  loss.backward() --> optimizer.step()

SynthConfigParams (--mode global, NN zmrazena)
  +--> log_beat_scale  --> beat_hz_k *= beat_scale
  +--> log_noise_level --> A_noise * noise_level
       |
       v  (stejna proxy render cesta)
       |
       v  MRSTFT loss --> optimizer.step()
```

`eq_net`, `wf_net`, `dur_net` gradient skrze proxy nemaji (proxy EQ nema, je mono).

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

## Velocity konvence

**Trenovaci data:** vel = band 0-7 (WAV soubory `m060-vel3-f44.wav`)

**Render server** (`IthacaRenderServer`, `RenderClient`): vel = band 0-7

**MIDI vstup** (klaviatura, plugin): vel = raw MIDI 0-127, VoiceManager mapuje automaticky:
```
vel_pos  = midi_vel * 7 / 127   // 0.0 - 7.0 float pro LUT interpolaci
vel_band = round(vel_pos)        // 0-7 pro amplitudovy vzorec
```

**Amplitudovy vzorec** (stejny ve vsech mistech):
```
vel_gain = ((vel_band + 1) / 8)^vel_gamma
```
Max vel_gain = 1.0 (pro vel_band=7), target_rms=0.06 → max RMS = 0.06 bez clippingu.

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
| `analysis/profile.pt` | vahy surrogate modelu (99 214 params) | train_instrument_profile.py |
| `analysis/profile-finetuned.pt` | doladeny model po MRSTFT fine-tuningu | closed_loop_finetune.py |
| `analysis/profile.synth_config.json` | optimalizovane SynthConfig (global mode) | closed_loop_finetune.py |
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

### MRSTFT finetune: loss = NaN po epochach 80-130

**Projev:** loss klesa normalne do epochy ~80-100, pak narazove skoci na NaN a zustava NaN.
Vzorky z epochy pred NaN znejo normalne; vzorky v epose NaN obsahuji 100% NaN vzorky
("sumovy vystrel — ticho — sumovy vystrel").

**Pricina (B_net → cos(∞) → NaN):**

```
B_net vystup drift → velke log_B
→ B = exp(velke) → f_hzs = ∞ pro vysoke k
→ valid maska = 0 (nad Nyquistem)
→ ale NaN × 0 = NaN v IEEE 754
→ cos(∞) = NaN → celý audio tensor NaN
→ MRSTFT loss = NaN → gradient = NaN → permanentni NaN
```

**Oprava (uz aplikovana v `analysis/torch_synth.py`):**

```python
# Clamp log_B pred exp():
B = torch.exp(model.forward_B(mf).clamp(max=0.0)).squeeze()  # B ≤ 1.0

# isfinite guard ve valid masce:
valid = ((f_hzs < sr * 0.495) & torch.isfinite(f_hzs)).float().unsqueeze(1)
```

Grand piano B je fyzikalne v [1e-5, 0.01]. Clamp na max=0.0 dovoluje B ≤ 1.0 — dostatecna
rezerva pro trening, zaroven zamezuje f_hzs → ∞.

Kdyz NaN uz nastalo, je nutne zacit znovu od checkpointu pred epochou exploze (nebo od
zakladniho `profile.pt`). Model po NaN explozi nema zachranitelne vahy.

### UnicodeEncodeError na Windows (cp1252)

Pridej na zacatek skriptu:
```python
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
```
Vsechny skripty v tomto projektu to delaji automaticky.

### Nacteni starsich profile.pt (bez phi_net)

`load_model()` pouziva `strict=False` → phi_net se inicializuje nahodne, ostatni vahy
se zachovaji. Doporucene: spusti kratsim fine-tuningem (20-50 epoch) pro inicializaci phi_net:
```bash
python analysis/closed_loop_finetune.py \
    --mode finetune \
    --model analysis/profile.pt \
    --bank  "C:/SoundBanks/IthacaPlayer/ks-grand" \
    --epochs 50 --lr 5e-5
```
