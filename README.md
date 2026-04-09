# IthacaCoreResonator

Fyzikálně založený aditivní syntetizér klavíru — C++17 real-time engine s Python training pipeline.

---

## Co projekt řeší

Cílem je vytvořit syntetizér, který zní jako skutečný klavír, ale je plně parametrický a deterministický. Místo vzorkování (sampling) se zvuk generuje čistě z fyzikálních rovnic — každá struna je simulována jako sada harmonických parciálů s reálnou inharmonicitou, dozvukem a vzájemným beatem mezi strunami.

Druhý cíl je, aby tento syntetizér šlo použít jako **diferenciabilní proxy** v neural network training loopu — tedy aby bylo možné trénovat neuronovou síť, která odhaduje parametry syntetizéru z nahrávek skutečného klavíru, a gradient šel zpět přes syntézu.

---

## Jak to funguje — celý pipeline

```
Nahrávky klavíru (WAV)
        │
        ▼
[1] train_instrument_profile.py
    Neuronová síť (InstrumentProfile, ~99k params) se naučí
    mapovat MIDI nota + velocity → fyzikální parametry syntézy.
    Výstup: profile-v2.pt  +  params-ks-grand-v2-nn.json
        │
        ▼
[2] closed_loop_finetune.py
    MRSTFT fine-tuning: diferenciabilní PyTorch syntetizér
    (torch_synth.py) renderuje noty, loss se počítá vůči
    originálním nahrávkám, gradienty jdou zpět do sítě.
    Výstup: profile-v2-finetuned.pt
        │
        ▼
[3] params JSON (soundbanks/*.json)
    Vygenerovaná banka fyzikálních parametrů pro 88 not × 8
    velocity vrstev. C++ syntetizér ji načte při startu.
        │
        ▼
[4] C++ IthacaCoreResonator
    Real-time přehrávání, MIDI vstup, GUI.
    Offline renderer pro export WAV.
```

---

## Fyzikální model syntézy

Každá nota je modelována jako součet inharmonických parciálů:

```
fk = k · f0 · sqrt(1 + B·k²)
```

kde `B` je koeficient inharmonicity (extrahovano z nahrávek per-nota). Amplituda každého parciálu sleduje bi-exponenciální obálku simulující útok a dozvuk. Klavírní nota má 2–3 struny mírně rozladěné — jejich beating vytváří charakteristický "živý" zvuk. Výstup je stereo se Schroederovou dekorelací, spektrální EQ korekcí a M/S stereo widening.

---

## Architektura C++ projektu

Tři build targety:

| Binárka | Použití |
|---|---|
| `IthacaCoreResonator.exe` | CLI real-time přehrávání + MIDI |
| `IthacaCoreResonatorGUI.exe` | Stejný engine + ImGui overlay s live parametry |
| `IthacaRenderServer.exe` | Headless TCP server pro Python training loop |

Zdrojový kód je organizován do tří modulů:

```
synth/core/       — sdílené typy: SynthConfig, NoteParams, BiquadEQ, NoteLUT
synth/realtime/   — real-time engine: ResonatorVoice, VoiceManager, ResonatorEngine
synth/offline/    — offline renderer + RenderServer (TCP JSON protokol)
```

---

## Soubory v repozitáři

### Soundbanky (`soundbanks/`)

| Soubor | Popis |
|---|---|
| `params-ks-grand-nn.json` | Parametry z první generace NN (1-string proxy) |
| `params-ks-grand-ft.json` | MRSTFT finetuned — první generace, nejlépe ověřená |
| `params-ks-grand-v2-nn.json` | Parametry z v2 NN (2-string proxy s phi_net) |

### Modely (`analysis/`)

| Soubor | Popis |
|---|---|
| `profile-v2.pt` | Supervisory trained model, 800 epoch, eval loss 1.298 |
| `profile-v2-finetuned.pt` | MRSTFT finetuned, 300 epoch, mean MRSTFT 8.369 |
| `params-nn-profile-ks-grand-v2.json` | Plná 88×8 banka vygenerovaná z profile-v2.pt |

### Python pipeline (`analysis/`)

| Soubor | Popis |
|---|---|
| `physics_synth.py` | Referenční Python implementace syntézy (ground truth) |
| `torch_synth.py` | Diferenciabilní PyTorch proxy pro training loop |
| `train_instrument_profile.py` | Supervisory trénink NN |
| `closed_loop_finetune.py` | MRSTFT fine-tuning |
| `compare_cpp_python.py` | Numerické porovnání C++ vs Python výstupu (MRSTFT/MSE/SNR) |
| `render_client.py` | Python klient pro komunikaci s RenderServerem přes TCP |

---

## Co je MRSTFT a proč ho používáme

MRSTFT (Multi-Resolution Short-Time Fourier Transform) je perceptuální loss funkce — měří spektrální vzdálenost mezi dvěma audio signály na více časových/frekvenčních rozlišeních najednou. Hodí se pro audio protože lidské ucho vnímá zvuk logaritmicky, ne lineárně.

Hodnoty v projektu:

- **~1.0–1.5** — stochastický floor (dvě identické syntézy stejné noty se liší kvůli náhodné počáteční fázi strun)
- **~1.0–2.5** — C++ vs Python, konvergovaný střed klavíru
- **~8–10** — syntetizér vs originální nahrávka (fyzikální model není perfektní replika)

---

## Zprovoznění

### Požadavky

- CMake ≥ 3.16
- MSVC BuildTools 2022 (Windows) nebo GCC/Clang (Linux/macOS)
- Python 3.10+ s PyTorch, numpy, scipy, soundfile
- Internet při prvním buildu (FetchContent stáhne GLFW + Dear ImGui)

### Build (Windows)

Spustit z VS Developer PowerShell nebo po `vcvars64.bat`:

```bat
cd C:\cesta\k\ICR

cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target IthacaCoreResonator
cmake --build build --config Release --target IthacaCoreResonatorGUI
cmake --build build --config Release --target IthacaRenderServer
```

Binárky skončí v `build\bin\Release\`. Soundbanky se automaticky zkopírují do `build\bin\Release\soundbanks\`.

### Build (Linux / macOS)

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Na Linuxu je potřeba `libasound-dev` (ALSA).

---

## Spuštění syntetizéru

### GUI — doporučeno pro hraní

```bat
cd build\bin\Release
IthacaCoreResonatorGUI.exe --params soundbanks\params-ks-grand-ft.json
```

Přepínače:

| Přepínač | Popis |
|---|---|
| `--params <cesta>` | Soundbanka (JSON), **povinné** |
| `--config <cesta>` | Volitelný SynthConfig JSON (beat_scale, eq_strength, …) |
| `--port <N>` | Index MIDI vstupního portu (default: 0; na startu vypíše dostupné porty) |

Keyboard fallback (bez MIDI klávesnice):
```
a s d f g h j k  →  C4 D4 E4 F4 G4 A4 B4 C5
z                →  sustain pedal (toggle)
q                →  quit
```

### CLI — headless real-time

```bat
IthacaCoreResonator.exe --params soundbanks\params-ks-grand-ft.json
```

Stejné přepínače a keyboard fallback jako GUI.

### RenderServer — pro Python pipeline

```bat
IthacaRenderServer.exe --params soundbanks\params-ks-grand-ft.json --port 9876
```

Server po startu pošle `{"status":"ready"}` na každé přijaté TCP spojení.
Logy jdou do `analysis/runtime-logs/render-server.log`.

TCP protokol (jeden JSON per řádek, `\n`-terminated):

```json
{"cmd":"ping"}
{"cmd":"render","midi":60,"vel":3,"sr":44100,"duration":3.0,"output":"exports/note.wav"}
{"cmd":"set_config","params":{"beat_scale":1.5,"eq_strength":0.8,"noise_level":1.0}}
{"cmd":"get_config"}
{"cmd":"reload","params":"soundbanks/jiny-profil.json"}
{"cmd":"quit"}
```

---

## Generování soundbanky

Celý pipeline ve třech krocích:

### Krok 1 — Supervisory trénink NN

```bash
python -m analysis.train_instrument_profile \
  --in  analysis/params-ks-grand.json \       # extrahované parametry z nahrávek
  --out analysis/params-nn-profile-ks-grand-v2.json \  # výstupní banka
  --model analysis/profile-v2.pt \            # uložený model
  --epochs 800 \
  --hidden 64 \
  --lr 3e-3 \
  --eval-every 10
```

| Přepínač | Popis |
|---|---|
| `--in` | Vstupní JSON s extrahovanými parametry (ground truth z nahrávek) |
| `--out` | Výstupní params JSON (88 not × 8 vel vrstev) |
| `--model` | Cesta pro uložení/načtení `.pt` modelu |
| `--epochs` | Počet epoch (doporučeno 800) |
| `--hidden` | Velikost skrytých vrstev sítě (default: 64) |
| `--lr` | Learning rate (default: 3e-3) |
| `--midi-from/--midi-to` | Omezení rozsahu not (default: celý klavír) |
| `--eval-every N` | Eval každých N not (default: 10) |

### Krok 2 — MRSTFT fine-tuning

```bash
python -m analysis.closed_loop_finetune \
  --mode finetune \
  --model analysis/profile-v2.pt \
  --bank  C:/SoundBanks/ddsp/ks-grand \       # adresář s originálními WAV
  --out   analysis/profile-v2-finetuned.pt \
  --epochs 300 \
  --lr 1e-4 \
  --batch-size 8 \
  --seed 7 \
  --log analysis/runtime-logs/finetune-v2.log
```

| Přepínač | Popis |
|---|---|
| `--mode` | `finetune` (MRSTFT loss), `eval` (jen vyhodnocení), `global` (globální params) |
| `--model` | Vstupní `.pt` model |
| `--bank` | Adresář s WAV soubory ve formátu `m060-vel3-f44.wav` |
| `--out` | Výstupní finetuned `.pt` model |
| `--epochs` | Počet epoch (doporučeno 300) |
| `--lr` | Learning rate (doporučeno 1e-4) |
| `--batch-size` | Not per gradient step (default: 8) |
| `--seed` | Random seed pro reprodukovatelnost |
| `--noise-level` | Úroveň šumu v syntéze (default: 1.0) |
| `--beat-scale` | Škálování beatingu (default: 1.0) |
| `--render-dir` | Ukládat WAV checkpointy každých N epoch |
| `--log` | Cesta k logu |

### Krok 3 — Nasazení banky

```bat
copy analysis\params-nn-profile-ks-grand-v2.json soundbanks\params-ks-grand-v2-nn.json
copy soundbanks\params-ks-grand-v2-nn.json build\bin\Release\soundbanks\
```

---

## Ověření C++ vs Python parity

```bash
# Spustit nejdřív RenderServer (v jiném terminálu):
cd build\bin\Release && IthacaRenderServer.exe --params soundbanks\params-ks-grand-ft.json

# Rychlý test — C4 vel 3:
python -m analysis.compare_cpp_python

# Celý klavír:
python -m analysis.compare_cpp_python --batch --save

# Jedna nota se spektrogramem:
python -m analysis.compare_cpp_python --midi 60 --vel 3 --plot --save

# Regresní kontrola (CI použití):
python -m analysis.compare_cpp_python --batch --check

# Uložit aktuální výsledky jako nový baseline:
python -m analysis.compare_cpp_python --batch --save-baseline
```

| Přepínač | Popis |
|---|---|
| `--midi N` | MIDI číslo noty (default: 60 = C4) |
| `--vel N` | Velocity band 0–7 (default: 3) |
| `--duration S` | Délka renderované noty v sekundách |
| `--params <cesta>` | Params JSON pro RenderServer |
| `--batch` | Test celého reprezentativního rozsahu not |
| `--plot` | Zobrazit spektrogram |
| `--save` | Uložit WAV a grafy |
| `--check` | Porovnat s baseline, exit 1 při regresi |
| `--save-baseline` | Uložit aktuální výsledky jako nový baseline (+0.3 margin) |

---

## SynthConfig parametry

Globální parametry syntézy lze předat přes `--config synth_config.json` nebo měnit live v GUI:

| Parametr | Default | Popis |
|---|---|---|
| `beat_scale` | 1.0 | Škálování beatingu mezi strunami (0 = žádný beating, 3 = výrazný) |
| `eq_strength` | 1.0 | Intenzita spektrální EQ korekce (0 = vypnuto, 1 = plná korekce) |
| `noise_level` | 1.0 | Úroveň šumové složky (0 = bez šumu, 2 = zdvojnásobeno) |
| `pan_spread` | 0.55 | Prostorové rozložení strun ve stereo poli (radiány) |
| `pan_tilt` | 0.20 | Náklon stereo obrazu (bas vlevo, výšky vpravo) |
| `stereo_decorr` | 1.0 | Intenzita Schroederovy dekorelace L/R kanálů |
| `onset_ms` | 10.0 | Délka onset rampy v ms |
| `vel_gamma` | 1.0 | Gamma křivka velocity → hlasitost |
| `target_rms` | 0.06 | Cílová RMS úroveň výstupu (~−24 dBFS) |

---

## Aktuální stav projektu

Podrobná historie vývoje, rozhodnutí a výsledky jednotlivých fází: [PROGRESS.md](PROGRESS.md)



- Signal chain C++ ↔ Python parita ověřena (Phase 1 ✅)
- Modulární refaktor kódu dokončen (Phase 2 ✅)
- KI-1 noise formula fix, GUI live slidery, regresní baseline (Phase 3 ✅)
- V2 model natrénován s 2-string proxy a phi_net architekturou
- Zbývá: ověření v2 profilu přes `compare_cpp_python --batch`, případný export finetuned params JSON
