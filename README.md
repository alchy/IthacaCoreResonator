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

## Jak spustit

### Real-time přehrávání (GUI)

```bat
cd build\bin\Release
IthacaCoreResonatorGUI.exe --params soundbanks\params-ks-grand-ft.json
```

Keyboard fallback (bez MIDI klávesnice):
```
a s d f g h j k  →  C4 D4 E4 F4 G4 A4 B4 C5
z                →  sustain pedal
q                →  quit
```

### Build

```bat
cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Trénink od nuly

```bash
# 1. Supervisory training
python -m analysis.train_instrument_profile \
  --in analysis/params-ks-grand.json \
  --out analysis/params-nn-profile-ks-grand-v2.json \
  --model analysis/profile-v2.pt \
  --epochs 800

# 2. MRSTFT fine-tuning
python -m analysis.closed_loop_finetune \
  --mode finetune \
  --model analysis/profile-v2.pt \
  --bank /cesta/k/nahravkam \
  --out analysis/profile-v2-finetuned.pt \
  --epochs 300
```

---

## Aktuální stav projektu

Podrobná historie vývoje, rozhodnutí a výsledky jednotlivých fází: [PROGRESS.md](PROGRESS.md)



- Signal chain C++ ↔ Python parita ověřena (Phase 1 ✅)
- Modulární refaktor kódu dokončen (Phase 2 ✅)
- KI-1 noise formula fix, GUI live slidery, regresní baseline (Phase 3 ✅)
- V2 model natrénován s 2-string proxy a phi_net architekturou
- Zbývá: ověření v2 profilu přes `compare_cpp_python --batch`, případný export finetuned params JSON
