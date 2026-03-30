# IthacaRenderServer — Headless Offline Render Server

## Proč existuje

IthacaCoreResonator (hlavní syntetizér) běží v reálném čase s audio zařízením a MIDI vstupem.
Pro trénování surrogate modelu je potřeba renderovat tisíce not rychle, bez latence audio zařízení
a bez GUI. `IthacaRenderServer` je oddělený binární soubor, který sdílí celý syntetický engine,
ale komunikuje přes stdin/stdout JSON protokol — ideální pro Python subprocess IPC.

Architektura sdílí DSP kód (`SYNTH_SOURCES`), takže jakákoliv změna enginu se automaticky projeví
ve všech třech binárních souborech (IthacaCoreResonator, IthacaCoreResonatorGUI, IthacaRenderServer)
po jediném buildu.

---

## Architektura

```
Python training loop
       │  subprocess
       ▼
IthacaRenderServer.exe
   ├── server_main.cpp        — arg parsing, Logger init, heap-alloc RenderServer
   ├── synth/render_server.cpp — JSON command dispatcher
   ├── synth/offline_renderer.cpp — headless note renderer (no audio device)
   ├── synth/voice_manager.cpp    — shared synth engine (same as ICR)
   ├── synth/sysex.cpp            — SysEx codec (set/get config via byte arrays)
   └── dsp/dsp_chain.cpp          — limiter + BBE post-processing
```

**Není součástí serveru:**
- `resonator_engine.cpp` — miniaudio real-time audio device
- `midi_input.cpp` — RtMidi MIDI input
- žádné audio systémové knihovny (WinMM, CoreMIDI, ALSA/JACK)

---

## Sestavení

```bash
cmake --build build --config Release --target IthacaRenderServer
# výstup: build/bin/Release/IthacaRenderServer.exe
```

---

## Spuštění

```
IthacaRenderServer.exe [params.json] [--verbose]
```

| Argument | Výchozí | Popis |
|---|---|---|
| `params.json` | `analysis/params-ks-grand.json` | Physics parameter tabulka |
| `--verbose` / `-v` | muted | Log výstup na stderr |

Na startu server zapíše `{"status":"ready"}` na stdout a čeká na příkazy.

---

## Protokol

**Formát:** jeden JSON objekt na řádek (stdin → příkaz, stdout → odpověď).
**Odpovědi** vždy obsahují `"status":"ok"` nebo `"status":"error","msg":"..."`.
**stderr** je volný pro diagnostický výstup (neinterferuje s protokolem).

### Příkazy

#### `ping`
```json
→ {"cmd":"ping"}
← {"status":"ok"}
```
Ověření živosti serveru.

#### `render`
```json
→ {"cmd":"render","midi":60,"vel":80,"sr":44100,"duration":3.0,"output":"exports/note.wav"}
← {"status":"ok","frames":132300}
```
| Pole | Výchozí | Popis |
|---|---|---|
| `midi` | — | MIDI nota (21–108) |
| `vel` | — | MIDI velocity (1–127) |
| `sr` | — | Sample rate v Hz |
| `duration` | `0` | Délka v sekundách; `0` = auto-detect silence tail |
| `output` | — | Výstupní WAV cesta (parent adresáře se vytvoří automaticky) |

Výstupní formát: IEEE float32 WAV, stereo, zadaný `sr`.

#### `set_config`
```json
→ {"cmd":"set_config","params":{"beat_scale":1.5,"eq_strength":0.8}}
← {"status":"ok"}
```
Partial update `SynthConfig` — nastavují se pouze uvedené klíče. Klíče odpovídají
názvům fieldů v `SynthConfig` (viz `docs/SYSEX.md` — tabulka R/W parametrů).

#### `get_config`
```json
→ {"cmd":"get_config"}
← {"status":"ok","params":{"beat_scale":1.0,"pan_spread":0.55,...}}
```
Vrátí kompletní `SynthConfig` jako JSON objekt (19 R/W parametrů).

#### `sysex`
```json
→ {"cmd":"sysex","bytes":[240,125,73,67,82,1,0,32,0,0,0,0,0,4,247]}
← {"status":"ok","applied":true}
```
Aplikuje raw ICR SysEx zprávu (jako pole int hodnot 0–255).
Po úspěšné aplikaci synchronizuje `SynthConfig` — výsledek viditelný přes `get_config`.
Používá se pro verifikaci SysEx routování (viz `analysis/sysex_test.py --verify-routing`).

#### `reload`
```json
→ {"cmd":"reload","params":"analysis/params-ks-grand.json"}
← {"status":"ok"}
```
Znovu načte physics params JSON za běhu (např. po aktualizaci `extract_params.py`).
Bez `"params"` klíče použije původní cestu.

#### `quit`
```json
→ {"cmd":"quit"}
← {"status":"ok"}
```
Server korektně ukončí smyčku a exits.

---

## Python klient

```python
from analysis.render_client import RenderClient

with RenderClient("build/bin/Release/IthacaRenderServer.exe",
                  "analysis/params-ks-grand.json") as rc:

    rc.ping()
    rc.set_config(beat_scale=1.5, harmonic_brightness=0.5)
    cfg = rc.get_config()          # dict se všemi 19 parametry
    n = rc.render(midi=60, vel=80, duration=3.0,
                  output="exports/m060_vel80.wav")
    rc.reload(params="analysis/params-ks-grand.json")
```

### Batch render

```python
from analysis.render_client import batch_render

jobs = [(midi, vel, f"exports/m{midi:03d}_v{vel:03d}.wav")
        for midi in range(21, 109) for vel in [40, 80, 110]]

frame_counts = batch_render(
    "build/bin/Release/IthacaRenderServer.exe",
    "analysis/params-ks-grand.json",
    jobs, sr=44100, duration=3.0
)
```

---

## Logování

| Kontext | Logger nastavení |
|---|---|
| Výchozí (headless, subprocess) | muted — žádný výstup |
| `--verbose` flag | stderr — diagnostické zprávy |

Stdout je vyhrazen výhradně pro JSON protokol.

---

## Implementační poznámky

- **Stack overflow fix**: `RenderServer` je alokován na heapu (`std::make_unique<RenderServer>()`)
  protože `ResonatorVoiceManager` obsahuje 88 hlasů × ~5 KB = ~400 KB → přetéká zásobník.
- **SysEx sync**: po aplikaci SysEx se volá `renderer_.setSynthConfig(vm.getSynthConfig())`
  aby `cfg_` odpovídal stavu `vm_` a `get_config` vrátil aktuální hodnoty.
- **Sdílený DSP kód**: `RENDER_SYNTH_SOURCES` ⊂ `SYNTH_SOURCES` v `CMakeLists.txt`.
  Změna enginu se propaguje do všech targetů při buildu.
