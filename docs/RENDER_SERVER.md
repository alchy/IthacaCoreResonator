# IthacaRenderServer — Headless Offline Render Server

## Proč existuje

IthacaCoreResonator (hlavní syntetizér) běží v reálném čase s audio zařízením a MIDI vstupem.
Pro trénování surrogate modelu je potřeba renderovat tisíce not rychle, bez latence audio zařízení
a bez GUI. `IthacaRenderServer` je oddělený binární soubor, který sdílí celý syntetický engine,
ale komunikuje přes **TCP JSON protokol** — ideální pro Python subprocess IPC.

Architektura sdílí DSP kód (`RENDER_SYNTH_SOURCES` ⊂ `SYNTH_SOURCES` v CMakeLists.txt), takže
jakákoliv změna enginu se automaticky projeví ve všech binárních souborech po jediném buildu.

---

## Architektura

```
Python training loop
       │  subprocess + TCP socket
       ▼
IthacaRenderServer.exe
   ├── server_main.cpp             — arg parsing, Logger init, heap-alloc RenderServer
   ├── synth/render_server.cpp     — TCP listener + JSON command dispatcher
   ├── synth/offline_renderer.cpp  — headless note renderer (no audio device)
   ├── synth/voice_manager.cpp     — shared synth engine (same as ICR)
   └── synth/sysex.cpp             — SysEx codec (set/get config via byte arrays)
```

**Není součástí serveru:**
- `resonator_engine.cpp` — miniaudio real-time audio device
- `midi_input.cpp` — RtMidi MIDI input
- `dsp/dsp_chain.cpp` — BBE + limiter post-processing (pouze real-time GUI path)
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
IthacaRenderServer [params.json] [--port <N>] [--log <path>] [--no-log]
```

| Argument | Výchozí | Popis |
|---|---|---|
| `params.json` | `analysis/params-ks-grand.json` | Physics parameter tabulka |
| `--port <N>` | `9876` | TCP port, musí odpovídat `--port` klienta |
| `--log <path>` | `analysis/runtime-logs/render-server.log` | Diagnostický log do souboru |
| `--no-log` | — | Zakáže veškeré logování |

Na startu server naváže TCP spojení na `127.0.0.1:PORT` a na každém přijatém spojení
odešle `{"status":"ready"}` jako první řádek.

---

## Protokol

**Transport:** TCP socket, `127.0.0.1:PORT`.
**Formát:** jeden JSON objekt na řádek, zakončený `\n`.
**Směr:** klient → server (příkaz), server → klient (odpověď).
**Odpovědi** vždy obsahují `"status":"ok"` nebo `"status":"error","msg":"..."`.
**Stdin / stdout / stderr** nejsou používány vůbec.

### Příkazy

#### `ping`
```json
→ {"cmd":"ping"}
← {"status":"ok"}
```
Ověření živosti serveru.

#### `render`
```json
→ {"cmd":"render","midi":60,"vel":3,"sr":44100,"duration":3.0,"output":"exports/note.wav"}
← {"status":"ok","frames":132300}
```
| Pole | Výchozí | Popis |
|---|---|---|
| `midi` | — | MIDI nota (21–108) |
| `vel` | — | **Velocity band 0–7** (shodné s trénovací konvencí, viz sekci níže) |
| `sr` | `44100` | Sample rate v Hz |
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
Znovu načte physics params JSON za běhu.
Bez `"params"` klíče použije původní cestu.

#### `quit`
```json
→ {"cmd":"quit"}
← {"status":"ok"}
```
Server korektně ukončí smyčku a exits.

---

## Velocity konvence

**Render server i trénovací data používají velocity band 0–7** (ne raw MIDI 0–127).

| `vel` band | Přibližný MIDI ekvivalent | vel_gain = ((band+1)/8)^γ |
|---|---|---|
| 0 | 0 | 0.0 (ticho) |
| 1 | 18 | ~0.25 |
| 3 | 54 | ~0.50 |
| 5 | 90 | ~0.72 |
| 7 | 127 | 1.0 |

MIDI klaviatura (0–127) se mapuje automaticky ve VoiceManageru: `vel_pos = vel * 7 / 127`.

---

## Python klient

```python
from analysis.render_client import RenderClient

with RenderClient("build/bin/Release/IthacaRenderServer.exe",
                  "analysis/params-nn-profile-ks-grand.json") as rc:

    rc.ping()
    rc.set_config(beat_scale=1.5, harmonic_brightness=0.5)
    cfg = rc.get_config()          # dict se všemi 19 parametry

    # vel = velocity band 0-7
    n = rc.render(midi=60, vel=3, duration=0.0,   # duration=0 → auto-detect
                  output="exports/m060_vel3.wav")
    rc.reload(params="analysis/params-nn-profile-ks-grand.json")
```

### Batch render

```python
from analysis.render_client import batch_render

jobs = [(midi, vel, f"exports/m{midi:03d}_v{vel}.wav")
        for midi in range(21, 109) for vel in [1, 3, 5, 7]]

frame_counts = batch_render(
    "build/bin/Release/IthacaRenderServer.exe",
    "analysis/params-nn-profile-ks-grand.json",
    jobs, sr=44100, duration=0.0   # duration=0 → auto-detect
)
```

---

## Logování

Všechna diagnostika jde do souboru `--log`. Stdin/stdout/stderr nejsou použity vůbec.

---

## Implementační poznámky

- **Stack overflow fix**: `RenderServer` je alokován na heapu (`std::make_unique<RenderServer>()`)
  protože `ResonatorVoiceManager` obsahuje 88 hlasů × ~5 KB = ~400 KB → přetéká zásobník.
- **SysEx sync**: po aplikaci SysEx se volá `renderer_.setSynthConfig(vm.getSynthConfig())`
  aby `cfg_` odpovídal stavu `vm_` a `get_config` vrátil aktuální hodnoty.
- **Sdílený DSP kód**: `RENDER_SYNTH_SOURCES` ⊂ `SYNTH_SOURCES` v `CMakeLists.txt`.
  Změna enginu se propaguje do všech targetů při buildu.
- **Post-render RMS normalizace**: `OfflineRenderer::renderNote()` po celkovém renderu
  přepočítá skutečný RMS a normalizuje výstup na `target_rms * vel_gain`. Opravuje
  inter-string fázové cross-termy (náhodné počáteční fáze způsobují rozptyl u not
  s málo parciály). Výsledný RMS je přesně `target_rms * vel_gain`.
- **Crest factor u vysokých not**: při `duration=0` (auto-detect) se normalizace počítá
  přes skutečně vyrendovanou délku → správné výsledky. Při fixní `duration=3.0` u krátkých
  not (m096–m108, rychlý útlum) může normalizace nadměrně zesílit počáteční transient.
  Pro export vzorků doporučujeme `duration=0`.
