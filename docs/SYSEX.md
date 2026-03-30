# IthacaCoreResonator — MIDI SysEx Parameter Protocol

Každý R/W parametr syntezátoru lze nastavit nebo číst přes standardní MIDI System Exclusive zprávu.
Implementace: `synth/sysex.h` + `synth/sysex.cpp`.

---

## Identifikátor zařízení

```
Manufacturer ID:  0x7D              (non-commercial / educational)
Device signature: 0x49 0x43 0x52   ("ICR" — IthacaCoreResonator)
```

---

## Schéma parametrického ID (16 bit)

```
 15  14  13  12 | 11  10   9   8   7   6   5   4   3   2   1   0
[  CATEGORY  ]  [            PARAMETER INDEX                    ]
   4 bity            12 bitů

ID = (category << 12) | param_index
```

### Tabulka kategorií

| Cat | Hex prefix | Název      | R/W    |
|-----|------------|------------|--------|
|  3  | `0x3___`   | STEREO     | R/W    |
|  4  | `0x4___`   | TIMBRE     | R/W    |
|  5  | `0x5___`   | LEVEL/ENV  | R/W    |

---

## Formát zpráv

Všechny bajty v SysEx musí být ≤ 0x7F (7-bit MIDI data).

### Typy zpráv

| Byte | Název            | Směr       | Délka      |
|------|------------------|------------|------------|
| 0x01 | SET_PARAM        | Host → ICR | 15 bajtů   |
| 0x02 | GET_PARAM        | Host → ICR | 9 bajtů    |
| 0x03 | PARAM_RESPONSE   | ICR → Host | 15 bajtů   |
| 0x04 | SET_ALL          | Host → ICR | 167 bajtů  |
| 0x05 | REQUEST_ALL      | Host → ICR | 7 bajtů    |
| 0x06 | ALL_PARAMS_DUMP  | ICR → Host | 167 bajtů  |

### SET_PARAM (15 bajtů)

```
F0  7D  49 43 52  01  [id0] [id1] [id2]  [v0] [v1] [v2] [v3] [v4]  F7
```

**Kódování ID** (16 bit → 3 × 7-bit):
```
id0 = (param_id >> 14) & 0x03
id1 = (param_id >>  7) & 0x7F
id2 =  param_id        & 0x7F
```

**Kódování float** (IEEE 754 → 5 × 7-bit):
```c
uint32_t raw;
memcpy(&raw, &value, 4);
v0 =  raw        & 0x7F   // bity  6–0
v1 = (raw >>  7) & 0x7F   // bity 13–7
v2 = (raw >> 14) & 0x7F   // bity 20–14
v3 = (raw >> 21) & 0x7F   // bity 27–21
v4 = (raw >> 28) & 0x0F   // bity 31–28
```

Celé číslo (pitch_glide_vel_thresh) se kóduje jako `(float)int_value`.

### GET_PARAM (9 bajtů)

```
F0  7D  49 43 52  02  [id0] [id1] [id2]  F7
```

ICR odpoví PARAM_RESPONSE se stejným kódováním.

### SET_ALL / ALL_PARAMS_DUMP (167 bajtů)

Záhlaví (6 B) + 20 param bloků × 8 B (3 B ID + 5 B hodnota) + F7 (1 B).

```
F0  7D  49 43 52  04/06
    [id0 id1 id2 v0 v1 v2 v3 v4]  × 20
F7
```

Pořadí bloků je fixní (viz tabulka níže, shora dolů).

---

## Kompletní tabulka R/W parametrů

### 0x3 — STEREO

| ID     | Konstanta             | SynthConfig field          | Default | Min   | Max   |
|--------|-----------------------|---------------------------|---------|-------|-------|
| 0x3000 | `PID_PAN_SPREAD`      | `pan_spread`              | 0.55    | 0.00  | 1.57  |
| 0x3001 | `PID_STEREO_DECORR`   | `stereo_decorr`           | 1.00    | 0.00  | 1.00  |
| 0x3002 | `PID_STEREO_BOOST`    | `stereo_boost`            | 1.00    | 0.00  | 4.00  |
| 0x3003 | `PID_PAN_TILT`        | `pan_tilt`                | 0.20    | 0.00  | 1.00  |
| 0x3004 | `PID_DECORR_MIDI_LO`  | `stereo_decorr_midi_lo`   | 40.0    | 0.0   | 127.0 |
| 0x3005 | `PID_DECORR_MIDI_HI`  | `stereo_decorr_midi_hi`   | 100.0   | 0.0   | 127.0 |
| 0x3006 | `PID_DECORR_MAX`      | `stereo_decorr_max`       | 0.45    | 0.00  | 1.00  |

### 0x4 — TIMBRE

| ID     | Konstanta                   | SynthConfig field           | Default | Min    | Max    |
|--------|-----------------------------|-----------------------------|---------|--------|--------|
| 0x4000 | `PID_BEAT_SCALE`            | `beat_scale`               | 1.00    | 0.00   | 5.00   |
| 0x4001 | `PID_HARMONIC_BRIGHTNESS`   | `harmonic_brightness`      | 0.00    | -2.00  | 4.00   |
| 0x4002 | `PID_EQ_STRENGTH`           | `eq_strength`              | 1.00    | 0.00   | 1.00   |
| 0x4003 | `PID_EQ_FREQ_MIN`           | `eq_freq_min`              | 400.0   | 20.0   | 2000.0 |
| 0x4004 | `PID_PITCH_GLIDE`           | `pitch_glide`              | 0.000   | 0.000  | 0.050  |
| 0x4005 | `PID_PITCH_GLIDE_TAU_MS`    | `pitch_glide_tau_ms`       | 80.0    | 1.0    | 500.0  |
| 0x4006 | `PID_PITCH_GLIDE_VEL_THRESH`| `pitch_glide_vel_thresh`   | 100     | 0      | 127    |

### 0x5 — LEVEL/ENV

| ID     | Konstanta                    | SynthConfig field            | Default | Min   | Max  |
|--------|------------------------------|------------------------------|---------|-------|------|
| 0x5000 | `PID_TARGET_RMS`             | `target_rms`                | 0.060   | 0.001 | 0.50 |
| 0x5001 | `PID_VEL_GAMMA`              | `vel_gamma`                 | 0.700   | 0.10  | 3.00 |
| 0x5002 | `PID_NOISE_LEVEL`            | `noise_level`               | 1.000   | 0.00  | 4.00 |
| 0x5003 | `PID_ONSET_MS`               | `onset_ms`                  | 3.00    | 0.00  | 50.0 |
| 0x5004 | `PID_LONGITUDINAL_PRECURSOR` | `longitudinal_precursor`    | 0.000   | 0.00  | 1.00 |
| 0x5005 | `PID_RENDER_REF_DURATION`    | `render_ref_duration_s`     | 3.000   | 0.10  | 60.0 |

Poznámka k `render_ref_duration_s`: referenční délka (v sekundách) použitá v level_scale formuli
pro real-time syntézu. Musí odpovídat délce render volání v tréninkovém pipeline (default 3.0 s).
Pro offline rendering (IthacaRenderServer) je výsledek normalizován post-render — hodnota ovlivňuje
pouze live playback.

**Celkem R/W parametrů:** 20
**SET_ALL / ALL_PARAMS_DUMP:** 167 bajtů

---

## Příklady zpráv

### SET pan_spread = 0.8 rad (0x3000)

```
ID 0x3000 → id0=0x00 id1=0x18 id2=0x00
float 0.8f → 0x3F4CCCCD → v0=0x4D v1=0x19 v2=0x53 v3=0x7A v4=0x03

F0 7D 49 43 52  01  00 18 00  4D 19 53 7A 03  F7
```

### GET stereo_decorr (0x3001)

```
F0 7D 49 43 52  02  00 18 01  F7
```

### SET beat_scale = 2.0 (0x4000)

```
ID 0x4000 → id0=0x00 id1=0x20 id2=0x00
float 2.0f → 0x40000000 → v0=0x00 v1=0x00 v2=0x00 v3=0x00 v4=0x04

F0 7D 49 43 52  01  00 20 00  00 00 00 00 04  F7
```

### REQUEST_ALL (žádost o dump)

```
F0 7D 49 43 52  05  F7
```

---

## Implementace

### Příjem (C++ strana)

SysEx přichází přes RtMidi callback v `midi_input.cpp`.
Zpracování delegováno do `sysex.cpp::sysexApply()`.

```cpp
// Decode + apply SET_PARAM nebo SET_ALL
sysexApply(msg_bytes, engine.getVoiceManager());

// Přečíst hodnotu parametru
float v;
sysexReadParam(PID_BEAT_SCALE, engine.getSynthConfig(), v);

// Sestavit odpověď na GET_PARAM
auto resp = sysexBuildParamResponse(param_id, v);
// resp odeslat přes MIDI output
```

### Odeslání patche (Python)

```python
import struct

def encode_float(f):
    raw = struct.unpack('<I', struct.pack('<f', f))[0]
    return [(raw >> (7*i)) & 0x7F for i in range(4)] + [(raw >> 28) & 0x0F]

def encode_id(pid):
    return [(pid >> 14) & 0x03, (pid >> 7) & 0x7F, pid & 0x7F]

def set_param(pid, value):
    return bytes([0xF0, 0x7D, 0x49, 0x43, 0x52, 0x01]
                 + encode_id(pid) + encode_float(value) + [0xF7])

# Příklad: beat_scale = 1.5
msg = set_param(0x4000, 1.5)
midi_out.send_message(list(msg))
```

### Clamping hodnot

Všechny příchozí hodnoty jsou clamped na definovaný rozsah (viz tabulky).
Neznámá `param_id` → `sysexApplyParam()` vrátí `false`, zpráva se ignoruje.

### Logování (ICR strana)

SysEx callback loguje přes `Logger` (stdout v interaktivním režimu, stderr s `--verbose`, muted v server kontextu):

```
[INF][SYSEX] Applied type=0x01 (15 bytes)    — SET_PARAM OK
[INF][SYSEX] Applied type=0x04 (159 bytes)   — SET_ALL OK
[WRN][SYSEX] Unhandled type=0x02 (10 bytes)  — GET_PARAM (žádný MIDI výstup)
[WRN][SYSEX] Unhandled type=0x05 (7 bytes)   — REQUEST_ALL (žádný MIDI výstup)
[WRN][SYSEX] Invalid SysEx (10 bytes) — ignored  — špatný MFR/SIG
```

GET_PARAM (0x02) a REQUEST_ALL (0x05) jsou validovány a logovány, ale odpověď není odesílána —
ICR nemá MIDI výstupní port. Tyto typy jsou připraveny pro budoucí bi-directional patch editor.

---

## Testování

### Prerekvizity

- **loopMIDI** (Windows) nebo ekvivalentní virtuální MIDI kabel (macOS: IAC Driver)
- `python-rtmidi`: `pip install python-rtmidi`
- `analysis/sysex_test.py` — testovací helper (součást repozitáře)

### Postup

**1. Spustit IthacaCoreResonator** s MIDI vstupem na loopMIDI:

```
IthacaCoreResonator.exe analysis/params-ks-grand.json 0
```

Ověřit v logu:
```
[MIDI] Opened: loopMIDI Port 0
```

**2. Spustit test helper** (druhý terminál):

```
python analysis/sysex_test.py
```

Auto-vybere první non-GS port (loopMIDI výstup). Explicitní port: `--port 1`.

Dostupné přepínače:
```
--list     vypíše MIDI výstupní porty a skončí
--port N   zvolí port N
--verify   pouze codec round-trip test (bez MIDI)
```

### Výsledky ověřeného testu (2026-03-30)

```
python analysis/sysex_test.py --port 1
```

ICR log:
```
[INF][SYSEX] Applied type=0x01 (15 bytes)   × 11  — SET_PARAM (8 params + 3 clamp testy)
[INF][SYSEX] Applied type=0x04 (167 bytes)  × 2   — SET_ALL (defaults, restore)
[WRN][SYSEX] Unhandled type=0x02 (10 bytes) × 3   — GET_PARAM (očekáváno)
[WRN][SYSEX] Unhandled type=0x05 (7 bytes)  × 1   — REQUEST_ALL (očekáváno)
[WRN][SYSEX] Invalid SysEx (10 bytes) — ignored   × 1   — bad signature test OK
```

Codec round-trip: **20 params + 20 random floats — vše OK**.
