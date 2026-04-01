# IthacaCoreResonator — Developer Reference (branch: dev-ICR)

Physics-based inharmonic additive piano synthesizer in C++17.
Core goal: offline renderer output must match `analysis/physics_synth.py` 1:1 (MRSTFT loss),
to serve as a differentiable proxy in a neural network training loop.

---

## Architecture

### Three build targets

| Target | Binary | Purpose |
|---|---|---|
| `IthacaCoreResonator` | CLI | Real-time playback, MIDI input, keyboard fallback |
| `IthacaCoreResonatorGUI` | GUI | Same real-time engine + ImGui overlay |
| `IthacaRenderServer` | Server | Headless TCP server for Python training-loop IPC |

```
IthacaCoreResonator/
├── main.cpp                     — CLI entry point
├── gui_main.cpp                 — GUI entry point
├── server_main.cpp              — RenderServer entry point
│
├── synth/
│   ├── resonator_voice.cpp/h    — Per-voice physics DSP (core)
│   ├── voice_manager.cpp/h      — 88-voice polyphony manager
│   ├── note_lut.cpp/h           — params.json → NoteParams lookup table
│   ├── note_params.h            — PartialParams, NoiseParams, NoteParams, NoteLUT
│   ├── synth_config.h           — SynthConfig (global rendering parameters)
│   ├── synth_config_io.h/cpp    — JSON I/O for SynthConfig
│   ├── biquad_eq.h/cpp          — Per-voice spectral EQ (biquad cascade)
│   ├── sysex.h/cpp              — SysEx parameter update protocol
│   │
│   ├── resonator_engine.cpp/h   — Real-time engine (miniaudio + MIDI queue)
│   ├── midi_input.h/cpp         — RtMidi wrapper
│   │
│   ├── offline_renderer.cpp/h   — Headless note renderer (no audio device)
│   └── render_server.cpp/h      — TCP JSON IPC server
│
├── dsp/
│   ├── limiter/                 — Peak limiter (real-time chain only)
│   ├── bbe/                     — BBE sonic maximizer (real-time chain only)
│   └── dsp_chain.cpp/h          — Limiter + BBE wrappers
│
├── gui/
│   └── resonator_gui.cpp/h      — ImGui UI (voice monitor, last-note display)
│
├── analysis/
│   ├── physics_synth.py         — Python ground-truth synthesizer (numpy)
│   ├── compare_cpp_python.py    — MRSTFT/MSE/SNR comparison tool
│   └── closed_loop_finetune.py  — MRSTFT-based parameter fine-tuning
│
├── soundbanks/
│   ├── params-ks-grand-nn.json  — NN-trained parameter profile
│   └── params-ks-grand-ft.json  — MRSTFT-finetuned profile (use this for playback)
│
└── third_party/
    ├── miniaudio.h              — Cross-platform audio (header-only)
    ├── RtMidi.cpp/h             — MIDI I/O
    └── nlohmann/json.hpp        — JSON parsing
```

### Data flow

```
params.json ──► NoteLUT [88×8]
                    │
    MIDI noteOn ──► VoiceManager ──► ResonatorVoice × N
                                          │  processBlockUninterleaved()
                                          ▼
                                    Signal chain (per block, per voice):
                                    1. Oscillator sum (partials × strings)
                                    2. Longitudinal precursor (MIDI < 50)
                                    3. Noise L+R (independent channels)
                                    4. Release ramp
                                    5. Schroeder all-pass decorr  ← BEFORE EQ
                                    6. BiquadEQ (spectral correction)
                                    7. M/S stereo width
                                    8. Onset ramp                 ← LAST
                                    9. Accumulate into output

Real-time path:  output → DSP chain (Limiter + BBE) → miniaudio → speakers
Offline path:    output → post-hoc RMS normalization → WAV / TCP response
```

---

## Stereo Architecture

### Parameter sources

| Parameter | Source | Description |
|---|---|---|
| `pan_spread` | `SynthConfig` (from `synth_config.json`, default 0.55 rad) | Spatial spread of strings across the stereo field |
| `pan_tilt` | `SynthConfig` (default 0.20) | Keyboard tilt: bass left, treble right |
| `n_strings` | `params.json` per-note field | Number of strings (1/2/3 depending on piano register) |
| `width_factor` | `params.json` per-note field | M/S stereo width multiplier (not pan spread) |
| `stereo_boost` | `SynthConfig` | Additional M/S side-channel boost |
| `stereo_decorr` / `decorr_max` | `SynthConfig` | Schroeder all-pass depth control |

`pan_spread` is **not** a per-note value from `params.json` — it comes from the global `SynthConfig`
(loaded from `soundbanks/params-ks-grand-ft.synth_config.json` or defaults to 0.55).

### String count per register

Real piano string counts are extracted per-note during parameter fitting and stored in `params.json`:

| Register | `n_strings` | Notes |
|---|---|---|
| Deep bass (~A0–F#2) | 1 | Single string, no beating |
| Mid bass | 2 | Two strings, detuned ±beat_hz/2 |
| Treble | 3 | Three strings, {−beat/2, 0, +beat/2} |

### String spatial placement (equal-power panning)

```
center = π/4 + (midi − 64.5) / 87 · pan_tilt   ← bass left, treble right
half   = pan_spread / 2

n=1:  angles = [center]
n=2:  angles = [center − half,  center + half]
n=3:  angles = [center − half,  center,  center + half]

pan_L[s] = cos(angle[s])
pan_R[s] = sin(angle[s])                          ← equal-power

output L += signal · pan_L[s] / n_strings
output R += signal · pan_R[s] / n_strings
```

### String detuning (beating)

```
freq_string[s] = f_hz + beat_hz · beat_scale · STRING_SIGNS[n_strings][s]

STRING_SIGNS:
  n=1: { 0,     0,    0   }   no detuning
  n=2: {−0.5, +0.5,   0   }   symmetric ±beat/2
  n=3: {−0.5,  0,   +0.5  }   symmetric ±beat/2 + center
```

Note: C++ string ordering is reversed relative to Python (string0 gets −beat/2 vs Python +beat/2),
but the stereo sum is symmetric so the perceived image is equivalent.

### Post-oscillator stereo chain (per block)

```
[oscillator sum with per-string pan]
        │
        ▼
1. Schroeder all-pass decorrelation  (L and R independently, different g coefficients)
     g_L =  0.35 + ds·0.25
     g_R = −(0.35 + ds·0.20)
     y[n] = −g·x[n] + x[n−1] − g·y[n−1]
     out  = x·(1−ds) + y·ds
     ds   = clamp((midi − decorr_lo)/(decorr_hi − decorr_lo), 0, 1) · decorr_max · stereo_decorr
        │
        ▼
2. Spectral BiquadEQ  (per-note curve from params.json, eq_strength blend)
        │
        ▼
3. M/S stereo width
     eff  = width_factor · stereo_boost
     outL = L·(1+eff)/2 + R·(1−eff)/2
     outR = R·(1+eff)/2 + L·(1−eff)/2
        │
        ▼
4. Onset ramp (linear, onset_ms, applied LAST — matches Python)
        │
        ▼
5. Accumulate into output buffers
```

Noise channels (L and R) are independent (separate `std::rand()` draws per sample),
matching Python which generates L and R noise buffers separately.

---

## Calculation Methods

### Inharmonic partial frequencies
```
fk = k · f0 · sqrt(1 + B·k²)
```
`B` = inharmonicity coefficient (extracted per-note from recordings).
`f0_fitted_hz` = physically fitted fundamental (not equal-temperament).

### Bi-exponential amplitude envelope
```
A(t) = A0 · (a1·exp(-t/τ1) + (1-a1)·exp(-t/τ2))
```
Parameters `A0`, `a1`, `τ1`, `τ2` are per-partial, per-velocity-layer, per-MIDI-note.
Mono partials (single string) use single-exponential (no bi-exp beating contribution).

### Inter-string beating
```
f_string[i] = fk + STRING_SIGNS[n_strings][i] · beat_hz · beat_scale
```
`beat_hz` is per-partial, extracted from the recording.
`STRING_SIGNS`: n=2 → {−0.5, +0.5}; n=3 → {−0.5, 0, +0.5}

### Schroeder all-pass stereo decorrelation
```
g = 0.35 + ds·0.25         (left channel)
g = -(0.35 + ds·0.20)      (right channel)
y[n] = -g·x[n] + x[n−1] - g·y[n−1]
out[n] = x[n]·(1 - ds) + y[n]·ds
```
`ds` = `min(1, (midi - lo)/(hi - lo)) · decorr_max · stereo_decorr`

### Velocity mapping
```
vel_gain = ((vel_band + 1) / 8)^vel_gamma
```
`vel_band` ∈ {0..7}. Render server accepts vel_band directly.
MIDI velocity → band: `vel_band = midi_vel · 7 / 127`

### Equal-power string panning
```
center = π/4 + (midi - 64.5)/87 · pan_tilt
angle[i] = center ± pan_spread/2
pan_L = cos(angle),  pan_R = sin(angle)
```

### Post-hoc RMS normalization (offline renderer only)
```
target_rms = cfg.target_rms · vel_gain     (default: 0.06 · vel_gain)
scale = target_rms / actual_rms
out[*] *= scale
```
Applied after full buffer render. Corrects for inter-string phase cross-terms.
**Not used in real-time path** (causal — uses onset-based level calibration).

---

## Differences from Original Implementation

| Area | Original | dev-ICR |
|---|---|---|
| **Signal chain order** | EQ → onset → (decorr+M/S combined) | decorr → EQ → M/S → onset (matches Python) |
| **SIMD** | `/arch:AVX2` / `-mavx2 -mfma` | `/arch:AVX` / `-mavx` (AMD Steamroller compatible) |
| **MIDI last-note** | Only updated on keyboard press | Also updated from MIDI queue in audio thread |
| **pan_spread** | Hardcoded 0.55 | From `SynthConfig.pan_spread` |
| **beat_scale** | Always 1.0 | From `SynthConfig.beat_scale` |
| **Offline RMS** | Per-block level calibration | Post-hoc full-buffer normalization (Python parity) |

---

## Build Instructions

### Prerequisites
- CMake ≥ 3.16
- MSVC BuildTools 2022 (Windows) or GCC/Clang (Linux/macOS)
- Internet access for FetchContent (GLFW, Dear ImGui — downloaded once)

### Windows (MSVC)

```bat
:: Run from VS Developer PowerShell or after vcvars64.bat
cd C:\Users\Jindra\PycharmProjects\ICR

cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target IthacaCoreResonator
cmake --build build --config Release --target IthacaCoreResonatorGUI
cmake --build build --config Release --target IthacaRenderServer
```

Binaries land in `build\bin\Release\`.
Soundbanks are automatically copied to `build\bin\Release\soundbanks\` post-build.

### Linux / macOS

```bash
cd /path/to/ICR
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target IthacaCoreResonatorGUI
```

ALSA (`libasound-dev`) required on Linux; CoreAudio/CoreMIDI linked automatically on macOS.

---

## Running

### GUI (recommended for testing)

```bat
cd build\bin\Release
IthacaCoreResonatorGUI.exe --params soundbanks\params-ks-grand-ft.json
```

Options:
- `--params <path>` — physics parameter profile (required)
- `--config <path>` — optional SynthConfig JSON (`beat_scale`, `eq_strength`, etc.)
- `--port <N>` — MIDI input port index (default: 0; lists available ports on startup)

Keyboard fallback (when no MIDI hardware):
```
a s d f g h j k  →  C4 D4 E4 F4 G4 A4 B4 C5
z                →  sustain pedal (toggle)
q                →  quit
```

### Headless CLI

```bat
cd build\bin\Release
IthacaCoreResonator.exe --params soundbanks\params-ks-grand-ft.json
```

Same options and keyboard fallback as GUI.

### RenderServer (for Python training loop)

```bat
cd build\bin\Release
IthacaRenderServer.exe --params soundbanks\params-ks-grand-ft.json --port 9876
```

The server logs to `analysis/runtime-logs/render-server.log`.
On startup it sends `{"status":"ready"}` to each accepted TCP connection.

**TCP protocol** (one JSON per line, newline-terminated):
```json
{"cmd":"ping"}
{"cmd":"render","midi":60,"vel":3,"sr":44100,"duration":3.0,"output":"exports/note.wav"}
{"cmd":"set_config","params":{"beat_scale":1.5,"eq_strength":0.8}}
{"cmd":"get_config"}
{"cmd":"reload","params":"soundbanks/new-profile.json"}
{"cmd":"quit"}
```

### Numerical comparison (C++ vs Python)

```bash
# Quick test: MIDI 60, velocity band 3
python -m analysis.compare_cpp_python

# Full piano range test + save results
python -m analysis.compare_cpp_python --batch --save

# Single note with spectrogram plot
python -m analysis.compare_cpp_python --midi 60 --vel 3 --plot --save
```

Requires `IthacaRenderServer` running on port 9876 and `analysis/physics_synth.py` in path.

---

## Soundbank Profiles

| File | Description |
|---|---|
| `params-ks-grand-nn.json` | Neural-network trained profile (step 4 output) |
| `params-ks-grand-ft.json` | MRSTFT-finetuned profile (step 5 output, best quality) |
| `params-ks-grand-ft.synth_config.json` | Global SynthConfig for the finetuned profile |

Use `params-ks-grand-ft.json` + the matching `.synth_config.json` for highest fidelity.

---

## Key Constants

| Constant | Value | Location |
|---|---|---|
| `BLOCK_SIZE` | 256 frames | `offline_renderer.h` |
| `RESONATOR_DEFAULT_CHANNELS` | 2 (stereo) | `resonator_engine.h` |
| `MAX_PARTIALS` | 96 | `note_params.h` |
| `MAX_STRINGS` | 3 | `note_params.h` |
| `MIDI_COUNT` | 88 (A0–C8) | `note_params.h` |
| `VEL_LAYERS` | 8 | `note_params.h` |
| `MIDI_QUEUE_SIZE` | 256 | `resonator_engine.cpp` |
| Default sample rate | 48000 Hz | `resonator_engine.h` |
| Default TCP port | 9876 | `render_server.h` |
| Target RMS | 0.06 (−24.4 dBFS) | `synth_config.h` |
