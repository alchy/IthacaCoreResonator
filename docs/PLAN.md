# IthacaCoreResonator — Implementation Plan

Physics-based real-time piano synthesizer engine.
Replaces WAV sample playback (IthacaCore) with additive synthesis from
a pre-computed parameter table (`params-nn-profile-{bank}.json`).

---

## Architecture Overview

```
MIDI note-on (midi, vel)
        │
        ▼
   NoteLUT[midi-21][vel]          ← loaded from params.json at startup
        │  PartialParams[K]
        │  SpectralEQCurve
        ▼
  ResonatorVoice::noteOn()
        │
        ▼
  ResonatorVoice::processBlock()  ← called each audio buffer (256 samples)
   ├─ envelope: bi-exp decay coefficients (pre-computed at note-on)
   ├─ oscillators: K partials × N_strings (SIMD AVX2)
   │   inharmonic freq: f_k = k·f0·√(1 + B·k²)
   │   beating: f_ki = f_k + beat_hz[k] × string_detune[str]
   ├─ spectral EQ: biquad cascade (8 bands, designed from EQCurve at note-on)
   └─ stereo: Mid = mono sum, Side = mono × width_factor → L=M+S, R=M-S
        │
        ▼
  VoiceManager::processBlockUninterleaved(L, R, n)
        │  sum active voices, apply limiter + BBE
        ▼
  audio output (float32 stereo)
```

---

## Component Map

### Synthesis core (shared across all targets)

| File | Responsibility |
|------|----------------|
| `synth/note_params.h` | `PartialParams`, `NoiseParams`, `NoteParams`, `NoteLUT` structs |
| `synth/note_lut.h/.cpp` | Load `params.json` → `NoteLUT[88][8]`, `interpolateNoteLayers()` |
| `synth/resonator_voice.h/.cpp` | Single voice: oscillators + bi-exp envelope + EQ + stereo + pitch glide |
| `synth/voice_manager.h/.cpp` | Polyphony pool (88 voices), note-on/off, sustain pedal, SynthConfig |
| `synth/biquad_eq.h/.cpp` | 8-band peaking biquad cascade designed from 64-point EQ curve |
| `synth/synth_config.h` | `SynthConfig` struct — 20 R/W parameters (vel, EQ, stereo, timbre) |
| `synth/sysex.h/.cpp` | MIDI SysEx codec: SET_PARAM / SET_ALL / GET_PARAM / ALL_PARAMS_DUMP |

### Real-time playback (IthacaCoreResonator / GUI)

| File | Responsibility |
|------|----------------|
| `synth/resonator_engine.h/.cpp` | miniaudio real-time audio callback, audio device management |
| `synth/midi_input.h/.cpp` | RtMidi MIDI input callback, SysEx routing |
| `dsp/dsp_chain.h/.cpp` | Serial DSP chain (BBE + limiter on master bus) |
| `dsp/bbe/*` | BBE harmonic enhancer |
| `dsp/limiter/*` | Soft limiter |

### Offline render server (IthacaRenderServer)

| File | Responsibility |
|------|----------------|
| `synth/offline_renderer.h/.cpp` | Headless note renderer; post-render RMS normalization |
| `synth/render_server.h/.cpp` | TCP JSON dispatcher: render / set_config / get_config / sysex / reload |
| `server_main.cpp` | Entry point, arg parsing, Logger init |

### Shared infrastructure

| File | Responsibility |
|------|----------------|
| `sampler/core_logger.h/.cpp` | RT-safe ring-buffer logger |

### Removed vs. IthacaCore

- `InstrumentLoader` — no WAV files
- `SampleRateConverter` — synthesis at native sample rate
- `EnvelopeStaticData` / ADSR — replaced by bi-exponential decay
- `libsndfile` submodule — not needed (keep for WAV export tests only)
- `speexdsp` submodule — not needed

---

## Key Data Structures (`synth/note_params.h`)

```cpp
static constexpr int MAX_PARTIALS = 96;   // bass notes up to ~90 partials
static constexpr int MAX_STRINGS  = 3;
static constexpr int EQ_POINTS    = 64;   // log-spaced EQ curve points from params.json

struct PartialParams {
    int   k;              // partial number (1-based)
    float f_hz;           // inharmonic frequency (Hz)
    float A0;             // initial amplitude
    float tau1, tau2;     // bi-exp decay time constants (s)
    float a1;             // bi-exp mixing weight (fast component)
    float beat_hz;        // inter-string detuning (Hz)
    float beat_depth;     // beating depth (0..1)
    bool  mono;           // true → no inter-string beating
    bool  is_longitudinal;// true → phantom partial (f≈2·f_k, τ≈τ/2, mono)
};

struct NoiseParams {
    float attack_tau_s;         // noise envelope decay (s)
    float floor_rms;            // noise RMS level
    float centroid_hz;          // spectral centroid (FIR filter design)
    float spectral_slope_db_oct;// roll-off slope
};

struct NoteParams {
    int   midi, vel;
    float f0_hz;           // fundamental frequency
    float B;               // inharmonicity coefficient
    float width_factor;    // M/S stereo width (from spectral_eq.stereo_width_factor)
    float duration_s;      // full note duration from recording
    int   sr;              // source sample rate
    int   n_partials;      // valid entries in partials[]
    int   n_strings;       // 1, 2, or 3
    PartialParams partials[MAX_PARTIALS];
    NoiseParams   noise;
    // Raw spectral EQ curve from params.json (64 log-spaced points, 20–20000 Hz)
    float eq_freqs_hz[EQ_POINTS];
    float eq_gains_db[EQ_POINTS];
    bool  valid;
};

// Pre-loaded at startup — zero allocation in audio path
using NoteLUT = std::array<std::array<NoteParams, VEL_LAYERS>, MIDI_COUNT>;
```

---

## Voice Rendering

### Envelope (no `expf()` in audio loop)

```cpp
// At noteOn: pre-compute per-sample decay multipliers
for (int k = 0; k < n_partials; k++) {
    decay1[k] = expf(-1.f / (p.partials[k].tau1 * sample_rate));
    decay2[k] = expf(-1.f / (p.partials[k].tau2 * sample_rate));
    env1[k]   = a1[k]       * amplitude[k];
    env2[k]   = (1-a1[k])   * amplitude[k];
}

// processBlock inner loop (per sample):
//   env1[k] *= decay1[k];
//   env2[k] *= decay2[k];
//   float amp = env1[k] + env2[k];
```

### SIMD Strategy (AVX2)

Inner loop processes 8 partials simultaneously:
- Phase accumulation: 8× `_mm256_add_ps`
- `cosf` approximation: degree-5 polynomial (error < 1e-5, no libm)
- Envelope multiply: 8× `_mm256_mul_ps`

Estimated: ~1 µs for 64 partials × 3 strings per 256-sample buffer.

### A0 Normalization

At `noteOn`, amplitudes are normalized before rendering:
```cpp
float A0_ref = partials[0].A0;          // first partial as reference
float sum_sq  = Σ (A0_k / A0_ref)²;    // expected instantaneous power
float level_scale = target_rms * sqrt(2) / sqrt(sum_sq) * vel_gain;
```

This matches the Python `physics_synth.py` normalization (E[cos²(φ)] = 0.5 for random phases).

---

## Spectral EQ (`synth/biquad_eq.h`)

Variable-band biquad cascade designed at note-on from `eq_freqs_hz[64]` + `eq_gains_db[64]`.
The 64-point curve comes from `compute_spectral_eq.py` (LTASE method, 20–20 kHz log-spaced).
`BiquadEQ::design()` selects the active bands up to `cfg.eq_freq_min` and fits peaking biquads.
Velocity interpolation: `interpolateNoteLayers()` lerps `eq_gains_db` across the 8 velocity layers
so each note-on receives the correct velocity-dependent spectral shape.

---

## Build System

Three CMake targets — all in `CMakeLists.txt`:

| Target | Binary | Contents |
|--------|--------|----------|
| `IthacaCoreResonatorGUI` | `build/bin/Release/IthacaCoreResonatorGUI.exe` | Full synth: engine + MIDI + DSP chain + ImGui GUI |
| `IthacaCoreResonator` | `build/bin/Release/IthacaCoreResonator.exe` | Headless CLI (same engine, no GUI) |
| `IthacaRenderServer` | `build/bin/Release/IthacaRenderServer.exe` | Offline renderer: TCP JSON server, no audio device |

`IthacaRenderServer` shares `RENDER_SYNTH_SOURCES` (note_lut, resonator_voice, voice_manager, sysex,
offline_renderer) with the other targets. It does **not** include dsp_chain (BBE+limiter),
resonator_engine, or MIDI input.

```bash
cmake --build build --config Release                           # all targets
cmake --build build --config Release --target IthacaRenderServer  # server only
```

Dependencies (all vendored or FetchContent — no package manager):
- `nlohmann/json` — `third_party/json.hpp` (MIT, vendored)
- `miniaudio` — `third_party/miniaudio.h` (MIT, vendored)
- `RtMidi` — `third_party/RtMidi.h/.cpp` (MIT, vendored)
- `GLFW 3.4` — FetchContent from GitHub
- `Dear ImGui v1.91.9` — FetchContent from GitHub
- `ws2_32` — Windows Sockets (IthacaRenderServer only)

---

## Implementation Phases

### Phase 0 — Parameter extraction (complete)
- [x] `analysis/extract_params.py` — extract physics params from WAV bank
- [x] `analysis/params.json` — 88 × 8 velocity layers, per-partial PartialParams
- [x] LTASE spectral EQ method, window resolution analysis

### Phase 1 — Core synthesis (complete)
- [x] `synth/note_params.h` — structs
- [x] `synth/note_lut.cpp` — parse params.json → NoteLUT, interpolate missing notes
- [x] `synth/resonator_voice.cpp` — scalar oscillator + bi-exp envelope + EQ + stereo
- [x] `synth/biquad_eq.cpp` — 8-band peaking EQ designer
- [x] A0 normalization + target_rms level calibration
- [x] Onset ramp (post-EQ), noise model

### Phase 2 — Polyphony + MIDI (complete)
- [x] `synth/voice_manager.cpp` — pool, note-on/off, sustain pedal
- [x] `synth/resonator_engine.cpp` — audio callback, miniaudio integration
- [x] DSP chain: BBE + limiter on master bus

### Phase 3 — GUI (complete)
- [x] `gui/resonator_gui.cpp` — ImGui frontend
- [x] Piano keyboard display, voice matrix
- [x] Per-note parameter display (NoteParams from LUT)
- [x] Synthesis config panel with live edit
- [x] Peak metering, seed display

### Phase 4 — Physics refinement (in progress)
- [x] Velocity-dependent spectral EQ interpolation — `interpolateNoteLayers()` lerps `eq_gains_db[64]` across 8 velocity layers
- [x] Pitch glide at forte — `pitch_glide` / `pitch_glide_tau_ms` / `pitch_glide_vel_thresh` in `SynthConfig`; fractional f₀ offset in `resonator_voice.cpp::processBlock()`
- [x] Longitudinal precursor (bass, MIDI < 50) — short noise burst at noteOn, `longitudinal_precursor` SynthConfig param
- [x] Phantom partials C++ render path — `PartialParams::is_longitudinal` flag; rendered at f≈2·f_k, τ≈τ/2, mono; **pending**: `extract_params.py` extension to detect and emit longitudinal peaks

### Phase 5 — Differentiable training (in progress)
- [x] `IthacaRenderServer` — headless TCP JSON server; used by training pipeline for fast note rendering
- [x] `analysis/torch_synth.py` — differentiable 2-string additive synth proxy; `render_note_differentiable(model, midi, vel, ...)`
- [x] `analysis/closed_loop_finetune.py` — MRSTFT fine-tuning (`--mode finetune`) and global SynthConfig optimization (`--mode global`)
- [x] `analysis/train_instrument_profile.py` — surrogate NN with `phi_net`, `df_net`, `eq_net` (99,214 params)
- [ ] Reparametrize τ as physical b1/b3 coefficients (Simionato 2024)
- [ ] Multi-instrument latent space (Steinway ↔ Bösendorfer)

### Phase 6 — Plugin wrapper (future)
- [ ] JUCE AudioProcessor wrapping ResonatorVoiceManager
- [ ] VST3 / CLAP export

---

## Comparison: IthacaCore vs. IthacaCoreResonator

| Aspect | IthacaCore | IthacaCoreResonator |
|--------|-----------|---------------------|
| Sound source | WAV samples (PCM playback) | Additive synthesis from params |
| Per-note data | `float* pcm[8]` (MB of audio) | `NoteParams[8]` (~4 KB total) |
| Memory footprint | Hundreds of MB (WAV bank) | < 1 MB (param table) |
| Envelope | ADSR | Bi-exponential decay (physics) |
| Tuning | Fixed (recorded pitch) | Exact physics (inharmonicity) |
| Velocity layers | 8 discrete samples | 8 discrete param sets + interp |
| Beating / chorus | None | Explicit per-partial beat_hz |
| Stereo | Fixed (recorded) | Parametric M/S + per-string pan |
