# IthacaCoreResonator — Project Roadmap
_Last updated: 2026-04-01_

## Phase 1 — Signal Chain Parity ✅ COMPLETE

### Completed
- [x] Signal chain order aligned with Python: decorr → EQ → M/S → onset_ramp
- [x] SIMD downgraded AVX2 → AVX (AMD Steamroller / Athlon X4 870K compatibility)
- [x] Post-hoc RMS normalization in offline_renderer (matches Python)
- [x] vel_band round-trip fix in offline_renderer: `(vel*127+6)/7` instead of `vel*127/7`
      — old formula mapped band=3 → midi_vel=54 → band_back=2 (wrong LUT layer)
- [x] compare_cpp_python.py: vel_gain applied to target_rms for Python synthesis
      — matches training loop convention (physics_synth.py normalize_dataset)
- [x] Numerical verification via compare_cpp_python --batch

### Verification Results (2026-04-01, params-ks-grand-ft.json)

| Note | MRSTFT (C++) | Stoch. floor | Status |
|---|---|---|---|
| A0  MIDI 21 vel 3 | 2.36 | 0.83 | ⚠️ noise-dominated |
| C2  MIDI 36 vel 3 | 1.09 | 1.42 | ✅ below floor |
| C3  MIDI 48 vel 3 | 1.30 | ~1.3 | ✅ at floor |
| C4  MIDI 60 vel 3 | 1.27 | 1.10 | ✅ at floor |
| C4  MIDI 60 vel 6 | 2.35 | 1.28 | ⚠️ noise-dominated |
| C5  MIDI 72 vel 3 | 1.21 | ~1.2 | ✅ at floor |
| C6  MIDI 84 vel 3 | 1.80 | 1.29 | ⚠️ noise-dominated |
| C7  MIDI 96 vel 3 | 1.41 | ~1.4 | ✅ at floor |
| C8  MIDI108 vel 3 | 4.32 | 1.52 | ⚠️ noise-dominated |

Mid-range piano (C2–C7 mezzo) converged to stochastic floor. Divergence only
in noise-dominated notes — fixed in Phase 3 (KI-1).

---

## Phase 2 — Modular Refactor ✅ COMPLETE

### Completed
- [x] `synth/core/` — NoteParams, SynthConfig, SynthConfigIO, BiquadEQ, NoteLUT
- [x] `synth/realtime/` — ResonatorVoice, VoiceManager, ResonatorEngine, MidiInput, Sysex
- [x] `synth/offline/` — OfflineRenderer, RenderServer
- [x] `third_party/miniaudio.h` — duplicate in `synth/` removed; canonical in `third_party/`
- [x] CMakeLists.txt — source paths and include_directories updated for all 3 targets
- [x] External includes updated — main.cpp, gui_main.cpp, server_main.cpp, gui/
- [x] Relative `../third_party/` includes fixed → flat names via include path
- [x] Build verified — all 3 targets clean (IthacaCoreResonator, GUI, RenderServer)

---

## Phase 3 — Quality & Tooling ✅ COMPLETE

### Completed
- [x] KI-1 noise formula fix — `SynthConfig.offline_mode` flag; offline path uses
      absolute `floor_rms * noise_level`, post-hoc normalization preserves noise/signal ratio
- [x] GUI SynthConfig live sliders — `beat_scale` (0–3), `eq_strength` (0–1), `noise_level` (0–2)
      via `ResonatorEngine::setSynthBeatScale/EqStrength/NoiseLevel()`
- [x] compare_cpp_python.py — `run_batch()` vel_gain logic centralized in `synthesize_python()`
- [x] render_client.py — explicit `--params` flag on server subprocess launch

---

## Known Issues

### KI-2: Stochastic phase variance
Random initial phase per partial/string means two renders of the same note
always differ. MRSTFT floor is ~1.0–1.5 even for Python vs Python comparisons.
This is expected and irreducible without fixing the random seed.

**Root cause**: each string gets a random initial phase at noteOn via `std::rand()`:
```cpp
phase_[s][k] = (float)std::rand() / RAND_MAX * TAU;
```
With multiple strings per note (2–3), the inter-string phase difference changes the
amplitude of the sum: constructive at φ=0, destructive at φ=π. Two renders of the
same note are physically different signals even if all parameters are identical.

**Why MRSTFT is affected**: MRSTFT measures spectral distance. Phase-shifted sinusoids
at the same frequency produce different spectra → non-zero loss even for correct code.
Python vs Python comparison of the same note gives MRSTFT ~1.0–1.5 — this is the
irreducible stochastic floor. Anything at or below this level is considered converged.

**When it would be worth fixing**: only if the training loop needs deterministic
gradients (e.g. same note rendered twice in one batch). Currently not a problem —
each note is synthesized once per batch and phase variability is part of the training
distribution. Fix would require a shared PRNG with a fixed seed passed to both C++
and Python, or eliminating random phases entirely (loses natural tone variability).

---

## Technical Debt

- GUI does not expose remaining SynthConfig params (pan_spread, stereo_decorr, onset_ms, vel_gamma)
- compare_cpp_python.py: no automated regression baseline — MRSTFT targets are manual
- closed_loop_finetune.py: not verified against new offline_mode noise fix (re-run recommended)
