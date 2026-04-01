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
in noise-dominated notes — see Known Issues below.

---

## Known Issues (deferred)

### KI-1: Noise formula mismatch for high-noise / extreme notes
**Affected**: A0, C4 forte, C6, C8 (A_noise > 0.75 in params.json)

**Root cause**: C++ voice synthesizes noise at `floor_rms * target_rms * vel_gain`
(pre-scaled). Post-hoc normalization then scales everything by
`target_rms * vel_gain / actual_rms`. For fast-decaying notes (C8: tau=50ms,
render=3s), actual_rms << target_rms → large scale factor → noise/signal ratio
deviates from Python.

Python adds noise at absolute `A_noise` level before normalization; normalization
scales partials+noise together, preserving the ratio.

**Fix**: In offline path, use `noise_env = floor_rms * noise_level` (absolute,
without target_rms*vel_gain) so post-hoc normalization handles level parity.
Requires either offline-mode flag in SynthConfig or two-pass rendering.

**Impact on training**: Low — extreme notes (A0, C8) are rare in training data;
mid-range forte notes contribute to loss but the C++ rendering is still usable.

### KI-2: Stochastic phase variance
Random initial phase per partial/string means two renders of the same note
always differ. MRSTFT floor is ~1.0–1.5 even for Python vs Python comparisons.
This is expected and irreducible without fixing the random seed.

---

## Phase 2 — Modular Refactor (active)

### Goal
Restructure `synth/` into clean separation of concerns:
```
synth/core/     — shared: NoteParams, SynthConfig, BiquadEQ, NoteLUT, constants
synth/offline/  — OfflineRenderer, RenderServer (buffer-based, Python-aligned)
synth/realtime/ — ResonatorVoice, VoiceManager, ResonatorEngine (causal, miniaudio)
```

### Principles
- No code duplication between offline and realtime paths
- Logging: unified via existing core_logger.h
- GUI, MIDI, audio device wrappers: untouched
- CMakeLists.txt: targets updated to reflect new paths
- All existing functionality preserved

### Next 3 Priority Actions
1. Create `synth/core/` with NoteParams, SynthConfig, BiquadEQ, NoteLUT (headers + move .cpp)
2. Move offline_renderer + render_server → `synth/offline/`
3. Move resonator_voice + voice_manager + resonator_engine → `synth/realtime/`

---

## Technical Debt
- compare_cpp_python.py: `synthesize_python()` and batch loop duplicate vel_gain logic — consolidate
- `render_client.py` start() uses positional args for server launch (missing `--params` flag) — verify
- GUI does not expose SynthConfig sliders (beat_scale, noise_level, eq_strength) — post-Phase 2
