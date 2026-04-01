# dev-pluggable-cascade — Quick Start

## Export params from soundbank

```bash
python analysis/export_soundbank_params.py \
    --soundbank soundbanks/params-ks-grand-ft.json \
    --out       analysis/params-piano-soundbank.json
```

Optional flags:
```
--sr 44100          sample rate for RMS normalization (default: 44100)
--duration 3.0      render duration used for RMS computation (default: 3.0)
--target-rms 0.06   target output RMS (default: 0.06)
--rng-seed 0        base RNG seed for phi generation (default: 0)
```

Output: `analysis/params-piano-soundbank.json` (~5 MB, 704 notes, includes biquad EQ cascade per note)

## Build

```bash
cd build
cmake --build . --config Release --target IthacaCoreResonatorGUI
```

## Run synth with GUI

```bash
build/bin/Release/IthacaCoreResonatorGUI.exe \
    --core   PianoCore \
    --params analysis/params-piano-soundbank.json
```

With optional SynthConfig overrides:
```bash
build/bin/Release/IthacaCoreResonatorGUI.exe \
    --core        PianoCore \
    --params      analysis/params-piano-soundbank.json \
    --config      analysis/profile-finetuned.synth_config.json \
    --core-param  pan_spread=0.55 \
    --core-param  stereo_decorr=1.0 \
    --core-param  noise_level=1.0
```

## PianoCore parameters (adjustable in GUI)

| Key | Default | Range | Effect |
|-----|---------|-------|--------|
| `beat_scale` | 1.0 | 0–4 | Scales beating frequency of all partials |
| `noise_level` | 1.0 | 0–4 | Scales hammer noise amplitude |
| `pan_spread` | 0.55 rad | 0–π | Stereo width (0 = mono) |
| `stereo_decorr` | 1.0 | 0–2 | Schroeder all-pass decorrelation strength |

## Signal chain (per voice)

```
noteOn
  → initVoice: load params, compute pan gains, biquad EQ coeffs,
                per-partial random phi_diff, all-pass coefficients

processBlock (per sample):
  partials:  s1=cos(carrier+beat+phi), s2=cos(carrier−beat+phi+phi_diff_k)
             → panned to L/R via constant-power angles (MIDI-dependent)
  noise:     independent L/R Gaussian × exp(−t/attack_tau)
  decorr:    Schroeder first-order all-pass (different g for L and R)
  EQ:        5-section biquad cascade (Direct Form II, fitted from soundbank)
  gate:      onset ramp (3 ms) + release fade (100 ms)
```

## Files

| File | Description |
|------|-------------|
| `soundbanks/params-ks-grand-ft.json` | Source: extracted params from real KS Grand recordings |
| `analysis/params-piano-soundbank.json` | Exported PianoCore params (generated, do not hand-edit) |
| `analysis/export_soundbank_params.py` | Export script (soundbank → PianoCore JSON) |
| `synth-core/piano/piano_core.h/.cpp` | C++ PianoCore implementation |
