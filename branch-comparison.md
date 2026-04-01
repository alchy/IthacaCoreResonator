# Branch comparison: `dev-pluggable-spectralEQ` vs `dev-pluggable-cascade`

Both branches implement the same soundbank-based piano synthesis with spectral EQ.
`dev-pluggable-cascade` was built incrementally; `dev-pluggable-spectralEQ` is a clean
rebase onto `dev-pluggable` incorporating all lessons learned.

## Feature matrix

| Feature | cascade (final) | spectralEQ |
|---|:---:|:---:|
| Soundbank params (real extracted, not NN) | yes | yes |
| Bi-exponential envelope | yes | yes |
| Stereo constant-power panning | yes | yes |
| Schroeder all-pass decorrelation | yes | yes |
| Per-partial `phi_diff` (independent L/R phase per harmonic) | **no** — per-voice | **yes** |
| Keyboard spread in pan formula | yes | yes |
| `keyboard_spread` GUI param | yes | yes |
| `eq_strength` GUI param (EQ dry/wet blend) | yes | yes |
| Spectral EQ — biquad cascade (DF-II, 5 sections) | yes | yes |
| EQ: L/R independent filter state | yes | yes |
| Noise power included in `rms_gain` denominator | yes | yes |
| `attack_tau` capped at `tau1` of k=1 partial | yes | yes |
| EQ fitting: `_invfreqz` (custom lstsq, scipy 1.16 compat) | yes | yes |
| EQ fitting: `_mag_to_min_phase` (cepstral) | yes | yes |
| Clean single-commit implementation | no — 3 commits | **yes** |

## Key functional difference

| | cascade | spectralEQ |
|---|---|---|
| `phi_diff` scope | One value per voice — all partials share the same phase offset between string 1 and string 2. Mono partials (beat=0) therefore produce identical L/R signal. | One independent random value **per partial** — each harmonic gets its own L/R phase decorrelation. Mono partials have true stereo width. |

## Parameters exposed (both branches identical)

| Key | Group | Range | Default |
|---|---|---|---|
| `beat_scale` | Timbre | 0 – 4 × | 1.0 |
| `noise_level` | Timbre | 0 – 4 × | 1.0 |
| `eq_strength` | Timbre | 0 – 1 × | 1.0 |
| `pan_spread` | Stereo | 0 – π rad | 0.55 |
| `keyboard_spread` | Stereo | 0 – π rad | 0.60 |
| `stereo_decorr` | Stereo | 0 – 2 × | 1.0 |
| `rng_seed` | Debug | 0 – 9999 | 0 |

## Recommendation

Use `dev-pluggable-spectralEQ` — it supersedes `dev-pluggable-cascade` in every respect
and additionally provides true per-partial stereo width.
