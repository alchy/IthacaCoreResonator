#pragma once
/*
 * synth_config.h
 * ────────────────
 * Per-session synthesis rendering parameters.
 * Mirrors physics_synth.py  synthesize_note()  keyword arguments exactly.
 *
 * Stored in ResonatorVoiceManager and forwarded to each ResonatorVoice at
 * noteOn time.  Changing a field affects notes started after the change.
 */

struct SynthConfig {
    // ── Stereo geometry ───────────────────────────────────────────────────────
    float pan_spread          = 0.55f;   // string spread in rad  (half = pan_spread/2)
    float pan_tilt            = 0.20f;   // keyboard tilt: center += (midi-64.5)/87 * pan_tilt
    float stereo_decorr       = 1.0f;    // Schroeder all-pass blend multiplier (0=off, 1=full)
    float stereo_decorr_midi_lo = 40.f;  // MIDI note where decorrelation starts
    float stereo_decorr_midi_hi = 100.f; // MIDI note where decorrelation reaches maximum
    float stereo_decorr_max   = 0.45f;  // max decorrelation depth before stereo_decorr scale
    float stereo_boost        = 1.0f;    // M/S side-channel boost on top of width_factor

    // ── Timbre ────────────────────────────────────────────────────────────────
    float beat_scale          = 1.0f;    // beat_hz multiplier (1.0=extracted, 1.5-2.5=vivid)
    float harmonic_brightness = 0.0f;    // upper-partial boost: gain = 1 + hb*log2(k)

    // ── Spectral EQ ───────────────────────────────────────────────────────────
    float eq_strength         = 1.0f;    // EQ blend (0=bypass, 1=full)
    float eq_freq_min         = 400.0f;  // EQ flat below this Hz (room-acoustics guard)

    // ── Noise ─────────────────────────────────────────────────────────────────
    float noise_level         = 1.0f;    // noise amplitude multiplier

    // ── Pitch glide (geometric nonlinearity at forte) ─────────────────────────
    // At large amplitudes f₀ starts slightly high and falls over ~100 ms.
    // Papers: RR-9516, RR-8181, Simionato 2024.
    float pitch_glide           = 0.0f;   // initial fractional frequency offset (e.g. 0.003)
    float pitch_glide_tau_ms    = 80.0f;  // glide decay time constant (ms)
    int   pitch_glide_vel_thresh= 100;    // apply only when MIDI vel >= this (forte gate)

    // ── Longitudinal precursor (bass string wave, MIDI < 50) ─────────────────
    // Short high-frequency noise burst before the transverse wave arrives.
    // Paper: RR_9530 (Chabassier 2023). Duration auto-scaled to ~2 string cycles.
    float longitudinal_precursor= 0.0f;   // burst amplitude relative to note level

    // ── Attack ────────────────────────────────────────────────────────────────
    float onset_ms            = 3.0f;    // linear ramp length to prevent click

    // ── Level ─────────────────────────────────────────────────────────────────
    // Target RMS for each synthesized note (matches Python target_rms=0.06).
    // Implemented via A0_ref normalization + exact tau-integral energy formula.
    float target_rms             = 0.06f;   // −24.4 dBFS RMS; Python default
    // Reference render duration used in the level calibration formula.
    // Must match the duration used when generating training data (Python default=3.0s).
    // p.duration_s (recording length) must NOT be used here — it varies per sample.
    float render_ref_duration_s  = 3.0f;    // seconds; matches Python render(duration=3.0)

    // ── Velocity ──────────────────────────────────────────────────────────────
    // vel_gain = ((midi_vel + 1) / 8)^vel_gamma  — matches Python synthesize_note().
    // Range: vel=1 → ~0.16 (ppp), vel=64 → ~2.1 (mf), vel=127 → ~7.0 (fff).
    // Applied as multiplier on target_rms for both partials and noise.
    float vel_gamma           = 0.7f;
};
