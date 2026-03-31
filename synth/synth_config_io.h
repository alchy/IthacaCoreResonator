#pragma once
/*
 * synth_config_io.h
 * ──────────────────
 * Load SynthConfig fields from a JSON file.
 *
 * Only keys present in the JSON are applied; missing keys keep their current
 * values (defaults or previously set).  Unknown keys are silently ignored.
 *
 * JSON key names match SynthConfig field names exactly (snake_case).
 * Typical use — loading profile.synth_config.json produced by global opt:
 *
 *   SynthConfig cfg;
 *   std::string err;
 *   if (!loadSynthConfig("soundbanks/params-ks-grand-ft.synth_config.json", cfg, &err))
 *       fprintf(stderr, "Config load failed: %s\n", err.c_str());
 *
 * Requires: third_party/json.hpp (nlohmann/json single-header)
 */

#include "synth_config.h"
#include "../third_party/json.hpp"
#include <fstream>
#include <string>

// Load SynthConfig fields from a JSON file.
// Returns true on success, false if the file cannot be opened or parsed.
// On failure, *error_out (if non-null) receives a human-readable description.
inline bool loadSynthConfig(const std::string& path, SynthConfig& cfg,
                             std::string* error_out = nullptr)
{
    std::ifstream f(path);
    if (!f) {
        if (error_out) *error_out = "Cannot open: " + path;
        return false;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        if (error_out) *error_out = std::string("JSON parse error: ") + e.what();
        return false;
    }

    auto get_f = [&](const char* k, float& v) {
        if (j.contains(k) && j[k].is_number()) v = j[k].get<float>();
    };
    auto get_i = [&](const char* k, int& v) {
        if (j.contains(k) && j[k].is_number()) v = j[k].get<int>();
    };

    // ── Timbre ────────────────────────────────────────────────────────────────
    get_f("beat_scale",             cfg.beat_scale);
    get_f("noise_level",            cfg.noise_level);
    get_f("harmonic_brightness",    cfg.harmonic_brightness);

    // ── Stereo geometry ───────────────────────────────────────────────────────
    get_f("pan_spread",             cfg.pan_spread);
    get_f("pan_tilt",               cfg.pan_tilt);
    get_f("stereo_decorr",          cfg.stereo_decorr);
    get_f("stereo_decorr_midi_lo",  cfg.stereo_decorr_midi_lo);
    get_f("stereo_decorr_midi_hi",  cfg.stereo_decorr_midi_hi);
    get_f("stereo_decorr_max",      cfg.stereo_decorr_max);
    get_f("stereo_boost",           cfg.stereo_boost);

    // ── Spectral EQ ───────────────────────────────────────────────────────────
    get_f("eq_strength",            cfg.eq_strength);
    get_f("eq_freq_min",            cfg.eq_freq_min);

    // ── Pitch glide ───────────────────────────────────────────────────────────
    get_f("pitch_glide",            cfg.pitch_glide);
    get_f("pitch_glide_tau_ms",     cfg.pitch_glide_tau_ms);
    get_i("pitch_glide_vel_thresh", cfg.pitch_glide_vel_thresh);

    // ── Longitudinal precursor ────────────────────────────────────────────────
    get_f("longitudinal_precursor", cfg.longitudinal_precursor);

    // ── Attack + level ────────────────────────────────────────────────────────
    get_f("onset_ms",               cfg.onset_ms);
    get_f("target_rms",             cfg.target_rms);
    get_f("render_ref_duration_s",  cfg.render_ref_duration_s);

    // ── Velocity ──────────────────────────────────────────────────────────────
    get_f("vel_gamma",              cfg.vel_gamma);

    return true;
}
