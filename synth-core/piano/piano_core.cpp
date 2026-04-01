/*
 * synth-core/piano/piano_core.cpp
 * ─────────────────────────────────
 * C++ port of analysis/torch_synth.py (2-string bi-exponential piano synth).
 *
 * Phase computation matches Python's:
 *   t[i] = float32(i) / sr
 *   phase_carrier = 2π * f_hz * t[i]
 *   phase_beat    = 2π * beat_hz * 0.5 * t[i]
 *   s1 = cos(phase_carrier + phase_beat + phi)
 *   s2 = cos(phase_carrier - phase_beat + phi + phi_diff)
 *
 * Envelope uses a multiplicative decay factor (avoids per-sample exp()):
 *   decay = exp(-1 / (tau * sr))  →  env[n] = env[n-1] * decay
 */
#include "piano_core.h"
#include "synth/synth_core_registry.h"
#include "third_party/json.hpp"

#include <fstream>
#include <algorithm>
#include <cstdio>

using json = nlohmann::json;

// Self-register
REGISTER_SYNTH_CORE("PianoCore", PianoCore)

static constexpr float PI  = 3.14159265358979f;
static constexpr float TAU = 2.f * PI;

// ── Constructor ───────────────────────────────────────────────────────────────

PianoCore::PianoCore() {
    for (auto& d : delayed_offs_) d.store(false, std::memory_order_relaxed);
}

// ── JSON loading ──────────────────────────────────────────────────────────────

bool PianoCore::load(const std::string& params_path, float sr, Logger& logger) {
    sample_rate_ = sr;
    inv_sr_      = 1.f / sr;

    if (params_path.empty()) {
        logger.log("PianoCore", LogSeverity::Error,
                   "params_path is required (export with analysis/export_piano_params.py)");
        return false;
    }

    std::ifstream f(params_path);
    if (!f.is_open()) {
        logger.log("PianoCore", LogSeverity::Error,
                   "Cannot open params: " + params_path);
        return false;
    }

    json root;
    try {
        f >> root;
    } catch (const std::exception& e) {
        logger.log("PianoCore", LogSeverity::Error,
                   std::string("JSON parse error: ") + e.what());
        return false;
    }

    if (!root.contains("notes")) {
        logger.log("PianoCore", LogSeverity::Error, "JSON missing 'notes' key");
        return false;
    }

    // Clear existing params
    for (int m = 0; m < 128; m++)
        for (int v = 0; v < 8; v++)
            note_params_[m][v] = PianoNoteParam{};

    int loaded_count = 0;
    const auto& notes = root["notes"];
    for (auto it = notes.begin(); it != notes.end(); ++it) {
        const auto& s = it.value();
        int midi    = s["midi"].get<int>();
        int vel_idx = s["vel"].get<int>();

        if (midi < 0 || midi > 127 || vel_idx < 0 || vel_idx > 7) continue;

        PianoNoteParam& np = note_params_[midi][vel_idx];
        np.valid     = true;
        np.phi_diff  = s["phi_diff"].get<float>();
        np.attack_tau= s["attack_tau"].get<float>();
        np.A_noise   = s["A_noise"].get<float>();
        np.rms_gain  = s["rms_gain"].get<float>();
        np.f0_hz     = s["f0_hz"].get<float>();

        const auto& partials = s["partials"];
        int K = std::min((int)partials.size(), PIANO_MAX_PARTIALS);
        np.K  = K;

        for (int ki = 0; ki < K; ki++) {
            const auto& p = partials[ki];
            PianoPartialParam& pp = np.partials[ki];
            pp.f_hz    = p["f_hz"].get<float>();
            pp.A0      = p["A0"].get<float>();
            pp.tau1    = p["tau1"].get<float>();
            pp.tau2    = p["tau2"].get<float>();
            pp.a1      = p["a1"].get<float>();
            pp.beat_hz = p["beat_hz"].get<float>();
            pp.phi     = p["phi"].get<float>();
        }

        // Spectral EQ biquad cascade (optional — absent in NN-exported params)
        np.n_biquad = 0;
        if (s.contains("eq_biquads")) {
            const auto& bqs = s["eq_biquads"];
            int nB = std::min((int)bqs.size(), PIANO_N_BIQUAD);
            for (int bi = 0; bi < nB; bi++) {
                const auto& bq = bqs[bi];
                PianoBiquadCoeffs& c = np.eq[bi];
                c.b0 = bq["b"][0].get<float>();
                c.b1 = bq["b"][1].get<float>();
                c.b2 = bq["b"][2].get<float>();
                c.a1 = bq["a"][0].get<float>();
                c.a2 = bq["a"][1].get<float>();
            }
            np.n_biquad = nB;
        }

        ++loaded_count;
    }

    loaded_ = (loaded_count > 0);
    if (!loaded_) {
        logger.log("PianoCore", LogSeverity::Error, "No valid notes in params file");
        return false;
    }

    logger.log("PianoCore", LogSeverity::Info,
               "Loaded " + std::to_string(loaded_count) + " notes from " + params_path
               + "  SR=" + std::to_string((int)sr));
    return true;
}

void PianoCore::setSampleRate(float sr) {
    sample_rate_ = sr;
    inv_sr_      = 1.f / sr;
    // Active voices will drift after SR change; not worth re-computing mid-note.
}

// ── MIDI (RT thread) ──────────────────────────────────────────────────────────

void PianoCore::noteOn(uint8_t midi, uint8_t velocity) {
    if (midi >= PIANO_MAX_VOICES) return;
    if (velocity == 0) { noteOff(midi); return; }
    handleNoteOn(midi, velocity);
}

void PianoCore::noteOff(uint8_t midi) {
    if (midi >= PIANO_MAX_VOICES) return;
    if (sustain_.load(std::memory_order_relaxed))
        delayed_offs_[midi].store(true, std::memory_order_relaxed);
    else
        handleNoteOff(midi);
}

void PianoCore::sustainPedal(bool down) {
    sustain_.store(down, std::memory_order_relaxed);
    if (!down) {
        for (int m = 0; m < PIANO_MAX_VOICES; m++) {
            if (delayed_offs_[m].load(std::memory_order_relaxed)) {
                handleNoteOff((uint8_t)m);
                delayed_offs_[m].store(false, std::memory_order_relaxed);
            }
        }
    }
}

void PianoCore::allNotesOff() {
    for (int m = 0; m < PIANO_MAX_VOICES; m++) {
        if (voices_[m].active) handleNoteOff((uint8_t)m);
        delayed_offs_[m].store(false, std::memory_order_relaxed);
    }
    sustain_.store(false, std::memory_order_relaxed);
}

void PianoCore::handleNoteOn(uint8_t midi, uint8_t vel) noexcept {
    int vel_idx = midiVelToIdx(vel);
    // Fall back to nearest available vel if exact not present
    if (!note_params_[midi][vel_idx].valid) {
        for (int dv = 1; dv < 8; dv++) {
            int lo = vel_idx - dv, hi = vel_idx + dv;
            if (lo >= 0 && note_params_[midi][lo].valid) { vel_idx = lo; break; }
            if (hi <= 7 && note_params_[midi][hi].valid) { vel_idx = hi; break; }
        }
    }
    if (!note_params_[midi][vel_idx].valid) return;  // no params for this note

    initVoice(voices_[midi], midi, vel_idx,
              beat_scale_.load(std::memory_order_relaxed),
              noise_level_.load(std::memory_order_relaxed),
              rng_seed_.load(std::memory_order_relaxed),
              pan_spread_.load(std::memory_order_relaxed),
              stereo_decorr_.load(std::memory_order_relaxed),
              keyboard_spread_.load(std::memory_order_relaxed));

    last_midi_.store(midi, std::memory_order_relaxed);
    last_vel_ .store(vel,  std::memory_order_relaxed);
}

void PianoCore::handleNoteOff(uint8_t midi) noexcept {
    PianoVoice& v = voices_[midi];
    if (!v.active) return;
    v.releasing = true;
    v.rel_gain  = v.in_onset ? v.onset_gain : 1.f;
    v.rel_step  = -v.rel_gain / (PIANO_RELEASE_MS * 0.001f * sample_rate_);
}

void PianoCore::initVoice(PianoVoice& v, int midi, int vel_idx,
                           float beat_scale, float noise_level,
                           int rng_seed, float pan_spread,
                           float stereo_decorr,
                           float keyboard_spread) noexcept {
    const PianoNoteParam& np = note_params_[midi][vel_idx];

    v.active     = true;
    v.releasing  = false;
    v.in_onset   = true;
    v.midi       = midi;
    v.vel_idx    = vel_idx;
    v.t_samples  = 0;

    // Noise
    v.A_noise_sc  = np.A_noise * np.rms_gain * noise_level;
    v.noise_env   = 1.f;
    v.noise_decay = std::exp(-1.f / std::max(np.attack_tau * sample_rate_, 1.f));
    v.rng.seed((uint32_t)(rng_seed + midi * 256 + vel_idx));
    v.ndist = std::normal_distribution<float>(0.f, 1.f);

    // Onset ramp
    v.onset_gain = 0.f;
    v.onset_step = 1.f / (PIANO_ONSET_MS * 0.001f * sample_rate_);
    v.rel_gain   = 1.f;
    v.rel_step   = 0.f;

    // Compute max voice duration: 10× longest tau2 or 60 s, whichever is less
    float max_tau = 0.f;
    for (int ki = 0; ki < np.K; ki++)
        if (np.partials[ki].tau2 > max_tau) max_tau = np.partials[ki].tau2;
    float dur_s = std::min(10.f * max_tau, 60.f);
    if (dur_s < 3.f) dur_s = 3.f;
    v.max_t_samp = (uint64_t)(dur_s * sample_rate_);

    // Initialise per-partial state (phi_diff: independent random per-partial)
    std::uniform_real_distribution<float> udist(0.f, TAU);
    v.n_partials = np.K;
    for (int ki = 0; ki < np.K; ki++) {
        const PianoPartialParam& pp = np.partials[ki];
        PianoPartialState& ps       = v.partials[ki];

        ps.env_fast   = 1.f;
        ps.env_slow   = 1.f;
        ps.decay_fast = std::exp(-1.f / std::max(pp.tau1 * sample_rate_, 1.f));
        ps.decay_slow = std::exp(-1.f / std::max(pp.tau2 * sample_rate_, 1.f));
        ps.A0_scaled  = pp.A0 * np.rms_gain;
        ps.a1         = pp.a1;
        ps.f_hz       = pp.f_hz;
        ps.beat_hz_h  = pp.beat_hz * beat_scale * 0.5f;
        ps.phi        = pp.phi;
        ps.phi_diff   = udist(v.rng);
    }

    // Stereo panning: constant-power pan per string, MIDI-dependent center
    // center = pi/4 + (midi-64.5)/87 * keyboard_spread/2  (±spread/2 across keyboard)
    // String 1 (s1 = carrier+beat/2) → angle1 = center - half
    // String 2 (s2 = carrier-beat/2) → angle2 = center + half
    {
        const float center = (PI / 4.f) + ((float)midi - 64.5f) / 87.0f * keyboard_spread * 0.5f;
        const float half   = pan_spread * 0.5f;
        const float a1     = center - half;
        const float a2     = center + half;
        v.gl1 = std::cos(a1);  v.gr1 = std::sin(a1);
        v.gl2 = std::cos(a2);  v.gr2 = std::sin(a2);
    }

    // Schroeder first-order all-pass decorrelation (matches physics_synth.py)
    // decor_strength = clamp((midi-40)/60, 0,1) * 0.45 * stereo_decorr
    // L all-pass: g_L = 0.35 + ds*0.25   (positive)
    // R all-pass: g_R = -(0.35 + ds*0.20) (negative → phase flip at Nyquist)
    // Difference equation: y[n] = -g*x[n] + x[n-1] - g*y[n-1]
    {
        float ds = std::min(1.0f, std::max(0.0f, ((float)midi - 40.0f) / 60.0f))
                   * 0.45f * stereo_decorr;
        v.decor_str = ds;
        v.ap_g_L    =   0.35f + ds * 0.25f;
        v.ap_g_R    = -(0.35f + ds * 0.20f);
        v.ap_x_L = v.ap_y_L = v.ap_x_R = v.ap_y_R = 0.f;
    }

    // Spectral EQ biquad cascade: copy coeffs, zero filter state
    v.n_biquad    = np.n_biquad;
    v.eq_strength = eq_strength_.load(std::memory_order_relaxed);
    for (int bi = 0; bi < np.n_biquad; bi++)
        v.eq_coeffs[bi] = np.eq[bi];
    std::memset(v.eq_wL, 0, sizeof(v.eq_wL));
    std::memset(v.eq_wR, 0, sizeof(v.eq_wR));
}

// ── Audio (RT thread, additive) ──────────────────────────────────────────────

bool PianoCore::processBlock(float* out_l, float* out_r, int n_samples) noexcept {
    bool any = false;

    for (int m = 0; m < PIANO_MAX_VOICES; m++) {
        PianoVoice& v = voices_[m];
        if (!v.active) continue;
        any = true;

        for (int i = 0; i < n_samples; i++) {
            // ── Onset ramp ──────────────────────────────────────────────────
            float env_gate = 1.f;
            if (v.in_onset) {
                v.onset_gain += v.onset_step;
                if (v.onset_gain >= 1.f) { v.onset_gain = 1.f; v.in_onset = false; }
                env_gate = v.onset_gain;
            }

            // ── Phase base ──────────────────────────────────────────────────
            // t = t_samples / sr  (matching Python's float32 t[i] = i/sr)
            const float t_f  = (float)v.t_samples * inv_sr_;
            const float tpi2 = TAU * t_f;

            // ── Partials (stereo: s1 → pan angle1, s2 → pan angle2) ────────
            float samp_L = 0.f, samp_R = 0.f;
            for (int ki = 0; ki < v.n_partials; ki++) {
                PianoPartialState& ps = v.partials[ki];

                float env = ps.a1 * ps.env_fast + (1.f - ps.a1) * ps.env_slow;
                ps.env_fast *= ps.decay_fast;
                ps.env_slow *= ps.decay_slow;

                if (ps.A0_scaled * env < PIANO_SKIP_THRESH) continue;

                float phase_c = tpi2 * ps.f_hz + ps.phi;
                float phase_b = tpi2 * ps.beat_hz_h;

                float s1 = std::cos(phase_c + phase_b);
                float s2 = std::cos(phase_c - phase_b + ps.phi_diff);

                float base = ps.A0_scaled * env * 0.5f;
                samp_L += base * (s1 * v.gl1 + s2 * v.gl2);
                samp_R += base * (s1 * v.gr1 + s2 * v.gr2);
            }

            // ── Noise (independent L/R) ──────────────────────────────────────
            float noise_sc = v.A_noise_sc * v.noise_env;
            samp_L += noise_sc * v.ndist(v.rng);
            samp_R += noise_sc * v.ndist(v.rng);
            v.noise_env *= v.noise_decay;

            // ── Schroeder all-pass decorrelation ─────────────────────────────
            // y[n] = -g*x[n] + x[n-1] - g*y[n-1]
            if (v.decor_str > 0.01f) {
                float Lap = -v.ap_g_L * samp_L + v.ap_x_L - v.ap_g_L * v.ap_y_L;
                float Rap = -v.ap_g_R * samp_R + v.ap_x_R - v.ap_g_R * v.ap_y_R;
                v.ap_x_L = samp_L;  v.ap_y_L = Lap;
                v.ap_x_R = samp_R;  v.ap_y_R = Rap;
                float inv = 1.f - v.decor_str;
                samp_L = samp_L * inv + Lap * v.decor_str;
                samp_R = samp_R * inv + Rap * v.decor_str;
            }

            // ── Spectral EQ biquad cascade (Direct Form II, L/R independent) ──
            if (v.n_biquad > 0 && v.eq_strength > 0.001f) {
                float wetL = samp_L, wetR = samp_R;
                for (int bi = 0; bi < v.n_biquad; bi++) {
                    const PianoBiquadCoeffs& c = v.eq_coeffs[bi];
                    float w0L = wetL - c.a1*v.eq_wL[bi][0] - c.a2*v.eq_wL[bi][1];
                    wetL = c.b0*w0L + c.b1*v.eq_wL[bi][0] + c.b2*v.eq_wL[bi][1];
                    v.eq_wL[bi][1] = v.eq_wL[bi][0];  v.eq_wL[bi][0] = w0L;
                    float w0R = wetR - c.a1*v.eq_wR[bi][0] - c.a2*v.eq_wR[bi][1];
                    wetR = c.b0*w0R + c.b1*v.eq_wR[bi][0] + c.b2*v.eq_wR[bi][1];
                    v.eq_wR[bi][1] = v.eq_wR[bi][0];  v.eq_wR[bi][0] = w0R;
                }
                float dry = 1.f - v.eq_strength;
                samp_L = samp_L * dry + wetL * v.eq_strength;
                samp_R = samp_R * dry + wetR * v.eq_strength;
            }

            // ── Onset / release gates ────────────────────────────────────────
            samp_L *= env_gate;
            samp_R *= env_gate;
            if (v.releasing) {
                samp_L *= v.rel_gain;
                samp_R *= v.rel_gain;
                v.rel_gain += v.rel_step;
                if (v.rel_gain <= 0.f) {
                    v.active   = false;
                    v.rel_gain = 0.f;
                }
            }

            out_l[i] += samp_L;
            out_r[i] += samp_R;

            v.t_samples++;

            if (!v.active) break;

            // Auto-stop after natural decay
            if ((uint64_t)v.t_samples >= v.max_t_samp) {
                v.active = false;
                break;
            }
        }
    }

    return any;
}

// ── Parameters (GUI thread) ───────────────────────────────────────────────────

bool PianoCore::setParam(const std::string& key, float value) {
    if (key == "beat_scale") {
        beat_scale_.store(std::max(0.f, std::min(4.f, value)),
                          std::memory_order_relaxed);
        return true;
    }
    if (key == "noise_level") {
        noise_level_.store(std::max(0.f, std::min(4.f, value)),
                           std::memory_order_relaxed);
        return true;
    }
    if (key == "rng_seed") {
        rng_seed_.store((int)value, std::memory_order_relaxed);
        return true;
    }
    if (key == "pan_spread") {
        pan_spread_.store(std::max(0.f, std::min(3.14159f, value)),
                          std::memory_order_relaxed);
        return true;
    }
    if (key == "stereo_decorr") {
        stereo_decorr_.store(std::max(0.f, std::min(2.f, value)),
                             std::memory_order_relaxed);
        return true;
    }
    if (key == "keyboard_spread") {
        keyboard_spread_.store(std::max(0.f, std::min(3.14159f, value)),
                               std::memory_order_relaxed);
        return true;
    }
    if (key == "eq_strength") {
        eq_strength_.store(std::max(0.f, std::min(1.f, value)),
                           std::memory_order_relaxed);
        return true;
    }
    return false;
}

bool PianoCore::getParam(const std::string& key, float& out) const {
    if (key == "beat_scale")   { out = beat_scale_   .load(std::memory_order_relaxed); return true; }
    if (key == "noise_level")  { out = noise_level_  .load(std::memory_order_relaxed); return true; }
    if (key == "rng_seed")     { out = (float)rng_seed_.load(std::memory_order_relaxed); return true; }
    if (key == "pan_spread")   { out = pan_spread_   .load(std::memory_order_relaxed); return true; }
    if (key == "stereo_decorr")    { out = stereo_decorr_    .load(std::memory_order_relaxed); return true; }
    if (key == "keyboard_spread")  { out = keyboard_spread_  .load(std::memory_order_relaxed); return true; }
    if (key == "eq_strength")      { out = eq_strength_      .load(std::memory_order_relaxed); return true; }
    return false;
}

std::vector<CoreParamDesc> PianoCore::describeParams() const {
    return {
        { "beat_scale",   "Beat Scale",    "Timbre",  "×",   beat_scale_   .load(), 0.f,    4.f,     false },
        { "noise_level",  "Noise Level",   "Timbre",  "×",   noise_level_  .load(), 0.f,    4.f,     false },
        { "pan_spread",   "Pan Spread",    "Stereo",  "rad", pan_spread_   .load(), 0.f,    3.14159f,false },
        { "stereo_decorr",   "Stereo Decorr",    "Stereo",  "×",   stereo_decorr_  .load(), 0.f,    2.f,      false },
        { "keyboard_spread", "Keyboard Spread",  "Stereo",  "rad", keyboard_spread_.load(), 0.f,    3.14159f, false },
        { "eq_strength",     "EQ Strength",      "Timbre",  "×",   eq_strength_    .load(), 0.f,    1.f,      false },
        { "rng_seed",        "RNG Seed",         "Debug",   "",    (float)rng_seed_.load(), 0.f,    9999.f,   true  },
    };
}

// ── Visualization (GUI thread) ────────────────────────────────────────────────

CoreVizState PianoCore::getVizState() const {
    CoreVizState vs;
    vs.sustain_active = sustain_.load(std::memory_order_relaxed);

    for (int m = 0; m < PIANO_MAX_VOICES; m++) {
        if (voices_[m].active) {
            vs.active_midi_notes.push_back(m);
            vs.active_voice_count++;
        }
    }

    int last_midi = last_midi_.load(std::memory_order_relaxed);
    int last_vel  = last_vel_ .load(std::memory_order_relaxed);
    if (last_midi >= 0 && last_midi < 128) {
        int vi = midiVelToIdx((uint8_t)std::max(1, std::min(127, last_vel)));
        const PianoNoteParam& np = note_params_[last_midi][vi];

        CoreVoiceViz vv;
        vv.midi       = last_midi;
        vv.vel        = last_vel;
        vv.f0_hz      = np.f0_hz;
        vv.n_partials = np.K;

        for (int ki = 0; ki < np.K && ki < 16; ki++) {   // cap at 16 for GUI
            const PianoPartialParam& pp = np.partials[ki];
            CorePartialViz cpv;
            cpv.k       = ki + 1;
            cpv.f_hz    = pp.f_hz;
            cpv.A0      = pp.A0;
            cpv.tau1    = pp.tau1;
            cpv.tau2    = pp.tau2;
            cpv.a1      = pp.a1;
            cpv.beat_hz = pp.beat_hz;
            cpv.mono    = (pp.a1 >= 0.99f);
            vv.partials.push_back(cpv);
        }

        // Spectral EQ frequency response (evaluated from biquad coefficients)
        // 32 log-spaced frequencies 30 Hz – 18 kHz, cascade magnitude in dB
        if (np.n_biquad > 0) {
            constexpr int N_EQ = 32;
            const float f_lo = 30.f, f_hi = 18000.f;
            const float log_lo = std::log(f_lo), log_hi = std::log(f_hi);
            vv.eq_freqs_hz.resize(N_EQ);
            vv.eq_gains_db.resize(N_EQ);
            for (int fi = 0; fi < N_EQ; fi++) {
                float f   = std::exp(log_lo + (log_hi - log_lo) * fi / (N_EQ - 1));
                float w   = TAU * f * inv_sr_;
                float cw  = std::cos(w), sw = std::sin(w);
                float c2w = std::cos(2.f * w), s2w = std::sin(2.f * w);
                // Product of biquad section magnitudes²
                float mag2 = 1.f;
                for (int bi = 0; bi < np.n_biquad; bi++) {
                    const PianoBiquadCoeffs& c = np.eq[bi];
                    float nr = c.b0 + c.b1 * cw  + c.b2 * c2w;
                    float ni = -(c.b1 * sw + c.b2 * s2w);
                    float dr = 1.f  + c.a1 * cw  + c.a2 * c2w;
                    float di = -(c.a1 * sw + c.a2 * s2w);
                    mag2 *= (nr*nr + ni*ni) / std::max(dr*dr + di*di, 1e-30f);
                }
                vv.eq_freqs_hz[fi] = f;
                vv.eq_gains_db[fi] = 10.f * std::log10(std::max(mag2, 1e-12f));
            }
        }

        vs.last_note       = std::move(vv);
        vs.last_note_valid = true;
    }

    return vs;
}
