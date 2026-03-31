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
#include <cstring>
#include <cstdio>

using json = nlohmann::json;

// Self-register
REGISTER_SYNTH_CORE("PianoCore", PianoCore)

static constexpr float PI  = 3.14159265358979f;
static constexpr float TAU = 2.f * PI;

// ── Constructor ───────────────────────────────────────────────────────────────

PianoCore::PianoCore() {
    std::memset(delayed_offs_, 0, sizeof(delayed_offs_));
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
    if (sustain_)
        delayed_offs_[midi] = true;
    else
        handleNoteOff(midi);
}

void PianoCore::sustainPedal(bool down) {
    sustain_ = down;
    if (!down) {
        for (int m = 0; m < PIANO_MAX_VOICES; m++) {
            if (delayed_offs_[m]) {
                handleNoteOff((uint8_t)m);
                delayed_offs_[m] = false;
            }
        }
    }
}

void PianoCore::allNotesOff() {
    for (int m = 0; m < PIANO_MAX_VOICES; m++) {
        if (voices_[m].active) handleNoteOff((uint8_t)m);
        delayed_offs_[m] = false;
    }
    sustain_ = false;
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
              rng_seed_.load(std::memory_order_relaxed));

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
                           int rng_seed) noexcept {
    const PianoNoteParam& np = note_params_[midi][vel_idx];

    v.active     = true;
    v.releasing  = false;
    v.in_onset   = true;
    v.midi       = midi;
    v.vel_idx    = vel_idx;
    v.t_samples  = 0;
    v.phi_diff   = np.phi_diff;

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

    // Initialise per-partial state
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
    }
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

            // ── Partials ────────────────────────────────────────────────────
            float voice_samp = 0.f;
            for (int ki = 0; ki < v.n_partials; ki++) {
                PianoPartialState& ps = v.partials[ki];

                float env = ps.a1 * ps.env_fast + (1.f - ps.a1) * ps.env_slow;
                ps.env_fast *= ps.decay_fast;
                ps.env_slow *= ps.decay_slow;

                if (ps.A0_scaled * env < PIANO_SKIP_THRESH) continue;

                float phase_c = tpi2 * ps.f_hz + ps.phi;
                float phase_b = tpi2 * ps.beat_hz_h;

                float s1 = std::cos(phase_c + phase_b);
                float s2 = std::cos(phase_c - phase_b + v.phi_diff);

                voice_samp += ps.A0_scaled * env * (s1 + s2) * 0.5f;
            }

            // ── Noise ───────────────────────────────────────────────────────
            float noise_samp = v.ndist(v.rng);
            voice_samp += v.A_noise_sc * noise_samp * v.noise_env;
            v.noise_env *= v.noise_decay;

            // ── Onset / release gates ────────────────────────────────────────
            voice_samp *= env_gate;
            if (v.releasing) {
                voice_samp *= v.rel_gain;
                v.rel_gain  += v.rel_step;
                if (v.rel_gain <= 0.f) {
                    v.active    = false;
                    v.rel_gain  = 0.f;
                }
            }

            out_l[i] += voice_samp;
            out_r[i] += voice_samp;   // mono → identical L/R

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
    return false;
}

bool PianoCore::getParam(const std::string& key, float& out) const {
    if (key == "beat_scale")  { out = beat_scale_ .load(std::memory_order_relaxed); return true; }
    if (key == "noise_level") { out = noise_level_.load(std::memory_order_relaxed); return true; }
    if (key == "rng_seed")    { out = (float)rng_seed_.load(std::memory_order_relaxed); return true; }
    return false;
}

std::vector<CoreParamDesc> PianoCore::describeParams() const {
    return {
        { "beat_scale",  "Beat Scale",  "Timbre", "×",  beat_scale_ .load(), 0.f, 4.f, false },
        { "noise_level", "Noise Level", "Timbre", "×",  noise_level_.load(), 0.f, 4.f, false },
        { "rng_seed",    "RNG Seed",    "Debug",  "",   (float)rng_seed_.load(), 0.f, 9999.f, true  },
    };
}

// ── Visualization (GUI thread) ────────────────────────────────────────────────

CoreVizState PianoCore::getVizState() const {
    CoreVizState vs;
    vs.sustain_active = sustain_;

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

        vs.last_note       = std::move(vv);
        vs.last_note_valid = true;
    }

    return vs;
}
