/*
 * synth-core/sine/sine_core.cpp
 * ──────────────────────────────
 */
#include "sine_core.h"
#include "synth/synth_core_registry.h"

// Self-register into the global SynthCoreRegistry.
REGISTER_SYNTH_CORE("SineCore", SineCore)

static constexpr float PI  = 3.14159265358979f;
static constexpr float TAU = 2.f * PI;

// ── Constructor ───────────────────────────────────────────────────────────────

SineCore::SineCore() {
    voices_.fill(SineVoice{});
    for (auto& d : delayed_offs_) d.store(false, std::memory_order_relaxed);
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

bool SineCore::load(const std::string& /*params_path*/, float sr, Logger& logger) {
    sample_rate_ = sr;
    loaded_      = true;
    // Recompute omega for voices already active (e.g. after setSampleRate call)
    for (int m = 0; m < N_VOICES; m++) {
        if (voices_[m].active) {
            float f = midiToHz(m, detune_cents_.load());
            voices_[m].omega = TAU * f / sample_rate_;
        }
    }
    logger.log("SineCore", LogSeverity::Info,
               "Ready. SR=" + std::to_string((int)sr));
    return true;
}

void SineCore::setSampleRate(float sr) {
    sample_rate_ = sr;
    for (int m = 0; m < N_VOICES; m++) {
        if (voices_[m].active) {
            float f = midiToHz(m, detune_cents_.load());
            voices_[m].omega = TAU * f / sample_rate_;
        }
    }
}

// ── MIDI (RT thread) ──────────────────────────────────────────────────────────

void SineCore::noteOn(uint8_t midi, uint8_t vel) {
    if (midi >= N_VOICES) return;
    handleNoteOn(midi, vel);
}

void SineCore::noteOff(uint8_t midi) {
    if (midi >= N_VOICES) return;
    if (sustain_.load(std::memory_order_relaxed))
        delayed_offs_[midi].store(true, std::memory_order_relaxed);
    else
        handleNoteOff(midi);
}

void SineCore::sustainPedal(bool down) {
    sustain_.store(down, std::memory_order_relaxed);
    if (!down) {
        for (int m = 0; m < N_VOICES; m++) {
            if (delayed_offs_[m].load(std::memory_order_relaxed)) {
                handleNoteOff((uint8_t)m);
                delayed_offs_[m].store(false, std::memory_order_relaxed);
            }
        }
    }
}

void SineCore::allNotesOff() {
    for (int m = 0; m < N_VOICES; m++) {
        if (voices_[m].active)
            handleNoteOff((uint8_t)m);
        delayed_offs_[m].store(false, std::memory_order_relaxed);
    }
    sustain_.store(false, std::memory_order_relaxed);
}

void SineCore::handleNoteOn(uint8_t midi, uint8_t vel) noexcept {
    SineVoice& v = voices_[midi];
    float f      = midiToHz(midi, detune_cents_.load(std::memory_order_relaxed));
    v.omega      = TAU * f / sample_rate_;
    v.amp        = (vel / 127.f) * gain_.load(std::memory_order_relaxed);
    v.phase      = 0.f;
    v.in_onset   = true;
    v.onset_gain = 0.f;
    v.onset_step = 1.f / (ONSET_MS * 0.001f * sample_rate_);
    v.releasing  = false;
    v.rel_gain   = 1.f;
    v.active     = true;

    last_midi_.store(midi, std::memory_order_relaxed);
    last_vel_ .store(vel,  std::memory_order_relaxed);
}

void SineCore::handleNoteOff(uint8_t midi) noexcept {
    SineVoice& v = voices_[midi];
    if (!v.active) return;
    v.releasing = true;
    v.rel_step  = -1.f / (RELEASE_MS * 0.001f * sample_rate_);
    v.rel_gain  = v.in_onset ? v.onset_gain : 1.f;
}

// ── Audio rendering (RT thread, additive) ────────────────────────────────────

bool SineCore::processBlock(float* out_l, float* out_r, int n_samples) noexcept {
    bool any = false;
    for (int m = 0; m < N_VOICES; m++) {
        SineVoice& v = voices_[m];
        if (!v.active) continue;
        any = true;
        for (int i = 0; i < n_samples; i++) {
            // Onset ramp
            float env = 1.f;
            if (v.in_onset) {
                v.onset_gain += v.onset_step;
                if (v.onset_gain >= 1.f) { v.onset_gain = 1.f; v.in_onset = false; }
                env = v.onset_gain;
            }
            // Release ramp
            if (v.releasing) {
                v.rel_gain += v.rel_step;
                if (v.rel_gain <= 0.f) {
                    v.active = v.releasing = false;
                    v.rel_gain = 0.f;
                }
                env *= v.rel_gain;
            }

            float s = v.amp * env * std::sin(v.phase);
            out_l[i] += s;
            out_r[i] += s;

            v.phase += v.omega;
            if (v.phase >= TAU) v.phase -= TAU;
            if (!v.active) break;
        }
    }
    return any;
}

// ── Parameters (GUI thread) ───────────────────────────────────────────────────

bool SineCore::setParam(const std::string& key, float value) {
    if (key == "gain") {
        gain_.store(std::max(0.f, std::min(2.f, value)), std::memory_order_relaxed);
        return true;
    }
    if (key == "detune_cents") {
        detune_cents_.store(std::max(-100.f, std::min(100.f, value)),
                            std::memory_order_relaxed);
        return true;
    }
    return false;
}

bool SineCore::getParam(const std::string& key, float& out) const {
    if (key == "gain")         { out = gain_.load(std::memory_order_relaxed);         return true; }
    if (key == "detune_cents") { out = detune_cents_.load(std::memory_order_relaxed); return true; }
    return false;
}

std::vector<CoreParamDesc> SineCore::describeParams() const {
    return {
        { "gain",         "Gain",   "Output", "",   gain_.load(),         0.f,   2.f,   false },
        { "detune_cents", "Detune", "Tuning", "ct", detune_cents_.load(), -100.f, 100.f, false },
    };
}

// ── Visualization (GUI thread) ────────────────────────────────────────────────

CoreVizState SineCore::getVizState() const {
    CoreVizState vs;
    vs.sustain_active = sustain_.load(std::memory_order_relaxed);

    for (int m = 0; m < N_VOICES; m++) {
        if (voices_[m].active) {
            vs.active_midi_notes.push_back(m);
            vs.active_voice_count++;
        }
    }

    int last_midi = last_midi_.load(std::memory_order_relaxed);
    int last_vel  = last_vel_ .load(std::memory_order_relaxed);
    if (last_midi >= 0) {
        CoreVoiceViz vv;
        vv.midi  = last_midi;
        vv.vel   = last_vel;
        vv.f0_hz = midiToHz(last_midi, detune_cents_.load(std::memory_order_relaxed));
        vs.last_note       = std::move(vv);
        vs.last_note_valid = true;
    }

    return vs;
}
