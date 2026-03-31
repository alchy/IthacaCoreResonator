#pragma once
/*
 * synth-core/piano/piano_core.h
 * ──────────────────────────────
 * Faithful C++ port of analysis/torch_synth.py.
 *
 * Synthesis algorithm (matches Python proxy):
 *  - 2-string model per partial:
 *      s1 = cos(2π*f*t + 2π*(beat/2)*t + phi)
 *      s2 = cos(2π*f*t - 2π*(beat/2)*t + phi + phi_diff)
 *      partial = A0 * env * (s1 + s2) / 2
 *  - Bi-exponential envelope: a1*exp(-t/tau1) + (1-a1)*exp(-t/tau2)
 *  - Gaussian noise: A_noise * randn() * exp(-t/attack_tau)
 *  - RMS normalisation: per-note rms_gain pre-computed at export time
 *
 * Parameters are loaded from a JSON file exported by
 *   analysis/export_piano_params.py
 * which runs the trained InstrumentProfile NN model for all 88×8 notes
 * and bakes the results into a format suitable for real-time playback.
 *
 * Phases (phi) are pre-computed in the export script using the same
 * torch.Generator seed as torch_synth.py, enabling byte-comparable
 * partial synthesis.  Noise uses an independent C++ mt19937 PRNG.
 *
 * Threading: same as ISynthCore — see i_synth_core.h.
 */

#include "synth/i_synth_core.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <unordered_map>
#include <string>

// ── Internal constants ────────────────────────────────────────────────────────

static constexpr int   PIANO_MAX_PARTIALS = 60;
static constexpr int   PIANO_MAX_VOICES   = 128;   // one slot per MIDI note
static constexpr float PIANO_RELEASE_MS   = 100.f; // key-release fade-out
static constexpr float PIANO_ONSET_MS     = 3.f;   // click-prevention onset
static constexpr float PIANO_SKIP_THRESH  = 2e-7f; // skip silent partials

// ── Loaded-param structs (constant per note) ─────────────────────────────────

struct PianoPartialParam {
    float f_hz     = 0.f;
    float A0       = 0.f;
    float tau1     = 0.f;
    float tau2     = 0.f;
    float a1       = 1.f;
    float beat_hz  = 0.f;
    float phi      = 0.f;   // initial phase (precomputed, matching Python RNG)
};

struct PianoNoteParam {
    bool  valid      = false;
    int   K          = 0;
    float phi_diff   = 0.f;
    float attack_tau = 0.05f;
    float A_noise    = 0.04f;
    float rms_gain   = 1.f;
    float f0_hz      = 440.f;
    PianoPartialParam partials[PIANO_MAX_PARTIALS];
};

// ── Voice runtime state ───────────────────────────────────────────────────────

struct PianoPartialState {
    // Exponential decay state
    float env_fast   = 1.f;
    float env_slow   = 1.f;
    float decay_fast = 0.f;   // exp(-1/(tau1*sr))
    float decay_slow = 0.f;   // exp(-1/(tau2*sr))
    // Precomputed at noteOn (const during voice lifetime)
    float A0_scaled   = 0.f;  // A0 * rms_gain * (current beat_scale)
    float a1          = 1.f;
    float f_hz        = 0.f;
    float beat_hz_h   = 0.f;  // beat_hz * beat_scale * 0.5
    float phi         = 0.f;
};

struct PianoVoice {
    bool     active      = false;
    bool     releasing   = false;
    bool     in_onset    = false;
    int      midi        = -1;
    int      vel_idx     = -1;
    uint32_t t_samples   = 0;
    uint64_t max_t_samp  = 0;  // auto-stop (silence) threshold

    // phi_diff (constant per note, loaded from params)
    float phi_diff       = 0.f;

    // Noise state
    float A_noise_sc     = 0.f;  // A_noise * rms_gain * noise_level
    float noise_env      = 1.f;
    float noise_decay    = 0.f;

    // Release / onset ramps
    float rel_gain       = 1.f;
    float rel_step       = 0.f;
    float onset_gain     = 0.f;
    float onset_step     = 0.f;

    // Noise PRNG (independent of Python RNG — noise not required to match exactly)
    std::mt19937 rng;
    std::normal_distribution<float> ndist{0.f, 1.f};

    // Active partial state
    int n_partials = 0;
    PianoPartialState partials[PIANO_MAX_PARTIALS];
};

// ── PianoCore ─────────────────────────────────────────────────────────────────

class PianoCore final : public ISynthCore {
public:
    PianoCore();

    bool load(const std::string& params_path, float sr, Logger& logger) override;
    void setSampleRate(float sr) override;

    void noteOn      (uint8_t midi, uint8_t velocity) override;
    void noteOff     (uint8_t midi)                   override;
    void sustainPedal(bool down)                      override;
    void allNotesOff ()                               override;

    bool processBlock(float* out_l, float* out_r, int n_samples) noexcept override;

    bool setParam(const std::string& key, float value)      override;
    bool getParam(const std::string& key, float& out) const override;
    std::vector<CoreParamDesc> describeParams()        const override;

    CoreVizState getVizState() const override;

    std::string coreName()    const override { return "PianoCore"; }
    std::string coreVersion() const override { return "1.0"; }
    bool        isLoaded()    const override { return loaded_; }

private:
    // Loaded note params [midi 0..127][vel_idx 0..7]
    PianoNoteParam note_params_[128][8];

    // Active voices (one slot per MIDI note)
    PianoVoice voices_[PIANO_MAX_VOICES];

    // Sustain pedal state
    bool sustain_                   = false;
    bool delayed_offs_[PIANO_MAX_VOICES] = {};

    float sample_rate_ = 44100.f;
    float inv_sr_      = 1.f / 44100.f;
    bool  loaded_      = false;

    // GUI-settable parameters (read from RT thread via atomic)
    std::atomic<float> beat_scale_  {1.0f};   // scales beat_hz for all notes
    std::atomic<float> noise_level_ {1.0f};   // scales noise amplitude
    std::atomic<int>   rng_seed_    {0};       // base seed (applied at noteOn)

    // Last note info for GUI viz
    std::atomic<int>   last_midi_   {-1};
    std::atomic<int>   last_vel_    {0};

    // Helpers
    void handleNoteOn (uint8_t midi, uint8_t vel) noexcept;
    void handleNoteOff(uint8_t midi)              noexcept;
    void initVoice    (PianoVoice& v, int midi, int vel_idx,
                       float beat_scale, float noise_level, int rng_seed) noexcept;

    // Map MIDI velocity 1-127 to vel index 0-7
    static int midiVelToIdx(uint8_t velocity) {
        return std::min(7, (int)(velocity - 1) / 16);
    }
};
