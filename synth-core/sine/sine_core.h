#pragma once
/*
 * synth-core/sine/sine_core.h
 * ────────────────────────────
 * Minimal sine-wave synthesizer — first ISynthCore implementation.
 *
 * Purpose: validate the CoreEngine + ISynthCore API end-to-end before
 * implementing synth-core/piano (the faithful Python proxy C++ port).
 *
 * Design:
 *  - 128 polyphonic voices (one per MIDI note)
 *  - No ADSR; amplitude = (velocity/127) × gain, constant while held
 *  - 3 ms linear onset ramp (click prevention)
 *  - 10 ms linear release ramp on noteOff (click prevention)
 *  - Sustain pedal: delays note-off until pedal released
 *  - Stereo output: identical L and R (mono core)
 *
 * Parameters:
 *  - "gain"         (Output):  overall amplitude scale, 0..2, default 1.0
 *  - "detune_cents" (Tuning):  global pitch shift in cents, -100..100, default 0
 */

#include "synth/i_synth_core.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cmath>

class SineCore final : public ISynthCore {
public:
    SineCore();

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

    std::string coreName()    const override { return "SineCore"; }
    std::string coreVersion() const override { return "1.0"; }
    bool        isLoaded()    const override { return loaded_; }

private:
    static constexpr int   N_VOICES   = 128;
    static constexpr float ONSET_MS   = 3.f;
    static constexpr float RELEASE_MS = 10.f;

    struct SineVoice {
        bool  active     = false;
        bool  releasing  = false;
        float phase      = 0.f;   // current phase (radians)
        float omega      = 0.f;   // angular frequency per sample
        float amp        = 0.f;   // target amplitude (vel-scaled)
        float onset_gain = 0.f;   // onset ramp state 0→1
        float onset_step = 0.f;   // per-sample onset increment
        bool  in_onset   = false;
        float rel_gain   = 1.f;   // release ramp state 1→0
        float rel_step   = 0.f;   // per-sample release decrement (negative)
    };

    std::array<SineVoice, N_VOICES> voices_{};
    std::atomic<bool> sustain_      {false};
    std::atomic<bool> delayed_offs_[N_VOICES];

    float sample_rate_   = 44100.f;
    bool  loaded_        = false;

    // Parameters: written by GUI thread, read by RT thread (atomic)
    std::atomic<float> gain_        {1.0f};
    std::atomic<float> detune_cents_{0.0f};

    // Last note info (written on noteOn, read in getVizState)
    std::atomic<int> last_midi_{-1};
    std::atomic<int> last_vel_ { 0};

    void handleNoteOn (uint8_t midi, uint8_t vel) noexcept;
    void handleNoteOff(uint8_t midi)              noexcept;

    static float midiToHz(int midi, float detune_cents) {
        return 440.f * std::pow(2.f, (midi - 69 + detune_cents / 100.f) / 12.f);
    }
};
