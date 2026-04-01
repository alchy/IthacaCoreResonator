#pragma once
/*
 * i_synth_core.h
 * ───────────────
 * Pluggable synthesis core interface.
 *
 * An ISynthCore handles a complete synthesis engine internally:
 *  - polyphony (voice pool), envelopes, oscillators, panning, EQ, noise
 *  - accepts MIDI events (noteOn, noteOff, sustainPedal, allNotesOff)
 *  - outputs stereo float32 PCM, additive (caller zeroes buffers)
 *  - exposes parameters via generic key/value API (GUI-friendly)
 *  - provides visualization snapshot (active voices, last note detail)
 *
 * What is NOT part of the core (lives in CoreEngine):
 *  - master gain / master pan
 *  - LFO panning
 *  - DspChain (limiter, BBE)
 *  - audio device (miniaudio)
 *  - MIDI port management / SysEx
 *
 * Threading model:
 *  - noteOn/noteOff/sustainPedal/allNotesOff are called ONLY from the RT
 *    thread (CoreEngine drains the MIDI queue before calling processBlock).
 *  - setParam/getParam are called from GUI thread; implementations MUST be
 *    safe with respect to concurrent processBlock (use atomics or a param
 *    queue; direct float write is acceptable on x86 as a pragmatic choice).
 *  - getVizState/describeParams may allocate and are called from GUI thread.
 *  - processBlock is called ONLY from the RT thread; MUST NOT allocate memory,
 *    acquire locks, or perform IO.
 */

#include "../sampler/core_logger.h"
#include <string>
#include <vector>
#include <cstdint>

// ── Param descriptor — returned by describeParams() ──────────────────────────

struct CoreParamDesc {
    std::string key;      // identifier used in setParam/getParam ("beat_scale")
    std::string label;    // human-readable label ("Beat Scale")
    std::string group;    // panel group / tab ("Timbre", "Stereo", ...)
    std::string unit;     // display unit ("Hz", "ms", "") — appended after value
    float       value;    // current value (snapshot at call time)
    float       min;      // slider minimum
    float       max;      // slider maximum
    bool        is_int;   // true → display as integer, step = 1
};

// ── Visualization structs — returned by getVizState() ────────────────────────

struct CorePartialViz {
    int   k        = 0;
    float f_hz     = 0.f;
    float A0       = 0.f;
    float tau1     = 0.f;
    float tau2     = 0.f;
    float a1       = 1.f;
    float beat_hz  = 0.f;
    bool  mono     = false;
};

struct CoreVoiceViz {
    int   midi              = -1;
    int   vel               = 0;
    float f0_hz             = 0.f;
    float B                 = 0.f;      // inharmonicity
    int   n_strings         = 0;
    int   n_partials        = 0;
    float width_factor      = 0.f;
    float noise_centroid_hz = 0.f;
    float noise_floor_rms   = 0.f;
    float noise_tau_s       = 0.f;
    // EQ curve (optional — only filled by cores that use it)
    std::vector<float> eq_freqs_hz;     // EQ_POINTS values
    std::vector<float> eq_gains_db;
    // Partials table (optional — only filled if n_partials > 0)
    std::vector<CorePartialViz> partials;
};

struct CoreVizState {
    int  active_voice_count = 0;
    bool sustain_active     = false;
    // Brief list of currently active voices (midi note only, no heavy data)
    std::vector<int> active_midi_notes;
    // Detailed snapshot of the last triggered note (for right panel)
    CoreVoiceViz last_note;
    bool         last_note_valid = false;
};

// ── Main interface ────────────────────────────────────────────────────────────

class ISynthCore {
public:
    virtual ~ISynthCore() = default;

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    // Load parameters from file and prepare synthesis at the given sample rate.
    // params_path: core-specific JSON (or "" for cores that need no file).
    // Returns false on error; error description written to logger.
    virtual bool load(const std::string& params_path,
                      float              sr,
                      Logger&            logger) = 0;

    // Change sample rate after load (recomputes coefficients).
    // THREADING: must only be called before CoreEngine::start() or after
    // CoreEngine::stop(). Implementations write non-atomic fields (e.g.
    // inv_sr_); calling this while the RT thread runs processBlock is a data race.
    virtual void setSampleRate(float sr) = 0;

    // ── MIDI — called from RT thread only ─────────────────────────────────────
    virtual void noteOn      (uint8_t midi, uint8_t velocity) = 0;
    virtual void noteOff     (uint8_t midi)                   = 0;
    virtual void sustainPedal(bool down)                      = 0;
    virtual void allNotesOff ()                               = 0;

    // ── Audio rendering — RT thread, NO alloc, NO lock, NO IO ────────────────
    // Adds core output into out_l / out_r (ADDITIVE — caller zeroes buffers).
    // Returns true if any voice is active.
    virtual bool processBlock(float* out_l, float* out_r,
                              int n_samples) noexcept = 0;

    // ── Parameters — called from GUI thread ──────────────────────────────────
    // setParam: returns false if key unknown.
    // getParam: returns false if key unknown.
    virtual bool setParam(const std::string& key, float value)       = 0;
    virtual bool getParam(const std::string& key, float& out)  const = 0;
    // Full param list with metadata; snapshot of current values.
    virtual std::vector<CoreParamDesc> describeParams() const         = 0;

    // ── Visualization — called from GUI thread, may allocate ─────────────────
    virtual CoreVizState getVizState() const = 0;

    // ── Meta ──────────────────────────────────────────────────────────────────
    virtual std::string coreName()    const = 0;
    virtual std::string coreVersion() const = 0;
    virtual bool        isLoaded()    const = 0;
};
