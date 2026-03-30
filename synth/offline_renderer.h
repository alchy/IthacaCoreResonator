#pragma once
/*
 * offline_renderer.h
 * ────────────────────
 * Headless note renderer — synthesizes a single note to an in-memory PCM
 * buffer and optionally writes it as a 32-bit float WAV file.
 *
 * No audio device required; pure software rendering through ResonatorVoiceManager.
 *
 * Usage:
 *   OfflineRenderer r;
 *   r.initialize("analysis/params-ks-grand.json", logger);
 *   int n = r.renderNoteToFile(60, 80, 3.0f, 44100, "exports/m060_vel80.wav");
 */

#include "voice_manager.h"
#include "../sampler/core_logger.h"
#include <string>
#include <vector>

class OfflineRenderer {
public:
    // Load params JSON and prepare voice pool at the given sample rate.
    // Calling initialize() again reloads params (e.g. after params.json update).
    bool initialize(const std::string& params_json_path, Logger& logger,
                    float sr = 44100.f);

    // Render one note to an interleaved stereo float32 PCM buffer.
    //   midi       — MIDI note number (21–108)
    //   vel        — MIDI velocity (1–127)
    //   duration_s — target duration in seconds; <=0 uses auto-detect
    //   sr         — sample rate (re-initializes voice pool if changed)
    // Returns interleaved [L0,R0,L1,R1,...] frames.  Empty on error.
    std::vector<float> renderNote(int midi, int vel, float duration_s, int sr);

    // Convenience: render and write to a 32-bit float WAV file.
    // Returns number of frames written, or -1 on error.
    int renderNoteToFile(int midi, int vel, float duration_s,
                         int sr, const std::string& output_path);

    // SynthConfig access (changes take effect on the next renderNote call).
    void              setSynthConfig(const SynthConfig& cfg);
    SynthConfig&      getSynthConfig()       { return cfg_; }
    const SynthConfig& getSynthConfig() const { return cfg_; }

    bool isInitialized() const { return initialized_; }

    ResonatorVoiceManager& getVoiceManager() { return vm_; }

private:
    ResonatorVoiceManager vm_;
    SynthConfig           cfg_;
    Logger                logger_;               // silent by default; set in initialize()
    bool                  initialized_ = false;
    float                 current_sr_  = 0.f;

    static constexpr int   BLOCK_SIZE     = 512;
    static constexpr float MAX_DURATION_S = 15.f;   // hard ceiling for auto-detect
    static constexpr float SILENCE_DB     = -72.f;  // RMS threshold for tail cut
    static constexpr float SILENCE_SEC    = 0.3f;   // consecutive silence before stop
};

// ── WAV file writer ───────────────────────────────────────────────────────────
// Write interleaved stereo (or mono) float32 PCM as an IEEE-float WAV file.
bool writeWavF32(const std::string& path,
                 const float*       data,
                 int                n_frames,
                 int                sr,
                 int                channels = 2);
