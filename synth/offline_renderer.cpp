/*
 * offline_renderer.cpp
 * ─────────────────────
 * Headless note rendering — no audio device, pure software.
 *
 * Algorithm:
 *   1. Trigger noteOn on the VoiceManager.
 *   2. Render block-by-block until either:
 *        a. Requested duration is reached, or
 *        b. Output RMS drops below SILENCE_DB for SILENCE_SEC (auto-detect).
 *   3. Optionally write the result as a 32-bit float WAV file.
 */

#include "offline_renderer.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <fstream>
#include <algorithm>

// ── initialize ────────────────────────────────────────────────────────────────

bool OfflineRenderer::initialize(const std::string& params_json_path,
                                  Logger& logger,
                                  float   sr)
{
    logger_ = &logger;
    try {
        vm_.initialize(params_json_path, sr, logger);
        vm_.prepareToPlay(BLOCK_SIZE);
        // Push our config into the voice manager (no-op initially, cfg_ is default)
        current_sr_  = sr;
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        logger.log("OfflineRenderer", LogSeverity::Error,
                   std::string("initialize failed: ") + e.what());
        return false;
    }
}

// ── renderNote ────────────────────────────────────────────────────────────────

std::vector<float> OfflineRenderer::renderNote(int   midi,
                                                int   vel,
                                                float duration_s,
                                                int   sr)
{
    if (!initialized_) return {};

    // Re-initialize voice pool if sample rate changed
    if (static_cast<float>(sr) != current_sr_) {
        vm_.changeSampleRate(static_cast<float>(sr), *logger_);
        vm_.prepareToPlay(BLOCK_SIZE);
        current_sr_ = static_cast<float>(sr);
    }

    // Push current SynthConfig into vm_ field-by-field
    vm_.setSynthPanSpread         (cfg_.pan_spread);
    vm_.setSynthBeatScale         (cfg_.beat_scale);
    vm_.setSynthStereoDecorr      (cfg_.stereo_decorr);
    vm_.setSynthStereoBoost       (cfg_.stereo_boost);
    vm_.setSynthEqStrength        (cfg_.eq_strength);
    vm_.setSynthEqFreqMin         (cfg_.eq_freq_min);
    vm_.setSynthNoiseLevel        (cfg_.noise_level);
    vm_.setSynthOnsetMs           (cfg_.onset_ms);
    vm_.setSynthHarmonicBrightness(cfg_.harmonic_brightness);
    vm_.setSynthTargetRms         (cfg_.target_rms);
    vm_.setSynthVelGamma          (cfg_.vel_gamma);

    // Stop any lingering voices
    vm_.stopAllVoices();

    // Trigger the note
    vm_.setNoteStateMIDI(static_cast<uint8_t>(midi), true,
                          static_cast<uint8_t>(vel));

    // Determine sample budget
    const float  max_dur   = (duration_s > 0.f) ? duration_s : MAX_DURATION_S;
    const int    max_frames = static_cast<int>(max_dur * static_cast<float>(sr));

    // Silence detection counters
    const float silence_lin    = std::pow(10.f, SILENCE_DB / 20.f);
    const int   silence_frames = static_cast<int>(SILENCE_SEC * static_cast<float>(sr));
    int         quiet_count    = 0;
    const bool  auto_detect    = (duration_s <= 0.f);

    // Output buffer (interleaved stereo)
    std::vector<float> out;
    out.reserve(static_cast<size_t>(max_frames) * 2);

    // Temp mono buffers for processBlockUninterleaved
    std::vector<float> buf_l(BLOCK_SIZE, 0.f);
    std::vector<float> buf_r(BLOCK_SIZE, 0.f);

    int rendered = 0;
    while (rendered < max_frames) {
        const int block = std::min(BLOCK_SIZE, max_frames - rendered);

        std::fill(buf_l.begin(), buf_l.begin() + block, 0.f);
        std::fill(buf_r.begin(), buf_r.begin() + block, 0.f);

        vm_.processBlockUninterleaved(buf_l.data(), buf_r.data(), block);

        // Interleave
        for (int i = 0; i < block; ++i) {
            out.push_back(buf_l[i]);
            out.push_back(buf_r[i]);
        }
        rendered += block;

        // Silence detection (only in auto-detect mode)
        if (auto_detect) {
            float sum2 = 0.f;
            for (int i = 0; i < block; ++i) {
                sum2 += buf_l[i] * buf_l[i] + buf_r[i] * buf_r[i];
            }
            const float rms = std::sqrt(sum2 / (2.f * static_cast<float>(block)));
            if (rms < silence_lin) {
                quiet_count += block;
                if (quiet_count >= silence_frames) break;
            } else {
                quiet_count = 0;
            }
        }
    }

    vm_.stopAllVoices();
    return out;
}

// ── renderNoteToFile ──────────────────────────────────────────────────────────

int OfflineRenderer::renderNoteToFile(int midi, int vel, float duration_s,
                                       int sr, const std::string& output_path)
{
    auto pcm = renderNote(midi, vel, duration_s, sr);
    if (pcm.empty()) return -1;

    const int n_frames = static_cast<int>(pcm.size()) / 2;
    if (!writeWavF32(output_path, pcm.data(), n_frames, sr, 2)) {
        if (logger_) logger_->log("OfflineRenderer", LogSeverity::Error,
                                   "writeWavF32 failed: " + output_path);
        return -1;
    }
    return n_frames;
}

// ── setSynthConfig ────────────────────────────────────────────────────────────

void OfflineRenderer::setSynthConfig(const SynthConfig& cfg) {
    cfg_ = cfg;
}

// ── writeWavF32 ───────────────────────────────────────────────────────────────
//
// IEEE-float (format code 3) WAV file, 32-bit, 2 channels.
// Layout: RIFF → fmt (18 bytes + 2-byte extension size) → fact → data.

bool writeWavF32(const std::string& path,
                 const float*       data,
                 int                n_frames,
                 int                sr,
                 int                channels)
{
    // Helper: write little-endian integer
    auto write16 = [](std::ofstream& f, uint16_t v) {
        f.write(reinterpret_cast<const char*>(&v), 2);
    };
    auto write32 = [](std::ofstream& f, uint32_t v) {
        f.write(reinterpret_cast<const char*>(&v), 4);
    };

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    const uint32_t n_samples    = static_cast<uint32_t>(n_frames * channels);
    const uint32_t data_bytes   = n_samples * 4;  // 4 bytes per float32 sample
    const uint16_t block_align  = static_cast<uint16_t>(channels * 4);
    const uint32_t byte_rate    = static_cast<uint32_t>(sr) * block_align;

    // ── RIFF header ─────────────────────────────────────────────────────────
    // chunk size = 4 (WAVE) + 8+26 (fmt) + 8+4 (fact) + 8+data_bytes (data)
    const uint32_t riff_size = 4 + (8 + 26) + (8 + 4) + (8 + data_bytes);
    f.write("RIFF", 4);
    write32(f, riff_size);
    f.write("WAVE", 4);

    // ── fmt chunk (IEEE float needs 18-byte fmt + 2-byte extension) ─────────
    f.write("fmt ", 4);
    write32(f, 18);              // chunk size
    write16(f, 3);               // wFormatTag = WAVE_FORMAT_IEEE_FLOAT
    write16(f, static_cast<uint16_t>(channels));
    write32(f, static_cast<uint32_t>(sr));
    write32(f, byte_rate);
    write16(f, block_align);
    write16(f, 32);              // wBitsPerSample
    write16(f, 0);               // cbSize (extension size = 0)

    // ── fact chunk (required for non-PCM formats) ────────────────────────────
    f.write("fact", 4);
    write32(f, 4);               // chunk size
    write32(f, static_cast<uint32_t>(n_frames));  // dwSampleLength

    // ── data chunk ──────────────────────────────────────────────────────────
    f.write("data", 4);
    write32(f, data_bytes);
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(data_bytes));

    return f.good();
}
