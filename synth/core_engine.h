#pragma once
/*
 * core_engine.h
 * ──────────────
 * Generic real-time engine wrapping an ISynthCore.
 * Replaces ResonatorEngine; works with any registered core.
 *
 * Responsibilities:
 *  - Create and own an ISynthCore (via SynthCoreRegistry)
 *  - Open audio device (miniaudio), run RT callback
 *  - Thread-safe MIDI queue (lock-free ring buffer)
 *  - Master gain / pan (post-core)
 *  - LFO panning (electric-piano style, post-core)
 *  - DspChain (limiter + BBE, master bus)
 *  - Peak metering
 *
 * Usage:
 *   CoreEngine engine;
 *   engine.initialize("ResonatorCore", "soundbanks/params-ks-grand-ft.json",
 *                     "soundbanks/params-ks-grand-ft.synth_config.json", logger);
 *   engine.start();
 *   engine.noteOn(60, 80);
 *   engine.stop();
 */

#include "i_synth_core.h"
#include "../dsp/dsp_chain.h"
#include "../sampler/core_logger.h"
#include <memory>
#include <string>
#include <atomic>
#include <cstdint>

struct ma_device;

static constexpr int CORE_ENGINE_DEFAULT_SR         = 48000;
static constexpr int CORE_ENGINE_DEFAULT_BLOCK_SIZE = 256;

class CoreEngine {
public:
    CoreEngine();
    ~CoreEngine();

    // ── Initialization ────────────────────────────────────────────────────────

    // Phase 1: instantiate core by name, load params, apply optional config JSON.
    bool initialize(const std::string& core_name,
                    const std::string& params_path,
                    const std::string& config_json_path,
                    Logger&            logger);

    // Phase 2: open audio device and start RT callback.
    bool start();

    // Phase 3: stop audio device (blocks until callback thread exits).
    void stop();

    bool isRunning()     const { return running_.load(); }
    bool isInitialized() const { return core_ && core_->isLoaded(); }

    // ── Thread-safe MIDI ──────────────────────────────────────────────────────
    void noteOn      (uint8_t midi, uint8_t velocity);
    void noteOff     (uint8_t midi);
    void sustainPedal(uint8_t val);  // >=64 = down

    // ── Master mix ────────────────────────────────────────────────────────────
    void setMasterGain (uint8_t midi_val, Logger& logger);  // 0..127 → level
    void setMasterPan  (uint8_t midi_val) noexcept;         // 64 = center
    void setPanSpeed   (uint8_t midi_val) noexcept;         // 0..127 → 0..2 Hz
    void setPanDepth   (uint8_t midi_val) noexcept;         // 0..127 → 0..1

    // ── DSP chain ─────────────────────────────────────────────────────────────
    void setLimiterThreshold(uint8_t v) noexcept;
    void setLimiterRelease  (uint8_t v) noexcept;
    void setLimiterEnabled  (uint8_t v) noexcept;
    void setBBEDefinition   (uint8_t v) noexcept;
    void setBBEBassBoost    (uint8_t v) noexcept;

    // ── Accessors ─────────────────────────────────────────────────────────────
    ISynthCore*  core()        noexcept { return core_.get();  }
    DspChain*    getDspChain() noexcept { return &dsp_;        }
    Logger&      getLogger()   noexcept { return logger_;      }

    int   activeVoices()     const noexcept;
    float getOutputPeakLin() const noexcept { return output_peak_lin_.load(std::memory_order_relaxed); }

    uint8_t getLastNoteMidi() const noexcept { return last_note_midi_.load(std::memory_order_relaxed); }
    uint8_t getLastNoteVel()  const noexcept { return last_note_vel_ .load(std::memory_order_relaxed); }

    int sampleRate() const { return sample_rate_; }
    int blockSize()  const { return block_size_;  }

private:
    static void audioCallback(ma_device* device, void* output,
                               const void* input, uint32_t frame_count);
    void processBlock(float* out_l, float* out_r, int n_samples) noexcept;
    void applyMasterAndLfo(float* out_l, float* out_r, int n_samples) noexcept;

    std::unique_ptr<ISynthCore> core_;
    DspChain                    dsp_;
    Logger                      logger_;

    // Master mix state
    float master_gain_ = 1.f;
    float pan_l_       = 1.f;
    float pan_r_       = 1.f;

    // LFO panning
    float lfo_speed_   = 0.f;   // Hz
    float lfo_depth_   = 0.f;   // 0..1
    float lfo_phase_   = 0.f;   // radians

    // Audio device
    ma_device*          device_      = nullptr;
    std::atomic<bool>   running_    {false};
    int                 sample_rate_ = CORE_ENGINE_DEFAULT_SR;
    int                 block_size_  = CORE_ENGINE_DEFAULT_BLOCK_SIZE;

    float* buf_l_ = nullptr;
    float* buf_r_ = nullptr;

    // Peak metering (audio → GUI, relaxed atomic)
    std::atomic<float> output_peak_lin_{0.f};
    float              peak_decay_coeff_ = 0.9878f;

    // Last note (written on noteOn, read by GUI)
    std::atomic<uint8_t> last_note_midi_{60};
    std::atomic<uint8_t> last_note_vel_ {80};
};

// ── Convenience: full startup + interactive loop ──────────────────────────────
// Like runResonator, but selects core by name.
int runCoreEngine(Logger&            logger,
                  const std::string& core_name,
                  const std::string& params_path,
                  int                midi_port       = 0,
                  const std::string& config_json_path = "");
