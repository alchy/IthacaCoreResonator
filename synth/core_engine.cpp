/*
 * core_engine.cpp
 * ────────────────
 * Generic RT engine for any ISynthCore.
 *
 * Audio thread flow:
 *   audioCallback()
 *     → processBlock(L*, R*, n)
 *         → drain MIDI queue → core->noteOn/Off/sustainPedal
 *         → memset buffers to 0
 *         → core->processBlock(L, R, n)     [additive]
 *         → applyMasterAndLfo(L, R, n)
 *         → dsp_.process(L, R, n)
 *         → interleave L+R → float32 stereo output
 *         → update peak meter
 */

// miniaudio implementation — compiled once here
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "core_engine.h"
#include "synth_core_registry.h"
#include "midi_input.h"
#include "../third_party/json.hpp"

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <fstream>

#ifdef _WIN32
  #include <conio.h>
#else
  #include <termios.h>
  #include <unistd.h>
  #include <fcntl.h>
#endif

// ── MIDI event queue — see pushMidiEvt (now a CoreEngine member) ─────────────

// ── MIDI queue (instance-local, SPSC lock-free) ───────────────────────────────

void CoreEngine::pushMidiEvt(MidiEvt::Type t, uint8_t midi, uint8_t val) noexcept {
    int w    = midi_w_.load(std::memory_order_relaxed);
    int next = (w + 1) % MIDI_Q_SIZE;
    if (next == midi_r_.load(std::memory_order_acquire)) return;  // full, drop
    midi_q_[w] = {t, midi, val};
    midi_w_.store(next, std::memory_order_release);
}

// ── Constructor / Destructor ──────────────────────────────────────────────────

CoreEngine::CoreEngine()
    : device_(new ma_device{}) {}

CoreEngine::~CoreEngine() {
    stop();
    delete[] buf_l_;
    delete[] buf_r_;
    delete device_;
}

// ── Helper: apply a JSON config file to a core via setParam ──────────────────

static void applyConfigJson(const std::string& path, ISynthCore* core,
                             Logger& logger) {
    if (path.empty()) return;
    std::ifstream f(path);
    if (!f) {
        logger.log("CoreEngine", LogSeverity::Warning,
                   "Config file not found: " + path);
        return;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) {
        logger.log("CoreEngine", LogSeverity::Warning,
                   std::string("Config parse error: ") + e.what());
        return;
    }
    int applied = 0;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it->is_number()) {
            float v = it->get<float>();
            if (core->setParam(it.key(), v)) ++applied;
        }
    }
    logger.log("CoreEngine", LogSeverity::Info,
               "Config loaded: " + path +
               " (" + std::to_string(applied) + " params applied)");
}

// ── initialize ────────────────────────────────────────────────────────────────

bool CoreEngine::initialize(const std::string& core_name,
                             const std::string& params_path,
                             const std::string& config_json_path,
                             Logger&            logger) {
    logger_ = logger;
    logger_.log("CoreEngine", LogSeverity::Info,
                "Initializing core: " + core_name);

    core_ = SynthCoreRegistry::instance().create(core_name);
    if (!core_) {
        logger_.log("CoreEngine", LogSeverity::Error,
                    "Unknown core: '" + core_name + "'. Available:");
        for (const auto& n : SynthCoreRegistry::instance().availableCores())
            logger_.log("CoreEngine", LogSeverity::Info, "  - " + n);
        return false;
    }

    if (!core_->load(params_path, (float)sample_rate_, logger_)) {
        logger_.log("CoreEngine", LogSeverity::Error, "Core load failed");
        return false;
    }

    applyConfigJson(config_json_path, core_.get(), logger_);

    delete[] buf_l_;
    delete[] buf_r_;
    buf_l_ = new float[block_size_];
    buf_r_ = new float[block_size_];
    dsp_.prepare((float)sample_rate_, block_size_);

    float bps = (float)sample_rate_ / (float)block_size_;
    peak_decay_coeff_ = std::pow(10.f, -1.f / bps);  // -20 dB/s

    logger_.log("CoreEngine", LogSeverity::Info,
        std::string("Ready. Core=") + core_->coreName() +
        " SR=" + std::to_string(sample_rate_) +
        " block=" + std::to_string(block_size_));
    return true;
}

// ── Audio callback ────────────────────────────────────────────────────────────

void CoreEngine::audioCallback(ma_device*  device,
                                void*       output,
                                const void* /*input*/,
                                uint32_t    frame_count) {
    auto* eng = reinterpret_cast<CoreEngine*>(device->pUserData);
    // Interleave L+R into float32 output
    uint32_t rem = frame_count;
    uint32_t off = 0;
    while (rem > 0) {
        uint32_t chunk = rem < (uint32_t)eng->block_size_
                       ? rem : (uint32_t)eng->block_size_;
        eng->processBlock(eng->buf_l_, eng->buf_r_, (int)chunk);
        float* dst = reinterpret_cast<float*>(output) + off * 2;
        for (uint32_t i = 0; i < chunk; i++) {
            dst[i*2]   = eng->buf_l_[i];
            dst[i*2+1] = eng->buf_r_[i];
        }
        off += chunk;
        rem -= chunk;
    }
}

void CoreEngine::processBlock(float* out_l, float* out_r, int n) noexcept {
    // Drain MIDI queue
    int r = midi_r_.load(std::memory_order_acquire);
    int w = midi_w_.load(std::memory_order_relaxed);
    while (r != w) {
        const MidiEvt& ev = midi_q_[r];
        switch (ev.type) {
            case MidiEvt::NOTE_ON:       core_->noteOn(ev.midi, ev.value);    break;
            case MidiEvt::NOTE_OFF:      core_->noteOff(ev.midi);             break;
            case MidiEvt::SUSTAIN:       core_->sustainPedal(ev.value >= 64); break;
            case MidiEvt::ALL_NOTES_OFF: core_->allNotesOff();                break;
        }
        r = (r + 1) % MIDI_Q_SIZE;
    }
    midi_r_.store(r, std::memory_order_release);

    // Zero buffers (core output is additive)
    std::memset(out_l, 0, n * sizeof(float));
    std::memset(out_r, 0, n * sizeof(float));

    // Core synthesis
    if (core_) core_->processBlock(out_l, out_r, n);

    // Master gain / LFO pan / DSP
    applyMasterAndLfo(out_l, out_r, n);
    dsp_.process(out_l, out_r, n);

    // Peak metering
    float peak = 0.f;
    for (int i = 0; i < n; i++) {
        float s = std::abs(out_l[i]) > std::abs(out_r[i])
                ? std::abs(out_l[i]) : std::abs(out_r[i]);
        if (s > peak) peak = s;
    }
    float cur = output_peak_lin_.load(std::memory_order_relaxed);
    cur = cur * peak_decay_coeff_;
    if (peak > cur) cur = peak;
    output_peak_lin_.store(cur, std::memory_order_relaxed);
}

void CoreEngine::applyMasterAndLfo(float* out_l, float* out_r,
                                    int n) noexcept {
    static constexpr float PI  = 3.14159265358979f;
    static constexpr float TAU = 2.f * PI;

    // Load atomics once — avoid repeated atomic reads in inner loop
    const float mg    = master_gain_.load(std::memory_order_relaxed);
    const float pl    = pan_l_      .load(std::memory_order_relaxed);
    const float pr    = pan_r_      .load(std::memory_order_relaxed);
    const float speed = lfo_speed_  .load(std::memory_order_relaxed);
    const float depth = lfo_depth_  .load(std::memory_order_relaxed);

    float mg_l = mg * pl;
    float mg_r = mg * pr;

    if (speed > 0.f && depth > 0.f) {
        float d_phase = TAU * speed / (float)sample_rate_;
        for (int i = 0; i < n; i++) {
            float lfo   = depth * std::sin(lfo_phase_);
            float lm    = mg_l * (1.f - lfo);
            float rm    = mg_r * (1.f + lfo);
            out_l[i] *= lm;
            out_r[i] *= rm;
            lfo_phase_ += d_phase;
            if (lfo_phase_ >= TAU) lfo_phase_ -= TAU;
        }
    } else {
        for (int i = 0; i < n; i++) {
            out_l[i] *= mg_l;
            out_r[i] *= mg_r;
        }
    }
}

// ── start / stop ─────────────────────────────────────────────────────────────

bool CoreEngine::start() {
    if (!isInitialized()) return false;

    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format    = ma_format_f32;
    cfg.playback.channels  = 2;
    cfg.sampleRate         = (ma_uint32)sample_rate_;
    cfg.dataCallback       = audioCallback;
    cfg.pUserData          = this;
    cfg.periodSizeInFrames = (ma_uint32)block_size_;

    if (ma_device_init(nullptr, &cfg, device_) != MA_SUCCESS) {
        logger_.log("CoreEngine", LogSeverity::Error, "Failed to open audio device");
        return false;
    }
    if (ma_device_start(device_) != MA_SUCCESS) {
        logger_.log("CoreEngine", LogSeverity::Error, "Failed to start audio device");
        ma_device_uninit(device_);
        return false;
    }
    running_.store(true);
    logger_.log("CoreEngine", LogSeverity::Info,
        "Audio started: " + std::string(device_->playback.name));
    return true;
}

void CoreEngine::stop() {
    if (!running_.load()) return;
    ma_device_stop(device_);
    ma_device_uninit(device_);
    running_.store(false);
    logger_.log("CoreEngine", LogSeverity::Info, "Audio stopped");
}

// ── Thread-safe MIDI ──────────────────────────────────────────────────────────

void CoreEngine::noteOn(uint8_t midi, uint8_t vel) {
    last_note_midi_.store(midi, std::memory_order_relaxed);
    last_note_vel_ .store(vel,  std::memory_order_relaxed);
    pushMidiEvt(MidiEvt::NOTE_ON, midi, vel);
}
void CoreEngine::noteOff(uint8_t midi) {
    pushMidiEvt(MidiEvt::NOTE_OFF, midi, 0);
}
void CoreEngine::sustainPedal(uint8_t val) {
    pushMidiEvt(MidiEvt::SUSTAIN, 0, val);
}

void CoreEngine::allNotesOff() {
    pushMidiEvt(MidiEvt::ALL_NOTES_OFF, 0, 0);
}

// ── Master mix ────────────────────────────────────────────────────────────────

void CoreEngine::setMasterGain(uint8_t v, Logger& logger) {
    master_gain_.store((v / 127.f) * (v / 127.f) * 2.f,  // square law, 0..2
                       std::memory_order_relaxed);
    logger.log("CoreEngine", LogSeverity::Info,
               "Master gain MIDI=" + std::to_string(v));
}

void CoreEngine::setMasterPan(uint8_t v) noexcept {
    float norm = (v - 64) / 64.f;  // -1..+1
    if (norm <= 0.f) {
        pan_l_.store(1.f,          std::memory_order_relaxed);
        pan_r_.store(1.f + norm,   std::memory_order_relaxed);
    } else {
        pan_l_.store(1.f - norm,   std::memory_order_relaxed);
        pan_r_.store(1.f,          std::memory_order_relaxed);
    }
}

void CoreEngine::setPanSpeed(uint8_t v) noexcept {
    lfo_speed_.store(2.f * (v / 127.f), std::memory_order_relaxed);   // 0..2 Hz
}

void CoreEngine::setPanDepth(uint8_t v) noexcept {
    lfo_depth_.store(v / 127.f, std::memory_order_relaxed);
}

// ── DSP chain ─────────────────────────────────────────────────────────────────

void CoreEngine::setLimiterThreshold(uint8_t v) noexcept { dsp_.setLimiterThreshold(v); }
void CoreEngine::setLimiterRelease  (uint8_t v) noexcept { dsp_.setLimiterRelease(v);   }
void CoreEngine::setLimiterEnabled  (uint8_t v) noexcept { dsp_.setLimiterEnabled(v);   }
void CoreEngine::setBBEDefinition   (uint8_t v) noexcept { dsp_.setBBEDefinition(v);    }
void CoreEngine::setBBEBassBoost    (uint8_t v) noexcept { dsp_.setBBEBassBoost(v);     }

// ── Stats ─────────────────────────────────────────────────────────────────────

int CoreEngine::activeVoices() const {
    if (!core_) return 0;
    return core_->getVizState().active_voice_count;
}

// ── runCoreEngine — interactive loop ─────────────────────────────────────────

int runCoreEngine(Logger&            logger,
                  const std::string& core_name,
                  const std::string& params_path,
                  int                midi_port,
                  const std::string& config_json_path) {
    logger.log("runCoreEngine", LogSeverity::Info,
               "=== IthacaCoreResonator STARTING ===");

    auto engine = std::make_unique<CoreEngine>();
    if (!engine->initialize(core_name, params_path, config_json_path, logger)) {
        logger.log("runCoreEngine", LogSeverity::Error, "Initialization failed");
        return 1;
    }
    if (!engine->start()) {
        logger.log("runCoreEngine", LogSeverity::Error, "Audio start failed");
        return 1;
    }

    MidiInput midi;
    auto ports = MidiInput::listPorts();
    if (!ports.empty()) {
        for (int i = 0; i < (int)ports.size(); i++)
            logger.log("MIDI", LogSeverity::Info,
                       "port [" + std::to_string(i) + "] " + ports[i]);
        midi.open(*engine, midi_port);
    }
#ifndef _WIN32
    if (!midi.isOpen()) midi.openVirtual(*engine);
#endif

    const char  keys[] = "asdfghjk";
    const int  midis[] = { 60, 62, 64, 65, 67, 69, 71, 72 };
    bool       sustain = false;
    logger.log("runCoreEngine", LogSeverity::Info,
               "Keyboard: a-k = C4-C5  |  z = sustain  |  q = quit");

#ifdef _WIN32
    while (true) {
        if (_kbhit()) {
            int ch = _getch();
            if (ch == 'q' || ch == 'Q') break;
            if (ch == 'z') {
                sustain = !sustain;
                engine->sustainPedal(sustain ? 127 : 0);
                continue;
            }
            for (int i = 0; i < 8; i++) {
                if (ch == keys[i]) {
                    engine->noteOn((uint8_t)midis[i], 80);
                    ma_sleep(300);
                    engine->noteOff((uint8_t)midis[i]);
                }
            }
        }
        ma_sleep(1);
    }
#else
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    while (true) {
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1) {
            if (ch == 'q' || ch == 'Q') break;
            if (ch == 'z') {
                sustain = !sustain;
                engine->sustainPedal(sustain ? 127 : 0);
            }
            for (int i = 0; i < 8; i++) {
                if (ch == keys[i]) {
                    engine->noteOn((uint8_t)midis[i], 80);
                    ma_sleep(300);
                    engine->noteOff((uint8_t)midis[i]);
                }
            }
        }
        ma_sleep(1);
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
#endif

    midi.close();
    engine->stop();
    logger.log("runCoreEngine", LogSeverity::Info,
               "=== IthacaCoreResonator STOPPED ===");
    return 0;
}
