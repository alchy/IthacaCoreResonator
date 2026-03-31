#pragma once
/*
 * midi_input.h
 * ─────────────
 * Cross-platform MIDI input via RtMidi.
 * Receives note-on/off, sustain pedal, and passes events to CoreEngine.
 *
 * CoreEngine is forward-declared to avoid circular inclusion
 * (core_engine.cpp includes midi_input.h).
 */

#include "../third_party/RtMidi.h"
#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

class CoreEngine;

// ── MIDI activity timestamps (updated from callback thread, read from GUI) ────
// Each field holds the steady_clock millisecond timestamp of the last event.
// 0 = never received.
struct MidiActivity {
    std::atomic<uint64_t> any_ms     {0};  // any MIDI message
    std::atomic<uint64_t> sysex_ms   {0};  // SysEx applied
    std::atomic<uint64_t> note_on_ms {0};  // Note On
    std::atomic<uint64_t> note_off_ms{0};  // Note Off
    std::atomic<uint64_t> pedal_ms   {0};  // CC 64 sustain pedal
};

class MidiInput {
public:
    MidiInput() = default;
    ~MidiInput() { close(); }

    // List available MIDI input ports (for user selection)
    static std::vector<std::string> listPorts();

    // Open port by index (0 = first available). Returns false if none found.
    bool open(CoreEngine& engine, int port_index = 0);

    // Open virtual port (macOS/Linux — allows DAW to send MIDI)
    bool openVirtual(CoreEngine& engine, const std::string& name = "IthacaCoreResonator");

    void close();
    bool isOpen() const { return midi_ && midi_->isPortOpen(); }
    std::string portName() const { return port_name_; }

    // Activity timestamps — read from any thread (GUI, main)
    const MidiActivity& activity() const { return activity_; }

private:
    static void callback(double /*timestamp*/,
                         std::vector<unsigned char>* msg,
                         void* user_data);

    RtMidiIn*   midi_    = nullptr;
    CoreEngine* engine_  = nullptr;
    std::string port_name_;
    MidiActivity activity_;
};
