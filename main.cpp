/*
 * main.cpp — IthacaCoreResonator (headless real-time)
 * ─────────────────────────────────────────────────────
 * Usage:
 *   IthacaCoreResonator --core <name> [--params <file.json>]
 *                       [--config <synth_config.json>]
 *                       [--core-param key=value ...]
 *                       [--port <N>]
 *   IthacaCoreResonator --help
 *   IthacaCoreResonator --list-cores
 *
 * Options:
 *   --core <name>          Synthesis core (default: SineCore)
 *   --params <path>        Core parameter file (JSON, core-specific)
 *   --config <path>        Additional SynthConfig JSON applied via setParam
 *   --core-param key=value Override a core parameter (repeatable)
 *   --port <N>             MIDI input port index (default: 0)
 *   --list-cores           Print available cores and exit
 *   --help                 Show this message
 *
 * Cores and their parameter files:
 *   SineCore   No params file needed
 *   PianoCore  --params analysis/params-piano-ks-grand.json
 *              (generate with: python analysis/export_piano_params.py)
 */

#include "synth/core_engine.h"
#include "synth/synth_core_registry.h"
#include "synth/midi_input.h"

#if defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
#  include <immintrin.h>
#  define ICR_ENABLE_FTZ() \
     _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); \
     _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)
#else
#  define ICR_ENABLE_FTZ() ((void)0)
#endif

// Pull in core registrations by including headers.
// REGISTER_SYNTH_CORE() fires at static-init time in the corresponding .cpp.
#include "synth-core/sine/sine_core.h"
#include "synth-core/piano/piano_core.h"

#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <chrono>

#ifdef _WIN32
  #include <conio.h>
#else
  #include <termios.h>
  #include <unistd.h>
  #include <fcntl.h>
#endif

static void sleepMs(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

static void printHelp(const char* argv0) {
    std::fprintf(stdout,
        "IthacaCoreResonator — Pluggable Synthesizer\n"
        "\n"
        "Usage:\n"
        "  %s --core <name> [--params <file>] [--config <file>]\n"
        "             [--core-param key=value ...] [--port <N>]\n"
        "  %s --list-cores\n"
        "  %s --help\n"
        "\n"
        "Options:\n"
        "  --core <name>          Synthesis core (default: SineCore)\n"
        "  --params <path>        Core parameter JSON (core-specific)\n"
        "  --config <path>        SynthConfig JSON applied via setParam\n"
        "  --core-param key=val   Override a core parameter (repeatable)\n"
        "  --port <N>             MIDI input port index (default: 0)\n"
        "  --list-cores           List registered cores and exit\n"
        "  --help                 Show this message\n"
        "\n"
        "Keyboard fallback (no MIDI hardware):\n"
        "  a s d f g h j k  ->  C4 D4 E4 F4 G4 A4 B4 C5\n"
        "  z                ->  sustain (toggle)\n"
        "  q                ->  quit\n",
        argv0, argv0, argv0);
}

int main(int argc, char* argv[]) {
    ICR_ENABLE_FTZ();  // prevent denormal stalls in biquad / IIR filters
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::string core_name   = "SineCore";
    std::string params_json;
    std::string config_json;
    int         midi_port   = 0;
    std::vector<std::pair<std::string,float>> core_params;  // --core-param key=value

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--help" || a == "-h") {
            printHelp(argv[0]);
            return 0;
        } else if (a == "--list-cores") {
            auto cores = SynthCoreRegistry::instance().availableCores();
            std::fprintf(stdout, "Available cores:\n");
            for (const auto& c : cores)
                std::fprintf(stdout, "  %s\n", c.c_str());
            return 0;
        } else if (a == "--core" && i + 1 < argc) {
            core_name = argv[++i];
        } else if (a == "--params" && i + 1 < argc) {
            params_json = argv[++i];
        } else if (a == "--config" && i + 1 < argc) {
            config_json = argv[++i];
        } else if (a == "--core-param" && i + 1 < argc) {
            std::string kv(argv[++i]);
            auto eq = kv.find('=');
            if (eq != std::string::npos) {
                std::string key = kv.substr(0, eq);
                float val = std::stof(kv.substr(eq + 1));
                core_params.emplace_back(key, val);
            } else {
                std::fprintf(stderr, "--core-param: expected key=value, got: %s\n",
                             kv.c_str());
                return 1;
            }
        } else if (a == "--port" && i + 1 < argc) {
            midi_port = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n\n", a.c_str());
            printHelp(argv[0]);
            return 1;
        }
    }

    Logger logger(stdout, stdout);
    logger.log("main", LogSeverity::Info, "IthacaCoreResonator — " + core_name);

    try {
        auto engine = std::make_unique<CoreEngine>();
        if (!engine->initialize(core_name, params_json, config_json, logger))
            return 1;

        // Apply --core-param overrides
        for (const auto& kv : core_params) {
            if (!engine->core()->setParam(kv.first, kv.second)) {
                logger.log("main", LogSeverity::Warning,
                           "Unknown core param: " + kv.first);
            }
        }

        if (!engine->start()) return 1;

        // Interactive loop (keyboard fallback + MIDI)
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

        // Keyboard fallback
        const char  keys[] = "asdfghjk";
        const int  midis[] = { 60, 62, 64, 65, 67, 69, 71, 72 };
        bool        sus    = false;
        logger.log("main", LogSeverity::Info,
                   "Keyboard: a-k = C4-C5  |  z = sustain  |  q = quit");

#ifdef _WIN32
        while (true) {
            if (_kbhit()) {
                int ch = _getch();
                if (ch == 'q' || ch == 'Q') break;
                if (ch == 'z') {
                    sus = !sus;
                    engine->sustainPedal(sus ? 127 : 0);
                    continue;
                }
                for (int i = 0; i < 8; i++) {
                    if (ch == keys[i]) {
                        engine->noteOn((uint8_t)midis[i], 80);
                        sleepMs(300);
                        engine->noteOff((uint8_t)midis[i]);
                    }
                }
            }
            sleepMs(1);
        }
#else
        // POSIX raw terminal
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
                if (ch == 'z') { sus = !sus; engine->sustainPedal(sus ? 127 : 0); }
                for (int i = 0; i < 8; i++) {
                    if (ch == keys[i]) {
                        engine->noteOn((uint8_t)midis[i], 80);
                        sleepMs(300);
                        engine->noteOff((uint8_t)midis[i]);
                    }
                }
            }
            sleepMs(1);
        }
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
#endif

        midi.close();
        engine->stop();
        logger.log("main", LogSeverity::Info, "=== STOPPED ===");
        return 0;

    } catch (const std::exception& e) {
        logger.log("main", LogSeverity::Critical, std::string("FATAL: ") + e.what());
        return 1;
    }
}
