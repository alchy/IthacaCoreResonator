/*
 * main.cpp — IthacaCoreResonator (headless real-time)
 * ─────────────────────────────────────────────────────
 * Usage:
 *   IthacaCoreResonator --params <profile.json> [--config <synthconfig.json>] [--port <N>]
 *   IthacaCoreResonator --help
 *
 * Options:
 *   --params <path>   Physics parameter profile JSON
 *                     (default: soundbanks/salamander.json)
 *   --config <path>   SynthConfig JSON — global parameters
 *                     (beat_scale, noise_level, eq_strength, …)
 *                     Applied after engine init; missing keys keep defaults.
 *   --port <N>        MIDI input port index (default: 0)
 *   --help            Show this help message
 *
 * Soundbanks (relative to executable):
 *   soundbanks/params-<bank>-nn.json    NN-trained profile (step 4)
 *   soundbanks/params-<bank>-ft.json    MRSTFT-finetuned profile (step 5)
 *   soundbanks/params-<bank>-ft.synth_config.json   global SynthConfig
 */

#include "synth/realtime/resonator_engine.h"
#include <string>
#include <cstdlib>
#include <cstdio>

static void printHelp(const char* argv0) {
    std::fprintf(stdout,
        "IthacaCoreResonator — Physics Piano Synthesizer\n"
        "\n"
        "Usage:\n"
        "  %s [--params <profile.json>] [--config <synthconfig.json>] [--port <N>]\n"
        "  %s --help\n"
        "\n"
        "Options:\n"
        "  --params <path>   Physics parameter profile JSON\n"
        "                    (default: soundbanks/salamander.json)\n"
        "  --config <path>   SynthConfig JSON with global parameters\n"
        "                    (beat_scale, noise_level, eq_strength, ...)\n"
        "                    Applied at startup; missing keys keep defaults.\n"
        "  --port <N>        MIDI input port index (default: 0)\n"
        "  --help            Show this help message\n"
        "\n"
        "Soundbank files (place next to executable in soundbanks/):\n"
        "  soundbanks/params-<bank>-nn.json              NN profile (after step 4)\n"
        "  soundbanks/params-<bank>-ft.json              Finetuned profile (after step 5)\n"
        "  soundbanks/params-<bank>-ft.synth_config.json Global SynthConfig\n"
        "\n"
        "Keyboard fallback (no MIDI hardware):\n"
        "  a s d f g h j k  ->  C4 D4 E4 F4 G4 A4 B4 C5\n"
        "  z                ->  sustain pedal (toggle)\n"
        "  q                ->  quit\n",
        argv0, argv0);
}

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::string params_json = "soundbanks/salamander.json";
    std::string config_json;
    int midi_port = 0;

    if (argc == 1) {
        // No arguments: show help and exit
        printHelp(argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--help" || a == "-h") {
            printHelp(argv[0]);
            return 0;
        } else if (a == "--params" && i + 1 < argc) {
            params_json = argv[++i];
        } else if (a == "--config" && i + 1 < argc) {
            config_json = argv[++i];
        } else if (a == "--port" && i + 1 < argc) {
            midi_port = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n\n", a.c_str());
            printHelp(argv[0]);
            return 1;
        }
    }

    Logger logger(stdout, stdout);
    logger.log("main", LogSeverity::Info,
               "IthacaCoreResonator v1.0 — Physics Piano Synthesizer");

    try {
        return runResonator(logger, params_json, midi_port, config_json);
    } catch (const std::exception& e) {
        logger.log("main", LogSeverity::Critical,
                   std::string("FATAL: ") + e.what());
        return 1;
    } catch (...) {
        logger.log("main", LogSeverity::Critical, "UNKNOWN FATAL ERROR");
        return 1;
    }
}
