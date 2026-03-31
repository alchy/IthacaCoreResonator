/*
 * gui_main.cpp — IthacaCoreResonatorGUI
 * ────────────────────────────────────
 * Usage:
 *   IthacaCoreResonatorGUI [--params <profile.json>] [--config <synthconfig.json>]
 *   IthacaCoreResonatorGUI --help
 *
 * Options:
 *   --params <path>   Physics parameter profile JSON
 *                     (default: soundbanks/salamander.json)
 *   --config <path>   SynthConfig JSON — global parameters applied at startup
 *   --help            Show this help message
 */
#include "synth/resonator_engine.h"
#include "synth/synth_config_io.h"
#include "gui/resonator_gui.h"
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>

static void printHelp(const char* argv0) {
    std::fprintf(stdout,
        "IthacaCoreResonatorGUI — Physics Piano Synthesizer (GUI)\n"
        "\n"
        "Usage:\n"
        "  %s [--params <profile.json>] [--config <synthconfig.json>]\n"
        "  %s --help\n"
        "\n"
        "Options:\n"
        "  --params <path>   Physics parameter profile JSON\n"
        "                    (default: soundbanks/salamander.json)\n"
        "  --config <path>   SynthConfig JSON with global parameters\n"
        "                    (beat_scale, noise_level, eq_strength, ...)\n"
        "                    Applied at startup; can be overridden via GUI sliders.\n"
        "  --help            Show this help message\n"
        "\n"
        "Soundbank files (place next to executable in soundbanks/):\n"
        "  soundbanks/params-<bank>-nn.json              NN profile (after step 4)\n"
        "  soundbanks/params-<bank>-ft.json              Finetuned profile (after step 5)\n"
        "  soundbanks/params-<bank>-ft.synth_config.json Global SynthConfig\n",
        argv0, argv0);
}

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::string params_json = "soundbanks/salamander.json";
    std::string config_json;

    if (argc == 1) {
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
        } else {
            std::fprintf(stderr, "Unknown option: %s\n\n", a.c_str());
            printHelp(argv[0]);
            return 1;
        }
    }

    Logger logger(stdout, stdout);
    logger.log("main", LogSeverity::Info, "=== IthacaCoreResonatorGUI STARTING ===");

    try {
        auto engine = std::make_unique<ResonatorEngine>();
        if (!engine->initialize(params_json, logger)) {
            logger.log("main", LogSeverity::Error, "Engine init failed");
            return 1;
        }

        // Apply optional SynthConfig JSON before audio starts
        if (!config_json.empty()) {
            SynthConfig cfg = engine->getSynthConfig();
            std::string err;
            if (loadSynthConfig(config_json, cfg, &err)) {
                engine->getVoiceManager().setSynthConfig(cfg);
                logger.log("main", LogSeverity::Info,
                           "SynthConfig loaded: " + config_json);
            } else {
                logger.log("main", LogSeverity::Warning,
                           "SynthConfig load failed: " + err);
            }
        }

        if (!engine->start()) {
            logger.log("main", LogSeverity::Error, "Audio start failed");
            return 1;
        }

        int ret = runResonatorGui(*engine, logger, params_json);

        engine->stop();
        logger.log("main", LogSeverity::Info, "=== IthacaCoreResonatorGUI STOPPED ===");
        return ret;

    } catch (const std::exception& e) {
        logger.log("main", LogSeverity::Critical,
                   std::string("FATAL: ") + e.what());
        return 1;
    }
}
