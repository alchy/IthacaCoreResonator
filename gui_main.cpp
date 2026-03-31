/*
 * gui_main.cpp — IthacaCoreResonatorGUI
 * ───────────────────────────────────────
 * Usage:
 *   IthacaCoreResonatorGUI --core <name> [--params <file.json>]
 *                          [--config <file.json>]
 *                          [--core-param key=value ...]
 *   IthacaCoreResonatorGUI --help
 *
 * Options:
 *   --core <name>          Synthesis core (default: SineCore)
 *   --params <path>        Core parameter JSON
 *   --config <path>        SynthConfig JSON applied via setParam
 *   --core-param key=val   Override a core parameter (repeatable)
 *   --list-cores           List registered cores and exit
 *   --help                 Show this message
 */

#include "synth/core_engine.h"
#include "synth/synth_core_registry.h"
#include "gui/resonator_gui.h"

// Core registrations
#include "synth-core/sine/sine_core.h"

#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <memory>

static void printHelp(const char* argv0) {
    std::fprintf(stdout,
        "IthacaCoreResonatorGUI — Pluggable Synthesizer (GUI)\n"
        "\n"
        "Usage:\n"
        "  %s --core <name> [--params <file>] [--config <file>]\n"
        "             [--core-param key=value ...]\n"
        "  %s --list-cores\n"
        "  %s --help\n"
        "\n"
        "Options:\n"
        "  --core <name>          Synthesis core (default: SineCore)\n"
        "  --params <path>        Core parameter JSON\n"
        "  --config <path>        SynthConfig JSON applied via setParam\n"
        "  --core-param key=val   Override a core parameter (repeatable)\n"
        "  --list-cores           List registered cores and exit\n"
        "  --help                 Show this message\n",
        argv0, argv0, argv0);
}

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::string core_name = "SineCore";
    std::string params_json;
    std::string config_json;
    std::vector<std::pair<std::string,float>> core_params;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--help" || a == "-h") {
            printHelp(argv[0]);
            return 0;
        } else if (a == "--list-cores") {
            for (const auto& c : SynthCoreRegistry::instance().availableCores())
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
                core_params.emplace_back(kv.substr(0, eq),
                                         std::stof(kv.substr(eq + 1)));
            } else {
                std::fprintf(stderr, "--core-param: expected key=value, got: %s\n",
                             kv.c_str());
                return 1;
            }
        } else {
            std::fprintf(stderr, "Unknown option: %s\n\n", a.c_str());
            printHelp(argv[0]);
            return 1;
        }
    }

    Logger logger(stdout, stdout);
    logger.log("main", LogSeverity::Info,
               "=== IthacaCoreResonatorGUI STARTING — " + core_name + " ===");

    try {
        auto engine = std::make_unique<CoreEngine>();
        if (!engine->initialize(core_name, params_json, config_json, logger)) {
            logger.log("main", LogSeverity::Error, "Engine init failed");
            return 1;
        }

        // Apply --core-param overrides
        for (const auto& kv : core_params) {
            if (!engine->core()->setParam(kv.first, kv.second))
                logger.log("main", LogSeverity::Warning,
                           "Unknown core param: " + kv.first);
        }

        if (!engine->start()) {
            logger.log("main", LogSeverity::Error, "Audio start failed");
            return 1;
        }

        int ret = runResonatorGui(*engine, logger);

        engine->stop();
        logger.log("main", LogSeverity::Info,
                   "=== IthacaCoreResonatorGUI STOPPED ===");
        return ret;

    } catch (const std::exception& e) {
        logger.log("main", LogSeverity::Critical,
                   std::string("FATAL: ") + e.what());
        return 1;
    }
}
