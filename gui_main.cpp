/*
 * gui_main.cpp — IthacaCoreResonatorGUI
 * ────────────────────────────────────
 * Launches the engine + Dear ImGui window.
 * Usage:
 *   IthacaCoreResonatorGUI [params.json] [midi_port]
 */
#include "synth/resonator_engine.h"
#include "gui/resonator_gui.h"
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string params_json = (argc > 1)
        ? argv[1]
        : "soundbanks/salamander.json";

    // log() and logRT() both go to stdout in interactive mode.
    Logger logger(stdout, stdout);
    logger.log("main", LogSeverity::Info, "=== IthacaCoreResonatorGUI STARTING ===");

    try {
        auto engine = std::make_unique<ResonatorEngine>();
        if (!engine->initialize(params_json, logger)) {
            logger.log("main", LogSeverity::Error, "Engine init failed");
            return 1;
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
