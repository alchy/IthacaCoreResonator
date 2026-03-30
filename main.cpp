/*
 * main.cpp — IthacaCoreResonator
 * ─────────────────────────────────
 * Usage:
 *   IthacaCoreResonator [params.json] [midi_port]
 *
 *   params.json  — physics parameter table (default: ../analysis/params.json)
 *   midi_port    — MIDI input port index (default: 0, first available)
 *
 * At startup, lists all available MIDI ports.
 * If no MIDI hardware is present, keyboard fallback is active (a-k = C4-C5).
 * On macOS/Linux, also opens a virtual MIDI port for DAW routing.
 */

#include "synth/resonator_engine.h"
#include <string>
#include <cstdlib>
#include <cstdio>

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string params_json = (argc > 1)
        ? argv[1]
        : "soundbanks/salamander.json";
    int midi_port = (argc > 2) ? std::atoi(argv[2]) : 0;

    // log() and logRT() both go to stdout in interactive mode.
    Logger logger(stdout, stdout);
    logger.log("main", LogSeverity::Info,
               "IthacaCoreResonator v1.0 — Physics Piano Synthesizer");

    try {
        return runResonator(logger, params_json, midi_port);
    } catch (const std::exception& e) {
        logger.log("main", LogSeverity::Critical,
                   std::string("FATAL: ") + e.what());
        return 1;
    } catch (...) {
        logger.log("main", LogSeverity::Critical, "UNKNOWN FATAL ERROR");
        return 1;
    }
}
