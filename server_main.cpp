/*
 * server_main.cpp — IthacaRenderServer
 * ───────────────────────────────────────
 * Headless render server for Python training-loop IPC.
 * Communicates via stdin/stdout JSON protocol (one JSON object per line).
 *
 * Usage:
 *   IthacaRenderServer [params.json]
 *
 *   params.json — physics parameter table
 *                 (default: analysis/params-ks-grand.json)
 *
 * On startup, writes {"status":"ready"} to stdout.
 * Then reads commands from stdin until {"cmd":"quit"}.
 * All log output goes to stderr so stdout stays clean for the protocol.
 */

#include "synth/render_server.h"
#include <iostream>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    // Unbuffered I/O — critical for line-by-line subprocess communication
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    const std::string params_json = (argc > 1)
        ? argv[1]
        : "analysis/params-ks-grand.json";

    std::fprintf(stderr, "[IthacaRenderServer] params: %s\n", params_json.c_str());

    try {
        Logger logger(".", /*use_stderr=*/true);
        // Heap-allocate: ResonatorVoiceManager is ~400 KB (88 voices × 5 KB each)
        auto server = std::make_unique<RenderServer>();

        if (!server->initialize(params_json, logger)) {
            std::fprintf(stderr, "[IthacaRenderServer] initialization failed\n");
            return 1;
        }

        std::fprintf(stderr, "[IthacaRenderServer] ready\n");
        return server->run();

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[IthacaRenderServer] FATAL: %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "[IthacaRenderServer] UNKNOWN FATAL ERROR\n");
        return 1;
    }
}
