/*
 * server_main.cpp — IthacaRenderServer
 * ───────────────────────────────────────
 * Headless render server for Python training-loop IPC.
 * Communicates via stdin/stdout JSON protocol (one JSON object per line).
 *
 * Usage:
 *   IthacaRenderServer [params.json] [--verbose]
 *
 *   params.json — physics parameter table
 *                 (default: analysis/params-ks-grand.json)
 *   --verbose   — print log messages to stderr (default: muted)
 *
 * On startup, writes {"status":"ready"} to stdout.
 * Then reads commands from stdin until {"cmd":"quit"}.
 * Log output goes to stderr only when --verbose is given.
 */

#include "synth/render_server.h"
#include <iostream>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

int main(int argc, char* argv[]) {
    // Unbuffered I/O — critical for line-by-line subprocess communication
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Parse args: optional params path + optional --verbose flag
    std::string params_json = "analysis/params-ks-grand.json";
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--verbose" || a == "-v") verbose = true;
        else if (a[0] != '-')             params_json = a;
    }

    if (verbose)
        std::fprintf(stderr, "[IthacaRenderServer] params: %s\n", params_json.c_str());

    try {
        // nullptr = muted; stderr = verbose
        Logger logger(".", verbose ? stderr : nullptr);
        // Heap-allocate: ResonatorVoiceManager is ~400 KB (88 voices × 5 KB each)
        auto server = std::make_unique<RenderServer>();

        if (!server->initialize(params_json, logger)) {
            std::fprintf(stderr, "[IthacaRenderServer] initialization failed\n");
            return 1;
        }

        if (verbose)
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
