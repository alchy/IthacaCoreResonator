/*
 * server_main.cpp — IthacaRenderServer
 * ───────────────────────────────────────
 * Headless render server for Python training-loop IPC.
 * Communicates via stdin/stdout JSON protocol (one JSON object per line).
 *
 * Usage:
 *   IthacaRenderServer [params.json] [--log <path>] [--no-log]
 *
 *   params.json  — physics parameter table
 *                  (default: analysis/params-ks-grand.json)
 *   --log <path> — write diagnostics to file
 *                  (default: analysis/runtime-logs/render-server.log)
 *   --no-log     — disable all logging
 *
 * stdout is reserved exclusively for the JSON protocol.
 * stderr is never written to — all diagnostics go to the log file.
 *
 * On startup, writes {"status":"ready"} to stdout.
 * Then reads commands from stdin until {"cmd":"quit"}.
 */

#include "synth/render_server.h"
#include <filesystem>
#include <memory>
#include <string>
#include <cstdio>

int main(int argc, char* argv[]) {
    // Unbuffered stdout — critical for line-by-line subprocess communication.
    // stderr is intentionally never used.
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::string params_json = "analysis/params-ks-grand.json";
    std::string log_path    = "analysis/runtime-logs/render-server.log";
    bool        logging     = true;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--log" && i + 1 < argc) { log_path = argv[++i]; }
        else if (a == "--no-log")         { logging = false; }
        else if (a[0] != '-')             { params_json = a; }
    }

    // All diagnostics go to log file — stdout stays clean for the JSON protocol.
    std::FILE* log_file = nullptr;
    if (logging) {
        try {
            std::filesystem::create_directories(
                std::filesystem::path(log_path).parent_path());
        } catch (...) {}
        log_file = std::fopen(log_path.c_str(), "w");
    }

    auto flog = [&](const char* msg) {
        if (log_file) { std::fprintf(log_file, "%s\n", msg); std::fflush(log_file); }
    };

    flog(("[IthacaRenderServer] params: " + params_json).c_str());

    try {
        Logger logger(".", log_file);
        auto server = std::make_unique<RenderServer>();

        if (!server->initialize(params_json, logger)) {
            flog("[IthacaRenderServer] initialization failed");
            if (log_file) std::fclose(log_file);
            return 1;
        }

        flog("[IthacaRenderServer] ready");
        int ret = server->run();
        if (log_file) std::fclose(log_file);
        return ret;

    } catch (const std::exception& e) {
        flog((std::string("[IthacaRenderServer] FATAL: ") + e.what()).c_str());
        if (log_file) std::fclose(log_file);
        return 1;
    } catch (...) {
        flog("[IthacaRenderServer] UNKNOWN FATAL ERROR");
        if (log_file) std::fclose(log_file);
        return 1;
    }
}
