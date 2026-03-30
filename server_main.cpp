/*
 * server_main.cpp — IthacaRenderServer
 * ───────────────────────────────────────
 * Headless TCP render server for Python training-loop IPC.
 *
 * Usage:
 *   IthacaRenderServer [params.json] [--port <N>] [--log <path>] [--no-log]
 *
 *   params.json  — physics parameter table
 *                  (default: analysis/params-ks-grand.json)
 *   --port <N>   — TCP port to listen on (default: 9876)
 *   --log <path> — write diagnostics to file
 *                  (default: analysis/runtime-logs/render-server.log)
 *   --no-log     — disable all logging
 *
 * No stdin, stdout, or stderr is used.
 * Client connects via TCP to 127.0.0.1:PORT after the process starts.
 * Server sends {"status":"ready"} as the first line on each accepted connection.
 */

#include "synth/render_server.h"
#include <filesystem>
#include <memory>
#include <string>
#include <cstdio>

int main(int argc, char* argv[]) {
    std::string params_json = "analysis/params-ks-grand.json";
    std::string log_path    = "analysis/runtime-logs/render-server.log";
    int         port        = 9876;
    bool        logging     = true;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a == "--port"   && i + 1 < argc) { port     = std::stoi(argv[++i]); }
        else if (a == "--log"    && i + 1 < argc) { log_path = argv[++i]; }
        else if (a == "--no-log")                 { logging  = false; }
        else if (a[0] != '-')                     { params_json = a; }
    }

    // All diagnostics go to log file — no stdout/stderr used.
    std::FILE* log_file = nullptr;
    if (logging) {
        try {
            std::filesystem::create_directories(
                std::filesystem::path(log_path).parent_path());
        } catch (...) {}
        log_file = std::fopen(log_path.c_str(), "w");
    }

    // Logger created before try so it's accessible in catch blocks.
    // log() writes to log_file; no RT channel needed (offline rendering).
    Logger logger(log_file, nullptr);
    logger.log("RenderServer", LogSeverity::Info, "params: " + params_json);
    logger.log("RenderServer", LogSeverity::Info, "port:   " + std::to_string(port));

    try {
        auto server = std::make_unique<RenderServer>();

        if (!server->initialize(params_json, logger)) {
            logger.log("RenderServer", LogSeverity::Error, "initialization failed");
            if (log_file) std::fclose(log_file);
            return 1;
        }

        int ret = server->runTCP(port);
        logger.log("RenderServer", LogSeverity::Info, "stopped");
        if (log_file) std::fclose(log_file);
        return ret;

    } catch (const std::exception& e) {
        logger.log("RenderServer", LogSeverity::Critical,
                   std::string("FATAL: ") + e.what());
        if (log_file) std::fclose(log_file);
        return 1;
    } catch (...) {
        logger.log("RenderServer", LogSeverity::Critical, "UNKNOWN FATAL ERROR");
        if (log_file) std::fclose(log_file);
        return 1;
    }
}
