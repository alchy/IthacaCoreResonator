/*
 * server_main.cpp — IthacaRenderServer
 * ───────────────────────────────────────
 * Headless TCP render server for Python training-loop IPC.
 *
 * Usage:
 *   IthacaRenderServer --params <profile.json> [--config <synthconfig.json>]
 *                      [--port <N>] [--log <path>] [--no-log]
 *   IthacaRenderServer --help
 *
 * Options:
 *   --params <path>   Physics parameter profile JSON
 *                     (default: soundbanks/params-ks-grand-nn.json)
 *   --config <path>   SynthConfig JSON — global parameters applied at startup
 *                     (beat_scale, noise_level, …). Can also be changed via
 *                     TCP {"cmd":"set_config",...} at runtime.
 *   --port <N>        TCP port to listen on (default: 9876)
 *   --log <path>      Write diagnostics to file
 *                     (default: analysis/runtime-logs/render-server.log)
 *   --no-log          Disable all logging
 *   --help            Show this help message
 *
 * No stdin, stdout, or stderr is used during operation.
 * Client connects via TCP to 127.0.0.1:PORT after the process starts.
 * Server sends {"status":"ready"} as the first line on each accepted connection.
 */

#include "synth/render_server.h"
#include "synth/synth_config_io.h"
#include <filesystem>
#include <memory>
#include <string>
#include <cstdio>

static void printHelp(const char* argv0) {
    std::fprintf(stdout,
        "IthacaRenderServer — Headless offline render server\n"
        "\n"
        "Usage:\n"
        "  %s --params <profile.json> [--config <synthconfig.json>]\n"
        "      [--port <N>] [--log <path>] [--no-log]\n"
        "  %s --help\n"
        "\n"
        "Options:\n"
        "  --params <path>   Physics parameter profile JSON\n"
        "                    (default: soundbanks/params-ks-grand-nn.json)\n"
        "  --config <path>   SynthConfig JSON applied at startup\n"
        "                    (beat_scale, noise_level, ...)\n"
        "  --port <N>        TCP port (default: 9876)\n"
        "  --log <path>      Log file path\n"
        "                    (default: analysis/runtime-logs/render-server.log)\n"
        "  --no-log          Disable all logging\n"
        "  --help            Show this help message\n"
        "\n"
        "TCP protocol: one JSON object per line.\n"
        "Commands: ping, render, set_config, get_config, reload, sysex, quit\n",
        argv0, argv0);
}

int main(int argc, char* argv[]) {
    std::string params_json = "soundbanks/params-ks-grand-nn.json";
    std::string config_json;
    std::string log_path    = "analysis/runtime-logs/render-server.log";
    int         port        = 9876;
    bool        logging     = true;

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
        } else if (a == "--port"   && i + 1 < argc) {
            port        = std::stoi(argv[++i]);
        } else if (a == "--log"    && i + 1 < argc) {
            log_path    = argv[++i];
        } else if (a == "--no-log") {
            logging     = false;
        } else if (a[0] != '-') {
            // backward-compat: bare positional treated as params path
            params_json = a;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n\n", a.c_str());
            printHelp(argv[0]);
            return 1;
        }
    }

    // All diagnostics go to log file — no stdout/stderr used during operation.
    std::FILE* log_file = nullptr;
    if (logging) {
        try {
            std::filesystem::create_directories(
                std::filesystem::path(log_path).parent_path());
        } catch (...) {}
        log_file = std::fopen(log_path.c_str(), "w");
    }

    Logger logger(log_file, nullptr);
    logger.log("RenderServer", LogSeverity::Info, "params: " + params_json);
    if (!config_json.empty())
        logger.log("RenderServer", LogSeverity::Info, "config: " + config_json);
    logger.log("RenderServer", LogSeverity::Info, "port:   " + std::to_string(port));

    try {
        auto server = std::make_unique<RenderServer>();

        if (!server->initialize(params_json, logger)) {
            logger.log("RenderServer", LogSeverity::Error, "initialization failed");
            if (log_file) std::fclose(log_file);
            return 1;
        }

        // Apply optional SynthConfig JSON at startup
        if (!config_json.empty()) {
            SynthConfig cfg = server->getRenderer().getSynthConfig();
            std::string err;
            if (loadSynthConfig(config_json, cfg, &err)) {
                server->getRenderer().setSynthConfig(cfg);
                logger.log("RenderServer", LogSeverity::Info,
                           "SynthConfig loaded: " + config_json);
            } else {
                logger.log("RenderServer", LogSeverity::Warning,
                           "SynthConfig load failed: " + err);
            }
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
