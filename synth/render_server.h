#pragma once
/*
 * render_server.h
 * ─────────────────
 * stdin/stdout JSON render server for Python training-loop IPC.
 *
 * One JSON object per line on stdin; one JSON response per line on stdout.
 * All responses have at minimum {"status":"ok"} or {"status":"error","msg":"..."}.
 *
 * Supported commands
 * ──────────────────
 *   {"cmd":"ping"}
 *       → {"status":"ok"}
 *
 *   {"cmd":"render","midi":60,"vel":80,"sr":44100,
 *                   "duration":3.0,"output":"exports/note.wav"}
 *       duration: seconds; 0 or omitted = auto-detect silence tail
 *       → {"status":"ok","frames":N}
 *
 *   {"cmd":"set_config","params":{"beat_scale":1.5,"eq_strength":0.8,...}}
 *       (any subset of SynthConfig fields by name)
 *       → {"status":"ok"}
 *
 *   {"cmd":"get_config"}
 *       → {"status":"ok","params":{...full SynthConfig...}}
 *
 *   {"cmd":"reload","params":"path/to/params.json"}
 *       Reload params JSON at runtime (e.g. after extract-params finishes).
 *       → {"status":"ok"}
 *
 *   {"cmd":"sysex","bytes":[0xF0,0x7D,...,0xF7]}
 *       Apply a raw ICR SysEx message; syncs SynthConfig so get_config reflects result.
 *       → {"status":"ok","applied":true/false}
 *
 *   {"cmd":"quit"}
 *       → {"status":"ok"}   (then server exits)
 */

#include "offline_renderer.h"
#include "../sampler/core_logger.h"
#include <string>

class RenderServer {
public:
    // Initialize renderer with params JSON path.
    bool initialize(const std::string& params_json_path, Logger& logger);

    // Run the stdin/stdout command loop.  Blocks until {"cmd":"quit"}.
    // Returns 0 on clean exit, non-zero on fatal error.
    int run();

private:
    // Process one JSON line, return a JSON response string (no trailing newline).
    std::string handleLine(const std::string& line);

    OfflineRenderer renderer_;
    Logger*         logger_      = nullptr;
    std::string     params_path_;
};
