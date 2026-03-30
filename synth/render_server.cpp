/*
 * render_server.cpp
 * ──────────────────
 * stdin/stdout JSON command dispatcher.
 *
 * Uses nlohmann/json (third_party/json.hpp) for parsing and serialisation.
 * One line in → one line out; stderr is free for diagnostic messages.
 */

#include "render_server.h"

#include "../third_party/json.hpp"

#include <iostream>
#include <string>
#include <filesystem>

using json = nlohmann::json;

// ── helpers ───────────────────────────────────────────────────────────────────

static json ok()                     { return {{"status","ok"}}; }
static json err(const std::string& m){ return {{"status","error"},{"msg",m}}; }

// Serialize SynthConfig → json object
static json config_to_json(const SynthConfig& c) {
    return {
        {"pan_spread",              c.pan_spread},
        {"pan_tilt",                c.pan_tilt},
        {"stereo_decorr",           c.stereo_decorr},
        {"stereo_decorr_midi_lo",   c.stereo_decorr_midi_lo},
        {"stereo_decorr_midi_hi",   c.stereo_decorr_midi_hi},
        {"stereo_decorr_max",       c.stereo_decorr_max},
        {"stereo_boost",            c.stereo_boost},
        {"beat_scale",              c.beat_scale},
        {"harmonic_brightness",     c.harmonic_brightness},
        {"eq_strength",             c.eq_strength},
        {"eq_freq_min",             c.eq_freq_min},
        {"noise_level",             c.noise_level},
        {"pitch_glide",             c.pitch_glide},
        {"pitch_glide_tau_ms",      c.pitch_glide_tau_ms},
        {"pitch_glide_vel_thresh",  c.pitch_glide_vel_thresh},
        {"longitudinal_precursor",  c.longitudinal_precursor},
        {"onset_ms",                c.onset_ms},
        {"target_rms",              c.target_rms},
        {"vel_gamma",               c.vel_gamma},
    };
}

// Apply json object → SynthConfig (partial update — only listed keys change)
static void apply_config(const json& j, SynthConfig& c) {
    auto f = [&](const char* k, float& v) {
        if (j.contains(k)) v = j.at(k).get<float>();
    };
    auto i = [&](const char* k, int& v) {
        if (j.contains(k)) v = j.at(k).get<int>();
    };
    f("pan_spread",             c.pan_spread);
    f("pan_tilt",               c.pan_tilt);
    f("stereo_decorr",          c.stereo_decorr);
    f("stereo_decorr_midi_lo",  c.stereo_decorr_midi_lo);
    f("stereo_decorr_midi_hi",  c.stereo_decorr_midi_hi);
    f("stereo_decorr_max",      c.stereo_decorr_max);
    f("stereo_boost",           c.stereo_boost);
    f("beat_scale",             c.beat_scale);
    f("harmonic_brightness",    c.harmonic_brightness);
    f("eq_strength",            c.eq_strength);
    f("eq_freq_min",            c.eq_freq_min);
    f("noise_level",            c.noise_level);
    f("pitch_glide",            c.pitch_glide);
    f("pitch_glide_tau_ms",     c.pitch_glide_tau_ms);
    i("pitch_glide_vel_thresh", c.pitch_glide_vel_thresh);
    f("longitudinal_precursor", c.longitudinal_precursor);
    f("onset_ms",               c.onset_ms);
    f("target_rms",             c.target_rms);
    f("vel_gamma",              c.vel_gamma);
}

// ── initialize ────────────────────────────────────────────────────────────────

bool RenderServer::initialize(const std::string& params_json_path,
                               Logger& logger)
{
    logger_      = &logger;
    params_path_ = params_json_path;
    return renderer_.initialize(params_json_path, logger);
}

// ── handleLine ────────────────────────────────────────────────────────────────

std::string RenderServer::handleLine(const std::string& line) {
    json req, resp;
    try {
        req = json::parse(line);
    } catch (const std::exception& e) {
        return err(std::string("JSON parse error: ") + e.what()).dump();
    }

    const std::string cmd = req.value("cmd", "");

    // ── ping ─────────────────────────────────────────────────────────────────
    if (cmd == "ping") {
        return ok().dump();
    }

    // ── render ────────────────────────────────────────────────────────────────
    if (cmd == "render") {
        if (!renderer_.isInitialized())
            return err("renderer not initialized").dump();

        if (!req.contains("midi") || !req.contains("vel") || !req.contains("output"))
            return err("render requires: midi, vel, output").dump();

        const int         midi       = req.at("midi").get<int>();
        const int         vel        = req.at("vel").get<int>();
        const float       duration   = req.value("duration", 0.f);
        const int         sr         = req.value("sr", 44100);
        const std::string output     = req.at("output").get<std::string>();

        // Create parent directory if needed
        try {
            std::filesystem::path p(output);
            if (p.has_parent_path())
                std::filesystem::create_directories(p.parent_path());
        } catch (...) {}

        const int n_frames = renderer_.renderNoteToFile(midi, vel, duration, sr, output);
        if (n_frames < 0)
            return err("render failed for midi=" + std::to_string(midi)).dump();

        json r = ok();
        r["frames"] = n_frames;
        return r.dump();
    }

    // ── set_config ────────────────────────────────────────────────────────────
    if (cmd == "set_config") {
        if (!req.contains("params"))
            return err("set_config requires 'params' object").dump();
        SynthConfig cfg = renderer_.getSynthConfig();
        apply_config(req.at("params"), cfg);
        renderer_.setSynthConfig(cfg);
        return ok().dump();
    }

    // ── get_config ────────────────────────────────────────────────────────────
    if (cmd == "get_config") {
        json r = ok();
        r["params"] = config_to_json(renderer_.getSynthConfig());
        return r.dump();
    }

    // ── reload ────────────────────────────────────────────────────────────────
    if (cmd == "reload") {
        const std::string path = req.value("params", params_path_);
        if (!renderer_.initialize(path, *logger_)) {
            return err("reload failed: " + path).dump();
        }
        params_path_ = path;
        return ok().dump();
    }

    // ── quit ──────────────────────────────────────────────────────────────────
    if (cmd == "quit") {
        // Caller checks for this response and exits the loop
        return json{{"status","ok"},{"quit",true}}.dump();
    }

    return err("unknown command: " + cmd).dump();
}

// ── run ───────────────────────────────────────────────────────────────────────

int RenderServer::run() {
    // Signal readiness to the Python client
    std::cout << json{{"status","ready"}}.dump() << "\n";
    std::cout.flush();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        // Strip trailing carriage return (Windows line endings in mixed envs)
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty()) continue;

        const std::string resp = handleLine(line);
        std::cout << resp << "\n";
        std::cout.flush();

        // Parse response to check for quit flag
        try {
            auto j = json::parse(resp);
            if (j.value("quit", false)) break;
        } catch (...) {}
    }
    return 0;
}
