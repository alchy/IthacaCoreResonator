/*
 * render_server.cpp
 * ──────────────────
 * TCP JSON command dispatcher.
 *
 * Uses nlohmann/json (third_party/json.hpp) for parsing and serialisation.
 * One JSON line per message in each direction; newline-terminated.
 * No stdin, stdout, or stderr is used.
 */

#include "render_server.h"

#include "../third_party/json.hpp"
#include "sysex.h"

#include <string>
#include <cstring>
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
    logger_      = logger;
    params_path_ = params_json_path;
    return renderer_.initialize(params_json_path, logger_);
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
        if (!renderer_.initialize(path, logger_)) {
            return err("reload failed: " + path).dump();
        }
        params_path_ = path;
        return ok().dump();
    }

    // ── sysex ─────────────────────────────────────────────────────────────────
    // Apply a raw SysEx message (as array of ints) and sync SynthConfig back.
    // Useful for verifying that key→value routing is correct via get_config.
    if (cmd == "sysex") {
        if (!req.contains("bytes"))
            return err("sysex requires 'bytes' array").dump();
        std::vector<uint8_t> msg;
        for (auto& b : req.at("bytes"))
            msg.push_back(static_cast<uint8_t>(b.get<int>()));
        const bool applied = sysexApply(msg, renderer_.getVoiceManager());
        if (applied) {
            // Sync the vm's updated SynthConfig back to cfg_ so get_config reflects it
            renderer_.setSynthConfig(renderer_.getVoiceManager().getSynthConfig());
        }
        json r = ok();
        r["applied"] = applied;
        return r.dump();
    }

    // ── quit ──────────────────────────────────────────────────────────────────
    if (cmd == "quit") {
        // Caller checks for this response and exits the loop
        return json{{"status","ok"},{"quit",true}}.dump();
    }

    return err("unknown command: " + cmd).dump();
}

// ── Socket helpers ────────────────────────────────────────────────────────────

bool RenderServer::recvLine(sock_t fd, std::string& out) {
    out.clear();
    char c;
    while (true) {
#ifdef _WIN32
        int n = ::recv(fd, &c, 1, 0);
#else
        ssize_t n = ::recv(fd, &c, 1, 0);
#endif
        if (n <= 0) return false;   // disconnect or error
        if (c == '\n') return true;
        if (c != '\r') out += c;
    }
}

bool RenderServer::sendAll(sock_t fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
#ifdef _WIN32
        int n = ::send(fd, data.c_str() + sent, (int)(data.size() - sent), 0);
        if (n == SOCKET_ERROR) return false;
#else
        ssize_t n = ::send(fd, data.c_str() + sent, data.size() - sent, 0);
        if (n < 0) return false;
#endif
        sent += (size_t)n;
    }
    return true;
}

// ── runTCP ────────────────────────────────────────────────────────────────────

int RenderServer::runTCP(int port) {
#ifdef _WIN32
    WSADATA wsa{};
    if (::WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        logger_.log("RenderServer", LogSeverity::Error, "WSAStartup failed");
        return 1;
    }
#endif

    sock_t srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (!sock_valid(srv)) {
        logger_.log("RenderServer", LogSeverity::Error, "socket() failed");
        return 1;
    }

    // Allow immediate reuse after restart
    int opt = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // 127.0.0.1 only

    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        logger_.log("RenderServer", LogSeverity::Error,
                    "bind() failed on port " + std::to_string(port));
        sock_close(srv);
        return 1;
    }

    ::listen(srv, 1);
    logger_.log("RenderServer", LogSeverity::Info,
                "Listening on 127.0.0.1:" + std::to_string(port));

    bool server_quit = false;
    while (!server_quit) {
        sock_t cli = ::accept(srv, nullptr, nullptr);
        if (!sock_valid(cli)) break;

        logger_.log("RenderServer", LogSeverity::Info, "Client connected");

        // Greet the client
        sendAll(cli, json{{"status", "ready"}}.dump() + "\n");

        // Command loop for this connection
        std::string line;
        while (recvLine(cli, line)) {
            if (line.empty()) continue;
            const std::string resp = handleLine(line) + "\n";
            if (!sendAll(cli, resp)) break;

            try {
                if (json::parse(resp).value("quit", false))
                    { server_quit = true; break; }
            } catch (...) {}
        }

        sock_close(cli);
        logger_.log("RenderServer", LogSeverity::Info, "Client disconnected");
    }

    sock_close(srv);
#ifdef _WIN32
    ::WSACleanup();
#endif
    return 0;
}
