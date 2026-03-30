#pragma once
/*
 * core_logger.h — minimal Logger compatible with IthacaCore API.
 * Replace with the full IthacaCore core_logger.h when copying DSP files.
 */
#include <string>
#include <cstdio>

enum class LogSeverity { Debug = 0, Info, Warning, Error, Critical };

class Logger {
public:
    // output: where log lines go (nullptr = muted)
    //   nullptr  — silent (default for headless/server contexts)
    //   stdout   — interactive use (IthacaCoreResonator main)
    //   stderr   — verbose server mode (--verbose flag)
    explicit Logger(const std::string& /*log_dir*/ = ".",
                    std::FILE* output = nullptr)
        : out_(output) {}

    void log(const char* tag, LogSeverity sev, const std::string& msg) const {
        if (!out_) return;
        const char* prefix[] = {"DBG","INF","WRN","ERR","CRT"};
        std::fprintf(out_, "[%s][%s] %s\n", prefix[(int)sev], tag, msg.c_str());
    }

    // RT-safe ring-buffer variant (stub — flushes immediately here)
    void logRT(const char* tag, LogSeverity sev, const std::string& msg) const {
        log(tag, sev, msg);
    }

private:
    std::FILE* out_ = nullptr;
};
