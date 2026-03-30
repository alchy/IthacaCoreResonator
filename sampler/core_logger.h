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
    // use_stderr: write log output to stderr instead of stdout.
    // Set true in headless/server contexts so protocol stdout stays clean.
    explicit Logger(const std::string& /*log_dir*/ = ".",
                    bool use_stderr = false)
        : use_stderr_(use_stderr) {}

    void log(const char* tag, LogSeverity sev, const std::string& msg) const {
        const char* prefix[] = {"DBG","INF","WRN","ERR","CRT"};
        std::FILE* out = use_stderr_ ? stderr : stdout;
        std::fprintf(out, "[%s][%s] %s\n", prefix[(int)sev], tag, msg.c_str());
    }

    // RT-safe ring-buffer variant (stub — flushes immediately here)
    void logRT(const char* tag, LogSeverity sev, const std::string& msg) const {
        log(tag, sev, msg);
    }

private:
    bool use_stderr_ = false;
};
