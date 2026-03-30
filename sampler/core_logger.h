#pragma once
/*
 * core_logger.h — minimal Logger compatible with IthacaCore API.
 * Replace with the full IthacaCore core_logger.h when copying DSP files.
 *
 * Two output channels:
 *   log()   — diagnostic channel (file_out), fflush after each write.
 *             Use for startup, errors, lifecycle events.
 *   logRT() — realtime channel (rt_out), no fflush.
 *             Use from audio/callback threads where disk I/O must not block.
 *             Typically points to stdout (line-buffered on a terminal).
 *
 * Either channel may be nullptr (silent).  Logger is copyable — store by value.
 */
#include <string>
#include <cstdio>

enum class LogSeverity { Debug = 0, Info, Warning, Error, Critical };

class Logger {
public:
    // file_out  — destination for log()   (e.g. a log file, or stdout)
    // rt_out    — destination for logRT() (e.g. stdout; no flush performed)
    explicit Logger(std::FILE* file_out = nullptr,
                    std::FILE* rt_out   = nullptr)
        : file_out_(file_out), rt_out_(rt_out) {}

    void log(const char* tag, LogSeverity sev, const std::string& msg) const {
        if (!file_out_) return;
        const char* prefix[] = {"DBG","INF","WRN","ERR","CRT"};
        std::fprintf(file_out_, "[%s][%s] %s\n", prefix[(int)sev], tag, msg.c_str());
        std::fflush(file_out_);
    }

    // Realtime variant — writes to rt_out without fflush.
    void logRT(const char* tag, LogSeverity sev, const std::string& msg) const {
        if (!rt_out_) return;
        const char* prefix[] = {"DBG","INF","WRN","ERR","CRT"};
        std::fprintf(rt_out_, "[%s][%s] %s\n", prefix[(int)sev], tag, msg.c_str());
    }

private:
    std::FILE* file_out_ = nullptr;
    std::FILE* rt_out_   = nullptr;
};
