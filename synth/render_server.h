#pragma once
/*
 * render_server.h
 * ─────────────────
 * TCP JSON render server for Python training-loop IPC.
 *
 * Listens on 127.0.0.1:PORT (default 9876).
 * One JSON object per line in each direction; newline-terminated.
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
 *       → {"status":"ok"}
 *
 *   {"cmd":"get_config"}
 *       → {"status":"ok","params":{...full SynthConfig...}}
 *
 *   {"cmd":"reload","params":"path/to/params.json"}
 *       → {"status":"ok"}
 *
 *   {"cmd":"sysex","bytes":[0xF0,0x7D,...,0xF7]}
 *       → {"status":"ok","applied":true/false}
 *
 *   {"cmd":"quit"}
 *       → {"status":"ok"}  (closes connection; server exits)
 *
 * On accept, server sends {"status":"ready"} as the first line.
 * No stdin, stdout, or stderr is used during operation.
 */

#include "offline_renderer.h"
#include "../sampler/core_logger.h"
#include <string>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
   using sock_t = SOCKET;
   static inline void sock_close(sock_t s) { ::closesocket(s); }
   static inline bool sock_valid(sock_t s) { return s != INVALID_SOCKET; }
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <unistd.h>
   using sock_t = int;
   static inline void sock_close(sock_t s) { ::close(s); }
   static inline bool sock_valid(sock_t s) { return s >= 0; }
   static constexpr sock_t INVALID_SOCKET_VAL = -1;
#endif

class RenderServer {
public:
    // Initialize renderer with params JSON path.
    bool initialize(const std::string& params_json_path, Logger& logger);

    // Bind to 127.0.0.1:port and serve connections until quit command.
    // Returns 0 on clean exit, non-zero on fatal error.
    int runTCP(int port = 9876);

private:
    std::string handleLine(const std::string& line);

    // Read one newline-terminated line from socket. Returns false on disconnect.
    bool recvLine(sock_t fd, std::string& out);

    // Send string over socket. Returns false on error.
    bool sendAll(sock_t fd, const std::string& data);

    OfflineRenderer renderer_;
    Logger          logger_;               // silent by default; set in initialize()
    std::string     params_path_;
};
