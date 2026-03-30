"""
analysis/render_client.py
─────────────────────────
Python client for IthacaRenderServer (TCP JSON protocol).

The server listens on 127.0.0.1:PORT (default 9876).
All messages are single-line JSON objects terminated with \\n.
On each accepted connection the server sends {"status":"ready"} first.

No stdin, stdout, or stderr pipes are used — communication is exclusively
via the TCP socket.

Usage
-----
    from analysis.render_client import RenderClient

    # Auto-start server subprocess + connect:
    with RenderClient("build/bin/IthacaRenderServer",
                      "analysis/params-ks-grand.json") as rc:
        rc.set_config(beat_scale=1.5, eq_strength=0.8)
        # vel = velocity band 0-7 (matching training data convention)
        frames = rc.render(midi=60, vel=3, duration=3.0,
                           output="exports/m060_vel3.wav")

    # Connect to already-running server (no subprocess):
    with RenderClient(server_exe=None, port=9876) as rc:
        cfg = rc.get_config()
"""

import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


class RenderClientError(RuntimeError):
    pass


class RenderClient:
    """TCP client for IthacaRenderServer.

    Can be used as a context manager (``with RenderClient(...) as rc:``)
    or manually controlled via start() / stop().

    If ``server_exe`` is provided the server subprocess is started
    automatically and terminated on stop().  If ``server_exe`` is None
    the client connects to an already-running server.
    """

    DEFAULT_PORT    = 9876
    CONNECT_RETRIES = 50     # × 0.1 s = 5 s max startup wait
    CONNECT_DELAY   = 0.1    # seconds between connection attempts

    def __init__(self,
                 server_exe: Optional[str] = "build/bin/IthacaRenderServer.exe",
                 params_json: str          = "analysis/params-ks-grand.json",
                 host: str                 = "127.0.0.1",
                 port: int                 = DEFAULT_PORT,
                 timeout: float            = 30.0,
                 log_path: str             = "analysis/runtime-logs/render-server.log"):
        """
        Parameters
        ----------
        server_exe   Path to the IthacaRenderServer binary, or None to skip
                     auto-start and connect to an already-running server.
        params_json  Params JSON passed to the server on startup.
        host         Server host (default: 127.0.0.1).
        port         TCP port (default: 9876, must match --port server arg).
        timeout      Per-command socket timeout in seconds.
        log_path     Server log file path forwarded via --log.
                     Pass None or "" to disable server logging.
        """
        self._exe       = str(server_exe) if server_exe else None
        self._params    = str(params_json)
        self._host      = host
        self._port      = port
        self._timeout   = timeout
        self._log_path  = log_path or None
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket]    = None
        self._file                             = None   # socket.makefile() handle
        self._lock                             = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the server subprocess (if exe provided) and connect via TCP."""
        if self._sock is not None:
            return  # already connected

        # Optionally launch server subprocess (no pipes needed)
        if self._exe is not None:
            exe_path    = Path(self._exe).resolve()
            params_path = Path(self._params).resolve()

            if not exe_path.exists():
                raise RenderClientError(
                    f"Server executable not found: {exe_path}\n"
                    "Build with: cmake --build build --target IthacaRenderServer"
                )

            cmd = [str(exe_path), str(params_path), "--port", str(self._port)]
            if self._log_path:
                cmd += ["--log", str(Path(self._log_path).resolve())]
            else:
                cmd += ["--no-log"]

            # No stdin/stdout/stderr pipes — server is fully autonomous
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Poll until the server accepts a connection
        sock = None
        for attempt in range(self.CONNECT_RETRIES):
            try:
                sock = socket.create_connection((self._host, self._port),
                                                timeout=self._timeout)
                break
            except (ConnectionRefusedError, OSError):
                if self._proc is not None and self._proc.poll() is not None:
                    raise RenderClientError(
                        f"Server process exited early (code {self._proc.returncode}). "
                        f"Check log: {self._log_path}"
                    )
                time.sleep(self.CONNECT_DELAY)
        else:
            if self._proc:
                self._proc.kill()
            raise RenderClientError(
                f"Could not connect to server at {self._host}:{self._port} "
                f"after {self.CONNECT_RETRIES * self.CONNECT_DELAY:.1f}s"
            )

        sock.settimeout(self._timeout)
        self._sock = sock
        self._file = sock.makefile("r", encoding="utf-8")

        # Read and verify the ready greeting
        ready_line = self._file.readline()
        try:
            msg = json.loads(ready_line)
        except json.JSONDecodeError as e:
            self._sock.close()
            raise RenderClientError(
                f"Server sent non-JSON greeting: {ready_line!r}"
            ) from e

        if msg.get("status") != "ready":
            self._sock.close()
            raise RenderClientError(f"Unexpected greeting: {msg}")

    def stop(self) -> None:
        """Send quit, close socket, and terminate subprocess (if any)."""
        if self._sock is not None:
            try:
                self._send_recv({"cmd": "quit"})
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._file = None

        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Commands ──────────────────────────────────────────────────────────────

    def ping(self) -> None:
        """Check that the server is alive."""
        self._send_recv({"cmd": "ping"})

    def render(self,
               midi: int,
               vel: int,
               output: str,
               duration: float = 0.0,
               sr: int = 44100) -> int:
        """Render one note to a WAV file. Returns number of frames written."""
        resp = self._send_recv({
            "cmd":      "render",
            "midi":     int(midi),
            "vel":      int(vel),
            "sr":       int(sr),
            "duration": float(duration),
            "output":   str(output),
        })
        return resp["frames"]

    def set_config(self, **kwargs) -> None:
        """Set SynthConfig fields by name, e.g. set_config(beat_scale=1.5)."""
        self._send_recv({"cmd": "set_config", "params": kwargs})

    def get_config(self) -> dict:
        """Return the current SynthConfig as a dict."""
        return self._send_recv({"cmd": "get_config"})["params"]

    def sysex(self, bytes_: list) -> bool:
        """Send raw SysEx bytes; returns True if applied."""
        resp = self._send_recv({"cmd": "sysex", "bytes": [int(b) for b in bytes_]})
        return resp.get("applied", False)

    def reload(self, params: Optional[str] = None) -> None:
        """Reload params JSON at runtime."""
        req: dict = {"cmd": "reload"}
        if params is not None:
            req["params"] = str(params)
        self._send_recv(req)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send_recv(self, msg: dict) -> dict:
        """Thread-safe send + receive of one JSON message over TCP."""
        if self._sock is None:
            raise RenderClientError("Not connected. Call start() first.")

        with self._lock:
            line = json.dumps(msg) + "\n"
            try:
                self._sock.sendall(line.encode("utf-8"))
                resp_line = self._file.readline()
            except OSError as e:
                raise RenderClientError(f"Socket error: {e}") from e

            if not resp_line:
                raise RenderClientError("Server closed connection unexpectedly.")

            try:
                resp = json.loads(resp_line)
            except json.JSONDecodeError as e:
                raise RenderClientError(f"Non-JSON response: {resp_line!r}") from e

            if resp.get("status") == "error":
                raise RenderClientError(f"Server error: {resp.get('msg', '?')}")

            return resp


# ── Convenience: batch render ─────────────────────────────────────────────────

def batch_render(server_exe: str,
                 params_json: str,
                 jobs: list,
                 port: int = RenderClient.DEFAULT_PORT,
                 sr: int = 44100,
                 duration: float = 0.0) -> list:
    """Render a list of (midi, vel, output_path) tuples.

    Returns list of frame counts in the same order as jobs.
    """
    results = []
    with RenderClient(server_exe, params_json, port=port) as rc:
        for midi, vel, output in jobs:
            n = rc.render(midi=midi, vel=vel, output=output,
                          duration=duration, sr=sr)
            results.append(n)
    return results
