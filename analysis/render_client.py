"""
analysis/render_client.py
─────────────────────────
Python client for IthacaRenderServer (stdin/stdout JSON protocol).

The server executable is a subprocess; communication is one JSON line per
message.  On startup the server emits {"status":"ready"} before accepting
commands.

Usage
-----
    from analysis.render_client import RenderClient

    with RenderClient("build/bin/IthacaRenderServer",
                      "analysis/params-ks-grand.json") as rc:
        rc.set_config(beat_scale=1.5, eq_strength=0.8)
        frames = rc.render(midi=60, vel=80, duration=3.0,
                           output="exports/m060_vel80.wav")
        print(f"rendered {frames} frames")

        cfg = rc.get_config()
        rc.reload(params="analysis/params-ks-grand.json")

Protocol
--------
All messages are single-line JSON objects terminated with \\n.
stdout is the exclusive JSON channel; stderr of the subprocess is closed.
Server diagnostics go to a log file (default: analysis/runtime-logs/render-server.log).
"""

import json
import subprocess
import threading
import os
from pathlib import Path
from typing import Optional


class RenderClientError(RuntimeError):
    pass


class RenderClient:
    """Subprocess wrapper for IthacaRenderServer.

    Can be used as a context manager (``with RenderClient(...) as rc:``)
    or manually controlled via start() / stop().
    """

    def __init__(self,
                 server_exe: str = "build/bin/IthacaRenderServer.exe",
                 params_json: str = "analysis/params-ks-grand.json",
                 timeout: float = 30.0,
                 log_path: str = "analysis/runtime-logs/render-server.log"):
        """
        Parameters
        ----------
        server_exe   Path to the IthacaRenderServer binary.
        params_json  Params JSON file passed as the first argv to the server.
        timeout      Default per-command timeout in seconds.
        log_path     Server writes diagnostics here (never to stderr).
                     Pass None or "" to disable server logging (--no-log).
        """
        self._exe        = str(server_exe)
        self._params     = str(params_json)
        self._timeout    = timeout
        self._log_path   = log_path or None
        self._proc: Optional[subprocess.Popen] = None
        self._lock       = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the server subprocess and wait for {"status":"ready"}."""
        if self._proc is not None and self._proc.poll() is None:
            return  # already running

        exe_path    = Path(self._exe).resolve()
        params_path = Path(self._params).resolve()

        if not exe_path.exists():
            raise RenderClientError(
                f"Server executable not found: {exe_path}\n"
                "Build with: cmake --build build --target IthacaRenderServer"
            )

        cmd = [str(exe_path), str(params_path)]
        if self._log_path:
            cmd += ["--log", str(Path(self._log_path).resolve())]
        else:
            cmd += ["--no-log"]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,   # server never writes to stderr
            text=True,
            bufsize=1,                   # line-buffered
            encoding="utf-8",
        )

        # Wait for the ready signal
        ready_line = self._proc.stdout.readline()
        try:
            msg = json.loads(ready_line)
        except json.JSONDecodeError as e:
            self._proc.kill()
            raise RenderClientError(
                f"Server sent non-JSON on startup: {ready_line!r}"
            ) from e

        if msg.get("status") != "ready":
            self._proc.kill()
            raise RenderClientError(f"Unexpected startup message: {msg}")

    def stop(self) -> None:
        """Send quit command and terminate the subprocess."""
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._send_recv({"cmd": "quit"})
        except Exception:
            pass
        finally:
            self._proc.stdin.close()
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
        """Render one note to a WAV file.

        Parameters
        ----------
        midi      MIDI note number (21–108).
        vel       MIDI velocity (1–127).
        output    Output WAV path (parent directories are created automatically).
        duration  Duration in seconds; 0 = auto-detect silence tail.
        sr        Sample rate in Hz.

        Returns
        -------
        Number of frames written.
        """
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
        """Set one or more SynthConfig fields.

        Keyword arguments correspond 1:1 to SynthConfig field names, e.g.:
            rc.set_config(beat_scale=1.5, eq_strength=0.8)
        """
        self._send_recv({"cmd": "set_config", "params": kwargs})

    def get_config(self) -> dict:
        """Return the current SynthConfig as a dict."""
        return self._send_recv({"cmd": "get_config"})["params"]

    def sysex(self, bytes_: list) -> bool:
        """Send raw SysEx bytes; returns True if the server applied them."""
        resp = self._send_recv({"cmd": "sysex", "bytes": [int(b) for b in bytes_]})
        return resp.get("applied", False)

    def reload(self, params: Optional[str] = None) -> None:
        """Reload params JSON at runtime (e.g. after re-running extract_params).

        Parameters
        ----------
        params  Path to the new params JSON; None = use the original path.
        """
        req: dict = {"cmd": "reload"}
        if params is not None:
            req["params"] = str(params)
        self._send_recv(req)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send_recv(self, msg: dict) -> dict:
        """Thread-safe send+receive of a single JSON message."""
        if self._proc is None or self._proc.poll() is not None:
            raise RenderClientError("Server is not running. Call start() first.")

        with self._lock:
            line = json.dumps(msg) + "\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

            resp_line = self._proc.stdout.readline()
            if not resp_line:
                raise RenderClientError("Server closed stdout unexpectedly.")

            try:
                resp = json.loads(resp_line)
            except json.JSONDecodeError as e:
                raise RenderClientError(
                    f"Non-JSON response: {resp_line!r}"
                ) from e

            if resp.get("status") == "error":
                raise RenderClientError(f"Server error: {resp.get('msg','?')}")

            return resp


# ── Convenience: batch render ─────────────────────────────────────────────────

def batch_render(server_exe: str,
                 params_json: str,
                 jobs: list,
                 sr: int = 44100,
                 duration: float = 0.0) -> list:
    """Render a list of (midi, vel, output_path) tuples.

    Parameters
    ----------
    server_exe   Path to IthacaRenderServer binary.
    params_json  Params JSON path.
    jobs         List of (midi, vel, output_path) tuples.
    sr           Sample rate for all notes.
    duration     Duration per note (0 = auto).

    Returns
    -------
    List of frame counts (same order as jobs).
    """
    results = []
    with RenderClient(server_exe, params_json) as rc:
        for midi, vel, output in jobs:
            n = rc.render(midi=midi, vel=vel, output=output,
                          duration=duration, sr=sr)
            results.append(n)
    return results
