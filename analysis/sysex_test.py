"""
sysex_test.py — SysEx send/test helper for IthacaCoreResonator.

Usage:
    python analysis/sysex_test.py [--port N] [--list]

Requires:
    pip install python-rtmidi

IthacaCoreResonator must be running and its MIDI input set to the same port.
"""

import struct, sys, time, argparse
from pathlib import Path
import rtmidi

# ── Param IDs (mirrors sysex.h) ───────────────────────────────────────────────

PARAMS = {
    # 0x3 STEREO
    "pan_spread":             (0x3000, 0.00, 1.57, 0.55),
    "stereo_decorr":          (0x3001, 0.00, 1.00, 1.00),
    "stereo_boost":           (0x3002, 0.00, 4.00, 1.00),
    "pan_tilt":               (0x3003, 0.00, 1.00, 0.20),
    "stereo_decorr_midi_lo":  (0x3004, 0.00, 127., 40.0),
    "stereo_decorr_midi_hi":  (0x3005, 0.00, 127., 100.),
    "stereo_decorr_max":      (0x3006, 0.00, 1.00, 0.45),
    # 0x4 TIMBRE
    "beat_scale":             (0x4000, 0.00, 5.00, 1.00),
    "harmonic_brightness":    (0x4001,-2.00, 4.00, 0.00),
    "eq_strength":            (0x4002, 0.00, 1.00, 1.00),
    "eq_freq_min":            (0x4003, 20.0, 2000., 400.),
    "pitch_glide":            (0x4004, 0.00, 0.05, 0.00),
    "pitch_glide_tau_ms":     (0x4005, 1.00, 500., 80.0),
    "pitch_glide_vel_thresh": (0x4006, 0,   127,  100),
    # 0x5 LEVEL/ENV
    "target_rms":             (0x5000, 0.001, 0.50, 0.060),
    "vel_gamma":              (0x5001, 0.10,  3.00, 0.700),
    "noise_level":            (0x5002, 0.00,  4.00, 1.000),
    "onset_ms":               (0x5003, 0.00,  50.0, 3.000),
    "longitudinal_precursor": (0x5004, 0.00,  1.00, 0.000),
}

MFR  = 0x7D
SIG  = [0x49, 0x43, 0x52]
SET_PARAM       = 0x01
GET_PARAM       = 0x02
SET_ALL         = 0x04
REQUEST_ALL     = 0x05

# ── Codec ─────────────────────────────────────────────────────────────────────

def encode_float(f: float) -> list[int]:
    raw = struct.unpack('<I', struct.pack('<f', f))[0]
    return [(raw >> (7 * i)) & 0x7F for i in range(4)] + [(raw >> 28) & 0x0F]

def encode_id(pid: int) -> list[int]:
    return [(pid >> 14) & 0x03, (pid >> 7) & 0x7F, pid & 0x7F]

def decode_float(v: list[int]) -> float:
    raw = (v[0] & 0x7F) | ((v[1] & 0x7F) << 7) | ((v[2] & 0x7F) << 14) \
        | ((v[3] & 0x7F) << 21) | ((v[4] & 0x0F) << 28)
    return struct.unpack('<f', struct.pack('<I', raw))[0]

def build_set_param(pid: int, value: float) -> list[int]:
    return [0xF0, MFR] + SIG + [SET_PARAM] + encode_id(pid) + encode_float(value) + [0xF7]

def build_get_param(pid: int) -> list[int]:
    return [0xF0, MFR] + SIG + [GET_PARAM] + encode_id(pid) + [0xF7]

def build_set_all(values: dict) -> list[int]:
    payload = []
    for name, (pid, lo, hi, default) in PARAMS.items():
        v = values.get(name, default)
        payload += encode_id(pid) + encode_float(float(v))
    return [0xF0, MFR] + SIG + [SET_ALL] + payload + [0xF7]

def build_request_all() -> list[int]:
    return [0xF0, MFR] + SIG + [REQUEST_ALL, 0xF7]

# ── Port helpers ──────────────────────────────────────────────────────────────

def list_ports() -> list[str]:
    m = rtmidi.MidiOut()
    return m.get_ports()

def open_port(index: int) -> rtmidi.MidiOut:
    m = rtmidi.MidiOut()
    ports = m.get_ports()
    if not ports:
        raise RuntimeError("No MIDI output ports found")
    if index >= len(ports):
        raise RuntimeError(f"Port {index} out of range (found {len(ports)})")
    m.open_port(index)
    print(f"Opened MIDI out: {ports[index]}")
    return m

# ── Test suite ────────────────────────────────────────────────────────────────

def run_tests(midi: rtmidi.MidiOut):
    delay = 0.05  # 50 ms between messages

    print("\n--- Test 1: SET_PARAM individual ---")
    tests = [
        ("beat_scale",          2.0),
        ("harmonic_brightness", 1.0),
        ("pan_spread",          0.8),
        ("stereo_boost",        2.0),
        ("target_rms",          0.08),
        ("onset_ms",            10.0),
        ("pitch_glide",         0.02),
        ("pitch_glide_tau_ms",  120.0),
    ]
    for name, val in tests:
        pid = PARAMS[name][0]
        msg = build_set_param(pid, val)
        midi.send_message(msg)
        print(f"  SET {name:30s} = {val:8.4f}  ({len(msg)} bytes)")
        time.sleep(delay)

    print("\n--- Test 2: out-of-range clamping ---")
    clamptests = [
        ("pan_spread", 99.0),      # should clamp to 1.57
        ("target_rms",  0.0),      # should clamp to 0.001
        ("beat_scale",  -1.0),     # should clamp to 0.0
    ]
    for name, val in clamptests:
        pid, lo, hi, _ = PARAMS[name]
        clamped = max(lo, min(hi, val))
        msg = build_set_param(pid, val)
        midi.send_message(msg)
        print(f"  SET {name:30s} = {val:8.4f}  (expect clamp -> {clamped:.4f})")
        time.sleep(delay)

    print("\n--- Test 3: SET_ALL (defaults) ---")
    msg = build_set_all({})
    midi.send_message(msg)
    print(f"  SET_ALL sent ({len(msg)} bytes, expected 159)")

    time.sleep(delay)

    print("\n--- Test 4: GET_PARAM (no response read, check ICR log) ---")
    for name in ["beat_scale", "target_rms", "pan_spread"]:
        pid = PARAMS[name][0]
        msg = build_get_param(pid)
        midi.send_message(msg)
        print(f"  GET {name:30s}  pid=0x{pid:04X}")
        time.sleep(delay)

    print("\n--- Test 5: REQUEST_ALL ---")
    msg = build_request_all()
    midi.send_message(msg)
    print(f"  REQUEST_ALL sent ({len(msg)} bytes)")

    print("\n--- Test 6: invalid SysEx (bad signature) ---")
    bad = [0xF0, 0x7D, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xF7]
    midi.send_message(bad)
    print(f"  Sent bad-sig SysEx ({len(bad)} bytes) — expect WRN in ICR log")

    time.sleep(delay)

    print("\n--- Test 7: restore defaults ---")
    msg = build_set_all({})
    midi.send_message(msg)
    print(f"  SET_ALL (defaults) sent")

    print("\nDone. Check IthacaCoreResonator console for [SYSEX] log lines.")

# ── Verify codec round-trip (no MIDI needed) ──────────────────────────────────

def verify_codec():
    print("=== Codec round-trip verification ===")
    import random
    errors = 0
    for name, (pid, lo, hi, default) in PARAMS.items():
        v = default
        enc = encode_float(float(v))
        dec = decode_float(enc)
        diff = abs(dec - v)
        ok = diff < 1e-5
        if not ok:
            print(f"  FAIL {name}: {v} -> {dec} (diff={diff:.2e})")
            errors += 1
    # random floats
    for _ in range(20):
        v = random.uniform(-100, 100)
        enc = encode_float(v)
        dec = decode_float(enc)
        diff = abs(dec - v)
        if diff > 1e-5:
            print(f"  FAIL random {v}: -> {dec} (diff={diff:.2e})")
            errors += 1
    # ID encode/decode
    for pid in [pid for pid, *_ in PARAMS.values()]:
        enc = encode_id(pid)
        dec = (enc[0] << 14) | (enc[1] << 7) | enc[2]
        if dec != pid:
            print(f"  FAIL pid 0x{pid:04X} -> 0x{dec:04X}")
            errors += 1
    if errors == 0:
        print(f"  All {len(PARAMS)} params + 20 random floats OK")
    else:
        print(f"  {errors} errors")
    print()

# ── Routing verification (via IthacaRenderServer) ─────────────────────────────

def verify_routing(server_exe: str, params_json: str):
    """
    Verify that every param ID routes to the correct SynthConfig field.

    Sends SET_PARAM for each of the 19 params via the RenderServer 'sysex'
    command (JSON IPC — no real MIDI needed), then reads back with get_config
    and compares. Encoding is lossless (full 32-bit IEEE 754), so values must
    match exactly (within float epsilon).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.render_client import RenderClient

    # Test values chosen to be within range and distinct from defaults
    TEST_VALUES = {
        "pan_spread":             0.77,
        "stereo_decorr":          0.33,
        "stereo_boost":           2.5,
        "pan_tilt":               0.65,
        "stereo_decorr_midi_lo":  55.0,
        "stereo_decorr_midi_hi":  90.0,
        "stereo_decorr_max":      0.30,
        "beat_scale":             1.75,
        "harmonic_brightness":    0.5,
        "eq_strength":            0.6,
        "eq_freq_min":            600.0,
        "pitch_glide":            0.015,
        "pitch_glide_tau_ms":     200.0,
        "pitch_glide_vel_thresh": 64,     # int param
        "target_rms":             0.04,
        "vel_gamma":              1.2,
        "noise_level":            0.5,
        "onset_ms":               8.0,
        "longitudinal_precursor": 0.3,
    }

    print("=== Routing verification via IthacaRenderServer ===")
    print(f"  server: {server_exe}")
    print(f"  params: {params_json}\n")

    passed = 0
    failed = 0

    with RenderClient(server_exe, params_json) as rc:
        for name, test_val in TEST_VALUES.items():
            pid, lo, hi, _ = PARAMS[name]

            # Send SET_PARAM via sysex command
            msg = build_set_param(pid, float(test_val))
            applied = rc.sysex(msg)
            if not applied:
                print(f"  FAIL  {name:30s}  sysex not applied (unknown pid?)")
                failed += 1
                continue

            # Read back via get_config
            cfg = rc.get_config()
            got = cfg.get(name)
            if got is None:
                print(f"  FAIL  {name:30s}  key missing in get_config response")
                failed += 1
                continue

            # For int param: compare as int
            if name == "pitch_glide_vel_thresh":
                expected = int(test_val)
                ok = int(got) == expected
                desc = f"expected {expected}  got {int(got)}"
            else:
                expected = float(test_val)
                diff = abs(got - expected)
                ok = diff < 1e-5
                desc = f"expected {expected:.6f}  got {got:.6f}  diff={diff:.2e}"

            status = "OK   " if ok else "FAIL "
            print(f"  {status} {name:30s}  {desc}")
            if ok:
                passed += 1
            else:
                failed += 1

    print(f"\n  {passed}/{passed+failed} passed")
    if failed:
        print("  ROUTING ERRORS — check sysexApplyParam() switch-case")
        sys.exit(1)
    else:
        print("  All key->value routes verified")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICR SysEx test helper")
    parser.add_argument("--list",    action="store_true", help="List MIDI output ports and exit")
    parser.add_argument("--port",    type=int, default=-1, help="MIDI output port index")
    parser.add_argument("--verify",  action="store_true", help="Run codec round-trip only (no MIDI)")
    parser.add_argument("--verify-routing", action="store_true",
                        help="Verify key->value routing via IthacaRenderServer (no MIDI needed)")
    parser.add_argument("--server",  default="build/bin/Release/IthacaRenderServer.exe",
                        help="Path to IthacaRenderServer executable")
    parser.add_argument("--params",  default="analysis/params-ks-grand.json",
                        help="Params JSON for IthacaRenderServer")
    args = parser.parse_args()

    verify_codec()

    if args.verify:
        sys.exit(0)

    if args.verify_routing:
        verify_routing(args.server, args.params)
        sys.exit(0)

    ports = list_ports()
    print("Available MIDI output ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p}")
    print()

    if args.list:
        sys.exit(0)

    # Auto-select: prefer loopMIDI / virtual ports over GS Wavetable
    port_idx = args.port
    if port_idx < 0:
        for i, p in enumerate(ports):
            if "GS Wavetable" not in p:
                port_idx = i
                break
        if port_idx < 0:
            port_idx = 0

    midi = open_port(port_idx)
    try:
        run_tests(midi)
    finally:
        midi.close_port()
