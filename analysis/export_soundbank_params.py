"""
analysis/export_soundbank_params.py
────────────────────────────────────
Export PianoCore JSON params from the soundbank (real extracted params)
instead of the NN model.  Produces the same format as export_piano_params.py
so the C++ PianoCore can load it directly.

Usage:
    python analysis/export_soundbank_params.py \\
        --soundbank soundbanks/params-ks-grand-ft.json \\
        --out       analysis/params-piano-soundbank.json \\
        [--sr 44100] [--duration 3.0] [--target-rms 0.06] [--rng-seed 0]
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PIANO_MAX_PARTIALS = 60
SR_DEFAULT         = 44100
DURATION_DEFAULT   = 3.0
TARGET_RMS         = 0.06
VEL_GAMMA          = 0.7
RNG_SEED           = 0


# ── Render a note (same algorithm as piano_core.cpp) ─────────────────────────

def render_note(partials_info: list, phi_diff: float,
                rms_gain: float, sr: int, duration: float) -> np.ndarray:
    """
    Render one note using the C++ piano_core algorithm (numpy replica).
    partials_info: list of dicts with keys f_hz, A0, tau1, tau2, a1, beat_hz, phi
    Returns float32 mono audio (used only to compute rms_gain, no noise included).
    """
    N      = int(duration * sr)
    inv_sr = np.float32(1.0) / np.float32(sr)
    t_idx  = np.arange(N, dtype=np.float32)
    t_f    = t_idx * inv_sr
    tpi2   = np.float32(2.0 * math.pi) * t_f

    audio = np.zeros(N, dtype=np.float32)
    for p in partials_info:
        tau1    = np.float32(p["tau1"])
        tau2    = np.float32(p["tau2"])
        a1      = np.float32(p["a1"])
        A0      = np.float32(p["A0"])
        f_hz    = np.float32(p["f_hz"])
        beat_hz = np.float32(p["beat_hz"])
        phi     = np.float32(p["phi"])

        df       = np.exp(np.float32(-1.0) / np.maximum(tau1 * np.float32(sr), np.float32(1.0)))
        ds       = np.exp(np.float32(-1.0) / np.maximum(tau2 * np.float32(sr), np.float32(1.0)))
        env_fast = np.power(df, t_idx)
        env_slow = np.power(ds, t_idx)
        env      = a1 * env_fast + (np.float32(1.0) - a1) * env_slow

        phase_c = tpi2 * f_hz + phi
        phase_b = tpi2 * (beat_hz * np.float32(0.5))
        s1      = np.cos(phase_c + phase_b)
        s2      = np.cos(phase_c - phase_b + np.float32(phi_diff))

        audio += A0 * rms_gain * env * (s1 + s2) * np.float32(0.5)

    return audio


# ── Main export ───────────────────────────────────────────────────────────────

def export(soundbank_path: str, out_path: str,
           sr: int, duration: float, target_rms: float, rng_seed: int) -> None:

    with open(soundbank_path) as f:
        sb = json.load(f)
    samples = sb["samples"]

    out = {
        "format":     "piano_core_v1",
        "source":     "soundbank:" + str(Path(soundbank_path).name),
        "sr":         sr,
        "target_rms": target_rms,
        "vel_gamma":  VEL_GAMMA,
        "k_max":      PIANO_MAX_PARTIALS,
        "rng_seed":   rng_seed,
        "duration_s": duration,
        "n_notes":    0,
        "notes":      {},
    }

    n_done = 0
    for midi in range(21, 109):       # 88 piano keys
        for vel_idx in range(8):
            key = f"m{midi:03d}_vel{vel_idx}"
            if key not in samples:
                continue
            s = samples[key]

            # ── Precompute phis ───────────────────────────────────────────────
            seed = rng_seed + midi * 256 + vel_idx
            rng  = np.random.default_rng(seed)

            partials_raw = s["partials"]
            K = min(len(partials_raw), PIANO_MAX_PARTIALS)

            # phi_diff: random per note (matches physics_synth.py where each
            # string gets independent random phase)
            phi_diff = float(rng.uniform(0, 2 * math.pi))

            # phis for K partials
            phis = rng.uniform(0, 2 * math.pi, K).astype(np.float32)

            # ── Build partial list ────────────────────────────────────────────
            partials_out = []
            for ki in range(K):
                p   = partials_raw[ki]
                phi = float(phis[ki])

                # Mono partial (single string): suppress beating
                beat = float(p.get("beat_hz", 0.0) or 0.0)
                if p.get("mono", False):
                    beat = 0.0

                # Sanitise tau — physical extraction can yield None or very short taus
                raw_tau1 = p.get("tau1")
                tau1 = max(float(raw_tau1) if raw_tau1 is not None else 0.5, 0.005)
                raw_tau2 = p.get("tau2")
                tau2 = float(raw_tau2) if raw_tau2 is not None else tau1
                tau2 = max(tau2, tau1)   # tau2 >= tau1

                # a1: use extracted value; if no tau2/biexp, set a1=1
                raw_a1 = p.get("a1")
                a1 = float(raw_a1) if raw_a1 is not None else 1.0
                if tau2 <= tau1 * 1.001:
                    a1 = 1.0

                partials_out.append({
                    "f_hz":    float(p["f_hz"]),
                    "A0":      float(p["A0"]),
                    "tau1":    tau1,
                    "tau2":    tau2,
                    "a1":      a1,
                    "beat_hz": beat,
                    "phi":     phi,
                })

            # ── Compute rms_gain (render with rms_gain=1 first) ──────────────
            vel_gain  = ((vel_idx + 1) / 8.0) ** VEL_GAMMA
            raw_audio = render_note(partials_out, phi_diff,
                                    rms_gain=1.0, sr=sr, duration=duration)
            raw_rms   = float(np.sqrt(np.mean(raw_audio ** 2) + 1e-12))
            rms_gain  = (target_rms * vel_gain) / raw_rms if raw_rms > 1e-10 else 1.0

            # ── Noise params ──────────────────────────────────────────────────
            noise     = s.get("noise", {})
            attack_tau = float(noise.get("attack_tau_s", 0.05) or 0.05)
            A_noise    = float(noise.get("A_noise", 0.04) or 0.04)

            out["notes"][key] = {
                "midi":       midi,
                "vel":        vel_idx,
                "f0_hz":      float(s.get("f0_fitted_hz") or s.get("f0_nominal_hz", 440.0)),
                "K_valid":    K,
                "phi_diff":   phi_diff,
                "attack_tau": attack_tau,
                "A_noise":    A_noise,
                "rms_gain":   rms_gain,
                "partials":   partials_out,
            }
            n_done += 1
            if n_done % 88 == 0:
                print(f"  {n_done} notes done ...", flush=True)

    out["n_notes"] = n_done
    print(f"\nTotal: {n_done} notes")

    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Written: {out_path}  ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundbank",  default="soundbanks/params-ks-grand-ft.json")
    ap.add_argument("--out",        default="analysis/params-piano-soundbank.json")
    ap.add_argument("--sr",         type=int,   default=SR_DEFAULT)
    ap.add_argument("--duration",   type=float, default=DURATION_DEFAULT)
    ap.add_argument("--target-rms", type=float, default=TARGET_RMS)
    ap.add_argument("--rng-seed",   type=int,   default=RNG_SEED)
    args = ap.parse_args()

    print(f"Source: {args.soundbank}")
    print(f"Output: {args.out}")
    print(f"SR={args.sr}  duration={args.duration}s  target_rms={args.target_rms}")
    export(args.soundbank, args.out, args.sr, args.duration, args.target_rms, args.rng_seed)


if __name__ == "__main__":
    main()
