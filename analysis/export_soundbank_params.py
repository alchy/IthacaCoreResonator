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
import argparse, json, math, sys, warnings
from pathlib import Path

import numpy as np
from scipy.signal import tf2sos

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PIANO_MAX_PARTIALS = 60
PIANO_N_BIQUAD     = 5    # must match piano_core.h PIANO_N_BIQUAD
SR_DEFAULT         = 44100
DURATION_DEFAULT   = 3.0
TARGET_RMS         = 0.06
VEL_GAMMA          = 0.7
RNG_SEED           = 0


# ── Spectral EQ fitting: log-spaced magnitude -> biquad SOS ──────────────────

def _mag_to_min_phase(H_mag_interp):
    """Cepstral minimum-phase reconstruction from one-sided magnitude spectrum.
    H_mag_interp: 1-D array on uniform frequency grid (DC to Nyquist).
    Returns complex spectrum of same length.
    """
    N_h = len(H_mag_interp)
    N   = (N_h - 1) * 2   # corresponding FFT size (even)

    log_H = np.log(np.maximum(H_mag_interp, 1e-8))
    # Build full two-sided log spectrum (symmetric)
    log_full = np.concatenate([log_H[:-1], log_H[-1::-1][:-1]])  # length N

    cep = np.real(np.fft.ifft(log_full))   # real cepstrum
    # Keep only causal (positive-time) quefrencies
    win        = np.zeros(N)
    win[0]     = 1.0
    win[1:N//2]= 2.0
    if N % 2 == 0:
        win[N//2] = 1.0
    log_min = np.fft.fft(cep * win)        # complex cepstrum → complex log spectrum
    H_complex = np.exp(log_min)             # complex min-phase spectrum (full, two-sided)
    return H_complex[:N_h]                  # return one-sided


def _invfreqz(H_complex, w, order):
    """Least-squares IIR design from complex frequency samples.
    Solves: B(e^jw) ≈ A(e^jw) * H  (equation error method).
    Returns (b, a) coefficient arrays with a[0] = 1.
    """
    nb = na = order
    M  = len(w)
    cols = []
    for k in range(nb + 1):
        cols.append(np.exp(-1j * k * w))
    for k in range(1, na + 1):
        cols.append(-H_complex * np.exp(-1j * k * w))
    A_mat = np.column_stack(cols)           # (M, nb+1+na), complex
    rhs   = H_complex
    # Stack real + imaginary for real solution
    A_r   = np.vstack([A_mat.real, A_mat.imag])
    rhs_r = np.concatenate([rhs.real, rhs.imag])
    x, *_ = np.linalg.lstsq(A_r, rhs_r, rcond=None)
    b = x[:nb + 1]
    a = np.concatenate([[1.0], x[nb + 1:]])
    return b, a


def _stabilize(a):
    """Reflect poles outside the unit circle inside it (|z| -> 0.999/|z|)."""
    poles = np.roots(a)
    mask  = np.abs(poles) >= 0.999
    poles[mask] = 0.999 * poles[mask] / np.abs(poles[mask])
    return np.poly(poles).real


def eq_to_biquads(freqs_hz, gains_db, sr, n_sections=PIANO_N_BIQUAD):
    """Fit spectral-EQ magnitude curve to a cascade of biquad IIR sections.

    freqs_hz / gains_db: 64 log-spaced points from the soundbank.
    sr:                  sample rate (Hz).
    n_sections:          number of biquad sections (must equal PIANO_N_BIQUAD).

    Returns list of dicts  {"b": [b0,b1,b2], "a": [a1,a2]}  (a0=1 always).
    Each dict drives one Direct-Form-II biquad in piano_core.cpp.
    """
    N_FFT = 2048
    # 1. Interpolate 64 log-spaced points to uniform frequency grid
    f_uni  = np.linspace(0, sr / 2, N_FFT // 2 + 1)
    gains_interp = np.interp(f_uni,
                             np.array(freqs_hz, dtype=np.float64),
                             np.array(gains_db,  dtype=np.float64),
                             left=gains_db[0], right=gains_db[-1])
    H_mag = 10.0 ** (gains_interp / 20.0)

    # 2. Compute minimum-phase complex spectrum
    H_min = _mag_to_min_phase(H_mag)

    # 3. Sub-sample to 256 log-spaced frequencies for fitting
    #    (avoid DC and very near Nyquist; weight perceptually important range)
    f_fit = np.geomspace(max(f_uni[1], 30.0), min(sr * 0.47, 18000.0), 256)
    w_fit = 2.0 * np.pi * f_fit / sr
    # Interpolate complex spectrum to fit frequencies
    H_fit = np.interp(f_fit, f_uni, H_min.real) + \
            1j * np.interp(f_fit, f_uni, H_min.imag)

    # 4. Fit IIR (order = 2 * n_sections)
    order = n_sections * 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b, a = _invfreqz(H_fit, w_fit, order)

    # 5. Stabilize poles
    a_s = _stabilize(a)

    # 6. Convert to second-order sections
    try:
        sos = tf2sos(b, a_s)    # shape (n_sections, 6): [b0,b1,b2, 1,a1,a2]
        # Pad / trim to exactly n_sections
        if len(sos) < n_sections:
            pad = np.tile([1., 0., 0., 1., 0., 0.], (n_sections - len(sos), 1))
            sos = np.vstack([sos, pad])
        else:
            sos = sos[:n_sections]
    except Exception:
        sos = np.tile([1., 0., 0., 1., 0., 0.], (n_sections, 1))

    result = []
    for row in sos:
        b0, b1, b2, a0, a1, a2 = row
        a0 = a0 if abs(a0) > 1e-10 else 1.0
        result.append({"b": [float(b0/a0), float(b1/a0), float(b2/a0)],
                        "a": [float(a1/a0), float(a2/a0)]})
    return result


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

            # ── Noise params ──────────────────────────────────────────────────
            noise      = s.get("noise", {})
            attack_tau_raw = float(noise.get("attack_tau_s", 0.05) or 0.05)
            A_noise        = float(noise.get("A_noise", 0.04) or 0.04)

            # Cap noise decay at tau1 of k=1 partial (matches physics_synth.py):
            #   "noise never outlasts the string"
            tau1_k1    = partials_out[0]["tau1"] if partials_out else 3.0
            attack_tau = min(attack_tau_raw, tau1_k1)

            # ── Compute rms_gain — include noise power in normalization ───────
            # physics_synth.py normalises (partials + noise) together; if we only
            # normalise partials, high notes (weak partials, large rms_gain) end
            # up with catastrophically loud noise.  Use analytical noise RMS:
            #   E[noise²] = A_noise² * ∫₀ᵀ exp(-2t/τ) dt = A_noise² * τ/2*(1-e^{-2T/τ})
            vel_gain    = ((vel_idx + 1) / 8.0) ** VEL_GAMMA
            raw_audio   = render_note(partials_out, phi_diff,
                                      rms_gain=1.0, sr=sr, duration=duration)
            partial_rms = float(np.sqrt(np.mean(raw_audio ** 2) + 1e-12))
            tau_n       = max(attack_tau, 1e-4)
            dur         = duration
            noise_rms   = A_noise * float(np.sqrt(
                tau_n / 2.0 * (1.0 - np.exp(-2.0 * dur / tau_n)) + 1e-12))
            total_rms   = float(np.sqrt(partial_rms ** 2 + noise_rms ** 2 + 1e-12))
            rms_gain    = (target_rms * vel_gain) / total_rms if total_rms > 1e-10 else 1.0

            # ── Spectral EQ biquad cascade ─────────────────────────────────────
            eq_biquads = []
            eq_data    = s.get("spectral_eq")
            if eq_data and eq_data.get("freqs_hz") and eq_data.get("gains_db"):
                try:
                    eq_biquads = eq_to_biquads(eq_data["freqs_hz"],
                                               eq_data["gains_db"], sr)
                except Exception as e:
                    eq_biquads = []  # skip on failure, EQ disabled for this note

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
                "eq_biquads": eq_biquads,
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
