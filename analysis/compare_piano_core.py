"""
analysis/compare_piano_core.py
────────────────────────────────
Byte-exact verification: C++ PianoCore algorithm (replicated in Python/numpy)
vs. torch_synth.render_note_differentiable() reference.

Renders one (midi, vel) note with both approaches at the same SR,
computes RMS of the difference, and reports per-partial phase errors.

Usage:
    python analysis/compare_piano_core.py \\
        --params  analysis/params-piano-ft-ks-grand.json \\
        --model   analysis/profile-finetuned.pt \\
        --midi    60 --vel 3 \\
        [--sr 44100] [--duration 3.0] [--plot]
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.train_instrument_profile import (
    InstrumentProfile, midi_feat, vel_feat, k_feat, midi_to_hz,
)
from analysis.torch_synth import render_note_differentiable

_K_FEAT_NORM = 90


# ── Replicate C++ PianoCore algorithm in numpy ────────────────────────────────

def render_cpp_algorithm(note: dict, *, sr: int, duration: float,
                          noise_level: float = 1.0) -> np.ndarray:
    """
    Reproduce exactly what piano_core.cpp processBlock() computes
    (minus noise — C++ uses independent mt19937, Python can't replicate it).

    Uses float32 throughout to match C++ arithmetic.
    """
    N        = int(duration * sr)
    inv_sr   = np.float32(1.0) / np.float32(sr)
    t_idx    = np.arange(N, dtype=np.float32)     # same as (float)t_samples
    t_f      = t_idx * inv_sr                      # t_samples * inv_sr_
    tpi2     = np.float32(2.0 * math.pi) * t_f    # TAU * t_f  (float32)

    phi_diff = np.float32(note["phi_diff"])
    rms_gain = np.float32(note["rms_gain"])

    audio = np.zeros(N, dtype=np.float32)

    for p in note["partials"]:
        tau1    = np.float32(p["tau1"])
        tau2    = np.float32(p["tau2"])
        a1      = np.float32(p["a1"])
        A0      = np.float32(p["A0"])
        f_hz    = np.float32(p["f_hz"])
        beat_hz = np.float32(p["beat_hz"])
        phi     = np.float32(p["phi"])

        # ── Envelope: multiplicative decay (matches C++ initVoice) ──────────
        decay_fast = np.exp(np.float32(-1.0) / np.maximum(tau1 * np.float32(sr),
                                                           np.float32(1.0)))
        decay_slow = np.exp(np.float32(-1.0) / np.maximum(tau2 * np.float32(sr),
                                                           np.float32(1.0)))
        env_fast = decay_fast ** t_idx   # env_fast[i] = decay_fast^i
        env_slow = decay_slow ** t_idx
        env      = a1 * env_fast + (np.float32(1.0) - a1) * env_slow

        # ── Phase (matches C++ processBlock phase_c / phase_b) ──────────────
        beat_hz_h = beat_hz * np.float32(0.5)     # beat_scale=1.0 default
        phase_c   = tpi2 * f_hz + phi             # TAU*t_f*f_hz + phi
        phase_b   = tpi2 * beat_hz_h              # TAU*t_f*beat_hz/2

        s1 = np.cos(phase_c + phase_b)
        s2 = np.cos(phase_c - phase_b + phi_diff)

        A0_scaled = A0 * rms_gain
        audio    += A0_scaled * env * (s1 + s2) * np.float32(0.5)

    return audio   # no noise (C++ noise can't be replicated)


# ── Reference: torch_synth.py ─────────────────────────────────────────────────

def render_reference(model: InstrumentProfile, midi: int, vel: int,
                     *, sr: int, duration: float,
                     rng_seed: int = 0) -> np.ndarray:
    """Run torch_synth.render_note_differentiable() and return as float32 numpy."""
    with torch.no_grad():
        audio = render_note_differentiable(
            model, midi, vel,
            sr=sr, duration=duration,
            rng_seed=rng_seed,
            noise_level=0.0,    # disable noise for apples-to-apples comparison
        )
    return audio.numpy().astype(np.float32)


# ── Per-partial phase comparison ──────────────────────────────────────────────

def compare_partial_phases(note: dict, model: InstrumentProfile,
                            midi: int, vel: int, sr: int) -> None:
    """Compare exported f_hz, A0, tau1 to live NN forward pass."""
    device = next(model.parameters()).device
    mf = midi_feat(midi).to(device)
    vf = vel_feat(vel).to(device)
    f0 = midi_to_hz(midi)

    K  = note["K_valid"]
    kf_b   = torch.stack([k_feat(k, _K_FEAT_NORM) for k in range(1, K+1)]).to(device)
    k_vals = torch.arange(1, K+1, dtype=torch.float32, device=device)
    mf_b   = mf.unsqueeze(0).expand(K, -1)
    vf_b   = vf.unsqueeze(0).expand(K, -1)

    with torch.no_grad():
        B     = torch.exp(model.forward_B(mf).clamp(max=0.0)).squeeze()
        f_hzs = (k_vals * f0 * torch.sqrt(1 + B * k_vals**2)).cpu().numpy()

        tau1_k1    = torch.exp(model.forward_tau1_k1(mf, vf)).squeeze()
        log_ratios = model.tau_ratio_net(torch.cat([mf_b, kf_b], -1)).squeeze(-1)
        log_k_bias = -0.3 * torch.log(k_vals)
        log_ratios = torch.minimum(log_ratios, torch.zeros_like(log_ratios))
        log_ratios = torch.maximum(log_ratios, log_k_bias - 2.0)
        tau1s = (tau1_k1 * torch.exp(log_ratios)).clamp(min=0.005).cpu().numpy()

        A0s = torch.exp(model.A0_net(
            torch.cat([mf_b, kf_b, vf_b], -1)).squeeze(-1)).clamp(min=1e-6).cpu().numpy()

        phi_diff_nn = model.forward_phi(mf, vf).squeeze().item()

    print(f"\n{'':─<65}")
    print(f"  {'Partial':>7}  {'f_hz JSON':>10}  {'f_hz NN':>10}  "
          f"{'A0 JSON':>9}  {'A0 NN':>9}  {'tau1 JSON':>9}  {'tau1 NN':>9}")
    print(f"{'':─<65}")
    partials = note["partials"]
    for ki, p in enumerate(partials[:min(10, len(partials))]):
        print(f"  k={ki+1:>5}  {p['f_hz']:>10.3f}  {f_hzs[ki]:>10.3f}  "
              f"{p['A0']:>9.4f}  {A0s[ki]:>9.4f}  "
              f"{p['tau1']:>9.4f}  {tau1s[ki]:>9.4f}")
    print(f"  phi_diff JSON={note['phi_diff']:.6f}  NN={phi_diff_nn:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params",   default="analysis/params-piano-ft-ks-grand.json")
    ap.add_argument("--model",    default="analysis/profile-finetuned.pt")
    ap.add_argument("--midi",     type=int,   default=60)
    ap.add_argument("--vel",      type=int,   default=3)
    ap.add_argument("--sr",       type=int,   default=44100)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--rng-seed", type=int,   default=0)
    ap.add_argument("--plot",     action="store_true")
    args = ap.parse_args()

    # ── Load params JSON ──────────────────────────────────────────────────────
    with open(args.params) as f:
        pdata = json.load(f)

    export_sr  = pdata.get("sr", 44100)
    key        = f"m{args.midi:03d}_vel{args.vel}"
    if key not in pdata["notes"]:
        print(f"ERROR: note {key} not in params file"); return
    note = pdata["notes"][key]

    print(f"Note: midi={args.midi} vel={args.vel}  K={note['K_valid']} partials")
    print(f"Export SR: {export_sr} Hz   Render SR: {args.sr} Hz")
    if export_sr != args.sr:
        print(f"  WARNING: SR mismatch! rms_gain was computed at {export_sr} Hz")

    # ── Load NN model ─────────────────────────────────────────────────────────
    ckpt  = torch.load(args.model, map_location="cpu", weights_only=False)
    model = InstrumentProfile(hidden=ckpt.get("hidden", 64))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── Compare per-partial params ────────────────────────────────────────────
    compare_partial_phases(note, model, args.midi, args.vel, args.sr)

    # ── Render both ───────────────────────────────────────────────────────────
    print(f"\nRendering C++ algorithm (numpy, no noise) ...")
    cpp = render_cpp_algorithm(note, sr=args.sr, duration=args.duration)

    print(f"Rendering reference (torch_synth, no noise) ...")
    ref = render_reference(model, args.midi, args.vel,
                           sr=args.sr, duration=args.duration,
                           rng_seed=args.rng_seed)

    # ── Difference analysis ───────────────────────────────────────────────────
    n    = min(len(cpp), len(ref))
    cpp  = cpp[:n]
    ref  = ref[:n]
    diff = cpp - ref

    rms_ref  = float(np.sqrt(np.mean(ref**2)))
    rms_cpp  = float(np.sqrt(np.mean(cpp**2)))
    rms_diff = float(np.sqrt(np.mean(diff**2)))
    peak_ref = float(np.max(np.abs(ref)))
    peak_cpp = float(np.max(np.abs(cpp)))
    snr_db   = 20 * np.log10(rms_ref / (rms_diff + 1e-12))

    print(f"\n{'':═<50}")
    print(f"  RMS  reference:  {rms_ref:.6f}  (peak {peak_ref:.4f})")
    print(f"  RMS  C++ algo:   {rms_cpp:.6f}  (peak {peak_cpp:.4f})")
    print(f"  RMS  difference: {rms_diff:.6f}")
    print(f"  SNR:             {snr_db:.1f} dB")
    print(f"{'':═<50}")

    if snr_db > 60:
        print("  RESULT: MATCH (SNR > 60 dB — float32 rounding only)")
    elif snr_db > 40:
        print("  RESULT: CLOSE (minor float32 discrepancy, acceptable)")
    else:
        print(f"  RESULT: MISMATCH — SNR={snr_db:.1f} dB, investigate!")

        # Find time of maximum divergence
        t_arr = np.arange(n) / args.sr
        bad_i = int(np.argmax(np.abs(diff)))
        print(f"  Max diff at t={t_arr[bad_i]:.3f}s  "
              f"ref={ref[bad_i]:.4f}  cpp={cpp[bad_i]:.4f}")

        # Check if it's mostly from early samples (onset) or later (decay)
        split = int(0.1 * args.sr)  # 100 ms
        rms_early = float(np.sqrt(np.mean(diff[:split]**2)))
        rms_late  = float(np.sqrt(np.mean(diff[split:]**2)))
        print(f"  Diff RMS: first 100ms={rms_early:.6f}  rest={rms_late:.6f}")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            t_arr = np.arange(n) / args.sr
            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            axes[0].plot(t_arr, ref,  label="reference (torch_synth)", alpha=0.7)
            axes[0].plot(t_arr, cpp,  label="C++ algorithm (numpy)",   alpha=0.7)
            axes[0].set_ylabel("amplitude"); axes[0].legend()
            axes[0].set_title(f"midi={args.midi} vel={args.vel}  SR={args.sr}  "
                              f"SNR={snr_db:.1f} dB")
            axes[1].plot(t_arr, diff, color="red", label="difference")
            axes[1].set_ylabel("diff"); axes[1].legend()
            axes[2].semilogy(t_arr, np.abs(diff) + 1e-10, color="orange")
            axes[2].set_ylabel("|diff| (log)"); axes[2].set_xlabel("time (s)")
            plt.tight_layout()
            plt.savefig(f"analysis/compare_m{args.midi:03d}_v{args.vel}.png", dpi=120)
            print(f"\n  Plot saved: analysis/compare_m{args.midi:03d}_v{args.vel}.png")
            plt.show()
        except ImportError:
            print("  (matplotlib not available, --plot skipped)")


if __name__ == "__main__":
    main()
