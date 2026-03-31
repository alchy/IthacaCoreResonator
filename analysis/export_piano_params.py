"""
analysis/export_piano_params.py
────────────────────────────────
Export all synthesis parameters needed by synth-core/piano/PianoCore from a
trained InstrumentProfile model.

Runs all NN forward passes (eval mode, no_grad) for MIDI 21-108 × vel 0-7
and writes a compact JSON file that PianoCore can load directly without
requiring the PyTorch model at runtime.

Per (midi, vel) output:
  - Per-partial: f_hz, A0, tau1, tau2, a1, beat_hz, phi (precomputed)
  - Scalars:     phi_diff, attack_tau, A_noise, rms_gain

phi values are precomputed via torch.Generator with seed = rng_seed+midi*256+vel,
matching torch_synth.py exactly.  rms_gain is computed from the full-duration
audio render so C++ can apply it directly.

Usage:
    python -u analysis/export_piano_params.py \\
        --model    analysis/profile.pt \\
        --out      analysis/params-piano-ks-grand.json \\
        [--k-max   60] [--sr 44100] [--duration 3.0] \\
        [--target-rms 0.06] [--vel-gamma 0.7] [--rng-seed 0] \\
        [--midi-from 21] [--midi-to 108]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.train_instrument_profile import (   # noqa: E402
    InstrumentProfile,
    midi_feat,
    vel_feat,
    k_feat,
    midi_to_hz,
)

_K_FEAT_NORM = 90   # must match train_instrument_profile.py


# ── Per-note export ────────────────────────────────────────────────────────────

def export_note(model: InstrumentProfile, midi: int, vel: int, *,
                sr: int, duration: float, k_max: int,
                target_rms: float, vel_gamma: float,
                rng_seed: int) -> dict:
    """
    Compute all synthesis parameters for one (midi, vel) note.
    Replicates torch_synth.render_note_differentiable() exactly
    (no_grad, same arithmetic).
    """
    device = next(model.parameters()).device

    mf = midi_feat(midi).to(device)
    vf = vel_feat(vel).to(device)
    f0 = midi_to_hz(midi)

    n_max_nyquist = max(1, int(sr / 2 / f0))
    K = min(k_max, n_max_nyquist)

    n = int(duration * sr)
    t = torch.arange(n, dtype=torch.float32, device=device) / sr   # (N,)

    kf_b   = torch.stack(
        [k_feat(k, k_max=_K_FEAT_NORM) for k in range(1, K + 1)]
    ).to(device)                                                     # (K, 3)
    k_vals = torch.arange(1, K + 1, dtype=torch.float32, device=device)
    mf_b   = mf.unsqueeze(0).expand(K, -1)
    vf_b   = vf.unsqueeze(0).expand(K, -1)

    # ── Partial frequencies ────────────────────────────────────────────────────
    B      = torch.exp(model.forward_B(mf).clamp(max=0.0)).squeeze()
    f_hzs  = k_vals * f0 * torch.sqrt(1.0 + B * k_vals ** 2)       # (K,)
    valid  = (f_hzs < sr * 0.495) & torch.isfinite(f_hzs)          # (K,) bool

    # ── Beat frequencies ───────────────────────────────────────────────────────
    log_df  = model.forward_df(mf_b, kf_b).squeeze(-1)
    beat_hz = torch.exp(log_df).clamp(min=1e-4)                     # (K,)

    # ── Decay times ───────────────────────────────────────────────────────────
    tau1_k1    = torch.exp(model.forward_tau1_k1(mf, vf)).squeeze()
    log_ratios = model.tau_ratio_net(
        torch.cat([mf_b, kf_b], dim=-1)
    ).squeeze(-1)
    log_k_bias = -0.3 * torch.log(k_vals)
    log_ratios = torch.minimum(log_ratios, torch.zeros_like(log_ratios))
    log_ratios = torch.maximum(log_ratios, log_k_bias - 2.0)
    tau1s      = (tau1_k1 * torch.exp(log_ratios)).clamp(min=0.005)  # (K,)

    # ── Amplitudes ────────────────────────────────────────────────────────────
    A0s = torch.exp(
        model.A0_net(torch.cat([mf_b, kf_b, vf_b], dim=-1)).squeeze(-1)
    ).clamp(min=1e-6)                                                 # (K,)

    # ── Bi-exponential ─────────────────────────────────────────────────────────
    biexp       = model.biexp_net(torch.cat([mf_b, kf_b, vf_b], dim=-1))
    a1s         = torch.sigmoid(biexp[:, 0]).clamp(0.05, 0.99)       # (K,)
    tau2_ratios = torch.exp(biexp[:, 1]).clamp(min=3.0)
    tau2s       = tau1s * tau2_ratios                                  # (K,)

    # ── Noise ─────────────────────────────────────────────────────────────────
    noise_pred = model.forward_noise(mf, vf).squeeze()               # (3,)
    attack_tau = torch.exp(noise_pred[0]).clamp(0.002, 1.0).item()
    A_noise    = torch.exp(noise_pred[2]).clamp(0.001, 0.5).item()

    # ── phi_diff ──────────────────────────────────────────────────────────────
    phi_diff = model.forward_phi(mf, vf).squeeze().item()

    # ── Fixed random phis (matching torch_synth.py exactly) ──────────────────
    seed = rng_seed + midi * 256 + vel
    gen  = torch.Generator()           # CPU generator
    gen.manual_seed(seed)
    phis      = torch.rand(K, generator=gen).mul(2 * math.pi)       # (K,)
    noise_raw = torch.randn(n, generator=gen)                        # (N,)

    # ── RMS normalisation (full render, matching torch_synth.py) ─────────────
    half_beat  = beat_hz.unsqueeze(1) * 0.5
    phase_base = 2.0 * math.pi * t.unsqueeze(0) * f_hzs.unsqueeze(1)  # (K,N)
    beat_phase = 2.0 * math.pi * t.unsqueeze(0) * half_beat
    phi_init   = phis.unsqueeze(1)                                   # (K,1)

    osc1 = torch.cos(phase_base + beat_phase + phi_init)
    osc2 = torch.cos(phase_base - beat_phase + phi_init + phi_diff)
    oscs = (osc1 + osc2) * 0.5                                       # (K,N)

    valid_f  = valid.float().unsqueeze(1)
    env_fast = torch.exp(-t.unsqueeze(0) / tau1s.unsqueeze(1))
    env_slow = torch.exp(-t.unsqueeze(0) / tau2s.unsqueeze(1))
    envs     = a1s.unsqueeze(1) * env_fast + (1.0 - a1s).unsqueeze(1) * env_slow

    audio      = (A0s.unsqueeze(1) * envs * oscs * valid_f).sum(0)  # (N,)
    noise_env  = torch.exp(-t / max(attack_tau, 1e-4))
    audio      = audio + A_noise * noise_raw * noise_env

    vel_gain   = ((vel + 1) / 8.0) ** float(vel_gamma)
    rms        = torch.sqrt(audio.pow(2).mean() + 1e-10).item()
    rms_gain   = float(target_rms) * vel_gain / rms

    # ── Build per-partial list (valid partials only) ──────────────────────────
    valid_cpu = valid.cpu().tolist()
    phis_cpu  = phis.cpu().tolist()
    partials  = []
    for ki in range(K):
        if not valid_cpu[ki]:
            continue
        partials.append({
            "f_hz":    round(f_hzs[ki].item(), 6),
            "A0":      round(A0s[ki].item(), 8),
            "tau1":    round(tau1s[ki].item(), 8),
            "tau2":    round(tau2s[ki].item(), 8),
            "a1":      round(a1s[ki].item(), 8),
            "beat_hz": round(beat_hz[ki].item(), 8),
            "phi":     round(phis_cpu[ki], 8),
        })

    return {
        "midi":       midi,
        "vel":        vel,
        "f0_hz":      round(float(f0), 6),
        "K_valid":    len(partials),
        "phi_diff":   round(phi_diff, 8),
        "attack_tau": round(attack_tau, 8),
        "A_noise":    round(A_noise, 8),
        "rms_gain":   round(rms_gain, 8),
        "partials":   partials,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export PianoCore parameters from a trained InstrumentProfile"
    )
    ap.add_argument("--model",      default="analysis/profile.pt",
                    help="Trained checkpoint (default: analysis/profile.pt)")
    ap.add_argument("--out",        default="analysis/params-piano-ks-grand.json",
                    help="Output JSON (default: analysis/params-piano-ks-grand.json)")
    ap.add_argument("--k-max",      type=int,   default=60)
    ap.add_argument("--sr",         type=int,   default=44100)
    ap.add_argument("--duration",   type=float, default=3.0,
                    help="Duration used to compute RMS normalisation (s)")
    ap.add_argument("--target-rms", type=float, default=0.06)
    ap.add_argument("--vel-gamma",  type=float, default=0.7)
    ap.add_argument("--rng-seed",   type=int,   default=0)
    ap.add_argument("--midi-from",  type=int,   default=21)
    ap.add_argument("--midi-to",    type=int,   default=108)
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    ckpt   = torch.load(args.model, map_location="cpu", weights_only=False)
    hidden = ckpt.get("hidden", 64)
    model  = InstrumentProfile(hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  hidden={hidden}, params={sum(p.numel() for p in model.parameters())}")

    midi_range = range(args.midi_from, args.midi_to + 1)
    total      = len(midi_range) * 8
    done       = 0
    notes: dict = {}

    with torch.no_grad():
        for midi in midi_range:
            for vel in range(8):
                key  = f"m{midi:03d}_vel{vel}"
                data = export_note(
                    model, midi, vel,
                    sr=args.sr,
                    duration=args.duration,
                    k_max=args.k_max,
                    target_rms=args.target_rms,
                    vel_gamma=args.vel_gamma,
                    rng_seed=args.rng_seed,
                )
                notes[key] = data
                done += 1
                if done % 88 == 0 or done == total:
                    print(f"  {done}/{total}  ({100*done//total}%)")

    output = {
        "format":     "piano-core-v1",
        "sr":         args.sr,
        "target_rms": args.target_rms,
        "vel_gamma":  args.vel_gamma,
        "k_max":      args.k_max,
        "rng_seed":   args.rng_seed,
        "duration_s": args.duration,
        "n_notes":    len(notes),
        "notes":      notes,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {len(notes)} notes -> {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
