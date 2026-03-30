"""
analysis/torch_synth.py
────────────────────────
Differentiable PyTorch proxy of physics_synth.synthesize_note().

Purpose: gradient computation for closed-loop NN fine-tuning.
Faithful C++ rendering (RenderClient) is used for evaluation.
This proxy supplies approximate gradients for backpropagation.

Simplifications vs physics_synth.py (acceptable for gradient quality):
  • Mono output — stereo pan/decorrelation adds non-differentiable structure
  • Single oscillator per partial (no multi-string beat splitting)
  • Noise: envelope-shaped Gaussian (no IIR spectral coloring)
  • Spectral EQ skipped (eq_net evaluated separately in eval mode)
  • No onset ramp (≤ 3 ms, negligible for MRSTFT)

All synthesis operations use native PyTorch; gradients flow from
MRSTFT loss back through audio → physics parameters → NN weights.

Batching: vectorised over K partials, serial over notes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# ── Import shared utilities from train script ─────────────────────────────────
# (feature encoders + InstrumentProfile class)
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.train_instrument_profile import (  # noqa: E402
    InstrumentProfile,
    midi_feat   as _midi_feat,
    vel_feat    as _vel_feat,
    k_feat      as _k_feat,
    midi_to_hz,
)


# ── Constants ─────────────────────────────────────────────────────────────────

# k_feat normalisation constant — must match train_instrument_profile.py
_K_FEAT_NORM = 90

# Default partial cap — 60 covers all audible partials for MIDI ≥ 40
# Lower to 30 to halve memory use on constrained machines.
K_MAX_DEFAULT = 60


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kf_batch(n_partials: int, device: torch.device) -> torch.Tensor:
    """(K, 3) constant tensor of k_feat vectors."""
    return torch.stack(
        [_k_feat(k, k_max=_K_FEAT_NORM) for k in range(1, n_partials + 1)]
    ).to(device)


# ── Main differentiable render ────────────────────────────────────────────────

def render_note_differentiable(
    model:       InstrumentProfile,
    midi:        int,
    vel:         int,
    *,
    sr:          int   = 44_100,
    duration:    float = 3.0,
    beat_scale:  float = 1.0,   # reserved — df not included in proxy (no multi-string)
    noise_level: float = 1.0,
    target_rms:  float = 0.06,
    vel_gamma:   float = 0.7,
    k_max:       int   = K_MAX_DEFAULT,
    rng_seed:    int   = 0,
) -> torch.Tensor:
    """
    Differentiable mono proxy for one (midi, vel) note.

    Runs InstrumentProfile forward pass + physics-based synthesis.
    All NN parameter tensors carry gradients; random phases and noise
    realizations are fixed per (midi, vel, rng_seed) for a smooth loss surface.

    Args:
        model:       InstrumentProfile (on any device)
        midi:        MIDI note number 21–108
        vel:         velocity index 0–7
        sr:          sample rate
        duration:    render duration in seconds
        noise_level: noise amplitude multiplier (SynthConfig.noise_level)
        target_rms:  normalisation target
        vel_gamma:   velocity curve exponent (SynthConfig.vel_gamma)
        k_max:       max number of partials (reduces memory for low MIDI notes)
        rng_seed:    base seed; actual seed = rng_seed + midi*256 + vel

    Returns:
        (N,) float32 mono audio tensor, normalised to target_rms * vel_gain.
        Differentiable w.r.t. all InstrumentProfile parameters.

    Memory note:
        Internal (K, N) matrices dominate memory.
        K=60, duration=3 s → ~128 MB per note (gradient tape ≈ 3×).
        Use k_max=30 or duration=1.5 on memory-constrained machines.
    """
    device = next(model.parameters()).device

    # ── Feature vectors (constants) ───────────────────────────────────────────
    mf = _midi_feat(midi).to(device)   # (6,)
    vf = _vel_feat(vel).to(device)     # (3,)
    f0 = midi_to_hz(midi)              # float

    # ── Partial count ─────────────────────────────────────────────────────────
    n_max_nyquist = max(1, int(sr / 2 / f0))
    K = min(k_max, n_max_nyquist)

    # ── Time vector (no grad) ─────────────────────────────────────────────────
    n = int(duration * sr)
    t = torch.arange(n, dtype=torch.float32, device=device) / sr  # (N,)

    # ── Batched feature tensors (constants) ───────────────────────────────────
    kf_b  = _kf_batch(K, device)                         # (K, 3)
    k_vals = torch.arange(1, K + 1, dtype=torch.float32, device=device)  # (K,)
    mf_b  = mf.unsqueeze(0).expand(K, -1)                # (K, 6)
    vf_b  = vf.unsqueeze(0).expand(K, -1)                # (K, 3)

    # ── Inharmonicity B → partial frequencies ─────────────────────────────────
    # f_k = k * f0 * sqrt(1 + B * k²)   [grad through B_net]
    B      = torch.exp(model.forward_B(mf)).squeeze()    # scalar
    f_hzs  = k_vals * f0 * torch.sqrt(1.0 + B * k_vals ** 2)  # (K,)
    valid  = (f_hzs < sr * 0.495).float().unsqueeze(1)   # (K, 1) mask

    # ── Decay times ───────────────────────────────────────────────────────────
    # tau1 for k=1 from dedicated net; k>1 from tau1_k1 * exp(clamped ratio)
    tau1_k1 = torch.exp(model.forward_tau1_k1(mf, vf)).squeeze()  # scalar

    log_ratios = model.tau_ratio_net(
        torch.cat([mf_b, kf_b], dim=-1)
    ).squeeze(-1)                                         # (K,)

    # Same clamping as generate_samples() in train_instrument_profile.py
    # torch.clamp cannot mix Tensor min with float max → use minimum/maximum
    log_k_bias = -0.3 * torch.log(k_vals)                # (K,)
    log_ratios = torch.minimum(log_ratios, torch.zeros_like(log_ratios))   # max=0.0
    log_ratios = torch.maximum(log_ratios, log_k_bias - 2.0)              # min=tensor

    tau1s = (tau1_k1 * torch.exp(log_ratios)).clamp(min=0.005)   # (K,)

    # ── Amplitudes ────────────────────────────────────────────────────────────
    A0s = torch.exp(
        model.A0_net(torch.cat([mf_b, kf_b, vf_b], dim=-1)).squeeze(-1)
    ).clamp(min=1e-6)                                     # (K,)

    # ── Bi-exponential parameters ─────────────────────────────────────────────
    # biexp_net → [logit(a1), log(tau2/tau1)]
    biexp = model.biexp_net(
        torch.cat([mf_b, kf_b, vf_b], dim=-1)
    )                                                     # (K, 2)
    a1s         = torch.sigmoid(biexp[:, 0]).clamp(0.05, 0.99)   # (K,)
    tau2_ratios = torch.exp(biexp[:, 1]).clamp(min=3.0)          # (K,)
    tau2s       = tau1s * tau2_ratios                             # (K,)

    # ── Noise parameters ──────────────────────────────────────────────────────
    # noise_net → [log(attack_tau_s), log(centroid_hz), log(A_noise)]
    noise_pred = model.forward_noise(mf, vf).squeeze()   # (3,)
    attack_tau = torch.exp(noise_pred[0]).clamp(0.002, 1.0)  # scalar
    A_noise    = torch.exp(noise_pred[2]).clamp(0.001, 0.5)  # scalar

    # ── Fixed random realizations (same per note, deterministic) ─────────────
    seed = rng_seed + midi * 256 + vel
    gen  = torch.Generator()          # CPU generator — move tensors to device after
    gen.manual_seed(seed)
    phis      = torch.rand(K, generator=gen).mul(2 * math.pi)  # (K,) phase
    noise_raw = torch.randn(n, generator=gen)                   # (N,) white noise
    phis      = phis.to(device)
    noise_raw = noise_raw.to(device)

    # ── Partial synthesis (vectorised over K) ─────────────────────────────────
    # Bi-exponential envelope: (K, N)
    #   env = a1 * exp(-t/tau1) + (1-a1) * exp(-t/tau2)
    env_fast = torch.exp(-t.unsqueeze(0) / tau1s.unsqueeze(1))  # (K, N)
    env_slow = torch.exp(-t.unsqueeze(0) / tau2s.unsqueeze(1))  # (K, N)
    envs = (a1s.unsqueeze(1) * env_fast
            + (1.0 - a1s).unsqueeze(1) * env_slow)              # (K, N)

    # Oscillator: (K, N)   grad flows through f_hzs → B
    oscs = torch.cos(
        2.0 * math.pi * f_hzs.unsqueeze(1) * t.unsqueeze(0)
        + phis.unsqueeze(1)
    )                                                            # (K, N)

    # Sum over partials → (N,)
    audio = (A0s.unsqueeze(1) * envs * oscs * valid).sum(0)    # (N,)

    # ── Noise signal ──────────────────────────────────────────────────────────
    # Envelope-shaped Gaussian — grad through A_noise and attack_tau.
    # Centroid shaping (IIR) is not differentiable; amplitude + decay are enough.
    noise_env = torch.exp(-t / attack_tau.clamp(min=1e-4))      # (N,)
    audio = audio + A_noise * float(noise_level) * noise_raw * noise_env

    # ── RMS normalisation ─────────────────────────────────────────────────────
    # vel_gain = ((vel+1)/8)^gamma — constant (vel_gamma is SynthConfig scalar)
    vel_gain = ((vel + 1) / 8.0) ** float(vel_gamma)            # float
    rms      = torch.sqrt(audio.pow(2).mean() + 1e-10)          # scalar, grad
    audio_norm = audio * (float(target_rms) * vel_gain / rms)   # (N,)

    return audio_norm


# ── Batch helper ──────────────────────────────────────────────────────────────

def render_batch_differentiable(
    model:    InstrumentProfile,
    notes:    list[tuple[int, int]],   # list of (midi, vel)
    **kwargs,
) -> list[torch.Tensor]:
    """
    Render a list of (midi, vel) pairs sequentially.
    Returns list of (N_i,) tensors — lengths may differ if notes have
    different durations. Pass a fixed ``duration`` kwarg to unify lengths.
    """
    return [render_note_differentiable(model, midi, vel, **kwargs)
            for midi, vel in notes]
