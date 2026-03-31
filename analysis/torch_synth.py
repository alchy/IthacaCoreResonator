"""
analysis/torch_synth.py
────────────────────────
Differentiable PyTorch proxy of physics_synth.synthesize_note().

Purpose: gradient computation for closed-loop NN fine-tuning.
Faithful C++ rendering (RenderClient) is used for evaluation.
This proxy supplies approximate gradients for backpropagation.

Simplifications vs physics_synth.py (acceptable for gradient quality):
  • Mono output — stereo pan/decorrelation adds non-differentiable structure
  • 2-string per partial with phi_diff learned by phi_net  ← NEW
  • Noise: envelope-shaped Gaussian (no IIR spectral coloring)
  • Spectral EQ skipped (eq_net evaluated separately in eval mode)
  • No onset ramp (≤ 3 ms, negligible for MRSTFT)

All synthesis operations use native PyTorch; gradients flow from
MRSTFT loss back through audio → physics parameters → NN weights.

Batching: vectorised over K partials, serial over notes.

2-string model per partial:
  s1 = cos(2π*(f + beat/2)*t + phi1)
  s2 = cos(2π*(f - beat/2)*t + phi1 + phi_diff)
  partial = A0 * env * (s1 + s2) / 2

  phi_diff from phi_net: 0 → constructive (max attack), π → destructive.
  beat_hz from df_net, scaled by beat_scale.
  Grad flows through: B_net (f_hzs), tau nets, A0_net, biexp_net,
                      noise_net, df_net (beat_hz), phi_net (phi_diff).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Union

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
    beat_scale:  Union[float, torch.Tensor] = 1.0,
    noise_level: Union[float, torch.Tensor] = 1.0,
    target_rms:  float = 0.06,
    vel_gamma:   float = 0.7,
    k_max:       int   = K_MAX_DEFAULT,
    rng_seed:    int   = 0,
) -> torch.Tensor:
    """
    Differentiable mono proxy for one (midi, vel) note.

    Uses 2-string synthesis per partial.  phi_diff comes from phi_net
    (trainable initial phase offset between the two strings).  Beat
    frequency comes from df_net, scaled by beat_scale.

    Args:
        model:       InstrumentProfile (on any device)
        midi:        MIDI note number 21–108
        vel:         velocity index 0–7
        sr:          sample rate
        duration:    render duration in seconds
        beat_scale:  df multiplier — float or differentiable tensor
        noise_level: noise amplitude multiplier — float or differentiable tensor
        target_rms:  normalisation target
        vel_gamma:   velocity curve exponent (SynthConfig.vel_gamma)
        k_max:       max number of partials (reduces memory for low MIDI notes)
        rng_seed:    base seed; actual seed = rng_seed + midi*256 + vel

    Returns:
        (N,) float32 mono audio tensor, normalised to target_rms * vel_gain.
        Differentiable w.r.t. InstrumentProfile parameters and any
        tensor-valued beat_scale / noise_level.

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
    kf_b   = _kf_batch(K, device)                         # (K, 3)
    k_vals = torch.arange(1, K + 1, dtype=torch.float32, device=device)  # (K,)
    mf_b   = mf.unsqueeze(0).expand(K, -1)                # (K, 6)
    vf_b   = vf.unsqueeze(0).expand(K, -1)                # (K, 3)

    # ── Inharmonicity B → partial frequencies ─────────────────────────────────
    # f_k = k * f0 * sqrt(1 + B * k²)   [grad through B_net]
    # Clamp log_B to prevent B from growing large enough that f_hzs → inf,
    # which causes cos(inf) = NaN and poisons the entire gradient.
    # Grand piano B is physically in [1e-5, 0.01]; clamp(max=0) → B ≤ 1.0 is generous.
    B      = torch.exp(model.forward_B(mf).clamp(max=0.0)).squeeze()  # scalar, B ≤ 1.0
    f_hzs  = k_vals * f0 * torch.sqrt(1.0 + B * k_vals ** 2)  # (K,)
    # isfinite guards against any remaining inf/nan before they enter cos()
    valid  = ((f_hzs < sr * 0.495) & torch.isfinite(f_hzs)).float().unsqueeze(1)  # (K, 1)

    # ── Beat frequencies from df_net ──────────────────────────────────────────
    # df_net → log(beat_hz) per partial   [grad through df_net]
    log_df  = model.forward_df(mf_b, kf_b).squeeze(-1)    # (K,)
    beat_hz = torch.exp(log_df).clamp(min=1e-4)            # (K,)
    if isinstance(beat_scale, torch.Tensor):
        beat_hz = beat_hz * beat_scale
    else:
        beat_hz = beat_hz * float(beat_scale)

    # ── Decay times ───────────────────────────────────────────────────────────
    # tau1 for k=1 from dedicated net; k>1 from tau1_k1 * exp(clamped ratio)
    tau1_k1 = torch.exp(model.forward_tau1_k1(mf, vf)).squeeze()  # scalar

    log_ratios = model.tau_ratio_net(
        torch.cat([mf_b, kf_b], dim=-1)
    ).squeeze(-1)                                          # (K,)

    # Same clamping as generate_samples() in train_instrument_profile.py
    # torch.clamp cannot mix Tensor min with float max → use minimum/maximum
    log_k_bias = -0.3 * torch.log(k_vals)                 # (K,)
    log_ratios = torch.minimum(log_ratios, torch.zeros_like(log_ratios))   # max=0.0
    log_ratios = torch.maximum(log_ratios, log_k_bias - 2.0)              # min=tensor

    tau1s = (tau1_k1 * torch.exp(log_ratios)).clamp(min=0.005)   # (K,)

    # ── Amplitudes ────────────────────────────────────────────────────────────
    A0s = torch.exp(
        model.A0_net(torch.cat([mf_b, kf_b, vf_b], dim=-1)).squeeze(-1)
    ).clamp(min=1e-6)                                      # (K,)

    # ── Bi-exponential parameters ─────────────────────────────────────────────
    # biexp_net → [logit(a1), log(tau2/tau1)]
    biexp = model.biexp_net(
        torch.cat([mf_b, kf_b, vf_b], dim=-1)
    )                                                      # (K, 2)
    a1s         = torch.sigmoid(biexp[:, 0]).clamp(0.05, 0.99)   # (K,)
    tau2_ratios = torch.exp(biexp[:, 1]).clamp(min=3.0)          # (K,)
    tau2s       = tau1s * tau2_ratios                             # (K,)

    # ── Noise parameters ──────────────────────────────────────────────────────
    # noise_net → [log(attack_tau_s), log(centroid_hz), log(A_noise)]
    noise_pred = model.forward_noise(mf, vf).squeeze()    # (3,)
    attack_tau = torch.exp(noise_pred[0]).clamp(0.002, 1.0)   # scalar
    A_noise    = torch.exp(noise_pred[2]).clamp(0.001, 0.5)   # scalar

    # ── Learned initial phase offset from phi_net ─────────────────────────────
    # phi_diff is the relative phase between string 2 and string 1 (K-broadcast).
    # grad flows back through phi_net.
    phi_diff = model.forward_phi(mf, vf).squeeze()        # scalar

    # ── Fixed random realizations (same per note, deterministic) ─────────────
    seed = rng_seed + midi * 256 + vel
    gen  = torch.Generator()          # CPU generator — move tensors to device after
    gen.manual_seed(seed)
    phis      = torch.rand(K, generator=gen).mul(2 * math.pi)  # (K,) phase for string 1
    noise_raw = torch.randn(n, generator=gen)                   # (N,) white noise
    phis      = phis.to(device)
    noise_raw = noise_raw.to(device)

    # ── Bi-exponential envelope (shared by both strings) ─────────────────────
    env_fast = torch.exp(-t.unsqueeze(0) / tau1s.unsqueeze(1))  # (K, N)
    env_slow = torch.exp(-t.unsqueeze(0) / tau2s.unsqueeze(1))  # (K, N)
    envs = (a1s.unsqueeze(1) * env_fast
            + (1.0 - a1s).unsqueeze(1) * env_slow)              # (K, N)

    # ── 2-string oscillators per partial ─────────────────────────────────────
    # String 1: f + beat/2, phase = phi1
    # String 2: f - beat/2, phase = phi1 + phi_diff   [grad through phi_diff, beat_hz]
    half_beat = beat_hz.unsqueeze(1) * 0.5                       # (K, 1)
    phase_base = 2.0 * math.pi * t.unsqueeze(0) * f_hzs.unsqueeze(1)  # (K, N)
    beat_phase = 2.0 * math.pi * t.unsqueeze(0) * half_beat     # (K, N)
    phi_init   = phis.unsqueeze(1)                                # (K, 1)

    osc1 = torch.cos(phase_base + beat_phase + phi_init)         # (K, N)
    osc2 = torch.cos(phase_base - beat_phase + phi_init + phi_diff)  # (K, N)
    oscs = (osc1 + osc2) * 0.5                                   # (K, N), range [-1, 1]

    # Sum over partials → (N,)
    audio = (A0s.unsqueeze(1) * envs * oscs * valid).sum(0)     # (N,)

    # ── Noise signal ──────────────────────────────────────────────────────────
    # Envelope-shaped Gaussian — grad through A_noise and attack_tau.
    # Centroid shaping (IIR) is not differentiable; amplitude + decay are enough.
    noise_env = torch.exp(-t / attack_tau.clamp(min=1e-4))       # (N,)
    if isinstance(noise_level, torch.Tensor):
        audio = audio + A_noise * noise_level * noise_raw * noise_env
    else:
        audio = audio + A_noise * float(noise_level) * noise_raw * noise_env

    # ── RMS normalisation ─────────────────────────────────────────────────────
    # vel_gain = ((vel+1)/8)^gamma — constant (vel_gamma is SynthConfig scalar)
    vel_gain   = ((vel + 1) / 8.0) ** float(vel_gamma)           # float
    rms        = torch.sqrt(audio.pow(2).mean() + 1e-10)         # scalar, grad
    audio_norm = audio * (float(target_rms) * vel_gain / rms)    # (N,)

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
