"""
analysis/mrstft_loss.py
────────────────────────
Multi-Resolution STFT Loss (MRSTFT).

Combines spectral convergence + log-magnitude loss at three FFT scales:
  n_fft = 256   (≈ 6 ms resolution — transients / attack)
  n_fft = 1024  (≈ 23 ms — balanced time-frequency)
  n_fft = 4096  (≈ 93 ms — fine spectral / pitch accuracy / sustain)

Reference: Yamamoto et al. "Parallel WaveGAN", ICASSP 2020.

Usage
─────
    from analysis.mrstft_loss import mrstft, mrstft_numpy

    # torch tensors (N,) or (N,2) stereo:
    loss = mrstft(pred_audio, ref_audio)       # scalar tensor, differentiable

    # numpy arrays:
    loss_val = mrstft_numpy(pred_np, ref_np)   # float
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Sequence

# Default (n_fft, hop_length, win_length) triples
DEFAULT_SCALES: list[tuple[int, int, int]] = [
    (256,   64,  256),
    (1024, 256, 1024),
    (4096, 1024, 4096),
]

_EPS = 1e-7


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_mono(x: torch.Tensor) -> torch.Tensor:
    """Convert stereo to mono if needed.
    Accepts: (N,) | (N,2) | (2,N) — always returns (N,).
    """
    if x.dim() == 1:
        return x
    if x.shape[-1] == 2:     # (N, 2)
        return x.mean(-1)
    if x.shape[0] == 2:      # (2, N)
        return x.mean(0)
    return x


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win_len: int) -> torch.Tensor:
    """STFT magnitude spectrogram, shape (F, T) where F = n_fft//2 + 1."""
    window = torch.hann_window(win_len, device=x.device, dtype=x.dtype)
    S = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win_len,
        window=window, return_complex=True,
        center=True, pad_mode='reflect',
    )
    return S.abs()


# ── Individual terms ──────────────────────────────────────────────────────────

def spectral_convergence_loss(
    pred_mag: torch.Tensor,
    ref_mag:  torch.Tensor,
) -> torch.Tensor:
    """Frobenius-norm spectral convergence: ||pred − ref||_F / ||ref||_F."""
    return (pred_mag - ref_mag).norm('fro') / (ref_mag.norm('fro') + _EPS)


def log_magnitude_loss(
    pred_mag: torch.Tensor,
    ref_mag:  torch.Tensor,
) -> torch.Tensor:
    """Mean absolute log-magnitude difference: E[|log(|S_pred|) − log(|S_ref|)|]."""
    return F.l1_loss(
        torch.log(pred_mag + _EPS),
        torch.log(ref_mag  + _EPS),
    )


# ── Main loss ─────────────────────────────────────────────────────────────────

def mrstft(
    pred:      torch.Tensor,
    ref:       torch.Tensor,
    scales:    Sequence[tuple[int, int, int]] = DEFAULT_SCALES,
    sc_weight: float = 1.0,
    lm_weight: float = 1.0,
    mono:      bool  = True,
) -> torch.Tensor:
    """
    Multi-Resolution STFT loss — differentiable scalar tensor.

    Args:
        pred:       predicted audio  (N,) | (N,2) | (2,N)
        ref:        reference audio, same or compatible shape
        scales:     list of (n_fft, hop_length, win_length)
        sc_weight:  spectral convergence weight per scale
        lm_weight:  log-magnitude weight per scale
        mono:       convert stereo to mono before STFT (default True)

    Returns:
        Scalar tensor — mean of (sc + lm) across all valid scales.
        Differentiable w.r.t. pred.
    """
    if mono:
        pred = _to_mono(pred)
        ref  = _to_mono(ref)

    # Align lengths to shorter signal
    n = min(pred.shape[-1], ref.shape[-1])
    pred = pred[..., :n].float()
    ref  = ref [..., :n].float()

    total       = pred.new_zeros(())
    valid_scales = 0

    for n_fft, hop, win_len in scales:
        if n < win_len:
            continue  # signal too short for this scale — skip gracefully

        pm = _stft_mag(pred, n_fft, hop, win_len)
        rm = _stft_mag(ref,  n_fft, hop, win_len)

        total = (total
                 + sc_weight * spectral_convergence_loss(pm, rm)
                 + lm_weight * log_magnitude_loss(pm, rm))
        valid_scales += 1

    if valid_scales == 0:
        return pred.new_zeros(())

    return total / valid_scales


# ── Numpy convenience wrapper ─────────────────────────────────────────────────

def mrstft_numpy(
    pred: 'np.ndarray',
    ref:  'np.ndarray',
    **kwargs,
) -> float:
    """Compute MRSTFT on numpy arrays. Returns float (no gradient)."""
    import numpy as np
    pred_t = torch.from_numpy(np.asarray(pred, dtype=np.float32))
    ref_t  = torch.from_numpy(np.asarray(ref,  dtype=np.float32))
    with torch.no_grad():
        return mrstft(pred_t, ref_t, **kwargs).item()
