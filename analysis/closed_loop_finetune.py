"""
analysis/closed_loop_finetune.py
─────────────────────────────────
Closed-loop MRSTFT fine-tuning of the InstrumentProfile neural network.

After training (train_instrument_profile.py) the NN predicts synthesis
parameters from (midi, vel).  This script refines those parameters further
by minimising the Multi-Resolution STFT loss (MRSTFT) between proxy-rendered
audio and the original piano WAV recordings.

Modes
─────
  eval      Report MRSTFT per note and aggregate statistics (no update).
  finetune  Gradient-based update of NN weights via torch proxy + MRSTFT.
  global    Optimise scalar SynthConfig parameters (beat_scale, noise_level,
            harmonic_brightness) with NN weights frozen.  Saves best values
            to a JSON file alongside the model.

Architecture
────────────
  Proxy synth (torch_synth.py)   — differentiable, mono, approximate
      ↓  (gradient path)
  MRSTFT loss (mrstft_loss.py)   — vs original WAV (mono crop)
      ↓
  Adam optimiser → NN weights  (finetune mode)
                → SynthConfigParams scalars  (global mode)

Evaluation is always done on the proxy (fast, no C++ dependency).
For faithful C++ evaluation use RenderClient separately after fine-tuning.

Usage
─────
    # Evaluate current model
    python analysis/closed_loop_finetune.py \
        --mode eval \
        --model analysis/profile.pt \
        --bank  soundbanks/ks-grand

    # Fine-tune NN weights
    python analysis/closed_loop_finetune.py \
        --mode finetune \
        --model analysis/profile.pt \
        --bank  soundbanks/ks-grand \
        --epochs 200 --lr 3e-4 --batch-size 8

    # Optimise global SynthConfig parameters (NN frozen)
    python analysis/closed_loop_finetune.py \
        --mode global \
        --model analysis/profile.pt \
        --bank  soundbanks/ks-grand \
        --opt-params beat_scale,noise_level \
        --epochs 100 --lr 0.05

Arguments (see --help)
──────────────────────
  --mode        eval | finetune | global
  --model       profile.pt path (input; also output unless --out given)
  --out         output model path (default: overwrite --model)
  --bank        directory with original WAV files (m060-vel3-f44.wav format)
  --wav-pattern glob pattern inside --bank (default: m*-vel*-f44.wav)
  --epochs      finetune/global epochs (default: 200)
  --lr          learning rate (default: 3e-4 for finetune, 0.05 for global)
  --batch-size  notes per gradient step, gradient-accumulated (default: 8)
  --duration    proxy render + reference crop in seconds (default: 3.0)
  --sr          sample rate (default: 44100)
  --target-rms  normalisation target (default: 0.06)
  --vel-gamma   velocity curve exponent — SynthConfig (default: 0.7)
  --noise-level noise multiplier — SynthConfig initial value (default: 1.0)
  --beat-scale  beat multiplier — SynthConfig initial value (default: 1.0)
  --opt-params  comma-separated SynthConfig params to optimise in global mode
                (choices: beat_scale, noise_level; default: beat_scale,noise_level)
  --config-out  JSON path for optimised SynthConfig (default: next to --out)
  --k-max       max partials in proxy (default: 60; use 30 to halve memory)
  --eval-every  eval full dataset every N epochs (default: 20)
  --seed        random seed (default: 42)
  --log         log file (default: analysis/runtime-logs/finetune.log)
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# Windows: force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Default notes rendered as WAV checkpoints: A0, C3, C4, C5, C7 across vel layers
_DEFAULT_SAMPLE_NOTES = "21:3,48:3,60:3,72:5,96:5"

# ── Project imports ───────────────────────────────────────────────────────────

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.train_instrument_profile import (
    InstrumentProfile,
    midi_feat, vel_feat,
)
from analysis.mrstft_loss  import mrstft
from analysis.torch_synth  import render_note_differentiable


# ── Global SynthConfig optimisation ──────────────────────────────────────────

class SynthConfigParams(nn.Module):
    """
    Differentiable wrapper around scalar SynthConfig parameters.

    Parameters are stored in log-space (nn.Parameter) for gradient stability:
      log_beat_scale  → beat_scale  = exp(p)   init: 0.0 → beat_scale=1.0
      log_noise_level → noise_level = exp(p)   init: 0.0 → noise_level=1.0

    Only parameters listed in ``opt_params`` are optimised; others are fixed
    at their initial values.

    Usage:
        cfg_params = SynthConfigParams(beat_scale=1.0, noise_level=1.0,
                                       opt_params=['beat_scale', 'noise_level'])
        optimizer = torch.optim.Adam(cfg_params.parameters(), lr=0.05)
        ...
        bs = cfg_params.get_beat_scale()   # differentiable tensor
        nl = cfg_params.get_noise_level()  # differentiable tensor
    """
    _SUPPORTED = ('beat_scale', 'noise_level')

    def __init__(
        self,
        beat_scale:  float = 1.0,
        noise_level: float = 1.0,
        opt_params:  list[str] | None = None,
    ) -> None:
        super().__init__()
        if opt_params is None:
            opt_params = list(self._SUPPORTED)

        # Always create both params; freeze the ones not being optimised
        self.log_beat_scale  = nn.Parameter(
            torch.tensor(math.log(max(beat_scale,  1e-4))),
            requires_grad='beat_scale'  in opt_params,
        )
        self.log_noise_level = nn.Parameter(
            torch.tensor(math.log(max(noise_level, 1e-4))),
            requires_grad='noise_level' in opt_params,
        )
        self._opt_params = opt_params

    def get_beat_scale(self)  -> torch.Tensor:
        return torch.exp(self.log_beat_scale).clamp(min=0.1, max=10.0)

    def get_noise_level(self) -> torch.Tensor:
        return torch.exp(self.log_noise_level).clamp(min=0.01, max=5.0)

    def to_dict(self) -> dict:
        return {
            'beat_scale':  float(self.get_beat_scale()),
            'noise_level': float(self.get_noise_level()),
        }


# ── Logging ───────────────────────────────────────────────────────────────────

class _Logger:
    """Tee to stdout + optional log file."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self._f = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._f = open(log_path, 'w', encoding='utf-8', buffering=1)

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        if self._f:
            self._f.write(msg + '\n')
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()


# ── Model I/O ─────────────────────────────────────────────────────────────────

def load_model(path: str) -> tuple[InstrumentProfile, dict]:
    """Load profile.pt; returns (model, metadata)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    hidden   = ckpt.get('hidden', 64)
    eq_freqs = ckpt.get('eq_freqs', None)
    model = InstrumentProfile(hidden=hidden)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if missing:
        print(f"  [load_model] new params (initialized fresh): {missing}", flush=True)
    if unexpected:
        print(f"  [load_model] unexpected keys (ignored): {unexpected}", flush=True)
    return model, {'hidden': hidden, 'eq_freqs': eq_freqs}


def save_model(model: InstrumentProfile, path: str, meta: dict) -> None:
    """Save profile.pt with same format as train_instrument_profile.py."""
    torch.save({
        'state_dict': model.state_dict(),
        'hidden':     meta['hidden'],
        'eq_freqs':   meta.get('eq_freqs'),
    }, path)


# ── WAV discovery and loading ─────────────────────────────────────────────────

_WAV_RE = re.compile(r'm(\d+)-vel(\d+)-.*\.wav', re.IGNORECASE)


def find_reference_wavs(
    bank_dir: str,
    pattern:  str = 'm*-vel*-f44.wav',
) -> list[tuple[int, int, Path]]:
    """
    Scan bank_dir for original WAV files matching pattern.
    Returns sorted list of (midi, vel, path).
    """
    bank = Path(bank_dir)
    if not bank.is_dir():
        raise FileNotFoundError(f"Bank directory not found: {bank}")

    results: list[tuple[int, int, Path]] = []
    for p in sorted(bank.glob(pattern)):
        m = _WAV_RE.match(p.name)
        if m:
            results.append((int(m.group(1)), int(m.group(2)), p))

    return sorted(results)


def load_wav_mono(
    path: Path,
    sr:   int,
    duration: float,
) -> Optional[torch.Tensor]:
    """
    Load WAV, resample if needed, convert to mono, crop to duration.
    Returns (N,) float32 tensor or None on error.
    """
    try:
        info = sf.info(str(path))
        audio, file_sr = sf.read(str(path), dtype='float32', always_2d=True)
    except Exception:
        return None

    # Mono downmix
    mono = audio.mean(axis=1)  # (N,)

    # Resample if needed (simple linear — good enough for reference loading)
    if file_sr != sr:
        try:
            import resampy
            mono = resampy.resample(mono, file_sr, sr)
        except ImportError:
            # Fallback: naive integer ratio or skip
            if file_sr % sr == 0:
                ratio = file_sr // sr
                mono = mono[::ratio]
            elif sr % file_sr == 0:
                ratio = sr // file_sr
                mono = np.repeat(mono, ratio)
            # else: use as-is (slight pitch shift — acceptable for loss comparison)

    # Crop to duration
    n = int(duration * sr)
    if len(mono) >= n:
        mono = mono[:n]
    else:
        # Zero-pad if WAV is shorter than requested duration
        pad = np.zeros(n - len(mono), dtype=np.float32)
        mono = np.concatenate([mono, pad])

    return torch.from_numpy(mono)


# ── Evaluation ────────────────────────────────────────────────────────────────

def render_checkpoint_samples(
    model:      InstrumentProfile,
    epoch:      int,
    args:       argparse.Namespace,
    log:        _Logger,
) -> None:
    """
    Render a small set of notes via proxy synth and save as WAV files.

    Output: <args.render_dir>/epoch-<N>/m<midi>_vel<vel>.wav
    Use these WAV files to audition how synthesis quality evolves during training.

    Note: proxy synth output is mono and simplified (no EQ, no stereo decorr).
    For high-quality stereo samples use IthacaRenderServer after training.
    """
    if not args.render_dir:
        return

    out_dir = Path(args.render_dir) / f"epoch-{epoch:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse --sample-notes "midi:vel,midi:vel,..."
    notes: list[tuple[int, int]] = []
    for token in args.sample_notes.split(','):
        token = token.strip()
        if ':' in token:
            m, v = token.split(':', 1)
            notes.append((int(m), int(v)))

    model.eval()
    with torch.no_grad():
        for midi, vel in notes:
            try:
                pred = render_note_differentiable(
                    model, midi, vel,
                    sr=args.sr,
                    duration=args.duration,
                    noise_level=args.noise_level,
                    target_rms=args.target_rms,
                    vel_gamma=args.vel_gamma,
                    k_max=args.k_max,
                )
                wav_path = out_dir / f"m{midi:03d}_vel{vel}.wav"
                sf.write(str(wav_path), pred.numpy(), args.sr, subtype='FLOAT')
                log.log(f"  sample -> {wav_path.name}")
            except Exception as e:
                log.log(f"  WARN sample m{midi:03d}v{vel}: {e}")


def evaluate(
    model:    InstrumentProfile,
    ref_notes: list[tuple[int, int, torch.Tensor]],
    args:     argparse.Namespace,
    log:      _Logger,
) -> float:
    """
    Compute MRSTFT for every reference note via proxy synth.
    Prints a per-note table and returns mean loss.
    """
    model.eval()
    losses: list[float] = []

    with torch.no_grad():
        for midi, vel, ref_wav in ref_notes:
            try:
                pred = render_note_differentiable(
                    model, midi, vel,
                    sr=args.sr,
                    duration=args.duration,
                    noise_level=args.noise_level,
                    target_rms=args.target_rms,
                    vel_gamma=args.vel_gamma,
                    k_max=args.k_max,
                )
                loss = mrstft(pred, ref_wav.to(pred.device)).item()
            except Exception as e:
                log.log(f"  WARN  m{midi:03d} vel{vel} failed: {e}")
                loss = float('nan')

            losses.append(loss)
            log.log(f"  m{midi:03d} vel{vel:1d}  MRSTFT={loss:.4f}")

    valid = [l for l in losses if not math.isnan(l)]
    mean  = float(np.mean(valid)) if valid else float('nan')
    log.log(f"  ── mean MRSTFT = {mean:.4f}  ({len(valid)}/{len(losses)} notes) ──")
    return mean


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(
    model:     InstrumentProfile,
    ref_notes: list[tuple[int, int, torch.Tensor]],
    args:      argparse.Namespace,
    meta:      dict,
    log:       _Logger,
) -> None:
    """
    Gradient-based NN weight update via torch proxy + MRSTFT.

    Training loop:
      1. Sample a mini-batch of reference notes.
      2. Render each via differentiable proxy.
      3. Compute MRSTFT vs reference mono WAV.
      4. Gradient-accumulate, step optimizer.
      5. Log progress; evaluate on full set every --eval-every epochs.
      6. Save updated model.
    """
    if not ref_notes:
        log.log("ERROR: no reference notes found — nothing to fine-tune")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    n_notes   = len(ref_notes)
    batch_sz  = min(args.batch_size, n_notes)
    rng       = random.Random(args.seed)

    log.log(f"\n── Fine-tune: {n_notes} reference notes, "
            f"batch={batch_sz}, epochs={args.epochs}, lr={args.lr} ──")

    # Initial evaluation
    log.log(f"\n[epoch 0 / {args.epochs}] initial evaluation:")
    mean0 = evaluate(model, ref_notes, args, log)
    if args.render_dir:
        render_checkpoint_samples(model, 0, args, log)

    best_loss = mean0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        # Shuffle notes for this epoch
        shuffled = list(ref_notes)
        rng.shuffle(shuffled)

        # Process in mini-batches
        n_steps = math.ceil(n_notes / batch_sz)
        step_loss_sum = 0.0

        for step in range(n_steps):
            batch = shuffled[step * batch_sz : (step + 1) * batch_sz]
            if not batch:
                continue

            optimizer.zero_grad()
            step_loss = torch.zeros((), dtype=torch.float32)
            n_valid   = 0

            for midi, vel, ref_wav in batch:
                try:
                    pred = render_note_differentiable(
                        model, midi, vel,
                        sr=args.sr,
                        duration=args.duration,
                        noise_level=args.noise_level,
                        target_rms=args.target_rms,
                        vel_gamma=args.vel_gamma,
                        k_max=args.k_max,
                        rng_seed=epoch,   # vary seed per epoch → different phase realizations
                    )
                    loss_i = mrstft(pred, ref_wav.to(pred.device))
                    # Accumulate gradient (scale by 1/batch to keep magnitude consistent)
                    (loss_i / len(batch)).backward()
                    step_loss = step_loss + loss_i.detach()
                    n_valid  += 1
                except Exception as e:
                    log.log(f"  WARN step {step} m{midi:03d}v{vel}: {e}")

            if n_valid > 0:
                # Gradient clipping prevents explosion through large (K,N) tensors
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                step_loss_sum += (step_loss / n_valid).item()

        scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss   = step_loss_sum / n_steps if n_steps else float('nan')
        epoch_loss = avg_loss

        if epoch % 10 == 0 or epoch == args.epochs:
            log.log(f"[epoch {epoch:4d}/{args.epochs}] "
                    f"loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"t={epoch_time:.1f}s")

        # Periodic full evaluation + optional WAV checkpoint
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            log.log(f"\n[epoch {epoch} eval]")
            mean = evaluate(model, ref_notes, args, log)
            if args.render_dir:
                render_checkpoint_samples(model, epoch, args, log)
            if mean < best_loss:
                best_loss  = mean
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                log.log(f"  ✓ new best: {best_loss:.4f}")

    # Restore best weights
    model.load_state_dict(best_state)
    log.log(f"\n── Restored best model (MRSTFT={best_loss:.4f}) ──")

    # Save
    out_path = args.out or args.model
    save_model(model, out_path, meta)
    log.log(f"Saved: {out_path}")


# ── Global SynthConfig optimisation loop ──────────────────────────────────────

def optimize_global(
    model:      InstrumentProfile,
    ref_notes:  list[tuple[int, int, torch.Tensor]],
    args:       argparse.Namespace,
    log:        _Logger,
) -> dict:
    """
    Optimise scalar SynthConfig parameters with NN weights frozen.

    Returns dict of best parameter values.
    Saves best config to --config-out JSON path.
    """
    if not ref_notes:
        log.log("ERROR: no reference notes found")
        return {}

    opt_params = [p.strip() for p in args.opt_params.split(',') if p.strip()]
    log.log(f"\n── Global SynthConfig optimisation ──")
    log.log(f"  opt_params: {opt_params}")
    log.log(f"  initial beat_scale={args.beat_scale:.3f}, "
            f"noise_level={args.noise_level:.3f}")

    cfg_params = SynthConfigParams(
        beat_scale=args.beat_scale,
        noise_level=args.noise_level,
        opt_params=opt_params,
    )

    lr = args.lr if args.lr != 3e-4 else 0.05   # sane default for global mode
    optimizer = torch.optim.Adam(cfg_params.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr * 0.01,
    )

    n_notes  = len(ref_notes)
    batch_sz = min(args.batch_size, n_notes)
    rng      = random.Random(args.seed)

    # NN weights frozen
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    best_loss  = float('inf')
    best_cfg   = cfg_params.to_dict()

    for epoch in range(1, args.epochs + 1):
        shuffled = list(ref_notes)
        rng.shuffle(shuffled)

        n_steps   = math.ceil(n_notes / batch_sz)
        step_loss_sum = 0.0

        for step in range(n_steps):
            batch = shuffled[step * batch_sz : (step + 1) * batch_sz]
            if not batch:
                continue

            optimizer.zero_grad()
            step_loss = torch.zeros(())
            n_valid   = 0

            for midi, vel, ref_wav in batch:
                try:
                    pred = render_note_differentiable(
                        model, midi, vel,
                        sr=args.sr,
                        duration=args.duration,
                        beat_scale=cfg_params.get_beat_scale(),
                        noise_level=cfg_params.get_noise_level(),
                        target_rms=args.target_rms,
                        vel_gamma=args.vel_gamma,
                        k_max=args.k_max,
                        rng_seed=epoch,
                    )
                    loss_i = mrstft(pred, ref_wav.to(pred.device))
                    (loss_i / len(batch)).backward()
                    step_loss = step_loss + loss_i.detach()
                    n_valid  += 1
                except Exception as e:
                    log.log(f"  WARN m{midi:03d}v{vel}: {e}")

            if n_valid > 0:
                nn.utils.clip_grad_norm_(cfg_params.parameters(), max_norm=2.0)
                optimizer.step()
                step_loss_sum += (step_loss / n_valid).item()

        scheduler.step()
        avg_loss = step_loss_sum / n_steps if n_steps else float('nan')

        if epoch % 10 == 0 or epoch == args.epochs:
            cur = cfg_params.to_dict()
            log.log(f"[epoch {epoch:4d}/{args.epochs}] "
                    f"loss={avg_loss:.4f}  "
                    f"beat_scale={cur['beat_scale']:.4f}  "
                    f"noise_level={cur['noise_level']:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.3e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_cfg  = cfg_params.to_dict()

    # Unfreeze NN (caller may need it)
    for p in model.parameters():
        p.requires_grad_(True)

    log.log(f"\n── Best SynthConfig (MRSTFT={best_loss:.4f}) ──")
    for k, v in best_cfg.items():
        log.log(f"  {k}: {v:.6f}")

    # Save JSON
    import json
    out_path = args.out or args.model
    cfg_out  = args.config_out or str(Path(out_path).with_suffix('.synth_config.json'))
    with open(cfg_out, 'w', encoding='utf-8') as f:
        json.dump(best_cfg, f, indent=2)
    log.log(f"Saved config: {cfg_out}")

    return best_cfg


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Closed-loop MRSTFT fine-tuning of InstrumentProfile NN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--mode',        required=True,    choices=['eval', 'finetune', 'global'])
    p.add_argument('--model',       required=True,    help='profile.pt path')
    p.add_argument('--out',         default=None,     help='output model path (default: overwrite --model)')
    p.add_argument('--bank',        required=True,    help='directory with original WAV files')
    p.add_argument('--wav-pattern', default='m*-vel*-f44.wav',
                   help='glob pattern for WAVs inside --bank (default: m*-vel*-f44.wav)')
    # Training
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--batch-size',  type=int,   default=8)
    p.add_argument('--eval-every',  type=int,   default=20,
                   help='run full eval every N epochs (default: 20)')
    # WAV sample checkpoints
    p.add_argument('--render-dir',   default=None,
                   help='save proxy WAV samples at each eval checkpoint (e.g. exports/finetune-samples)')
    p.add_argument('--sample-notes', default=_DEFAULT_SAMPLE_NOTES,
                   help='notes to render as WAV checkpoints, "midi:vel,..." '
                        f'(default: {_DEFAULT_SAMPLE_NOTES})')
    # Render / SynthConfig
    p.add_argument('--duration',    type=float, default=3.0,
                   help='proxy render + reference crop in seconds (default: 3.0)')
    p.add_argument('--sr',          type=int,   default=44_100)
    p.add_argument('--target-rms',  type=float, default=0.06)
    p.add_argument('--vel-gamma',   type=float, default=0.7)
    p.add_argument('--noise-level', type=float, default=1.0)
    p.add_argument('--beat-scale',  type=float, default=1.0)
    p.add_argument('--opt-params',  default='beat_scale,noise_level',
                   help='SynthConfig params to optimise in global mode '
                        '(default: beat_scale,noise_level)')
    p.add_argument('--config-out',  default=None,
                   help='JSON path for optimised SynthConfig '
                        '(default: <out>.synth_config.json)')
    p.add_argument('--k-max',       type=int,   default=60,
                   help='max partials in proxy (default: 60; use 30 to save memory)')
    # Misc
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--log',         default='analysis/runtime-logs/finetune.log')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-stamp render-dir so each run gets its own folder:
    #   exports/finetune-samples  →  exports/finetune-samples-20260331-0810
    if args.render_dir:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        args.render_dir = f"{args.render_dir}-{ts}"

    log  = _Logger(args.log)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    log.log(f"Loading model: {args.model}")
    model, meta = load_model(args.model)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.log(f"  hidden={meta['hidden']}, parameters={n_params:,}")

    # ── Discover reference WAVs ───────────────────────────────────────────────
    log.log(f"Scanning bank: {args.bank}  pattern={args.wav_pattern}")
    wav_list = find_reference_wavs(args.bank, args.wav_pattern)
    log.log(f"  Found {len(wav_list)} WAV files")

    if not wav_list:
        log.log("ERROR: no WAV files found — check --bank and --wav-pattern")
        sys.exit(1)

    # ── Load reference WAVs ───────────────────────────────────────────────────
    log.log(f"Loading references (duration={args.duration}s, sr={args.sr} Hz) …")
    ref_notes: list[tuple[int, int, torch.Tensor]] = []
    skipped = 0
    for midi, vel, path in wav_list:
        wav = load_wav_mono(path, args.sr, args.duration)
        if wav is None:
            skipped += 1
            continue
        ref_notes.append((midi, vel, wav))

    log.log(f"  Loaded {len(ref_notes)} notes"
            + (f" ({skipped} skipped)" if skipped else ""))

    # ── Run requested mode ────────────────────────────────────────────────────
    if args.mode == 'eval':
        log.log(f"\n── Eval mode ──")
        evaluate(model, ref_notes, args, log)

    elif args.mode == 'finetune':
        finetune(model, ref_notes, args, meta, log)

    elif args.mode == 'global':
        optimize_global(model, ref_notes, args, log)

    log.close()


if __name__ == '__main__':
    main()
