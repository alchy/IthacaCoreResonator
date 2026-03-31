"""
Generate comparison WAV files:
  tmp/python-proxy-m045-vel5-m44.wav   -- torch_synth.py reference (no noise for clean comparison)
  tmp/ICR-m045-vel5-m44.wav            -- C++ PianoCore algorithm replicated in numpy

Usage:
    python analysis/gen_comparison_wavs.py
"""
import json, math, sys, struct, array
from pathlib import Path

import numpy as np
import torch

_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.train_instrument_profile import (
    InstrumentProfile, midi_feat, vel_feat, k_feat, midi_to_hz,
)

MIDI      = 45
VEL       = 5
SR        = 44100
DURATION  = 3.0
RNG_SEED  = 0
PARAMS    = "analysis/params-piano-ft-ks-grand.json"
MODEL     = "analysis/profile-finetuned.pt"
OUT_DIR   = Path("tmp")
_K_FEAT   = 90


# ── WAV writer (no scipy dependency) ─────────────────────────────────────────

def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write float32 mono audio as 16-bit PCM WAV."""
    # Normalise to [-1,1] if needed, then scale to int16
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        print(f"  WARNING: clipping at peak={peak:.3f}, normalising for WAV")
        audio = audio / peak
    pcm = (audio * 32767).astype(np.int16)
    n_samples = len(pcm)
    with open(path, "wb") as f:
        # RIFF header
        data_size = n_samples * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH",
            16,       # chunk size
            1,        # PCM
            1,        # mono
            sr,       # sample rate
            sr * 2,   # byte rate
            2,        # block align
            16,       # bits per sample
        ))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm.tobytes())
    print(f"  Wrote {path}  ({n_samples} samples, peak={peak:.4f})")


# ── Python proxy render ───────────────────────────────────────────────────────

def render_python_proxy(model, midi, vel, sr, duration, rng_seed,
                         with_noise: bool = True) -> np.ndarray:
    """Exact torch_synth.py algorithm."""
    device = next(model.parameters()).device
    mf = midi_feat(midi).to(device)
    vf = vel_feat(vel).to(device)
    f0 = midi_to_hz(midi)

    n_max_nyquist = max(1, int(sr / 2 / f0))
    K = min(60, n_max_nyquist)
    N = int(duration * sr)
    t = torch.arange(N, dtype=torch.float32, device=device) / sr

    kf_b   = torch.stack([k_feat(k, _K_FEAT) for k in range(1, K+1)]).to(device)
    k_vals = torch.arange(1, K+1, dtype=torch.float32, device=device)
    mf_b   = mf.unsqueeze(0).expand(K, -1)
    vf_b   = vf.unsqueeze(0).expand(K, -1)

    B     = torch.exp(model.forward_B(mf).clamp(max=0.0)).squeeze()
    f_hzs = k_vals * f0 * torch.sqrt(1.0 + B * k_vals**2)
    valid = ((f_hzs < sr * 0.495) & torch.isfinite(f_hzs)).float().unsqueeze(1)

    log_df  = model.forward_df(mf_b, kf_b).squeeze(-1)
    beat_hz = torch.exp(log_df).clamp(min=1e-4)

    tau1_k1    = torch.exp(model.forward_tau1_k1(mf, vf)).squeeze()
    log_ratios = model.tau_ratio_net(torch.cat([mf_b, kf_b], -1)).squeeze(-1)
    log_k_bias = -0.3 * torch.log(k_vals)
    log_ratios = torch.minimum(log_ratios, torch.zeros_like(log_ratios))
    log_ratios = torch.maximum(log_ratios, log_k_bias - 2.0)
    tau1s = (tau1_k1 * torch.exp(log_ratios)).clamp(min=0.005)

    A0s = torch.exp(model.A0_net(
        torch.cat([mf_b, kf_b, vf_b], -1)).squeeze(-1)).clamp(min=1e-6)

    biexp       = model.biexp_net(torch.cat([mf_b, kf_b, vf_b], -1))
    a1s         = torch.sigmoid(biexp[:, 0]).clamp(0.05, 0.99)
    tau2s       = tau1s * torch.exp(biexp[:, 1]).clamp(min=3.0)

    noise_pred = model.forward_noise(mf, vf).squeeze()
    attack_tau = torch.exp(noise_pred[0]).clamp(0.002, 1.0).item()
    A_noise    = torch.exp(noise_pred[2]).clamp(0.001, 0.5).item()
    phi_diff   = model.forward_phi(mf, vf).squeeze().item()

    seed = rng_seed + midi * 256 + vel
    gen  = torch.Generator()
    gen.manual_seed(seed)
    phis      = torch.rand(K, generator=gen).mul(2 * math.pi)
    noise_raw = torch.randn(N, generator=gen)

    half_beat  = beat_hz.unsqueeze(1) * 0.5
    phase_base = 2.0 * math.pi * t.unsqueeze(0) * f_hzs.unsqueeze(1)
    beat_phase = 2.0 * math.pi * t.unsqueeze(0) * half_beat
    phi_init   = phis.unsqueeze(1)

    osc1 = torch.cos(phase_base + beat_phase + phi_init)
    osc2 = torch.cos(phase_base - beat_phase + phi_init + phi_diff)
    oscs = (osc1 + osc2) * 0.5

    env_fast = torch.exp(-t.unsqueeze(0) / tau1s.unsqueeze(1))
    env_slow = torch.exp(-t.unsqueeze(0) / tau2s.unsqueeze(1))
    envs     = a1s.unsqueeze(1) * env_fast + (1.0 - a1s).unsqueeze(1) * env_slow

    audio = (A0s.unsqueeze(1) * envs * oscs * valid).sum(0)

    if with_noise:
        noise_env = torch.exp(-t / max(attack_tau, 1e-4))
        audio = audio + A_noise * noise_raw * noise_env

    vel_gain = ((vel + 1) / 8.0) ** 0.7
    rms      = torch.sqrt(audio.pow(2).mean() + 1e-10)
    audio    = audio * (0.06 * vel_gain / rms)

    return audio.cpu().numpy().astype(np.float32)


# ── C++ algorithm (numpy replication) ────────────────────────────────────────

def render_cpp_algorithm(note: dict, sr: int, duration: float,
                          with_noise: bool = False,
                          rng_seed: int = 0) -> np.ndarray:
    """
    Replicate piano_core.cpp processBlock() exactly.
    Uses float32 arithmetic.
    with_noise=False: skip noise (C++ noise uses independent mt19937)
    """
    N      = int(duration * sr)
    inv_sr = np.float32(1.0) / np.float32(sr)
    t_idx  = np.arange(N, dtype=np.float32)
    t_f    = t_idx * inv_sr
    TAU    = np.float32(2.0 * math.pi)
    tpi2   = TAU * t_f

    phi_diff = np.float32(note["phi_diff"])
    rms_gain = np.float32(note["rms_gain"])
    audio    = np.zeros(N, dtype=np.float32)

    for p in note["partials"]:
        tau1    = np.float32(p["tau1"])
        tau2    = np.float32(p["tau2"])
        a1      = np.float32(p["a1"])
        A0      = np.float32(p["A0"])
        f_hz    = np.float32(p["f_hz"])
        beat_hz = np.float32(p["beat_hz"])
        phi     = np.float32(p["phi"])

        # Multiplicative decay — matches C++ initVoice
        df = np.exp(np.float32(-1.0) / np.maximum(tau1 * np.float32(sr), np.float32(1.0)))
        ds = np.exp(np.float32(-1.0) / np.maximum(tau2 * np.float32(sr), np.float32(1.0)))
        env_fast = np.power(df, t_idx, dtype=np.float32)
        env_slow = np.power(ds, t_idx, dtype=np.float32)
        env      = a1 * env_fast + (np.float32(1.0) - a1) * env_slow

        # Phase — matches C++ processBlock
        phase_c = tpi2 * f_hz + phi
        phase_b = tpi2 * (beat_hz * np.float32(0.5))

        s1 = np.cos(phase_c + phase_b).astype(np.float32)
        s2 = np.cos(phase_c - phase_b + phi_diff).astype(np.float32)

        audio += (A0 * rms_gain) * env * (s1 + s2) * np.float32(0.5)

    return audio


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyse(ref: np.ndarray, cpp: np.ndarray, sr: int) -> dict:
    n     = min(len(ref), len(cpp))
    ref   = ref[:n];  cpp = cpp[:n]
    diff  = cpp - ref
    rms_r = float(np.sqrt(np.mean(ref**2)))
    rms_c = float(np.sqrt(np.mean(cpp**2)))
    rms_d = float(np.sqrt(np.mean(diff**2)))
    snr   = 20 * np.log10(max(rms_r, 1e-12) / max(rms_d, 1e-12))

    print(f"\n  RMS ref  = {rms_r:.6f}   peak={np.max(np.abs(ref)):.4f}")
    print(f"  RMS cpp  = {rms_c:.6f}   peak={np.max(np.abs(cpp)):.4f}")
    print(f"  RMS diff = {rms_d:.6f}")
    print(f"  SNR      = {snr:.1f} dB", end="  ")
    if   snr > 60: print("PASS (float32 rounding)")
    elif snr > 40: print("CLOSE (minor discrepancy)")
    else:          print("MISMATCH - bug present!")

    # Early vs late
    sp = int(0.1 * sr)
    print(f"  diff RMS  0..100ms = {np.sqrt(np.mean(diff[:sp]**2)):.6f}")
    print(f"  diff RMS 100ms..end= {np.sqrt(np.mean(diff[sp:]**2)):.6f}")
    return {"snr_db": snr, "rms_ref": rms_r, "rms_cpp": rms_c, "rms_diff": rms_d}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL}")
    ckpt  = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = InstrumentProfile(hidden=ckpt.get("hidden", 64))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Loading params: {PARAMS}")
    with open(PARAMS) as f:
        pdata = json.load(f)
    export_sr = pdata.get("sr", 44100)
    key       = f"m{MIDI:03d}_vel{VEL}"
    note      = pdata["notes"][key]
    print(f"  Note {key}: K={note['K_valid']} partials  export_sr={export_sr}")

    OUT_DIR.mkdir(exist_ok=True)

    print(f"\n[1] Python proxy (with noise, SR={SR}) ...")
    with torch.no_grad():
        ref_audio = render_python_proxy(model, MIDI, VEL, SR, DURATION, RNG_SEED,
                                         with_noise=True)
    write_wav(OUT_DIR / f"python-proxy-m{MIDI:03d}-vel{VEL}-m44.wav", ref_audio, SR)

    print(f"\n[2] C++ algorithm replica (no noise, SR={SR}) ...")
    cpp_audio = render_cpp_algorithm(note, SR, DURATION, with_noise=False)
    write_wav(OUT_DIR / f"ICR-m{MIDI:03d}-vel{VEL}-m44.wav", cpp_audio, SR)

    print(f"\n[3] Reference without noise (apples-to-apples) ...")
    with torch.no_grad():
        ref_no_noise = render_python_proxy(model, MIDI, VEL, SR, DURATION, RNG_SEED,
                                            with_noise=False)

    print("\n-- Analysis (no noise, both) --")
    results = analyse(ref_no_noise, cpp_audio, SR)

    # Partial-level diagnostics if SNR is bad
    if results["snr_db"] < 40:
        print("\n-- Partial-level diagnosis --")
        # Compare first partial alone
        p0       = note["partials"][0]
        phi_diff = float(note["phi_diff"])
        N        = int(DURATION * SR)
        t        = np.arange(N, dtype=np.float32) / SR
        TAU      = float(2.0 * math.pi)

        # Reference partial 0 (from Python, no beat for simplicity)
        phase_r  = TAU * float(p0["f_hz"]) * t + float(p0["phi"])
        # C++ partial 0
        inv_sr   = np.float32(1.0) / np.float32(SR)
        t_cpp    = np.arange(N, dtype=np.float32) * inv_sr
        phase_c  = np.float32(TAU) * np.float32(p0["f_hz"]) * t_cpp + np.float32(p0["phi"])

        cos_r = np.cos(phase_r).astype(np.float32)
        cos_c = np.cos(phase_c).astype(np.float32)
        pdiff = cos_c - cos_r
        print(f"  Partial 0 carrier cos() max diff: {np.max(np.abs(pdiff)):.2e}")
        print(f"  at t=0: ref={cos_r[0]:.8f}  cpp={cos_c[0]:.8f}")
        print(f"  at t=1s: ref={cos_r[SR]:.8f}  cpp={cos_c[SR]:.8f}")
        print(f"  at t=3s: ref={cos_r[-1]:.8f}  cpp={cos_c[-1]:.8f}")

        # Envelope comparison
        df = float(np.exp(-1.0 / max(float(p0["tau1"]) * SR, 1.0)))
        ef_cpp = np.power(df, np.arange(N, dtype=np.float32))
        ef_ref = np.exp(-t / float(p0["tau1"]))
        ediff  = ef_cpp - ef_ref.astype(np.float32)
        print(f"  Partial 0 env_fast max diff: {np.max(np.abs(ediff)):.2e}")


if __name__ == "__main__":
    main()
