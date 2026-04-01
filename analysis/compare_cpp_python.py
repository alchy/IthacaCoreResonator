"""
analysis/compare_cpp_python.py
───────────────────────────────
Numerická komparace výstupu C++ RenderServeru a Python physics_synth.py.

Cíl: kvantifikovat divergenci mezi oběma syntézními cestami a sledovat
     pokrok refaktoringu. Výstup je report s číselnými metrikami a
     volitelnými spektrogramy.

Metriky:
  - MRSTFT loss (multi-resolution STFT, stejná funkce jako v training loop)
  - MSE a SNR na mono sumě
  - RMS porovnání (level matching check)
  - Korelační koeficient (R²) — 1.0 = identické signály

Použití:
    # Rychlý test: MIDI 60, vel 3 (mezzo, A440)
    python -m analysis.compare_cpp_python

    # Konkrétní nota, spektrogram uložit:
    python -m analysis.compare_cpp_python --midi 60 --vel 3 --plot --save

    # Dávkový test (vybere reprezentativní noty přes celý klavír):
    python -m analysis.compare_cpp_python --batch

    # Dávkový test s regresní kontrolou (exitcode 1 při regresi):
    python -m analysis.compare_cpp_python --batch --check

    # Uloží aktuální výsledky jako nový baseline (přidá margin 0.3):
    python -m analysis.compare_cpp_python --batch --save-baseline
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# ── Projekt v sys.path ────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.physics_synth import synthesize_note, load_synth_config, synth_config_to_kwargs
from analysis.render_client  import RenderClient, RenderClientError
from analysis.mrstft_loss    import mrstft_numpy


# ── Výchozí cesty ─────────────────────────────────────────────────────────────
DEFAULT_PARAMS   = str(_ROOT / "soundbanks" / "params-ks-grand-ft.json")
DEFAULT_SERVER   = str(_ROOT / "build" / "bin" / "Release" / "IthacaRenderServer.exe")
DEFAULT_CONFIG   = str(_ROOT / "analysis" / "synth_config.json")
EXPORT_DIR       = _ROOT / "analysis" / "compare_exports"
BASELINE_PATH    = _ROOT / "analysis" / "regression_baseline.json"

# ── Regresní baseline — konzervativní horní meze MRSTFT ───────────────────────
# Odvozeno z Phase 1 verifikace (params-ks-grand-ft.json, 2026-04-01) + margin 0.3.
# Noise-dominated noty (A0, C4 forte, C6, C8) mají vyšší práh odpovídající
# stochastickému flooru + typické noise divergenci.
# Po KI-1 fix (offline_mode noise) by noise noty měly mít nižší MRSTFT —
# spusť --batch --save-baseline pro aktualizaci po potvrzení zlepšení.
_HARDCODED_BASELINE = {
    "m021_vel3": 3.5,   # A0  — noise dominated (pre-fix: 2.36)
    "m036_vel3": 1.8,   # C2  — at stochastic floor (pre-fix: 1.09)
    "m048_vel3": 1.8,   # C3  — at stochastic floor (pre-fix: 1.30)
    "m060_vel3": 1.8,   # C4  — at stochastic floor (pre-fix: 1.27)
    "m060_vel6": 3.5,   # C4 forte — noise dominated (pre-fix: 2.35)
    "m072_vel3": 1.8,   # C5  — at stochastic floor (pre-fix: 1.21)
    "m084_vel3": 2.8,   # C6  — noise dominated (pre-fix: 1.80)
    "m096_vel3": 1.8,   # C7  — at stochastic floor (pre-fix: 1.41)
    "m108_vel3": 5.0,   # C8  — noise dominated (pre-fix: 4.32)
}


def load_baseline() -> dict:
    """Načte baseline z JSON souboru, nebo vrátí hardcoded výchozí hodnoty."""
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            data = json.load(f)
        return data.get("targets", _HARDCODED_BASELINE)
    return _HARDCODED_BASELINE


def save_baseline(results: list, margin: float = 0.3) -> None:
    """Uloží aktuální výsledky jako nový baseline (target = mrstft + margin)."""
    targets = {r["label"]: round(r["mrstft"] + margin, 4) for r in results}
    payload = {
        "updated":  __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "margin":   margin,
        "targets":  targets,
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Baseline uložen: {BASELINE_PATH}")


# ── Reprezentativní noty pro dávkový test ─────────────────────────────────────
BATCH_NOTES = [
    (21, 3),   # A0  — nejhlubší nota, 1 struna
    (36, 3),   # C2  — 2 struny, hluboký bas
    (48, 3),   # C3  — hranice 2/3 strun
    (60, 3),   # C4  — střední C, mezzo
    (60, 6),   # C4  — forte
    (72, 3),   # C5  — výška
    (84, 3),   # C6  — vysoká poloha
    (96, 3),   # C7  — velmi vysoká
    (108, 3),  # C8  — nejvyšší nota
]


# ── Metrika: MRSTFT + pomocné ─────────────────────────────────────────────────

def compute_metrics(cpp_audio: np.ndarray, py_audio: np.ndarray,
                    sr: int, label: str = "") -> dict:
    """
    Vypočítá sadu metrik mezi C++ a Python výstupem.

    Oba vstupy musí být (N,2) float32 stereo pole.
    Délka je zkrácena na kratší z obou (offline renderer může vygenerovat
    mírně jiný počet vzorků kvůli silence detection).
    """
    n = min(len(cpp_audio), len(py_audio))
    cpp = cpp_audio[:n]
    py  = py_audio[:n]

    # Mono sumy pro skalární metriky
    cpp_mono = cpp.mean(axis=1)
    py_mono  = py.mean(axis=1)

    # MRSTFT (stejná funkce jako v closed_loop_finetune.py)
    mrstft = mrstft_numpy(cpp_mono, py_mono)

    # MSE a SNR na mono
    mse  = float(np.mean((cpp_mono - py_mono) ** 2))
    rms_diff = float(math.sqrt(mse))
    py_rms   = float(math.sqrt(np.mean(py_mono ** 2)))
    snr_db   = float(20 * math.log10(py_rms / (rms_diff + 1e-10)))

    # Korelace
    corr = float(np.corrcoef(cpp_mono, py_mono)[0, 1]) if len(cpp_mono) > 1 else 0.0

    # RMS úrovně
    cpp_rms_stereo = float(math.sqrt((np.mean(cpp ** 2))))
    py_rms_stereo  = float(math.sqrt((np.mean(py  ** 2))))
    rms_ratio_db   = float(20 * math.log10(cpp_rms_stereo / (py_rms_stereo + 1e-10)))

    # Attack divergence: MRSTFT jen na prvních 100 ms (attack window)
    attack_n = min(int(0.1 * sr), n)
    mrstft_attack = mrstft_numpy(cpp_mono[:attack_n], py_mono[:attack_n])

    return {
        "label":         label,
        "n_samples":     n,
        "duration_s":    n / sr,
        "mrstft":        round(mrstft, 4),
        "mrstft_attack": round(mrstft_attack, 4),
        "mse":           round(mse, 8),
        "snr_db":        round(snr_db, 2),
        "corr_r2":       round(corr ** 2, 6),
        "cpp_rms_db":    round(20 * math.log10(cpp_rms_stereo + 1e-10), 2),
        "py_rms_db":     round(20 * math.log10(py_rms_stereo  + 1e-10), 2),
        "rms_diff_db":   round(rms_ratio_db, 2),
    }


# ── Python syntéza ─────────────────────────────────────────────────────────────

def synthesize_python(params_path: str, midi: int, vel: int,
                      duration: float = 3.0, sr: int = 44100,
                      config_kwargs: dict = None) -> np.ndarray:
    """Syntéza jedné noty přes Python physics_synth.py.

    Aplikuje vel_gain na target_rms — shoduje se s training loop
    (physics_synth.py normalize_dataset, řádek ~537):
        vel_rms = target_rms * ((vel+1)/8)^vel_gamma
    C++ offline_renderer dělá totéž; bez tohoto by srovnání ukazovalo
    systematický -4.2 dB offset na každé notě vel=3.
    """
    with open(params_path) as f:
        data = json.load(f)

    key = f"m{midi:03d}_vel{vel}"
    samples = data.get("samples", {})
    if key not in samples:
        raise ValueError(f"Klíč {key} nenalezen v {params_path}. "
                         f"Dostupné klíče (prvních 5): {list(samples.keys())[:5]}")

    kwargs = dict(config_kwargs or {})
    # vel_gain není parametr synthesize_note — zpracujeme ho zde
    vel_gamma = kwargs.pop("vel_gamma", 0.7)
    base_rms  = kwargs.get("target_rms", 0.06)
    vel_gain  = ((vel + 1) / 8.0) ** vel_gamma
    kwargs["target_rms"] = base_rms * vel_gain

    audio = synthesize_note(samples[key], duration=duration, sr=sr, **kwargs)
    return audio  # (N, 2) float32


# ── C++ syntéza přes RenderServer ─────────────────────────────────────────────

def synthesize_cpp(params_path: str, server_exe: str,
                   midi: int, vel: int,
                   duration: float = 3.0, sr: int = 44100,
                   config_kwargs: dict = None) -> np.ndarray:
    """Syntéza jedné noty přes C++ IthacaRenderServer."""
    with RenderClient(server_exe=server_exe, params_json=params_path) as rc:
        # Přenést config parametry na server
        if config_kwargs:
            cpp_cfg = {}
            key_map = {
                "beat_scale":         "beat_scale",
                "eq_strength":        "eq_strength",
                "noise_level":        "noise_level",
                "stereo_boost":       "stereo_boost",
                "stereo_decorr":      "stereo_decorr",
                "pan_spread":         "pan_spread",
                "harmonic_brightness":"harmonic_brightness",
                "target_rms":         "target_rms",
                "onset_ms":           "onset_ms",
            }
            for py_key, cpp_key in key_map.items():
                if py_key in config_kwargs:
                    cpp_cfg[cpp_key] = config_kwargs[py_key]
            if cpp_cfg:
                rc.set_config(**cpp_cfg)

        tmp_path = str(EXPORT_DIR / f"_cpp_m{midi:03d}_vel{vel}.wav")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        frames = rc.render(midi=midi, vel=vel, duration=duration,
                           sr=sr, output=tmp_path)

    audio, file_sr = sf.read(tmp_path, dtype='float32', always_2d=True)
    if file_sr != sr:
        raise RuntimeError(f"Server vrátil sr={file_sr}, očekáváno {sr}")
    return audio  # (N, 2) float32


# ── Spektrogram diff (volitelný) ───────────────────────────────────────────────

def plot_comparison(cpp_audio: np.ndarray, py_audio: np.ndarray,
                    sr: int, midi: int, vel: int, save_path: Path = None):
    """Vykreslí spektrogramy C++, Python a rozdíl."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("  [!] matplotlib není dostupný — přeskočuji plot")
        return

    n = min(len(cpp_audio), len(py_audio))
    cpp_mono = cpp_audio[:n].mean(axis=1)
    py_mono  = py_audio[:n].mean(axis=1)

    # STFT parametry: n_fft=2048, hop=512 (≈23ms při 44100)
    from scipy.signal import spectrogram as scipy_spec
    f_cpp, t_cpp, S_cpp = scipy_spec(cpp_mono, sr, nperseg=2048, noverlap=1536,
                                     scaling='spectrum')
    f_py,  t_py,  S_py  = scipy_spec(py_mono,  sr, nperseg=2048, noverlap=1536,
                                     scaling='spectrum')

    S_cpp_db = 10 * np.log10(S_cpp + 1e-10)
    S_py_db  = 10 * np.log10(S_py  + 1e-10)
    S_diff   = S_cpp_db - S_py_db

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"C++ vs Python — MIDI {midi} vel {vel} (sr={sr})", fontsize=13)

    vmin, vmax = -80, 0
    dmax = max(abs(S_diff).max(), 1.0)

    axes[0].pcolormesh(t_cpp, f_cpp, S_cpp_db, vmin=vmin, vmax=vmax, cmap='magma')
    axes[0].set_title("C++ RenderServer"); axes[0].set_ylabel("Hz")

    axes[1].pcolormesh(t_py,  f_py,  S_py_db,  vmin=vmin, vmax=vmax, cmap='magma')
    axes[1].set_title("Python physics_synth"); axes[1].set_ylabel("Hz")

    im = axes[2].pcolormesh(t_cpp, f_cpp, S_diff, vmin=-dmax, vmax=dmax, cmap='RdBu_r')
    axes[2].set_title(f"Diff [C++ − Python] dB  (max ±{dmax:.1f} dB)")
    axes[2].set_ylabel("Hz"); axes[2].set_xlabel("s")
    fig.colorbar(im, ax=axes[2], label="dB")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Spektrogram uložen: {save_path}")
    else:
        plt.show()
    plt.close()


# ── Formátování výsledku ───────────────────────────────────────────────────────

def print_metrics(m: dict):
    qual = "DOBRY" if m["mrstft"] < 0.5 else ("OK" if m["mrstft"] < 1.0 else "SPATNY")
    print(f"  MRSTFT={m['mrstft']:.4f} [{qual}]  "
          f"MRSTFT_attack={m['mrstft_attack']:.4f}  "
          f"SNR={m['snr_db']:.1f}dB  "
          f"R²={m['corr_r2']:.4f}  "
          f"RMS_diff={m['rms_diff_db']:+.1f}dB  "
          f"dur={m['duration_s']:.2f}s")


# ── Hlavní logika ─────────────────────────────────────────────────────────────

def run_single(args):
    midi     = args.midi
    vel      = args.vel
    duration = args.duration
    sr       = args.sr

    print(f"\n{'='*60}")
    print(f"  Porovnání: MIDI {midi} vel {vel}  dur={duration}s  sr={sr}")
    print(f"{'='*60}")

    # Načíst config
    config_kwargs = {}
    if Path(DEFAULT_CONFIG).exists():
        config_kwargs = synth_config_to_kwargs(load_synth_config(DEFAULT_CONFIG))
    config_kwargs.setdefault("target_rms", 0.06)
    config_kwargs.setdefault("onset_ms",   3.0)

    # Python syntéza
    print("  [1/2] Python physics_synth... ", end="", flush=True)
    t0 = time.time()
    try:
        py_audio = synthesize_python(args.params, midi, vel, duration, sr, config_kwargs)
        print(f"OK ({time.time()-t0:.2f}s, {len(py_audio)} vzorků)")
    except Exception as e:
        print(f"CHYBA: {e}")
        return None

    # C++ syntéza
    print("  [2/2] C++ RenderServer... ", end="", flush=True)
    t0 = time.time()
    try:
        cpp_audio = synthesize_cpp(args.params, args.server, midi, vel,
                                   duration, sr, config_kwargs)
        print(f"OK ({time.time()-t0:.2f}s, {len(cpp_audio)} vzorků)")
    except RenderClientError as e:
        print(f"CHYBA (server): {e}")
        print("  [!] Je server zkompilovaný? Zkuste: cmake --build build --config Release")
        return None
    except Exception as e:
        print(f"CHYBA: {e}")
        return None

    # Metriky
    label = f"m{midi:03d}_vel{vel}"
    m = compute_metrics(cpp_audio, py_audio, sr, label)
    print_metrics(m)

    # Uložit WAV pro auditívní porovnání
    if args.save:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(str(EXPORT_DIR / f"py_{label}.wav"),   py_audio,  sr)
        sf.write(str(EXPORT_DIR / f"cpp_{label}.wav"),  cpp_audio, sr)
        print(f"  WAV uloženy do {EXPORT_DIR}/")

    # Spektrogram
    if args.plot:
        plot_path = (EXPORT_DIR / f"spec_{label}.png") if args.save else None
        plot_comparison(cpp_audio, py_audio, sr, midi, vel, plot_path)

    return m


def run_batch(args):
    print(f"\n{'='*60}")
    print(f"  DÁVKOVÝ TEST — {len(BATCH_NOTES)} not")
    print(f"  Parametry: {args.params}")
    print(f"{'='*60}")

    config_kwargs = {}
    if Path(DEFAULT_CONFIG).exists():
        config_kwargs = synth_config_to_kwargs(load_synth_config(DEFAULT_CONFIG))
    config_kwargs.setdefault("target_rms", 0.06)
    config_kwargs.setdefault("onset_ms",   3.0)

    results = []
    try:
        with RenderClient(server_exe=args.server, params_json=args.params) as rc:
            if config_kwargs:
                rc.set_config(**{k: v for k, v in config_kwargs.items()
                                 if k in ("beat_scale", "eq_strength", "noise_level",
                                          "stereo_boost", "stereo_decorr", "pan_spread",
                                          "harmonic_brightness", "target_rms", "onset_ms")})

            for midi, vel in BATCH_NOTES:
                label = f"m{midi:03d}_vel{vel}"
                print(f"\n  MIDI {midi:3d} vel {vel}  ({label})")

                # Python syntéza — vel_gain logika centralizovaná v synthesize_python()
                try:
                    py_audio = synthesize_python(args.params, midi, vel,
                                                 args.duration, args.sr, config_kwargs)
                except ValueError as e:
                    print(f"    [!] Přeskočeno — {e}")
                    continue
                except Exception as e:
                    print(f"    [!] Python chyba: {e}")
                    continue

                # C++
                try:
                    tmp = str(EXPORT_DIR / f"_batch_m{midi:03d}_vel{vel}.wav")
                    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
                    rc.render(midi=midi, vel=vel, duration=args.duration,
                              sr=args.sr, output=tmp)
                    cpp_audio, _ = sf.read(tmp, dtype='float32', always_2d=True)
                except Exception as e:
                    print(f"    [!] C++ chyba: {e}")
                    continue

                m = compute_metrics(cpp_audio, py_audio, args.sr, label)
                print_metrics(m)
                results.append(m)

    except RenderClientError as e:
        print(f"\n  [!] Server nelze spustit: {e}")
        print("  Zkompilujte: cmake --build build --config Release")
        return

    # Souhrnná statistika
    if results:
        mrstft_vals = [r["mrstft"] for r in results]
        print(f"\n{'='*60}")
        print(f"  SOUHRN  ({len(results)} not)")
        print(f"  MRSTFT  avg={np.mean(mrstft_vals):.4f}  "
              f"min={np.min(mrstft_vals):.4f}  "
              f"max={np.max(mrstft_vals):.4f}")
        worst = max(results, key=lambda r: r["mrstft"])
        best  = min(results, key=lambda r: r["mrstft"])
        print(f"  Nejhorší: {worst['label']} MRSTFT={worst['mrstft']:.4f}")
        print(f"  Nejlepší: {best['label']}  MRSTFT={best['mrstft']:.4f}")

        # Uložit CSV
        if args.save:
            import csv
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            csv_path = EXPORT_DIR / "batch_results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"  CSV uložen: {csv_path}")

        # Regresní kontrola
        if args.check or args.save_baseline:
            baseline = load_baseline()
            print(f"\n{'='*60}")
            print(f"  REGRESNÍ KONTROLA  (baseline: "
                  f"{'soubor' if BASELINE_PATH.exists() else 'hardcoded'})")
            failures = []
            for r in results:
                label  = r["label"]
                target = baseline.get(label)
                if target is None:
                    print(f"  {label:16s}  MRSTFT={r['mrstft']:.4f}  [bez cíle]")
                    continue
                status = "PASS" if r["mrstft"] <= target else "FAIL"
                marker = "" if status == "PASS" else f"  >> target={target:.4f}"
                print(f"  {label:16s}  MRSTFT={r['mrstft']:.4f}  [{status}]{marker}")
                if status == "FAIL":
                    failures.append((label, r["mrstft"], target))
            if failures:
                print(f"\n  [FAIL] REGRESE na {len(failures)} notach:")
                for lbl, got, exp in failures:
                    print(f"    {lbl}: {got:.4f} > {exp:.4f}  (o {got-exp:+.4f})")
            else:
                print(f"\n  [PASS] Vsechny noty splnuji baseline.")

            if args.save_baseline:
                save_baseline(results)

            if args.check and failures:
                raise SystemExit(1)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="C++ vs Python syntéza — numerická komparace")
    p.add_argument("--midi",     type=int,   default=60,           help="MIDI nota (default: 60 = C4)")
    p.add_argument("--vel",      type=int,   default=3,            help="Velocity band 0-7 (default: 3)")
    p.add_argument("--duration", type=float, default=3.0,          help="Délka v sekundách (default: 3.0)")
    p.add_argument("--sr",       type=int,   default=44100,        help="Sample rate (default: 44100)")
    p.add_argument("--params",   type=str,   default=DEFAULT_PARAMS, help="Cesta k params JSON")
    p.add_argument("--server",   type=str,   default=DEFAULT_SERVER, help="Cesta k IthacaRenderServer.exe")
    p.add_argument("--plot",          action="store_true", help="Zobrazit spektrogram")
    p.add_argument("--save",          action="store_true", help="Uložit WAV a grafy")
    p.add_argument("--batch",         action="store_true", help="Dávkový test všech reprezentativních not")
    p.add_argument("--check",         action="store_true", help="Regresní kontrola vs baseline (exitcode 1 při regresi)")
    p.add_argument("--save-baseline", action="store_true", help="Uloží aktuální výsledky jako nový baseline (+0.3 margin)",
                   dest="save_baseline")
    args = p.parse_args()

    if args.batch:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
