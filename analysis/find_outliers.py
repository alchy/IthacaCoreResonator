"""
analysis/find_outliers.py
──────────────────────────
Detects samples with extraction errors by comparing each sample's physics
parameters against a locally-smoothed trend over adjacent MIDI notes
(same velocity layer).

Parameters checked (all should vary smoothly across the keyboard):
  B           — inharmonicity coefficient
  tau1_mean   — mean of partial tau1 values (first 6 partials)
  A0_mean     — mean of partial A0 values   (first 6 partials)
  f0_ratio    — f0_fitted_hz / f0_nominal_hz  (should be ≈ 1.0)

Outlier criterion: residual from 5-note median smoother > Z_THRESH sigma
(computed per velocity layer using MAD-based robust sigma estimate).

Usage:
    python analysis/find_outliers.py --params analysis/params-ks-grand.json
    python analysis/find_outliers.py --params analysis/params-ks-grand.json --plot
    python analysis/find_outliers.py --params analysis/params-ks-grand.json --drop
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

Z_THRESH      = 3.5    # residual threshold in robust-sigma units
N_PARTIALS    = 6      # how many partials to average for tau1 / A0
SMOOTH_WINDOW = 5      # half-width of median smoother (notes each side)
VEL_LAYERS    = 8


# ── Load + parse ──────────────────────────────────────────────────────────────

def load_params(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_features(data: dict) -> dict[tuple[int,int], dict]:
    """Return {(midi, vel): feature_dict} for all samples."""
    out = {}
    for key, s in data['samples'].items():
        midi  = s['midi']
        vel   = s['vel']
        parts = s['partials'][:N_PARTIALS]
        if not parts:
            continue

        tau1_vals = [p['tau1'] for p in parts if p.get('tau1') is not None]
        A0_vals   = [p['A0']   for p in parts]
        f0_nom    = s.get('f0_nominal_hz', 0.0)
        f0_fit    = s.get('f0_fitted_hz', 0.0)

        out[(midi, vel)] = {
            'key':       key,
            'B':         s.get('B', 0.0),
            'tau1_mean': float(np.mean(tau1_vals)) if tau1_vals else 0.0,
            'A0_mean':   float(np.mean(A0_vals))   if A0_vals   else 0.0,
            'f0_ratio':  (f0_fit / f0_nom) if f0_nom > 0 else 1.0,
            'n_partials': s['n_partials'],
        }
    return out


# ── Robust statistics ─────────────────────────────────────────────────────────

def mad_sigma(x: np.ndarray) -> float:
    """Median absolute deviation → Gaussian-equivalent sigma."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12)


def median_smooth(vals: np.ndarray, hw: int = SMOOTH_WINDOW) -> np.ndarray:
    """Per-element median over a sliding window of width 2*hw+1."""
    n = len(vals)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        out[i] = np.median(vals[lo:hi])
    return out


# ── Outlier detection ─────────────────────────────────────────────────────────

FEATURES = ['B', 'tau1_mean', 'A0_mean', 'f0_ratio']


def detect_outliers(features: dict, z_thresh: float = Z_THRESH) -> list[dict]:
    """
    For each velocity layer, sort samples by MIDI, compute residuals from a
    median smoother, flag samples exceeding z_thresh robust sigmas.
    """
    # Group by velocity
    by_vel: dict[int, list] = defaultdict(list)
    for (midi, vel), feat in features.items():
        by_vel[vel].append((midi, feat))

    outliers = []

    for vel in sorted(by_vel):
        entries = sorted(by_vel[vel], key=lambda x: x[0])  # sort by MIDI
        if len(entries) < 4:
            continue

        midis = np.array([e[0] for e in entries])

        for feat_name in FEATURES:
            vals = np.array([e[1][feat_name] for e in entries])

            # log-space for B and A0 (span many decades)
            log_space = feat_name in ('B', 'A0_mean')
            if log_space:
                vals = np.log1p(vals)

            smoothed  = median_smooth(vals)
            residuals = vals - smoothed
            sigma     = mad_sigma(residuals)

            for i, (midi, feat) in enumerate(entries):
                z = abs(residuals[i]) / sigma
                if z > z_thresh:
                    outliers.append({
                        'key':      feat['key'],
                        'midi':     midi,
                        'vel':      vel,
                        'feature':  feat_name,
                        'value':    feat[feat_name],
                        'smoothed': float(np.expm1(smoothed[i])) if log_space else float(smoothed[i]),
                        'z_score':  float(z),
                    })

    # Sort by z_score descending, deduplicate by key (keep worst feature)
    seen: dict[str, dict] = {}
    for o in sorted(outliers, key=lambda x: -x['z_score']):
        k = o['key']
        if k not in seen or o['z_score'] > seen[k]['z_score']:
            seen[k] = o

    return sorted(seen.values(), key=lambda x: -x['z_score'])


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_outliers(features: dict, outlier_keys: set, feat_name: str = 'B') -> None:
    import matplotlib.pyplot as plt

    by_vel: dict[int, list] = defaultdict(list)
    for (midi, vel), feat in features.items():
        by_vel[vel].append((midi, feat[feat_name]))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    fig.suptitle(f'Parameter: {feat_name}  (outliers in red)', fontsize=13)

    for vel in range(VEL_LAYERS):
        ax = axes[vel // 4][vel % 4]
        entries = sorted(by_vel.get(vel, []))
        if not entries:
            ax.set_title(f'vel={vel} (no data)'); continue

        midis = [e[0] for e in entries]
        vals  = [e[1] for e in entries]

        for k, v in features.items():
            if k[1] == vel:
                pass

        colors = ['red' if features[(m, vel)]['key'] in outlier_keys else 'steelblue'
                  for m in midis]
        ax.scatter(midis, vals, c=colors, s=12, zorder=3)
        ax.plot(midis, vals, 'k-', lw=0.5, alpha=0.3, zorder=2)
        ax.set_title(f'vel={vel}')
        ax.set_xlabel('MIDI')

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Find extraction-error outliers in params JSON'
    )
    parser.add_argument('--params', default='analysis/params-ks-grand.json')
    parser.add_argument('--z',      type=float, default=Z_THRESH,
                        help='Z-score threshold (default: %(default)s)')
    parser.add_argument('--plot',   action='store_true',
                        help='Show scatter plots (requires matplotlib)')
    parser.add_argument('--drop',   action='store_true',
                        help='Remove outlier samples from params JSON in-place')
    parser.add_argument('--feature', default='B',
                        choices=FEATURES,
                        help='Feature to plot (with --plot)')
    args = parser.parse_args()

    data     = load_params(args.params)
    features = parse_features(data)
    outliers = detect_outliers(features, z_thresh=args.z)

    if not outliers:
        print("No outliers detected.")
    else:
        print(f"\n{'KEY':<18} {'MIDI':>4} {'VEL':>3}  {'FEATURE':<12} {'VALUE':>12}  {'EXPECTED':>12}  {'Z':>6}")
        print('-' * 80)
        for o in outliers:
            print(f"{o['key']:<18} {o['midi']:>4} {o['vel']:>3}  {o['feature']:<12} "
                  f"{o['value']:>12.4g}  {o['smoothed']:>12.4g}  {o['z_score']:>6.1f}s")
        print(f"\n{len(outliers)} outlier sample(s) found (z > {args.z}s)")

    if args.plot:
        outlier_keys = {o['key'] for o in outliers}
        plot_outliers(features, outlier_keys, feat_name=args.feature)

    if args.drop and outliers:
        outlier_keys = {o['key'] for o in outliers}
        for key in outlier_keys:
            del data['samples'][key]
        data['n_samples'] = len(data['samples'])
        with open(args.params, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nDropped {len(outlier_keys)} samples. Saved {args.params}")


if __name__ == '__main__':
    main()
