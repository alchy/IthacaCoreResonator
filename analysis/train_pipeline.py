"""
analysis/train_pipeline.py
───────────────────────────
Wrapper pro celý tréninkový pipeline IthacaCoreResonator.

Spustí kroky v pořadí:
  1. extract_params.py       — fyzikální parametry z WAV banky
  2. find_outliers.py        — odstranění chybně extrahovaných samplov
  3. compute_spectral_eq.py  — LTASE spektrální EQ křivka
  4. train_instrument_profile.py — surrogate NN model (parametrový profil)
  5. closed_loop_finetune.py — MRSTFT fine-tuning (volitelný, --finetune)

Výstupní soubory (v --out-dir, default analysis/):
  params-<banka>.json              naměřené fyzikální parametry
  params-nn-profile-<banka>.json   NN-smoothed profil pro syntetizér
  profile-<banka>.pt               váhy modelu (znovupoužitelné)

Použití
───────
  # Celý pipeline (doporučené)
  python analysis/train_pipeline.py --bank soundbanks/ks-grand

  # S explicitním pojmenováním banky a výstupem
  python analysis/train_pipeline.py \\
      --bank soundbanks/ks-grand \\
      --bank-name ks-grand \\
      --out-dir analysis/

  # Přeskočit EQ krok (rychlejší, ale profil bez tělové rezonance)
  python analysis/train_pipeline.py --bank soundbanks/ks-grand --skip-eq

  # Přidat closed-loop MRSTFT fine-tuning po NN tréninku
  python analysis/train_pipeline.py --bank soundbanks/ks-grand --finetune

  # Jen výpis příkazů (bez spuštění)
  python analysis/train_pipeline.py --bank soundbanks/ks-grand --dry-run

  # Od kroku 4 (přeskočit extrakci, params JSON existuje)
  python analysis/train_pipeline.py --bank soundbanks/ks-grand --start-at 4

Argumenty
─────────
  --bank         Adresář s WAV soubory (m060-vel3-f44.wav formát) [povinný]
  --bank-name    Název banky pro pojmenování souborů (default: jméno adresáře)
  --out-dir      Výstupní adresář (default: analysis/)
  --workers      Paralelní workery pro extrakci + EQ (default: CPU count)
  --epochs       Epochy NN tréninku (default: 1800)
  --hidden       Velikost MLP hidden vrstvy (default: 64)
  --lr           Learning rate (default: 0.003)
  --finetune     Spustit closed-loop MRSTFT fine-tuning po NN tréninku
  --ft-epochs    Epochy fine-tuningu (default: 200)
  --skip-outlier Přeskočit krok 2 (outlier removal)
  --skip-eq      Přeskočit krok 3 (spektrální EQ)
  --start-at     Spustit od kroku N (1-5, default: 1)
  --dry-run      Jen výpis příkazů, bez spuštění
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows: force UTF-8 output so Czech characters render correctly
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(step: int, title: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f" Krok {step}: {title}", flush=True)
    print(f"{'='*60}", flush=True)


def _run(cmd: list[str], dry_run: bool) -> bool:
    """Spustí příkaz; vrátí True při úspěchu."""
    print(" ".join(cmd), flush=True)
    if dry_run:
        print("  [dry-run — přeskočeno]", flush=True)
        return True
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  CHYBA: příkaz skončil s kódem {result.returncode}", flush=True)
        return False
    print(f"  OK  ({elapsed:.0f}s)", flush=True)
    return True


def _python(*args) -> list[str]:
    """Vrátí [sys.executable, ...] pro použití ve _run()."""
    return [sys.executable, "-u"] + list(args)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IthacaCoreResonator — tréninkový pipeline wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--bank',         required=True,
                   help='Adresář s WAV soubory banky')
    p.add_argument('--bank-name',    default=None,
                   help='Název banky (default: jméno adresáře)')
    p.add_argument('--out-dir',      default='analysis',
                   help='Výstupní adresář (default: analysis/)')
    # Paralelismus
    p.add_argument('--workers',      type=int, default=None,
                   help='Paralelní workery pro extrakci + EQ (default: CPU count)')
    # NN trénink
    p.add_argument('--epochs',       type=int,   default=1800)
    p.add_argument('--hidden',       type=int,   default=64)
    p.add_argument('--lr',           type=float, default=0.003)
    # Fine-tuning
    p.add_argument('--finetune',     action='store_true',
                   help='Closed-loop MRSTFT fine-tuning po NN tréninku')
    p.add_argument('--ft-epochs',    type=int,   default=200,
                   help='Epochy fine-tuningu (default: 200)')
    # Přeskočení kroků
    p.add_argument('--skip-outlier', action='store_true',
                   help='Přeskočit krok 2 (outlier removal)')
    p.add_argument('--skip-eq',      action='store_true',
                   help='Přeskočit krok 3 (spektrální EQ)')
    p.add_argument('--start-at',     type=int, default=1, choices=range(1, 6),
                   metavar='N',
                   help='Spustit od kroku N (1–5, default: 1)')
    # Ostatní
    p.add_argument('--dry-run',      action='store_true',
                   help='Jen výpis příkazů, bez spuštění')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    bank_dir  = Path(args.bank).resolve()
    bank_name = args.bank_name or bank_dir.name
    out_dir   = Path(args.out_dir)
    workers   = args.workers or os.cpu_count() or 4

    # Odvozené cesty výstupních souborů
    params_json  = out_dir / f"params-{bank_name}.json"
    profile_json = out_dir / f"params-nn-profile-{bank_name}.json"
    profile_pt   = out_dir / f"profile-{bank_name}.pt"

    # Výpis plánu
    print("\nIthacaCoreResonator — tréninkový pipeline", flush=True)
    print(f"  banka:   {bank_dir}", flush=True)
    print(f"  výstup:  {out_dir}/", flush=True)
    print(f"  params:  {params_json.name}", flush=True)
    print(f"  profil:  {profile_json.name}", flush=True)
    print(f"  model:   {profile_pt.name}", flush=True)
    print(f"  epochy:  {args.epochs}  hidden={args.hidden}  lr={args.lr}", flush=True)
    if args.finetune:
        print(f"  fine-tune: {args.ft_epochs} epoch po NN tréninku", flush=True)
    if args.dry_run:
        print("  [DRY RUN — příkazy se jen vypíší]", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    # ── Krok 1 — Extrakce fyzikálních parametrů ───────────────────────────────
    if args.start_at <= 1:
        _banner(1, "Extrakce fyzikálních parametrů")
        ok = _run(_python(
            "analysis/extract_params.py",
            "--bank",    str(bank_dir),
            "--out",     str(params_json),
            "--workers", str(workers),
        ), args.dry_run)
        if not ok:
            return 1

    # ── Krok 2 — Filtrace outlierů ────────────────────────────────────────────
    if args.start_at <= 2 and not args.skip_outlier:
        _banner(2, "Filtrace outlierů (z=10, in-place drop)")
        ok = _run(_python(
            "analysis/find_outliers.py",
            "--params", str(params_json),
            "--z",      "10",
            "--drop",
        ), args.dry_run)
        if not ok:
            return 1
    elif args.skip_outlier:
        print("\n  [Krok 2 přeskočen: --skip-outlier]", flush=True)

    # ── Krok 3 — Spektrální EQ ────────────────────────────────────────────────
    if args.start_at <= 3 and not args.skip_eq:
        _banner(3, "Spektrální EQ (LTASE, in-place)")
        ok = _run(_python(
            "analysis/compute_spectral_eq.py",
            "--params",  str(params_json),
            "--bank",    str(bank_dir),
            "--workers", str(workers),
        ), args.dry_run)
        if not ok:
            return 1
    elif args.skip_eq:
        print("\n  [Krok 3 přeskočen: --skip-eq]", flush=True)

    # ── Krok 4 — Surrogate NN trénink ─────────────────────────────────────────
    if args.start_at <= 4:
        _banner(4, f"Surrogate NN trénink ({args.epochs} epoch)")
        ok = _run(_python(
            "analysis/train_instrument_profile.py",
            "--in",         str(params_json),
            "--out",        str(profile_json),
            "--model",      str(profile_pt),
            "--epochs",     str(args.epochs),
            "--hidden",     str(args.hidden),
            "--lr",         str(args.lr),
            "--eval-every", "10",
        ), args.dry_run)
        if not ok:
            return 1

    # ── Krok 5 — Closed-loop MRSTFT fine-tuning (volitelný) ──────────────────
    if args.start_at <= 5 and args.finetune:
        _banner(5, f"Closed-loop MRSTFT fine-tuning ({args.ft_epochs} epoch)")
        ok = _run(_python(
            "analysis/closed_loop_finetune.py",
            "--mode",    "finetune",
            "--model",   str(profile_pt),
            "--out",     str(profile_pt),   # přepsat in-place
            "--bank",    str(bank_dir),
            "--epochs",  str(args.ft_epochs),
        ), args.dry_run)
        if not ok:
            return 1
    elif not args.finetune and args.start_at <= 5:
        print("\n  [Krok 5 přeskočen: přidej --finetune pro MRSTFT fine-tuning]",
              flush=True)

    print(f"\n{'='*60}", flush=True)
    print(" Pipeline dokončen.", flush=True)
    print(f"  params:  {params_json}", flush=True)
    print(f"  profil:  {profile_json}", flush=True)
    print(f"  model:   {profile_pt}", flush=True)
    print(f"{'='*60}\n", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
