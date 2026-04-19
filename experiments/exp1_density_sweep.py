"""
Experiment 1 — Density sweep.

Sweeps input density ρ, records FD / SFD / Adaptive wall-clock and
covariance error. Plots speedup curve + overlay ρ*, ρ̂_drift, empirical
crossover.

Usage:
    python experiments/exp1_density_sweep.py          # quick (default)
    python experiments/exp1_density_sweep.py --full   # PDF-scale grid
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

# Allow running as `python experiments/exp1_*.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from benchmark import run_seeds
from calibrate import calibrate
from datasets import make_synthetic_lowrank
from metrics import rho_star_paper, summarize_runs


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def run(n: int, d: int, ell: int, densities: list[float], n_seeds: int) -> dict:
    calib = calibrate(d=d, ell=ell)
    seeds = list(range(n_seeds))

    rows = []
    for rho in densities:
        print(f"\n--- rho={rho} ---")
        A = make_synthetic_lowrank(n, d, rank=10, rho=rho, seed=0)
        entry = {"rho": rho, "n": n, "d": d, "ell": ell}
        for algo in ("fd", "sfd", "adaptive"):
            runs = run_seeds(A, ell, algo=algo, seeds=seeds, calib=calib)
            stats = summarize_runs(runs)
            entry[algo] = stats
            print(f"  {algo:9s}  wall={stats['wall_median']:.3f}s "
                  f"rel_err={stats['rel_err_median']:.4f}")
        rows.append(entry)

    return {
        "params": {"n": n, "d": d, "ell": ell, "n_seeds": n_seeds,
                   "densities": densities},
        "calib": calib,
        "rows": rows,
    }


def plot(data: dict, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    rows = data["rows"]
    ell = data["params"]["ell"]
    d = data["params"]["d"]
    rho = np.array([r["rho"] for r in rows])

    fd_wall  = np.array([r["fd"]["wall_median"] for r in rows])
    sfd_wall = np.array([r["sfd"]["wall_median"] for r in rows])
    ad_wall  = np.array([r["adaptive"]["wall_median"] for r in rows])

    fd_err  = np.array([r["fd"]["rel_err_median"] for r in rows])
    sfd_err = np.array([r["sfd"]["rel_err_median"] for r in rows])
    ad_err  = np.array([r["adaptive"]["rel_err_median"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.loglog(rho, fd_wall  / sfd_wall, "b-o", label="FD / SFD")
    ax1.loglog(rho, fd_wall  / ad_wall,  "g--^", label="FD / Adaptive")
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    rp = rho_star_paper(ell, d)
    ax1.axvline(rp, color="purple", ls="--", lw=1,
                label=f"Paper ρ* = 1-ℓ/d = {rp:.2f}")
    ax1.set_xlabel("Density ρ")
    ax1.set_ylabel("Speedup")
    ax1.set_title("Wall-clock speedup vs density")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.semilogx(rho, fd_err,  "b-o",  label="FD")
    ax2.semilogx(rho, sfd_err, "r--s", label="SFD")
    ax2.semilogx(rho, ad_err,  "g:^",  label="Adaptive")
    ax2.set_xlabel("Density ρ")
    ax2.set_ylabel("Relative covariance error")
    ax2.set_title("Approximation quality")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Figure → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="PDF-scale grid")
    args = parser.parse_args()

    if args.full:
        n, d, ell, n_seeds = 10000, 1000, 20, 5
        densities = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]
    else:
        n, d, ell, n_seeds = 2000, 500, 20, 3
        densities = [0.005, 0.02, 0.05, 0.1, 0.3]

    data = run(n, d, ell, densities, n_seeds)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "exp1.json"
    out_json.write_text(json.dumps(data, indent=2))
    print(f"Results → {out_json}")

    try:
        plot(data, FIGURES / "exp1_density_sweep.pdf")
        plot(data, FIGURES / "exp1_density_sweep.png")
    except ImportError as e:
        print(f"[skip plot] {e}. Results JSON is ready; rerun plot later.")


if __name__ == "__main__":
    main()
