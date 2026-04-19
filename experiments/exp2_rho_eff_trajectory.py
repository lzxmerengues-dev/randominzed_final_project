"""
Experiment 2 — ρ_eff trajectory (Contribution D evidence).

Runs SFD with instrumentation on three input densities; plots ρ_eff(t)
trajectories. Evidence that after the first shrink B is dense, so ρ_eff
is bounded below by ell/(ell+r_t) regardless of input ρ.

Usage:
    python experiments/exp2_rho_eff_trajectory.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from benchmark import run_once
from datasets import make_synthetic_lowrank


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def run(n: int = 5000, d: int = 500, ell: int = 20,
        rhos=(0.005, 0.05, 0.2)) -> dict:
    out = {"params": {"n": n, "d": d, "ell": ell, "rhos": list(rhos)},
           "trajectories": []}
    for rho in rhos:
        print(f"\n--- rho={rho} ---")
        A = make_synthetic_lowrank(n, d, rank=10, rho=rho, seed=0)
        r = run_once(A, ell, algo="sfd", seed=0, instrument=True,
                     measure_error=False)
        log = r.pop("log")
        out["trajectories"].append({
            "rho": rho,
            "n_shrinks": len(log),
            "log": log,
            "wall": r["wall"],
        })
        print(f"  shrinks={len(log)}  first ρ_eff={log[0]['rho_eff']:.3f}  "
              f"last ρ_eff={log[-1]['rho_eff']:.3f}")
    return out


def plot(data: dict, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    ell = data["params"]["ell"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for traj, color in zip(data["trajectories"], colors):
        rho = traj["rho"]
        log = traj["log"]
        ts = [r["t"] for r in log]
        rho_eff = [r["rho_eff"] for r in log]
        rho_batch = [r["rho_batch"] for r in log]
        ax.plot(ts, rho_eff, "-o", color=color, markersize=4,
                label=f"ρ_eff, input ρ={rho}")
        ax.plot(ts, rho_batch, ":", color=color, alpha=0.5,
                label=f"ρ_batch, input ρ={rho}")
        ax.axhline(rho, color=color, lw=0.5, alpha=0.4)

    # Lower bound ell/(ell + r_t). r_t varies per batch; show ell/(2ell).
    ax.axhline(0.5, color="red", lw=1, ls="--",
               label=f"Lower bound ℓ/(ℓ+ℓ) = 0.5")

    ax.set_xlabel("Shrink index t")
    ax.set_ylabel("Density")
    ax.set_title(f"ρ_eff trajectory (ℓ={ell}): stays high regardless of input ρ")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Figure → {save_path}")


def main() -> None:
    data = run()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "exp2.json"
    out_json.write_text(json.dumps(data, indent=2))
    print(f"Results → {out_json}")

    try:
        plot(data, FIGURES / "exp2_rho_eff_trajectory.pdf")
        plot(data, FIGURES / "exp2_rho_eff_trajectory.png")
    except ImportError as e:
        print(f"[skip plot] {e}. Results JSON is ready; rerun plot later.")


if __name__ == "__main__":
    main()
