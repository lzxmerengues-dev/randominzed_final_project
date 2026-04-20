"""
Experiment 2 — ρ_eff trajectory (Contribution D evidence).

Runs SFD with instrumentation across a range of input densities. For each
density, n is chosen so that the stream generates ≈ 15 shrinks — this keeps
the trajectory visualisation readable even at very low ρ (where otherwise
the stream would run out of rows before the first shrink).

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


def _auto_n(rho: float, d: int, ell: int, shrinks_target: int = 15,
            n_min: int = 2000, n_max: int = 80000) -> int:
    """Choose n so SFD runs ≈ shrinks_target shrinks.

    Per-batch row count ≈ (ell * d) / (rho * d) = ell / rho.  We size n so
    that n / rows_per_batch ≈ shrinks_target, clipped to [n_min, n_max].
    """
    nnz_per_row = max(1, int(round(rho * d)))
    rows_per_batch = max(1, (ell * d) // nnz_per_row)
    n = shrinks_target * rows_per_batch
    return int(np.clip(n, n_min, n_max))


def run(d: int = 1000, ell: int = 20,
        rhos=(0.005, 0.02, 0.05, 0.1, 0.2),
        shrinks_target: int = 15) -> dict:
    out = {"params": {"d": d, "ell": ell, "rhos": list(rhos),
                      "shrinks_target": shrinks_target},
           "trajectories": []}
    for rho in rhos:
        n = _auto_n(rho, d, ell, shrinks_target)
        print(f"\n--- rho={rho}  n={n} (auto) ---")
        A = make_synthetic_lowrank(n, d, rank=10, rho=rho, seed=0)
        r = run_once(A, ell, algo="sfd", seed=0, instrument=True,
                     measure_error=False)
        log = r.pop("log")
        out["trajectories"].append({
            "rho": rho, "n": n,
            "n_shrinks": len(log), "log": log, "wall": r["wall"],
        })
        rho_effs = [rec["rho_eff"] for rec in log]
        print(f"  shrinks={len(log)}  ρ_eff: first={rho_effs[0]:.3f}  "
              f"median={np.median(rho_effs):.3f}  last={rho_effs[-1]:.3f}  "
              f"(input ρ={rho})")
    return out


def plot(data: dict, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    ell = data["params"]["ell"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(data["trajectories"])))

    # Left: ρ_eff trajectory per input ρ
    for traj, color in zip(data["trajectories"], colors):
        rho = traj["rho"]
        log = traj["log"]
        ts = [rec["t"] for rec in log]
        rho_eff = [rec["rho_eff"] for rec in log]
        ax1.plot(ts, rho_eff, "-o", color=color, markersize=4,
                 label=f"ρ_in={rho}")
        ax1.axhline(rho, color=color, linewidth=0.6, linestyle=":", alpha=0.5)

    ax1.set_xlabel("Shrink index t")
    ax1.set_ylabel(r"$\rho_{\mathrm{eff}}(t)$")
    ax1.set_title(f"ρ_eff trajectory (ℓ={ell})\nDashed = input ρ; solid = ρ_eff")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="best")
    ax1.set_ylim(0, 1.05)

    # Right: ρ_eff / ρ ratio vs input ρ (the "drift" quantity)
    rhos = [t["rho"] for t in data["trajectories"]]
    first_eff = [t["log"][0]["rho_eff"] for t in data["trajectories"]]
    last_eff  = [t["log"][-1]["rho_eff"] for t in data["trajectories"]]
    ax2.loglog(rhos, np.array(first_eff) / np.array(rhos),
               "b-o", label="ρ_eff(1) / ρ")
    ax2.loglog(rhos, np.array(last_eff)  / np.array(rhos),
               "r--s", label="ρ_eff(T) / ρ")
    ax2.axhline(1.0, color="k", lw=0.6, ls=":")
    ax2.set_xlabel("Input density ρ")
    ax2.set_ylabel("Drift factor ρ_eff / ρ")
    ax2.set_title("Sketch-drift amplification")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)

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
