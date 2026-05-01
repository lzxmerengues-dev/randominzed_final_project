"""
Experiment 3 — Mixed-density stream (Contribution A evidence).

A stream with a density phase transition (sparse → dense). Pure FD wastes
work on the sparse half; pure SFD retries heavily on the dense half.
Adaptive should approach the lower envelope.

Usage:
    python experiments/exp3_mixed_stream.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from benchmark import run_once, run_seeds
from calibrate import calibrate
from datasets import make_mixed_stream
from metrics import summarize_runs


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def run(n1: int = 20000, n2: int = 20000, d: int = 1000, ell: int = 20,
        rho1: float = 0.005, rho2: float = 0.5, n_seeds: int = 3) -> dict:
    calib = calibrate(d=d, ell=ell)
    A = make_mixed_stream(n1, n2, d, rho1, rho2, seed=0)
    print(f"Mixed stream: {n1}+{n2}={n1+n2} rows, rho1={rho1} rho2={rho2}")

    seeds = list(range(n_seeds))
    summary = {}
    for algo in ("fd", "sfd", "adaptive"):
        runs = run_seeds(A, ell, algo=algo, seeds=seeds, calib=calib)
        summary[algo] = summarize_runs(runs)
        print(f"  {algo:9s} wall={summary[algo]['wall_median']:.3f}s "
              f"rel_err={summary[algo]['rel_err_median']:.4f}")

    # Run Adaptive once more with instrumentation for the choice timeline.
    r = run_once(A, ell, algo="adaptive", seed=0, calib=calib,
                 instrument=True, measure_error=False)

    return {
        "params": {"n1": n1, "n2": n2, "d": d, "ell": ell,
                   "rho1": rho1, "rho2": rho2, "n_seeds": n_seeds},
        "calib": calib,
        "summary": summary,
        "adaptive_log": r["log"],
    }


def plot(data: dict, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    summary = data["summary"]
    log = data["adaptive_log"]
    ell = data["params"]["ell"]

    # Left: wall-clock bar chart with IQR
    algos = ["fd", "sfd", "adaptive"]
    colors = ["tab:blue", "tab:red", "tab:green"]
    walls = [summary[a]["wall_median"] for a in algos]
    lo = [summary[a]["wall_median"] - summary[a]["wall_q25"] for a in algos]
    hi = [summary[a]["wall_q75"] - summary[a]["wall_median"] for a in algos]
    ax1.bar(algos, walls, yerr=[lo, hi], color=colors, capsize=5)
    ax1.set_ylabel("Wall-clock (s, median ± IQR)")
    ax1.set_title("Mixed-density stream: total time")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Adaptive's per-batch choice on a log-spaced x-axis so the few
    # SFD batches near t=0 are visible alongside the long FD tail. Each
    # batch is a coloured tick on a horizontal line; phase boundary is
    # marked.
    import numpy as np
    from matplotlib.patches import Patch
    ts = np.array([r["t"] for r in log], dtype=float)
    is_fd = np.array([1 if r["choice"] == "fd" else 0 for r in log])
    n_fd = int(is_fd.sum())
    n_sfd = int((1 - is_fd).sum())
    phase_t = None
    for r in log:
        if r["start"] >= data["params"]["n1"]:
            phase_t = r["t"]
            break

    # Plot SFD and FD batches as separate scatter series (log x, t+1).
    sfd_ts = ts[is_fd == 0] + 1
    fd_ts  = ts[is_fd == 1] + 1
    ax2.scatter(sfd_ts, np.zeros_like(sfd_ts), marker="|",
                color="tab:red", s=400, linewidths=2, label="SFD chosen")
    ax2.scatter(fd_ts,  np.zeros_like(fd_ts),  marker="|",
                color="tab:blue", s=400, linewidths=1, alpha=0.6,
                label="FD chosen")
    if phase_t is not None:
        ax2.axvline(phase_t + 1 - 0.5, color="k", ls="--", lw=1.2,
                    label=f"phase boundary (t={phase_t})")
    ax2.set_xscale("log")
    ax2.set_xlim(0.8, ts.max() + 5)
    ax2.set_yticks([])
    ax2.set_xlabel("Shrink index t (log scale, +1 offset)")
    ax2.set_title(f"Adaptive's per-batch subroutine "
                  f"(SFD×{n_sfd}, FD×{n_fd})")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.set_ylim(-1, 1)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Figure → {save_path}")


def main() -> None:
    data = run()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "exp3.json"
    out_json.write_text(json.dumps(data, indent=2))
    print(f"Results → {out_json}")

    try:
        plot(data, FIGURES / "exp3_mixed_stream.pdf")
        plot(data, FIGURES / "exp3_mixed_stream.png")
    except ImportError as e:
        print(f"[skip plot] {e}. Results JSON is ready; rerun plot later.")


if __name__ == "__main__":
    main()
