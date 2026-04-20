"""
Experiment 4 — Real datasets (MovieLens + 20 Newsgroups).

Runs FD / SFD / Adaptive across a small sweep of ℓ on real sparse matrices,
using the fast iterative error metrics from metrics.py (precomputed tail_sq,
power-iteration spectral norm). This avoids the O(n·d²) SVD that made the
naive pipeline take > 1 hour on 20 Newsgroups.

Falls back gracefully if a dataset is not downloaded.

Usage:
    python experiments/exp4_real_data.py                  # all available
    python experiments/exp4_real_data.py --movielens-1m
    python experiments/exp4_real_data.py --movielens-100k
    python experiments/exp4_real_data.py --newsgroups
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

from fd import frequent_directions
from sfd import sparse_frequent_directions
from adaptive import adaptive_frequent_directions
from calibrate import calibrate
from datasets import load_movielens_100k, load_movielens_1m, load_20newsgroups
from metrics import (
    spectral_norm_cov_diff, precompute_tail_sq, relative_error_fast,
    summarize_runs,
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


# ---------------------------------------------------------------------------

def _run_once_fast(A, ell: int, algo: str, seed: int, calib: dict,
                   tail_sq: float, err_iter: int = 30) -> dict:
    """Single run with iterative error evaluation. Never calls A.toarray()."""
    np.random.seed(seed)
    t0 = time.perf_counter()
    if algo == "fd":
        B = frequent_directions(A, ell)
    elif algo == "sfd":
        B = sparse_frequent_directions(A, ell)
    elif algo == "adaptive":
        B = adaptive_frequent_directions(A, ell, calib)
    else:
        raise ValueError(algo)
    wall = time.perf_counter() - t0

    cov_err = spectral_norm_cov_diff(A, B, n_iter=err_iter, seed=seed)
    rel_err = cov_err / tail_sq if tail_sq > 1e-12 else 0.0
    return {"algo": algo, "seed": seed, "ell": ell, "wall": wall,
            "cov_err": cov_err, "rel_err": rel_err}


def run_dataset(name: str, A, ells: list[int], n_seeds: int = 3) -> dict:
    print(f"\n=== {name}  shape={A.shape}  nnz={A.nnz} ===")

    print(f"  [precompute] tail_sq via Lanczos SVD (k=1) …")
    t0 = time.perf_counter()
    tail_sq = precompute_tail_sq(A, k=1)
    print(f"  [precompute] tail_sq={tail_sq:.3e} in {time.perf_counter()-t0:.1f}s")

    out = {"name": name, "n": int(A.shape[0]), "d": int(A.shape[1]),
           "nnz": int(A.nnz),
           "density": float(A.nnz / (A.shape[0] * A.shape[1])),
           "tail_sq": tail_sq,
           "ells": ells, "results": []}

    for ell in ells:
        calib = calibrate(d=A.shape[1], ell=ell)
        entry: dict = {"ell": ell}
        for algo in ("fd", "sfd", "adaptive"):
            runs = [
                _run_once_fast(A, ell, algo, seed, calib, tail_sq)
                for seed in range(n_seeds)
            ]
            entry[algo] = summarize_runs(runs)
            print(f"  ell={ell:3d}  {algo:9s}  "
                  f"wall={entry[algo]['wall_median']:.3f}s  "
                  f"rel_err={entry[algo]['rel_err_median']:.4f}")
        out["results"].append(entry)
    return out


def plot(all_data: list[dict], save_path: Path) -> None:
    import matplotlib.pyplot as plt
    n_datasets = len(all_data)
    if n_datasets == 0:
        return
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4.5),
                              squeeze=False)
    for ax, data in zip(axes[0], all_data):
        ells = data["ells"]
        fd_wall  = [r["fd"]["wall_median"]  for r in data["results"]]
        sfd_wall = [r["sfd"]["wall_median"] for r in data["results"]]
        ad_wall  = [r["adaptive"]["wall_median"] for r in data["results"]]
        ax.plot(ells, fd_wall,  "b-o",  label="FD")
        ax.plot(ells, sfd_wall, "r--s", label="SFD")
        ax.plot(ells, ad_wall,  "g:^",  label="Adaptive")
        ax.set_xlabel("Sketch size ℓ")
        ax.set_ylabel("Wall-clock (s)")
        ax.set_title(f"{data['name']}\n(ρ={data['density']:.4f}, "
                     f"{data['n']}×{data['d']})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Figure → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movielens-1m",   action="store_true")
    parser.add_argument("--movielens-100k", action="store_true")
    parser.add_argument("--newsgroups",     action="store_true")
    args = parser.parse_args()

    run_all = not any([args.movielens_1m, args.movielens_100k, args.newsgroups])

    all_data = []
    if args.movielens_1m or run_all:
        try:
            A = load_movielens_1m()
            all_data.append(run_dataset("MovieLens-1M", A, ells=[10, 30, 100]))
        except FileNotFoundError as e:
            print(f"[skip movielens-1m] {e}")

    if args.movielens_100k or run_all:
        try:
            A = load_movielens_100k()
            all_data.append(run_dataset("MovieLens-100K", A, ells=[10, 30, 100]))
        except FileNotFoundError as e:
            print(f"[skip movielens-100k] {e}")

    if args.newsgroups or run_all:
        try:
            A = load_20newsgroups(max_features=5000)
            all_data.append(run_dataset("20Newsgroups", A, ells=[10, 30, 100]))
        except ImportError as e:
            print(f"[skip 20newsgroups] {e}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "exp4.json"
    out_json.write_text(json.dumps(all_data, indent=2))
    print(f"\nResults → {out_json}")

    try:
        plot(all_data, FIGURES / "exp4_real_data.pdf")
        plot(all_data, FIGURES / "exp4_real_data.png")
    except ImportError as e:
        print(f"[skip plot] {e}. Results JSON is ready; rerun plot later.")


if __name__ == "__main__":
    main()
