"""
benchmark.py — Compare FD vs SFD across density, sketch size, and real datasets.

Usage:
    python benchmark.py                  # synthetic density sweep
    python benchmark.py --movielens      # MovieLens experiment
    python benchmark.py --newsgroups     # 20Newsgroups experiment
"""

import time
import warnings
import argparse
import numpy as np
import scipy.sparse as sp

# Suppress false-positive BLAS warnings from macOS Accelerate
warnings.filterwarnings("ignore", category=RuntimeWarning)

from fd import frequent_directions
from sfd import sparse_frequent_directions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def covariance_error(A_dense: np.ndarray, B: np.ndarray) -> float:
    """Spectral norm ||A^T A - B^T B||_2."""
    ATA = A_dense.T @ A_dense
    BTB = B.T @ B
    return float(np.linalg.norm(ATA - BTB, ord=2))


def relative_error(A_dense: np.ndarray, B: np.ndarray, k: int = 1) -> float:
    """Covariance error normalized by ||A - A_k||_F^2 (tail energy)."""
    _, s, _ = np.linalg.svd(A_dense, full_matrices=False)
    tail_sq = float(np.sum(s[k:] ** 2))
    if tail_sq < 1e-12:
        return 0.0
    return covariance_error(A_dense, B) / tail_sq


def theoretical_flops(n: int, d: int, ell: int, nnz: int, n_iter: int = 4) -> dict:
    """
    Theoretical floating-point operation counts per algorithm.

    FD:  O(n * d * ell)  — exact SVD of ell×d matrix, n/ell times
    SFD: O(nnz * ell * n_iter + n * ell^2)  — sparse matvec in power iteration

    Returns dict with fd_flops, sfd_flops, and speedup ratio.
    """
    fd  = n * d * ell
    sfd = nnz * ell * n_iter + n * ell ** 2
    return {
        "fd_flops":  fd,
        "sfd_flops": sfd,
        "theoretical_speedup": fd / max(sfd, 1),
    }


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_synthetic(
    n: int, d: int, rho: float, rank: int = 10, seed: int = 42
) -> sp.csr_matrix:
    """
    Sparse n x d matrix with approximate rank `rank` and density ≈ rho.
    """
    rng = np.random.default_rng(seed)
    # Scale down to avoid Accelerate BLAS NaN issues on macOS
    scale = 1.0 / (rank * np.sqrt(max(d, n)))
    U = rng.standard_normal((n, rank)) * scale
    V = rng.standard_normal((d, rank)) * scale
    signal = U @ V.T
    mask = rng.random((n, d)) < rho
    noise = rng.standard_normal((n, d)) * (scale * 0.1)
    A_dense = np.nan_to_num((signal + noise) * mask)
    return sp.csr_matrix(A_dense)


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(A: sp.csr_matrix, ell: int, label: str = "") -> dict:
    A_dense = A.toarray()
    n, d = A_dense.shape
    density = A.nnz / (n * d)

    # FD — convert to dense first (FD ignores sparsity)
    t0 = time.perf_counter()
    B_fd = frequent_directions(A_dense.copy(), ell)
    fd_time = time.perf_counter() - t0

    # SFD — pass sparse directly
    t0 = time.perf_counter()
    B_sfd = sparse_frequent_directions(A, ell)
    sfd_time = time.perf_counter() - t0

    fd_err  = covariance_error(A_dense, B_fd)
    sfd_err = covariance_error(A_dense, B_sfd)
    fd_rel  = relative_error(A_dense, B_fd)
    sfd_rel = relative_error(A_dense, B_sfd)

    theory = theoretical_flops(n, d, ell, A.nnz if sp.issparse(A) else np.count_nonzero(A_dense))
    return {
        "label": label,
        "n": n, "d": d, "ell": ell,
        "density": density,
        "fd_time":  fd_time,
        "sfd_time": sfd_time,
        "speedup":  fd_time / max(sfd_time, 1e-9),
        "fd_error":  fd_err,
        "sfd_error": sfd_err,
        "fd_rel_error":  fd_rel,
        "sfd_rel_error": sfd_rel,
        **theory,
    }


def print_row(r: dict) -> None:
    print(
        f"rho={r['density']:.4f} | "
        f"FD {r['fd_time']:.3f}s  SFD {r['sfd_time']:.3f}s  "
        f"wall {r['speedup']:5.2f}x | "
        f"theory {r['theoretical_speedup']:6.1f}x | "
        f"FD_rel {r['fd_rel_error']:.4f}  SFD_rel {r['sfd_rel_error']:.4f}"
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_density_sweep(
    n: int = 2000, d: int = 1000, ell: int = 10,
    densities=None,
) -> list[dict]:
    """Main experiment: sweep ρ to find FD/SFD crossover."""
    if densities is None:
        densities = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]

    print(f"\n{'='*65}")
    print(f"Density Sweep  n={n} d={d} ell={ell}")
    print(f"{'='*65}")

    results = []
    for rho in densities:
        A = make_synthetic(n, d, rho)
        r = run_benchmark(A, ell, label=f"rho={rho}")
        print_row(r)
        results.append(r)

    rho_paper  = 1.0 - ell / d
    rho_adj    = (1.0 - ell / d) / 4   # adjusted for n_iter=4 power steps
    theory_cross = [r for r in results if r["theoretical_speedup"] >= 1.0]
    if theory_cross:
        print(f"\nTheory crossover (speedup >= 1):")
        print(f"  Paper rho*   = 1 - ell/d          = {rho_paper:.4f}")
        print(f"  Adjusted rho = (1 - ell/d)/n_iter = {rho_adj:.4f}")
        print(f"  Empirical (theory >= 1x): rho <= {max(r['density'] for r in theory_cross):.4f}")

    return results


def experiment_sketch_size(n: int = 600, d: int = 1000, rho: float = 0.01) -> list[dict]:
    """Fix density, sweep ell."""
    ells = [5, 10, 20, 40, 80]
    print(f"\n{'='*65}")
    print(f"Sketch-size Sweep  n={n} d={d} rho={rho}")
    print(f"{'='*65}")
    A = make_synthetic(n, d, rho)
    results = []
    for ell in ells:
        r = run_benchmark(A, ell, label=f"ell={ell}")
        print_row(r)
        results.append(r)
    return results


def experiment_real(dataset: str = "movielens", ell: int = 30) -> dict:
    from datasets import load_movielens, load_20newsgroups
    A = load_movielens() if dataset == "movielens" else load_20newsgroups()
    print(f"\nReal dataset: {dataset}")
    r = run_benchmark(A, ell, label=dataset)
    print_row(r)
    return r


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_density_sweep(results: list[dict], save_path: str = "density_sweep.png") -> None:
    """
    Two-panel figure:
      Left:  theoretical speedup vs density (log scale) with crossover lines
      Right: relative error FD vs SFD vs density
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. pip install matplotlib")
        return

    densities = [r["density"] for r in results]
    theory_su = [r["theoretical_speedup"] for r in results]
    wall_su   = [r["speedup"] for r in results]
    fd_err    = [r["fd_rel_error"] for r in results]
    sfd_err   = [r["sfd_rel_error"] for r in results]

    ell = results[0]["ell"]
    d   = results[0]["d"]
    rho_paper = 1.0 - ell / d
    rho_adj   = rho_paper / 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: speedup ---
    ax1.loglog(densities, theory_su, "b-o", label="Theoretical speedup")
    ax1.loglog(densities, wall_su,   "r--s", label="Wall-clock speedup (Python)")
    ax1.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax1.axvline(rho_paper, color="green", linewidth=1, linestyle="--",
                label=f"Paper ρ* = 1−ℓ/d = {rho_paper:.2f}")
    ax1.axvline(rho_adj, color="orange", linewidth=1, linestyle="--",
                label=f"Adjusted ρ = ρ*/n_iter ≈ {rho_adj:.2f}")
    ax1.set_xlabel("Density ρ")
    ax1.set_ylabel("Speedup (FD / SFD)")
    ax1.set_title("FD vs SFD Speedup")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # --- Right: error ---
    ax2.semilogx(densities, fd_err,  "b-o",  label="FD relative error")
    ax2.semilogx(densities, sfd_err, "r--s", label="SFD relative error")
    ax2.set_xlabel("Density ρ")
    ax2.set_ylabel("Relative covariance error")
    ax2.set_title("Approximation Quality")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--movielens",   action="store_true")
    parser.add_argument("--newsgroups",  action="store_true")
    parser.add_argument("--ell-sweep",   action="store_true")
    args = parser.parse_args()

    if args.movielens:
        experiment_real("movielens")
    elif args.newsgroups:
        experiment_real("newsgroups")
    elif args.ell_sweep:
        experiment_sketch_size()
    else:
        results = experiment_density_sweep()
        plot_density_sweep(results)
