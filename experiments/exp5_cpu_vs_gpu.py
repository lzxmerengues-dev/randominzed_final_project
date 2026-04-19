"""
Experiment 5 — CPU vs GPU crossover.

If cupy is available and a CUDA device is visible, runs FD and SFD on GPU
via cupy.ndarray / cupyx.scipy.sparse. Compares calibration coefficients
and wall-clock crossover density.

Falls back to CPU-only output (with a note) if cupy is not installed.

Usage:
    python experiments/exp5_cpu_vs_gpu.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from benchmark import run_seeds
from calibrate import calibrate
from datasets import make_synthetic_lowrank
from metrics import summarize_runs


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _have_cupy() -> bool:
    try:
        import cupy  # noqa: F401
        import cupyx.scipy.sparse  # noqa: F401
        return True
    except Exception:
        return False


def cpu_sweep(n: int, d: int, ell: int, densities: list[float],
              n_seeds: int = 3) -> dict:
    calib = calibrate(d=d, ell=ell, device="cpu")
    rows = []
    for rho in densities:
        A = make_synthetic_lowrank(n, d, rank=10, rho=rho, seed=0)
        entry = {"rho": rho}
        for algo in ("fd", "sfd"):
            runs = run_seeds(A, ell, algo=algo,
                             seeds=list(range(n_seeds)), calib=calib)
            entry[algo] = summarize_runs(runs)
        rows.append(entry)
        print(f"  rho={rho}  fd={entry['fd']['wall_median']:.3f}  "
              f"sfd={entry['sfd']['wall_median']:.3f}")
    return {"calib": calib, "rows": rows}


def gpu_sweep(n: int, d: int, ell: int, densities: list[float]) -> dict:
    """Minimal GPU FD / SFD via cupy.

    We intentionally write a self-contained, simple GPU port here rather
    than parameterise fd.py / sfd.py — keeps the CPU code paths clean.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    def fd_gpu(A_cp, ell):
        n, d = A_cp.shape
        B = cp.zeros((2 * ell, d))
        zr = 0
        for i in range(n):
            B[zr] = A_cp[i]
            zr += 1
            if zr == 2 * ell:
                _, s, Vt = cp.linalg.svd(B, full_matrices=False)
                delta = s[ell] ** 2
                s2 = cp.sqrt(cp.maximum(s ** 2 - delta, 0.0))
                new_top = s2[:ell, None] * Vt[:ell]
                B = cp.zeros((2 * ell, d))
                B[:ell] = new_top
                zr = ell
        return B[:ell]

    def sfd_gpu(A_sp, ell, n_iter=4):
        """Simple SFD: nnz batching + power iteration on M^T M, no VerifySpectral."""
        n, d = A_sp.shape
        B = cp.zeros((ell, d))
        target = ell * d
        start = 0
        indptr = A_sp.indptr.get()
        while start < n:
            # Find batch end
            row_nnz = np.diff(indptr[start:])
            cs = np.cumsum(row_nnz)
            idx = int(np.searchsorted(cs, target, side="left"))
            end = min(start + idx + 1, n) if idx < len(cs) else n
            end = max(start + 1, end)
            batch = A_sp[start:end]
            Omega = cp.random.standard_normal((d, ell))
            # Power iteration
            for _ in range(n_iter):
                Y_top = B @ Omega
                Y_bot = batch @ Omega
                MtY = B.T @ Y_top + batch.T @ Y_bot
                Om_new = MtY
                Om_new, _ = cp.linalg.qr(Om_new)
                Omega = Om_new
            # Project and SVD
            Y_top = B @ Omega
            Y_bot = batch @ Omega
            Y = cp.vstack([Y_top, Y_bot])
            Q, _ = cp.linalg.qr(Y)
            B_small = (B.T @ Q[: B.shape[0]] + batch.T @ Q[B.shape[0]:]).T
            _, s, Vt = cp.linalg.svd(B_small, full_matrices=False)
            delta = float(s[ell - 1].get() ** 2) if len(s) >= ell else 0.0
            s2 = cp.sqrt(cp.maximum(s[:ell] ** 2 - delta, 0.0))
            B = s2[:, None] * Vt[:ell]
            start = end
        return B

    rows = []
    for rho in densities:
        A_np = make_synthetic_lowrank(n, d, rank=10, rho=rho, seed=0)
        A_dense = cp.asarray(A_np.toarray())
        A_sp = cpsp.csr_matrix(A_np)

        # Warm-up (CUDA kernel compilation)
        _ = fd_gpu(A_dense[:100], ell); cp.cuda.Stream.null.synchronize()
        _ = sfd_gpu(A_sp[:100], ell);   cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        fd_gpu(A_dense, ell); cp.cuda.Stream.null.synchronize()
        t_fd = time.perf_counter() - t0

        t0 = time.perf_counter()
        sfd_gpu(A_sp, ell); cp.cuda.Stream.null.synchronize()
        t_sfd = time.perf_counter() - t0

        rows.append({"rho": rho, "fd_wall": t_fd, "sfd_wall": t_sfd})
        print(f"  rho={rho}  fd_gpu={t_fd:.3f}  sfd_gpu={t_sfd:.3f}")

    return {"calib": None, "rows": rows}


def plot(data: dict, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, key, title in zip(axes, ("cpu", "gpu"), ("CPU", "GPU")):
        block = data.get(key)
        if block is None:
            ax.set_title(f"{title} (unavailable)")
            ax.axis("off")
            continue
        rhos = [r["rho"] for r in block["rows"]]
        if key == "cpu":
            fd_w  = [r["fd"]["wall_median"]  for r in block["rows"]]
            sfd_w = [r["sfd"]["wall_median"] for r in block["rows"]]
        else:
            fd_w  = [r["fd_wall"]  for r in block["rows"]]
            sfd_w = [r["sfd_wall"] for r in block["rows"]]
        ax.loglog(rhos, fd_w,  "b-o",  label="FD")
        ax.loglog(rhos, sfd_w, "r--s", label="SFD")
        ax.set_xlabel("Density ρ")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Wall-clock (s)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Figure → {save_path}")


def main() -> None:
    n, d, ell = 2000, 500, 20
    densities = [0.005, 0.02, 0.05, 0.1, 0.3]

    print("=== CPU sweep ===")
    cpu = cpu_sweep(n, d, ell, densities)

    if _have_cupy():
        print("\n=== GPU sweep ===")
        gpu = gpu_sweep(n, d, ell, densities)
    else:
        print("\n[skip GPU] cupy not available. "
              "pip install cupy-cuda12x  (or run in Colab).")
        gpu = None

    data = {"params": {"n": n, "d": d, "ell": ell, "densities": densities},
            "cpu": cpu, "gpu": gpu}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "exp5.json"
    out_json.write_text(json.dumps(data, indent=2))
    print(f"Results → {out_json}")

    try:
        plot(data, FIGURES / "exp5_cpu_vs_gpu.pdf")
        plot(data, FIGURES / "exp5_cpu_vs_gpu.png")
    except ImportError as e:
        print(f"[skip plot] {e}. Results JSON is ready; rerun plot later.")


if __name__ == "__main__":
    main()
