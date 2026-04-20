"""
Experiment 5 — CPU vs GPU crossover.

Key fix vs. previous version: the GPU FD is now batched (ell rows per
iteration) instead of per-row assignment. The per-row version was doing
O(n) kernel launches, which dominated for small matrices and made GPU FD
look slower than CPU FD. The batched version does O(n/ell) launches, which
is what exercises the GPU's strength.

If cupy is available and a CUDA device is visible, runs FD and SFD on GPU.
Falls back to CPU-only output (with a note) otherwise.

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
        cupy.cuda.runtime.getDeviceCount()
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


def _fd_gpu_batched(A_cp, ell: int):
    """Batched GPU FD: ell rows per iteration, O(n/ell) SVDs.

    Unlike the per-row version, this amortises GPU kernel launch overhead
    by working with a fixed 2*ell × d buffer that is refilled ell rows at
    a time.
    """
    import cupy as cp
    n, d = A_cp.shape
    B = cp.zeros((ell, d), dtype=cp.float64)
    start = 0
    while start < n:
        end = min(start + ell, n)
        chunk = A_cp[start:end]
        if chunk.shape[0] < ell:
            pad = cp.zeros((ell - chunk.shape[0], d), dtype=chunk.dtype)
            chunk = cp.vstack([chunk, pad])
        buffer_ = cp.vstack([B, chunk])                                     # (2ell, d)
        _, s, Vt = cp.linalg.svd(buffer_, full_matrices=False)
        k = min(ell, s.shape[0] - 1)
        delta = s[k] ** 2
        s2 = cp.sqrt(cp.maximum(s ** 2 - delta, 0.0))
        B = s2[:ell, None] * Vt[:ell]
        start = end
    return B


def _sfd_gpu(A_sp, ell: int, n_iter: int = 4):
    """Minimal GPU SFD (no VerifySpectral): nnz batching + SI on implicit M."""
    import cupy as cp
    n, d = A_sp.shape
    B = cp.zeros((ell, d), dtype=cp.float64)
    target = ell * d
    indptr = A_sp.indptr.get()
    start = 0
    while start < n:
        row_nnz = np.diff(indptr[start:])
        if row_nnz.size == 0:
            break
        cs = np.cumsum(row_nnz)
        idx = int(np.searchsorted(cs, target, side="left"))
        end = min(start + idx + 1, n) if idx < len(cs) else n
        end = max(start + 1, end)
        batch = A_sp[start:end]

        Omega = cp.random.standard_normal((d, ell))
        for _ in range(n_iter):
            Y_top = B @ Omega
            Y_bot = batch @ Omega
            MtY = B.T @ Y_top + batch.T @ Y_bot
            Omega, _ = cp.linalg.qr(MtY)
        Y_top = B @ Omega
        Y_bot = batch @ Omega
        Y = cp.vstack([Y_top, Y_bot])
        Q, _ = cp.linalg.qr(Y)
        B_small = (B.T @ Q[:B.shape[0]] + batch.T @ Q[B.shape[0]:]).T
        _, s, Vt = cp.linalg.svd(B_small, full_matrices=False)
        k = min(ell - 1, s.shape[0] - 1)
        delta = s[k] ** 2
        s2 = cp.sqrt(cp.maximum(s[:ell] ** 2 - delta, 0.0))
        B = s2[:, None] * Vt[:ell]
        start = end
    return B


def gpu_sweep(n: int, d: int, ell: int, densities: list[float],
              n_seeds: int = 3) -> dict:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    rows_out = []
    for rho in densities:
        A_np = make_synthetic_lowrank(n, d, rank=10, rho=rho, seed=0)
        A_dense = cp.asarray(A_np.toarray(), dtype=cp.float64)
        A_sp = cpsp.csr_matrix(A_np.astype(np.float64))

        # Warm-up (compilation + cache warm)
        _ = _fd_gpu_batched(A_dense[:min(2 * ell, n)], ell)
        _ = _sfd_gpu(A_sp[:min(2 * ell, n)], ell)
        cp.cuda.Stream.null.synchronize()

        fd_times, sfd_times = [], []
        for _ in range(n_seeds):
            t0 = time.perf_counter()
            _fd_gpu_batched(A_dense, ell)
            cp.cuda.Stream.null.synchronize()
            fd_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _sfd_gpu(A_sp, ell)
            cp.cuda.Stream.null.synchronize()
            sfd_times.append(time.perf_counter() - t0)

        entry = {
            "rho": rho,
            "fd_wall": float(np.median(fd_times)),
            "fd_q25":  float(np.percentile(fd_times, 25)),
            "fd_q75":  float(np.percentile(fd_times, 75)),
            "sfd_wall": float(np.median(sfd_times)),
            "sfd_q25":  float(np.percentile(sfd_times, 25)),
            "sfd_q75":  float(np.percentile(sfd_times, 75)),
        }
        rows_out.append(entry)
        print(f"  rho={rho}  fd_gpu={entry['fd_wall']:.3f}  "
              f"sfd_gpu={entry['sfd_wall']:.3f}")
    return {"rows": rows_out}


def plot(data: dict, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
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
        ax.set_ylabel("Wall-clock (s)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Figure → {save_path}")


def main() -> None:
    # Bigger n than before so GPU overhead doesn't dominate.
    n, d, ell = 10000, 1000, 20
    densities = [0.005, 0.02, 0.05, 0.1, 0.3]

    print("=== CPU sweep ===")
    cpu = cpu_sweep(n, d, ell, densities)

    if _have_cupy():
        print("\n=== GPU sweep (batched FD) ===")
        gpu = gpu_sweep(n, d, ell, densities)
    else:
        print("\n[skip GPU] cupy + CUDA device not available. "
              "On Colab: Runtime → Change runtime type → T4 GPU.")
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
