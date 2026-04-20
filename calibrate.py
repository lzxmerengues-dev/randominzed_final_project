"""
calibrate.py — Hardware calibration for the Adaptive FD/SFD cost model.

Fits a two-parameter (slope + intercept) cost model:
    T_FD  ≈ alpha_fd  * (rows * d * ell)                    + fixed_fd
    T_SFD ≈ beta_sfd  * (nnz * ell * n_iter + rows * ell^2) + fixed_sfd

The intercept captures BLAS / LAPACK fixed overhead that is significant for
small matrices and dominates the per-batch cost for tiny shrinks. A slope-
only fit systematically underestimates the advantage of SFD on small dense
batches, which was the cause of Adaptive being slower than pure SFD on the
mixed-density stream in the first experiment run.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fd import frequent_directions
from sfd import sparse_frequent_directions


# ---------------------------------------------------------------------------

def _time_fn(fn, n_repeat: int = 3) -> float:
    """Best-of-n wall-clock timing."""
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def _fit_affine(data: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares: y = slope * x + intercept. Returns (slope, max(0, intercept))."""
    xs = np.array([x for x, _ in data], dtype=float)
    ys = np.array([y for _, y in data], dtype=float)
    if len(xs) < 2 or float(np.var(xs)) < 1e-30:
        slope = float(np.sum(xs * ys) / max(float(np.sum(xs ** 2)), 1e-30))
        return slope, 0.0
    A = np.column_stack([xs, np.ones_like(xs)])
    coefs, *_ = np.linalg.lstsq(A, ys, rcond=None)
    slope, intercept = float(coefs[0]), float(coefs[1])
    # Intercept is physically a fixed cost; negative values are fit noise.
    return max(slope, 0.0), max(intercept, 0.0)


def _cache_path(d: int, ell: int, device: str) -> Path:
    key = f"{platform.system()}_{platform.machine()}_{device}_d{d}_ell{ell}"
    return Path.home() / ".cache" / "sfd_calibration" / f"{key}.json"


# ---------------------------------------------------------------------------

def calibrate(
    d: int = 1000,
    ell: int = 20,
    n_iter: int = 4,
    seed: int = 0,
    device: str = "cpu",
    use_cache: bool = True,
    verbose: bool = True,
) -> dict:
    """Fit (alpha_fd, fixed_fd, beta_sfd, fixed_sfd) on the local hardware.

    Returns dict with keys:
        alpha_fd, fixed_fd, beta_sfd, fixed_sfd, d, ell, n_iter, device, grid.
    """
    if use_cache:
        path = _cache_path(d, ell, device)
        if path.exists():
            if verbose:
                print(f"[calibrate] loading cached result from {path}")
            return json.loads(path.read_text())

    rng = np.random.default_rng(seed)
    # Grid spans small → larger shrink sizes and sparse → dense densities.
    grid = [
        (100,  0.5),  (100,  0.05),
        (300,  0.5),  (300,  0.05),
        (600,  0.5),  (600,  0.05),
        (1000, 0.05), (1000, 0.01),
    ]

    fd_data: list[tuple[float, float]] = []
    sfd_data: list[tuple[float, float]] = []

    for rows, rho in grid:
        A = sp.random(rows, d, density=rho, random_state=rng, format="csr")
        A_dense = A.toarray()

        t_fd = _time_fn(lambda A=A_dense: frequent_directions(A, ell))
        fd_data.append((rows * d * ell, t_fd))

        t_sfd = _time_fn(lambda A=A: sparse_frequent_directions(A, ell, n_iter=n_iter))
        sfd_flops = A.nnz * ell * n_iter + rows * ell * ell
        sfd_data.append((sfd_flops, t_sfd))

        if verbose:
            print(f"[calibrate]  rows={rows:5d} rho={rho:.2f}  "
                  f"t_fd={t_fd:6.3f}s  t_sfd={t_sfd:6.3f}s")

    alpha_fd, fixed_fd = _fit_affine(fd_data)
    beta_sfd, fixed_sfd = _fit_affine(sfd_data)

    result = {
        "alpha_fd": alpha_fd,
        "fixed_fd": fixed_fd,
        "beta_sfd": beta_sfd,
        "fixed_sfd": fixed_sfd,
        "d": d,
        "ell": ell,
        "n_iter": n_iter,
        "device": device,
        "grid": grid,
    }
    if verbose:
        print(f"[calibrate] fit: alpha_fd={alpha_fd:.3e}  fixed_fd={fixed_fd*1000:.3f}ms  "
              f"beta_sfd={beta_sfd:.3e}  fixed_sfd={fixed_sfd*1000:.3f}ms")

    if use_cache:
        path = _cache_path(d, ell, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        if verbose:
            print(f"[calibrate] cached to {path}")

    return result


if __name__ == "__main__":
    c = calibrate(use_cache=False)
    print(json.dumps(c, indent=2))
