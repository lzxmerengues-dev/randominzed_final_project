"""
calibrate.py — Hardware calibration for the Adaptive FD/SFD cost model.

Fits scalar coefficients (alpha_fd, beta_sfd) such that
    T_FD  ≈ alpha_fd * (rows * d * ell)
    T_SFD ≈ beta_sfd * (nnz * ell * n_iter + rows * ell^2)

One-time cost (~30s on a laptop). Cached to ~/.cache/sfd_calibration/.
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


def _fit_slope(data: list[tuple[float, float]]) -> float:
    """Least-squares through the origin: slope = sum(x*y) / sum(x^2)."""
    xs = np.array([x for x, _ in data], dtype=float)
    ys = np.array([y for _, y in data], dtype=float)
    denom = float(np.sum(xs ** 2))
    if denom < 1e-30:
        return 0.0
    return float(np.sum(xs * ys) / denom)


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
    """Fit (alpha_fd, beta_sfd) on the local hardware.

    Returns dict with keys {alpha_fd, beta_sfd, d, ell, device, grid}.
    """
    if use_cache:
        path = _cache_path(d, ell, device)
        if path.exists():
            if verbose:
                print(f"[calibrate] loading cached result from {path}")
            return json.loads(path.read_text())

    rng = np.random.default_rng(seed)
    # A small grid that spans the (rows, nnz) regime FD/SFD will see.
    grid = [(200, 0.5), (200, 0.05), (500, 0.5), (500, 0.05), (1000, 0.01)]

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

    alpha_fd = _fit_slope(fd_data)
    beta_sfd = _fit_slope(sfd_data)

    result = {
        "alpha_fd": alpha_fd,
        "beta_sfd": beta_sfd,
        "d": d,
        "ell": ell,
        "n_iter": n_iter,
        "device": device,
        "grid": grid,
    }
    if verbose:
        print(f"[calibrate] fitted  alpha_fd={alpha_fd:.3e}  beta_sfd={beta_sfd:.3e}")

    if use_cache:
        path = _cache_path(d, ell, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        if verbose:
            print(f"[calibrate] cached to {path}")

    return result


if __name__ == "__main__":
    # CLI: python calibrate.py — forces recalibration.
    c = calibrate(use_cache=False)
    print(json.dumps(c, indent=2))
