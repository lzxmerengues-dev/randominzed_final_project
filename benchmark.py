"""
benchmark.py — Shared runners used by every experiment.

Not a standalone entry point any more; experiments/*.py are the runnable
scripts. This module provides run_once / run_seeds only.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp

# Silence spurious IEEE-flag warnings from Apple Accelerate BLAS matmul on
# finite inputs (macOS/arm64). Results are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
np.seterr(over="ignore", invalid="ignore", divide="ignore")

from fd import frequent_directions
from sfd import sparse_frequent_directions
from adaptive import adaptive_frequent_directions
from metrics import covariance_error, relative_error


def _seed_all(seed: int) -> None:
    np.random.seed(seed)


def run_once(
    A,
    ell: int,
    algo: str = "fd",
    seed: int = 0,
    calib: dict | None = None,
    instrument: bool = False,
    measure_error: bool = True,
    **kwargs: Any,
) -> dict:
    """Run one algorithm once and record metrics.

    Args:
        A:            input matrix.
        ell:          sketch size.
        algo:         "fd" | "sfd" | "adaptive".
        seed:         RNG seed (controls numpy global RNG used inside SFD/adaptive).
        calib:        calibration dict, required if algo == "adaptive".
        instrument:   if True, attach per-shrink log in return dict.
        measure_error: if False, skip the O(n*d^2) SVD needed for ||A||_F.
    """
    _seed_all(seed)
    t0 = time.perf_counter()

    if algo == "fd":
        B = frequent_directions(A, ell)
        log = None
    elif algo == "sfd":
        out = sparse_frequent_directions(A, ell, instrument=instrument, **kwargs)
        if instrument:
            B, log = out
        else:
            B, log = out, None
    elif algo == "adaptive":
        if calib is None:
            raise ValueError("adaptive requires calib={alpha_fd, beta_sfd}")
        out = adaptive_frequent_directions(A, ell, calib, instrument=instrument, **kwargs)
        if instrument:
            B, log = out
        else:
            B, log = out, None
    else:
        raise ValueError(f"unknown algo {algo!r}")

    wall = time.perf_counter() - t0

    result: dict[str, Any] = {
        "algo": algo, "seed": seed, "ell": ell, "wall": wall,
        "n": int(A.shape[0]), "d": int(A.shape[1]),
        "nnz": int(A.nnz if sp.issparse(A) else np.count_nonzero(A)),
    }

    if measure_error:
        A_dense = A.toarray() if sp.issparse(A) else A
        result["cov_err"] = covariance_error(A_dense, B)
        result["rel_err"] = relative_error(A_dense, B, k=1)

    if log is not None:
        result["log"] = log

    return result


def run_seeds(
    A, ell: int, algo: str, seeds: list[int],
    calib: dict | None = None, **kwargs: Any,
) -> list[dict]:
    """Run `algo` once per seed on the same `A`."""
    results = []
    for s in seeds:
        r = run_once(A, ell, algo=algo, seed=s, calib=calib, **kwargs)
        results.append(r)
    return results
