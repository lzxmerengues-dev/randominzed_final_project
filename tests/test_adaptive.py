"""Tests for adaptive.py — per-batch selection and correctness."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

from adaptive import adaptive_frequent_directions, adaptive_stats
from metrics import covariance_error


def _fake_calib(prefer: str) -> dict:
    """Construct calibration that forces a specific branch."""
    if prefer == "fd":
        return {"alpha_fd": 1e-12, "beta_sfd": 1.0}
    elif prefer == "sfd":
        return {"alpha_fd": 1.0, "beta_sfd": 1e-12}
    raise ValueError(prefer)


def test_shape() -> None:
    np.random.seed(0)
    rng = np.random.default_rng(0)
    A = sp.random(300, 50, density=0.1, random_state=rng, format="csr")
    calib = {"alpha_fd": 1e-9, "beta_sfd": 1e-9}
    B = adaptive_frequent_directions(A, ell=10, calib=calib)
    assert B.shape == (10, 50)


def test_fd_branch_exclusive() -> None:
    """When calib says FD is far cheaper, every batch should pick FD."""
    np.random.seed(0)
    rng = np.random.default_rng(0)
    A = sp.random(300, 50, density=0.1, random_state=rng, format="csr")
    B, log = adaptive_frequent_directions(
        A, ell=10, calib=_fake_calib("fd"), instrument=True,
    )
    stats = adaptive_stats(log)
    assert stats["frac_fd"] == 1.0


def test_sfd_branch_exclusive() -> None:
    np.random.seed(0)
    rng = np.random.default_rng(0)
    A = sp.random(300, 50, density=0.1, random_state=rng, format="csr")
    B, log = adaptive_frequent_directions(
        A, ell=10, calib=_fake_calib("sfd"), instrument=True,
    )
    stats = adaptive_stats(log)
    assert stats["frac_sfd"] == 1.0


def test_fd_invariant_preserved() -> None:
    """Regardless of path, the output should satisfy the FD-style covariance
    bound (with alpha_mix >= 6/41)."""
    np.random.seed(0)
    rng = np.random.default_rng(0)
    A = sp.random(500, 80, density=0.05, random_state=rng, format="csr")
    ell, k = 40, 2
    for prefer in ("fd", "sfd"):
        B = adaptive_frequent_directions(A, ell=ell, calib=_fake_calib(prefer))
        A_dense = A.toarray()
        lhs = covariance_error(A_dense, B)
        _, s, _ = np.linalg.svd(A_dense, full_matrices=False)
        tail = float(np.sum(s[k:] ** 2))
        alpha_mix = 1.0 if prefer == "fd" else 6.0 / 41.0
        rhs = tail / (alpha_mix * ell - k)
        assert lhs <= rhs * 1.1, \
            f"{prefer} invariant violated: {lhs:.3e} > {rhs:.3e}"


if __name__ == "__main__":
    test_shape()
    test_fd_branch_exclusive()
    test_sfd_branch_exclusive()
    test_fd_invariant_preserved()
    print("adaptive: all tests passed")
