"""Tests for sfd.py — Sparse Frequent Directions bound, batching, verification."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

from sfd import (
    sparse_frequent_directions,
    _collect_batch, _simultaneous_iteration, _verify_spectral,
)
from metrics import covariance_error


def test_shape() -> None:
    np.random.seed(0)
    rng = np.random.default_rng(0)
    A = sp.random(300, 50, density=0.1, random_state=rng, format="csr")
    B = sparse_frequent_directions(A, ell=10)
    assert B.shape == (10, 50)


def test_bound() -> None:
    """SFD guarantee: ||A^T A - B^T B||_2 <= ||A - A_k||_F^2 / (alpha*ell - k)."""
    np.random.seed(0)
    rng = np.random.default_rng(0)
    A = sp.random(500, 80, density=0.05, random_state=rng, format="csr")
    ell, k = 40, 2
    B = sparse_frequent_directions(A, ell, n_iter=6)

    A_dense = A.toarray()
    lhs = covariance_error(A_dense, B)
    _, s, _ = np.linalg.svd(A_dense, full_matrices=False)
    tail = float(np.sum(s[k:] ** 2))
    alpha = 6.0 / 41.0
    rhs = tail / (alpha * ell - k)
    # Allow 10% slack for randomization
    assert lhs <= rhs * 1.1, f"SFD bound violated: {lhs:.3e} > {rhs:.3e}"


def test_nnz_batching() -> None:
    """_collect_batch should accumulate up to ≈ ell*d nnz, not ell rows."""
    rng = np.random.default_rng(0)
    A = sp.random(1000, 100, density=0.02, random_state=rng, format="csr")
    ell, d = 10, 100
    batch, end = _collect_batch(A, 0, ell, d)
    # Target is ell*d = 1000; with density 0.02 each row has ~2 nnz, so we
    # need ~500 rows to hit the target. The old row-based version would
    # have stopped at ~10 rows.
    assert batch.shape[0] > 50, \
        f"nnz batching should collect many rows at low density; got {batch.shape[0]}"
    assert batch.nnz >= ell * d or end == A.shape[0]


def test_instrument() -> None:
    np.random.seed(0)
    rng = np.random.default_rng(0)
    A = sp.random(500, 50, density=0.1, random_state=rng, format="csr")
    B, log = sparse_frequent_directions(A, ell=10, instrument=True)
    assert B.shape == (10, 50)
    assert isinstance(log, list) and len(log) >= 1
    for rec in log:
        assert 0.0 <= rec["rho_eff"] <= 1.0
        assert rec["attempts"] >= 1


def test_verify_spectral_accepts_truth() -> None:
    """VerifySpectral should accept M itself as a perfect sketch."""
    np.random.seed(0)
    rng = np.random.default_rng(0)
    A = sp.random(200, 40, density=0.2, random_state=rng, format="csr")
    dense_sketch = np.zeros((5, 40))
    # B candidate = A stacked with dense_sketch (trivially perfect)
    B_exact = np.vstack([dense_sketch, A.toarray()])
    # ||M^T M - B^T B|| = 0, tolerance 1.0 should pass
    assert _verify_spectral(dense_sketch, A, B_exact, tol=1.0)


if __name__ == "__main__":
    test_shape()
    test_bound()
    test_nnz_batching()
    test_instrument()
    test_verify_spectral_accepts_truth()
    print("sfd: all tests passed")
