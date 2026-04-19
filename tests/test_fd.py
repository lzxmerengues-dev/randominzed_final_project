"""Tests for fd.py — Frequent Directions bound and shape contract."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

from fd import frequent_directions
from metrics import covariance_error


def test_shape() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((300, 50))
    B = frequent_directions(A, ell=10)
    assert B.shape == (10, 50)


def test_bound() -> None:
    """FD guarantee: ||A^T A - B^T B||_2 <= ||A - A_k||_F^2 / (ell - k)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, 80))
    ell, k = 20, 5
    B = frequent_directions(A, ell)

    lhs = covariance_error(A, B)
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    rhs = float(np.sum(s[k:] ** 2)) / (ell - k)
    assert lhs <= rhs * 1.05, f"bound violated: {lhs:.3f} > {rhs:.3f}"


def test_sparse_input() -> None:
    rng = np.random.default_rng(0)
    A = sp.random(300, 50, density=0.1, random_state=rng, format="csr")
    B = frequent_directions(A, ell=10)
    assert B.shape == (10, 50)


def test_fewer_rows_than_ell() -> None:
    """Sketch degrades gracefully when n < ell."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 50))
    B = frequent_directions(A, ell=10)
    assert B.shape == (10, 50)


if __name__ == "__main__":
    test_shape()
    test_bound()
    test_sparse_input()
    test_fewer_rows_than_ell()
    print("fd: all tests passed")
