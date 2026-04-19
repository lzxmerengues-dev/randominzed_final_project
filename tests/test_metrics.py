"""Tests for metrics.py — error metrics and threshold formulas."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from metrics import (
    covariance_error, relative_error, rho_eff,
    rho_star_paper, rho_hat_drift, summarize_runs,
)


def test_covariance_error_zero() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 20))
    assert covariance_error(A, A) < 1e-8


def test_relative_error_zero() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 20))
    assert relative_error(A, A, k=1) < 1e-8


def test_rho_eff_bounds() -> None:
    val = rho_eff(batch_nnz=500, ell=10, d=100, batch_rows=10)
    assert 0.0 < val <= 1.0


def test_rho_star() -> None:
    assert abs(rho_star_paper(20, 1000) - 0.98) < 1e-8


def test_rho_hat_drift_lt_paper() -> None:
    """Drift threshold is always ≤ paper threshold."""
    p = rho_star_paper(20, 1000)
    d = rho_hat_drift(20, 1000, avg_batch_nnz=20_000)
    assert d <= p


def test_summarize_empty() -> None:
    assert summarize_runs([]) == {}


def test_summarize_nonempty() -> None:
    runs = [{"wall": 1.0, "cov_err": 0.1, "rel_err": 0.01},
            {"wall": 2.0, "cov_err": 0.2, "rel_err": 0.02},
            {"wall": 3.0, "cov_err": 0.3, "rel_err": 0.03}]
    s = summarize_runs(runs)
    assert s["n_seeds"] == 3
    assert abs(s["wall_median"] - 2.0) < 1e-8


if __name__ == "__main__":
    test_covariance_error_zero()
    test_relative_error_zero()
    test_rho_eff_bounds()
    test_rho_star()
    test_rho_hat_drift_lt_paper()
    test_summarize_empty()
    test_summarize_nonempty()
    print("metrics: all tests passed")
