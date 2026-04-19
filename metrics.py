"""
metrics.py — Evaluation metrics shared across FD, SFD, and Adaptive.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def covariance_error(A_dense: np.ndarray, B: np.ndarray) -> float:
    """Spectral norm ||A^T A - B^T B||_2.

    Given an exact dense A and a sketch B, returns the spectral-norm gap
    between their Gram matrices. This is the quantity bounded by FD (by
    1/ell * ||A - A_k||_F^2) and by SFD (by (6/41) * 1/ell * ||A - A_k||_F^2).
    """
    ATA = A_dense.T @ A_dense
    BTB = B.T @ B
    return float(np.linalg.norm(ATA - BTB, ord=2))


def relative_error(A_dense: np.ndarray, B: np.ndarray, k: int = 1) -> float:
    """Covariance error normalized by tail energy ||A - A_k||_F^2."""
    _, s, _ = np.linalg.svd(A_dense, full_matrices=False)
    tail_sq = float(np.sum(s[k:] ** 2))
    if tail_sq < 1e-12:
        return 0.0
    return covariance_error(A_dense, B) / tail_sq


def rho_eff(batch_nnz: int, ell: int, d: int, batch_rows: int) -> float:
    """Effective density of the working buffer M = [B (dense); batch (sparse)]
    at shrink time. After the first shrink, B is dense so nnz(B) = ell * d.
    """
    return (ell * d + batch_nnz) / ((ell + batch_rows) * d)


def rho_star_paper(ell: int, d: int) -> float:
    """Closed-form ρ* = 1 - ell/d from the March abstract."""
    return 1.0 - ell / d


def rho_hat_drift(ell: int, d: int, avg_batch_nnz: float) -> float:
    """Drift-corrected threshold ρ̂_drift = ρ* · nnz(batch) / (ell*d + nnz(batch))."""
    paper = rho_star_paper(ell, d)
    return paper * avg_batch_nnz / (ell * d + avg_batch_nnz)


def summarize_runs(runs: list[dict]) -> dict:
    """Aggregate a list of single-seed runs into median + IQR."""
    if not runs:
        return {}
    keys = ["wall", "cov_err", "rel_err"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs if k in r])
        out[f"{k}_median"] = float(np.median(vals))
        out[f"{k}_q25"] = float(np.percentile(vals, 25))
        out[f"{k}_q75"] = float(np.percentile(vals, 75))
    out["n_seeds"] = len(runs)
    return out
