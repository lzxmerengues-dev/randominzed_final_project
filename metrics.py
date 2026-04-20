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


def spectral_norm_cov_diff(
    A, B: np.ndarray, n_iter: int = 30, seed: int = 0,
) -> float:
    """Estimate ||A^T A - B^T B||_2 via power iteration, never forming A^T A.

    Each power step costs O(nnz(A) + ell·d) instead of O(d^3), which turns
    an hour-long evaluation on 20 Newsgroups (18846x5000) into < 1 second.
    """
    rng = np.random.default_rng(seed)
    d = B.shape[1]
    x = rng.standard_normal(d)
    x /= (float(np.linalg.norm(x)) + 1e-30)
    is_sparse = sp.issparse(A)

    def apply_ata(v: np.ndarray) -> np.ndarray:
        if is_sparse:
            return np.asarray(A.T @ (A @ v)).ravel()
        return A.T @ (A @ v)

    def apply_btb(v: np.ndarray) -> np.ndarray:
        return B.T @ (B @ v)

    for _ in range(n_iter):
        y = apply_ata(x) - apply_btb(x)
        nrm = float(np.linalg.norm(y))
        if nrm < 1e-15:
            return 0.0
        x = y / nrm
    y = apply_ata(x) - apply_btb(x)
    return float(np.linalg.norm(y))


def precompute_tail_sq(A, k: int = 1) -> float:
    """Precompute ||A - A_k||_F^2 once per dataset (≈ ||A||_F^2 - sum top-k σ^2).

    Uses Frobenius norm (O(nnz)) + Lanczos top-k SVD via scipy.sparse.linalg.svds
    (~1 sec on 18846x5000). For small matrices, falls back to exact SVD.
    """
    n, d = A.shape
    if sp.issparse(A):
        fro_sq = float(A.multiply(A).sum())
    else:
        fro_sq = float(np.sum(A ** 2))

    if min(n, d) <= k + 1:
        return max(fro_sq, 0.0)

    try:
        from scipy.sparse.linalg import svds
        A_sp = A if sp.issparse(A) else sp.csr_matrix(A)
        _, s, _ = svds(A_sp.astype(np.float64), k=k)
        top_sq = float(np.sum(s ** 2))
    except Exception:
        A_d = A.toarray() if sp.issparse(A) else A
        _, s, _ = np.linalg.svd(A_d, full_matrices=False)
        top_sq = float(np.sum(s[:k] ** 2))
    return max(fro_sq - top_sq, 0.0)


def relative_error_fast(
    A, B: np.ndarray, k: int = 1,
    tail_sq: float | None = None, n_iter: int = 30, seed: int = 0,
) -> float:
    """Fast variant of relative_error — O(nnz · n_iter) per call.

    If tail_sq is passed (precomputed once per dataset), each call is
    dominated by ~30 sparse matvecs — roughly 1000x faster than the exact
    formulation on 20 Newsgroups.
    """
    if tail_sq is None:
        tail_sq = precompute_tail_sq(A, k)
    if tail_sq < 1e-12:
        return 0.0
    return spectral_norm_cov_diff(A, B, n_iter=n_iter, seed=seed) / tail_sq


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
