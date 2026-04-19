"""
sfd.py — Sparse Frequent Directions (Ghashami, Liberty, Phillips, KDD 2016).

Faithful implementation with:
  (a) nnz-based buffer filling (accumulate rows until nnz ≈ ell*d),
  (b) SimultaneousIteration on an implicit M = [B; batch] (never materialised),
  (c) real BoostedSparseShrink backed by randomized VerifySpectral,
  (d) instrumentation hooks for ρ_eff trajectory analysis.
"""
from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp
from typing import Union


# ---------------------------------------------------------------------------
# Building blocks for implicit M = [dense_sketch; sparse_batch]
# ---------------------------------------------------------------------------

def _apply_M(dense_sketch: np.ndarray, sparse_batch: sp.spmatrix, x: np.ndarray) -> np.ndarray:
    """M @ x where M = [dense_sketch; sparse_batch], never materialised."""
    top = dense_sketch @ x
    bot = np.asarray(sparse_batch @ x)
    return np.vstack([top, bot])


def _apply_MT(dense_sketch: np.ndarray, sparse_batch: sp.spmatrix, y: np.ndarray) -> np.ndarray:
    """M^T @ y where M = [dense_sketch; sparse_batch]."""
    sr = dense_sketch.shape[0]
    return dense_sketch.T @ y[:sr] + np.asarray(sparse_batch.T @ y[sr:])


def _apply_MtM(dense_sketch: np.ndarray, sparse_batch: sp.spmatrix, x: np.ndarray) -> np.ndarray:
    """M^T M @ x without forming M."""
    y_top = dense_sketch @ x
    y_bot = np.asarray(sparse_batch @ x)
    return dense_sketch.T @ y_top + np.asarray(sparse_batch.T @ y_bot)


# ---------------------------------------------------------------------------
# Randomised SVD of the implicit M (Halko-Martinsson-Tropp style)
# ---------------------------------------------------------------------------

def _simultaneous_iteration(
    dense_sketch: np.ndarray,
    sparse_batch: sp.spmatrix,
    k: int,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate top-k right singular vectors of M = [dense_sketch; sparse_batch].

    Returns (s, Vt) with len(s) <= k and Vt shape (len(s), d).
    Cost: O(n_iter * k * (sketch_rows * d + nnz(sparse_batch))).
    """
    sketch_rows, d = dense_sketch.shape
    batch_rows = sparse_batch.shape[0]
    m = sketch_rows + batch_rows
    k_eff = max(1, min(k, d, m))

    # Y = M @ Omega, Omega ~ N(0, I) in R^{d x k}
    Omega = np.random.standard_normal((d, k_eff))
    Y = _apply_M(dense_sketch, sparse_batch, Omega)  # (m, k_eff)

    # Power iteration on M M^T: Y <- M (M^T Y)
    for _ in range(n_iter):
        MtY = _apply_MT(dense_sketch, sparse_batch, Y)  # (d, k_eff)
        Y = _apply_M(dense_sketch, sparse_batch, MtY)    # (m, k_eff)
        # Orthonormalise periodically to avoid blow-up
        Y, _ = np.linalg.qr(Y)

    # Final range orthonormalisation
    Q, _ = np.linalg.qr(Y)  # (m, k_eff)

    # B_small = Q^T @ M, shape (k_eff, d). Apply via M^T @ Q and transpose.
    B_small = _apply_MT(dense_sketch, sparse_batch, Q).T  # (k_eff, d)

    _, s, Vt = np.linalg.svd(B_small, full_matrices=False)
    return s, Vt


# ---------------------------------------------------------------------------
# Shrink + verify
# ---------------------------------------------------------------------------

def _shrink(s: np.ndarray, Vt: np.ndarray, ell: int) -> np.ndarray:
    """Apply FD-style shrink to the ell-truncated SVD approximation.

    Uses δ = s[ell-1]^2 when only ell singular values are available;
    δ = s[ell]^2 when more are available (cleaner zeroing of the tail).
    """
    if len(s) > ell:
        delta = float(s[ell] ** 2)
        s_top, Vt_top = s[:ell], Vt[:ell]
    elif len(s) == ell:
        delta = float(s[ell - 1] ** 2)
        s_top, Vt_top = s, Vt
    else:
        # Pad with zero rows to reach ell
        pad = ell - len(s)
        s_top = np.concatenate([s, np.zeros(pad)])
        Vt_top = np.vstack([Vt, np.zeros((pad, Vt.shape[1]))])
        delta = 0.0

    s_shrunk = np.sqrt(np.maximum(s_top ** 2 - delta, 0.0))
    return s_shrunk[:, None] * Vt_top


def _verify_spectral(
    dense_sketch: np.ndarray,
    sparse_batch: sp.spmatrix,
    B_cand: np.ndarray,
    tol: float,
    n_probes: int = 10,
    n_power: int = 3,
) -> bool:
    """Randomised test: is ||M^T M - B_cand^T B_cand||_2 ≤ tol?

    Uses block power iteration on the difference (M^T M - B^T B), which
    never materialises M or its Gram matrix. A single pass would under-
    estimate; n_power=3 is a practical choice.
    """
    d = B_cand.shape[1]
    Omega = np.random.standard_normal((d, n_probes))
    for _ in range(n_power):
        # y = (M^T M - B^T B) @ Omega
        MtM_Om = _apply_MtM(dense_sketch, sparse_batch, Omega)
        BtB_Om = B_cand.T @ (B_cand @ Omega)
        Y = MtM_Om - BtB_Om
        # Renormalise via QR
        Omega, _ = np.linalg.qr(Y)
    MtM_Om = _apply_MtM(dense_sketch, sparse_batch, Omega)
    BtB_Om = B_cand.T @ (B_cand @ Omega)
    diff = MtM_Om - BtB_Om
    est = float(np.linalg.norm(diff, 2))
    return est <= tol


def _boosted_shrink(
    dense_sketch: np.ndarray,
    sparse_batch: sp.spmatrix,
    ell: int,
    n_iter: int,
    delta_prob: float,
    tol_factor: float = 2.0,
) -> tuple[np.ndarray, int]:
    """BoostedSparseShrink: retry SimultaneousIteration until VerifySpectral passes.

    Returns (B_new, n_attempts). Falls back to the last attempt on budget exhaustion.
    By the Chernoff bound, E[n_attempts] = O(log 1/δ).
    """
    max_retries = max(1, math.ceil(math.log(1.0 / max(delta_prob, 1e-10))))
    B_cand = np.zeros((ell, sparse_batch.shape[1]))
    for attempt in range(1, max_retries + 1):
        s, Vt = _simultaneous_iteration(dense_sketch, sparse_batch, ell, n_iter)
        B_cand = _shrink(s, Vt, ell)
        sigma_ell_sq = float(s[ell - 1] ** 2) if len(s) >= ell else 0.0
        tol = tol_factor * sigma_ell_sq
        if _verify_spectral(dense_sketch, sparse_batch, B_cand, tol):
            return B_cand, attempt
    return B_cand, max_retries


# ---------------------------------------------------------------------------
# nnz-based batching (paper's Algorithm 1)
# ---------------------------------------------------------------------------

def _collect_batch(A_csr: sp.csr_matrix, start: int, ell: int, d: int) -> tuple[sp.csr_matrix, int]:
    """Accumulate rows of A from `start` until total nnz reaches ell*d (or end).

    Returns (batch, end) where batch = A[start:end] and `end` is the first
    row NOT included.
    """
    n = A_csr.shape[0]
    if start >= n:
        return A_csr[n:n], n
    target = ell * d
    indptr = A_csr.indptr
    row_nnz = np.diff(indptr[start:])
    if row_nnz.size == 0:
        return A_csr[start:start], start
    cumsum = np.cumsum(row_nnz)
    idx = int(np.searchsorted(cumsum, target, side="left"))
    if idx < len(cumsum):
        end = start + idx + 1   # include the row that pushed us past target
    else:
        end = n
    end = max(start + 1, min(end, n))  # always consume at least one row
    return A_csr[start:end], end


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def sparse_frequent_directions(
    A: Union[np.ndarray, sp.spmatrix],
    ell: int,
    n_iter: int = 4,
    delta_prob: float = 0.1,
    instrument: bool = False,
):
    """Sparse Frequent Directions.

    Guarantee: with probability ≥ 1-δ, for α = 6/41 and any k < α*ell,
        ||A^T A - B^T B||_2 ≤ (1 / (α*ell - k)) * ||A - A_k||_F^2.

    Time: Õ(nnz(A) * ell + n * ell^2), dominated by the sparse matvecs in
    SimultaneousIteration.

    Args:
        A:          n x d matrix (dense or sparse; converted to csr).
        ell:        sketch size.
        n_iter:     SimultaneousIteration power steps.
        delta_prob: failure probability for BoostedSparseShrink retry loop.
        instrument: if True, also return per-shrink log (list of dicts).

    Returns:
        B (ell x d) — or (B, log) if instrument=True.
    """
    A = sp.csr_matrix(A) if not sp.issparse(A) else A.tocsr()
    n, d = A.shape
    B = np.zeros((ell, d), dtype=np.float64)
    start = 0
    log: list[dict] | None = [] if instrument else None

    while start < n:
        batch, end = _collect_batch(A, start, ell, d)
        if batch.shape[0] == 0:
            break
        B_new, attempts = _boosted_shrink(B, batch, ell, n_iter, delta_prob)

        if log is not None:
            batch_rows = batch.shape[0]
            batch_nnz = int(batch.nnz)
            log.append({
                "t": len(log),
                "start": int(start),
                "end": int(end),
                "batch_rows": batch_rows,
                "batch_nnz": batch_nnz,
                "attempts": attempts,
                "rho_eff": (ell * d + batch_nnz) / ((ell + batch_rows) * d),
                "rho_batch": batch_nnz / (batch_rows * d) if batch_rows > 0 else 0.0,
            })
        B = B_new
        start = end

    return (B, log) if instrument else B
