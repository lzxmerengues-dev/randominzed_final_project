import math
import numpy as np
import scipy.sparse as sp
from typing import Union


def _simultaneous_iteration_mixed(
    dense_sketch: np.ndarray,
    sparse_batch: sp.spmatrix,
    k: int,
    n_iter: int,
) -> tuple:
    """
    Approximate top-k SVD of M = [dense_sketch; sparse_batch] via block power method.

    The key sparse speedup: sparse_batch @ v costs O(nnz(sparse_batch))
    rather than O(rows * d).  dense_sketch @ v is O(sketch_rows * d).

    Returns: (s, Vt) — singular values and right singular vectors.
    """
    _, d = dense_sketch.shape
    k = min(k, d)

    Omega = np.random.standard_normal((d, k))

    # Z = M @ Omega, exploiting sparsity of sparse_batch
    Z_dense  = dense_sketch @ Omega                              # O(sketch_rows * d)
    Z_sparse = np.asarray(sparse_batch @ Omega)                 # O(nnz(batch))
    Z = np.vstack([Z_dense, Z_sparse])

    m = Z.shape[0]
    k = min(k, m)

    for _ in range(n_iter):
        Z, _ = np.linalg.qr(Z[:, :k])
        # M^T @ Z = dense^T @ Z_top + sparse^T @ Z_bot
        n_dense = dense_sketch.shape[0]
        MtZ = dense_sketch.T @ Z[:n_dense] + np.asarray(sparse_batch.T @ Z[n_dense:])
        Z = np.vstack([
            dense_sketch @ MtZ,                                  # O(sketch_rows * d)
            np.asarray(sparse_batch @ MtZ),                     # O(nnz(batch))
        ])

    Q, _ = np.linalg.qr(Z[:, :k])
    # B_small = Q^T @ M  (k × d, small → cheap exact SVD)
    n_dense = dense_sketch.shape[0]
    B_small = Q[:n_dense].T @ dense_sketch + np.asarray(Q[n_dense:].T @ sparse_batch)
    _, s, Vt = np.linalg.svd(B_small, full_matrices=False)
    return s, Vt


def _shrink(s: np.ndarray, Vt: np.ndarray, ell: int) -> tuple[np.ndarray, int]:
    """Apply FD shrink step: subtract δ = σ_ell² from all squared singular values."""
    delta_val = float(s[ell - 1] ** 2) if len(s) >= ell else 0.0
    s_shrunk = np.sqrt(np.maximum(s ** 2 - delta_val, 0.0))
    B_new = s_shrunk[:, None] * Vt
    zero_row = min(int(np.sum(s_shrunk > 1e-12)), ell - 1)
    return B_new, zero_row


def sparse_frequent_directions(
    A: Union[np.ndarray, sp.spmatrix],
    ell: int,
    n_iter: int = 4,
    delta_prob: float = 0.1,
) -> np.ndarray:
    """
    Sparse Frequent Directions (SFD).

    Replaces FD's exact SVD with SimultaneousIteration using sparse matrix-vector
    products — touching only nonzero entries of each batch from A.

    Covariance error guarantee:
        ||A^T A - B^T B||_2 <= (6/41) * (1/ell) * ||A - A_k||_F^2

    Time: O~(nnz(A) * ell + n * ell^2)   vs FD's O(n * d * ell)

    Args:
        A:          n x d input matrix (dense or scipy sparse)
        ell:        sketch size
        n_iter:     SimultaneousIteration power steps (4 is usually enough)
        delta_prob: failure probability for BoostedSparseShrink retry loop

    Returns:
        B: ell x d sketch matrix
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.tocsr()

    n, d = A.shape
    max_retries = max(1, math.ceil(math.log(1.0 / max(delta_prob, 1e-10))))

    # Sketch: dense ell × d, tracks accumulated directions
    B = np.zeros((ell, d), dtype=np.float64)
    sketch_rows = 0   # non-zero rows of B (the "real" sketch part)
    batch_start = 0

    while batch_start < n:
        # Fill up to ell rows total: keep sketch_rows sketch + fresh sparse rows
        free_slots = ell - sketch_rows
        batch_end  = min(batch_start + free_slots, n)
        sparse_batch = A[batch_start:batch_end]            # sparse! O(nnz) matvecs
        batch_start = batch_end

        # BoostedSparseShrink: retry loop bounding failure prob to delta_prob
        # (Chernoff bound → expected retries = O(log 1/δ))
        s, Vt = None, None
        dense_sketch = B[:sketch_rows]
        for _ in range(max_retries):
            s, Vt = _simultaneous_iteration_mixed(
                dense_sketch, sparse_batch, ell, n_iter
            )
            if len(s) > 0 and s[0] > 1e-12:
                break

        if s is None or len(s) == 0:
            # Degenerate: keep current sketch, skip batch
            continue

        B, sketch_rows = _shrink(s, Vt, ell)

    return B
