"""
fd.py — Frequent Directions (Liberty 2013) with the standard 2ell buffer.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Union


def _shrink_fd(s: np.ndarray, Vt: np.ndarray, ell: int) -> np.ndarray:
    """Shrink a (2ell, d) SVD decomposition to an (ell, d) sketch.

    Sets δ = s[ell]^2 so that the top ell singular values remain positive
    and the bottom ell become exactly zero. Returns the top-ell rows of
    the shrunk decomposition.
    """
    if len(s) > ell:
        delta = float(s[ell] ** 2)
    elif len(s) == ell:
        delta = float(s[ell - 1] ** 2)
    else:
        delta = 0.0
    s_shrunk = np.sqrt(np.maximum(s[:ell] ** 2 - delta, 0.0))
    return s_shrunk[:, None] * Vt[:ell]


def frequent_directions(A: Union[np.ndarray, sp.spmatrix], ell: int) -> np.ndarray:
    """Deterministic Frequent Directions sketch.

    Guarantee (Liberty 2013): for any rank k < ell,
        ||A^T A - B^T B||_2 <= (1 / (ell - k)) * ||A - A_k||_F^2.

    Time: O(n * d * ell). Density-oblivious — a dense matrix is materialised
    before sketching to emphasise the asymptotic that SFD tries to beat.

    Args:
        A:   n x d matrix (dense ndarray or scipy sparse).
        ell: sketch size (number of rows returned).

    Returns:
        B: ell x d ndarray — the FD sketch.
    """
    if sp.issparse(A):
        A = A.toarray()
    n, d = A.shape
    B = np.zeros((2 * ell, d), dtype=np.float64)
    zero_row = 0

    for i in range(n):
        B[zero_row] = A[i]
        zero_row += 1
        if zero_row == 2 * ell:
            np.nan_to_num(B, copy=False)
            _, s, Vt = np.linalg.svd(B, full_matrices=False)
            top = _shrink_fd(s, Vt, ell)
            B = np.zeros((2 * ell, d), dtype=np.float64)
            B[:ell] = top
            zero_row = ell

    # Final shrink if buffer holds more than ell rows.
    if zero_row > ell:
        _, s, Vt = np.linalg.svd(B[:zero_row], full_matrices=False)
        return _shrink_fd(s, Vt, ell)
    return B[:ell].copy()
