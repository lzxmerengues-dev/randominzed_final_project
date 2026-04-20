"""
adaptive.py — Adaptive FD/SFD that picks per-batch between exact SVD and
SimultaneousIteration, using the hardware-calibrated cost model.

Correctness: both branches return a valid FD-invariant sketch, so the output
is always a legitimate rank-ell sketch regardless of per-batch choice.
Guarantee: ||A^T A - B^T B||_2 <= ||A - A_k||_F^2 / (alpha_mix * ell - k),
where alpha_mix in [6/41, 1] depends on the FD/SFD batch mix.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Union

from sfd import _boosted_shrink, _collect_batch, _shrink


def adaptive_frequent_directions(
    A: Union[np.ndarray, sp.spmatrix],
    ell: int,
    calib: dict,
    n_iter: int = 4,
    delta_prob: float = 0.1,
    instrument: bool = False,
):
    """Adaptive FD/SFD selector.

    Args:
        A:          input matrix (dense or sparse).
        ell:        sketch size.
        calib:      output of calibrate.calibrate(); must contain alpha_fd, beta_sfd.
        n_iter:     power steps inside SFD branch.
        delta_prob: failure probability for BoostedSparseShrink.
        instrument: if True, also return per-shrink log.

    Returns:
        B (ell x d) — or (B, log) if instrument=True.
    """
    A = sp.csr_matrix(A) if not sp.issparse(A) else A.tocsr()
    n, d = A.shape
    alpha_fd  = float(calib["alpha_fd"])
    beta_sfd  = float(calib["beta_sfd"])
    fixed_fd  = float(calib.get("fixed_fd", 0.0))
    fixed_sfd = float(calib.get("fixed_sfd", 0.0))

    B = np.zeros((ell, d), dtype=np.float64)
    start = 0
    log: list[dict] | None = [] if instrument else None

    while start < n:
        batch, end = _collect_batch(A, start, ell, d)
        if batch.shape[0] == 0:
            break

        rows_M = ell + batch.shape[0]
        nnz_M = ell * d + batch.nnz  # dense sketch contributes ell*d
        c_fd  = alpha_fd * rows_M * d * ell                              + fixed_fd
        c_sfd = beta_sfd * (nnz_M * ell * n_iter + rows_M * ell * ell)   + fixed_sfd

        if c_fd <= c_sfd:
            # Exact SVD branch
            M = np.vstack([B, batch.toarray()])
            _, s, Vt = np.linalg.svd(M, full_matrices=False)
            B_new = _shrink(s, Vt, ell)
            choice = "fd"
            attempts = 1
        else:
            # Randomised (SFD) branch
            B_new, attempts = _boosted_shrink(B, batch, ell, n_iter, delta_prob)
            choice = "sfd"

        if log is not None:
            log.append({
                "t": len(log),
                "start": int(start),
                "end": int(end),
                "batch_rows": int(batch.shape[0]),
                "batch_nnz": int(batch.nnz),
                "choice": choice,
                "attempts": attempts,
                "c_fd": c_fd,
                "c_sfd": c_sfd,
                "rho_eff": nnz_M / (rows_M * d),
                "rho_batch": batch.nnz / (batch.shape[0] * d) if batch.shape[0] > 0 else 0.0,
            })

        B = B_new
        start = end

    return (B, log) if instrument else B


def adaptive_stats(log: list[dict]) -> dict:
    """Summarise Adaptive's per-batch choices from an instrumented run."""
    if not log:
        return {"n_batches": 0, "frac_fd": 0.0, "frac_sfd": 0.0, "total_attempts": 0}
    n = len(log)
    n_fd = sum(1 for r in log if r["choice"] == "fd")
    return {
        "n_batches": n,
        "frac_fd": n_fd / n,
        "frac_sfd": (n - n_fd) / n,
        "total_attempts": sum(r["attempts"] for r in log),
    }
