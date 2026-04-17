import numpy as np
import scipy.sparse as sp
from typing import Union


def frequent_directions(A: Union[np.ndarray, sp.spmatrix], ell: int) -> np.ndarray:
    """
    Standard Frequent Directions (FD).

    Covariance error guarantee (optimal):
        ||A^T A - B^T B||_2 <= (1 / ell) * ||A - A_k||_F^2

    Time: O(n * d * ell)  — ignores sparsity entirely.

    Args:
        A:   n x d input matrix (dense or scipy sparse)
        ell: sketch size (number of rows in output)

    Returns:
        B: ell x d sketch matrix
    """
    if sp.issparse(A):
        A = A.toarray()

    n, d = A.shape
    B = np.zeros((ell, d), dtype=np.float64)
    zero_row = 0

    for i in range(n):
        B[zero_row] = A[i]
        zero_row += 1

        if zero_row == ell:
            np.nan_to_num(B, copy=False)
            _, s, Vt = np.linalg.svd(B, full_matrices=False)
            delta = s[-1] ** 2
            s_shrunk = np.sqrt(np.maximum(s ** 2 - delta, 0.0))
            B = s_shrunk[:, None] * Vt
            zero_row = min(int(np.sum(s_shrunk > 1e-12)), ell - 1)

    return B
