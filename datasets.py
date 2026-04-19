"""
datasets.py — Synthetic and real loaders.

Synthetic:
    make_synthetic_lowrank(n, d, rank, rho) — column-sparse low-rank matrix.
    make_mixed_stream(n1, n2, d, rho1, rho2) — two phase stream (for Exp 3).

Real:
    load_movielens_1m()  — 6040 x 3706 user-item, ρ ≈ 4.5%.
    load_movielens_100k() — 943 x 1682 user-item, ρ ≈ 6.3%.
    load_20newsgroups()  — TF-IDF docs x terms, ρ ≈ 0.3%.
"""
from __future__ import annotations

import os
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------

def make_synthetic_lowrank(
    n: int,
    d: int,
    rank: int = 10,
    rho: float = 0.05,
    noise: float = 0.1,
    seed: int = 0,
) -> sp.csr_matrix:
    """Row-wise sparse low-rank matrix.

    Each row has exactly ⌈ρ·d⌉ nonzeros at randomly chosen columns. Values
    are U[i]·V[cols]^T (low-rank signal) plus Gaussian noise, so the matrix
    has density = ⌈ρ·d⌉/d ≈ ρ exactly and rank ≤ `rank` approximately.

    Unlike elementwise masking, this preserves low-rank structure. Unlike a
    shared-column support, it allows ρ arbitrarily small without a rank
    floor (each row's support is independent).
    """
    rng = np.random.default_rng(seed)
    k = max(1, min(d, int(round(rho * d))))

    U = rng.standard_normal((n, rank))
    V = rng.standard_normal((d, rank))
    scale = 1.0 / max(1.0, float(np.sqrt(max(n, d))))

    # Per-row support: k random columns chosen via argpartition of random weights
    if k >= d:
        col_idx = np.tile(np.arange(d), (n, 1))
    else:
        weights = rng.random((n, d))
        col_idx = np.argpartition(weights, k, axis=1)[:, :k]

    row_idx = np.repeat(np.arange(n), k)
    col_flat = col_idx.ravel()

    # Low-rank signal at each (row, col) entry: U[row] · V[col]
    signal = np.einsum("ij,ij->i", U[row_idx], V[col_flat])
    noise_vec = rng.standard_normal(row_idx.size) * noise
    vals = (signal + noise_vec) * scale

    A = sp.csr_matrix((vals, (row_idx, col_flat)), shape=(n, d))
    return A


def make_mixed_stream(
    n1: int, n2: int, d: int,
    rho1: float, rho2: float,
    rank: int = 10, seed: int = 0,
) -> sp.csr_matrix:
    """Two-phase stream: first `n1` rows at density `rho1`, then `n2` at `rho2`."""
    A1 = make_synthetic_lowrank(n1, d, rank, rho1, seed=seed)
    A2 = make_synthetic_lowrank(n2, d, rank, rho2, seed=seed + 1)
    return sp.vstack([A1, A2]).tocsr()


# ---------------------------------------------------------------------------
# Real
# ---------------------------------------------------------------------------

def load_movielens_100k(path: str = "data/ml-100k/u.data") -> sp.csr_matrix:
    """MovieLens 100K (943 x 1682, ρ ≈ 6.3%).

    Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path!r} not found. Download ml-100k.zip from "
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip "
            "and unzip to data/ml-100k/"
        )
    data = np.loadtxt(path, dtype=int, usecols=(0, 1, 2))
    users = data[:, 0] - 1
    items = data[:, 1] - 1
    scores = data[:, 2].astype(np.float64)
    A = sp.csr_matrix((scores, (users, items)),
                      shape=(int(users.max()) + 1, int(items.max()) + 1))
    print(f"[MovieLens-100K] {A.shape[0]} x {A.shape[1]} "
          f"nnz={A.nnz} density={A.nnz / (A.shape[0] * A.shape[1]):.4f}")
    return A


def load_movielens_1m(path: str = "data/ml-1m/ratings.dat") -> sp.csr_matrix:
    """MovieLens 1M (6040 x 3706, ρ ≈ 4.5%).

    Download: https://files.grouplens.org/datasets/movielens/ml-1m.zip
    Format: user::movie::rating::timestamp
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path!r} not found. Download ml-1m.zip from "
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip "
            "and unzip to data/ml-1m/"
        )
    users, items, scores = [], [], []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            users.append(int(parts[0]) - 1)
            items.append(int(parts[1]) - 1)
            scores.append(float(parts[2]))
    users = np.array(users); items = np.array(items); scores = np.array(scores)
    A = sp.csr_matrix((scores, (users, items)),
                      shape=(int(users.max()) + 1, int(items.max()) + 1))
    print(f"[MovieLens-1M] {A.shape[0]} x {A.shape[1]} "
          f"nnz={A.nnz} density={A.nnz / (A.shape[0] * A.shape[1]):.4f}")
    return A


def load_20newsgroups(max_features: int = 5000) -> sp.csr_matrix:
    """20 Newsgroups (~18846 docs x 5000 terms, TF-IDF, ρ ≈ 0.3%).

    Fetched via scikit-learn (cached in ~/scikit_learn_data by default).
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        raise ImportError("pip install scikit-learn") from e
    news = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    vec = TfidfVectorizer(max_features=max_features, min_df=5, sublinear_tf=True)
    A = vec.fit_transform(news.data).astype(np.float64).tocsr()
    print(f"[20Newsgroups] {A.shape[0]} x {A.shape[1]} "
          f"nnz={A.nnz} density={A.nnz / (A.shape[0] * A.shape[1]):.4f}")
    return A
