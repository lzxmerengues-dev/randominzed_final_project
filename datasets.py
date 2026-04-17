"""
datasets.py — Loaders for real-world sparse benchmark matrices.

MovieLens 100K:
    Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip
    Unzip to: data/ml-100k/

20 Newsgroups:
    Fetched automatically via scikit-learn (requires: pip install scikit-learn).
"""

import os
import numpy as np
import scipy.sparse as sp


def load_movielens(path: str = "data/ml-100k/u.data") -> sp.csr_matrix:
    """
    Load MovieLens 100K as a user × item rating matrix.

    Shape: 943 users × 1682 items, density ≈ 6.3%
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.\n"
            "Download from https://files.grouplens.org/datasets/movielens/ml-100k.zip "
            "and unzip to data/ml-100k/"
        )

    data = np.loadtxt(path, dtype=int, usecols=(0, 1, 2))
    users  = data[:, 0] - 1
    items  = data[:, 1] - 1
    scores = data[:, 2].astype(np.float64)

    n_users = int(users.max()) + 1
    n_items = int(items.max()) + 1
    A = sp.csr_matrix((scores, (users, items)), shape=(n_users, n_items))

    density = A.nnz / (n_users * n_items)
    print(f"[MovieLens]  {n_users} users × {n_items} items  "
          f"nnz={A.nnz}  density={density:.4f}")
    return A


def load_20newsgroups(max_features: int = 5000) -> sp.csr_matrix:
    """
    Load 20 Newsgroups as a document × term TF-IDF matrix.

    Shape: ~18846 docs × max_features terms, density ≈ 0.2–0.5%
    Requires: pip install scikit-learn
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    news = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    vec = TfidfVectorizer(max_features=max_features, min_df=5, sublinear_tf=True)
    A = vec.fit_transform(news.data)

    density = A.nnz / (A.shape[0] * A.shape[1])
    print(f"[20Newsgroups]  {A.shape[0]} docs × {A.shape[1]} terms  "
          f"nnz={A.nnz}  density={density:.4f}")
    return A
