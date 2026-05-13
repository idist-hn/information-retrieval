"""
Distance-based retrieval engine.
Best config from paper: Cosine distance (Tables I–IV).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from config import SUPPORTED_METRICS


def compute_distances(
    queries: np.ndarray,
    database: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Pairwise distances between queries and database. Shape: (n_queries, n_db)."""
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_METRICS}")
    return cdist(queries, database, metric=metric)


def retrieve(
    query_features: np.ndarray,
    db_features: np.ndarray,
    top_k: int | None = None,
    metric: str = "cosine",
    return_distances: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Rank database images by distance to each query.

    Parameters
    ----------
    query_features   : (n_queries, D)
    db_features      : (n_db, D)
    top_k            : keep only top-k results per query
    metric           : distance metric name
    return_distances : if True, also return the sorted distance values

    Returns
    -------
    ranked_indices   : (n_queries, top_k or n_db)  — closest first
    distances        : (n_queries, top_k or n_db)  — only when return_distances=True
    """
    dists = compute_distances(query_features, db_features, metric)
    ranked = np.argsort(dists, axis=1)
    sorted_dists = np.take_along_axis(dists, ranked, axis=1)

    if top_k is not None:
        ranked = ranked[:, :top_k]
        sorted_dists = sorted_dists[:, :top_k]

    if return_distances:
        return ranked, sorted_dists
    return ranked
