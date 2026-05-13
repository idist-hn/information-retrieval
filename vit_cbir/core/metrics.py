"""
Evaluation metrics — Section III of arXiv:2101.03771.

- mAP : for INRIA Holidays, Oxford5K, Paris6K
- NS  : top-4 recall score for UKBench
"""

from __future__ import annotations

import numpy as np


def average_precision(
    ranked_indices: np.ndarray,
    relevant_set: set,
    junk_set: set | None = None,
) -> float:
    """
    Average Precision for a single query.

    Parameters
    ----------
    ranked_indices : 1-D array of ranked database indices (closest first).
    relevant_set   : indices that are ground-truth relevant.
    junk_set       : indices to skip entirely (Oxford/Paris protocol).
    """
    if junk_set is None:
        junk_set = set()

    n_relevant = len(relevant_set)
    if n_relevant == 0:
        return 0.0

    hits, position, precision_sum = 0, 0, 0.0
    for idx in ranked_indices:
        if idx in junk_set:
            continue
        position += 1
        if idx in relevant_set:
            hits += 1
            precision_sum += hits / position

    return precision_sum / n_relevant


def mean_average_precision(
    all_ranked: np.ndarray,
    relevant_sets: list[set],
    junk_sets: list[set] | None = None,
) -> float:
    """
    mAP across all queries. Returns value in [0, 100].

    Parameters
    ----------
    all_ranked    : (n_queries, n_db) ranked indices.
    relevant_sets : list of relevant-index sets, one per query.
    junk_sets     : list of junk-index sets (or None), one per query.
    """
    if junk_sets is None:
        junk_sets = [set()] * len(relevant_sets)

    aps = [
        average_precision(all_ranked[i], relevant_sets[i], junk_sets[i])
        for i in range(len(relevant_sets))
    ]
    return float(np.mean(aps)) * 100.0


def ns_score(all_ranked: np.ndarray, group_ids: np.ndarray) -> float:
    """
    UKBench N-S score: average true positives in top-4 results. Perfect = 4.0.

    Parameters
    ----------
    all_ranked : (n_images, n_db) ranked indices — self already excluded.
    group_ids  : group_ids[i] is the group label of image i.
    """
    hits = sum(
        sum(1 for idx in all_ranked[i, :4] if group_ids[idx] == group_ids[i])
        for i in range(len(all_ranked))
    )
    return hits / len(all_ranked)
