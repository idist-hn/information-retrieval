"""
Post-processing normalization — Section III of arXiv:2101.03771.

Best config reported in the paper: ROBUST + Cosine distance.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize as sk_normalize

from config import SUPPORTED_NORMS


def apply_robust(features: np.ndarray) -> np.ndarray:
    """ROBUST scaling per feature dimension (Axis=0).
    Formula: X' = (X − Q1) / (Q3 − Q1), Q1 = 25th, Q3 = 75th percentile.
    """
    return RobustScaler(quantile_range=(25.0, 75.0)).fit_transform(features)


def apply_l2_axis1(features: np.ndarray) -> np.ndarray:
    """L2-norm applied independently to each descriptor (row)."""
    return sk_normalize(features, norm="l2", axis=1)


def apply_l1_axis1(features: np.ndarray) -> np.ndarray:
    """L1-norm applied independently to each descriptor (row)."""
    return sk_normalize(features, norm="l1", axis=1)


def apply_l2_axis0(features: np.ndarray) -> np.ndarray:
    """L2-norm per feature dimension across all descriptors (column)."""
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def apply_l1_axis0(features: np.ndarray) -> np.ndarray:
    """L1-norm per feature dimension across all descriptors (column)."""
    norms = np.abs(features).sum(axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


_NORMALIZERS: dict[str, callable] = {
    "robust":   apply_robust,
    "l2_axis1": apply_l2_axis1,
    "l1_axis1": apply_l1_axis1,
    "l2_axis0": apply_l2_axis0,
    "l1_axis0": apply_l1_axis0,
}


def normalize_features(features: np.ndarray, method: str = "robust") -> np.ndarray:
    if method not in SUPPORTED_NORMS:
        raise ValueError(f"method must be one of {SUPPORTED_NORMS}")
    return _NORMALIZERS[method](features)
