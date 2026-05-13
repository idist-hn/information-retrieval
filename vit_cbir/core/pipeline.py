"""
End-to-end feature-building pipeline: load → extract → normalize → (cache).
Shared by demo.py and run_evaluate.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import DEFAULT_BATCH_SIZE, DEFAULT_MODEL, DEFAULT_NORM
from core.extractor import extract_features, load_model
from core.normalizer import normalize_features


def build_features(
    image_paths: list[str],
    model_name: str = DEFAULT_MODEL,
    norm_method: str = DEFAULT_NORM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: str | None = None,
) -> np.ndarray:
    """
    Load ViT model, extract raw features, normalize, then optionally cache to disk.

    On subsequent calls with the same cache_path the extraction step is skipped.

    Returns
    -------
    np.ndarray  shape (N, D), normalized descriptors ready for retrieval.
    """
    if cache_path and Path(cache_path).exists():
        print(f"[cache] Loading features from {cache_path}")
        return np.load(cache_path)

    model = load_model(model_name)
    raw = extract_features(image_paths, model, batch_size)
    feats = normalize_features(raw, norm_method)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, feats)
        print(f"[cache] Features saved to {cache_path}")

    return feats
