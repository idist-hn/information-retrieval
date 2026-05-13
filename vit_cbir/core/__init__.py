"""
Public API of the core CBIR package.

Consumers can import directly from `core` without knowing the internal layout:

    from core import build_features, retrieve, RetrievalResult
"""

from core.extractor import extract_features, load_model
from core.metrics import mean_average_precision, ns_score
from core.normalizer import normalize_features
from core.pipeline import build_features
from core.retrieval import retrieve
from core.types import RetrievalResult

__all__ = [
    "load_model",
    "extract_features",
    "normalize_features",
    "retrieve",
    "build_features",
    "mean_average_precision",
    "ns_score",
    "RetrievalResult",
]
