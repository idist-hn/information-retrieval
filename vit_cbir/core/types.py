"""Shared data types for the CBIR system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalResult:
    """A single ranked retrieval result."""

    path: Path
    rank: int
    distance: float

    @property
    def similarity(self) -> float:
        """Similarity score in [0, 1] derived from distance (lower distance → higher similarity)."""
        return max(0.0, 1.0 - self.distance)
