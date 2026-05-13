"""
Visualization utilities for image retrieval results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core.types import RetrievalResult


def visualize(
    query_path: Path,
    results: list[RetrievalResult],
    output_path: Path,
) -> None:
    """
    Save a result grid: query on the left, ranked results to the right.

    Each result panel shows rank and similarity score.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 4))
    fig.patch.set_facecolor("#1e1e1e")

    def _load(p: Path) -> np.ndarray:
        return np.array(Image.open(p).convert("RGB"))

    # Query panel
    axes[0].imshow(_load(query_path))
    axes[0].set_title("Query", fontsize=10, color="#4fc3f7", fontweight="bold", pad=6)
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#4fc3f7")
        spine.set_linewidth(2)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Result panels
    for res in results:
        ax = axes[res.rank]
        ax.imshow(_load(res.path))
        ax.set_title(
            f"#{res.rank}  sim={res.similarity:.3f}",
            fontsize=8,
            color="#e0e0e0",
            pad=4,
        )
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved visualization → {output_path}")
