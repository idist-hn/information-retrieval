"""
Interactive demo: retrieve the most visually similar images to a query.

Usage
-----
    python demo.py --query path/to/query.jpg --db_dir path/to/images/ --top_k 5

Output
------
    Console        : ranked list with similarity scores.
    demo_result.png: image grid (query + top-k).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_METRIC,
    DEFAULT_MODEL,
    DEFAULT_NORM,
    IMAGE_EXTENSIONS,
    SUPPORTED_METRICS,
    SUPPORTED_MODELS,
    SUPPORTED_NORMS,
)
from core import RetrievalResult, build_features, retrieve
from utils import visualize


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def collect_images(directory: Path) -> list[Path]:
    """Return a sorted list of all image files directly inside *directory*."""
    paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in: {directory}")
    return sorted(paths)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def run_retrieval(
    query_path: Path,
    db_paths: list[Path],
    model_name: str = DEFAULT_MODEL,
    norm_method: str = DEFAULT_NORM,
    metric: str = DEFAULT_METRIC,
    batch_size: int = DEFAULT_BATCH_SIZE,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Build ViT descriptors for query + database, rank by distance.

    The query participates in normalization together with the database,
    consistent with the paper's batch-evaluation protocol.

    Returns a list of RetrievalResult sorted closest-first.
    """
    all_paths = [str(query_path)] + [str(p) for p in db_paths]
    feats = build_features(all_paths, model_name, norm_method, batch_size)

    query_feat = feats[[0]]
    db_feats = feats[1:]

    ranked_idx, dists = retrieve(
        query_feat, db_feats,
        top_k=top_k,
        metric=metric,
        return_distances=True,
    )

    return [
        RetrievalResult(path=db_paths[idx], rank=r + 1, distance=float(dists[0, r]))
        for r, idx in enumerate(ranked_idx[0])
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViT Image Retrieval Demo  (arXiv:2101.03771)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--query",      required=True,              help="Path to query image")
    p.add_argument("--db_dir",     required=True,              help="Directory of database images")
    p.add_argument("--top_k",      type=int, default=5,        help="Number of results to retrieve")
    p.add_argument("--model",      default=DEFAULT_MODEL,      choices=SUPPORTED_MODELS)
    p.add_argument("--norm",       default=DEFAULT_NORM,       choices=SUPPORTED_NORMS)
    p.add_argument("--metric",     default=DEFAULT_METRIC,     choices=SUPPORTED_METRICS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--output",     default="demo_result.png",  help="Output image path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    query_path = Path(args.query)
    db_dir     = Path(args.db_dir)
    output     = Path(args.output)

    if not query_path.is_file():
        print(f"Error: query image not found: {query_path}", file=sys.stderr)
        sys.exit(1)
    if not db_dir.is_dir():
        print(f"Error: db_dir is not a directory: {db_dir}", file=sys.stderr)
        sys.exit(1)

    db_paths = collect_images(db_dir)
    print(f"Database: {len(db_paths)} images in {db_dir}")

    results = run_retrieval(
        query_path=query_path,
        db_paths=db_paths,
        model_name=args.model,
        norm_method=args.norm,
        metric=args.metric,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )

    print(f"\nTop-{args.top_k} results for: {query_path.name}")
    print(f"{'Rank':<6} {'Similarity':>10}  {'File'}")
    print("-" * 50)
    for res in results:
        print(f"{res.rank:<6} {res.similarity:>10.4f}  {res.path.name}")

    visualize(query_path, results, output)


if __name__ == "__main__":
    main()
