"""
Evaluation script — reproduces Tables I–IV of arXiv:2101.03771.

Usage
-----
    python run_evaluate.py --dataset inria     --data_dir data/inria
    python run_evaluate.py --dataset ukbench   --data_dir data/ukbench
    python run_evaluate.py --dataset oxford5k  --data_dir data/oxford5k
    python run_evaluate.py --dataset paris6k   --data_dir data/paris6k

Key options
-----------
    --model       vit_b16 | vit_b32 | vit_l16 | vit_l32   (default: vit_b16)
    --norm        robust | l2_axis1 | l1_axis1 | ...       (default: robust)
    --metric      cosine | euclidean | ...                 (default: cosine)
    --batch_size  (default: 16)
    --cache       .npy file — skip extraction if already exists
"""

from __future__ import annotations

import argparse

import numpy as np

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_METRIC,
    DEFAULT_MODEL,
    DEFAULT_NORM,
    SUPPORTED_METRICS,
    SUPPORTED_MODELS,
    SUPPORTED_NORMS,
)
from core import build_features, mean_average_precision, ns_score, retrieve
from datasets import inria, oxford_paris, ukbench


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build(args: argparse.Namespace, image_paths: list[str]) -> np.ndarray:
    return build_features(
        image_paths,
        model_name=args.model,
        norm_method=args.norm,
        batch_size=args.batch_size,
        cache_path=args.cache,
    )


def _print_result(dataset: str, args: argparse.Namespace, label: str, value: str) -> None:
    print(f"\n[{dataset.upper()}]  model={args.model}  norm={args.norm}  metric={args.metric}")
    print(f"{label} = {value}")


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def eval_inria(args: argparse.Namespace) -> None:
    paths, q_idxs, rel_sets = inria.load(args.data_dir)
    print(f"INRIA Holidays: {len(paths)} images, {len(q_idxs)} queries")

    feats  = _build(args, paths)
    ranked = retrieve(feats[q_idxs], feats, metric=args.metric)

    # Remove each query image from its own ranked list
    ranked_clean = np.array([ranked[i][ranked[i] != q] for i, q in enumerate(q_idxs)])
    score = mean_average_precision(ranked_clean, rel_sets)
    _print_result("inria", args, "mAP", f"{score:.2f}")


def eval_ukbench(args: argparse.Namespace) -> None:
    paths, group_ids = ukbench.load(args.data_dir)
    print(f"UKBench: {len(paths)} images, {len(np.unique(group_ids))} groups")

    feats  = _build(args, paths)
    ranked = retrieve(feats, feats, metric=args.metric)[:, 1:]  # skip rank-0 (self)

    score = ns_score(ranked, group_ids)
    _print_result("ukbench", args, "N-S score", f"{score:.4f}  (perfect = 4.0)")


def eval_oxford_paris(args: argparse.Namespace) -> None:
    paths, q_idxs, _rois, rel_sets, junk_sets = oxford_paris.load(
        args.data_dir, args.dataset
    )
    print(f"{args.dataset}: {len(paths)} images, {len(q_idxs)} queries")

    feats  = _build(args, paths)
    # Paper uses the full image as query — not the annotated ROI crop
    ranked = retrieve(feats[q_idxs], feats, metric=args.metric)

    score = mean_average_precision(ranked, rel_sets, junk_sets)
    _print_result(args.dataset, args, "mAP", f"{score:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EVAL_FN = {
    "inria":    eval_inria,
    "ukbench":  eval_ukbench,
    "oxford5k": eval_oxford_paris,
    "paris6k":  eval_oxford_paris,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViT CBIR Evaluation  (arXiv:2101.03771)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",    required=True,              choices=list(_EVAL_FN))
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--model",      default=DEFAULT_MODEL,      choices=SUPPORTED_MODELS)
    p.add_argument("--norm",       default=DEFAULT_NORM,       choices=SUPPORTED_NORMS)
    p.add_argument("--metric",     default=DEFAULT_METRIC,     choices=SUPPORTED_METRICS)
    p.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int)
    p.add_argument("--cache",      default=None, metavar="FILE.npy",
                   help="Cache extracted features; reuse on next run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _EVAL_FN[args.dataset](args)


if __name__ == "__main__":
    main()
