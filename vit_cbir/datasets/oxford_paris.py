"""
Oxford5K and Paris6K dataset loader.

Oxford5K:
  Download: https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
  - oxbuild_images.tgz  (5062 images)
  - gt_files_170407.tgz (ground truth .txt files)

Paris6K:
  Download: https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
  - paris_1.tgz, paris_2.tgz (6412 images)
  - paris_120310.tgz (ground truth .txt files)

Expected structure:
    data/oxford5k/
        images/         ← all .jpg images
        ground_truth/   ← *_query.txt, *_good.txt, *_ok.txt, *_junk.txt

Ground truth protocol:
  - relevant = good ∪ ok
  - ignored  = junk
  - query image itself is excluded from results
"""

import os
import glob
from pathlib import Path


def _load_list(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load(data_dir: str, dataset: str = "oxford5k"):
    """
    Returns:
        image_paths   : list of all image paths
        query_indices : list of int — query image indices
        query_rois    : list of (x1,y1,x2,y2) or None (paper uses full image)
        relevant_sets : list of set of relevant image indices per query
        junk_sets     : list of set of junk image indices per query
    """
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    ground_truth_dir = data_dir / "ground_truth"

    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    all_paths = sorted(glob.glob(str(img_dir / "*.jpg")))
    stem_to_idx = {Path(p).stem: i for i, p in enumerate(all_paths)}

    query_files = sorted(glob.glob(str(ground_truth_dir / "*_query.txt")))

    query_indices = []
    query_rois = []
    relevant_sets = []
    junk_sets = []

    for qf in query_files:
        base = qf.replace("_query.txt", "")

        query_line = _load_list(qf)[0]
        parts = query_line.split()

        # Oxford/Paris query format: "oxc1_<name> x1 y1 x2 y2"
        img_name = parts[0].replace("oxc1_", "").replace("paris_", "")
        roi = tuple(float(x) for x in parts[1:5]) if len(parts) >= 5 else None

        if img_name not in stem_to_idx:
            img_name_alt = parts[0]
            if img_name_alt not in stem_to_idx:
                continue
            img_name = img_name_alt

        query_idx = stem_to_idx[img_name]

        good = set(stem_to_idx[n] for n in _load_list(base + "_good.txt") if n in stem_to_idx)
        ok   = set(stem_to_idx[n] for n in _load_list(base + "_ok.txt")   if n in stem_to_idx)
        junk = set(stem_to_idx[n] for n in _load_list(base + "_junk.txt") if n in stem_to_idx)

        relevant = good | ok
        junk.add(query_idx)

        query_indices.append(query_idx)
        query_rois.append(roi)
        relevant_sets.append(relevant)
        junk_sets.append(junk)

    return all_paths, query_indices, query_rois, relevant_sets, junk_sets
