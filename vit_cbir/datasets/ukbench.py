"""
UKBench dataset loader.

Download: https://archive.org/details/ukbench  (ukbench.zip)
Expected structure:
    data/ukbench/full/
        ukbench00000.jpg
        ukbench00001.jpg
        ...
        ukbench10199.jpg

Every 4 consecutive images form a group (group 0 = images 0-3, group 1 = images 4-7, ...).
Evaluation: N-S score = average true positives in top-4 results.
"""

import glob
from pathlib import Path
import numpy as np


def load(data_dir: str):
    """
    Returns:
        image_paths : sorted list of all image paths
        group_ids   : numpy array of group id for each image
    """
    img_dir = Path(data_dir) / "full"
    if not img_dir.exists():
        raise FileNotFoundError(f"Directory not found: {img_dir}")

    all_paths = sorted(glob.glob(str(img_dir / "ukbench*.jpg")))
    if len(all_paths) == 0:
        raise FileNotFoundError(f"No images found in {img_dir}")

    group_ids = np.array([i // 4 for i in range(len(all_paths))], dtype=np.int32)
    return all_paths, group_ids
