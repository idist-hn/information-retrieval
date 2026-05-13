"""
INRIA Holidays dataset loader.

Download: http://lear.inrialpes.fr/~jegou/data.php  (jpg1.tar.gz + jpg2.tar.gz)
Expected structure:
    data/inria/jpg/
        100000.jpg  ← query of group 1000
        100001.jpg  ← relevant
        100002.jpg  ← relevant
        110000.jpg  ← query of group 1100
        ...

Group ID = first 4 digits of the filename.
The first image of each group (suffix 0) is the query.
"""

import os
import glob
from pathlib import Path


def load(data_dir: str):
    """
    Returns:
        image_paths : list of all image paths (sorted)
        query_indices : list of int — indices of query images
        relevant_sets : list of set — relevant image indices per query
    """
    jpg_dir = Path(data_dir) / "jpg"
    if not jpg_dir.exists():
        raise FileNotFoundError(f"Directory not found: {jpg_dir}")

    all_paths = sorted(glob.glob(str(jpg_dir / "*.jpg")))
    name_to_idx = {Path(p).stem: i for i, p in enumerate(all_paths)}

    groups = {}
    for stem, idx in name_to_idx.items():
        group_id = stem[:4]
        groups.setdefault(group_id, []).append(idx)

    query_indices = []
    relevant_sets = []

    for group_id in sorted(groups):
        idxs = sorted(groups[group_id])
        query_idx = idxs[0]
        relevant = set(idxs) - {query_idx}
        query_indices.append(query_idx)
        relevant_sets.append(relevant)

    return all_paths, query_indices, relevant_sets
