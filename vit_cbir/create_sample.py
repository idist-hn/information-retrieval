# -*- coding: utf-8 -*-
"""
Tao tap du lieu nho de test nhanh, dung symlink (khong copy file).

Ket qua tao ra trong thu muc data_sample/:
  ukbench  : 100 anh (25 nhom x 4)
  oxford5k : ~100 anh (5 query dau + relevant + distractor)
  paris6k  : ~100 anh (5 query dau + relevant + distractor)

Dung voi run_evaluate.py:
  python run_evaluate.py --dataset ukbench  --data_dir data_sample/ukbench
  python run_evaluate.py --dataset oxford5k --data_dir data_sample/oxford5k
  python run_evaluate.py --dataset paris6k  --data_dir data_sample/paris6k
"""

import glob
import os
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_DIR = Path("data")
OUT_DIR = Path("data_sample")

UKBENCH_GROUPS = 25
OXFORD_QUERIES = 5
PARIS_QUERIES = 5
MAX_IMAGES = 100


def symlink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def load_gt_names(gt_file):
    if not gt_file.exists():
        return []
    return [l.strip() for l in gt_file.read_text().splitlines() if l.strip()]


def sample_ukbench():
    src = DATA_DIR / "ukbench" / "full"
    dst = OUT_DIR / "ukbench" / "full"
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(glob.glob(str(src / "ukbench*.jpg")))
    selected = images[: UKBENCH_GROUPS * 4]

    for p in selected:
        symlink(Path(p), dst / Path(p).name)

    print("[ukbench]  %d anh  (%d nhom)" % (len(selected), UKBENCH_GROUPS))


def sample_oxford_paris(dataset, n_queries):
    src_images = DATA_DIR / dataset / "images"
    src_ground_truth = DATA_DIR / dataset / "ground_truth"
    dst_images = OUT_DIR / dataset / "images"
    dst_ground_truth = OUT_DIR / dataset / "ground_truth"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_ground_truth.mkdir(parents=True, exist_ok=True)

    query_files = sorted(glob.glob(str(src_ground_truth / "*_query.txt")))[:n_queries]

    needed_stems = set()

    for qf in query_files:
        base = qf.replace("_query.txt", "")
        for suffix in ("_query.txt", "_good.txt", "_ok.txt", "_junk.txt"):
            ground_truth_path = Path(base + suffix)
            if ground_truth_path.exists():
                symlink(ground_truth_path, dst_ground_truth / ground_truth_path.name)

        query_line = load_gt_names(Path(qf))
        if query_line:
            stem = query_line[0].split()[0].replace("oxc1_", "").replace("paris_", "")
            needed_stems.add(stem)

        for suffix in ("_good.txt", "_ok.txt"):
            for name in load_gt_names(Path(base + suffix)):
                needed_stems.add(name.strip())

    all_images = sorted(glob.glob(str(src_images / "*.jpg")))
    stem_to_path = {Path(p).stem: p for p in all_images}

    selected_paths = set()
    for stem in needed_stems:
        if stem in stem_to_path:
            selected_paths.add(stem_to_path[stem])

    remaining = MAX_IMAGES - len(selected_paths)
    if remaining > 0:
        non_relevant = [p for p in all_images if p not in selected_paths]
        distractors = random.sample(non_relevant, min(remaining, len(non_relevant)))
        selected_paths.update(distractors)

    final = sorted(selected_paths)[:MAX_IMAGES]

    for p in final:
        symlink(Path(p), dst_images / Path(p).name)

    print("[%s]  %d anh  (%d query + relevant + distractor)" % (dataset, len(final), n_queries))


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    if (DATA_DIR / "ukbench" / "full").exists():
        sample_ukbench()
    else:
        print("[ukbench]  bo qua - khong tim thay data/ukbench/full/")

    if (DATA_DIR / "oxford5k" / "images").exists():
        sample_oxford_paris("oxford5k", OXFORD_QUERIES)
    else:
        print("[oxford5k] bo qua - khong tim thay data/oxford5k/images/")

    if (DATA_DIR / "paris6k" / "images").exists():
        sample_oxford_paris("paris6k", PARIS_QUERIES)
    else:
        print("[paris6k]  bo qua - khong tim thay data/paris6k/images/")

    print("\nXong! Du lieu sample o: %s" % OUT_DIR.resolve())
