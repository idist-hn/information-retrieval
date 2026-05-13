"""
ViT feature extractor — Section II of arXiv:2101.03771.

Pipeline:
    image → resize 384×384 → normalize [-1, 1] → ViT encoder (no head) → descriptor
"""

from __future__ import annotations

import os

import certifi
import numpy as np
from PIL import Image
from tqdm import tqdm

# vit-keras 0.1.x requires Keras 2 (tf-keras shim) and certifi SSL certs on macOS.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from vit_keras import vit  # noqa: E402

from config import IMAGE_SIZE, SUPPORTED_MODELS

# Pillow >= 10 removed legacy constants; this resolves correctly across versions.
_RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR

_BUILDERS = {
    "vit_b16": vit.vit_b16,
    "vit_b32": vit.vit_b32,
    "vit_l16": vit.vit_l16,
    "vit_l32": vit.vit_l32,
}

_OUTPUT_DIM = {
    "vit_b16": 768,
    "vit_b32": 768,
    "vit_l16": 1024,
    "vit_l32": 1024,
}


def load_model(model_name: str = "vit_b16"):
    """Return pre-trained ViT encoder without the classification head."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODELS}")

    model = _BUILDERS[model_name](
        image_size=IMAGE_SIZE,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    print(f"Loaded {model_name}  (output dim: {_OUTPUT_DIM[model_name]})")
    return model


def preprocess(image_path: str) -> np.ndarray:
    """Open, resize to IMAGE_SIZE×IMAGE_SIZE, and normalize pixels to [-1, 1]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), _RESAMPLE)
    arr = np.array(img, dtype=np.float32)
    return (arr - 127.5) / 127.5


def extract_features(
    image_paths: list[str],
    model,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Extract raw ViT descriptors for a list of image paths.

    Returns
    -------
    np.ndarray  shape (N, D),  D = 768 for ViT-B,  1024 for ViT-L.
    """
    batches = [
        image_paths[i : i + batch_size]
        for i in range(0, len(image_paths), batch_size)
    ]
    features = []
    for batch in tqdm(batches, desc="Extracting features"):
        arr = np.stack([preprocess(p) for p in batch])
        features.append(model.predict(arr, verbose=0))

    return np.vstack(features)
