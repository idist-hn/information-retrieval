"""Central configuration — single source of truth for all constants."""

IMAGE_SIZE = 384  # Section II: all images resized to 384×384

DEFAULT_MODEL = "vit_b16"
DEFAULT_NORM = "robust"
DEFAULT_METRIC = "cosine"
DEFAULT_BATCH_SIZE = 16

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SUPPORTED_MODELS = ["vit_b16", "vit_b32", "vit_l16", "vit_l32"]

SUPPORTED_NORMS = ["robust", "l2_axis1", "l1_axis1", "l2_axis0", "l1_axis0"]

SUPPORTED_METRICS = [
    "cosine", "euclidean", "cityblock",
    "braycurtis", "canberra", "chebyshev", "correlation",
]
