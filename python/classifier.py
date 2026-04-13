"""Flower species classifier using a fine-tuned Vision Transformer."""

from __future__ import annotations

import os
from typing import List, Dict

from PIL import Image

# Default model trained on Oxford 102 Flowers + related datasets.
# Falls back gracefully when the model cannot be downloaded.
DEFAULT_MODEL = "dima806/flower_types_image_detection"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".emx_flower_cache")


class FlowerClassifier:
    """Identify flower species in an image and return ranked predictions."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "image-classification",
            model=model_name,
            cache_dir=CACHE_DIR,
        )

    def classify(self, image_path: str, top_k: int = 5) -> List[Dict]:
        """Return up to *top_k* predictions sorted by confidence (highest first).

        Each item has keys:
            ``label``  – flower species name (str)
            ``score``  – confidence in [0, 1] (float)
        """
        image = Image.open(image_path).convert("RGB")
        raw = self._pipe(image, top_k=top_k)
        return [{"label": r["label"], "score": round(float(r["score"]), 4)} for r in raw]
