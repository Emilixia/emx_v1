"""Flower species classifier using a fine-tuned Vision Transformer."""

from __future__ import annotations

import os
from typing import List, Dict

from PIL import Image

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".emx_flower_cache")

# Ordered list of candidate models.  The first one that loads successfully is
# used; this makes the classifier resilient to individual models being removed
# or renamed on the Hugging Face Hub.
_CANDIDATE_MODELS = [
    # ViT fine-tuned on TF-Flowers (daisy, dandelion, roses, sunflowers, tulips)
    "nickmuchi/vit-finetuned-flowers",
    # Broader ViT flower classifier
    "Falconsai/flower_classification",
    # General-purpose ViT (ImageNet-21k) as last resort — has many flower labels
    "google/vit-base-patch16-224",
]


class FlowerClassifier:
    """Identify flower species in an image and return ranked predictions."""

    def __init__(self, model_name: str | None = None) -> None:
        from transformers import pipeline

        candidates = [model_name] if model_name else _CANDIDATE_MODELS
        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                self._pipe = pipeline(
                    "image-classification",
                    model=candidate,
                    cache_dir=CACHE_DIR,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

        raise RuntimeError(
            f"Could not load any flower classification model. "
            f"Please check your internet connection. Last error: {last_exc}"
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
