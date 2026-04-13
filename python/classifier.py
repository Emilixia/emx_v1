"""Flower species classifier using a fine-tuned Vision Transformer."""

from __future__ import annotations

import os
from typing import List, Dict

from PIL import Image

# Custom cache directory — set via HF_HOME so all huggingface_hub / transformers
# code picks it up automatically.  This avoids passing cache_dir= to pipeline(),
# which is no longer a valid keyword argument in recent transformers releases.
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".emx_flower_cache")
os.environ.setdefault("HF_HOME", CACHE_DIR)
# Suppress the Windows symlink warning that clutters the error log.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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


def _best_device() -> int:
    """Return 0 (first CUDA GPU) when available, otherwise -1 (CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:  # noqa: BLE001
        pass
    return -1


class FlowerClassifier:
    """Identify flower species in an image and return ranked predictions."""

    def __init__(self, model_name: str | None = None) -> None:
        from transformers import pipeline

        device = _best_device()
        candidates = [model_name] if model_name else _CANDIDATE_MODELS
        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                self._pipe = pipeline(
                    "image-classification",
                    model=candidate,
                    device=device,
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

    def classify_crop(self, image: Image.Image, box: Dict, top_k: int = 3) -> List[Dict]:
        """Classify a specific region of an already-opened PIL image.

        The region defined by *box* (keys ``xmin``, ``ymin``, ``xmax``, ``ymax``)
        is cropped, padded to a square to avoid distortion, and then classified.

        Returns up to *top_k* predictions, same format as :meth:`classify`.
        """
        crop = image.crop((
            int(box["xmin"]), int(box["ymin"]),
            int(box["xmax"]), int(box["ymax"]),
        ))
        # Pad to square so the model receives a properly proportioned input.
        w, h = crop.size
        size = max(w, h)
        square = Image.new("RGB", (size, size), (255, 255, 255))
        square.paste(crop, ((size - w) // 2, (size - h) // 2))
        raw = self._pipe(square, top_k=top_k)
        return [{"label": r["label"], "score": round(float(r["score"]), 4)} for r in raw]
