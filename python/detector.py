"""Detect individual flower instances using OWL-ViT zero-shot object detection."""

from __future__ import annotations

import os
from typing import List, Dict

from PIL import Image

# Share the same HuggingFace cache directory used by classifier.py.
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".emx_flower_cache")
os.environ.setdefault("HF_HOME", CACHE_DIR)
# Suppress the Windows symlink warning that shows up as noise in the error log.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Text queries sent to the zero-shot detector.  A broad "flower" query finds
# arrangements in general; the species-specific queries help the detector focus
# on individual bloom heads for better per-instance detection.
_FLOWER_QUERIES: List[str] = [
    "flower",
    "flower head",
    "flower bloom",
    "rose",
    "tulip",
    "daisy",
    "sunflower",
    "lily",
    "peony",
    "orchid",
    "carnation",
    "ranunculus",
    "poppy",
    "chrysanthemum",
    "iris",
    "daffodil",
    "lavender",
    "hydrangea",
    "anemone",
]

_DETECTOR_MODELS: List[str] = [
    "google/owlvit-base-patch32",
    "google/owlvit-large-patch14",
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


def _iou(a: Dict, b: Dict) -> float:
    """Intersection over Union for two boxes, each with xmin/ymin/xmax/ymax."""
    ix1 = max(a["xmin"], b["xmin"])
    iy1 = max(a["ymin"], b["ymin"])
    ix2 = min(a["xmax"], b["xmax"])
    iy2 = min(a["ymax"], b["ymax"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a["xmax"] - a["xmin"]) * (a["ymax"] - a["ymin"])
    area_b = (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"])
    return inter / (area_a + area_b - inter)


def _nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Non-maximum suppression: keep highest-score non-overlapping detections."""
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: List[Dict] = []
    for det in detections:
        if all(_iou(det["box"], k["box"]) < iou_threshold for k in kept):
            kept.append(det)
    return kept


class FlowerDetector:
    """Locate individual flower heads in an image using zero-shot object detection."""

    def __init__(self) -> None:
        from transformers import pipeline

        device = _best_device()
        last_exc: Exception | None = None
        for candidate in _DETECTOR_MODELS:
            try:
                self._pipe = pipeline(
                    "zero-shot-object-detection",
                    model=candidate,
                    device=device,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

        raise RuntimeError(
            f"Could not load any flower detection model. Last error: {last_exc}"
        )

    def detect(self, image_path: str, threshold: float = 0.1) -> List[Dict]:
        """Return a deduplicated list of detected flower instances.

        Each item has keys:
            ``score``  – detection confidence in [0, 1] (float)
            ``label``  – matched query string (str)
            ``box``    – bounding box dict with ``xmin``, ``ymin``, ``xmax``, ``ymax`` (int)
        """
        image = Image.open(image_path).convert("RGB")
        try:
            raw = self._pipe(image, candidate_labels=_FLOWER_QUERIES, threshold=threshold)
        except TypeError:
            # Older pipeline versions may not accept `threshold` as a kwarg.
            raw = self._pipe(image, candidate_labels=_FLOWER_QUERIES)
            raw = [d for d in raw if d.get("score", 0.0) >= threshold]
        return _nms(raw, iou_threshold=0.5)
