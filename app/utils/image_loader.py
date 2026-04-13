"""
image_loader.py
---------------
Utilities for loading and preprocessing images before passing them to the
flower-identification and stem-counting models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────

# Standard ImageNet normalisation (used by EfficientNet / ResNet backbones).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default spatial size expected by EfficientNet-B0.
CLASSIFIER_INPUT_SIZE: Tuple[int, int] = (224, 224)

# YOLOv8 typically works well at 640×640.
DETECTOR_INPUT_SIZE: Tuple[int, int] = (640, 640)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Public helpers ─────────────────────────────────────────────────────────────


def load_pil(path: str | Path) -> Image.Image:
    """Load an image from *path* as an RGB PIL Image.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not in :data:`SUPPORTED_EXTENSIONS`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return Image.open(path).convert("RGB")


def load_cv2(path: str | Path) -> np.ndarray:
    """Load an image as a NumPy array in BGR format (OpenCV convention)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cv2 could not read image: {path}")
    return img


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a cv2 BGR ndarray."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a cv2 BGR ndarray to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def preprocess_for_classifier(
    image: Image.Image | np.ndarray,
    input_size: Tuple[int, int] = CLASSIFIER_INPUT_SIZE,
) -> np.ndarray:
    """Resize, normalise, and batch-expand an image for the ONNX classifier.

    Parameters
    ----------
    image:
        PIL RGB Image or NumPy HWC uint8 array (RGB).
    input_size:
        ``(height, width)`` expected by the model.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, 3, H, W)`` (NCHW), normalised with
        ImageNet mean/std.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    image = image.convert("RGB").resize((input_size[1], input_size[0]), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD  # normalise
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return np.expand_dims(arr, axis=0)  # CHW → NCHW


def preprocess_for_detector(
    image: Image.Image | np.ndarray,
    input_size: Tuple[int, int] = DETECTOR_INPUT_SIZE,
) -> np.ndarray:
    """Resize and normalise an image for the YOLOv8 stem detector.

    Returns
    -------
    np.ndarray
        Float32 NCHW array with values in ``[0, 1]``.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    resized = cv2.resize(image, (input_size[1], input_size[0]))
    arr = resized.astype(np.float32) / 255.0  # HWC, [0, 1]
    arr = arr.transpose(2, 0, 1)  # CHW
    return np.expand_dims(arr, axis=0)  # NCHW


def is_supported(path: str | Path) -> bool:
    """Return ``True`` if *path* has a supported image extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def collect_images(directory: str | Path) -> list[Path]:
    """Recursively collect all supported images under *directory*."""
    directory = Path(directory)
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and is_supported(p)
    )
