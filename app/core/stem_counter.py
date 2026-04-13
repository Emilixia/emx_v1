"""
stem_counter.py
---------------
Inference wrapper for the YOLOv8 stem-detection model.

Expected model
--------------
A YOLOv8 model (``stem_detector.pt`` or ``stem_detector.onnx``) fine-tuned on
a dataset where the sole class is ``stem``.

The model file should be placed at ``app/models/stem_detector.pt``.
To obtain the model, train it using ``training/train_stem_detector.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ── Optional Ultralytics import ───────────────────────────────────────────────
try:
    from ultralytics import YOLO

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "stem_detector.pt"

# Bounding-box colour for stem annotations (BGR for OpenCV).
_BOX_COLOR_BGR = (0, 200, 50)
_BOX_THICKNESS = 2
_LABEL_FONT_SCALE = 0.55


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class StemDetection:
    """A single detected stem."""

    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords
    confidence: float
    label: str = "stem"


@dataclass
class StemCountResult:
    """Aggregated result for one image."""

    count: int
    detections: List[StemDetection] = field(default_factory=list)


# ── Main class ────────────────────────────────────────────────────────────────


class StemCounter:
    """YOLOv8-based stem detector and counter.

    Parameters
    ----------
    model_path:
        Path to ``stem_detector.pt`` (or ``.onnx`` / ``.yaml``).
    confidence_threshold:
        Minimum confidence score for a detection to be counted.
    device:
        Torch device string, e.g. ``"cpu"``, ``"cuda:0"``, or ``"0"`` for the
        first GPU.  ``None`` lets Ultralytics choose automatically.

    Example
    -------
    >>> counter = StemCounter()
    >>> result = counter.count("flower.jpg")
    >>> print(f"Detected {result.count} stem(s)")
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            )
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Stem detection model not found at '{self.model_path}'.\n"
                "Train it with:  python training/train_stem_detector.py"
            )
        self.confidence_threshold = confidence_threshold
        self._model = YOLO(str(self.model_path))
        self._device = device

    # ── Inference ─────────────────────────────────────────────────────────────

    def count(self, image) -> StemCountResult:
        """Detect and count stems in *image*.

        Parameters
        ----------
        image:
            File path (str/Path), PIL Image, or NumPy HWC array (RGB or BGR).

        Returns
        -------
        :class:`StemCountResult` with ``count`` and per-detection details.
        """
        if isinstance(image, (str, Path)):
            from app.utils.image_loader import load_pil
            image = load_pil(image)

        results = self._model.predict(
            image,
            conf=self.confidence_threshold,
            device=self._device,
            verbose=False,
        )

        detections: List[StemDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    StemDetection(
                        box=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                    )
                )

        return StemCountResult(count=len(detections), detections=detections)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def draw_detections(
        self,
        image: Image.Image | np.ndarray,
        result: StemCountResult,
    ) -> np.ndarray:
        """Overlay detection bounding-boxes onto *image*.

        Parameters
        ----------
        image:
            Original image (PIL or NumPy BGR).
        result:
            Output of :meth:`count`.

        Returns
        -------
        NumPy BGR array with boxes drawn.
        """
        if isinstance(image, Image.Image):
            canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            canvas = image.copy()

        for det in result.detections:
            x1, y1, x2, y2 = det.box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _BOX_COLOR_BGR, _BOX_THICKNESS)
            label_text = f"stem {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, 1
            )
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), _BOX_COLOR_BGR, -1)
            cv2.putText(
                canvas,
                label_text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                _LABEL_FONT_SCALE,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Total count banner
        banner = f"Stems: {result.count}"
        cv2.putText(
            canvas,
            banner,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            _BOX_COLOR_BGR,
            2,
            cv2.LINE_AA,
        )
        return canvas
