"""
test_stem_counter.py
--------------------
Unit tests for app.core.stem_counter.

A fake Ultralytics YOLO object is injected so tests run without GPU or model
weights.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pre-import cv2 and PIL so patch.dict(sys.modules, ...) does not remove them
# when it restores state (patch.dict removes newly-added keys on exit).
import cv2  # noqa: E402
import PIL.Image  # noqa: E402


# ── Fake Ultralytics stubs ────────────────────────────────────────────────────


def _make_fake_box(x1, y1, x2, y2, conf):
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [float(x1), float(y1), float(x2), float(y2)]
    box.conf = [MagicMock()]
    box.conf[0].__float__ = lambda self: conf
    return box


def _make_fake_result(boxes_data):
    """boxes_data: list of (x1,y1,x2,y2,conf) tuples."""
    result = MagicMock()
    result.boxes = MagicMock()
    fake_boxes = [_make_fake_box(*d) for d in boxes_data]
    result.boxes.__iter__ = lambda self: iter(fake_boxes)
    result.boxes.__bool__ = lambda self: True
    return result


def _make_ultralytics_stub(detections):
    """Return a fake ultralytics module whose YOLO.predict returns *detections*."""
    ult_stub = types.ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, path):
            self._detections = detections

        def predict(self, image, conf=0.25, device=None, verbose=False):
            return [_make_fake_result(self._detections)]

    ult_stub.YOLO = FakeYOLO
    return ult_stub


# ── Helper: build a StemCounter backed by a given stub ────────────────────────


def _build_counter(ult_stub):
    """Import StemCounter while the stub is in sys.modules, then wire it up."""
    for mod in list(sys.modules):
        if "stem_counter" in mod:
            del sys.modules[mod]

    # stem_counter is imported here while the stub is active in sys.modules
    from app.core.stem_counter import StemCounter

    counter = StemCounter.__new__(StemCounter)
    counter.confidence_threshold = 0.25
    counter._device = None
    counter._model = ult_stub.YOLO("")
    return counter


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestStemCounterBasic(unittest.TestCase):
    """Tests run with a per-test ultralytics stub kept active for the whole test."""

    def _run_with_stub(self, detections, test_fn):
        """Activate *ult_stub* for sys.modules, build a counter, call *test_fn*."""
        ult_stub = _make_ultralytics_stub(detections)
        with patch.dict("sys.modules", {"ultralytics": ult_stub}):
            counter = _build_counter(ult_stub)
            test_fn(counter)

    def test_count_zero_stems(self):
        def _test(counter):
            from PIL import Image
            img = Image.new("RGB", (640, 640), color=(200, 180, 160))
            result = counter.count(img)
            self.assertEqual(result.count, 0)
            self.assertEqual(result.detections, [])

        self._run_with_stub([], _test)

    def test_count_multiple_stems(self):
        detections = [
            (10, 20, 50, 80, 0.9),
            (100, 150, 140, 220, 0.75),
            (300, 10, 340, 90, 0.6),
        ]

        def _test(counter):
            from PIL import Image
            img = Image.new("RGB", (640, 640))
            result = counter.count(img)
            self.assertEqual(result.count, 3)
            self.assertEqual(len(result.detections), 3)

        self._run_with_stub(detections, _test)

    def test_detection_box_values(self):
        detections = [(5, 10, 50, 100, 0.8)]

        def _test(counter):
            from PIL import Image
            img = Image.new("RGB", (640, 640))
            result = counter.count(img)
            det = result.detections[0]
            self.assertEqual(det.box, (5, 10, 50, 100))
            self.assertAlmostEqual(det.confidence, 0.8, places=3)
            self.assertEqual(det.label, "stem")

        self._run_with_stub(detections, _test)


class TestDrawDetections(unittest.TestCase):
    """Verify draw_detections returns a valid BGR ndarray."""

    def test_draw_returns_ndarray(self):
        ult_stub = _make_ultralytics_stub([])
        with patch.dict("sys.modules", {"ultralytics": ult_stub}):
            counter = _build_counter(ult_stub)
            from app.core.stem_counter import StemCountResult, StemDetection

            img = PIL.Image.new("RGB", (200, 200), color=(100, 150, 200))
            result = StemCountResult(
                count=1,
                detections=[StemDetection(box=(10, 10, 80, 160), confidence=0.9)],
            )
            canvas = counter.draw_detections(img, result)

        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape, (200, 200, 3))

    def test_draw_with_numpy_input(self):
        ult_stub = _make_ultralytics_stub([])
        with patch.dict("sys.modules", {"ultralytics": ult_stub}):
            counter = _build_counter(ult_stub)
            from app.core.stem_counter import StemCountResult

            bgr = np.zeros((300, 400, 3), dtype=np.uint8)
            result = StemCountResult(count=0)
            canvas = counter.draw_detections(bgr, result)

        self.assertEqual(canvas.shape, (300, 400, 3))


if __name__ == "__main__":
    unittest.main()

