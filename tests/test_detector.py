"""Tests for python/detector.py."""

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Pre-import real cv2 and PIL.Image at module level before any patch.dict call
# to prevent cv2 circular-import errors when sys.modules is temporarily patched.
import cv2  # noqa: F401
from PIL import Image  # noqa: F401


def _make_transformers_stub(detections=None):
    """Return a minimal stub for the `transformers` package."""
    if detections is None:
        detections = [
            {
                "score": 0.85,
                "label": "flower",
                "box": {"xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100},
            },
            {
                "score": 0.75,
                "label": "rose",
                "box": {"xmin": 200, "ymin": 50, "xmax": 300, "ymax": 150},
            },
        ]

    transformers = types.ModuleType("transformers")

    def pipeline(task, model=None, **kwargs):
        mock_pipe = MagicMock()
        mock_pipe.return_value = detections
        return mock_pipe

    transformers.pipeline = pipeline
    return transformers


class TestFlowerDetector(unittest.TestCase):
    def _make_detector(self, detections=None):
        stub = _make_transformers_stub(detections)
        self._patch = patch.dict(sys.modules, {"transformers": stub})
        self._patch.start()
        sys.path.insert(0, "python")
        import detector as det_mod
        importlib.reload(det_mod)
        self._det_mod = det_mod
        return det_mod.FlowerDetector()

    def tearDown(self):
        if hasattr(self, "_patch"):
            self._patch.stop()
        if sys.path and sys.path[0] == "python":
            sys.path.pop(0)

    # ── detect() ─────────────────────────────────────────────────────────────

    def test_detect_returns_list(self):
        det = self._make_detector()
        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_open.return_value = mock_img
            results = det.detect("dummy.jpg")
        self.assertIsInstance(results, list)

    def test_detect_items_have_required_keys(self):
        det = self._make_detector()
        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_open.return_value = mock_img
            results = det.detect("dummy.jpg")
        for item in results:
            self.assertIn("score", item)
            self.assertIn("label", item)
            self.assertIn("box", item)

    def test_detect_applies_nms(self):
        # Two heavily overlapping boxes → NMS should keep only the higher-score one
        overlapping = [
            {
                "score": 0.9,
                "label": "flower",
                "box": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
            },
            {
                "score": 0.7,
                "label": "rose",
                "box": {"xmin": 5, "ymin": 5, "xmax": 95, "ymax": 95},
            },
        ]
        det = self._make_detector(detections=overlapping)
        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_open.return_value = mock_img
            results = det.detect("dummy.jpg")
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["score"], 0.9)

    # ── _iou() ────────────────────────────────────────────────────────────────

    def test_iou_identical_boxes(self):
        det = self._make_detector()
        box = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        self.assertAlmostEqual(self._det_mod._iou(box, box), 1.0)

    def test_iou_non_overlapping(self):
        self._make_detector()
        a = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        b = {"xmin": 20, "ymin": 20, "xmax": 30, "ymax": 30}
        self.assertAlmostEqual(self._det_mod._iou(a, b), 0.0)

    def test_iou_partial_overlap(self):
        self._make_detector()
        a = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        b = {"xmin": 5, "ymin": 5, "xmax": 15, "ymax": 15}
        iou = self._det_mod._iou(a, b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    # ── _nms() ────────────────────────────────────────────────────────────────

    def test_nms_removes_overlapping_lower_score(self):
        self._make_detector()
        detections = [
            {
                "score": 0.9,
                "label": "flower",
                "box": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100},
            },
            {
                "score": 0.7,
                "label": "rose",
                "box": {"xmin": 5, "ymin": 5, "xmax": 95, "ymax": 95},
            },
        ]
        kept = self._det_mod._nms(detections, iou_threshold=0.5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["score"], 0.9)

    def test_nms_keeps_non_overlapping(self):
        self._make_detector()
        detections = [
            {
                "score": 0.9,
                "label": "flower",
                "box": {"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50},
            },
            {
                "score": 0.8,
                "label": "rose",
                "box": {"xmin": 200, "ymin": 200, "xmax": 250, "ymax": 250},
            },
        ]
        kept = self._det_mod._nms(detections, iou_threshold=0.5)
        self.assertEqual(len(kept), 2)

    def test_nms_empty_input(self):
        self._make_detector()
        self.assertEqual(self._det_mod._nms([]), [])


if __name__ == "__main__":
    unittest.main()
