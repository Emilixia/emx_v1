"""Tests for python/stem_counter.py."""

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Pre-import real cv2 and PIL.Image at module level before any patch.dict call
# to prevent cv2 circular-import errors when sys.modules is temporarily patched.
import cv2  # noqa: F401
from PIL import Image  # noqa: F401

import numpy as np


def _make_contour(area, bbox):
    """Helper to create a fake contour dict understood by the cv2 stub."""
    c = MagicMock()
    c.get = lambda k, d=None: {"area": area, "bbox": bbox}.get(k, d)
    return c


def _make_cv2_stub(contours):
    """Return a cv2-like stub with controlled contour output."""
    cv2_stub = types.ModuleType("cv2")

    cv2_stub.COLOR_BGR2HSV = 40
    cv2_stub.cvtColor = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    cv2_stub.inRange = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
    cv2_stub.bitwise_or = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
    cv2_stub.MORPH_CLOSE = 3
    cv2_stub.MORPH_OPEN = 2
    cv2_stub.morphologyEx = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
    cv2_stub.RETR_EXTERNAL = 0
    cv2_stub.CHAIN_APPROX_SIMPLE = 1
    cv2_stub.findContours = MagicMock(return_value=(contours, None))
    cv2_stub.contourArea = MagicMock(side_effect=lambda c: float(c.get("area", 0)))
    cv2_stub.boundingRect = MagicMock(side_effect=lambda c: c.get("bbox", (0, 0, 10, 30)))
    cv2_stub.imread = MagicMock(return_value=np.zeros((200, 200, 3), dtype=np.uint8))

    return cv2_stub


class TestStemCounter(unittest.TestCase):
    def _make_counter(self, contours):
        cv2_stub = _make_cv2_stub(contours)
        self._patch = patch.dict(sys.modules, {"cv2": cv2_stub})
        self._patch.start()
        sys.path.insert(0, "python")
        import stem_counter as sc_mod
        importlib.reload(sc_mod)
        return sc_mod.StemCounter(), cv2_stub

    def tearDown(self):
        if hasattr(self, "_patch"):
            self._patch.stop()
        if sys.path and sys.path[0] == "python":
            sys.path.pop(0)

    def test_returns_tuple(self):
        counter, _ = self._make_counter([])
        result = counter.count_stems("dummy.jpg")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_zero_stems_when_no_contours(self):
        counter, _ = self._make_counter([])
        count, _ = counter.count_stems("dummy.jpg")
        self.assertEqual(count, 0)

    def test_counts_valid_stem_contours(self):
        contours = [
            _make_contour(area=200, bbox=(0, 0, 10, 40)),
            _make_contour(area=200, bbox=(20, 0, 10, 40)),
            _make_contour(area=200, bbox=(40, 0, 10, 40)),
        ]
        counter, _ = self._make_counter(contours)
        count, method = counter.count_stems("dummy.jpg")
        self.assertEqual(count, 3)
        self.assertIsInstance(method, str)
        self.assertTrue(len(method) > 0)

    def test_small_area_contours_excluded(self):
        contours = [_make_contour(area=50, bbox=(0, 0, 5, 20))]
        counter, _ = self._make_counter(contours)
        count, _ = counter.count_stems("dummy.jpg")
        self.assertEqual(count, 0)

    def test_invalid_image_raises(self):
        counter, cv2_stub = self._make_counter([])
        cv2_stub.imread.return_value = None
        with self.assertRaises(ValueError):
            counter.count_stems("nonexistent.jpg")


if __name__ == "__main__":
    unittest.main()
