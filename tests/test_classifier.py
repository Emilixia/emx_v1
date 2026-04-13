"""Tests for python/classifier.py."""

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Pre-import real cv2 and PIL.Image at module level before any patch.dict call to
# prevent cv2 circular import errors. patch.dict removes newly-added keys on exit.
import cv2  # noqa: F401
from PIL import Image  # noqa: F401


def _make_transformers_stub():
    """Return a minimal stub for the `transformers` package."""
    transformers = types.ModuleType("transformers")

    def pipeline(task, model=None, **kwargs):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "rose", "score": 0.92},
            {"label": "tulip", "score": 0.05},
        ]
        return mock_pipe

    transformers.pipeline = pipeline
    return transformers


class TestFlowerClassifier(unittest.TestCase):
    def setUp(self):
        self._patch = patch.dict(sys.modules, {"transformers": _make_transformers_stub()})
        self._patch.start()
        sys.path.insert(0, "python")
        import classifier as clf_mod
        importlib.reload(clf_mod)
        self._clf = clf_mod.FlowerClassifier()

    def tearDown(self):
        self._patch.stop()
        if sys.path and sys.path[0] == "python":
            sys.path.pop(0)

    def _classify(self, top_k=5):
        with patch("PIL.Image.open") as mock_open:
            mock_img = MagicMock()
            mock_img.convert.return_value = mock_img
            mock_open.return_value = mock_img
            return self._clf.classify("dummy.jpg", top_k=top_k)

    def test_classify_returns_list(self):
        results = self._classify(top_k=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIn("label", results[0])
        self.assertIn("score", results[0])
        self.assertEqual(results[0]["label"], "rose")
        self.assertAlmostEqual(results[0]["score"], 0.92)

    def test_classify_score_is_float(self):
        results = self._classify()
        for r in results:
            self.assertIsInstance(r["score"], float)

    def test_classify_top_k_respected(self):
        results = self._classify(top_k=2)
        # Stub returns 2 items; result must not exceed top_k
        self.assertLessEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
