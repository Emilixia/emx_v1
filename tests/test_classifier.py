"""
test_classifier.py
------------------
Unit tests for app.core.flower_classifier and app.utils.image_loader.

These tests use a mock ONNX session so they run without a real model file.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ── Make the repo root importable ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pre-import cv2, PIL and numpy so that patch.dict(sys.modules, ...) does not
# remove them when it restores state after each test (patch.dict removes any
# keys that were not present in sys.modules at patch entry time).
import cv2  # noqa: E402
import PIL.Image  # noqa: E402


# ── Stub onnxruntime before importing the classifier ─────────────────────────


def _make_ort_stub(logits_shape=(1, 102)):
    """Return a minimal fake onnxruntime module."""
    ort_stub = types.ModuleType("onnxruntime")

    class FakeInput:
        name = "input"

    class FakeOutput:
        name = "logits"

    class FakeSession:
        def __init__(self, path, providers=None):
            pass

        def get_inputs(self):
            return [FakeInput()]

        def get_outputs(self):
            return [FakeOutput()]

        def run(self, output_names, feed):
            # Return uniform logits so softmax spreads evenly.
            return [np.ones(logits_shape, dtype=np.float32)]

    ort_stub.InferenceSession = FakeSession
    return ort_stub


class TestFlowerClassifierImport(unittest.TestCase):
    """Verify FlowerClassifier can be instantiated with a fake model."""

    def setUp(self):
        self._ort_patcher = patch.dict(
            "sys.modules", {"onnxruntime": _make_ort_stub()}
        )
        self._ort_patcher.start()

        # Patch Path.exists so the model file appears to exist
        self._exists_patcher = patch.object(Path, "exists", return_value=True)
        self._exists_patcher.start()

        # Now import (fresh)
        if "app.core.flower_classifier" in sys.modules:
            del sys.modules["app.core.flower_classifier"]
        from app.core.flower_classifier import FlowerClassifier, FLOWER_CLASSES

        self.FlowerClassifier = FlowerClassifier
        self.FLOWER_CLASSES = FLOWER_CLASSES

    def tearDown(self):
        self._ort_patcher.stop()
        self._exists_patcher.stop()

    def test_flower_classes_length(self):
        self.assertEqual(len(self.FLOWER_CLASSES), 102)

    def test_flower_classes_unique(self):
        self.assertEqual(len(self.FLOWER_CLASSES), len(set(self.FLOWER_CLASSES)))

    def test_predict_returns_top_k(self):
        clf = self.FlowerClassifier.__new__(self.FlowerClassifier)
        # Manually initialise without __init__ to avoid file-system checks
        import onnxruntime as ort

        clf._session = ort.InferenceSession("")
        clf._input_name = "input"
        clf._output_name = "logits"

        from PIL import Image

        dummy_image = Image.new("RGB", (224, 224), color=(100, 150, 200))
        results = clf.predict(dummy_image, top_k=5)

        self.assertEqual(len(results), 5)
        for species, conf in results:
            self.assertIsInstance(species, str)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_predict_confidences_sum_to_one(self):
        clf = self.FlowerClassifier.__new__(self.FlowerClassifier)
        import onnxruntime as ort

        clf._session = ort.InferenceSession("")
        clf._input_name = "input"
        clf._output_name = "logits"

        from PIL import Image

        img = Image.new("RGB", (224, 224))
        results = clf.predict(img, top_k=102)  # all classes
        total = sum(conf for _, conf in results)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_predict_top1_convenience(self):
        clf = self.FlowerClassifier.__new__(self.FlowerClassifier)
        import onnxruntime as ort

        clf._session = ort.InferenceSession("")
        clf._input_name = "input"
        clf._output_name = "logits"

        from PIL import Image

        img = Image.new("RGB", (100, 100))
        species, conf = clf.predict_top1(img)
        self.assertIsInstance(species, str)
        self.assertIsInstance(conf, float)


class TestImageLoader(unittest.TestCase):
    """Tests for app.utils.image_loader preprocessing helpers."""

    def test_preprocess_for_classifier_shape(self):
        from PIL import Image

        from app.utils.image_loader import preprocess_for_classifier

        img = Image.new("RGB", (320, 480))
        tensor = preprocess_for_classifier(img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)

    def test_preprocess_for_detector_shape(self):
        from PIL import Image

        from app.utils.image_loader import preprocess_for_detector

        img = Image.new("RGB", (800, 600))
        tensor = preprocess_for_detector(img)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)

    def test_preprocess_range_classifier(self):
        """After ImageNet normalisation values can exceed [0,1] — just check dtype."""
        from PIL import Image

        from app.utils.image_loader import preprocess_for_classifier

        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        tensor = preprocess_for_classifier(img)
        self.assertEqual(tensor.dtype, np.float32)

    def test_is_supported(self):
        from app.utils.image_loader import is_supported

        self.assertTrue(is_supported("photo.jpg"))
        self.assertTrue(is_supported("photo.PNG"))
        self.assertFalse(is_supported("document.pdf"))
        self.assertFalse(is_supported("data.csv"))

    def test_load_pil_not_found(self):
        from app.utils.image_loader import load_pil

        with self.assertRaises(FileNotFoundError):
            load_pil("/nonexistent/path/image.jpg")

    def test_load_pil_unsupported_ext(self):
        import tempfile

        from app.utils.image_loader import load_pil

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_pil(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestReportGenerator(unittest.TestCase):
    """Tests for CSV export."""

    def test_export_csv(self):
        import tempfile

        from app.utils.report_generator import AnalysisResult, export_csv

        results = [
            AnalysisResult(
                image_path=Path("rose.jpg"),
                flower_species="rose",
                confidence=0.92,
                stem_count=3,
            ),
            AnalysisResult(
                image_path=Path("tulip.png"),
                flower_species="tulip",
                confidence=0.85,
                stem_count=1,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results.csv"
            export_csv(results, out)
            self.assertTrue(out.exists())
            content = out.read_text()
            self.assertIn("rose", content)
            self.assertIn("tulip", content)
            self.assertIn("stem_count", content)


if __name__ == "__main__":
    unittest.main()
