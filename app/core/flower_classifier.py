"""
flower_classifier.py
--------------------
Inference wrapper for the ONNX EfficientNet-B0 flower-identification model.

Expected model
--------------
An ONNX model with:
  • Input  – float32 NCHW tensor (1, 3, 224, 224), ImageNet-normalised
  • Output – float32 logits tensor (1, 102) for the Oxford 102 Flowers classes

The model file should be placed at ``app/models/flower_id.onnx``.
To obtain the model, either train it using ``training/train_flower_classifier.py``
or convert a pre-trained checkpoint with ``torch.onnx.export``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.utils.image_loader import preprocess_for_classifier

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# ── Oxford 102 Flowers class names ────────────────────────────────────────────
# Source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html
FLOWER_CLASSES: List[str] = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barberton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia",
    "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "pink quill", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]

assert len(FLOWER_CLASSES) == 102, "FLOWER_CLASSES must contain exactly 102 entries."

# ── Default model path ─────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "flower_id.onnx"


class FlowerClassifier:
    """ONNX-based flower species classifier.

    Parameters
    ----------
    model_path:
        Path to the ``flower_id.onnx`` model file.
        Defaults to ``app/models/flower_id.onnx``.
    providers:
        ONNX Runtime execution providers, e.g.
        ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` for GPU.
        Defaults to CPU only.

    Example
    -------
    >>> clf = FlowerClassifier()
    >>> results = clf.predict("rose.jpg", top_k=3)
    >>> for species, conf in results:
    ...     print(f"{species}: {conf:.2%}")
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        providers: Optional[List[str]] = None,
    ) -> None:
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Flower classification model not found at '{self.model_path}'.\n"
                "Train it with:  python training/train_flower_classifier.py\n"
                "or download a pre-trained checkpoint and export to ONNX."
            )
        self._providers = providers or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(self.model_path), providers=self._providers
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Classify *image* and return the top-*k* flower species.

        Parameters
        ----------
        image:
            A file path (str/Path), PIL Image, or NumPy HWC uint8 RGB array.
        top_k:
            Number of top predictions to return.

        Returns
        -------
        List of ``(species_name, confidence)`` tuples sorted by confidence
        descending.
        """
        if isinstance(image, (str, Path)):
            from app.utils.image_loader import load_pil
            image = load_pil(image)

        input_tensor = preprocess_for_classifier(image)  # (1, 3, 224, 224)
        logits = self._session.run(
            [self._output_name], {self._input_name: input_tensor}
        )[0][0]  # shape: (102,)

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(FLOWER_CLASSES[i], float(probs[i])) for i in top_indices]

    def predict_top1(self, image) -> Tuple[str, float]:
        """Convenience method: return ``(species, confidence)`` for the best match."""
        results = self.predict(image, top_k=1)
        return results[0]
