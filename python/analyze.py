"""Main entry point for the Python analysis backend.

Usage:
    python analyze.py <image_path>

Prints a single JSON object to stdout:
    {
        "flowers": [{"label": "...", "score": 0.91}, ...],
        "species_counts": [{"label": "rose", "count": 7, "score": 0.91}, ...],
        "stem_count": 10,
        "stem_method": "..."
    }

``species_counts`` is present when the detection-based pipeline succeeds.
``flowers`` is always present (top species from species_counts, or whole-image
predictions on fallback) and is used for the Wikipedia lookup in the UI.

Errors are printed to stderr and the process exits with code 1.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        _error("Usage: python analyze.py <image_path>")

    image_path = sys.argv[1]

    flowers = None
    species_counts = None
    stem_count = None
    stem_method = None

    # ── Stage 1: Detection-based pipeline ────────────────────────────────────
    # OWL-ViT locates individual flower heads → each crop is classified by the
    # fine-tuned ViT → results are aggregated into a per-species histogram.
    try:
        from PIL import Image
        from detector import FlowerDetector
        from classifier import FlowerClassifier

        detector = FlowerDetector()
        classifier = FlowerClassifier()

        detections = detector.detect(image_path, threshold=0.1)

        if detections:
            image_pil = Image.open(image_path).convert("RGB")
            tally: dict = {}  # label -> {"count": int, "scores": list[float]}

            for det in detections:
                preds = classifier.classify_crop(image_pil, det["box"], top_k=1)
                if preds:
                    label = preds[0]["label"].lower().strip()
                    tally.setdefault(label, {"count": 0, "scores": []})
                    tally[label]["count"] += 1
                    tally[label]["scores"].append(preds[0]["score"])

            species_counts = sorted(
                [
                    {
                        "label": label,
                        "count": data["count"],
                        "score": round(
                            sum(data["scores"]) / len(data["scores"]), 4
                        ),
                    }
                    for label, data in tally.items()
                ],
                key=lambda x: x["count"],
                reverse=True,
            )

            flowers = [
                {"label": s["label"], "score": s["score"]} for s in species_counts
            ]
            stem_count = len(detections)
            stem_method = (
                "OWL-ViT zero-shot detection + ViT classification per bloom"
            )
    except Exception as exc:  # noqa: BLE001
        # Log why the detection pipeline failed so it's visible in the error log.
        print(f"Detection pipeline unavailable ({exc}); using whole-image fallback.", file=sys.stderr)

    # ── Stage 2: Whole-image fallback ─────────────────────────────────────────
    if flowers is None:
        try:
            from classifier import FlowerClassifier

            classifier = FlowerClassifier()
            flowers = classifier.classify(image_path, top_k=5)
        except Exception as exc:
            _error(f"Flower classification failed: {exc}")

    if stem_count is None:
        try:
            from stem_counter import StemCounter

            counter = StemCounter()
            stem_count, stem_method = counter.count_stems(image_path)
        except Exception as exc:
            _error(f"Stem counting failed: {exc}")

    # ── Output ────────────────────────────────────────────────────────────────
    result: dict = {
        "flowers": flowers,
        "stem_count": stem_count,
        "stem_method": stem_method,
    }
    if species_counts:
        result["species_counts"] = species_counts

    print(json.dumps(result, ensure_ascii=False))


def _error(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
