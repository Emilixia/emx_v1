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

# OWL-ViT query strings that are too generic to use as a species label.
# Detections that match these are re-attributed to the top whole-image species.
_GENERIC_QUERIES = {"flower", "flower head", "flower bloom"}


def main() -> None:
    if len(sys.argv) < 2:
        _error("Usage: python analyze.py <image_path>")

    image_path = sys.argv[1]

    flowers = None
    species_counts = None
    stem_count = None
    stem_method = None

    # ── Stage 1: Whole-image classification ──────────────────────────────────
    # Always run first: provides the species name used as a fallback for
    # generic OWL-ViT matches, and drives the Wikipedia lookup.
    try:
        from classifier import FlowerClassifier

        classifier = FlowerClassifier()
        flowers = classifier.classify(image_path, top_k=5)
    except Exception as exc:
        _error(f"Flower classification failed: {exc}")

    top_species = flowers[0]["label"].lower().strip() if flowers else "flower"

    # ── Stage 2: Detection-based per-bloom counting ───────────────────────────
    # OWL-ViT locates individual flower heads.  Each detection is labelled with
    # whichever of our flower query strings it best matched (e.g. "rose",
    # "ranunculus").  We use that label directly — never run a crop through the
    # image classifier, which would fall back to an ImageNet model and produce
    # nonsense labels like "Vase", "Cup", or "Lotion".
    #
    # Generic matches ("flower", "flower head", "flower bloom") are re-attributed
    # to *top_species* from the whole-image classifier above.
    try:
        from detector import FlowerDetector

        detector = FlowerDetector()
        detections = detector.detect(image_path, threshold=0.1)

        if detections:
            tally: dict = {}  # label -> {"count": int, "max_score": float}

            for det in detections:
                label = det["label"].lower().strip()
                if label in _GENERIC_QUERIES:
                    label = top_species
                entry = tally.setdefault(label, {"count": 0, "max_score": 0.0})
                entry["count"] += 1
                entry["max_score"] = max(entry["max_score"], float(det["score"]))

            species_counts = sorted(
                [
                    {
                        "label": label,
                        "count": data["count"],
                        "score": round(data["max_score"], 4),
                    }
                    for label, data in tally.items()
                ],
                key=lambda x: x["count"],
                reverse=True,
            )

            # Overwrite flowers so the wiki lookup targets the dominant detected species.
            flowers = [
                {"label": s["label"], "score": s["score"]} for s in species_counts
            ]
            stem_count = len(detections)
            stem_method = "OWL-ViT zero-shot detection per bloom"
    except Exception as exc:  # noqa: BLE001
        print(
            f"Detection pipeline unavailable ({exc}); using HSV stem counter.",
            file=sys.stderr,
        )

    # ── Stage 3: HSV stem-counter fallback (only if detection unavailable) ────
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
