"""Main entry point for the Python analysis backend.

Usage:
    python analyze.py <image_path>

Prints a single JSON object to stdout:
    {
        "flowers": [{"label": "...", "score": 0.98}, ...],
        "stem_count": 3,
        "stem_method": "..."
    }

Errors are printed to stderr and the process exits with code 1.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        _error("Usage: python analyze.py <image_path>")

    image_path = sys.argv[1]

    # Flower classification ------------------------------------------------
    try:
        from classifier import FlowerClassifier

        classifier = FlowerClassifier()
        flowers = classifier.classify(image_path, top_k=5)
    except Exception as exc:
        _error(f"Flower classification failed: {exc}")

    # Stem counting --------------------------------------------------------
    try:
        from stem_counter import StemCounter

        counter = StemCounter()
        stem_count, stem_method = counter.count_stems(image_path)
    except Exception as exc:
        _error(f"Stem counting failed: {exc}")

    # Output ---------------------------------------------------------------
    result = {
        "flowers": flowers,
        "stem_count": stem_count,
        "stem_method": stem_method,
    }
    print(json.dumps(result, ensure_ascii=False))


def _error(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
