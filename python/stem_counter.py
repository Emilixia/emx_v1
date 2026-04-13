"""Stem counter — estimates the number of flower stems in an image using OpenCV."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class StemCounter:
    """Count distinct flower stems in an image via colour segmentation + morphology."""

    # HSV ranges that capture green and brown/yellow stems
    _GREEN_LOW = np.array([25, 30, 30])
    _GREEN_HIGH = np.array([90, 255, 255])
    _BROWN_LOW = np.array([10, 30, 30])
    _BROWN_HIGH = np.array([25, 255, 200])

    def count_stems(self, image_path: str) -> Tuple[int, str]:
        """Return ``(stem_count, method_description)`` for the given image.

        The algorithm:
        1. Isolate green and brown/yellow pixels (typical stem colours).
        2. Apply morphological close + open to fill gaps and remove noise.
        3. Focus on the lower half of the frame where stems converge.
        4. Find external contours; keep those that are tall/narrow (stem-like).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv, self._GREEN_LOW, self._GREEN_HIGH)
        brown_mask = cv2.inRange(hsv, self._BROWN_LOW, self._BROWN_HIGH)
        stem_mask = cv2.bitwise_or(green_mask, brown_mask)

        kernel = np.ones((3, 3), np.uint8)
        stem_mask = cv2.morphologyEx(stem_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        stem_mask = cv2.morphologyEx(stem_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        h, w = stem_mask.shape
        # Analyse the lower half — stems are most distinct there
        lower = stem_mask[h // 2 :, :]

        contours, _ = cv2.findContours(lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stem_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            _x, _y, cw, ch = cv2.boundingRect(cnt)
            # Keep tall-ish blobs (stems are at least as tall as they are wide)
            if cw > 0 and ch / cw >= 0.5:
                stem_count += 1

        method = "HSV colour segmentation + contour analysis (lower-half region)"
        return stem_count, method
