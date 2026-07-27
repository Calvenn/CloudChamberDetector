"""Member template: standard, progressive probabilistic, and randomised Hough."""

from __future__ import annotations

import numpy as np

from cloud_chamber.models import DetectionResult


class HoughDetector:
    name = "Hough category winner"

    def detect(self, enhanced_image: np.ndarray) -> DetectionResult:
        """Implement and return the validated category-winning method."""
        raise NotImplementedError(
            "Hough member: implement the selected category winner"
        )

