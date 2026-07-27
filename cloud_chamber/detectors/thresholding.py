"""Member template: Otsu, Triangle, Yen, and Niblack thresholding."""

from __future__ import annotations

import numpy as np

from cloud_chamber.models import DetectionResult


class ThresholdingDetector:
    name = "Thresholding category winner"

    def detect(self, enhanced_image: np.ndarray) -> DetectionResult:
        """Implement and return the validated category-winning method."""
        raise NotImplementedError(
            "Thresholding member: implement the selected category winner"
        )

