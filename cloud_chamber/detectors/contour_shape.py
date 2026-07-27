"""Member template: contours, Hu moments, and skeleton analysis."""

from __future__ import annotations

import numpy as np

from cloud_chamber.models import DetectionResult


class ContourShapeDetector:
    name = "Contour/shape category winner"

    def detect(self, enhanced_image: np.ndarray) -> DetectionResult:
        """Implement and return the validated category-winning method."""
        raise NotImplementedError(
            "Contour/shape member: implement the selected category winner"
        )

