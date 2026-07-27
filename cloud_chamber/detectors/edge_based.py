"""Member template: Sobel, Laplacian of Gaussian, and Canny."""

from __future__ import annotations

import numpy as np

from cloud_chamber.models import DetectionResult


class EdgeBasedDetector:
    name = "Edge-based category winner"

    def detect(self, enhanced_image: np.ndarray) -> DetectionResult:
        """Implement and return the validated category-winning method."""
        raise NotImplementedError(
            "Edge-based member: implement the selected category winner"
        )

