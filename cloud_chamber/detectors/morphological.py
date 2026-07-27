"""Member template: watershed, directional openings, and reconstruction."""

from __future__ import annotations

import numpy as np

from cloud_chamber.models import DetectionResult


class MorphologicalDetector:
    name = "Morphological category winner"

    def detect(self, enhanced_image: np.ndarray) -> DetectionResult:
        """Implement and return the validated category-winning method."""
        raise NotImplementedError(
            "Morphological member: implement the selected category winner"
        )

