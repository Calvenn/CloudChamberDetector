"""Protocol that every individual detector must implement."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from cloud_chamber.models import DetectionResult


class Detector(Protocol):
    name: str

    def detect(self, enhanced_image: np.ndarray) -> DetectionResult:
        """Detect particle-track candidates in an enhanced greyscale image."""

