"""Shared data contracts used by every detection technique."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageSample:
    """An acquired image and traceable source metadata."""

    sample_id: str
    image: np.ndarray
    source_path: Path
    frame_number: int | None = None


@dataclass
class EnhancementResult:
    """Shared preprocessing outputs available for visual inspection."""

    grey: np.ndarray
    denoised: np.ndarray
    enhanced: np.ndarray
    local_contrast: np.ndarray | None = None

    @property
    def segmentation_input(self) -> np.ndarray:
        """Return the white-top-hat image prepared for thresholding."""
        return self.local_contrast if self.local_contrast is not None else self.enhanced


@dataclass
class SegmentationResult:
    """Output of the one shared threshold/morphology/contour segmenter."""

    method_name: str
    binary_mask: np.ndarray
    bounding_boxes: list[BoundingBox]
    contours: list[np.ndarray]
    processing_time_ms: float
    intermediate_images: dict[str, np.ndarray] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Comparable mask-level evaluation metrics."""

    method_name: str
    precision: float
    recall: float
    f1_score: float
    iou: float
    dice: float
    processing_time_ms: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "method_name": self.method_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "iou": self.iou,
            "dice": self.dice,
            "processing_time_ms": self.processing_time_ms,
        }

