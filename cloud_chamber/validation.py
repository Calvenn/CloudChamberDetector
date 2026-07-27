"""Validation of shared detector outputs."""

from __future__ import annotations

import numpy as np

from cloud_chamber.models import DetectionResult


def validate_detection_result(
    result: DetectionResult,
    expected_shape: tuple[int, int],
) -> None:
    if not result.method_name.strip():
        raise ValueError("Detector method_name cannot be empty")
    if result.binary_mask.ndim != 2:
        raise ValueError("Detector binary_mask must be two-dimensional")
    if result.binary_mask.shape != expected_shape:
        raise ValueError(
            f"Detector mask shape {result.binary_mask.shape} "
            f"does not match input shape {expected_shape}"
        )
    if result.binary_mask.dtype != np.uint8:
        raise ValueError("Detector binary_mask must use uint8")

    unique_values = set(np.unique(result.binary_mask).tolist())
    if not unique_values.issubset({0, 255}):
        raise ValueError("Detector binary_mask must contain only 0 and 255")
    if result.processing_time_ms < 0:
        raise ValueError("processing_time_ms cannot be negative")

    height, width = expected_shape
    for x, y, box_width, box_height in result.bounding_boxes:
        if min(x, y, box_width, box_height) < 0:
            raise ValueError("Bounding-box values cannot be negative")
        if box_width == 0 or box_height == 0:
            raise ValueError("Bounding boxes must have a positive area")
        if x + box_width > width or y + box_height > height:
            raise ValueError("Bounding box lies outside the input image")

