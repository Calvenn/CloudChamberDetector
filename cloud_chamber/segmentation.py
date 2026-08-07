"""Shared thresholding, morphology and contour segmentation pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np

from cloud_chamber.models import SegmentationResult


def segment_tracks(
    enhanced_image: np.ndarray,
    settings: dict[str, Any],
) -> SegmentationResult:
    """Segment bright particle tracks and return their external contours."""
    if enhanced_image.ndim != 2 or enhanced_image.dtype != np.uint8:
        raise ValueError("Segmentation input must be an 8-bit grayscale image")

    started = perf_counter()
    _, threshold_mask = cv2.threshold(
        enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel_size = int(settings["morphology_kernel"])
    iterations = int(settings.get("morphology_iterations", 1))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    opened = cv2.morphologyEx(
        threshold_mask, cv2.MORPH_OPEN, kernel, iterations=iterations
    )
    refined = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=iterations
    )

    found, _ = cv2.findContours(
        refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    minimum_area = float(settings["minimum_object_area"])
    contours = [c for c in found if cv2.contourArea(c) >= minimum_area]

    clean_mask = np.zeros_like(refined)
    if contours:
        cv2.drawContours(clean_mask, contours, -1, 255, cv2.FILLED)
    boxes = [tuple(int(v) for v in cv2.boundingRect(c)) for c in contours]

    return SegmentationResult(
        method_name="Otsu thresholding + morphological opening/closing",
        binary_mask=clean_mask,
        bounding_boxes=boxes,
        contours=contours,
        processing_time_ms=(perf_counter() - started) * 1000.0,
        intermediate_images={
            "threshold": threshold_mask,
            "morphological_opening": opened,
            "morphological_closing": refined,
        },
        parameters={
            "threshold": "Otsu",
            "kernel_size": kernel_size,
            "iterations": iterations,
            "minimum_area": minimum_area,
        },
    )
