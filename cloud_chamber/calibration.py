"""Spatial scaling and optional perspective rectification for input images."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated image and the scale used for physical measurements."""

    image: np.ndarray
    centimetres_per_pixel: float
    rectified: bool


@dataclass(frozen=True)
class SpatialScalingResult:
    """Aspect-ratio-preserving image and spatial scaling metadata."""

    image: np.ndarray
    scale: float
    original_width: int
    original_height: int
    scaled_width: int
    scaled_height: int


def spatial_scale(
    image: np.ndarray,
    target_longest_side: int = 1920,
) -> SpatialScalingResult:
    """Scale the longest side without cropping, padding or distortion."""
    if image is None or image.size == 0:
        raise ValueError("Spatial-normalisation input image is empty")
    if target_longest_side < 2:
        raise ValueError("Target longest side must exceed one pixel")

    source_height, source_width = image.shape[:2]
    scale = float(target_longest_side) / max(source_width, source_height)
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))
    if (scaled_width, scaled_height) == (source_width, source_height):
        scaled = image.copy()
    else:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(
            image, (scaled_width, scaled_height), interpolation=interpolation
        )
    return SpatialScalingResult(
        image=scaled,
        scale=float(scale),
        original_width=source_width,
        original_height=source_height,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
    )




def detect_chamber_corners(image: np.ndarray) -> np.ndarray:
    """Estimate the largest chamber-like quadrilateral in an image.

    The detector searches strong structural boundaries only for calibration;
    it does not change the selected particle-segmentation methodology.
    Returned points are ordered top-left, top-right, bottom-right, bottom-left.
    """
    if image is None or image.size == 0:
        raise ValueError("Corner-detection input image is empty")
    grey = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )
    blurred = cv2.GaussianBlur(grey, (7, 7), 1.4)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    image_area = float(image.shape[0] * image.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.20 * image_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            candidates.append((area, polygon.reshape(4, 2).astype(np.float32)))
    if not candidates:
        # A failed detection must not silently crop the image. Default to the
        # complete field of view; users can refine these safe coordinates with
        # the existing manual rectification controls when correction is needed.
        height, width = image.shape[:2]
        left, right = 0.0, float(width - 1)
        top, bottom = 0.0, float(height - 1)
        return np.asarray(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float32,
        )
    return _order_corners(max(candidates, key=lambda item: item[0])[1])


def calibrate_image(
    image: np.ndarray,
    known_length_cm: float,
    reference_points: Iterable[Iterable[float]],
    rectification_points: Iterable[Iterable[float]] | None = None,
) -> CalibrationResult:
    """Calculate spatial scale and optionally rectify a quadrilateral view.

    ``reference_points`` are two endpoints whose real separation is known.
    Rectification points, when supplied, must be ordered top-left, top-right,
    bottom-right and bottom-left.
    """
    if image is None or image.size == 0:
        raise ValueError("Calibration input image is empty")
    if known_length_cm <= 0:
        raise ValueError("Known physical length must be greater than zero")

    reference = np.asarray(list(reference_points), dtype=np.float32)
    if reference.shape != (2, 2):
        raise ValueError("Exactly two reference endpoints are required")
    _validate_points(reference, image.shape[1], image.shape[0])
    pixel_length = hypot(
        float(reference[1, 0] - reference[0, 0]),
        float(reference[1, 1] - reference[0, 1]),
    )
    if pixel_length < 1.0:
        raise ValueError("Reference endpoints must be at least one pixel apart")
    scale = float(known_length_cm) / pixel_length

    if rectification_points is None:
        return CalibrationResult(image.copy(), scale, False)

    corners = np.asarray(list(rectification_points), dtype=np.float32)
    rectified, transform = _rectify_with_transform(image, corners)
    # Perspective correction changes pixel distances. Transform the reference
    # endpoints through the same homography before calculating the final scale.
    transformed_reference = cv2.perspectiveTransform(
        reference.reshape(1, 2, 2), transform
    )[0]
    transformed_length = float(
        np.linalg.norm(transformed_reference[1] - transformed_reference[0])
    )
    if transformed_length < 1.0:
        raise ValueError("Rectified reference endpoints are too close together")
    rectified_scale = float(known_length_cm) / transformed_length
    return CalibrationResult(rectified, rectified_scale, True)


def rectify_image(
    image: np.ndarray,
    rectification_points: Iterable[Iterable[float]],
) -> np.ndarray:
    """Correct perspective without requiring a physical measurement scale."""
    if image is None or image.size == 0:
        raise ValueError("Rectification input image is empty")
    corners = np.asarray(list(rectification_points), dtype=np.float32)
    rectified, _ = _rectify_with_transform(image, corners)
    return rectified


def _rectify_with_transform(
    image: np.ndarray, corners: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a rectified image and its source-to-destination homography."""
    if corners.shape != (4, 2):
        raise ValueError("Rectification requires four corner points")
    _validate_points(corners, image.shape[1], image.shape[0])
    destination_width = max(
        int(round(np.linalg.norm(corners[1] - corners[0]))),
        int(round(np.linalg.norm(corners[2] - corners[3]))),
    )
    destination_height = max(
        int(round(np.linalg.norm(corners[3] - corners[0]))),
        int(round(np.linalg.norm(corners[2] - corners[1]))),
    )
    if destination_width < 2 or destination_height < 2:
        raise ValueError("Rectification corners do not define a valid area")
    destination = np.asarray(
        [
            [0, 0],
            [destination_width - 1, 0],
            [destination_width - 1, destination_height - 1],
            [0, destination_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    rectified = cv2.warpPerspective(
        image, transform, (destination_width, destination_height)
    )
    return rectified, transform


def _validate_points(points: np.ndarray, width: int, height: int) -> None:
    if not np.isfinite(points).all():
        raise ValueError("Calibration coordinates must be finite numbers")
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= height)
    ):
        raise ValueError("Calibration coordinates must lie inside the image")


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Return four quadrilateral points in a stable clockwise order."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sums = points.sum(axis=1)
    coordinate_differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(coordinate_sums)]
    ordered[2] = points[np.argmax(coordinate_sums)]
    ordered[1] = points[np.argmin(coordinate_differences)]
    ordered[3] = points[np.argmax(coordinate_differences)]
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("Detected chamber corners are ambiguous")
    return ordered
