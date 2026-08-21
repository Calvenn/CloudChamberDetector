"""Spatial scaling and optional perspective rectification for input images."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, hypot
from typing import Iterable

import cv2
import numpy as np


DEFAULT_PROCESSING_SIZE = (640, 640)


@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated image and the scale used for physical measurements."""

    image: np.ndarray
    centimetres_per_pixel: float
    rectified: bool


@dataclass(frozen=True)
class RoiCoordinates:
    """Integer ROI coordinates expressed in the original image space."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class RoiScalingResult:
    """Selected original-resolution ROI and its standardized image."""

    image: np.ndarray
    roi_image: np.ndarray
    coordinates: RoiCoordinates
    original_width: int
    original_height: int
    processing_width: int
    processing_height: int
    scale_x: float
    scale_y: float

    @property
    def scaling_operation(self) -> str:
        """Describe whether the selected ROI is enlarged, reduced or copied."""
        if self.processing_width == self.coordinates.width:
            return "none"
        return "upscale" if self.scale_x > 1.0 else "downscale"

    def metadata(self) -> dict[str, int | float]:
        """Return serialisable transform information for reports and mapping."""
        return {
            "original_image_width": self.original_width,
            "original_image_height": self.original_height,
            "roi_x": self.coordinates.x,
            "roi_y": self.coordinates.y,
            "roi_width": self.coordinates.width,
            "roi_height": self.coordinates.height,
            "processing_width": self.processing_width,
            "processing_height": self.processing_height,
            "processing_resolution": (
                f"{self.processing_width} × {self.processing_height}"
            ),
            "scale_factor": self.scale_x,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "scaling_operation": self.scaling_operation,
        }


def default_roi_coordinates(
    image: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_PROCESSING_SIZE,
) -> RoiCoordinates:
    """Return the largest centred ROI matching the processing aspect ratio."""
    _validate_image_and_target(image, target_size)
    image_height, image_width = image.shape[:2]
    target_width, target_height = target_size
    divisor = gcd(target_width, target_height)
    unit_width = target_width // divisor
    unit_height = target_height // divisor
    units = min(image_width // unit_width, image_height // unit_height)
    if units < 1:
        raise ValueError("Image is too small for the configured ROI aspect ratio")
    width = units * unit_width
    height = units * unit_height
    return RoiCoordinates(
        x=(image_width - width) // 2,
        y=(image_height - height) // 2,
        width=width,
        height=height,
    )


def constrain_roi(
    image: np.ndarray,
    roi_coordinates: RoiCoordinates | dict[str, int | float],
    target_size: tuple[int, int] = DEFAULT_PROCESSING_SIZE,
) -> RoiCoordinates:
    """Constrain a movable ROI to the image and target aspect ratio."""
    _validate_image_and_target(image, target_size)
    if isinstance(roi_coordinates, RoiCoordinates):
        values = roi_coordinates
    else:
        # ``streamlit-cropper`` names the origin ``left``/``top`` while the
        # application's serialised metadata uses ``x``/``y``. Accept both at
        # this boundary so the geometry functions remain reusable.
        x_value = roi_coordinates.get("x", roi_coordinates.get("left"))
        y_value = roi_coordinates.get("y", roi_coordinates.get("top"))
        if x_value is None or y_value is None:
            raise KeyError("ROI coordinates require x/y or left/top values")
        values = RoiCoordinates(
            x=int(round(float(x_value))),
            y=int(round(float(y_value))),
            width=max(1, int(round(float(roi_coordinates["width"])))),
            height=max(1, int(round(float(roi_coordinates["height"])))),
        )

    image_height, image_width = image.shape[:2]
    target_width, target_height = target_size
    divisor = gcd(target_width, target_height)
    unit_width = target_width // divisor
    unit_height = target_height // divisor
    if image_width < unit_width or image_height < unit_height:
        raise ValueError("Image is too small for the configured ROI aspect ratio")

    # Fit the requested rectangle from inside, then quantise to the exact
    # target ratio. This avoids non-uniform stretching after integer rounding.
    requested_units = min(
        values.width // unit_width,
        values.height // unit_height,
        image_width // unit_width,
        image_height // unit_height,
    )
    units = max(1, requested_units)
    width = units * unit_width
    height = units * unit_height
    x = min(max(values.x, 0), image_width - width)
    y = min(max(values.y, 0), image_height - height)
    return RoiCoordinates(x=x, y=y, width=width, height=height)


def extract_roi(
    image: np.ndarray,
    roi_coordinates: RoiCoordinates | dict[str, int | float],
    target_size: tuple[int, int] = DEFAULT_PROCESSING_SIZE,
) -> tuple[np.ndarray, RoiCoordinates]:
    """Extract a bounded fixed-aspect ROI without modifying the source image."""
    coordinates = constrain_roi(image, roi_coordinates, target_size)
    roi = image[
        coordinates.y : coordinates.y + coordinates.height,
        coordinates.x : coordinates.x + coordinates.width,
    ].copy()
    return roi, coordinates


def spatial_scale_roi(
    roi: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_PROCESSING_SIZE,
) -> tuple[np.ndarray, float, float]:
    """Resize only a selected ROI to the standardized processing resolution."""
    _validate_image_and_target(roi, target_size)
    roi_height, roi_width = roi.shape[:2]
    target_width, target_height = target_size
    if roi_width * target_height != roi_height * target_width:
        raise ValueError("ROI aspect ratio must match the processing resolution")
    scale_x = target_width / roi_width
    scale_y = target_height / roi_height
    if (roi_width, roi_height) == (target_width, target_height):
        return roi.copy(), 1.0, 1.0
    interpolation = (
        cv2.INTER_AREA
        if target_width < roi_width or target_height < roi_height
        else cv2.INTER_LINEAR
    )
    scaled = cv2.resize(
        roi, (target_width, target_height), interpolation=interpolation
    )
    return scaled, float(scale_x), float(scale_y)


def select_and_scale_roi(
    image: np.ndarray,
    roi_coordinates: RoiCoordinates | dict[str, int | float] | None = None,
    target_size: tuple[int, int] = DEFAULT_PROCESSING_SIZE,
) -> RoiScalingResult:
    """Extract the chosen original-image region, then spatially scale it."""
    requested = roi_coordinates or default_roi_coordinates(image, target_size)
    roi, coordinates = extract_roi(image, requested, target_size)
    scaled, scale_x, scale_y = spatial_scale_roi(roi, target_size)
    original_height, original_width = image.shape[:2]
    return RoiScalingResult(
        image=scaled,
        roi_image=roi,
        coordinates=coordinates,
        original_width=original_width,
        original_height=original_height,
        processing_width=target_size[0],
        processing_height=target_size[1],
        scale_x=scale_x,
        scale_y=scale_y,
    )


def map_processed_point_to_original(
    point: tuple[float, float],
    transform: RoiScalingResult | dict[str, int | float],
) -> tuple[float, float]:
    """Map a point from the scaled ROI back to original-image coordinates."""
    metadata = transform.metadata() if isinstance(transform, RoiScalingResult) else transform
    return (
        float(metadata["roi_x"]) + float(point[0]) / float(metadata["scale_x"]),
        float(metadata["roi_y"]) + float(point[1]) / float(metadata["scale_y"]),
    )


def _validate_image_and_target(
    image: np.ndarray, target_size: tuple[int, int]
) -> None:
    if image is None or image.size == 0:
        raise ValueError("ROI spatial-scaling input image is empty")
    if len(target_size) != 2 or min(int(value) for value in target_size) < 2:
        raise ValueError("Processing width and height must both exceed one pixel")


def detect_chamber_corners(image: np.ndarray) -> np.ndarray:
    """Set default chamber corners capturing only the inner dark active chamber region."""
    if image is None or image.size == 0:
        raise ValueError("Corner-detection input image is empty")
    height, width = image.shape[:2]
    is_portrait = height > width
    # Primary portrait frames have a thicker 12% bottom base and 7% top rim
    left_ratio = 0.04 if is_portrait else 0.03
    right_ratio = 0.04 if is_portrait else 0.03
    top_ratio = 0.10 if is_portrait else 0.03
    bottom_ratio = 0.12 if is_portrait else 0.03

    left = float(width * left_ratio)
    right = float(width * (1.0 - right_ratio))
    top = float(height * top_ratio)
    bottom = float(height * (1.0 - bottom_ratio))
    return np.asarray(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float32,
    )


def calibrate_image(
    image: np.ndarray,
    known_length_cm: float,
    reference_points: Iterable[Iterable[float]],
    rectification_points: Iterable[Iterable[float]] | None = None,
) -> CalibrationResult:
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
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Correct perspective, optionally retaining a required ROI output size."""
    if image is None or image.size == 0:
        raise ValueError("Rectification input image is empty")
    corners = np.asarray(list(rectification_points), dtype=np.float32)
    rectified, _ = _rectify_with_transform(image, corners, output_size)
    return rectified


def rectify_image_with_transform(
    image: np.ndarray,
    rectification_points: Iterable[Iterable[float]],
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rectified ROI and its source-to-rectified homography."""
    if image is None or image.size == 0:
        raise ValueError("Rectification input image is empty")
    corners = np.asarray(list(rectification_points), dtype=np.float32)
    return _rectify_with_transform(image, corners, output_size)


def _rectify_with_transform(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a rectified image and its source-to-destination homography."""
    if corners.shape != (4, 2):
        raise ValueError("Rectification requires four corner points")
    _validate_points(corners, image.shape[1], image.shape[0])
    if output_size is None:
        destination_width = max(
            int(round(np.linalg.norm(corners[1] - corners[0]))),
            int(round(np.linalg.norm(corners[2] - corners[3]))),
        )
        destination_height = max(
            int(round(np.linalg.norm(corners[3] - corners[0]))),
            int(round(np.linalg.norm(corners[2] - corners[1]))),
        )
    else:
        destination_width, destination_height = (
            int(value) for value in output_size
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
    points[:, 0] = np.clip(points[:, 0], 0.0, float(max(0, width - 1)))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(max(0, height - 1)))


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
