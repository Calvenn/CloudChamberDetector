"""Shared geometric feature extraction from detector masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cloud_chamber.core.contracts import BoundingBox


@dataclass(frozen=True)
class TrackFeatures:
    """The shared 16-feature contract for one segmented particle track."""
    track_id: int
    area_pixels: float
    perimeter_pixels: float
    major_axis_pixels: float
    mean_width_pixels: float
    orientation_degrees: float
    aspect_ratio: float
    solidity: float
    rectangularity: float
    thickness_pixels: float
    mean_intensity: float
    intensity_stddev: float
    circularity: float
    convexity: float
    perimeter_to_major_axis: float
    orientation_sin_2x: float
    orientation_cos_2x: float
    bounding_box: BoundingBox


def extract_track_features(
    binary_mask: np.ndarray,
    enhanced_image: np.ndarray,
    minimum_area: float = 20.0,
) -> list[TrackFeatures]:
    """Measure every valid contour using the project's fixed feature order."""
    if binary_mask.shape != enhanced_image.shape:
        raise ValueError("Mask and enhanced image must have equal shapes")

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    features: list[TrackFeatures] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        major_axis, minor_axis, orientation = _oriented_dimensions(contour)
        mean_width = area / major_axis if major_axis > 0 else 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        rectangle_area = major_axis * minor_axis

        region_mask = np.zeros_like(binary_mask)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
        intensity_mean, intensity_stddev = cv2.meanStdDev(
            enhanced_image, mask=region_mask
        )
        mean_intensity = float(intensity_mean[0, 0])
        intensity_stddev_value = float(intensity_stddev[0, 0])
        perimeter = float(cv2.arcLength(contour, closed=True))
        hull_perimeter = float(
            cv2.arcLength(cv2.convexHull(contour), closed=True)
        )
        orientation_radians = np.deg2rad(2.0 * orientation)

        features.append(
            TrackFeatures(
                track_id=len(features) + 1,
                area_pixels=area,
                perimeter_pixels=perimeter,
                major_axis_pixels=major_axis,
                mean_width_pixels=mean_width,
                orientation_degrees=orientation,
                aspect_ratio=major_axis / minor_axis if minor_axis > 0 else 0.0,
                solidity=area / hull_area if hull_area > 0 else 0.0,
                rectangularity=area / rectangle_area if rectangle_area > 0 else 0.0,
                thickness_pixels=mean_width,
                mean_intensity=mean_intensity,
                intensity_stddev=intensity_stddev_value,
                circularity=(
                    4.0 * np.pi * area / (perimeter * perimeter)
                    if perimeter > 0
                    else 0.0
                ),
                convexity=(
                    hull_perimeter / perimeter if perimeter > 0 else 0.0
                ),
                perimeter_to_major_axis=(
                    perimeter / major_axis if major_axis > 0 else 0.0
                ),
                orientation_sin_2x=float(np.sin(orientation_radians)),
                orientation_cos_2x=float(np.cos(orientation_radians)),
                bounding_box=(x, y, width, height),
            )
        )

    return features


def _oriented_dimensions(
    contour: np.ndarray,
) -> tuple[float, float, float]:
    (_, _), (side_a, side_b), angle = cv2.minAreaRect(contour)
    if side_a >= side_b:
        return float(side_a), float(side_b), float(angle)
    return float(side_b), float(side_a), float(angle + 90.0)

