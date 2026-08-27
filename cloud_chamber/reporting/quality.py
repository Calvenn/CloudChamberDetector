"""Explainable segmentation-quality assessment for classified contours."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from cloud_chamber.feature_extraction.contour_features import TrackFeatures


def assess_contour_quality(
    track: TrackFeatures,
    enhanced_image: np.ndarray,
    binary_mask: np.ndarray,
    segmentation_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Return an explainable heuristic quality assessment for one contour."""
    image_height, image_width = binary_mask.shape
    x, y, width, height = track.bounding_box
    padding = max(5, int(round(min(image_height, image_width) * 0.01)))
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    local_image = enhanced_image[top:bottom, left:right]
    local_mask = binary_mask[top:bottom, left:right] > 0
    foreground_values = local_image[local_mask]
    background_values = local_image[~local_mask]
    foreground_mean = float(np.mean(foreground_values)) if foreground_values.size else 0.0
    background_median = (
        float(np.median(background_values)) if background_values.size else foreground_mean
    )
    local_contrast = foreground_mean - background_median

    roi = segmentation_parameters.get("roi_pixels", {})
    roi_left = int(roi.get("left", 0))
    roi_top = int(roi.get("top", 0))
    roi_right = int(roi.get("right", image_width))
    roi_bottom = int(roi.get("bottom", image_height))
    boundary_margin = max(3, int(round(min(image_height, image_width) * 0.01)))
    near_boundary = (
        x <= roi_left + boundary_margin
        or y <= roi_top + boundary_margin
        or x + width >= roi_right - boundary_margin
        or y + height >= roi_bottom - boundary_margin
    )

    general_minimum = float(segmentation_parameters.get("minimum_area", 0))
    accepted_as_small_track = track.area_pixels < general_minimum
    very_thin = track.mean_width_pixels <= 5.0 and track.aspect_ratio >= 3.0

    score = 100
    warnings: list[str] = []
    evidence: list[str] = []
    if local_contrast < 5:
        score -= 35
        warnings.append("Very low local contrast; segmentation may be unstable.")
    elif local_contrast < 12:
        score -= 20
        warnings.append("Low local contrast; inspect the contour mask.")
    else:
        evidence.append(f"Local contrast is {local_contrast:.1f} intensity levels.")
    if near_boundary:
        score -= 20
        warnings.append("Touches the analysis boundary and may be incomplete.")
    if track.solidity < 0.35:
        score -= 15
        warnings.append("Irregular/fragment-like contour shape.")
    elif track.solidity >= 0.65:
        evidence.append(f"Contour solidity is {track.solidity:.2f}.")
    if accepted_as_small_track:
        score -= 5
        evidence.append("Accepted through the specialised thin-track rule.")
    else:
        evidence.append("Passed the normal contour size rule.")
    if very_thin:
        evidence.append(
            f"Thin elongated structure: width {track.mean_width_pixels:.1f}px, "
            f"aspect ratio {track.aspect_ratio:.1f}."
        )
    else:
        evidence.append(
            f"Length {track.major_axis_pixels:.1f}px and width "
            f"{track.mean_width_pixels:.1f}px."
        )
    score = int(np.clip(score, 0, 100))
    grade = "High" if score >= 75 else "Moderate" if score >= 50 else "Low"
    return {
        "track_id": track.track_id,
        "score": score,
        "grade": grade,
        "local_contrast": round(local_contrast, 3),
        "near_boundary": near_boundary,
        "very_thin": very_thin,
        "accepted_as_small_track": accepted_as_small_track,
        "warnings": warnings,
        "evidence": evidence,
        "method": "explainable contour-quality heuristic; not model confidence",
    }


def assess_all_contours(
    features: Iterable[TrackFeatures],
    enhanced_image: np.ndarray,
    binary_mask: np.ndarray,
    segmentation_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Assess contours in the same track order used by the classifier."""
    return [
        assess_contour_quality(
            track, enhanced_image, binary_mask, segmentation_parameters
        )
        for track in features
    ]


def reporting_status(
    confidence: float, confidence_threshold: float, quality_score: int
) -> str:
    """Combine two independent signals into a plain-language review status."""
    confident = confidence >= confidence_threshold
    good_contour = quality_score >= 50
    if confident and quality_score >= 75:
        return "Reliable candidate"
    if confident and not good_contour:
        return "Review segmentation"
    if not confident and good_contour:
        return "Ambiguous particle class"
    if not confident and not good_contour:
        return "Manual review required"
    return "Usable with caution"
