"""Binary particle-versus-artifact filtering after candidate segmentation."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np

from cloud_chamber.feature_extraction.contour_features import extract_track_features
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS as TRACK_COLUMNS


FEATURE_COLUMNS = (*TRACK_COLUMNS, "contour_complexity", "bbox_extent", "skeleton_endpoints", "skeleton_branch_ratio", "border_distance_ratio", "local_contrast")


def _skeleton_topology(mask: np.ndarray) -> tuple[int, int, int]:
    """Count skeleton pixels, endpoints, and branch points in a contour mask."""
    working = mask.copy()
    maximum_dimension = max(working.shape, default=0)
    if maximum_dimension > 256:
        scale = 256.0 / maximum_dimension
        working = cv2.resize(
            working,
            (
                max(1, int(round(working.shape[1] * scale))),
                max(1, int(round(working.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    skeleton = np.zeros_like(working)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max(working.shape) + 1):
        if not cv2.countNonZero(working):
            break
        eroded = cv2.erode(working, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(working, opened))
        if np.array_equal(eroded, working):
            skeleton = cv2.bitwise_or(skeleton, working)
            break
        working = eroded
    foreground = (skeleton > 0).astype(np.uint8)
    length = int(np.count_nonzero(foreground))
    if length == 0:
        return 0, 0, 0
    neighbours = cv2.filter2D(foreground, cv2.CV_16S, np.ones((3, 3), np.uint8)) - foreground
    endpoints = int(np.count_nonzero((foreground > 0) & (neighbours == 1)))
    branches = int(np.count_nonzero((foreground > 0) & (neighbours >= 3)))
    return length, endpoints, branches


def contour_feature_vector(contour: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """Describe one contour using track and artefact-specific measurements."""
    x, y, width, height = cv2.boundingRect(contour)
    padding = 8
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(enhanced.shape[1], x + width + padding)
    bottom = min(enhanced.shape[0], y + height + padding)
    local_contour = contour.copy()
    local_contour[:, 0, 0] -= left
    local_contour[:, 0, 1] -= top
    local_enhanced = enhanced[top:bottom, left:right]
    mask = np.zeros(local_enhanced.shape, dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, cv2.FILLED)
    measured = extract_track_features(mask, local_enhanced, minimum_area=0.0)
    if not measured:
        raise ValueError("Cannot measure an empty candidate contour")
    track = max(measured, key=lambda item: item.area_pixels)
    values = asdict(track)
    area = max(float(track.area_pixels), 1.0)
    perimeter = float(track.perimeter_pixels)
    local_x, local_y, local_width, local_height = cv2.boundingRect(local_contour)
    skeleton_length, endpoints, branches = _skeleton_topology(
        mask[local_y:local_y + local_height, local_x:local_x + local_width]
    )
    border_distance = min(x, y, enhanced.shape[1] - (x + width), enhanced.shape[0] - (y + height))
    border_ratio = max(0.0, float(border_distance)) / max(min(enhanced.shape), 1)
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ring = (dilated > 0) & (mask == 0)
    foreground_mean = float(np.mean(local_enhanced[mask > 0])) if np.any(mask > 0) else 0.0
    background_mean = float(np.mean(local_enhanced[ring])) if np.any(ring) else foreground_mean
    extras = {
        "contour_complexity": perimeter / np.sqrt(area),
        "bbox_extent": area / max(width * height, 1),
        "skeleton_endpoints": float(endpoints),
        "skeleton_branch_ratio": branches / max(skeleton_length, 1),
        "border_distance_ratio": border_ratio,
        "local_contrast": foreground_mean - background_mean,
    }
    return np.asarray([float(values[name]) if name in values else float(extras[name]) for name in FEATURE_COLUMNS], dtype=np.float64)


def candidate_matrix(contours: list[np.ndarray], enhanced: np.ndarray) -> np.ndarray:
    """Build one consistently ordered feature row for every candidate contour."""
    if not contours:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64)
    return np.vstack([contour_feature_vector(contour, enhanced) for contour in contours])


@lru_cache(maxsize=4)
def load_model(path: str | Path) -> dict:
    """Load a trained artefact-filter bundle and validate its structure."""
    bundle = joblib.load(Path(path).resolve())
    if tuple(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Saved artifact filter uses different feature columns")
    return bundle


def filter_mask(mask: np.ndarray, enhanced: np.ndarray, bundle: dict) -> tuple[np.ndarray, dict]:
    """Remove candidates whose learned particle probability is below threshold."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask.copy(), {"candidate_count": 0, "accepted_count": 0, "rejected_count": 0}
    matrix = candidate_matrix(contours, enhanced)
    model = bundle["model"]
    particle_index = list(model.classes_).index("particle")
    probabilities = model.predict_proba(matrix)[:, particle_index]
    threshold = float(bundle["acceptance_threshold"])
    accepted = [contour for contour, probability in zip(contours, probabilities, strict=True) if probability >= threshold]
    output = np.zeros_like(mask)
    if accepted:
        cv2.drawContours(output, accepted, -1, 255, cv2.FILLED)
    return output, {
        "candidate_count": len(contours),
        "accepted_count": len(accepted),
        "rejected_count": len(contours) - len(accepted),
        "acceptance_threshold": threshold,
    }
