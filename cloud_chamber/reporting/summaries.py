"""Image-level classification summaries and traceability metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np

from cloud_chamber.ml.member_models.mlp import DISPLAY_NAMES


def build_summary(
    predictions: list[dict[str, Any]],
    quality_assessments: list[dict[str, Any]],
    confidence_threshold: float,
    processing_time_ms: float,
) -> dict[str, Any]:
    """Build an image-level summary from particle and quality results."""
    class_counts = Counter(item["predicted_class"] for item in predictions)
    confident = sum(item["confidence"] >= confidence_threshold for item in predictions)
    quality_scores = [item["score"] for item in quality_assessments]
    mean_quality = float(np.mean(quality_scores)) if quality_scores else 0.0
    overall_grade = (
        "High" if mean_quality >= 75 else "Moderate" if mean_quality >= 50 else "Low"
    )
    dominant = class_counts.most_common(1)[0][0] if class_counts else None
    return {
        "detected_contours": len(predictions),
        "confident_classifications": confident,
        "uncertain_classifications": len(predictions) - confident,
        "class_counts": {
            DISPLAY_NAMES.get(name, name): class_counts.get(name, 0)
            for name in ("alpha", "electron_positron", "proton", "v_track")
        },
        "dominant_prediction": DISPLAY_NAMES.get(dominant, dominant or "None"),
        "mean_contour_quality": round(mean_quality, 2),
        "overall_contour_quality": overall_grade,
        "processing_time_ms": round(float(processing_time_ms), 3),
    }


def make_traceability_metadata(
    input_name: str,
    source_description: str,
    roi_profile: str,
    confidence_threshold: float,
    model_path: str,
    model_classes: Iterable[str],
    segmentation_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Capture enough context to reproduce and audit one report."""
    selected_parameter_names = (
        "otsu_value",
        "threshold_offset",
        "applied_threshold",
        "hysteresis_low_threshold_offset",
        "hysteresis_low_threshold_value",
        "hysteresis_seed_pixels",
        "top_hat_kernel",
        "closing_kernel",
        "directional_closing_length",
        "alignment_merge_gap",
        "minimum_area",
        "minimum_major_axis",
        "minimum_thin_area",
        "minimum_thin_major_axis",
        "minimum_thin_aspect_ratio",
    )
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_name": input_name,
        "source": source_description,
        "roi_profile": roi_profile,
        "model": model_path,
        "model_classes": [str(value) for value in model_classes],
        "class_mapping": {
            name: DISPLAY_NAMES.get(name, name) for name in model_classes
        },
        "confidence_threshold": confidence_threshold,
        "segmentation_parameters": {
            name: segmentation_parameters.get(name)
            for name in selected_parameter_names
            if name in segmentation_parameters
        },
    }
