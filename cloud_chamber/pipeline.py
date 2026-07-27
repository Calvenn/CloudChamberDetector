"""Integration pipeline shared by the command-line runner and future UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from cloud_chamber.acquisition import load_image
from cloud_chamber.detectors import create_detector
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import TrackFeatures, extract_track_features
from cloud_chamber.models import DetectionResult, EnhancementResult
from cloud_chamber.validation import validate_detection_result


@dataclass
class PipelineResult:
    sample_id: str
    enhancement: EnhancementResult
    detection: DetectionResult
    features: list[TrackFeatures]


def analyse_image(
    image_path: str | Path,
    detector_name: str,
    config: dict[str, Any],
) -> PipelineResult:
    sample = load_image(image_path)
    enhancement = enhance_image(sample.image, config["enhancement"])
    detector = create_detector(detector_name)
    detection = detector.detect(enhancement.enhanced)
    validate_detection_result(detection, enhancement.enhanced.shape)
    features = extract_track_features(
        detection.binary_mask,
        enhancement.enhanced,
        minimum_area=float(config["baseline"]["minimum_object_area"]),
    )

    return PipelineResult(
        sample_id=sample.sample_id,
        enhancement=enhancement,
        detection=detection,
        features=features,
    )


def save_pipeline_images(
    result: PipelineResult,
    output_folder: str | Path,
) -> Path:
    folder = Path(output_folder) / result.sample_id / _safe_name(
        result.detection.method_name
    )
    folder.mkdir(parents=True, exist_ok=True)

    images = {
        "01_grey.png": result.enhancement.grey,
        "02_denoised.png": result.enhancement.denoised,
        "03_clahe.png": result.enhancement.contrast_enhanced,
        "04_background_corrected.png": (
            result.enhancement.background_corrected
        ),
        "05_detection_mask.png": result.detection.binary_mask,
        **{
            f"intermediate_{_safe_name(name)}.png": image
            for name, image in result.detection.intermediate_images.items()
        },
    }
    for filename, image in images.items():
        if not cv2.imwrite(str(folder / filename), image):
            raise OSError(f"Unable to write result image: {folder / filename}")
    return folder


def _safe_name(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")
