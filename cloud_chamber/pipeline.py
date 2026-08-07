"""Integrated shared preprocessing, segmentation and feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from cloud_chamber.acquisition import load_image
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import TrackFeatures, extract_track_features
from cloud_chamber.models import EnhancementResult, SegmentationResult
from cloud_chamber.segmentation import segment_tracks


@dataclass
class PipelineResult:
    sample_id: str
    enhancement: EnhancementResult
    segmentation: SegmentationResult
    features: list[TrackFeatures]


def analyse_image(
    image_path: str | Path,
    config: dict[str, Any],
) -> PipelineResult:
    sample = load_image(image_path)
    enhancement = enhance_image(sample.image, config["enhancement"])
    segmentation = segment_tracks(enhancement.enhanced, config["segmentation"])
    features = extract_track_features(
        segmentation.binary_mask,
        enhancement.enhanced,
        minimum_area=float(config["segmentation"]["minimum_object_area"]),
    )

    return PipelineResult(
        sample_id=sample.sample_id,
        enhancement=enhancement,
        segmentation=segmentation,
        features=features,
    )


def save_pipeline_images(
    result: PipelineResult,
    output_folder: str | Path,
) -> Path:
    folder = Path(output_folder) / result.sample_id / "shared_pipeline"
    folder.mkdir(parents=True, exist_ok=True)

    images = {
        "01_grey.png": result.enhancement.grey,
        "02_denoised.png": result.enhancement.denoised,
        "03_threshold.png": result.segmentation.intermediate_images["threshold"],
        "04_morphological_opening.png": result.segmentation.intermediate_images[
            "morphological_opening"
        ],
        "05_morphological_closing.png": result.segmentation.intermediate_images[
            "morphological_closing"
        ],
        "06_segmentation_mask.png": result.segmentation.binary_mask,
        **{
            f"intermediate_{_safe_name(name)}.png": image
            for name, image in result.segmentation.intermediate_images.items()
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
