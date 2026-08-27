"""Integrated shared preprocessing, segmentation and feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from cloud_chamber.acquisition import load_image
from cloud_chamber.features import TrackFeatures
from cloud_chamber.models import EnhancementResult, SegmentationResult


@dataclass
class PipelineResult:
    sample_id: str
    enhancement: EnhancementResult
    segmentation: SegmentationResult
    features: list[TrackFeatures]
    spatial_scaling: dict[str, int | float]


def analyse_image(
    image_path: str | Path,
    config: dict[str, Any],
) -> PipelineResult:
    sample = load_image(image_path)
    # Local import avoids making the data-contract module initialise the GUI
    # during ordinary imports while keeping CLI and GUI preprocessing identical.
    from app import _process_tiled_pipeline_image

    output = _process_tiled_pipeline_image(sample.image, config)

    return PipelineResult(
        sample_id=sample.sample_id,
        enhancement=output["enhancement"],
        segmentation=output["segmentation"],
        features=output["features"],
        spatial_scaling=output["spatial_scaling"],
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
        "03_white_top_hat.png": result.enhancement.segmentation_input,
        "04_threshold.png": result.segmentation.intermediate_images["threshold"],
        "05_morphological_opening.png": result.segmentation.intermediate_images[
            "morphological_opening"
        ],
        "06_morphological_closing.png": result.segmentation.intermediate_images[
            "morphological_closing"
        ],
        "07_segmentation_mask.png": result.segmentation.binary_mask,
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
