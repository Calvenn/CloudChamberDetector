"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ``config.yaml`` is stored at the project root, two levels above this module
# (``cloud_chamber/core/config.py``).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML settings and reject missing or unsafe processing values."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_sections = {
        "paths",
        "acquisition",
        "enhancement",
        "spatial_scaling",
        "segmentation",
        "evaluation",
    }
    missing = required_sections.difference(config or {})
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing configuration sections: {names}")

    processing = config["spatial_scaling"]
    if int(processing["tile_size"]) < 2:
        raise ValueError("spatial_scaling.tile_size must exceed one pixel")
    overlap_ratio = float(processing["overlap_ratio"])
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("spatial_scaling.overlap_ratio must be in [0, 1)")
    for dimension in ("processing_width", "processing_height"):
        if int(processing[dimension]) < 2:
            raise ValueError(f"spatial_scaling.{dimension} must exceed one pixel")
    if int(processing["processing_width"]) != int(processing["processing_height"]):
        raise ValueError(
            "spatial_scaling processing width and height must match for a 1:1 ROI"
        )
    if int(processing.get("pixel_parameter_reference_size", 1920)) < 2:
        raise ValueError(
            "spatial_scaling.pixel_parameter_reference_size must exceed one pixel"
        )

    kernel = int(config["enhancement"]["gaussian_kernel"])
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("enhancement.gaussian_kernel must be a positive odd number")

    segmentation = config["segmentation"]
    for kernel_name in ("top_hat_kernel", "closing_kernel", "opening_kernel"):
        morphology_kernel = int(segmentation[kernel_name])
        if morphology_kernel < 1 or morphology_kernel % 2 == 0:
            raise ValueError(
                f"segmentation.{kernel_name} must be a positive odd number"
            )
    positive_settings = (
        "minimum_object_area",
        "minimum_major_axis",
        "minimum_thin_area",
        "minimum_thin_perimeter",
        "minimum_thin_major_axis",
        "minimum_thin_aspect_ratio",
        "maximum_thin_frame_span",
        "maximum_thin_frame_thickness",
    )
    for name in positive_settings:
        if float(segmentation[name]) <= 0:
            raise ValueError(f"segmentation.{name} must be greater than zero")

    for name in ("maximum_thin_frame_span", "maximum_thin_frame_thickness"):
        if float(segmentation[name]) > 1:
            raise ValueError(f"segmentation.{name} must not exceed 1")

    for profile_name, margins in segmentation["roi_profiles"].items():
        for side in ("left", "right", "top", "bottom"):
            value = float(margins[side])
            if not 0 <= value < 0.5:
                raise ValueError(
                    f"segmentation.roi_profiles.{profile_name}.{side} "
                    "must be between 0 and 0.5"
                )

    return config
