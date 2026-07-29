"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required_sections = {
        "paths",
        "acquisition",
        "enhancement",
        "segmentation",
        "evaluation",
        "classification",
    }
    missing = required_sections.difference(config or {})
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing configuration sections: {names}")

    kernel = int(config["enhancement"]["gaussian_kernel"])
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("enhancement.gaussian_kernel must be a positive odd number")

    threshold = float(config["segmentation"]["confidence_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "segmentation.confidence_threshold must be between 0 and 1"
        )

    return config
