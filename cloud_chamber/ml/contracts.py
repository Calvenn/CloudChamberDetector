"""Standard outputs shared by segmentation and member ML models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cloud_chamber.models import BoundingBox


@dataclass(frozen=True)
class SegmentedInstance:
    """One class-agnostic particle track predicted by Mask R-CNN."""

    instance_id: int
    bounding_box: BoundingBox
    mask: np.ndarray
    confidence: float


@dataclass(frozen=True)
class SegmentationResult:
    """Mask R-CNN output passed identically to every member model."""

    model_name: str
    instances: list[SegmentedInstance]
    processing_time_ms: float


@dataclass(frozen=True)
class ClassPrediction:
    """Classification of one segmented particle track."""

    instance_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResult:
    """Comparable output returned by every independently developed model."""

    model_name: str
    predictions: list[ClassPrediction]
    processing_time_ms: float
    feature_dimensions: int | None = None
