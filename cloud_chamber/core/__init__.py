"""Shared domain contracts used across processing, ML, and reporting."""

from cloud_chamber.core.contracts import (
    BoundingBox,
    EnhancementResult,
    EvaluationResult,
    ImageSample,
    SegmentationResult,
)

__all__ = [
    "BoundingBox",
    "EnhancementResult",
    "EvaluationResult",
    "ImageSample",
    "SegmentationResult",
]
