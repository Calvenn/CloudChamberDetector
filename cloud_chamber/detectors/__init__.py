"""Detector registration and built-in implementations."""

from cloud_chamber.detectors.base import Detector
from cloud_chamber.detectors.registry import (
    create_detector,
    list_detectors,
    register_detector,
)

__all__ = [
    "Detector",
    "OtsuBaselineDetector",
    "create_detector",
    "list_detectors",
    "register_detector",
]

