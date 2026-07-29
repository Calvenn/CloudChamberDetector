"""Protocols for the shared segmenter and independent member models."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from cloud_chamber.ml.contracts import ModelResult, SegmentationResult


class Segmenter(Protocol):
    name: str

    def segment(self, enhanced_image: np.ndarray) -> SegmentationResult:
        """Return class-agnostic particle instances."""


class ParticleModel(Protocol):
    name: str

    def predict(
        self,
        enhanced_image: np.ndarray,
        segmentation: SegmentationResult,
    ) -> ModelResult:
        """Automatically extract features and classify each instance."""
