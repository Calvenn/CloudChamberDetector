"""Shared enhancement stages used before every Mode A detector.

This module implements the selected shared enhancement: greyscale conversion
followed by Gaussian filtering. Every machine-learning model receives the same
filtered greyscale image.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cloud_chamber.models import EnhancementResult


def enhance_image(
    image: np.ndarray,
    settings: dict[str, Any],
) -> EnhancementResult:
    """Apply the fixed shared enhancement pipeline to one image."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    # Stage 1: standardise the detector input to an 8-bit greyscale image.
    grey = _to_greyscale_uint8(image)

    kernel = int(settings["gaussian_kernel"])
    sigma = float(settings["gaussian_sigma"])
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("gaussian_kernel must be a positive odd integer")

    # Stage 2: Gaussian smoothing reduces sensor noise and small bright dots.
    denoised = cv2.GaussianBlur(
        grey,
        (kernel, kernel),
        sigmaX=sigma,
    )

    return EnhancementResult(
        grey=grey,
        denoised=denoised,
        enhanced=denoised,
    )


def _to_greyscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or greyscale array into a normalised uint8 image."""
    if image.ndim == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        grey = image.copy()
    else:
        raise ValueError("Input image must be greyscale or BGR colour")
    return _as_uint8(grey)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise non-uint8 arrays to the image-processing range 0-255."""
    if image.dtype == np.uint8:
        return image
    normalised = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalised.astype(np.uint8)
