"""Shared grayscale-conversion and Gaussian-filtering preprocessing."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cloud_chamber.models import EnhancementResult


def enhance_image(
    image: np.ndarray,
    settings: dict[str, Any],
    segmentation_settings: dict[str, Any] | None = None,
) -> EnhancementResult:
    """Apply grayscale, Gaussian and optional white-top-hat enhancement."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    # Stage 1: standardise the detector input to an 8-bit greyscale image.
    grey = _to_greyscale_uint8(image)

    kernel = int(settings["gaussian_kernel"])
    sigma = float(settings["gaussian_sigma"])
    # Stage 2: Gaussian smoothing reduces sensor noise and small bright dots.
    denoised = cv2.GaussianBlur(
        grey,
        (kernel, kernel),
        sigmaX=sigma,
    )

    # White top-hat is conceptually an enhancement operation. Keep the
    # Gaussian result for the established intensity-feature contract and expose
    # local contrast separately as the thresholding input.
    local_contrast = None
    if segmentation_settings is not None:
        top_hat_size = int(segmentation_settings["top_hat_kernel"])
        if top_hat_size < 1 or top_hat_size % 2 == 0:
            raise ValueError("top_hat_kernel must be a positive odd number")
        top_hat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (top_hat_size, top_hat_size),
        )
        local_contrast = cv2.morphologyEx(
            denoised,
            cv2.MORPH_TOPHAT,
            top_hat_kernel,
        )

    return EnhancementResult(
        grey=grey,
        denoised=denoised,
        enhanced=denoised,
        local_contrast=local_contrast,
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
