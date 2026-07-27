"""Shared enhancement stages used before every Mode A detector.

This module converts images to greyscale, removes noise with a Gaussian filter,
enhances local contrast with CLAHE, and optionally subtracts an equivalently
processed reference frame. Individual detectors receive the final enhanced
greyscale image.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cloud_chamber.models import EnhancementResult


def enhance_image(
    image: np.ndarray,
    settings: dict[str, Any],
    background_reference: np.ndarray | None = None,
) -> EnhancementResult:
    """Apply the fixed shared enhancement pipeline to one image."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    # Stage 1: standardise the detector input to an 8-bit greyscale image.
    grey = _to_greyscale_uint8(image)

    kernel = int(settings["gaussian_kernel"])
    sigma = float(settings["gaussian_sigma"])
    grid_size = int(settings["clahe_grid_size"])

    # Stage 2: Gaussian smoothing reduces sensor noise and small bright dots.
    denoised = cv2.GaussianBlur(
        grey,
        (kernel, kernel),
        sigmaX=sigma,
    )

    # Stage 3: CLAHE improves local contrast without amplifying the complete
    # image as aggressively as ordinary histogram equalisation.
    clahe = cv2.createCLAHE(
        clipLimit=float(settings["clahe_clip_limit"]),
        tileGridSize=(grid_size, grid_size),
    )
    contrast_enhanced = clahe.apply(denoised)

    # Stage 4: subtract an equivalently processed reference when one is
    # available. This isolates newly appearing or moving tracks while avoiding
    # differences caused only by applying enhancement to one frame.
    background_corrected = contrast_enhanced
    subtraction_applied = background_reference is not None
    if background_reference is not None:
        reference_grey = _to_greyscale_uint8(background_reference)
        if reference_grey.shape != grey.shape:
            raise ValueError(
                "Background reference must have the same dimensions as input"
            )
        reference_denoised = cv2.GaussianBlur(
            reference_grey,
            (kernel, kernel),
            sigmaX=sigma,
        )
        reference_enhanced = clahe.apply(reference_denoised)
        background_corrected = cv2.absdiff(
            contrast_enhanced,
            reference_enhanced,
        )

    return EnhancementResult(
        grey=grey,
        denoised=denoised,
        contrast_enhanced=contrast_enhanced,
        background_corrected=background_corrected,
        enhanced=background_corrected,
        background_subtraction_applied=subtraction_applied,
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
