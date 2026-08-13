"""Shared thresholding, morphology and contour segmentation pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np

from cloud_chamber.models import SegmentationResult


def segment_tracks(
    enhanced_image: np.ndarray,
    settings: dict[str, Any],
    roi_margins: dict[str, float] | None = None,
) -> SegmentationResult:
    """Segment bright particle tracks and return their external contours."""
    if enhanced_image.ndim != 2 or enhanced_image.dtype != np.uint8:
        raise ValueError("Segmentation input must be an 8-bit grayscale image")

    started = perf_counter()
    top_hat_size = int(settings["top_hat_kernel"])
    top_hat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (top_hat_size, top_hat_size)
    )
    local_bright = cv2.morphologyEx(
        enhanced_image, cv2.MORPH_TOPHAT, top_hat_kernel
    )
    otsu_value, _ = cv2.threshold(
        local_bright, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_value = min(255.0, otsu_value + float(settings["threshold_offset"]))
    _, threshold_mask = cv2.threshold(
        local_bright, threshold_value, 255, cv2.THRESH_BINARY
    )

    image_height, image_width = enhanced_image.shape
    margins = roi_margins or {side: 0.0 for side in ("left", "right", "top", "bottom")}
    roi_left = int(round(image_width * float(margins["left"])))
    roi_right = image_width - int(round(image_width * float(margins["right"])))
    roi_top = int(round(image_height * float(margins["top"])))
    roi_bottom = image_height - int(round(image_height * float(margins["bottom"])))
    if roi_left >= roi_right or roi_top >= roi_bottom:
        raise ValueError("ROI margins leave no usable image region")
    roi_mask = np.zeros_like(threshold_mask)
    roi_mask[roi_top:roi_bottom, roi_left:roi_right] = 255
    # Mask before closing so chamber walls cannot bleed back into the ROI.
    threshold_mask = cv2.bitwise_and(threshold_mask, roi_mask)

    closing_size = int(settings["closing_kernel"])
    opening_size = int(settings["opening_kernel"])
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    # Closing must occur before opening: it reconnects short gaps while the
    # original thin track is still present. Opening then removes residual dots.
    closed = cv2.morphologyEx(
        threshold_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=int(settings["closing_iterations"]),
    )
    opening_applied = bool(settings.get("apply_opening", True))
    if opening_applied:
        refined = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            opening_kernel,
            iterations=int(settings["opening_iterations"]),
        )
    else:
        # Preserve the closed mask exactly. Noise rejection is performed using
        # contour area, length and aspect ratio below, rather than erosion.
        refined = closed.copy()

    found, _ = cv2.findContours(
        refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    minimum_area = float(settings["minimum_object_area"])
    minimum_major_axis = float(settings["minimum_major_axis"])
    minimum_thin_area = float(settings["minimum_thin_area"])
    minimum_thin_perimeter = float(settings["minimum_thin_perimeter"])
    minimum_thin_major_axis = float(settings["minimum_thin_major_axis"])
    minimum_thin_aspect = float(settings["minimum_thin_aspect_ratio"])
    enable_thin_track_rule = bool(settings.get("enable_thin_track_rule", True))
    maximum_span = float(settings["maximum_thin_frame_span"])
    maximum_thickness = float(settings["maximum_thin_frame_thickness"])
    contours = []
    rejected_contours = []
    for contour in found:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)
        major_axis = float(max(side_a, side_b))
        minor_axis = float(min(side_a, side_b))
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0
        x, y, width, height = cv2.boundingRect(contour)

        # A general path retains substantial tracks, while the second path
        # preserves thin electron-like contours near their labelled lower-tail
        # dimensions. Isolated dots cannot meet the area/perimeter combination.
        passes_general_track_rule = (
            area >= minimum_area and major_axis >= minimum_major_axis
        )
        passes_thin_track_rule = (
            area >= minimum_thin_area
            and perimeter >= minimum_thin_perimeter
            and major_axis >= minimum_thin_major_axis
            and aspect_ratio >= minimum_thin_aspect
        )
        # A chamber rim can appear as a nearly frame-wide one-pixel line. This
        # rule rejects only near-frame-spanning AND very thin regions, so an
        # ordinary long proton track is not rejected merely for being long.
        spans_width = width / image_width >= maximum_span
        spans_height = height / image_height >= maximum_span
        thin_horizontally = height / image_height <= maximum_thickness
        thin_vertically = width / image_width <= maximum_thickness
        is_thin_frame_artifact = (
            (spans_width and thin_horizontally)
            or (spans_height and thin_vertically)
        )
        if (
            (
                passes_general_track_rule
                or (enable_thin_track_rule and passes_thin_track_rule)
            )
            and not is_thin_frame_artifact
        ):
            contours.append(contour)
        else:
            rejected_contours.append(contour)

    clean_mask = np.zeros_like(refined)
    if contours:
        cv2.drawContours(clean_mask, contours, -1, 255, cv2.FILLED)
    # Re-read contours from the final accepted mask. This guarantees that the
    # returned contours and bounding boxes describe exactly the same white
    # regions shown in the GUI and later used for feature extraction.
    final_contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    rejected_mask = np.zeros_like(refined)
    if rejected_contours:
        cv2.drawContours(rejected_mask, rejected_contours, -1, 255, cv2.FILLED)
    boxes = [
        tuple(int(v) for v in cv2.boundingRect(contour))
        for contour in final_contours
    ]

    return SegmentationResult(
        method_name="Otsu thresholding + morphological opening/closing",
        binary_mask=clean_mask,
        bounding_boxes=boxes,
        contours=final_contours,
        processing_time_ms=(perf_counter() - started) * 1000.0,
        intermediate_images={
            "local_bright_tracks": local_bright,
            "region_of_interest": roi_mask,
            "threshold": threshold_mask,
            "morphological_closing": closed,
            "morphological_opening": refined,
            "rejected_small_blobs": rejected_mask,
        },
        parameters={
            "threshold": "White top-hat + Otsu plus offset",
            "top_hat_kernel": top_hat_size,
            "otsu_value": float(otsu_value),
            "threshold_offset": float(settings["threshold_offset"]),
            "applied_threshold": float(threshold_value),
            "closing_kernel": closing_size,
            "opening_kernel": opening_size,
            "opening_applied": opening_applied,
            "minimum_area": minimum_area,
            "minimum_major_axis": minimum_major_axis,
            "minimum_thin_area": minimum_thin_area,
            "minimum_thin_perimeter": minimum_thin_perimeter,
            "minimum_thin_major_axis": minimum_thin_major_axis,
            "minimum_thin_aspect_ratio": minimum_thin_aspect,
            "thin_track_rule_enabled": enable_thin_track_rule,
            "candidate_count_before_filter": len(found),
            "candidate_count_after_filter": len(final_contours),
            "rejected_candidate_count": len(rejected_contours),
            "roi_pixels": {
                "left": roi_left,
                "right": roi_right,
                "top": roi_top,
                "bottom": roi_bottom,
            },
        },
    )
