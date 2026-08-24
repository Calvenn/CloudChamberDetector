"""Shared thresholding, morphology and contour segmentation pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np


def scale_pixel_parameters(
    settings: dict,
    reference_size: int,
    processing_size: tuple[int, int],
) -> dict:
    """Preserve existing pixel-parameter proportions at a new resolution.

    This does not introduce a new enhancement or segmentation technique. It
    converts parameters previously expressed at the 1920-pixel reference into
    equivalent lengths and areas at the configured ROI processing size.
    """
    if reference_size < 2:
        raise ValueError("Pixel-parameter reference size must exceed one")
    ratio = max(processing_size) / float(reference_size)
    scaled = dict(settings)

    for name in (
        "minimum_major_axis",
        "minimum_thin_perimeter",
        "minimum_thin_major_axis",
        "alignment_merge_gap",
    ):
        if name in scaled:
            scaled[name] = max(1, int(round(float(scaled[name]) * ratio)))

    for name in ("minimum_object_area", "minimum_thin_area"):
        if name in scaled:
            scaled[name] = max(1, int(round(float(scaled[name]) * ratio * ratio)))

    for name in (
        "top_hat_kernel",
        "closing_kernel",
        "opening_kernel",
        "directional_closing_length",
    ):
        if name in scaled:
            value = max(1, int(round(float(scaled[name]) * ratio)))
            scaled[name] = value if value % 2 == 1 else value + 1

    if "hysteresis_seed_pixels" in scaled:
        scaled["hysteresis_seed_pixels"] = max(
            1, int(round(float(scaled["hysteresis_seed_pixels"]) * ratio * ratio))
        )
    scaled["pixel_parameter_scale"] = ratio
    scaled["pixel_parameter_reference_size"] = int(reference_size)
    return scaled

from cloud_chamber.models import SegmentationResult


def segment_tracks(
    enhanced_image: np.ndarray,
    settings: dict[str, Any],
    roi_margins: dict[str, float] | None = None,
) -> SegmentationResult:
    """Segment bright particle tracks and return their external contours."""
    if enhanced_image.ndim != 2 or enhanced_image.dtype != np.uint8:
        raise ValueError("Segmentation input must be an 8-bit grayscale image")
    threshold_method = str(
        settings.get("threshold_method", "white_tophat_otsu_hysteresis")
    )
    if threshold_method != "white_tophat_otsu_hysteresis":
        raise ValueError(
            "Unsupported threshold_method. Expected "
            "'white_tophat_otsu_hysteresis'."
        )

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
    # Optional Otsu-derived hysteresis retains faint extensions only when they
    # belong to a low-threshold component containing reliable high-threshold
    # pixels. Unlike simply lowering Otsu, isolated dim background texture has
    # no strong seed and is discarded.
    low_threshold_offset = float(
        settings.get("hysteresis_low_threshold_offset", settings["threshold_offset"])
    )
    hysteresis_seed_pixels = int(settings.get("hysteresis_seed_pixels", 1))
    low_threshold_value = min(255.0, otsu_value + low_threshold_offset)
    hysteresis_applied = low_threshold_value < threshold_value
    if hysteresis_applied:
        _, low_mask = cv2.threshold(
            local_bright, low_threshold_value, 255, cv2.THRESH_BINARY
        )
        component_count, component_labels = cv2.connectedComponents(
            low_mask, connectivity=8
        )
        seed_counts = np.bincount(
            component_labels[threshold_mask > 0], minlength=component_count
        )
        keep_component = seed_counts >= hysteresis_seed_pixels
        keep_component[0] = False
        threshold_mask = np.where(
            keep_component[component_labels], 255, 0
        ).astype(np.uint8)

    # Optional thin-line branch. The faint mask is opened with short line
    # elements at several orientations. Compact droplets cannot contain the
    # line element, while locally straight parts of a one-pixel track can.
    thin_line_length = int(settings.get("thin_line_opening_length", 1))
    thin_line_threshold_offset = float(
        settings.get("thin_line_threshold_offset", low_threshold_offset)
    )
    thin_line_mask = np.zeros_like(threshold_mask)
    if thin_line_length > 1:
        if thin_line_length % 2 == 0:
            raise ValueError("thin_line_opening_length must be odd")
        _, faint_mask = cv2.threshold(
            local_bright,
            min(255.0, otsu_value + thin_line_threshold_offset),
            255,
            cv2.THRESH_BINARY,
        )
        centre = thin_line_length // 2
        for angle_degrees in range(0, 180, 15):
            radians = np.deg2rad(angle_degrees)
            dx = int(round(centre * np.cos(radians)))
            dy = int(round(centre * np.sin(radians)))
            line_kernel = np.zeros(
                (thin_line_length, thin_line_length), dtype=np.uint8
            )
            cv2.line(
                line_kernel,
                (centre - dx, centre - dy),
                (centre + dx, centre + dy),
                1,
                1,
            )
            oriented = cv2.morphologyEx(
                faint_mask, cv2.MORPH_OPEN, line_kernel, iterations=1
            )
            thin_line_mask = cv2.bitwise_or(thin_line_mask, oriented)
        threshold_mask = cv2.bitwise_or(threshold_mask, thin_line_mask)

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
    # A long circular kernel joins objects in every direction and previously
    # merged unrelated droplets into giant boxes. Optional one-pixel line
    # kernels reconnect only fragments that are aligned along a likely track.
    directional_length = int(settings.get("directional_closing_length", 1))
    if directional_length > 1:
        if directional_length % 2 == 0:
            raise ValueError("directional_closing_length must be odd")
        centre = directional_length // 2
        directional_closed = closed.copy()
        for angle_degrees in range(0, 180, 30):
            angle = np.deg2rad(angle_degrees)
            dx = int(round(centre * np.cos(angle)))
            dy = int(round(centre * np.sin(angle)))
            line_kernel = np.zeros(
                (directional_length, directional_length), dtype=np.uint8
            )
            cv2.line(
                line_kernel,
                (centre - dx, centre - dy),
                (centre + dx, centre + dy),
                1,
                1,
            )
            aligned = cv2.morphologyEx(
                threshold_mask,
                cv2.MORPH_CLOSE,
                line_kernel,
                iterations=1,
            )
            directional_closed = cv2.bitwise_or(directional_closed, aligned)
        closed = cv2.bitwise_and(directional_closed, roi_mask)
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
    refined = cv2.bitwise_and(refined, roi_mask)
    found, _ = cv2.findContours(
        refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    # Link plausible fragments before size filtering. Previously, alignment
    # merging ran only after filtering, so short sections of one long dotted
    # track were discarded before they had a chance to form a valid contour.
    alignment_merge_gap = float(settings.get("alignment_merge_gap", 0))
    alignment_merge_angle = float(settings.get("alignment_merge_angle", 25))
    alignment_merge_minimum_aspect = float(
        settings.get("alignment_merge_minimum_aspect", 2.5)
    )
    prefilter_alignment_links = 0
    prefilter_alignment_minimum_aspect = float(
        settings.get(
            "prefilter_alignment_minimum_aspect",
            alignment_merge_minimum_aspect,
        )
    )
    if (
        bool(settings.get("prefilter_alignment_merge", True))
        and alignment_merge_gap > 0
        and len(found) > 1
    ):
        refined, prefilter_alignment_links = _link_aligned_contours(
            refined,
            found,
            maximum_gap=alignment_merge_gap,
            maximum_angle_difference=alignment_merge_angle,
            minimum_aspect_ratio=prefilter_alignment_minimum_aspect,
        )
        refined = cv2.bitwise_and(refined, roi_mask)
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

        # Reject any contour whose center lies outside the active ROI bounds
        cx = x + width // 2
        cy = y + height // 2
        if cx < roi_left or cx >= roi_right or cy < roi_top or cy >= roi_bottom:
            rejected_contours.append(contour)
            continue

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
    alignment_links = 0
    if alignment_merge_gap > 0 and len(contours) > 1:
        clean_mask, alignment_links = _link_aligned_contours(
            clean_mask,
            contours,
            maximum_gap=alignment_merge_gap,
            maximum_angle_difference=alignment_merge_angle,
            minimum_aspect_ratio=alignment_merge_minimum_aspect,
        )
    clean_mask = cv2.bitwise_and(clean_mask, roi_mask)
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
        method_name="Otsu-guided hysteresis + morphological refinement",
        binary_mask=clean_mask,
        bounding_boxes=boxes,
        contours=final_contours,
        processing_time_ms=(perf_counter() - started) * 1000.0,
        intermediate_images={
            "local_bright_tracks": local_bright,
            "region_of_interest": roi_mask,
            "threshold": threshold_mask,
            "thin_line_candidates": thin_line_mask,
            "morphological_closing": closed,
            "morphological_opening": refined,
            "rejected_small_blobs": rejected_mask,
        },
        parameters={
            "threshold": "White top-hat + Otsu-guided hysteresis",
            "threshold_method": threshold_method,
            "top_hat_kernel": top_hat_size,
            "otsu_value": float(otsu_value),
            "threshold_offset": float(settings["threshold_offset"]),
            "applied_threshold": float(threshold_value),
            "hysteresis_applied": hysteresis_applied,
            "hysteresis_low_threshold_offset": low_threshold_offset,
            "hysteresis_low_threshold_value": float(low_threshold_value),
            "hysteresis_seed_pixels": hysteresis_seed_pixels,
            "thin_line_opening_length": thin_line_length,
            "thin_line_threshold_offset": thin_line_threshold_offset,
            "thin_line_pixels": int(np.count_nonzero(thin_line_mask)),
            "closing_kernel": closing_size,
            "directional_closing_length": directional_length,
            "alignment_merge_gap": alignment_merge_gap,
            "prefilter_alignment_minimum_aspect": (
                prefilter_alignment_minimum_aspect
            ),
            "prefilter_alignment_links_created": prefilter_alignment_links,
            "alignment_links_created": alignment_links,
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


def _angle_difference(first: float, second: float) -> float:
    """Return the smallest difference between two undirected line angles."""
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _contour_axis(contour: np.ndarray) -> dict[str, object]:
    """Describe the centre, endpoints and direction of a contour's major axis."""
    (centre_x, centre_y), (side_a, side_b), rectangle_angle = cv2.minAreaRect(contour)
    if side_a >= side_b:
        major_axis = float(side_a)
        minor_axis = float(side_b)
        angle = float(rectangle_angle)
    else:
        major_axis = float(side_b)
        minor_axis = float(side_a)
        angle = float(rectangle_angle + 90.0)
    radians = np.deg2rad(angle)
    direction = np.asarray([np.cos(radians), np.sin(radians)], dtype=np.float64)
    centre = np.asarray([centre_x, centre_y], dtype=np.float64)
    half_axis = direction * (major_axis / 2.0)
    return {
        "angle": angle % 180.0,
        "aspect": major_axis / minor_axis if minor_axis > 0 else 0.0,
        "endpoints": (centre - half_axis, centre + half_axis),
    }


def _link_aligned_contours(
    mask: np.ndarray,
    contours: list[np.ndarray],
    maximum_gap: float,
    maximum_angle_difference: float,
    minimum_aspect_ratio: float,
) -> tuple[np.ndarray, int]:
    """Join nearby collinear accepted fragments using one-pixel connectors.

    Both fragments must already pass the contour filters, both must be
    elongated, and the closest major-axis endpoints must align with both track
    directions. Consequently a nearby round droplet is not used as a bridge.
    """
    descriptions = [_contour_axis(contour) for contour in contours]
    linked = mask.copy()
    link_count = 0
    for first_index, first in enumerate(descriptions):
        if float(first["aspect"]) < minimum_aspect_ratio:
            continue
        for second in descriptions[first_index + 1 :]:
            if float(second["aspect"]) < minimum_aspect_ratio:
                continue
            if _angle_difference(
                float(first["angle"]), float(second["angle"])
            ) > maximum_angle_difference:
                continue

            endpoint_pairs = [
                (first_point, second_point)
                for first_point in first["endpoints"]
                for second_point in second["endpoints"]
            ]
            start, stop = min(
                endpoint_pairs,
                key=lambda pair: float(np.linalg.norm(pair[1] - pair[0])),
            )
            connector = stop - start
            distance = float(np.linalg.norm(connector))
            if distance == 0 or distance > maximum_gap:
                continue
            connector_angle = float(
                np.degrees(np.arctan2(connector[1], connector[0])) % 180.0
            )
            if (
                _angle_difference(connector_angle, float(first["angle"]))
                > maximum_angle_difference
                or _angle_difference(connector_angle, float(second["angle"]))
                > maximum_angle_difference
            ):
                continue
            cv2.line(
                linked,
                tuple(np.rint(start).astype(int)),
                tuple(np.rint(stop).astype(int)),
                255,
                1,
                cv2.LINE_8,
            )
            link_count += 1
    return linked, link_count
