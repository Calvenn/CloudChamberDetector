"""Shared Streamlit workflow for acquisition, processing, and segmentation."""

from __future__ import annotations

import hashlib
import gc
import tempfile
from dataclasses import replace
from time import perf_counter, sleep
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from cloud_chamber.image_processing.calibration import (
    RoiScalingResult,
    default_roi_coordinates,
    detect_chamber_corners,
    extract_roi,
    rectify_image,
    rectify_image_with_transform,
    select_and_scale_roi,
    spatial_scale_roi,
)
from cloud_chamber.image_processing.enhancement import enhance_image
from cloud_chamber.feature_extraction.contour_features import extract_track_features
from cloud_chamber.ml.artifact_filter import (
    filter_mask as filter_artifact_candidates,
    load_model as load_artifact_filter,
)
from cloud_chamber.core.contracts import EnhancementResult, SegmentationResult
from cloud_chamber.image_processing.segmentation import scale_pixel_parameters, segment_tracks
from cloud_chamber.image_processing.tiling import (
    coverage_map,
    draw_tile_boundaries,
    generate_overlapping_tiles,
    map_processed_mask_to_image,
    merge_tile_masks,
    spatial_scale_tile,
)
from cloud_chamber.image_processing.video_processing import temporal_track_composite
from cloud_chamber.ui.feature_rows import (
    advanced_feature_row,
    basic_feature_row,
    feature_row,
)


def initialise_state() -> None:
    """Create session keys once so page changes do not erase user results."""
    st.session_state.setdefault("input_batch", [])
    st.session_state.setdefault("input_image", None)
    st.session_state.setdefault("input_name", None)
    st.session_state.setdefault("input_images", [])
    st.session_state.setdefault("selected_input_index", 0)
    st.session_state.setdefault("source_description", None)
    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("pipeline_result_signature", None)
    st.session_state.setdefault("pipeline_results_by_input", {})
    st.session_state.setdefault("input_source_type", "Image")
    st.session_state.setdefault("shared_segmentation_results", {})
    st.session_state.setdefault("image_upload_signature", None)
    st.session_state.setdefault("mlp_predictions", None)
    st.session_state.setdefault("cnn_predictions", None)
    st.session_state.setdefault("svm_predictions", None)
    st.session_state.setdefault("extra_trees_predictions", None)
    st.session_state.setdefault("extra_trees_prediction_signature", None)
    st.session_state.setdefault("decision_tree_predictions", None)
    st.session_state.setdefault(
        "layout_choice", "Auto-detect from image shape (recommended)"
    )
    st.session_state.setdefault("mlp_quality", None)
    st.session_state.setdefault("extra_trees_quality", None)
    st.session_state.setdefault("extra_trees_batch_results", {})
    st.session_state.setdefault("extra_trees_batch_reports", {})
    st.session_state.setdefault("decision_tree_quality", None)
    st.session_state.setdefault("batch_reports", {})
    st.session_state.setdefault("mlp_batch_results", {})
    st.session_state.setdefault("decision_tree_batch_results", {})
    st.session_state.setdefault("svm_batch_results", {})
    st.session_state.setdefault("calibration_settings", None)
    st.session_state.setdefault("rectification_settings_by_input", {})
    st.session_state.setdefault("config", None)


def render(config: dict) -> None:
    """Render acquisition, preprocessing, segmentation, and feature evidence."""
    st.title("Image Processing Pipeline")

    st.header("Image or video-frame acquisition")
    _acquisition_section(config)
    image = st.session_state.get("input_image")
    if image is None:
        return

    rectification_settings = _automatic_rectification_section(image)
    processing_settings = {**rectification_settings}
    result = _process_tiled_pipeline_image(image, config, processing_settings)
    analysis_image = result["input_image"]
    enhancement = result["enhancement"]
    segmentation = result["segmentation"]
    features = result["features"]
    scaling = result["spatial_scaling"]
    st.header("Automatic overlapping tiling and spatial scaling")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Tiles", scaling["tile_count"])
    metric_columns[1].metric("Tile size", f"{scaling['tile_size']} × {scaling['tile_size']}")
    metric_columns[2].metric("Processing size", scaling["processing_resolution"])
    metric_columns[3].metric("Minimum coverage", scaling["minimum_coverage"])
    st.caption(
        f"The complete rectified processing copy is covered using "
        f"{scaling['overlap_ratio']:.0%} overlap. Every tile is spatially "
        "scaled before enhancement; the original uploaded image remains unchanged."
    )
    with st.expander("Show tile coverage and metadata"):
        st.image(
            _bgr_to_rgb(result["tile_boundary_preview"]),
            caption="Automatic overlapping tile boundaries",
        )
        st.dataframe(result["tile_metadata"], use_container_width=True, hide_index=True)
    st.header("Grayscale, Gaussian and white top-hat enhancement")
    columns = st.columns(4)
    columns[0].image(_bgr_to_rgb(analysis_image), caption="Full processing image")
    columns[1].image(enhancement.grey, caption="Grayscale")
    columns[2].image(enhancement.denoised, caption="Gaussian filtered")
    columns[3].image(
        enhancement.segmentation_input,
        caption="White top-hat enhanced",
    )

    st.header("Thresholding, morphology and contour detection")
    intermediate = segmentation.intermediate_images
    columns = st.columns(3)
    columns[0].image(intermediate["threshold"], caption="Otsu-guided hysteresis")
    columns[1].image(
        intermediate["morphological_closing"],
        caption="Morphological operation: Closing",
    )
    columns[2].image(
        segmentation.binary_mask,
        caption="Contour detection and filtering",
    )

    current_signature = (
        st.session_state.get("input_name"),
        _pipeline_result_signature(result),
    )
    pipeline_changed = (
        st.session_state.get("pipeline_result_signature") != current_signature
    )
    st.session_state["pipeline_result"] = result
    st.session_state["pipeline_result_signature"] = current_signature
    st.session_state["pipeline_results_by_input"][
        st.session_state["input_name"]
    ] = result
    if pipeline_changed:
        # Invalidate classification only when the effective segmentation or
        # extracted features change. Navigation back to this page recomputes
        # an identical result and must preserve the existing model values.
        st.session_state["mlp_predictions"] = None
        st.session_state["cnn_predictions"] = None
        st.session_state["mlp_quality"] = None
        st.session_state["decision_tree_predictions"] = None
        st.session_state["decision_tree_quality"] = None
        st.session_state["svm_predictions"] = None
        st.session_state["svm_quality"] = None
        st.session_state["extra_trees_predictions"] = None
        st.session_state["extra_trees_prediction_signature"] = None
        st.session_state["mlp_batch_results"] = {}
        st.session_state["decision_tree_batch_results"] = {}
        st.session_state["svm_batch_results"] = {}
        st.session_state["extra_trees_batch_results"] = {}
        st.session_state["batch_reports"].pop(
            st.session_state.get("input_name"), None
        )

    overlay = _colour_instance_mask(
        image, result["original_segmentation_mask"], opacity=0.52
    )
    overlay = _draw_track_ids(
        overlay,
        features,
        result.get("original_bounding_boxes"),
    )
    _render_batch_segmentation_summary(config, result)

    st.header("Contour-based feature extraction")
    if features:
        st.subheader("Basic features")
        st.dataframe(
            [basic_feature_row(item) for item in features],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Show advanced features and measurement definitions"):
            st.dataframe(
                [advanced_feature_row(item) for item in features],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                """
- **Area** is the number of pixels inside the contour.
- **Length** is the contour's major-axis length; **mean width** estimates its average thickness.
- **Aspect ratio** compares length with minor-axis width.
- **Solidity** compares contour area with its convex hull, while **rectangularity** compares it with its oriented rectangle.
- **Circularity** is high for compact round regions and low for elongated tracks.
- **Convexity** compares convex-hull perimeter with contour perimeter.
- **Orientation sine/cosine** encode direction without treating opposite ends of the same axis as different track directions.
"""
            )
        if result["centimetres_per_pixel"] is not None:
            with st.expander("Show calibrated physical measurements"):
                st.dataframe(
                    [
                        feature_row(item, result["centimetres_per_pixel"])
                        for item in features
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.warning("No contour passed the configured minimum-area filter.")


def _processing_size(config: dict) -> tuple[int, int]:
    """Return the single configured ROI processing resolution."""
    settings = config.get("spatial_scaling", {})
    return (
        int(settings["processing_width"]),
        int(settings["processing_height"]),
    )


def _roi_source_key(image: np.ndarray) -> str:
    """Identify an acquired image independently of its display filename."""
    height, width = image.shape[:2]
    digest = hashlib.sha1(image.tobytes()).hexdigest()[:12]
    return f"{width}x{height}:{digest}"


def _roi_selection_section(image: np.ndarray, config: dict) -> dict | None:
    """Display a mouse-controlled ROI on the untouched original image."""
    st.header("Dynamic ROI selection")
    configured_width, configured_height = _processing_size(config)
    configured_resolution = configured_width
    candidates = (512, 640, 800, 992, 1024)
    resolution_options = [f"{value} × {value}" for value in candidates] + ["Custom"]
    configured_label = f"{configured_resolution} × {configured_resolution}"
    default_index = (
        resolution_options.index(configured_label)
        if configured_label in resolution_options
        else len(resolution_options) - 1
    )
    resolution_choice = st.selectbox(
        "Processing resolution",
        resolution_options,
        index=default_index,
        key="roi_processing_resolution_choice",
        help=(
            "Candidate values are experimental options, not a ranking of "
            "scientific quality. The selected ROI remains unchanged."
        ),
    )
    if resolution_choice == "Custom":
        processing_resolution = int(
            st.number_input(
                "Custom square resolution (pixels)",
                min_value=128,
                max_value=2048,
                value=configured_resolution,
                step=32,
                key="roi_custom_processing_resolution",
            )
        )
    else:
        processing_resolution = int(resolution_choice.split(" ")[0])
    target_size = (processing_resolution, processing_resolution)
    st.caption(
        f"The configured initial value is {configured_resolution} × "
        f"{configured_resolution}. No candidate is assumed to be universally "
        "best; use the comparison below to evaluate resolution and cost."
    )
    image_height, image_width = image.shape[:2]
    source_key = _roi_source_key(image)
    applied_settings = st.session_state["roi_settings_by_input"].get(source_key)
    stored = (
        applied_settings.get("roi_coordinates")
        if applied_settings and applied_settings.get("roi_coordinates")
        else applied_settings
    )
    if stored is None:
        default = default_roi_coordinates(image, target_size)
        stored = {
            "x": default.x,
            "y": default.y,
            "width": default.width,
            "height": default.height,
        }

    st.caption(
        "Drag the ROI to move it and use its handles to resize it. The aspect "
        "ratio is locked to 1:1. When the "
        "selection is ready, click **Apply selected ROI**. The cropper and the "
        "server-side boundary check prevent the ROI from extending outside "
        "the original image. Display fitting does not resize source data."
    )
    rgb_image = Image.fromarray(_bgr_to_rgb(image))
    box = st_cropper(
        rgb_image,
        realtime_update=True,
        default_coords=(
            int(stored["x"]),
            int(stored["x"] + stored["width"]),
            int(stored["y"]),
            int(stored["y"] + stored["height"]),
        ),
        box_color="#FFFF00",
        aspect_ratio=target_size,
        return_type="box",
        key=f"roi_cropper_{source_key}",
        should_resize_image=True,
        stroke_width=3,
    )
    preview_result = select_and_scale_roi(image, box, target_size)
    coordinates = preview_result.coordinates
    selected = {
        "x": coordinates.x,
        "y": coordinates.y,
        "width": coordinates.width,
        "height": coordinates.height,
    }
    st.dataframe(
        [
            {
                "Original image": f"{image_width} × {image_height}",
                "ROI origin": f"({coordinates.x}, {coordinates.y})",
                "ROI size": f"{coordinates.width} × {coordinates.height}",
                "Processing size": f"{target_size[0]} × {target_size[1]}",
                "Scale factor": round(preview_result.scale_x, 4),
                "Operation": preview_result.scaling_operation.title(),
            }
        ],
        use_container_width=True,
        hide_index=True,
    )
    if preview_result.scale_x >= 2.0:
        st.warning(
            "Large upscaling required. Additional pixels are interpolated "
            "and do not represent additional original image detail."
        )
    if st.button("Apply selected ROI", type="primary", key=f"apply_roi_{source_key}"):
        applied_settings = {
            "roi_coordinates": selected,
            "source_key": source_key,
            "processing_size": target_size,
            "is_custom": True,
        }
        st.session_state["roi_settings_by_input"][source_key] = applied_settings
        st.success(
            f"Applied ROI ({coordinates.x}, {coordinates.y}), "
            f"{coordinates.width} × {coordinates.height} pixels."
        )

    if applied_settings is None:
        st.info("Select a region and click **Apply selected ROI** to continue.")
        return None

    settings = applied_settings
    # Keep the existing shared model-page context key so member UIs receive
    # the same selected ROI without any model-specific changes.
    st.session_state["calibration_settings"] = settings
    return settings


def _rectification_section(image: np.ndarray, roi_settings: dict) -> dict:
    """Collect perspective-correction points relative to the extracted ROI."""
    st.header("Rectification")
    target_size = tuple(int(value) for value in roi_settings["processing_size"])
    selected_roi, coordinates = extract_roi(
        image, roi_settings["roi_coordinates"], target_size
    )
    roi_height, roi_width = selected_roi.shape[:2]
    detected_corners = detect_chamber_corners(selected_roi)
    rectification_id = hashlib.sha1(
        (
            f"{roi_settings['source_key']}:"
            f"{coordinates.x},{coordinates.y},"
            f"{coordinates.width},{coordinates.height}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    coordinate_keys = [
        (
            f"roi_rectification_{rectification_id}_x_{index}",
            f"roi_rectification_{rectification_id}_y_{index}",
        )
        for index in range(4)
    ]
    source_state_key = f"roi_rectification_source_{rectification_id}"
    if not st.session_state.get(source_state_key):
        st.session_state[source_state_key] = True
        for point, (x_key, y_key) in zip(detected_corners, coordinate_keys):
            st.session_state[x_key] = int(round(float(point[0])))
            st.session_state[y_key] = int(round(float(point[1])))

    if st.button(
        "Reset rectification coordinates to automatic values",
        key=f"reset_roi_rectification_{rectification_id}",
    ):
        for point, (x_key, y_key) in zip(detected_corners, coordinate_keys):
            st.session_state[x_key] = int(round(float(point[0])))
            st.session_state[y_key] = int(round(float(point[1])))

    names = ("Top-left", "Top-right", "Bottom-right", "Bottom-left")
    points = []
    with st.expander("Rectification coordinates within selected ROI", expanded=True):
        st.caption(
            "Coordinates are relative to the extracted ROI, not the complete "
            "source image. Automatic values may be adjusted manually."
        )
        for name, (x_key, y_key) in zip(names, coordinate_keys):
            columns = st.columns(2)
            point_x = columns[0].number_input(
                f"{name} X",
                min_value=0,
                max_value=roi_width - 1,
                step=1,
                key=x_key,
            )
            point_y = columns[1].number_input(
                f"{name} Y",
                min_value=0,
                max_value=roi_height - 1,
                step=1,
                key=y_key,
            )
            points.append((float(point_x), float(point_y)))

    adjusted = np.asarray(points, dtype=np.int32)
    boundary_preview = selected_roi.copy()
    for index, (point, label) in enumerate(
        zip(adjusted, ("TL", "TR", "BR", "BL"))
    ):
        next_point = adjusted[(index + 1) % 4]
        cv2.line(boundary_preview, tuple(point), tuple(next_point), (0, 255, 255), 2)
        cv2.circle(boundary_preview, tuple(point), 6, (0, 255, 255), -1)
        cv2.putText(
            boundary_preview,
            label,
            (int(point[0]) + 8, max(int(point[1]) - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    try:
        rectified_preview = rectify_image(
            selected_roi, points, output_size=(roi_width, roi_height)
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()
    preview_columns = st.columns(2)
    preview_columns[0].image(
        _bgr_to_rgb(boundary_preview),
        caption="Selected ROI with rectification boundary",
        width=420,
    )
    preview_columns[1].image(
        _bgr_to_rgb(rectified_preview),
        caption="Perspective-rectified ROI",
        width=420,
    )
    return {
        "rectification_points": points,
        "rectification_output_size": (roi_width, roi_height),
    }


def _resolution_comparison_section(
    image: np.ndarray,
    applied_settings: dict,
    config: dict,
) -> None:
    """Compare processing resolutions while holding the applied ROI constant."""
    with st.expander("Compare processing resolutions using this ROI"):
        st.caption(
            "Every candidate uses the exact same original-image ROI. Detection "
            "counts and processing time may be compared, but no accuracy is "
            "inferred without ground-truth annotations."
        )
        candidates = st.multiselect(
            "Candidate square resolutions",
            options=[512, 640, 800, 992, 1024],
            default=[512, 640, 800, 992, 1024],
            format_func=lambda value: f"{value} × {value}",
            key="roi_resolution_comparison_candidates",
        )
        source_key = _roi_source_key(image)
        comparison_key = (
            source_key,
            tuple(sorted(applied_settings["roi_coordinates"].items())),
            tuple(
                tuple(float(value) for value in point)
                for point in applied_settings.get("rectification_points", [])
            ),
        )
        if st.button(
            "Run resolution comparison",
            disabled=not candidates,
            key=f"compare_roi_resolution_{source_key}",
        ):
            rows = []
            original_height, original_width = image.shape[:2]
            for resolution in candidates:
                candidate_settings = {
                    "roi_coordinates": dict(applied_settings["roi_coordinates"]),
                    "source_key": source_key,
                    "processing_size": (int(resolution), int(resolution)),
                    "rectification_points": applied_settings.get(
                        "rectification_points"
                    ),
                    "rectification_output_size": applied_settings.get(
                        "rectification_output_size"
                    ),
                }
                started = perf_counter()
                output = _process_pipeline_image(image, config, candidate_settings)
                elapsed_ms = (perf_counter() - started) * 1000.0
                metadata = output["spatial_scaling"]
                rows.append(
                    {
                        "Original image": f"{original_width} × {original_height}",
                        "ROI origin": f"({metadata['roi_x']}, {metadata['roi_y']})",
                        "ROI size": f"{metadata['roi_width']} × {metadata['roi_height']}",
                        "Resolution": int(resolution),
                        "Target": f"{resolution} × {resolution}",
                        "Scale factor": round(float(metadata["scale_factor"]), 4),
                        "Operation": str(metadata["scaling_operation"]).title(),
                        "Processing time (ms)": round(elapsed_ms, 2),
                        "Detected contours": len(output["segmentation"].bounding_boxes),
                    }
                )
            st.session_state["roi_resolution_comparison"] = {
                "key": comparison_key,
                "rows": rows,
            }

        comparison = st.session_state.get("roi_resolution_comparison")
        if comparison and comparison.get("key") == comparison_key:
            rows = comparison["rows"]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.bar_chart(
                [
                    {
                        "Resolution": str(row["Resolution"]),
                        "Processing time (ms)": row["Processing time (ms)"],
                    }
                    for row in rows
                ],
                x="Resolution",
                y="Processing time (ms)",
                use_container_width=True,
            )


def _process_pipeline_image(
    image: np.ndarray,
    config: dict,
    calibration_settings: dict | None = None,
) -> dict:
    """Run the identical shared pipeline for one image without drawing UI."""
    # Select from original pixels first. Enhancement never sees the complete
    # source image or a longest-side-resized version of that image.
    profile_name = _detect_layout_profile(image)
    centimetres_per_pixel = None
    source_key = _roi_source_key(image)
    saved_settings = st.session_state.get("roi_settings_by_input", {}).get(source_key)
    if saved_settings is not None and "roi_coordinates" not in saved_settings:
        # Backward compatibility for ROI state saved before the Apply button.
        saved_settings = {
            "roi_coordinates": saved_settings,
            "source_key": source_key,
        }
    explicit_settings = calibration_settings
    if (
        explicit_settings
        and explicit_settings.get("source_key") is not None
        and explicit_settings.get("source_key") != source_key
    ):
        explicit_settings = None
    roi_settings = explicit_settings or saved_settings
    configured_size = _processing_size(config)
    target_size = tuple(
        int(value)
        for value in (
            roi_settings.get("processing_size", configured_size)
            if roi_settings
            else configured_size
        )
    )
    requested_roi = (
        roi_settings.get("roi_coordinates")
        if roi_settings and roi_settings.get("roi_coordinates")
        else default_roi_coordinates(image, target_size)
    )
    selected_roi_image, roi_coordinates = extract_roi(
        image, requested_roi, target_size
    )
    rectification_points = (
        roi_settings.get("rectification_points") if roi_settings else None
    )
    if rectification_points is not None:
        rectified_roi_image, rectification_transform = rectify_image_with_transform(
            selected_roi_image,
            rectification_points,
            output_size=(roi_coordinates.width, roi_coordinates.height),
        )
        rectified = True
    else:
        rectified_roi_image = selected_roi_image.copy()
        rectification_transform = np.eye(3, dtype=np.float32)
        rectified = False
    spatially_scaled_image, scale_x, scale_y = spatial_scale_roi(
        rectified_roi_image, target_size
    )
    original_height, original_width = image.shape[:2]
    roi_transform = RoiScalingResult(
        image=spatially_scaled_image,
        roi_image=selected_roi_image,
        coordinates=roi_coordinates,
        original_width=original_width,
        original_height=original_height,
        processing_width=target_size[0],
        processing_height=target_size[1],
        scale_x=scale_x,
        scale_y=scale_y,
    )
    analysis_image = spatially_scaled_image

    selected_profile = config["segmentation"]["roi_profiles"][profile_name]
    is_custom_roi = roi_settings is not None and roi_settings.get("is_custom", False)
    if is_custom_roi:
        roi_margins = {side: 0.0 for side in ("left", "right", "top", "bottom")}
    else:
        roi_margins = {
            side: float(selected_profile.get(side, 0.0))
            for side in ("left", "right", "top", "bottom")
        }
    segmentation_settings = {
        **config["segmentation"],
        **{
            name: value
            for name, value in selected_profile.items()
            if name not in ("left", "right", "top", "bottom")
        },
    }
    segmentation_settings = scale_pixel_parameters(
        segmentation_settings,
        int(config["spatial_scaling"].get("pixel_parameter_reference_size", 1920)),
        target_size,
    )
    enhancement = enhance_image(
        analysis_image,
        config["enhancement"],
        segmentation_settings,
    )
    segmentation = segment_tracks(
        enhancement.segmentation_input,
        segmentation_settings,
        roi_margins,
    )
    feature_minimum_area = float(segmentation_settings["minimum_object_area"])
    if segmentation_settings.get("enable_thin_track_rule", True):
        feature_minimum_area = min(
            feature_minimum_area,
            float(segmentation_settings["minimum_thin_area"]),
        )
    features = extract_track_features(
        segmentation.binary_mask,
        enhancement.enhanced,
        minimum_area=feature_minimum_area,
    )
    original_boxes = []
    inverse_rectification = np.linalg.inv(rectification_transform)
    for x, y, width, height in segmentation.bounding_boxes:
        rectified_points = np.asarray(
            [
                [x / roi_transform.scale_x, y / roi_transform.scale_y],
                [(x + width) / roi_transform.scale_x, y / roi_transform.scale_y],
                [
                    (x + width) / roi_transform.scale_x,
                    (y + height) / roi_transform.scale_y,
                ],
                [x / roi_transform.scale_x, (y + height) / roi_transform.scale_y],
            ],
            dtype=np.float32,
        ).reshape(1, 4, 2)
        roi_points = cv2.perspectiveTransform(
            rectified_points, inverse_rectification
        )[0]
        minimum = roi_points.min(axis=0)
        maximum = roi_points.max(axis=0)
        original_boxes.append(
            {
                "x": roi_transform.coordinates.x + float(minimum[0]),
                "y": roi_transform.coordinates.y + float(minimum[1]),
                "width": float(maximum[0] - minimum[0]),
                "height": float(maximum[1] - minimum[1]),
            }
        )
    return {
        "original_image": image,
        "input_image": analysis_image,
        "selected_roi_image": selected_roi_image,
        "rectified_roi_image": rectified_roi_image,
        "spatially_scaled_image": spatially_scaled_image,
        "enhancement": enhancement,
        "segmentation": segmentation,
        "features": features,
        "roi_profile": profile_name,
        "centimetres_per_pixel": centimetres_per_pixel,
        "rectified": rectified,
        "spatial_scaling": roi_transform.metadata(),
        "rectification": {
            "applied": rectified,
            "points_roi_coordinates": rectification_points,
            "source_to_rectified_homography": rectification_transform.tolist(),
        },
        "original_bounding_boxes": original_boxes,
    }
def _automatic_rectification_section(image: np.ndarray) -> dict:
    """Estimate and allow correction of full-image perspective coordinates."""
    st.header("Rectification")
    height, width = image.shape[:2]
    source_key = _roi_source_key(image)
    detected = detect_chamber_corners(image)
    state_prefix = f"full_rectification_{source_key}"
    keys = [(f"{state_prefix}_x_{index}", f"{state_prefix}_y_{index}") for index in range(4)]
    initialised_key = f"{state_prefix}_initialised"
    if not st.session_state.get(initialised_key):
        for point, (x_key, y_key) in zip(detected, keys, strict=True):
            st.session_state[x_key] = int(round(float(point[0])))
            st.session_state[y_key] = int(round(float(point[1])))
        st.session_state[initialised_key] = True
    if st.button("Reset rectification coordinates to automatic values", key=f"reset_{state_prefix}"):
        for point, (x_key, y_key) in zip(detected, keys, strict=True):
            st.session_state[x_key] = int(round(float(point[0])))
            st.session_state[y_key] = int(round(float(point[1])))

    points = []
    with st.expander("Rectification coordinates", expanded=False):
        st.caption(
            "Automatic corner values apply to the complete image processing copy. "
            "They remain editable when the estimated boundary needs correction."
        )
        for label, (x_key, y_key) in zip(
            ("Top-left", "Top-right", "Bottom-right", "Bottom-left"), keys, strict=True
        ):
            columns = st.columns(2)
            x_value = columns[0].number_input(
                f"{label} X", 0, width - 1, step=1, key=x_key
            )
            y_value = columns[1].number_input(
                f"{label} Y", 0, height - 1, step=1, key=y_key
            )
            points.append((float(x_value), float(y_value)))

    boundary = image.copy()
    integer_points = np.asarray(points, dtype=np.int32)
    for index, label in enumerate(("TL", "TR", "BR", "BL")):
        point = integer_points[index]
        following = integer_points[(index + 1) % 4]
        cv2.line(boundary, tuple(point), tuple(following), (0, 255, 255), 2)
        cv2.circle(boundary, tuple(point), 6, (0, 255, 255), -1)
        cv2.putText(
            boundary, label, (int(point[0]) + 8, max(16, int(point[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
        )
    try:
        rectified = rectify_image(image, points, output_size=(width, height))
    except ValueError as error:
        st.error(str(error))
        st.stop()
    columns = st.columns(2)
    columns[0].image(_bgr_to_rgb(boundary), caption="Automatic rectification boundary", width=420)
    columns[1].image(_bgr_to_rgb(rectified), caption="Rectified processing copy", width=420)
    return {"rectification_points": points, "source_key": source_key}


def _map_tile_intensity_to_image(
    processed: np.ndarray, tile, destination_sum: np.ndarray, destination_count: np.ndarray
) -> None:
    """Accumulate one processed intensity image in full-image coordinates."""
    item = tile.metadata
    valid_width = max(1, int(round(item.original_tile_width * item.scale_x)))
    valid_height = max(1, int(round(item.original_tile_height * item.scale_y)))
    valid = processed[:valid_height, :valid_width]
    mapped = cv2.resize(
        valid, (item.original_tile_width, item.original_tile_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    destination_sum[item.y_start:item.y_end, item.x_start:item.x_end] += mapped
    destination_count[item.y_start:item.y_end, item.x_start:item.x_end] += 1.0


def _process_tiled_pipeline_image(
    image: np.ndarray,
    config: dict,
    calibration_settings: dict | None = None,
) -> dict:
    """Analyse every source pixel through overlapping standardized tiles."""
    original = image
    original_height, original_width = original.shape[:2]
    settings = calibration_settings or {}
    rectification_points = settings.get("rectification_points")
    if rectification_points is not None:
        processing_copy, rectification_transform = rectify_image_with_transform(
            original, rectification_points, output_size=(original_width, original_height)
        )
        rectified = True
    else:
        processing_copy = original.copy()
        rectification_transform = np.eye(3, dtype=np.float32)
        rectified = False

    scaling_config = config["spatial_scaling"]
    target_size = _processing_size(config)
    tile_size = int(scaling_config["tile_size"])
    overlap_ratio = float(scaling_config["overlap_ratio"])
    tiles = generate_overlapping_tiles(
        processing_copy, tile_size, target_size, overlap_ratio
    )
    coverage = coverage_map(processing_copy.shape, tiles)
    if int(coverage.min()) < 1:
        raise RuntimeError("Automatic tiling left uncovered image pixels")

    profile_name = _detect_layout_profile(original)
    selected_profile = config["segmentation"]["roi_profiles"][profile_name]
    segmentation_settings = {
        **config["segmentation"],
        **{
            name: value for name, value in selected_profile.items()
            if name not in ("left", "right", "top", "bottom")
        },
    }
    segmentation_settings = scale_pixel_parameters(
        segmentation_settings,
        int(scaling_config.get("pixel_parameter_reference_size", 1920)),
        target_size,
    )
    tile_margins = {
        side: float(selected_profile.get(side, 0.0))
        for side in ("left", "right", "top", "bottom")
    }
    masks = []
    intermediate_masks: dict[str, list[np.ndarray]] = {}
    sums = {
        name: np.zeros(processing_copy.shape[:2], dtype=np.float32)
        for name in ("grey", "denoised", "enhanced", "local_contrast")
    }
    counts = np.zeros(processing_copy.shape[:2], dtype=np.float32)
    total_processing_ms = 0.0
    tile_previews = []
    for tile in tiles:
        scaled_tile, _, _ = spatial_scale_tile(tile.image, target_size)
        enhancement = enhance_image(
            scaled_tile,
            config["enhancement"],
            segmentation_settings,
        )
        segmentation = segment_tracks(
            enhancement.segmentation_input, segmentation_settings, tile_margins
        )
        masks.append(segmentation.binary_mask)
        total_processing_ms += segmentation.processing_time_ms
        for name, intermediate in segmentation.intermediate_images.items():
            intermediate_masks.setdefault(name, []).append(intermediate)
        local_count = np.zeros_like(counts)
        for name in sums:
            _map_tile_intensity_to_image(
                getattr(enhancement, name), tile, sums[name], local_count
            )
        counts += (local_count > 0).astype(np.float32)
        if len(tile_previews) < 4:
            tile_previews.append(scaled_tile)

    safe_counts = np.maximum(counts, 1.0)
    merged_enhancement = EnhancementResult(
        grey=np.clip(sums["grey"] / safe_counts, 0, 255).astype(np.uint8),
        denoised=np.clip(sums["denoised"] / safe_counts, 0, 255).astype(np.uint8),
        enhanced=np.clip(sums["enhanced"] / safe_counts, 0, 255).astype(np.uint8),
        local_contrast=np.clip(
            sums["local_contrast"] / safe_counts, 0, 255
        ).astype(np.uint8),
    )
    merged_mask = merge_tile_masks(
        processing_copy.shape,
        masks,
        tiles,
        minimum_overlap_agreement=float(
            scaling_config.get("minimum_overlap_agreement", 0.0)
        ),
    )
    merged_intermediate = {
        name: merge_tile_masks(processing_copy.shape, values, tiles)
        for name, values in intermediate_masks.items()
    }
    artifact_filter_stats = {
        "candidate_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
    }
    artifact_filter_enabled = bool(
        selected_profile.get(
            "artifact_filter_enabled",
            config["segmentation"].get("artifact_filter_enabled", False),
        )
    )
    if artifact_filter_enabled:
        model_path = Path(
            selected_profile.get(
                "artifact_filter_model",
                config["segmentation"].get(
                    "artifact_filter_model", "models/artifact_filter.joblib"
                ),
            )
        )
        if not model_path.is_absolute():
            # Configuration paths are project-relative, not relative to this
            # UI module (which is nested under ``cloud_chamber/ui``).
            model_path = Path(__file__).resolve().parents[3] / model_path
        merged_intermediate["candidates_before_artifact_filter"] = merged_mask.copy()
        merged_mask, artifact_filter_stats = filter_artifact_candidates(
            merged_mask,
            merged_enhancement.enhanced,
            load_artifact_filter(model_path),
        )
    contours, _ = cv2.findContours(
        merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    merged_segmentation = SegmentationResult(
        method_name="Overlapping tiles + " + segmentation.method_name,
        binary_mask=merged_mask,
        bounding_boxes=bounding_boxes,
        contours=contours,
        processing_time_ms=total_processing_ms,
        intermediate_images=merged_intermediate,
        parameters={
            **segmentation_settings,
            "tile_count": len(tiles),
            "tile_size": tile_size,
            "overlap_ratio": overlap_ratio,
            "artifact_filter_enabled": artifact_filter_enabled,
            **{
                f"artifact_filter_{name}": value
                for name, value in artifact_filter_stats.items()
            },
        },
    )
    feature_minimum_area = float(segmentation_settings["minimum_object_area"])
    if segmentation_settings.get("enable_thin_track_rule", True):
        feature_minimum_area = min(
            feature_minimum_area, float(segmentation_settings["minimum_thin_area"])
        )
    features = extract_track_features(
        merged_mask, merged_enhancement.enhanced, feature_minimum_area
    )
    # Geometry is reported in the standardized tile-processing scale used by
    # every classifier, while bounding boxes remain in full-image coordinates.
    feature_scale = target_size[0] / float(tile_size)
    features = [
        replace(
            item,
            area_pixels=item.area_pixels * feature_scale * feature_scale,
            perimeter_pixels=item.perimeter_pixels * feature_scale,
            major_axis_pixels=item.major_axis_pixels * feature_scale,
            mean_width_pixels=item.mean_width_pixels * feature_scale,
            thickness_pixels=item.thickness_pixels * feature_scale,
        )
        for item in features
    ]

    inverse_rectification = np.linalg.inv(rectification_transform)
    original_segmentation_mask = cv2.warpPerspective(
        merged_mask,
        inverse_rectification,
        (original_width, original_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    original_boxes = []
    for x, y, width, height in bounding_boxes:
        points = np.asarray(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        ).reshape(1, 4, 2)
        mapped = cv2.perspectiveTransform(points, inverse_rectification)[0]
        minimum, maximum = mapped.min(axis=0), mapped.max(axis=0)
        minimum_x = float(np.clip(minimum[0], 0, original_width - 1))
        minimum_y = float(np.clip(minimum[1], 0, original_height - 1))
        maximum_x = float(np.clip(maximum[0], 0, original_width - 1))
        maximum_y = float(np.clip(maximum[1], 0, original_height - 1))
        original_boxes.append({
            "x": minimum_x,
            "y": minimum_y,
            "width": max(0.0, maximum_x - minimum_x),
            "height": max(0.0, maximum_y - minimum_y),
        })
    tile_metadata = [tile.metadata.to_dict() for tile in tiles]
    scale_factor = target_size[0] / float(tile_size)
    return {
        "original_image": original,
        "input_image": processing_copy,
        "selected_roi_image": processing_copy,
        "rectified_roi_image": processing_copy,
        "spatially_scaled_image": tile_previews[0] if tile_previews else processing_copy,
        "enhancement": merged_enhancement,
        "segmentation": merged_segmentation,
        "features": features,
        "roi_profile": profile_name,
        "centimetres_per_pixel": None,
        "rectified": rectified,
        "spatial_scaling": {
            "method": "automatic_overlapping_tiling",
            "original_image_width": original_width,
            "original_image_height": original_height,
            "tile_count": len(tiles),
            "tile_size": tile_size,
            "overlap_ratio": overlap_ratio,
            "processing_width": target_size[0],
            "processing_height": target_size[1],
            "processing_resolution": f"{target_size[0]} × {target_size[1]}",
            "scale_factor": scale_factor,
            "scale_x": scale_factor,
            "scale_y": scale_factor,
            "scaling_operation": "none" if scale_factor == 1 else ("upscale" if scale_factor > 1 else "downscale"),
            "minimum_coverage": int(coverage.min()),
            "maximum_coverage": int(coverage.max()),
        },
        "rectification": {
            "applied": rectified,
            "points_original_coordinates": rectification_points,
            "source_to_rectified_homography": rectification_transform.tolist(),
        },
        "tile_metadata": tile_metadata,
        "coverage_map": coverage,
        "tile_boundary_preview": draw_tile_boundaries(processing_copy, tiles),
        "original_bounding_boxes": original_boxes,
        "original_segmentation_mask": original_segmentation_mask,
    }


def _acquisition_section(config: dict) -> None:
    """Acquire a batch of images or sampled video frames.

    The processing stages still analyse one image at a time. Keeping all
    acquired samples in a session batch lets the user move between them
    without uploading the files again.
    """
    source_type = st.radio("Input type", ["Image", "Video"], horizontal=True)
    st.session_state["input_source_type"] = source_type
    if source_type == "Image":
        uploads = st.file_uploader(
            "Upload one or more raw cloud-chamber images",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            accept_multiple_files=True,
        )
        if uploads:
            samples = []
            failures = []
            upload_signature = []
            for upload in uploads:
                try:
                    payload = upload.getvalue()
                    upload_signature.append(
                        (upload.name, len(payload), hashlib.sha1(payload).hexdigest())
                    )
                    image = _decode_uploaded_image(payload)
                    samples.append(
                        {
                            "image": image,
                            "name": upload.name,
                            "description": f"Uploaded image: {upload.name}",
                        }
                    )
                except ValueError:
                    failures.append(upload.name)

            if samples:
                signature = tuple(upload_signature)
                if signature != st.session_state.get("image_upload_signature"):
                    _replace_input_batch(samples)
                    st.session_state["image_upload_signature"] = signature
                    st.session_state["input_images"] = [
                        (sample["image"], sample["name"]) for sample in samples
                    ]
                    st.session_state["selected_input_index"] = 0
                    st.success(
                        f"Loaded {len(samples)} image(s). Choose one from the selector below."
                    )
            if failures:
                st.warning("Unreadable files skipped: " + ", ".join(failures))

        if st.session_state.get("input_batch"):
            names = [sample["name"] for sample in st.session_state["input_batch"]]
            selected = st.selectbox("Select image for analysis", names)
            index = names.index(selected)
            st.session_state["selected_input_index"] = index
            sample = st.session_state["input_batch"][index]
            _set_input(sample["image"], sample["name"], sample["description"])
        return

    upload = st.file_uploader(
        "Upload a cloud-chamber video", type=["mp4", "avi", "mov"]
    )
    if upload is not None:
        _video_acquisition(upload, config)

    _batch_selector()


def _video_acquisition(upload, config: dict) -> None:
    """Preview a video and acquire either one frame or a sampled frame batch."""
    with tempfile.NamedTemporaryFile(
        suffix=Path(upload.name).suffix, delete=False
    ) as temporary:
        temporary.write(upload.getvalue())
        video_path = Path(temporary.name)

    capture = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            st.error("The uploaded video could not be opened.")
            return
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps if fps > 0 else 0.0
        st.caption(
            f"{frame_count:,} frames | {fps:.2f} fps | {duration:.2f} seconds"
        )

        st.markdown("**Video track enhancement**")
        temporal_enabled = st.checkbox(
            "Combine neighbouring frames to strengthen faint tracks",
            value=True,
            help=(
                "Uses a temporal median to suppress stationary chamber texture "
                "and boosts bright changes across a short frame window."
            ),
        )
        temporal_controls = st.columns(3)
        temporal_window = temporal_controls[0].slider(
            "Temporal window", 3, 15, 7, step=2,
            disabled=not temporal_enabled,
            help="More frames reveal longer-lived tracks but may combine separate events.",
        )
        temporal_gain = temporal_controls[1].slider(
            "Track evidence gain", 0.5, 4.0, 2.0, step=0.25,
            disabled=not temporal_enabled,
        )
        stabilise_video = temporal_controls[2].checkbox(
            "Stabilise camera movement", value=False,
            disabled=not temporal_enabled,
            help="Enable for handheld video; leave disabled for a fixed camera.",
        )

        frame_number = st.slider("Preview frame", 0, frame_count - 1, 0)
        window_frames, reference_index = _read_video_window(
            capture,
            frame_number,
            int(temporal_window) if temporal_enabled else 1,
            frame_count,
        )
        success = bool(window_frames)
        if success:
            frame = window_frames[reference_index]
            processed_frame = frame
            evidence = None
            if temporal_enabled and len(window_frames) > 1:
                processed_frame, evidence = temporal_track_composite(
                    window_frames,
                    reference_index=reference_index,
                    evidence_gain=float(temporal_gain),
                    stabilise=bool(stabilise_video),
                )
            timestamp = frame_number / fps if fps > 0 else 0.0
            preview_columns = st.columns(3 if evidence is not None else 1)
            preview_columns[0].image(
                _bgr_to_rgb(frame), caption=f"Original frame {frame_number}", width=420
            )
            if evidence is not None:
                preview_columns[1].image(
                    evidence, caption="Transient track evidence", width=420
                )
                preview_columns[2].image(
                    _bgr_to_rgb(processed_frame),
                    caption="Frame sent to segmentation",
                    width=420,
                )
            st.caption(f"Preview time: {timestamp:.2f} seconds")
            if st.button("Load preview frame only"):
                sample = _video_sample(
                    upload.name,
                    processed_frame,
                    frame_number,
                    fps,
                    temporal_window=len(window_frames) if temporal_enabled else 1,
                )
                _replace_input_batch([sample])

        st.subheader("Video frame batch")
        controls = st.columns(3)
        start_frame = controls[0].number_input(
            "Start frame", 0, frame_count - 1, 0, step=1
        )
        frame_interval = controls[1].number_input(
            "Sample every N frames",
            min_value=1,
            max_value=frame_count,
            value=min(int(config["acquisition"]["frame_interval"]), frame_count),
            step=1,
            help="A larger interval reduces near-duplicate neighbouring frames.",
        )
        maximum_frames = controls[2].number_input(
            "Maximum frames", 1, 200, min(30, frame_count), step=1
        )
        if st.button("Extract video frame batch", type="primary"):
            samples = []
            for index in range(
                int(start_frame), frame_count, int(frame_interval)
            ):
                if len(samples) >= int(maximum_frames):
                    break
                window_frames, reference_index = _read_video_window(
                    capture,
                    index,
                    int(temporal_window) if temporal_enabled else 1,
                    frame_count,
                )
                if window_frames:
                    sampled_frame = window_frames[reference_index]
                    if temporal_enabled and len(window_frames) > 1:
                        sampled_frame, _ = temporal_track_composite(
                            window_frames,
                            reference_index=reference_index,
                            evidence_gain=float(temporal_gain),
                            stabilise=bool(stabilise_video),
                        )
                    samples.append(
                        _video_sample(
                            upload.name,
                            sampled_frame,
                            index,
                            fps,
                            temporal_window=(
                                len(window_frames) if temporal_enabled else 1
                            ),
                        )
                    )
            _replace_input_batch(samples)
            st.success(f"Extracted {len(samples)} frame(s) from the video.")
    finally:
        if capture is not None:
            capture.release()
            # OpenCV's FFmpeg backend can retain its Windows file handle until
            # the Python wrapper is collected, even after release().
            del capture
            gc.collect()
        for attempt in range(5):
            try:
                video_path.unlink(missing_ok=True)
                break
            except PermissionError:
                # Windows may release the decoder handle a few milliseconds
                # after VideoCapture.release(). Cleanup must never crash the UI.
                if attempt < 4:
                    sleep(0.05)


def _video_sample(
    video_name: str,
    frame: np.ndarray,
    frame_number: int,
    fps: float,
    temporal_window: int = 1,
) -> dict:
    """Create traceable metadata for one acquired video frame."""
    timestamp = frame_number / fps if fps > 0 else 0.0
    return {
        "image": frame.copy(),
        "name": f"{Path(video_name).stem}_frame_{frame_number:06d}.jpg",
        "description": (
            f"{video_name}, frame {frame_number}, {timestamp:.2f} seconds; "
            + (
                f"temporal track enhancement using {temporal_window} frames"
                if temporal_window > 1
                else "single-frame processing"
            )
        ),
    }


def _read_video_window(
    capture: cv2.VideoCapture,
    centre_frame: int,
    window_size: int,
    frame_count: int,
) -> tuple[list[np.ndarray], int]:
    """Read a centred, boundary-safe temporal window from a video."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("Video temporal window must be a positive odd number")
    available = min(window_size, frame_count)
    half = available // 2
    start = max(0, min(centre_frame - half, frame_count - available))
    indices = list(range(start, start + available))
    frames: list[np.ndarray] = []
    captured_indices: list[int] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = capture.read()
        if success:
            frames.append(frame)
            captured_indices.append(index)
    if not frames:
        return [], 0
    reference_index = min(
        range(len(captured_indices)),
        key=lambda position: abs(captured_indices[position] - centre_frame),
    )
    return frames, reference_index


def _replace_input_batch(samples: list[dict]) -> None:
    """Replace the acquisition queue and activate its first valid sample."""
    st.session_state["input_batch"] = samples
    st.session_state["shared_segmentation_results"] = {}
    st.session_state["mlp_batch_results"] = {}
    st.session_state["svm_batch_results"] = {}
    st.session_state["decision_tree_batch_results"] = {}
    st.session_state["batch_reports"] = {}
    st.session_state["extra_trees_batch_results"] = {}
    st.session_state["extra_trees_batch_reports"] = {}
    st.session_state["pipeline_results_by_input"] = {}
    if samples:
        first = samples[0]
        _set_input(first["image"], first["name"], first["description"])


def _render_batch_segmentation_summary(config: dict, current_result: dict) -> None:
    """Automatically review one segmentation or compare an acquired batch."""
    samples = st.session_state.get("input_batch", [])
    if not samples:
        samples = [{
            "image": st.session_state["input_image"],
            "name": st.session_state.get("input_name") or "Current input",
            "description": st.session_state.get("source_description") or "",
        }]

    is_batch = len(samples) > 1
    st.subheader(
        "Batch / video segmented-particle comparison"
        if is_batch else "Segmented-particle review"
    )

    if is_batch:
        segmented_results = st.session_state.get(
            "shared_segmentation_results", {}
        )
        expected_names = {sample["name"] for sample in samples}
        cache_is_complete = set(segmented_results) == expected_names
    else:
        segmented_results = {samples[0]["name"]: current_result}
        cache_is_complete = True

    if not cache_is_complete:
        segmented_results = {}
        progress = st.progress(0.0, text="Automatically segmenting batch...")
        current_name = st.session_state.get("input_name")
        for index, sample in enumerate(samples):
            sample_result = (
                current_result
                if sample["name"] == current_name
                else _process_tiled_pipeline_image(
                    sample["image"],
                    config,
                    st.session_state.get("calibration_settings"),
                )
            )
            segmented_results[sample["name"]] = sample_result
            progress.progress(
                (index + 1) / len(samples),
                text=f"Segmented {index + 1} of {len(samples)} inputs",
            )
        progress.empty()
        st.session_state["shared_segmentation_results"] = segmented_results

    rows = [
        {
            "Input / frame": name,
            "Detected segmented particles": len(item["features"]),
            "Segmentation time (ms)": round(
                float(item["segmentation"].processing_time_ms), 2
            ),
        }
        for name, item in segmented_results.items()
    ]
    total = sum(row["Detected segmented particles"] for row in rows)
    metrics = st.columns(3)
    metrics[0].metric("Processed inputs / frames", len(rows))
    metrics[1].metric("Detected particles", total)
    metrics[2].metric(
        "Frames with detections",
        sum(row["Detected segmented particles"] > 0 for row in rows),
    )
    if is_batch:
        st.caption(
            "The chart compares how many segmented particles were detected "
            "in each uploaded image or sampled video frame."
        )
        st.bar_chart(
            rows,
            x="Input / frame",
            y="Detected segmented particles",
            color="#FFD700",
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    selected_name = (
        st.selectbox(
            "Inspect segmented particles for an input / frame",
            list(segmented_results),
            key="shared_segmented_frame_preview",
        )
        if is_batch
        else next(iter(segmented_results))
    )
    selected_result = segmented_results[selected_name]
    sample_by_name = {sample["name"]: sample for sample in samples}
    preview = _colour_instance_mask(
        sample_by_name[selected_name]["image"],
        selected_result["original_segmentation_mask"],
        opacity=0.52,
    )
    preview = _draw_track_ids(
        preview,
        selected_result["features"],
        selected_result.get("original_bounding_boxes"),
    )
    st.image(
        _bgr_to_rgb(preview),
        caption=(
            f"{selected_name}: {len(selected_result['features'])} segmented "
            "particles detected during preprocessing"
        ),
    )


def _batch_selector() -> None:
    """Select the sample sent through the shared one-image pipeline."""
    samples = st.session_state.get("input_batch", [])
    if not samples:
        st.caption("No acquisition batch has been loaded yet.")
        return

    st.subheader("Acquisition batch")
    names = [sample["name"] for sample in samples]
    current_name = st.session_state.get("input_name")
    default_index = names.index(current_name) if current_name in names else 0
    selected_index = st.selectbox(
        f"Select an input to analyse ({len(samples)} available)",
        range(len(samples)),
        index=default_index,
        format_func=lambda index: f"Image {index + 1}: {names[index]}",
    )
    selected = samples[selected_index]
    if selected["name"] != current_name:
        _set_input(selected["image"], selected["name"], selected["description"])
    st.caption(selected["description"])

    if st.checkbox(f"Show batch gallery ({len(samples)} image entries)", value=False):
        previews_per_page = 12
        page_count = max(1, (len(samples) + previews_per_page - 1) // previews_per_page)
        page = st.number_input(
            "Preview page", min_value=1, max_value=page_count, value=1, step=1
        )
        start = (int(page) - 1) * previews_per_page
        stop = min(start + previews_per_page, len(samples))
        st.caption(f"Showing entries {start + 1}–{stop} of {len(samples)}.")
        for index in range(start, stop):
            sample = samples[index]
            active = " — active input" if index == selected_index else ""
            with st.expander(
                f"Image {index + 1}: {sample['name']}{active}", expanded=False
            ):
                st.image(
                    _bgr_to_rgb(sample["image"]),
                    caption=sample["description"],
                    width=420,
                )

def _draw_track_ids(
    image: np.ndarray,
    features: list,
    bounding_boxes: list | tuple | None = None,
) -> np.ndarray:
    """Draw readable track IDs that map the segmentation to table rows."""
    output = image.copy()
    boxes = (
        [item.bounding_box for item in features]
        if bounding_boxes is None
        else bounding_boxes
    )
    if len(boxes) != len(features):
        boxes = [item.bounding_box for item in features]
    for item, box in zip(features, boxes, strict=True):
        if isinstance(box, dict):
            x = int(round(float(box["x"])))
            y = int(round(float(box["y"])))
        else:
            x, y, _, _ = (int(round(float(value))) for value in box)
        label = f"T{item.track_id}"
        origin = (max(2, x), max(18, y - 5))
        cv2.putText(
            output, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            output, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (0, 255, 255), 2, cv2.LINE_AA,
        )
    return output


def _pipeline_result_signature(result: dict) -> str:
    """Identify the segmentation and feature records used by classifiers."""
    digest = hashlib.sha1()
    mask = np.ascontiguousarray(result["original_segmentation_mask"])
    digest.update(str(mask.shape).encode("ascii"))
    digest.update(mask.dtype.str.encode("ascii"))
    digest.update(mask.tobytes())
    for feature in result.get("features", []):
        digest.update(repr(feature).encode("utf-8"))
    return digest.hexdigest()


def _set_input(image: np.ndarray, name: str, description: str) -> None:
    """Select one input and invalidate cached results from the previous image."""
    st.session_state["input_image"] = image
    st.session_state["input_name"] = name
    st.session_state["source_description"] = description
    st.session_state["pipeline_result"] = None
    st.session_state["pipeline_result_signature"] = None
    st.session_state["mlp_predictions"] = None
    st.session_state["mlp_quality"] = None
    st.session_state["extra_trees_quality"] = None
    st.session_state["svm_predictions"] = None
    st.session_state["svm_quality"] = None
    st.session_state["decision_tree_predictions"] = None
    st.session_state["decision_tree_quality"] = None
    st.session_state["extra_trees_predictions"] = None
    st.session_state["extra_trees_prediction_signature"] = None





def _detect_layout_profile(image: np.ndarray) -> str:
    """Infer the known acquisition layout from its stable aspect ratio.

    Primary frames are portrait (1080x1920), including the full chamber and
    walls. Muller frames are landscape (1312x992) and already cropped.
    """
    height, width = image.shape[:2]
    return (
        "primary_full_chamber"
        if height > width
        else "external_muller"
    )


def _input_status() -> None:
    """Show the active input in the sidebar when one has been selected."""
    if st.session_state.get("input_image") is None:
        st.sidebar.warning("No image selected")
    else:
        st.sidebar.success(f"Input: {st.session_state['input_name']}")


def _decode_uploaded_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes as a BGR image and reject unreadable content."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded data is not a readable image")
    return image


def _colour_instance_mask(
    image: np.ndarray,
    binary_mask: np.ndarray,
    opacity: float = 0.45,
) -> np.ndarray:
    """Colour every connected accepted track while preserving image detail."""
    if image.shape[:2] != binary_mask.shape:
        raise ValueError("Instance mask must match the displayed image size")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Overlay opacity must be between zero and one")
    component_count, labels = cv2.connectedComponents(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )
    segmentation_colour = (0, 210, 255)
    output = image.copy()
    colour_layer = image.copy()
    for component_id in range(1, component_count):
        region = labels == component_id
        colour_layer[region] = segmentation_colour
    foreground = labels > 0
    blended = cv2.addWeighted(image, 1.0 - opacity, colour_layer, opacity, 0)
    output[foreground] = blended[foreground]
    contours, _ = cv2.findContours(
        (foreground.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    return output


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV colour order to the RGB order expected by Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Public adapters used by model pages and the non-interactive pipeline.
# The underlying private helpers retain their original names to minimise risk in
# this structure-only refactor.
process_pipeline_image = _process_tiled_pipeline_image
bgr_to_rgb = _bgr_to_rgb
input_status = _input_status
