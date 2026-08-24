"""Streamlit GUI for the corrected Mode A cloud-chamber pipeline."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import replace
from time import perf_counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from cloud_chamber.config import load_config
from cloud_chamber.calibration import (
    RoiScalingResult,
    default_roi_coordinates,
    detect_chamber_corners,
    extract_roi,
    rectify_image,
    rectify_image_with_transform,
    select_and_scale_roi,
    spatial_scale_roi,
)
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import extract_track_features
from cloud_chamber.ml.member_models.extra_trees import (
    EXTRA_TREES_CLASS_COLOURS,
    build_visual_report as build_extra_trees_visual_report,
    encode_report_csv as encode_extra_trees_report_csv,
    encode_report_png as encode_extra_trees_report_png,
    load_model as load_extra_trees_model,
    predict_tracks as predict_extra_trees_tracks,
    summarise_predictions as summarise_extra_trees_predictions
)
from cloud_chamber.models import EnhancementResult, SegmentationResult
from cloud_chamber.segmentation import scale_pixel_parameters, segment_tracks
from cloud_chamber.tiling import (
    coverage_map,
    draw_tile_boundaries,
    generate_overlapping_tiles,
    map_processed_mask_to_image,
    merge_tile_masks,
    spatial_scale_tile,
)
from cloud_chamber.ui.navigation import (
    COMPARISON_PAGE,
    MODEL_PAGES,
    PAGES,
    SHARED_PIPELINE_PAGE,
    render_selected_page,
)
from cloud_chamber.ui.model_pages import (
    decision_tree_page,
    extra_trees_page,
    mlp_page,
    svm_page,
)
from cloud_chamber.ui.model_pages.context import PageContext
from cloud_chamber.ui.comparison_page import render as render_comparison_page
from ui.cnn_page import render_cnn_page

ROI_PROFILE_LABELS = {
    "Auto-detect from image shape (recommended)": "auto",
    "External Muller / already cropped": "external_muller",
    "Primary dataset / full chamber": "primary_full_chamber",
}
PAGES = ["Shared Processing Pipeline", *MODEL_PAGES, "Final Model Comparison"]
ROI_PROFILE_LABELS = ("Auto-detect from image shape (recommended)",)


def main() -> None:
    st.set_page_config(
        page_title="Cloud Chamber Particle Classification",
        page_icon="â˜ï¸",
        layout="wide",
    )
    config = load_config()
    _initialise_state()

    st.sidebar.title("Cloud Chamber")
    page = st.sidebar.radio("Navigate", PAGES)
    _input_status()

    render_selected_page(page, config, _build_page_handlers())


def _build_page_handlers():
    """Connect navigation labels to the existing page-rendering functions."""
    context = PageContext(
        bgr_to_rgb=_bgr_to_rgb,
        process_pipeline_image=_process_tiled_pipeline_image,
        feature_row=_feature_row,
    )
    return {
        SHARED_PIPELINE_PAGE: _shared_pipeline_page,
        "CNN": lambda config: render_cnn_page(config, context),
        "SVM": lambda _config: svm_page.render(context),
        "Decision Tree": lambda _config: decision_tree_page.render(context),
        "MLP": lambda config: mlp_page.render(
            "MLP", MODEL_PAGES["MLP"], config, context
        ),
        "Extra Trees": lambda _config: extra_trees_page.render(context),
        COMPARISON_PAGE: lambda _config: render_comparison_page(),
    }


def _initialise_state() -> None:
    st.session_state.setdefault("input_batch", [])
    st.session_state.setdefault("input_image", None)
    st.session_state.setdefault("input_name", None)
    st.session_state.setdefault("input_images", [])
    st.session_state.setdefault("selected_input_index", 0)
    st.session_state.setdefault("source_description", None)
    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("mlp_predictions", None)
    st.session_state.setdefault("cnn_predictions", None)
    st.session_state.setdefault("svm_predictions", None)
    st.session_state.setdefault("extra_trees_predictions", None)
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


def _shared_pipeline_page(config: dict) -> None:
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
    st.header("Grayscale conversion and Gaussian filtering")
    columns = st.columns(3)
    columns[0].image(_bgr_to_rgb(analysis_image), caption="Full processing image")
    columns[1].image(enhancement.grey, caption="Grayscale")
    columns[2].image(enhancement.denoised, caption="Gaussian filtered")

    st.header("Thresholding, morphology and contour detection")
    intermediate = segmentation.intermediate_images
    columns = st.columns(4)
    columns[0].image(
        intermediate["local_bright_tracks"],
        caption="Morphological Operation: White top-hat",
    )
    columns[1].image(intermediate["threshold"], caption="Otsu thresholding")
    columns[2].image(
        intermediate["morphological_closing"],
        caption="Morphological operation: Closing",
    )
    columns[3].image(
        segmentation.binary_mask,
        caption="Contour detection and filtering",
    )

    st.session_state["pipeline_result"] = result
    # A changed image or segmentation produces different track IDs/features.
    # Never display predictions cached for the previous pipeline result.
    st.session_state["mlp_predictions"] = None
    st.session_state["cnn_predictions"] = None
    st.session_state["mlp_quality"] = None
    st.session_state["decision_tree_predictions"] = None
    st.session_state["decision_tree_quality"] = None
    st.session_state["svm_predictions"] = None
    st.session_state["svm_quality"] = None
    st.session_state["extra_trees_predictions"] = None
    # Batch classifier pages cache their own pipeline outputs. Clear them as
    # well, otherwise they can display contours from an older segmentation run.
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
    st.image(
        _bgr_to_rgb(overlay),
        caption=(
            "Final instance segmentation: each continuous coloured region is "
            "one detected particle track"
        ),
    )
    st.caption(
        f"Segmented particle instances: {len(features)}. The classification "
        "page must report the same count for this pipeline run."
    )

    st.header("Contour-based feature extraction")
    rows = [
        _feature_row(item, result["centimetres_per_pixel"])
        for item in features
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
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
    st.header("Perspective rectification")
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
    enhancement = enhance_image(analysis_image, config["enhancement"])
    segmentation = segment_tracks(
        enhancement.enhanced,
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
    st.header("Perspective rectification")
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
    sums = {name: np.zeros(processing_copy.shape[:2], dtype=np.float32) for name in ("grey", "denoised", "enhanced")}
    counts = np.zeros(processing_copy.shape[:2], dtype=np.float32)
    total_processing_ms = 0.0
    tile_previews = []
    for tile in tiles:
        scaled_tile, _, _ = spatial_scale_tile(tile.image, target_size)
        enhancement = enhance_image(scaled_tile, config["enhancement"])
        segmentation = segment_tracks(
            enhancement.enhanced, segmentation_settings, tile_margins
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
    )
    merged_mask = merge_tile_masks(processing_copy.shape, masks, tiles)
    merged_intermediate = {
        name: merge_tile_masks(processing_copy.shape, values, tiles)
        for name, values in intermediate_masks.items()
    }
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
    if source_type == "Image":
        uploads = st.file_uploader(
            "Upload one or more raw cloud-chamber images",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            accept_multiple_files=True,
        )
        if uploads:
            samples = []
            failures = []
            for upload in uploads:
                try:
                    image = _decode_uploaded_image(upload.getvalue())
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
                _replace_input_batch(samples)
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

        frame_number = st.slider("Preview frame", 0, frame_count - 1, 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if success:
            timestamp = frame_number / fps if fps > 0 else 0.0
            st.image(
                _bgr_to_rgb(frame),
                caption=f"Frame {frame_number} ({timestamp:.2f} seconds)",
                width=480,
            )
            if st.button("Load preview frame only"):
                sample = _video_sample(upload.name, frame, frame_number, fps)
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
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                captured, sampled_frame = capture.read()
                if captured:
                    samples.append(
                        _video_sample(upload.name, sampled_frame, index, fps)
                    )
            _replace_input_batch(samples)
            st.success(f"Extracted {len(samples)} frame(s) from the video.")
        capture.release()
    finally:
        video_path.unlink(missing_ok=True)


def _video_sample(
    video_name: str, frame: np.ndarray, frame_number: int, fps: float
) -> dict:
    """Create traceable metadata for one acquired video frame."""
    timestamp = frame_number / fps if fps > 0 else 0.0
    return {
        "image": frame.copy(),
        "name": f"{Path(video_name).stem}_frame_{frame_number:06d}.jpg",
        "description": (
            f"{video_name}, frame {frame_number}, {timestamp:.2f} seconds"
        ),
    }


def _model_page(short_name: str, full_name: str) -> None:
    st.title(f"{short_name} Classifier")
    st.info(
        f"Purpose: team-member workspace for the {full_name}. This model must "
        "use the shared dataset splits and shared processing pipeline."
    )
    if short_name == "CNN":
        render_cnn_page()
        return

    module_name = short_name.lower().replace(" ", "_")
    if short_name == "Extra Trees":
        module_name = "extra_trees"
    st.code(f"cloud_chamber/ml/member_models/{module_name}.py")
    st.markdown(
        """
        The member implementation should provide training, validation,
        prediction and model-saving functions. Do not duplicate or alter the
        shared enhancement, segmentation or feature extraction stages.

        This page is intentionally a placeholder until the assigned member
        connects their completed classifier.
        """
    )
    if short_name != "MLP":
        return

    st.subheader("Run the trained MLP")
    result = st.session_state.get("pipeline_result")
    if result is None:
        st.warning(
            "Process an image or video frame on the Shared Processing "
            "Pipeline page first."
        )
        return
    model_path = Path("models/mlp_classifier.joblib")
    if not model_path.exists():
        st.warning("Train the model first: `python scripts/train_mlp.py`")
        return
    confidence_threshold = st.slider(
        "Reporting confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help=(
            "Predictions below this probability remain visible but are marked "
            "Uncertain. This threshold does not retrain the model."
        ),
    )
    st.caption("This page remains a stub until the member model is connected.")


def _replace_input_batch(samples: list[dict]) -> None:
    """Replace the acquisition queue and activate its first valid sample."""
    st.session_state["input_batch"] = samples
    st.session_state["mlp_batch_results"] = {}
    st.session_state["svm_batch_results"] = {}
    st.session_state["decision_tree_batch_results"] = {}
    st.session_state["batch_reports"] = {}
    st.session_state["extra_trees_batch_results"] = {}
    st.session_state["extra_trees_batch_reports"] = {}
    if samples:
        first = samples[0]
        _set_input(first["image"], first["name"], first["description"])


def _batch_selector() -> None:
    """Select the sample sent through the existing one-image pipeline."""
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
        _set_input(
            selected["image"], selected["name"], selected["description"]
        )
    st.caption(selected["description"])

    # Expanders keep a large upload readable: users first see Image 1,
    # Image 2, etc., and reveal only the previews they want to inspect.
    show_gallery = st.checkbox(
        f"Show batch gallery ({len(samples)} image entries)", value=False
    )
    if show_gallery:
        previews_per_page = 12
        page_count = max(
            1, (len(samples) + previews_per_page - 1) // previews_per_page
        )
        page = st.number_input(
            "Preview page",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            help=(
                "Only 12 entries are placed on a page so a long video batch "
                "does not overload the browser."
            ),
        )
        start = (int(page) - 1) * previews_per_page
        stop = min(start + previews_per_page, len(samples))
        st.caption(f"Showing entries {start + 1}–{stop} of {len(samples)}.")
        for index in range(start, stop):
            sample = samples[index]
            active_text = " — active input" if index == selected_index else ""
            with st.expander(
                f"Image {index + 1}: {sample['name']}{active_text}",
                expanded=False,
            ):
                st.image(
                    _bgr_to_rgb(sample["image"]),
                    caption=sample["description"],
                    width=420,
                )












def _feature_row(item, centimetres_per_pixel: float | None = None) -> dict:
    row = {
        "Track": item.track_id,
        "Area (pxÂ²)": round(item.area_pixels, 3),
        "Perimeter (px)": round(item.perimeter_pixels, 3),
        "Length (px)": round(item.major_axis_pixels, 3),
        "Width (px)": round(item.mean_width_pixels, 3),
        "Aspect ratio": round(item.aspect_ratio, 3),
        "Solidity": round(item.solidity, 3),
        "Rectangularity": round(item.rectangularity, 3),
        "Thickness (px)": round(item.thickness_pixels, 3),
        "Orientation (Â°)": round(item.orientation_degrees, 3),
        "Mean intensity": round(item.mean_intensity, 3),
    }
    if centimetres_per_pixel is not None:
        scale = float(centimetres_per_pixel)
        row.update(
            {
                "Area (cm²)": round(item.area_pixels * scale * scale, 6),
                "Perimeter (cm)": round(item.perimeter_pixels * scale, 6),
                "Length (cm)": round(item.major_axis_pixels * scale, 6),
                "Width (cm)": round(item.mean_width_pixels * scale, 6),
                "Thickness (cm)": round(item.thickness_pixels * scale, 6),
            }
        )
        return
    model_path = Path("models/mlp_classifier.joblib")
    if not model_path.exists():
        st.warning("Train the model first: `python scripts/train_mlp.py`")
        return
    model_bundle = load_model(model_path)
    confidence_threshold = st.slider(
        "Confidence reporting threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        key="mlp_confidence_threshold",
        help=(
            "Predictions below this value are reported as uncertain. The "
            "predicted class and trained model do not change."
        ),
    )
    samples = st.session_state.get("input_batch") or [
        {
            "image": st.session_state["input_image"],
            "name": st.session_state["input_name"],
            "description": st.session_state["source_description"],
        }
    ]
    button_label = (
        f"Classify all {len(samples)} inputs and create reports"
        if len(samples) > 1
        else "Classify and create MLP report"
    )
    if st.button(button_label, type="primary"):
        batch_results = {}
        progress = st.progress(0.0, text="Processing batch...")
        for index, sample in enumerate(samples):
            sample_result = _process_pipeline_image(sample["image"], config)
            sample_predictions = predict_tracks(
                model_bundle, sample_result["features"]
            )
            sample_quality = assess_all_contours(
                sample_result["features"],
                sample_result["enhancement"].enhanced,
                sample_result["segmentation"].binary_mask,
                sample_result["segmentation"].parameters,
            )
            batch_results[sample["name"]] = {
                "image": sample["image"],
                "description": sample["description"],
                "result": sample_result,
                "predictions": sample_predictions,
                "quality": sample_quality,
            }
            progress.progress(
                (index + 1) / len(samples),
                text=f"Processed {index + 1} of {len(samples)} inputs",
            )
        progress.empty()
        st.session_state["mlp_batch_results"] = batch_results

    batch_results = st.session_state.get("mlp_batch_results", {})
    if batch_results:
        st.subheader("Batch classification results")
        for index, (name, entry) in enumerate(batch_results.items(), start=1):
            item_predictions = entry["predictions"]
            item_quality = entry["quality"]
            item_result = entry["result"]
            item_summary = build_summary(
                predictions=item_predictions,
                quality_assessments=item_quality,
                confidence_threshold=confidence_threshold,
                processing_time_ms=float(
                    item_result["segmentation"].processing_time_ms
                )
                + sum(
                    float(item["inference_time_ms"])
                    for item in item_predictions
                ),
            )
            st.session_state["batch_reports"][name] = {
                "Input": name,
                "Source": entry["description"],
                "Tracks": item_summary["detected_contours"],
                "Dominant prediction": item_summary["dominant_prediction"],
                "Confident": item_summary["confident_classifications"],
                "Uncertain": item_summary["uncertain_classifications"],
                "Alpha": item_summary["class_counts"]["Alpha"],
                "Electron/Positron": item_summary["class_counts"][
                    "Electron/Positron"
                ],
                "Proton": item_summary["class_counts"]["Proton"],
                "V-track": item_summary["class_counts"]["V-track"],
                "Contour quality": item_summary["overall_contour_quality"],
                "Mean quality score": item_summary["mean_contour_quality"],
                "Processing time (ms)": item_summary["processing_time_ms"],
            }
            with st.expander(
                f"Image {index}: {name} — "
                f"{item_summary['detected_contours']} detected — "
                f"dominant: {item_summary['dominant_prediction']}",
                expanded=False,
            ):
                if item_predictions:
                    item_overlay, _ = build_visual_report(
                        image=entry["image"],
                        features=item_result["features"],
                        predictions=item_predictions,
                        confidence_threshold=confidence_threshold,
                        quality_assessments=item_quality,
                    )
                    st.image(_bgr_to_rgb(item_overlay), width=700)
                else:
                    st.warning("No segmented track was available to classify.")
        selected_name = st.selectbox(
            "Choose an input for the detailed report",
            list(batch_results),
        )
        selected_entry = batch_results[selected_name]
        st.session_state["input_image"] = selected_entry["image"]
        st.session_state["input_name"] = selected_name
        st.session_state["source_description"] = selected_entry["description"]
        st.session_state["pipeline_result"] = selected_entry["result"]
        st.session_state["mlp_predictions"] = selected_entry["predictions"]
        st.session_state["mlp_quality"] = selected_entry["quality"]
        result = selected_entry["result"]

    predictions = st.session_state.get("mlp_predictions")
    if predictions is None:
        return
    if not predictions:
        st.warning("No segmented track is available for classification.")
        return
    quality_assessments = st.session_state.get("mlp_quality")
    if quality_assessments is None:
        quality_assessments = assess_all_contours(
            result["features"],
            result["enhancement"].enhanced,
            result["segmentation"].binary_mask,
            result["segmentation"].parameters,
        )
        st.session_state["mlp_quality"] = quality_assessments

    overlay, report_rows = build_visual_report(
        image=st.session_state["input_image"],
        features=result["features"],
        predictions=predictions,
        confidence_threshold=confidence_threshold,
        quality_assessments=quality_assessments,
    )
    for row, prediction, quality in zip(
        report_rows, predictions, quality_assessments, strict=True
    ):
        row["Reporting decision"] = reporting_status(
            confidence=float(prediction["confidence"]),
            confidence_threshold=confidence_threshold,
            quality_score=int(quality["score"]),
        )

    processing_time_ms = float(result["segmentation"].processing_time_ms) + sum(
        float(item["inference_time_ms"]) for item in predictions
    )
    summary = build_summary(
        predictions=predictions,
        quality_assessments=quality_assessments,
        confidence_threshold=confidence_threshold,
        processing_time_ms=processing_time_ms,
    )
    metadata = make_traceability_metadata(
        input_name=st.session_state["input_name"],
        source_description=st.session_state["source_description"],
        roi_profile=result["roi_profile"],
        confidence_threshold=confidence_threshold,
        model_path=str(model_path),
        model_classes=model_bundle["classes"],
        segmentation_parameters=result["segmentation"].parameters,
    )
    metadata["model_version"] = (
        f"{model_path.stat().st_size}-{model_path.stat().st_mtime_ns}"
    )
    track_reports = []
    for track, prediction, quality in zip(
        result["features"], predictions, quality_assessments, strict=True
    ):
        track_reports.append(
            {
                "track_id": track.track_id,
                "prediction": prediction,
                "quality": quality,
                "reporting_status": reporting_status(
                    confidence=float(prediction["confidence"]),
                    confidence_threshold=confidence_threshold,
                    quality_score=int(quality["score"]),
                ),
                "features": _feature_row(track),
            }
        )
    complete_report = {
        "metadata": metadata,
        "summary": summary,
        "tracks": track_reports,
        "interpretation_note": (
            "Model confidence estimates class preference. Contour quality is "
            "an explainable heuristic and is not a correctness probability."
        ),
    }

    st.subheader("Image-level result summary")
    summary_columns = st.columns(6)
    summary_columns[0].metric("Detected", summary["detected_contours"])
    summary_columns[1].metric("Confident", summary["confident_classifications"])
    summary_columns[2].metric("Uncertain", summary["uncertain_classifications"])
    summary_columns[3].metric("Dominant", summary["dominant_prediction"])
    summary_columns[4].metric(
        "Contour quality",
        f"{summary['overall_contour_quality']} ({summary['mean_contour_quality']:.0f})",
    )
    summary_columns[5].metric("Processing", f"{processing_time_ms:.1f} ms")
    st.dataframe(
        [
            {"Particle type": name, "Predicted count": count}
            for name, count in summary["class_counts"].items()
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Annotated classification overview")
    st.image(
        _bgr_to_rgb(overlay),
    )
    st.markdown(
        "**Legend:** Orange = Alpha | Blue = Electron/Positron | Green = Proton | "
        "Purple = V-track | Yellow = Uncertain | Grey Dashed = Low Quality (Review Segmentation)"
    )
    st.dataframe(report_rows, use_container_width=True, hide_index=True)

    _render_particle_evidence(
        result["segmentation"].binary_mask,
        result["features"],
        predictions,
        quality_assessments,
        confidence_threshold,
    )

    batch_row = {
        "Input": st.session_state["input_name"],
        "Source": st.session_state["source_description"],
        "Tracks": summary["detected_contours"],
        "Dominant prediction": summary["dominant_prediction"],
        "Confident": summary["confident_classifications"],
        "Uncertain": summary["uncertain_classifications"],
        "Alpha": summary["class_counts"]["Alpha"],
        "Electron/Positron": summary["class_counts"]["Electron/Positron"],
        "Proton": summary["class_counts"]["Proton"],
        "V-track": summary["class_counts"]["V-track"],
        "Contour quality": summary["overall_contour_quality"],
        "Mean quality score": summary["mean_contour_quality"],
        "Processing time (ms)": summary["processing_time_ms"],
    }
    st.session_state["batch_reports"][st.session_state["input_name"]] = batch_row
    _render_batch_summary()

    safe_name = Path(st.session_state["input_name"]).stem
    st.subheader("Download reproducible report")
    downloads = st.columns(3)
    downloads[0].download_button(
        "Download annotated image",
        data=encode_report_png(overlay),
        file_name=f"{safe_name}_mlp_report.png",
        mime="image/png",
    )
    downloads[1].download_button(
        "Particle CSV",
        data=encode_report_csv(report_rows),
        file_name=f"{safe_name}_mlp_report.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "PDF summary",
        data=encode_pdf_report(overlay, complete_report),
        file_name=f"{safe_name}_mlp_report.pdf",
        mime="application/pdf",
    )


def _render_particle_evidence(
    binary_mask: np.ndarray,
    features: list,
    predictions: list[dict],
    quality_assessments: list[dict],
    confidence_threshold: float,
    expand_cards: bool = True,
) -> None:
    st.subheader("Particle evidence cards")
    items_per_page = 10
    page_count = max(1, (len(features) + items_per_page - 1) // items_per_page)
    page = st.number_input(
        "Evidence page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key=f"evidence_page_{st.session_state['input_name']}",
    )
    start = (int(page) - 1) * items_per_page
    stop = min(start + items_per_page, len(features))
    for track, prediction, quality in zip(
        features[start:stop],
        predictions[start:stop],
        quality_assessments[start:stop],
        strict=True,
    ):
        decision = reporting_status(
            confidence=float(prediction["confidence"]),
            confidence_threshold=confidence_threshold,
            quality_score=int(quality["score"]),
        )
        title = (
            f"T{track.track_id} — {prediction['particle_type']} "
            f"({prediction['confidence']:.0%}) — {decision}"
        )
        card = (
            st.expander(title, expanded=False)
            if expand_cards
            else st.container(border=True)
        )
        with card:
            if not expand_cards:
                st.markdown(f"**{title}**")
            x, y, width, height = track.bounding_box
            padding = 10
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(binary_mask.shape[1], x + width + padding)
            bottom = min(binary_mask.shape[0], y + height + padding)
            st.image(
                binary_mask[top:bottom, left:right],
                caption="Contour mask sent to feature extraction",
                width=320,
            )
            st.markdown("**Class probability evidence**")
            for class_name, probability in sorted(
                prediction["probabilities"].items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                st.progress(
                    float(probability),
                    text=(
                        f"{DISPLAY_NAMES.get(class_name, class_name)}: "
                        f"{probability:.1%}"
                    ),
                )
            details = st.columns(2)
            details[0].markdown(
                f"**Contour quality:** {quality['grade']} "
                f"({quality['score']}/100)  \n"
                f"**Reporting decision:** {decision}  \n"
                f"**Local contrast:** {quality['local_contrast']:.1f}"
            )
            details[1].dataframe(
                [_feature_row(track)], use_container_width=True, hide_index=True
            )
            for item in quality["warnings"]:
                st.warning(item)


def _render_batch_summary() -> None:
    """Show accumulated image/frame results from the current GUI session."""
    rows = list(st.session_state.get("batch_reports", {}).values())
    if not rows:
        return
    st.subheader("Batch and video-frame summary")
    st.dataframe(rows, use_container_width=True, hide_index=True)

def _extra_trees_page() -> None:
    """Beginner-friendly Extra Trees classification page."""

    # =====================================================
    # PAGE TITLE
    # =====================================================

    st.title("Extra Trees Classifier")

    st.write(
        "Extra Trees analyses the detected particle tracks and predicts the "
        "particle type for each track."
    )

    project_root = Path(__file__).resolve().parent

    model_path = (
        project_root
        / "models"
        / "extra_trees_classifier.joblib"
    )

    report_path = (
        project_root
        / "models"
        / "extra_trees_training_report.json"
    )


    # =====================================================
    # 1. MODEL STATUS
    # =====================================================

    if not model_path.exists():

        st.warning(
            "Extra Trees has not been trained yet."
        )

        st.code(
            "python scripts/train_extra_trees.py"
        )

        return


    # =====================================================
    # 2. MODEL PERFORMANCE
    # =====================================================

    training_report = None

    if report_path.exists():

        training_report = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        validation = training_report["validation"]

        final_test = training_report["final_test"]

        selected_candidate = training_report.get(
            "selected_candidate"
        )


        st.header("Model Performance")


        # -------------------------------------------------
        # Main performance results
        # -------------------------------------------------

        columns = st.columns(3)


        columns[0].metric(
            "Validation Macro F1",
            f"{validation['macro_f1']:.3f}",
        )


        columns[1].metric(
            "Final Test Macro F1",
            f"{final_test['macro_f1']:.3f}",
        )


        columns[2].metric(
            "Final Test Accuracy",
            f"{final_test['accuracy']:.1%}",
        )


        if selected_candidate:

            st.write(f"Selected model: **{selected_candidate['name']}**")


        st.caption(
            "Validation Macro F1 was used to select the Extra Trees "
            "configuration. Final-test results measure performance on unseen "
            "test data."
        )

        # Report-ready tables matching Section 4.1.2 of the assignment.
        # Precision, recall and F1 are macro averages in the model row. The
        # saved report records one overall mean inference time rather than a
        # separate timing measurement for every class.
        with st.expander("View report-ready final-test tables", expanded=True):
            st.subheader("Extreme Random Tree - Model Configuration")
            macro_result = final_test["classification_report"]["macro avg"]
            model_configuration = (
                f"{selected_candidate.get('n_estimators', '-')} trees, "
                f"maximum depth = {selected_candidate.get('max_depth', '-')}, "
                f"minimum split = "
                f"{selected_candidate.get('min_samples_split', '-')}, "
                f"maximum features = "
                f"{selected_candidate.get('max_features', '-')}, "
                f"class weight = "
                f"{selected_candidate.get('class_weight', '-')}"
                if selected_candidate
                else "Selected Extra Trees configuration"
            )
            processing_time = float(
                final_test.get("mean_inference_ms_per_track", 0.0)
            )
            model_table = pd.DataFrame(
                [
                    {
                        "Model Configuration": model_configuration,
                        "Accuracy": f"{float(final_test['accuracy']):.2%}",
                        "Precision": f"{float(macro_result['precision']):.2%}",
                        "Recall": f"{float(macro_result['recall']):.2%}",
                        "F1-score": f"{float(macro_result['f1-score']):.2%}",
                        "Processing Time": f"{processing_time:.3f} ms/track",
                    }
                ]
            )
            st.dataframe(model_table, use_container_width=True, hide_index=True)

            st.subheader("Extreme Random Tree - Per-Class Performance")
            confusion = np.asarray(final_test["confusion_matrix"], dtype=np.int64)
            total_samples = int(confusion.sum())
            class_table = []
            display_names = {
                "alpha": "Alpha",
                "electron_positron": "Electron/Positron",
                "proton": "Proton",
                "v_track": "V-track",
            }
            class_processing_times = final_test.get(
                "mean_inference_ms_per_track_by_class", {}
            )
            for class_index, class_name in enumerate(final_test["class_names"]):
                class_result = final_test["classification_report"][class_name]
                true_positive = int(confusion[class_index, class_index])
                false_negative = int(confusion[class_index, :].sum()) - true_positive
                false_positive = int(confusion[:, class_index].sum()) - true_positive
                true_negative = (
                    total_samples - true_positive - false_negative - false_positive
                )
                one_vs_rest_accuracy = (
                    (true_positive + true_negative) / total_samples
                    if total_samples
                    else 0.0
                )
                class_table.append(
                    {
                        "Predicted": display_names.get(class_name, class_name),
                        "Accuracy": f"{one_vs_rest_accuracy:.2%}",
                        "Precision": f"{float(class_result['precision']):.2%}",
                        "Recall": f"{float(class_result['recall']):.2%}",
                        "F1-score": f"{float(class_result['f1-score']):.2%}",
                        "Processing Time": (
                            f"{float(class_processing_times[class_name]):.3f} ms/track"
                            if class_processing_times.get(class_name) is not None
                            else "Run training to calculate"
                        ),
                    }
                )
            st.dataframe(class_table, use_container_width=True, hide_index=True)
            st.caption(
                "Overall precision, recall and F1-score are macro averages. "
                "Per-class accuracy uses one-versus-rest calculation. The "
                "processing time for each class is the mean inference time of "
                "its final-test tracks."
            )


        # =================================================
        # MODEL TUNING EXTRA EFFORT
        # =================================================

        with st.expander(
            "View model training details"
        ):
            st.info(
                "Beginner summary: several Extra Trees settings were tested. "
                "The application kept the setting that classified all particle "
                "types most consistently on validation data."
            )

            if selected_candidate:
                st.subheader("Setting chosen by validation")
                detail_columns = st.columns(3)
                detail_columns[0].metric(
                    "Number of trees", selected_candidate.get("n_estimators", "-")
                )
                detail_columns[1].metric(
                    "Maximum tree depth",
                    selected_candidate.get("max_depth") or "No limit",
                )
                detail_columns[2].metric(
                    "Minimum samples to split",
                    selected_candidate.get("min_samples_split", "-"),
                )
                st.caption(
                    "You do not need to adjust these values when classifying an "
                    "image. They were chosen automatically during training."
                )

            tuning_rows = []


            for candidate in training_report.get(
                "candidate_results",
                [],
            ):

                is_selected = (
                    selected_candidate is not None
                    and candidate["name"]
                    == selected_candidate["name"]
                )


                tuning_rows.append(
                    {
                        "Model":
                            candidate["name"],

                        "Trees":
                            candidate["n_estimators"],

                        "Max Depth":
                            (
                                "Unlimited"
                                if candidate["max_depth"] is None
                                else str(candidate["max_depth"])
                            ),

                        "Max Features":
                            candidate.get("max_features", "Default"),

                        "Min Split":
                            candidate.get("min_samples_split", 2),

                        "Min Leaf":
                            candidate["min_samples_leaf"],

                        "Criterion":
                            candidate.get("criterion", "gini"),

                        "Class Weight":
                            (
                                "None"
                                if candidate["class_weight"] is None
                                else candidate["class_weight"]
                            ),

                        "Validation Macro F1":
                            round(
                                candidate[
                                    "validation_macro_f1"
                                ],
                                3,
                            ),

                        "Selected":
                            "Yes"
                            if is_selected
                            else "",
                    }
                )


            show_candidates = st.checkbox(
                "Show every tested training setting",
                key="extra_trees_show_training_candidates",
            )
            if show_candidates:
                st.subheader("All tested training settings")
                st.dataframe(
                    tuning_rows,
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "The row marked Yes achieved the highest Validation Macro "
                    "F1 and was selected automatically."
                )

            selected_parameters = training_report.get("parameters")
            show_parameters = st.checkbox(
                "Show all technical model parameters",
                key="extra_trees_show_training_parameters",
            )
            if selected_parameters and show_parameters:
                st.json(selected_parameters)


    # =====================================================
    # 3. CLASSIFY IMAGE OR BATCH (INCLUDING VIDEO FRAMES)
    # =====================================================

    st.header("Classify Input")

    config = load_config()

    result = st.session_state.get("pipeline_result")
    image = st.session_state.get("input_image")

    if result is None and not st.session_state.get("input_batch"):
        st.info(
            "Go to **Shared Processing Pipeline**, upload an image or video, "
            "and process it first."
        )
        return

    model_bundle = load_extra_trees_model(model_path)
    supported_classes = tuple(config["classification"]["supported_classes"])
    model_classes = tuple(str(value) for value in model_bundle["classes"])
    if set(model_classes) != set(supported_classes):
        st.warning(
            "The trained Extra Trees classes do not match the configured "
            "supported classes. Results below retain the trained model's "
            "classes; retrain only after confirming the intended taxonomy."
        )

    # Keep beginner reporting consistent without exposing an extra control.
    confidence_threshold = 0.50

    samples = st.session_state.get("input_batch") or (
        [
            {
                "image": image,
                "name": st.session_state["input_name"],
                "description": st.session_state["source_description"],
            }
        ]
        if image is not None
        else []
    )

    if not samples:
        st.info("No acquired input is available to classify.")
        return

    button_label = (
        "Classify All Inputs"
        if len(samples) > 1
        else "Classify Image"
    )

    if st.button(button_label, type="primary", key="btn_classify_extra_trees"):
        batch_results = {}
        progress = st.progress(0.0, text="Processing batch with Extra Trees...")
        for index, sample in enumerate(samples):
            sample_result = _process_pipeline_image(sample["image"], config)
            sample_predictions = predict_extra_trees_tracks(
                model_bundle, sample_result["features"]
            )
            sample_quality = assess_all_contours(
                sample_result["features"],
                sample_result["enhancement"].enhanced,
                sample_result["segmentation"].binary_mask,
                sample_result["segmentation"].parameters,
            )
            batch_results[sample["name"]] = {
                "image": sample["image"],
                "description": sample["description"],
                "result": sample_result,
                "predictions": sample_predictions,
                "quality": sample_quality,
            }
            progress.progress(
                (index + 1) / len(samples),
                text=f"Processed {index + 1} of {len(samples)} inputs",
            )
        progress.empty()
        st.session_state["extra_trees_batch_results"] = batch_results

    batch_results = st.session_state.get("extra_trees_batch_results", {})
    if batch_results:
        st.subheader("Batch classification results")
        for index, (name, entry) in enumerate(batch_results.items(), start=1):
            item_predictions = entry["predictions"]
            item_quality = entry["quality"]
            item_result = entry["result"]
            item_summary = build_summary(
                predictions=item_predictions,
                quality_assessments=item_quality,
                confidence_threshold=confidence_threshold,
                processing_time_ms=float(
                    item_result["segmentation"].processing_time_ms
                )
                + sum(
                    float(item["inference_time_ms"])
                    for item in item_predictions
                ),
            )
            item_summary["class_counts"] = _extra_trees_class_counts(
                item_predictions, model_classes
            )
            st.session_state["extra_trees_batch_reports"][name] = {
                "Input": name,
                "Source": entry["description"],
                "Tracks": item_summary["detected_contours"],
                "Dominant prediction": item_summary["dominant_prediction"],
                "Confident": item_summary["confident_classifications"],
                "Uncertain": item_summary["uncertain_classifications"],
                "Class counts": item_summary["class_counts"],
                "Contour quality": item_summary["overall_contour_quality"],
                "Mean quality score": item_summary["mean_contour_quality"],
                "Processing time (ms)": item_summary["processing_time_ms"],
            }
            with st.expander(
                f"Image {index}: {name} — "
                f"{item_summary['detected_contours']} tracks — "
                f"Main prediction: {item_summary['dominant_prediction']}",
                expanded=False,
            ):
                if item_predictions:
                    item_overlay, _ = build_extra_trees_visual_report(
                        image=entry["image"],
                        features=item_result["features"],
                        predictions=item_predictions,
                        confidence_threshold=confidence_threshold,
                        quality_assessments=item_quality,
                    )
                    st.image(_bgr_to_rgb(item_overlay), width=700)
                else:
                    st.warning("No segmented track was available to classify.")

        selected_name = st.selectbox(
            "Choose an input for the detailed report",
            list(batch_results),
            key="extra_trees_batch_selector",
        )
        selected_entry = batch_results[selected_name]
        st.session_state["input_image"] = selected_entry["image"]
        st.session_state["input_name"] = selected_name
        st.session_state["source_description"] = selected_entry["description"]
        st.session_state["pipeline_result"] = selected_entry["result"]
        st.session_state["extra_trees_predictions"] = selected_entry["predictions"]
        st.session_state["extra_trees_quality"] = selected_entry["quality"]
        result = selected_entry["result"]

    predictions = st.session_state.get("extra_trees_predictions")
    if predictions is None:
        return
    if not predictions:
        st.warning("No segmented track is available for classification.")
        return

    quality_assessments = st.session_state.get("extra_trees_quality")
    if quality_assessments is None and result is not None:
        quality_assessments = assess_all_contours(
            result["features"],
            result["enhancement"].enhanced,
            result["segmentation"].binary_mask,
            result["segmentation"].parameters,
        )
        st.session_state["extra_trees_quality"] = quality_assessments

    classified_image, report_rows = build_extra_trees_visual_report(
        image=st.session_state["input_image"],
        features=result["features"],
        predictions=predictions,
        confidence_threshold=confidence_threshold,
        quality_assessments=quality_assessments,
    )
    for row, prediction, quality in zip(
        report_rows, predictions, quality_assessments or [], strict=False
    ):
        if quality:
            row["Reporting decision"] = reporting_status(
                confidence=float(prediction["confidence"]),
                confidence_threshold=confidence_threshold,
                quality_score=int(quality["score"]),
            )

    processing_time_ms = float(result["segmentation"].processing_time_ms) + sum(
        float(item["inference_time_ms"]) for item in predictions
    )
    summary = build_summary(
        predictions=predictions,
        quality_assessments=quality_assessments or [],
        confidence_threshold=confidence_threshold,
        processing_time_ms=processing_time_ms,
    )
    summary["class_counts"] = _extra_trees_class_counts(
        predictions, model_classes
    )
    metadata = make_traceability_metadata(
        input_name=st.session_state["input_name"],
        source_description=st.session_state["source_description"],
        roi_profile=result["roi_profile"],
        confidence_threshold=confidence_threshold,
        model_path=str(model_path),
        model_classes=model_bundle["classes"],
        segmentation_parameters=result["segmentation"].parameters,
    )
    metadata["model_version"] = (
        f"{model_path.stat().st_size}-{model_path.stat().st_mtime_ns}"
    )
    track_reports = []
    for track, prediction, quality in zip(
        result["features"], predictions, quality_assessments or [], strict=False
    ):
        track_reports.append(
            {
                "track_id": track.track_id,
                "prediction": prediction,
                "quality": quality,
                "reporting_status": reporting_status(
                    confidence=float(prediction["confidence"]),
                    confidence_threshold=confidence_threshold,
                    quality_score=int(quality["score"]) if quality else 0,
                ),
                "features": _feature_row(track),
            }
        )
    complete_report = {
        "metadata": metadata,
        "summary": summary,
        "tracks": track_reports,
        "interpretation_note": (
            "Extra Trees confidence estimates class preference. Contour quality is "
            "an explainable heuristic and is not a correctness probability."
        ),
        "field_guide": [
            {
                "field": "Prediction and confidence",
                "description": (
                    "Prediction is the particle class preferred by Extra Trees. "
                    "Confidence is the model probability for that class, not a "
                    "guarantee that the prediction is correct."
                ),
            },
            {
                "field": "Reporting decision",
                "description": (
                    "Combines the selected confidence threshold with contour "
                    "quality. Ambiguous or review results should be checked "
                    "manually."
                ),
            },
            {
                "field": "Contour quality",
                "description": (
                    "A segmentation-quality heuristic based on the detected "
                    "shape and local image evidence. It is separate from model "
                    "confidence."
                ),
            },
            {
                "field": "Class probabilities",
                "description": (
                    "The CSV columns beginning with P(...) show the probability "
                    "assigned to every class available in the trained model."
                ),
            },
            {
                "field": "Bounding box",
                "description": (
                    "X and Y locate the upper-left corner. Width and height are "
                    "measured in pixels on the analysed image."
                ),
            },
            {
                "field": "Decision drivers",
                "description": (
                    "Lists the contour measurements that contributed most "
                    "strongly to the Extra Trees decision for that track."
                ),
            },
        ],
    }

    # =====================================================
    # CLASSIFICATION RESULT
    # =====================================================

    st.header("Classification Summary")

    summary_columns = st.columns(4)
    summary_columns[0].metric("Tracks Detected", summary["detected_contours"])
    summary_columns[1].metric("Main Prediction", summary["dominant_prediction"])
    summary_columns[2].metric(
        "Confident Tracks", summary["confident_classifications"]
    )
    summary_columns[3].metric(
        "Needs Review", summary["uncertain_classifications"]
    )

    st.subheader("Predicted Particle Counts")
    model_class_names = [
        DISPLAY_NAMES.get(str(class_name), str(class_name))
        for class_name in model_bundle["classes"]
    ]
    particle_count_rows = [
        {
            "Particle Type": class_name,
            "Number of Tracks": summary["class_counts"].get(class_name, 0),
        }
        for class_name in model_class_names
    ]
    st.dataframe(
        particle_count_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Where Were the Particles Detected?")
    st.image(
        _bgr_to_rgb(classified_image),
        caption=(
            "Each box represents one detected track. The label shows the "
            "predicted particle type and confidence."
        ),
    )
    st.markdown(_extra_trees_legend(model_bundle["classes"]))

    st.subheader("Track Results")
    show_review_only = st.checkbox(
        "Show only tracks that need review",
        key="extra_trees_show_review_only",
    )
    simplified_rows = []
    for row in report_rows:
        decision = str(row.get("Reporting decision") or row.get("Status") or "")
        simplified_rows.append(
            {
                "Track": f"T{row['Track']}",
                "Prediction": row["Particle type"],
                "Confidence": f"{float(row['Confidence']):.1%}",
                "Result": decision,
            }
        )
    if show_review_only:
        review_terms = ("uncertain", "review", "ambiguous", "manual")
        simplified_rows = [
            row
            for row in simplified_rows
            if any(term in row["Result"].lower() for term in review_terms)
        ]
    st.dataframe(simplified_rows, use_container_width=True, hide_index=True)
    st.info(
        "How to read the results: Prediction is the particle type selected by "
        "Extra Trees. Confidence shows how strongly the model prefers that "
        "prediction. Tracks marked Uncertain or Review segmentation should be "
        "checked manually."
    )

    batch_row = {
        "Input": st.session_state["input_name"],
        "Source": st.session_state["source_description"],
        "Tracks": summary["detected_contours"],
        "Dominant prediction": summary["dominant_prediction"],
        "Confident": summary["confident_classifications"],
        "Uncertain": summary["uncertain_classifications"],
        "Class counts": summary["class_counts"],
        "Contour quality": summary["overall_contour_quality"],
        "Mean quality score": summary["mean_contour_quality"],
        "Processing time (ms)": summary["processing_time_ms"],
    }
    st.session_state["extra_trees_batch_reports"][
        st.session_state["input_name"]
    ] = batch_row
    _render_extra_trees_batch_summary()

    safe_name = Path(st.session_state["input_name"]).stem
    st.subheader("Download Results")
    downloads = st.columns(3)
    downloads[0].download_button(
        "Download annotated image",
        data=encode_extra_trees_report_png(classified_image),
        file_name=f"{safe_name}_extra_trees_report.png",
        mime="image/png",
    )
    downloads[1].download_button(
        "Particle CSV",
        data=encode_extra_trees_report_csv(report_rows),
        file_name=f"{safe_name}_extra_trees_report.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "PDF summary",
        data=encode_pdf_report(classified_image, complete_report),
        file_name=f"{safe_name}_extra_trees_report.pdf",
        mime="application/pdf",
    )

    # =====================================================
    # TECHNICAL DETAILS
    # =====================================================

    with st.expander("Technical details and model evidence"):
        st.info(
            "Most users can rely on the summary, image, and simple track table "
            "above. This section is only for checking technical evidence."
        )
        st.subheader("Processing information")
        processing_columns = st.columns(3)
        processing_columns[0].metric(
            "Processing time", f"{processing_time_ms:.1f} ms"
        )
        processing_columns[1].metric(
            "Overall contour quality", summary["overall_contour_quality"]
        )
        processing_columns[2].metric(
            "Mean contour quality score",
            f"{summary['mean_contour_quality']:.0f}",
        )

        show_full_report = st.checkbox(
            "Show the full technical track table",
            key="extra_trees_show_full_report",
        )
        if show_full_report:
            st.subheader("Full technical track table")
            st.caption(
                "This contains probabilities, contour quality, bounding boxes, "
                "timing, and decision drivers. The same data is in the CSV."
            )
            st.dataframe(report_rows, use_container_width=True, hide_index=True)

        if training_report is not None:
            show_feature_importance = st.checkbox(
                "Show which measurements influenced the model overall",
                key="extra_trees_show_feature_importance",
            )
            if show_feature_importance:
                st.subheader("Which measurements mattered most?")
                st.write(
                    "A taller bar means the model used that measurement more "
                    "often across all training decisions. It does not mean the "
                    "measurement alone proves a particle type."
                )
                st.bar_chart(
                    {
                        item["feature"]: item["importance"]
                        for item in training_report.get("feature_importance", [])
                    },
                    height=280,
                )

            st.subheader("Final-Test Confusion Matrix")
            final_test = training_report["final_test"]
            st.caption(
                "Rows represent the correct particle classes. "
                "Columns represent the Extra Trees predictions."
            )
            st.write("Class order:", final_test["class_names"])
            cm_df = pd.DataFrame(
                final_test["confusion_matrix"],
                columns=final_test["class_names"],
                index=final_test["class_names"],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.subheader("Per-Class Performance")
            classification_rep = final_test["classification_report"]
            class_rows = []
            for class_name in final_test["class_names"]:
                class_result = classification_rep[class_name]
                class_rows.append(
                    {
                        "Class": class_name,
                        "Precision": round(class_result["precision"], 3),
                        "Recall": round(class_result["recall"], 3),
                        "F1 Score": round(class_result["f1-score"], 3),
                        "Support": int(class_result["support"]),
                    }
                )
            st.dataframe(class_rows, use_container_width=True, hide_index=True)



def _render_extra_trees_batch_summary() -> None:
    """Show accumulated image/frame Extra Trees results from the current GUI session."""
    rows = list(st.session_state.get("extra_trees_batch_reports", {}).values())
    if not rows:
        return
    st.subheader("Batch Results")
    simple_rows = [
        {
            "Input": row["Input"],
            "Tracks": row["Tracks"],
            "Main Prediction": row["Dominant prediction"],
            "Confident": row["Confident"],
            "Uncertain": row["Uncertain"],
        }
        for row in rows
    ]
    st.dataframe(simple_rows, use_container_width=True, hide_index=True)


def _extra_trees_class_styles(model_classes: list | tuple) -> list[tuple[str, str]]:
    """Return friendly class names and CSS colours for classes in the bundle."""
    styles = []
    for class_name in model_classes:
        class_id = str(class_name)
        bgr = EXTRA_TREES_CLASS_COLOURS.get(class_id, (255, 255, 255))
        blue, green, red = (int(value) for value in bgr)
        styles.append(
            (
                DISPLAY_NAMES.get(class_id, class_id),
                f"#{red:02X}{green:02X}{blue:02X}",
            )
        )
    return styles


def _extra_trees_class_counts(
    predictions: list[dict], model_classes: list | tuple
) -> dict[str, int]:
    """Count predictions using the classes declared by the saved model."""
    return {
        DISPLAY_NAMES.get(str(class_name), str(class_name)): sum(
            prediction["predicted_class"] == str(class_name)
            for prediction in predictions
        )
        for class_name in model_classes
    }


def _extra_trees_legend(model_classes: list | tuple) -> str:
    """Build a legend from the saved model classes and actual box colours."""
    colour_names = {
        "#FF0000": "Red",
        "#0000FF": "Blue",
        "#00FF00": "Green",
        "#B400B4": "Purple",
        "#FFFFFF": "White",
    }
    class_items = [
        f"{colour_names[colour]} = {display_name}"
        for display_name, colour in _extra_trees_class_styles(model_classes)
    ]
    return "**Legend:** " + " | ".join(
        [*class_items, "Yellow = Uncertain", "Grey dashed = Review segmentation"]
    )

def _comparison_page() -> None:
    st.title("Final Model Comparison")
    st.write(
        "Final-test metrics are shown only from each model's saved training "
        "report. Validation results are not used as final performance."
    )

    project_root = Path(__file__).resolve().parent
    report_files = {
        "CNN": "cnn_training_report.json",
        "SVM": "svm_training_report.json",
        "Decision Tree": "decision_tree_training_report.json",
        "MLP": "mlp_training_report.json",
        "Extra Trees": "extra_trees_training_report.json",
    }
    available_reports = {}
    unavailable_models = []
    comparison_rows = []

    for model_name, filename in report_files.items():
        path = project_root / "models" / filename
        report = None
        if path.exists():
            try:
                candidate_report = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(candidate_report.get("final_test"), dict):
                    report = candidate_report
            except (OSError, json.JSONDecodeError):
                report = None

        if report is None:
            unavailable_models.append(model_name)
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Accuracy": "Not available",
                    "Balanced Accuracy": "Not available",
                    "Macro F1": "Not available",
                    "Weighted F1": "Not available",
                    "Mean Inference Time / Track (ms)": "Not available",
                }
            )
            continue

        available_reports[model_name] = report
        final_test = report["final_test"]
        mean_time = final_test.get(
            "mean_inference_ms_per_track",
            final_test.get("mean_inference_time_per_track_ms"),
        )
        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy": f"{float(final_test['accuracy']):.3f}",
                "Balanced Accuracy": f"{float(final_test['balanced_accuracy']):.3f}",
                "Macro F1": f"{float(final_test['macro_f1']):.3f}",
                "Weighted F1": f"{float(final_test['weighted_f1']):.3f}",
                "Mean Inference Time / Track (ms)": (
                    f"{float(mean_time):.4f}" if mean_time is not None else "Not available"
                ),
            }
        )

    highest_macro_model = None
    if available_reports:
        highest_macro_model = max(
            available_reports,
            key=lambda name: float(
                available_reports[name]["final_test"]["macro_f1"]
            ),
        )
        for row in comparison_rows:
            if row["Model"] == highest_macro_model:
                row["Model"] = f"★ {row['Model']}"

    st.dataframe(comparison_rows, use_container_width=True, hide_index=True)

    if highest_macro_model:
        highest_score = available_reports[highest_macro_model]["final_test"][
            "macro_f1"
        ]
        st.success(
            f"Highest Final-Test Macro F1: **{highest_macro_model}** "
            f"({float(highest_score):.3f})."
        )

    if unavailable_models:
        st.warning(
            "Not included yet because no usable final-test report was found: "
            + ", ".join(unavailable_models)
            + "."
        )

    configured_classes = set(
        load_config()["classification"]["supported_classes"]
    )
    compatibility_signatures = {}
    compatibility_problems = []
    for model_name, report in available_reports.items():
        final_test = report["final_test"]
        report_classes = set(str(name) for name in final_test.get("class_names", []))
        if report_classes != configured_classes:
            compatibility_problems.append(
                f"{model_name} uses a different class taxonomy."
            )

        final_counts = report.get("class_counts", {}).get("final_test", {})
        final_signature = tuple(
            sorted((str(name), int(count)) for name, count in final_counts.items())
        )
        feature_signature = tuple(report.get("feature_columns", []))
        split_identity = report.get("final_test_split_id") or report.get(
            "final_test_dataset"
        )
        feature_method = report.get("feature_source")
        if not split_identity or not feature_method or not final_signature:
            compatibility_problems.append(
                f"{model_name} does not record enough dataset/method metadata "
                "to verify direct comparability."
            )
        compatibility_signatures[model_name] = (
            tuple(sorted(report_classes)),
            final_signature,
            feature_signature,
            split_identity,
            feature_method,
        )

    if len(set(compatibility_signatures.values())) > 1:
        compatibility_problems.append(
            "Available reports do not record matching final-test identity, "
            "class counts, and shared feature methodology, so their scores "
            "are shown for reference but should not be treated as a direct "
            "ranking."
        )

    if compatibility_problems:
        st.warning(" ".join(compatibility_problems))
    elif len(available_reports) > 1:
        st.caption(
            "Compatibility check passed: available reports record the same "
            "class taxonomy, final-test identity and counts, and feature "
            "methodology."
        )

    for model_name, report in available_reports.items():
        final_test = report["final_test"]
        class_names = [str(name) for name in final_test.get("class_names", [])]
        with st.expander(f"{model_name} final-test details"):
            st.subheader("Confusion matrix")
            confusion = final_test.get("confusion_matrix")
            if confusion and class_names:
                st.caption("Rows are true classes; columns are predicted classes.")
                st.dataframe(
                    pd.DataFrame(confusion, index=class_names, columns=class_names),
                    use_container_width=True,
                )
            else:
                st.info("Confusion-matrix data is not available in this report.")

            st.subheader("Per-class performance")
            classification_rep = final_test.get("classification_report", {})
            class_rows = []
            for class_name in class_names:
                class_result = classification_rep.get(class_name, {})
                class_rows.append(
                    {
                        "Class": DISPLAY_NAMES.get(class_name, class_name),
                        "Precision": class_result.get("precision"),
                        "Recall": class_result.get("recall"),
                        "F1": class_result.get("f1-score"),
                        "Support": class_result.get("support"),
                    }
                )
            st.dataframe(class_rows, use_container_width=True, hide_index=True)

            mean_time = final_test.get(
                "mean_inference_ms_per_track",
                final_test.get("mean_inference_time_per_track_ms"),
            )
            st.subheader("Processing time")
            if mean_time is None:
                st.write("Mean inference time per track: Not available")
            else:
                st.metric(
                    "Mean inference time / track",
                    f"{float(mean_time):.4f} ms",
                )


def _feature_row(item, centimetres_per_pixel: float | None = None) -> dict:
    """Convert one contour feature object into a display-table row."""
    row = {
        "Track": item.track_id,
        "Area (pxÂ²)": round(item.area_pixels, 3),
        "Perimeter (px)": round(item.perimeter_pixels, 3),
        "Length (px)": round(item.major_axis_pixels, 3),
        "Width (px)": round(item.mean_width_pixels, 3),
        "Aspect ratio": round(item.aspect_ratio, 3),
        "Solidity": round(item.solidity, 3),
        "Rectangularity": round(item.rectangularity, 3),
        "Thickness (px)": round(item.thickness_pixels, 3),
        "Orientation (Â°)": round(item.orientation_degrees, 3),
        "Mean intensity": round(item.mean_intensity, 3),
    }
    if centimetres_per_pixel is not None:
        scale = float(centimetres_per_pixel)
        row.update(
            {
                "Area (cm²)": round(item.area_pixels * scale * scale, 6),
                "Perimeter (cm)": round(item.perimeter_pixels * scale, 6),
                "Length (cm)": round(item.major_axis_pixels * scale, 6),
                "Width (cm)": round(item.mean_width_pixels * scale, 6),
                "Thickness (cm)": round(item.thickness_pixels * scale, 6),
            }
        )
    return row


def _set_input(image: np.ndarray, name: str, description: str) -> None:
    st.session_state["input_image"] = image
    st.session_state["input_name"] = name
    st.session_state["source_description"] = description
    st.session_state["pipeline_result"] = None
    st.session_state["mlp_predictions"] = None
    st.session_state["mlp_quality"] = None
    st.session_state["extra_trees_quality"] = None
    st.session_state["svm_predictions"] = None
    st.session_state["svm_quality"] = None
    st.session_state["decision_tree_predictions"] = None
    st.session_state["decision_tree_quality"] = None
    st.session_state["extra_trees_predictions"] = None





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
    if st.session_state.get("input_image") is None:
        st.sidebar.warning("No image selected")
    else:
        st.sidebar.success(f"Input: {st.session_state['input_name']}")


def _decode_uploaded_image(data: bytes) -> np.ndarray:
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
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
