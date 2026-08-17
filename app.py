"""Streamlit GUI for the corrected Mode A cloud-chamber pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cloud_chamber.config import load_config
from cloud_chamber.calibration import (
    detect_chamber_corners,
    spatial_scale,
    rectify_image,
)
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import extract_track_features
from cloud_chamber.segmentation import segment_tracks
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


def main() -> None:
    st.set_page_config(
        page_title="Cloud Chamber Particle Classification",
        page_icon="â˜ï¸",
        layout="wide",
    )
    config = load_config()
    _initialise_state()

    st.sidebar.title("Cloud Chamber")
    st.sidebar.caption("BMDS2133 Mode A comparative study")
    page = st.sidebar.radio("Navigate", PAGES)
    _input_status()

    render_selected_page(page, config, _build_page_handlers())


def _build_page_handlers():
    """Connect navigation labels to the existing page-rendering functions."""
    context = PageContext(
        bgr_to_rgb=_bgr_to_rgb,
        process_pipeline_image=_process_pipeline_image,
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
    st.session_state.setdefault("decision_tree_quality", None)
    st.session_state.setdefault("batch_reports", {})
    st.session_state.setdefault("mlp_batch_results", {})
    st.session_state.setdefault("decision_tree_batch_results", {})
    st.session_state.setdefault("svm_batch_results", {})
    st.session_state.setdefault("calibration_settings", None)
    st.session_state.setdefault("config", None)


def _shared_pipeline_page(config: dict) -> None:
    st.title("Shared Image Processing Pipeline")

    st.header("Image or video-frame acquisition")
    _acquisition_section(config)
    image = st.session_state.get("input_image")
    if image is None:
        return

    calibration_settings = _calibration_section(image)
    result = _process_pipeline_image(image, config, calibration_settings)
    analysis_image = result["input_image"]
    enhancement = result["enhancement"]
    segmentation = result["segmentation"]
    features = result["features"]
    scaling = result["spatial_normalisation"]
    st.header("Spatial scaling")
    scaling_columns = st.columns(2)
    scaling_columns[0].image(
        _bgr_to_rgb(result["rectified_image"]),
        caption=(
            f"Rectified source: {scaling['original_width']} × "
            f"{scaling['original_height']} pixels"
        ),
    )
    scaling_columns[1].image(
        _bgr_to_rgb(result["spatially_scaled_image"]),
        caption=(
            f"Spatially scaled image: {scaling['scaled_width']} × "
            f"{scaling['scaled_height']} pixels"
        ),
    )
    st.caption(
        f"Longest side normalised to {scaling['target_longest_side']} pixels; "
        f"scale factor = {scaling['scale']:.4f}. Scaling is performed after "
        "rectification and before image enhancement."
    )
    st.subheader("Spatial-resolution selection")
    resolution_report = Path("results/spatial_resolution_validation.json")
    if resolution_report.exists():
        resolution_rows = json.loads(
            resolution_report.read_text(encoding="utf-8")
        )
        best_f1 = max(row["f1"] for row in resolution_rows)
        st.dataframe(
            [
                {
                    "Longest side": row["longest_side"],
                    "Precision": round(row["precision"], 3),
                    "Recall": round(row["recall"], 3),
                    "F1": round(row["f1"], 3),
                    "Mean time (ms/image)": round(
                        row["mean_processing_ms"], 1
                    ),
                    "Result": (
                        "Highest validation F1"
                        if row["f1"] == best_f1
                        else (
                            "Current configured target"
                            if row["longest_side"]
                            == scaling["target_longest_side"]
                            else "Candidate"
                        )
                    ),
                }
                for row in resolution_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Measured on 127 validation images using one-to-one IoU ≥ 0.50. "
        "The combined result is dominated by the larger external validation "
        "set; primary and thin-track results must also be considered before "
        "fixing the final target."
    )
    st.header("Grayscale conversion and Gaussian filtering")
    columns = st.columns(3)
    columns[0].image(_bgr_to_rgb(analysis_image), caption="Calibrated input")
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
    st.session_state["batch_reports"].pop(
        st.session_state.get("input_name"), None
    )

    overlay = analysis_image.copy()
    # These boxes come only from the accepted contours used to create the
    # clean mask above. Rejected and out-of-ROI contours cannot appear here.
    for x, y, width, height in segmentation.bounding_boxes:
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
    st.image(_bgr_to_rgb(overlay), caption="Detected particle-track contours")

    st.header("Contour-based feature extraction")
    rows = [
        _feature_row(item, result["centimetres_per_pixel"])
        for item in features
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No contour passed the configured minimum-area filter.")


def _calibration_section(image: np.ndarray) -> dict | None:
    """Collect editable chamber corners for perspective rectification."""
    st.header("Perspective rectification")
    height, width = image.shape[:2]
    rectification_points = None
    if True:
        detected_corners = detect_chamber_corners(image)
        source_key = f"{st.session_state.get('input_name')}:{width}x{height}"
        coordinate_keys = [
            (f"rectification_x_{index}", f"rectification_y_{index}")
            for index in range(4)
        ]
        if st.session_state.get("calibration_corner_source") != source_key:
            st.session_state["calibration_corner_source"] = source_key
            for point, (x_key, y_key) in zip(
                detected_corners, coordinate_keys
            ):
                st.session_state[x_key] = int(round(float(point[0])))
                st.session_state[y_key] = int(round(float(point[1])))

        if st.button("Reset coordinates to automatic values"):
            for point, (x_key, y_key) in zip(
                detected_corners, coordinate_keys
            ):
                st.session_state[x_key] = int(round(float(point[0])))
                st.session_state[y_key] = int(round(float(point[1])))

        names = ("Top-left", "Top-right", "Bottom-right", "Bottom-left")
        points = []
        with st.expander("Rectification corner coordinates", expanded=True):
            st.caption(
                "Values are filled automatically. Adjust them if the preview "
                "does not follow the real chamber boundary."
            )
            for index, (name, (x_key, y_key)) in enumerate(
                zip(names, coordinate_keys)
            ):
                columns = st.columns(2)
                point_x = columns[0].number_input(
                    f"{name} X",
                    min_value=0,
                    max_value=width - 1,
                    step=1,
                    key=x_key,
                )
                point_y = columns[1].number_input(
                    f"{name} Y",
                    min_value=0,
                    max_value=height - 1,
                    step=1,
                    key=y_key,
                )
                points.append((float(point_x), float(point_y)))
        rectification_points = points

        adjusted_corners = np.asarray(points, dtype=np.int32)
        corner_preview = image.copy()
        for index, (point, corner_label) in enumerate(
            zip(adjusted_corners, ("TL", "TR", "BR", "BL"))
        ):
            x, y = int(point[0]), int(point[1])
            next_point = adjusted_corners[(index + 1) % 4]
            cv2.line(
                corner_preview,
                (x, y),
                (int(next_point[0]), int(next_point[1])),
                (0, 255, 255),
                3,
            )
            cv2.circle(corner_preview, (x, y), 10, (0, 255, 255), -1)
            cv2.putText(
                corner_preview,
                corner_label,
                (x + 12, max(y - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    settings = {
        "rectification_points": rectification_points,
    }
    try:
        preview_image = rectify_image(image, rectification_points)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    preview_columns = st.columns(2)
    preview_columns[0].image(
        _bgr_to_rgb(corner_preview),
        caption="Editable chamber boundary",
        width=420,
    )
    preview_columns[1].image(
        _bgr_to_rgb(preview_image),
        caption="Perspective-rectified image",
        width=420,
    )
    st.session_state["calibration_settings"] = settings
    return settings


def _process_pipeline_image(
    image: np.ndarray,
    config: dict,
    calibration_settings: dict | None = None,
) -> dict:
    """Run the identical shared pipeline for one image without drawing UI."""
    # The original orientation selects the appropriate dataset-specific ROI.
    profile_name = _detect_layout_profile(image)
    analysis_image = image
    centimetres_per_pixel = None
    rectified = False
    if calibration_settings and calibration_settings.get(
        "rectification_points"
    ) is not None:
        analysis_image = rectify_image(
            image, calibration_settings["rectification_points"]
        )
        rectified = True

    rectified_image = analysis_image
    normalisation_settings = config.get("spatial_normalisation", {})
    normalisation = spatial_scale(
        rectified_image,
        int(normalisation_settings.get("target_longest_side", 1920)),
    )
    spatially_scaled_image = normalisation.image
    analysis_image = spatially_scaled_image

    selected_profile = config["segmentation"]["roi_profiles"][profile_name]
    roi_margins = {
        side: selected_profile[side]
        for side in ("left", "right", "top", "bottom")
    }
    segmentation_settings = {
        **config["segmentation"],
        **{
            name: value
            for name, value in selected_profile.items()
            if name not in roi_margins
        },
    }
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
    return {
        "input_image": analysis_image,
        "rectified_image": rectified_image,
        "spatially_scaled_image": spatially_scaled_image,
        "enhancement": enhancement,
        "segmentation": segmentation,
        "features": features,
        "roi_profile": profile_name,
        "centimetres_per_pixel": centimetres_per_pixel,
        "rectified": rectified,
        "spatial_normalisation": {
            "target_longest_side": int(
                normalisation_settings.get("target_longest_side", 1920)
            ),
            "scale": normalisation.scale,
            "original_width": normalisation.original_width,
            "original_height": normalisation.original_height,
            "scaled_width": normalisation.scaled_width,
            "scaled_height": normalisation.scaled_height,
        },
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
                    use_container_width=True,
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
    return row


def _set_input(image: np.ndarray, name: str, description: str) -> None:
    st.session_state["input_image"] = image
    st.session_state["input_name"] = name
    st.session_state["source_description"] = description
    st.session_state["pipeline_result"] = None
    st.session_state["mlp_predictions"] = None
    st.session_state["mlp_quality"] = None
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


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
