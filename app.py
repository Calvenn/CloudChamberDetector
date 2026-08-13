"""Streamlit GUI for the corrected Mode A cloud-chamber pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cloud_chamber.config import load_config
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import extract_track_features
from cloud_chamber.ml.member_models.mlp import (
    build_visual_report,
    encode_report_csv,
    encode_report_png,
    load_model,
    predict_tracks,
)
from cloud_chamber.segmentation import segment_tracks


MODEL_PAGES = {
    "CNN": "Convolutional Neural Network",
    "SVM": "Support Vector Machine",
    "Decision Tree": "Decision Tree",
    "MLP": "Multilayer Perceptron",
    "Extra Trees": "Extremely Randomised Trees",
}
ROI_PROFILE_LABELS = {
    "Auto-detect from image shape (recommended)": "auto",
    "External Muller / already cropped": "external_muller",
    "Primary dataset / full chamber": "primary_full_chamber",
}
PAGES = ["Shared Processing Pipeline", *MODEL_PAGES, "Final Model Comparison"]


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

    if page == "Shared Processing Pipeline":
        _shared_pipeline_page(config)
    elif page == "Final Model Comparison":
        _comparison_page()
    else:
        _model_page(page, MODEL_PAGES[page])


def _initialise_state() -> None:
    st.session_state.setdefault("input_image", None)
    st.session_state.setdefault("input_name", None)
    st.session_state.setdefault("source_description", None)
    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("mlp_predictions", None)
    st.session_state.setdefault(
        "layout_choice", "Auto-detect from image shape (recommended)"
    )
    if st.session_state["layout_choice"] not in ROI_PROFILE_LABELS:
        # Replace an obsolete selection retained by an older Streamlit session.
        st.session_state["layout_choice"] = (
            "Auto-detect from image shape (recommended)"
        )


def _shared_pipeline_page(config: dict) -> None:
    st.title("Shared Image Processing Pipeline")
    st.info(
        "Purpose: acquire an image or video frame, apply grayscale conversion "
        "and Gaussian filtering, segment tracks using thresholding and "
        "morphology, then extract one common contour-based feature vector."
    )

    st.header("1. Image or video-frame acquisition")
    _acquisition_section()
    image = st.session_state.get("input_image")
    if image is None:
        return

    st.subheader("Analysis region")
    profile_label = st.selectbox(
        "Dataset/image layout",
        list(ROI_PROFILE_LABELS),
        key="layout_choice",
        help=(
            "MÃ¼ller images are already cropped. Primary images contain bright "
            "chamber walls, so analysis is restricted to the inner chamber."
        ),
    )
    profile_name = ROI_PROFILE_LABELS[profile_label]
    detected_profile = _detect_layout_profile(image)
    if profile_name == "auto":
        profile_name = detected_profile
        detected_label = (
            "Primary dataset / full chamber"
            if profile_name == "primary_full_chamber"
            else "External Muller / already cropped"
        )
        st.info(f"Automatically detected layout: **{detected_label}**")
    elif profile_name != detected_profile:
        expected = (
            "Primary dataset / full chamber"
            if detected_profile == "primary_full_chamber"
            else "External Muller / already cropped"
        )
        st.error(
            f"The selected profile conflicts with the image shape. Use "
            f"**{expected}**, or select automatic detection. Processing has "
            "been stopped to prevent an invalid segmentation result."
        )
        return
    selected_profile = config["segmentation"]["roi_profiles"][profile_name]
    roi_margins = {
        side: selected_profile[side]
        for side in ("left", "right", "top", "bottom")
    }
    # The image sources have different noise and framing. Keep the same
    # algorithm but apply the validation-selected parameters for that source.
    segmentation_settings = {
        **config["segmentation"],
        **{
            name: value
            for name, value in selected_profile.items()
            if name not in roi_margins
        },
    }
    st.success(
        f"Active profile: {profile_name} | "
        f"threshold offset={segmentation_settings['threshold_offset']}, "
        f"closing={segmentation_settings['closing_kernel']}x"
        f"{segmentation_settings['closing_kernel']}, "
        f"minimum area={segmentation_settings['minimum_object_area']} px^2, "
        f"minimum length={segmentation_settings['minimum_major_axis']} px."
    )

    st.header("2. Grayscale conversion and Gaussian filtering")
    enhancement = enhance_image(image, config["enhancement"])
    columns = st.columns(3)
    columns[0].image(_bgr_to_rgb(image), caption="Original input")
    columns[1].image(enhancement.grey, caption="Grayscale")
    columns[2].image(enhancement.denoised, caption="Gaussian filtered")

    st.header("3. Thresholding, morphology and contour detection")
    segmentation = segment_tracks(
        enhancement.enhanced,
        segmentation_settings,
        roi_margins,
    )
    intermediate = segmentation.intermediate_images
    columns = st.columns(4)
    columns[0].image(
        intermediate["local_bright_tracks"],
        caption="Morphological white top-hat: local bright tracks",
    )
    columns[1].image(
        intermediate["threshold"],
        caption=(
            f"Otsu + {segmentation.parameters['threshold_offset']:.0f} "
            f"(applied value {segmentation.parameters['applied_threshold']:.0f})"
        ),
    )
    columns[2].image(
        intermediate["morphological_closing"],
        caption="Closing first: reconnect short track gaps",
    )
    columns[3].image(
        segmentation.binary_mask,
        caption="Accepted regions after contour size/shape filtering",
    )
    st.caption(
        "Morphological opening is disabled to preserve very thin tracks. "
        "Small dots are rejected afterward using area, length and elongation."
    )
    st.caption(
        f"Small-blob filter: {segmentation.parameters['candidate_count_before_filter']} "
        f"initial contours, {segmentation.parameters['rejected_candidate_count']} "
        f"rejected, {segmentation.parameters['candidate_count_after_filter']} accepted."
    )
    with st.expander("Inspect rejected small/compact regions"):
        st.image(
            intermediate["rejected_small_blobs"],
            caption=(
                "White regions were rejected before feature extraction and "
                "are therefore not sent to the classifiers."
            ),
        )
        st.json(
            {
                "general-track minimum area (px^2)": segmentation.parameters[
                    "minimum_area"
                ],
                "minimum major axis (px)": segmentation.parameters[
                    "minimum_major_axis"
                ],
                "thin-track minimum area (px^2)": segmentation.parameters[
                    "minimum_thin_area"
                ],
                "thin-track minimum perimeter (px)": segmentation.parameters[
                    "minimum_thin_perimeter"
                ],
                "thin-track minimum major axis (px)": segmentation.parameters[
                    "minimum_thin_major_axis"
                ],
                "thin-track minimum aspect ratio": segmentation.parameters[
                    "minimum_thin_aspect_ratio"
                ],
            }
        )
    with st.expander("Inspect selected region of interest"):
        roi = segmentation.parameters["roi_pixels"]
        roi_preview = image.copy()
        cv2.rectangle(
            roi_preview,
            (roi["left"], roi["top"]),
            (roi["right"] - 1, roi["bottom"] - 1),
            (255, 255, 0),
            3,
        )
        st.image(
            _bgr_to_rgb(roi_preview),
            caption=(
                "Only pixels inside the cyan rectangle are eligible for "
                "accepted contours, features and bounding boxes."
            ),
        )

    features = extract_track_features(
        segmentation.binary_mask,
        enhancement.enhanced,
        minimum_area=float(segmentation_settings["minimum_object_area"]),
    )
    st.session_state["pipeline_result"] = {
        "enhancement": enhancement,
        "segmentation": segmentation,
        "features": features,
        "roi_profile": profile_name,
    }

    overlay = image.copy()
    # These boxes come only from the accepted contours used to create the
    # clean mask above. Rejected and out-of-ROI contours cannot appear here.
    for x, y, width, height in segmentation.bounding_boxes:
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
    st.image(_bgr_to_rgb(overlay), caption="Detected particle-track contours")

    st.header("4. Contour-based feature extraction")
    st.caption(
        "These identical numerical features are the controlled input for all "
        "five team-member classifiers."
    )
    rows = [_feature_row(item) for item in features]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No contour passed the configured minimum-area filter.")


def _acquisition_section() -> None:
    source_type = st.radio("Input type", ["Image", "Video"], horizontal=True)
    if source_type == "Image":
        upload = st.file_uploader(
            "Upload a raw cloud-chamber image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
        )
        if upload is not None:
            image = _decode_uploaded_image(upload.getvalue())
            _set_input(image, upload.name, "Uploaded image")
            st.success("Image loaded.")
        return

    upload = st.file_uploader(
        "Upload a cloud-chamber video", type=["mp4", "avi", "mov"]
    )
    if upload is None:
        st.caption("Select one video frame; that frame is analysed as an image.")
        return

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
        frame_number = st.slider("Frame number", 0, frame_count - 1, 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        capture.release()
        if not success:
            st.error("The selected frame could not be captured.")
            return
        timestamp = frame_number / fps if fps > 0 else 0.0
        st.image(
            _bgr_to_rgb(frame),
            caption=f"Frame {frame_number} ({timestamp:.2f} seconds)",
        )
        if st.button("Use this frame for processing", type="primary"):
            name = f"{Path(upload.name).stem}_frame_{frame_number:06d}"
            _set_input(frame, name, f"{upload.name}, frame {frame_number}")
            st.success("Video frame loaded.")
    finally:
        video_path.unlink(missing_ok=True)


def _model_page(short_name: str, full_name: str) -> None:
    st.title(f"{short_name} Classifier")
    st.info(
        f"Purpose: team-member workspace for the {full_name}. This model must "
        "use the shared dataset splits and shared processing pipeline."
    )
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
    if st.button("Classify and create MLP report", type="primary"):
        st.session_state["mlp_predictions"] = predict_tracks(
            load_model(model_path), result["features"]
        )

    predictions = st.session_state.get("mlp_predictions")
    if predictions is None:
        return
    if not predictions:
        st.warning("No segmented track is available for classification.")
        return

    overlay, report_rows = build_visual_report(
        st.session_state["input_image"],
        result["features"],
        predictions,
        confidence_threshold,
    )
    st.subheader("Particle classification report")
    st.image(
        _bgr_to_rgb(overlay),
        caption=(
            "MLP particle predictions. Yellow annotations are below the "
            "selected confidence threshold."
        ),
    )
    st.dataframe(report_rows, use_container_width=True, hide_index=True)
    st.caption(
        "The label is the model's most likely class for each segmented "
        "contour. It is a prediction, not direct physical confirmation."
    )

    safe_name = Path(st.session_state["input_name"]).stem
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download annotated image",
        data=encode_report_png(overlay),
        file_name=f"{safe_name}_mlp_report.png",
        mime="image/png",
    )
    downloads[1].download_button(
        "Download classification CSV",
        data=encode_report_csv(report_rows),
        file_name=f"{safe_name}_mlp_report.csv",
        mime="text/csv",
    )


def _comparison_page() -> None:
    st.title("Final Model Comparison")
    st.info(
        "Purpose: compare CNN, SVM, Decision Tree, MLP and Extra Trees using "
        "the same final-test split and the same metrics."
    )
    st.markdown(
        "Report per-class precision, recall and F1-score, macro F1-score, "
        "confusion matrix and processing time. Use validation results for "
        "model selection; use the final-test split only once after all models "
        "and parameters are fixed."
    )
    report_path = Path("models/mlp_training_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result = report["final_test"]
        st.subheader("Available final-test results")
        st.dataframe(
            [
                {
                    "Model": "MLP",
                    "Accuracy": result["accuracy"],
                    "Balanced accuracy": result["balanced_accuracy"],
                    "Macro F1": result["macro_f1"],
                    "Weighted F1": result["weighted_f1"],
                    "Mean time/track (ms)": result[
                        "mean_inference_ms_per_track"
                    ],
                }
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "MLP confusion matrix: rows are true classes and columns are "
            "predicted classes, in the displayed class order."
        )
        st.write(result["class_names"])
        st.dataframe(result["confusion_matrix"], use_container_width=True)
    else:
        st.warning("The MLP report is not available. Run `python scripts/train_mlp.py`.")

    st.info(
        "Add the other four models' final-test reports here only after their "
        "development and validation choices are fixed."
    )


def _feature_row(item) -> dict:
    return {
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


def _set_input(image: np.ndarray, name: str, description: str) -> None:
    st.session_state["input_image"] = image
    st.session_state["input_name"] = name
    st.session_state["source_description"] = description
    st.session_state["pipeline_result"] = None
    st.session_state["mlp_predictions"] = None


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

