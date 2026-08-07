"""Streamlit GUI for the corrected Mode A cloud-chamber pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cloud_chamber.config import load_config
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import extract_track_features
from cloud_chamber.segmentation import segment_tracks


MODEL_PAGES = {
    "CNN": "Convolutional Neural Network",
    "SVM": "Support Vector Machine",
    "Decision Tree": "Decision Tree",
    "MLP": "Multilayer Perceptron",
    "Extra Trees": "Extremely Randomised Trees",
}
PAGES = ["Shared Processing Pipeline", *MODEL_PAGES, "Final Model Comparison"]


def main() -> None:
    st.set_page_config(
        page_title="Cloud Chamber Particle Classification",
        page_icon="☁️",
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

    st.header("2. Grayscale conversion and Gaussian filtering")
    enhancement = enhance_image(image, config["enhancement"])
    columns = st.columns(3)
    columns[0].image(_bgr_to_rgb(image), caption="Original input")
    columns[1].image(enhancement.grey, caption="Grayscale")
    columns[2].image(enhancement.denoised, caption="Gaussian filtered")

    st.header("3. Thresholding, morphology and contour detection")
    segmentation = segment_tracks(enhancement.enhanced, config["segmentation"])
    intermediate = segmentation.intermediate_images
    columns = st.columns(3)
    columns[0].image(intermediate["threshold"], caption="Otsu threshold")
    columns[1].image(
        intermediate["morphological_opening"], caption="Morphological opening"
    )
    columns[2].image(
        segmentation.binary_mask,
        caption="Closing + filtered contour regions",
    )

    features = extract_track_features(
        segmentation.binary_mask,
        enhancement.enhanced,
        minimum_area=float(config["segmentation"]["minimum_object_area"]),
    )
    st.session_state["pipeline_result"] = {
        "enhancement": enhancement,
        "segmentation": segmentation,
        "features": features,
    }

    overlay = image.copy()
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
    st.warning("Comparison becomes available after members connect all models.")


def _feature_row(item) -> dict:
    return {
        "Track": item.track_id,
        "Area (px²)": round(item.area_pixels, 3),
        "Perimeter (px)": round(item.perimeter_pixels, 3),
        "Length (px)": round(item.major_axis_pixels, 3),
        "Width (px)": round(item.mean_width_pixels, 3),
        "Aspect ratio": round(item.aspect_ratio, 3),
        "Solidity": round(item.solidity, 3),
        "Rectangularity": round(item.rectangularity, 3),
        "Thickness (px)": round(item.thickness_pixels, 3),
        "Orientation (°)": round(item.orientation_degrees, 3),
        "Mean intensity": round(item.mean_intensity, 3),
    }


def _set_input(image: np.ndarray, name: str, description: str) -> None:
    st.session_state["input_image"] = image
    st.session_state["input_name"] = name
    st.session_state["source_description"] = description
    st.session_state["pipeline_result"] = None


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
