"""Streamlit GUI for the amended cloud-chamber ML project."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cloud_chamber.config import load_config
from cloud_chamber.enhancement import enhance_image


ALGORITHM_PAGES = {
    "Convolutional Neural Network (CNN)": (
        "cloud_chamber/ml/member_models/cnn.py"
    ),
    "Neuro-Explicit Model": (
        "cloud_chamber/ml/member_models/neuro_explicit.py"
    ),
    "YOLOv5": "cloud_chamber/ml/member_models/yolov5.py",
    "Modified U-Net": (
        "cloud_chamber/ml/member_models/modified_unet.py"
    ),
    "Mask R-CNN Model": (
        "cloud_chamber/ml/member_models/mask_rcnn_model.py"
    ),
}

PAGES = [
    "Acquisition, Enhancement and Segmentation",
    *ALGORITHM_PAGES,
    "Final Comparison",
]


def main() -> None:
    st.set_page_config(
        page_title="Cloud Chamber ML Comparison",
        page_icon="☁️",
        layout="wide",
    )
    config = load_config()
    _initialise_state()

    st.sidebar.title("Cloud Chamber")
    st.sidebar.caption("BMDS2133 machine-learning workflow")
    page = st.sidebar.radio("Navigate", PAGES)
    _input_status()

    if page == "Acquisition, Enhancement and Segmentation":
        _shared_pipeline_page(config)
    elif page in ALGORITHM_PAGES:
        _empty_algorithm_page(page, ALGORITHM_PAGES[page])
    else:
        _final_comparison_page()


def _initialise_state() -> None:
    st.session_state.setdefault("input_image", None)
    st.session_state.setdefault("input_name", None)
    st.session_state.setdefault("source_description", None)


def _shared_pipeline_page(config: dict) -> None:
    st.title("Acquisition, Enhancement and Segmentation")
    st.info(
        "This shared page prepares one consistent input before it is passed "
        "to the independently developed machine-learning algorithms."
    )

    st.header("1. Image / Video Acquisition")
    source_type = st.radio("Input type", ["Image", "Video"], horizontal=True)
    if source_type == "Image":
        upload = st.file_uploader(
            "Upload a cloud-chamber image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
        )
        if upload is not None:
            image = _decode_uploaded_image(upload.getvalue())
            _set_input(image, upload.name, "Uploaded image")
    else:
        _video_input()

    image = st.session_state.get("input_image")
    if image is None:
        st.caption("Select an image or video frame to continue.")
        return

    st.divider()
    st.header("2. Shared Image Enhancement")
    result = enhance_image(image, config["enhancement"])
    columns = st.columns(3)
    columns[0].image(_bgr_to_rgb(image), caption="Original input")
    columns[1].image(result.grey, caption="Grayscale conversion")
    columns[2].image(result.denoised, caption="Gaussian filtering")
    st.caption(
        "The Gaussian-filtered grayscale image is the shared input for "
        "segmentation."
    )

    st.divider()
    st.header("3. FCN within Mask R-CNN Segmentation")
    st.image(result.enhanced, caption="Input prepared for segmentation")
    weights = Path(config["segmentation"]["weights"])
    if weights.is_file():
        st.success(f"Segmentation weights found: {weights}")
        st.caption(
            "Inference controls will be enabled after the trained model is "
            "integrated."
        )
    else:
        st.warning(
            f"Segmentation is not trained yet. Expected weights: `{weights}`"
        )
        st.caption(
            "The current dataset contains bounding boxes but no exact instance "
            "masks. Mask R-CNN training will be completed after suitable mask "
            "annotations and the approved external dataset are added."
        )


def _video_input() -> None:
    upload = st.file_uploader(
        "Upload a cloud-chamber video",
        type=["mp4", "avi", "mov"],
    )
    if upload is None:
        return

    suffix = Path(upload.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(upload.getvalue())
        video_path = Path(temporary.name)

    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            st.error("The uploaded video could not be opened.")
            return
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_number = st.slider(
            "Select frame number",
            0,
            frame_count - 1,
            min(max(int(round(fps)), 1), frame_count - 1),
        )
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
        if st.button("Use this frame", type="primary"):
            _set_input(
                frame,
                f"{Path(upload.name).stem}_frame_{frame_number:06d}",
                f"{upload.name}, frame {frame_number}",
            )
    finally:
        video_path.unlink(missing_ok=True)


def _empty_algorithm_page(title: str, implementation_file: str) -> None:
    st.title(title)
    st.caption(f"Member implementation file: `{implementation_file}`")
    st.info(
        "This page is intentionally empty. The assigned member will develop "
        "and connect this algorithm independently."
    )


def _final_comparison_page() -> None:
    st.title("Final Comparison")
    st.info(
        "This page will be completed after all five independent algorithms "
        "produce a common prediction and evaluation result."
    )


def _set_input(image: np.ndarray, name: str, description: str) -> None:
    st.session_state["input_image"] = image
    st.session_state["input_name"] = name
    st.session_state["source_description"] = description


def _input_status() -> None:
    if st.session_state.get("input_image") is None:
        st.sidebar.warning("No image selected")
    else:
        st.sidebar.success(f"Input: {st.session_state['input_name']}")


def _decode_uploaded_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded data is not a readable image")
    return image


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
