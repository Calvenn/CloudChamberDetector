"""University-level Streamlit GUI for the Mode A comparative study."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cloud_chamber.config import load_config
from cloud_chamber.detectors import create_detector, list_detectors
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import extract_track_features
from cloud_chamber.models import DetectionResult
from cloud_chamber.validation import validate_detection_result


TECHNIQUE_PAGES = {
    "Classical Thresholding": {
        "purpose": (
            "This page compares thresholding methods that separate bright "
            "particle tracks from the darker chamber background."
        ),
        "methods": "Otsu, Triangle, Yen and Niblack thresholding",
        "registry_names": ["otsu_baseline", "thresholding"],
    },
    "Edge-Based Detection": {
        "purpose": (
            "This page studies intensity changes along particle-track "
            "boundaries and evaluates whether edges can localise faint tracks."
        ),
        "methods": "Sobel, Laplacian of Gaussian and Canny",
        "registry_names": ["edge_based"],
    },
    "Morphological Detection": {
        "purpose": (
            "This page studies shape-based operations for removing noise, "
            "joining broken tracks and isolating track regions."
        ),
        "methods": (
            "Distance-transform watershed, directional openings and "
            "morphological reconstruction"
        ),
        "registry_names": ["morphological"],
    },
    "Contour / Shape Analysis": {
        "purpose": (
            "This page measures detected track geometry, including area, "
            "length, width, orientation and shape."
        ),
        "methods": (
            "Geometric contour descriptors, Hu moments and skeleton analysis"
        ),
        "registry_names": ["contour_shape"],
    },
    "Hough Transform": {
        "purpose": (
            "This page evaluates line-based detection for straight or "
            "approximately straight particle trajectories."
        ),
        "methods": (
            "Standard Hough, progressive probabilistic Hough and "
            "randomised Hough"
        ),
        "registry_names": ["hough"],
    },
}

PAGES = [
    "Project Overview",
    "Analysis Workspace",
    *TECHNIQUE_PAGES,
]


def main() -> None:
    st.set_page_config(
        page_title="Cloud Chamber Track Comparison",
        page_icon="☁️",
        layout="wide",
    )
    config = load_config()
    _initialise_state()

    st.sidebar.title("Cloud Chamber")
    st.sidebar.caption("BMDS2133 Mode A comparative study")
    page = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.divider()
    _input_status()

    if page == "Project Overview":
        _overview_page()
    elif page == "Analysis Workspace":
        _analysis_workspace(config)
    elif page in TECHNIQUE_PAGES:
        _technique_page(page, TECHNIQUE_PAGES[page], config)


def _initialise_state() -> None:
    st.session_state.setdefault("input_image", None)
    st.session_state.setdefault("input_name", None)
    st.session_state.setdefault("source_description", None)
    st.session_state.setdefault("background_reference", None)


def _overview_page() -> None:
    st.title("Cloud Chamber Particle Track Comparison")
    st.info(
        "Purpose: provide one interface for comparing five classical "
        "image-processing categories using the same input and shared "
        "enhancement pipeline."
    )
    st.markdown(
        """
        This is a Mode A comparative system. Each technique is executed
        independently. The system records each output, candidate count and
        processing time for a consistent comparison.

        **Common workflow**

        1. Acquire one image or select one frame from a video.
        2. Apply the fixed shared enhancement.
        3. Run each detection technique independently.
        4. Extract common geometric features.
        5. Compare every technique using the same evaluation criteria.
        6. Recommend the technique with the strongest justified result.

        The GUI is a control and presentation layer. Individual algorithms
        remain inside `cloud_chamber/detectors/`.
        """
    )

    available = list_detectors()
    st.subheader("Registered detector implementations")
    if available:
        st.write(", ".join(available))
    else:
        st.warning("No detectors have been registered yet.")


def _analysis_workspace(config: dict) -> None:
    st.title("Analysis Workspace")
    st.info(
        "Purpose: acquire one cloud-chamber image or video frame, inspect the "
        "shared enhancement, and compare every completed detection technique "
        "using the same prepared input."
    )

    st.header("1. Image / Video Acquisition")
    _acquisition_page(embedded=True)

    if st.session_state.get("input_image") is None:
        return

    st.divider()
    st.header("2. Shared Image Enhancement")
    _enhancement_page(config, embedded=True)

    st.divider()
    st.header("3. Final Technique Comparison")
    _comparison_page(config, embedded=True)


def _acquisition_page(embedded: bool = False) -> None:
    if not embedded:
        st.title("Image / Video Acquisition")
    st.info(
        "Purpose: select a consistent image input or capture a reproducible "
        "frame from a cloud-chamber video before enhancement and detection."
    )
    source_type = st.radio(
        "Input type",
        ["Image", "Video"],
        horizontal=True,
    )

    if source_type == "Image":
        upload = st.file_uploader(
            "Upload a raw cloud-chamber image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
        )
        reference_upload = st.file_uploader(
            "Optional background reference image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help=(
                "Use a same-sized image of the chamber without the target "
                "track. Leave empty for ordinary single-image analysis."
            ),
        )
        if upload is not None:
            image = _decode_uploaded_image(upload.getvalue())
            reference = (
                _decode_uploaded_image(reference_upload.getvalue())
                if reference_upload is not None
                else None
            )
            _set_input(
                image,
                upload.name,
                "Uploaded image",
                background_reference=reference,
            )
            st.success("Image loaded for all technique pages.")
            st.image(_bgr_to_rgb(image), caption=upload.name)
        return

    upload = st.file_uploader(
        "Upload a cloud-chamber video",
        type=["mp4", "avi", "mov"],
    )
    if upload is None:
        st.caption(
            "The selected frame becomes a normal image input. The original "
            "video is not passed directly to the detection algorithms."
        )
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
            min_value=0,
            max_value=frame_count - 1,
            value=min(max(int(round(fps)), 1), frame_count - 1),
        )
        maximum_gap = max(min(frame_count - 1, 120), 1)
        default_gap = min(max(int(round(fps)), 1), maximum_gap)
        reference_gap = st.slider(
            "Background reference gap (frames)",
            min_value=1,
            max_value=maximum_gap,
            value=default_gap,
            help=(
                "The system compares the selected frame with an earlier frame "
                "to highlight moving or newly appearing tracks."
            ),
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        if not success:
            capture.release()
            st.error("The selected video frame could not be captured.")
            return

        reference_number = max(frame_number - reference_gap, 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, reference_number)
        reference_success, reference_frame = capture.read()
        capture.release()
        if not reference_success or reference_number == frame_number:
            reference_frame = None

        timestamp = frame_number / fps if fps > 0 else 0.0
        preview_columns = st.columns(2)
        preview_columns[0].image(
            _bgr_to_rgb(frame),
            caption=f"Selected frame {frame_number} ({timestamp:.2f} seconds)",
        )
        if reference_frame is not None:
            preview_columns[1].image(
                _bgr_to_rgb(reference_frame),
                caption=f"Reference frame {reference_number}",
            )
        if st.button("Use this frame for analysis", type="primary"):
            frame_name = f"{Path(upload.name).stem}_frame_{frame_number:06d}"
            _set_input(
                frame,
                frame_name,
                f"{upload.name}, frame {frame_number}",
                background_reference=reference_frame,
            )
            st.success("Frame captured for all technique pages.")
    finally:
        video_path.unlink(missing_ok=True)


def _enhancement_page(config: dict, embedded: bool = False) -> None:
    if not embedded:
        st.title("Shared Image Enhancement")
    st.info(
        "Purpose: apply exactly the same greyscale conversion, Gaussian noise "
        "reduction, CLAHE contrast enhancement and optional background "
        "subtraction before every individual detection technique."
    )
    image = _require_input()
    if image is None:
        return

    result = _enhance_current_input(image, config)
    columns = st.columns(2)
    columns[0].image(result.grey, caption="1. Greyscale")
    columns[1].image(result.denoised, caption="2. Gaussian filtered")
    st.image(result.contrast_enhanced, caption="3. CLAHE enhanced")
    if result.background_subtraction_applied:
        st.image(
            result.background_corrected,
            caption="4. Background-subtracted detector input",
        )
        st.caption(
            "A reference frame was supplied, so this fourth result is passed "
            "to every detector."
        )
    else:
        st.caption(
            "No background reference was supplied. The CLAHE image above is "
            "passed directly to every detector."
        )
    st.caption(
        "These parameters are shared and should remain fixed during the final "
        "comparison unless the entire team repeats all experiments."
    )
    st.json(config["enhancement"])

def _technique_page(title: str, details: dict, config: dict) -> None:
    st.title(title)
    st.info(f"Purpose: {details['purpose']}")
    st.markdown(f"**Algorithms documented for comparison:** {details['methods']}.")
    image = _require_input()
    if image is None:
        return

    available = set(list_detectors())
    compatible = [
        name for name in details["registry_names"] if name in available
    ]
    if not compatible:
        st.warning(
            "The GUI page is ready, but the assigned member has not registered "
            "a detector for this category. Implement the detector, decorate it "
            "with `@register_detector(\"registry_name\")`, and add that name "
            "to this page's `registry_names` list."
        )
        _show_enhanced_input(image, config)
        return

    selected = st.selectbox("Technique implementation", compatible)
    if st.button(f"Run {title}", type="primary"):
        with st.spinner("Processing the selected image..."):
            enhancement = _enhance_current_input(image, config)
            detection = create_detector(selected).detect(enhancement.enhanced)
            validate_detection_result(detection, enhancement.enhanced.shape)
            features = extract_track_features(
                detection.binary_mask,
                enhancement.enhanced,
                minimum_area=float(config["baseline"]["minimum_object_area"]),
            )
        st.session_state.setdefault("technique_results", {})[selected] = detection
        _display_detection(image, enhancement.enhanced, detection, features)


def _comparison_page(config: dict, embedded: bool = False) -> None:
    if not embedded:
        st.title("Final Comparison")
    st.info(
        "Purpose: run all completed techniques on the same enhanced input, "
        "compare their quantitative results, and identify the most suitable "
        "method for the final recommended pipeline."
    )
    image = _require_input()
    if image is None:
        return

    registered = list_detectors()
    if not registered:
        st.warning("No completed detector is registered.")
        return

    if st.button("Run all registered techniques", type="primary"):
        enhancement = _enhance_current_input(image, config)
        rows = []
        detections: dict[str, DetectionResult] = {}

        for name in registered:
            detector = create_detector(name)
            detection = detector.detect(enhancement.enhanced)
            validate_detection_result(detection, enhancement.enhanced.shape)
            detections[name] = detection
            row = {
                "Registry name": name,
                "Method": detection.method_name,
                "Time (ms)": round(detection.processing_time_ms, 3),
                "Candidates": len(detection.bounding_boxes),
            }
            rows.append(row)

        st.session_state["technique_results"] = detections
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.info(
            "This single-image view compares outputs, candidate counts and "
            "processing time. Select the final method only after evaluating "
            "the supplied bounding-box labels over the validation dataset."
        )

        st.subheader("Detection masks")
        columns = st.columns(min(len(detections), 3))
        for index, (name, detection) in enumerate(detections.items()):
            columns[index % len(columns)].image(
                detection.binary_mask,
                caption=f"{name}: {detection.method_name}",
            )


def _display_detection(
    original: np.ndarray,
    enhanced: np.ndarray,
    detection: DetectionResult,
    features: list,
) -> None:
    overlay = original.copy()
    for x, y, width, height in detection.bounding_boxes:
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)

    columns = st.columns(3)
    columns[0].image(_bgr_to_rgb(original), caption="Original input")
    columns[1].image(enhanced, caption="Shared enhanced input")
    columns[2].image(detection.binary_mask, caption="Detection mask")
    st.image(_bgr_to_rgb(overlay), caption="Detected candidate bounding boxes")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Method", detection.method_name)
    metric_columns[1].metric("Candidates", len(features))
    metric_columns[2].metric(
        "Processing time",
        f"{detection.processing_time_ms:.2f} ms",
    )
    if features:
        st.dataframe(
            [
                {
                    "Track": item.track_id,
                    "Area (px)": round(item.area_pixels, 2),
                    "Major axis (px)": round(item.major_axis_pixels, 2),
                    "Mean width (px)": round(item.mean_width_pixels, 2),
                    "Orientation (degrees)": round(
                        item.orientation_degrees, 2
                    ),
                    "Mean intensity": round(item.mean_intensity, 2),
                }
                for item in features
            ],
            use_container_width=True,
            hide_index=True,
        )


def _show_enhanced_input(image: np.ndarray, config: dict) -> None:
    enhancement = _enhance_current_input(image, config)
    columns = st.columns(2)
    columns[0].image(_bgr_to_rgb(image), caption="Original")
    columns[1].image(
        enhancement.enhanced,
        caption="Input prepared for this technique",
    )


def _require_input() -> np.ndarray | None:
    image = st.session_state.get("input_image")
    if image is None:
        st.warning(
            "No input is selected. Open the Analysis Workspace and load an "
            "image or capture a video frame first."
        )
        return None
    st.caption(f"Current input: {st.session_state['source_description']}")
    return image


def _input_status() -> None:
    if st.session_state.get("input_image") is None:
        st.sidebar.warning("No image selected")
    else:
        st.sidebar.success(f"Input: {st.session_state['input_name']}")


def _set_input(
    image: np.ndarray,
    name: str,
    source_description: str,
    background_reference: np.ndarray | None = None,
) -> None:
    st.session_state["input_image"] = image
    st.session_state["input_name"] = name
    st.session_state["source_description"] = source_description
    st.session_state["background_reference"] = background_reference
    st.session_state["technique_results"] = {}


def _enhance_current_input(image: np.ndarray, config: dict):
    """Run shared enhancement with the session's optional reference frame."""
    return enhance_image(
        image,
        config["enhancement"],
        st.session_state.get("background_reference"),
    )


def _decode_uploaded_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded data is not a readable image")
    return image


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
