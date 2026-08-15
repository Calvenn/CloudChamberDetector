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
from cloud_chamber.ml.member_models.extra_trees import (
    build_visual_report as build_extra_trees_visual_report,
    encode_report_csv as encode_extra_trees_report_csv,
    encode_report_png as encode_extra_trees_report_png,
    load_model as load_extra_trees_model,
    predict_tracks as predict_extra_trees_tracks,
    summarise_predictions as summarise_extra_trees_predictions,
)
from cloud_chamber.ml.member_models.mlp import (
    DISPLAY_NAMES,
    build_visual_report,
    encode_report_csv,
    encode_report_png,
    load_model,
    predict_tracks,
)
from cloud_chamber.reporting import (
    assess_all_contours,
    build_summary,
    encode_pdf_report,
    make_traceability_metadata,
    reporting_status,
)
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
    elif page == "Extra Trees":
        _extra_trees_page()
    elif page == "Final Model Comparison":
        _comparison_page()
    else:
        _model_page(page, MODEL_PAGES[page], config)


def _initialise_state() -> None:
    st.session_state.setdefault("input_batch", [])
    st.session_state.setdefault("input_image", None)
    st.session_state.setdefault("input_name", None)
    st.session_state.setdefault("source_description", None)
    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("mlp_predictions", None)
    st.session_state.setdefault("extra_trees_predictions", None)
    st.session_state.setdefault(
        "layout_choice", "Auto-detect from image shape (recommended)"
    )
    if st.session_state["layout_choice"] not in ROI_PROFILE_LABELS:
        # Replace an obsolete selection retained by an older Streamlit session.
        st.session_state["layout_choice"] = (
            "Auto-detect from image shape (recommended)"
        )
    st.session_state.setdefault("mlp_quality", None)
    st.session_state.setdefault("batch_reports", {})
    st.session_state.setdefault("mlp_batch_results", {})


def _shared_pipeline_page(config: dict) -> None:
    st.title("Shared Image Processing Pipeline")

    st.header("Image or video-frame acquisition")
    _acquisition_section(config)
    image = st.session_state.get("input_image")
    if image is None:
        return

    result = _process_pipeline_image(image, config)
    enhancement = result["enhancement"]
    segmentation = result["segmentation"]
    features = result["features"]
    st.header("Grayscale conversion and Gaussian filtering")
    columns = st.columns(3)
    columns[0].image(_bgr_to_rgb(image), caption="Original input")
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
    st.session_state["mlp_quality"] = None
    st.session_state["batch_reports"].pop(
        st.session_state.get("input_name"), None
    )

    overlay = image.copy()
    # These boxes come only from the accepted contours used to create the
    # clean mask above. Rejected and out-of-ROI contours cannot appear here.
    for x, y, width, height in segmentation.bounding_boxes:
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
    st.image(_bgr_to_rgb(overlay), caption="Detected particle-track contours")

    st.header("Contour-based feature extraction")
    rows = [_feature_row(item) for item in features]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No contour passed the configured minimum-area filter.")


def _process_pipeline_image(image: np.ndarray, config: dict) -> dict:
    """Run the identical shared pipeline for one image without drawing UI."""
    profile_name = _detect_layout_profile(image)
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
    enhancement = enhance_image(image, config["enhancement"])
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
        "enhancement": enhancement,
        "segmentation": segmentation,
        "features": features,
        "roi_profile": profile_name,
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
        if uploads and st.button("Load image batch", type="primary"):
            samples = []
            failures = []
            for upload in uploads:
                try:
                    samples.append(
                        {
                            "image": _decode_uploaded_image(upload.getvalue()),
                            "name": upload.name,
                            "description": f"Uploaded image: {upload.name}",
                        }
                    )
                except ValueError:
                    failures.append(upload.name)
            _replace_input_batch(samples)
            st.success(f"Loaded {len(samples)} image(s).")
            if failures:
                st.warning("Unreadable files skipped: " + ", ".join(failures))
    else:
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


def _replace_input_batch(samples: list[dict]) -> None:
    """Replace the acquisition queue and activate its first valid sample."""
    st.session_state["input_batch"] = samples
    st.session_state["mlp_batch_results"] = {}
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


def _model_page(short_name: str, full_name: str, config: dict) -> None:
    st.title(f"{short_name} Classifier")
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
        "**Legend:** 🟧 Alpha · 🟦 Electron/Positron · 🟩 Proton · "
        "🟪 V-track · 🟨 Uncertain · Grey dashed: review segmentation"
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
        with st.expander(title, expanded=False):
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
        "Extra Trees classifies each detected particle track as "
        "**Alpha**, **Electron/Positron**, or **Proton**."
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

    st.header("1. Model Status")

    if not model_path.exists():

        st.warning(
            "Extra Trees has not been trained yet."
        )

        st.code(
            "python scripts/train_extra_trees.py"
        )

        return


    st.success(
        "Extra Trees model is trained and ready."
    )


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


        st.header("2. Model Performance")


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

            st.success(
                "Selected Extra Trees configuration: "
                f"**{selected_candidate['name']}**"
            )


        st.caption(
            "Validation Macro F1 is used to choose the best Extra Trees "
            "configuration. The final-test results show the performance "
            "of the selected model on unseen test data."
        )


        # =================================================
        # MODEL TUNING EXTRA EFFORT
        # =================================================

        with st.expander(
            "View Model Tuning and Explanation"
        ):

            st.subheader(
                "Extra Trees Candidate Comparison"
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
                                else candidate["max_depth"]
                            ),

                        "Min Leaf":
                            candidate["min_samples_leaf"],

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


            st.dataframe(
                tuning_rows,
                use_container_width=True,
                hide_index=True,
            )


            st.caption(
                "The configuration with the highest Validation Macro F1 "
                "is automatically selected."
            )


            # ---------------------------------------------
            # Explain Model A
            # ---------------------------------------------

            st.subheader(
                "What Does Each Model Mean?"
            )


            st.markdown(
                """
### Model A - Baseline

This is the **basic Extra Trees configuration** used as the starting
point for comparison.

- **100 trees**
- No class balancing
- Unlimited tree depth
- Minimum leaf size of 1

It provides a baseline result before additional tuning is applied.


### Model B - Balanced 300

This configuration increases the number of trees and introduces
**class balancing**.

- **300 trees**
- Balanced class weights
- Unlimited tree depth
- Minimum leaf size of 1

Class balancing gives more importance to particle classes with fewer
training samples.


### Model C - Balanced 500 Depth 20

This configuration uses more trees while limiting how deep each tree
can grow.

- **500 trees**
- Balanced class weights
- Maximum tree depth of **20**
- Minimum leaf size of 1

The depth limit controls model complexity and may help reduce
overfitting.


### Model D - Balanced 500 Leaf 2

This configuration adds another restriction to make individual trees
more conservative.

- **500 trees**
- Balanced class weights
- Maximum tree depth of **20**
- Minimum leaf size of **2**

A larger minimum leaf size prevents a tree from creating leaves based
on only one training sample and may help reduce overfitting.
"""
            )


            # ---------------------------------------------
            # Explain parameters
            # ---------------------------------------------

            with st.expander(
                "What Do the Extra Trees Settings Mean?"
            ):

                st.markdown(
                    """
**Number of Trees**  
Controls how many decision trees are created. More trees can provide
more stable predictions but require more computation.

**Class Weight**  
Controls the importance given to each particle class. Balanced class
weights give additional importance to classes with fewer training
samples.

**Maximum Depth**  
Controls how deep an individual tree can grow. Limiting the depth can
reduce model complexity.

**Minimum Leaf Size**  
Controls the minimum number of training samples required in a final
tree leaf. A larger value makes the tree more conservative.

**Validation Macro F1**  
Measures classification performance while giving equal importance to
each particle class. The candidate with the highest Validation Macro F1
is selected.
"""
                )


    # =====================================================
    # 3. CLASSIFY CURRENT IMAGE
    # =====================================================

    st.header(
        "3. Classify Current Image"
    )


    result = st.session_state.get(
        "pipeline_result"
    )

    image = st.session_state.get(
        "input_image"
    )


    if result is None or image is None:

        st.info(
            "Go to **Shared Processing Pipeline**, upload an image "
            "and process it first."
        )

        return


    features = result["features"]


    if not features:

        st.warning(
            "No particle tracks were detected in this image."
        )

        return


    st.write(
        f"Detected tracks: **{len(features)}**"
    )


    if st.button(
        "Classify Tracks",
        type="primary",
    ):

        model = load_extra_trees_model(
            model_path
        )


        predictions = predict_extra_trees_tracks(
            model,
            features,
        )


        st.session_state[
            "extra_trees_predictions"
        ] = predictions


    predictions = st.session_state.get(
        "extra_trees_predictions"
    )


    if predictions is None:

        st.caption(
            "Click **Classify Tracks** to start Extra Trees classification."
        )

        return


    if not predictions:

        st.warning(
            "No segmented track is available for classification."
        )

        return


    # Fixed confidence threshold
    confidence_threshold = 0.60


    classified_image, report_rows = (
        build_extra_trees_visual_report(
            image,
            features,
            predictions,
            confidence_threshold,
        )
    )


    # =====================================================
    # 4. CLASSIFICATION RESULT
    # =====================================================

    st.header(
        "4. Classification Result"
    )


    summary = summarise_extra_trees_predictions(
        predictions,
        confidence_threshold,
    )


    # -----------------------------------------------------
    # Particle counts
    # -----------------------------------------------------

    count_columns = st.columns(4)


    count_columns[0].metric(
        "Alpha",
        summary["Alpha"],
    )


    count_columns[1].metric(
        "Electron / Positron",
        summary["Electron/Positron"],
    )


    count_columns[2].metric(
        "Proton",
        summary["Proton"],
    )


    count_columns[3].metric(
        "Uncertain",
        summary["Uncertain"],
    )


    # -----------------------------------------------------
    # Classified image
    # -----------------------------------------------------

    st.image(
        _bgr_to_rgb(
            classified_image
        ),
        caption=(
            "Extra Trees prediction for each detected particle track."
        ),
    )


    # -----------------------------------------------------
    # Simple results
    # -----------------------------------------------------

    st.subheader(
        "Track Predictions"
    )


    simple_rows = []


    for row in report_rows:

        simple_rows.append(
            {
                "Track":
                    row["Track"],

                "Prediction":
                    row["Particle type"],

                "Confidence":
                    f"{row['Confidence']:.1%}",

                "Status":
                    row["Status"],
            }
        )


    st.dataframe(
        simple_rows,
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        "Confidence shows how strongly Extra Trees supports its "
        "prediction. Predictions below 60% confidence are marked "
        "as Uncertain."
    )


    # =====================================================
    # 5. ADVANCED DETAILS
    # =====================================================

    with st.expander(
        "Advanced Extra Trees Details"
    ):

        # -------------------------------------------------
        # Per-class prediction probabilities
        # -------------------------------------------------

        st.subheader(
            "Prediction Probabilities"
        )

        st.write(
            "Extra Trees calculates a probability for every particle "
            "class. The class with the highest probability becomes "
            "the prediction."
        )


        probability_rows = []


        for prediction in predictions:

            row = {
                "Track":
                    prediction["track_id"],

                "Prediction":
                    prediction["particle_type"],

                "Confidence":
                    f"{prediction['confidence']:.1%}",
            }


            for class_name, probability in (
                prediction["probabilities"].items()
            ):

                row[
                    f"P({class_name})"
                ] = f"{probability:.1%}"


            probability_rows.append(
                row
            )


        st.dataframe(
            probability_rows,
            use_container_width=True,
            hide_index=True,
        )


        # -------------------------------------------------
        # Inference time
        # -------------------------------------------------

        st.subheader(
            "Extra Trees Classification Time"
        )


        average_time = sum(
            prediction["inference_time_ms"]
            for prediction in predictions
        ) / len(predictions)


        time_columns = st.columns(2)


        time_columns[0].metric(
            "Tracks Classified",
            len(predictions),
        )


        time_columns[1].metric(
            "Average Time per Track",
            f"{average_time:.4f} ms",
        )


        st.caption(
            "This measures the Extra Trees prediction stage only."
        )


        # -------------------------------------------------
        # Training information
        # -------------------------------------------------

        if training_report is not None:

            # ---------------------------------------------
            # Feature importance
            # ---------------------------------------------

            st.subheader(
                "Feature Importance"
            )


            st.write(
                "Feature importance shows which track measurements "
                "contributed more strongly to the Extra Trees model."
            )


            feature_rows = []


            for item in training_report.get(
                "feature_importance",
                [],
            ):

                feature_rows.append(
                    {
                        "Feature":
                            item["feature"],

                        "Importance":
                            round(
                                item["importance"],
                                4,
                            ),
                    }
                )


            st.dataframe(
                feature_rows,
                use_container_width=True,
                hide_index=True,
            )


            # ---------------------------------------------
            # Confusion matrix
            # ---------------------------------------------

            st.subheader(
                "Final-Test Confusion Matrix"
            )


            final_test = training_report[
                "final_test"
            ]


            st.caption(
                "Rows represent the correct particle classes. "
                "Columns represent the Extra Trees predictions."
            )


            st.write(
                "Class order:",
                final_test[
                    "class_names"
                ],
            )


            st.dataframe(
                final_test[
                    "confusion_matrix"
                ],
                use_container_width=True,
            )


            # ---------------------------------------------
            # Per-class results
            # ---------------------------------------------

            st.subheader(
                "Per-Class Performance"
            )


            classification_report = (
                final_test[
                    "classification_report"
                ]
            )


            class_rows = []


            for class_name in final_test[
                "class_names"
            ]:

                class_result = (
                    classification_report[
                        class_name
                    ]
                )


                class_rows.append(
                    {
                        "Class":
                            class_name,

                        "Precision":
                            round(
                                class_result[
                                    "precision"
                                ],
                                3,
                            ),

                        "Recall":
                            round(
                                class_result[
                                    "recall"
                                ],
                                3,
                            ),

                        "F1 Score":
                            round(
                                class_result[
                                    "f1-score"
                                ],
                                3,
                            ),

                        "Support":
                            int(
                                class_result[
                                    "support"
                                ]
                            ),
                    }
                )


            st.dataframe(
                class_rows,
                use_container_width=True,
                hide_index=True,
            )


    # =====================================================
    # 6. DOWNLOAD RESULTS
    # =====================================================

    with st.expander(
        "Download Results"
    ):

        safe_name = Path(
            st.session_state[
                "input_name"
            ]
        ).stem


        download_columns = (
            st.columns(2)
        )


        download_columns[0].download_button(
            "Download Classified Image",
            data=encode_extra_trees_report_png(
                classified_image
            ),
            file_name=(
                f"{safe_name}"
                "_extra_trees_report.png"
            ),
            mime="image/png",
        )


        download_columns[1].download_button(
            "Download Result CSV",
            data=encode_extra_trees_report_csv(
                report_rows
            ),
            file_name=(
                f"{safe_name}"
                "_extra_trees_report.csv"
            ),
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
    st.session_state["extra_trees_predictions"] = None
    st.session_state["mlp_quality"] = None


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
