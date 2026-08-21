"""Streamlit GUI for the corrected Mode A cloud-chamber pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cloud_chamber.config import load_config
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import extract_track_features
from cloud_chamber.ml.member_models.extra_trees import (
    EXTRA_TREES_CLASS_COLOURS,
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
    st.session_state.setdefault("extra_trees_quality", None)
    st.session_state.setdefault("batch_reports", {})
    st.session_state.setdefault("mlp_batch_results", {})
    st.session_state.setdefault("extra_trees_batch_results", {})
    st.session_state.setdefault("extra_trees_batch_reports", {})


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
    st.session_state["extra_trees_quality"] = None


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
