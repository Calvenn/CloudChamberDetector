from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from cloud_chamber.ml.member_models.cnn import (
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
from cloud_chamber.ui.model_pages.context import PageContext


MODEL_PATH = Path("models/cnn_classifier.pth")
TRAIN_COMMAND = "python -m cloud_chamber.ml.member_models.cnn"


def render(config: dict, context: PageContext) -> None:
    """Render the CNN panel in the same shared flow as the other model pages."""
    st.title("CNN Classifier")

    model_key = "cnn"
    state_key = f"{model_key}_predictions"
    quality_state_key = f"{model_key}_quality"
    batch_state_key = f"{model_key}_batch_results"

    st.subheader("Run the trained CNN")
    result = st.session_state.get("pipeline_result")
    if result is None:
        st.warning(
            "Process an image or video frame on the Shared Processing "
            "Pipeline page first."
        )
        return

    if not MODEL_PATH.exists():
        st.warning(f"Train the model first: `{TRAIN_COMMAND}`")
        return

    model_bundle = load_model(MODEL_PATH)
    confidence_threshold = st.slider(
        "Confidence reporting threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        key="cnn_confidence_threshold",
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
        f"Classify all {len(samples)} inputs and create CNN reports"
        if len(samples) > 1
        else "Classify and create CNN report"
    )

    if st.button(button_label, type="primary"):
        batch_results = {}
        progress = st.progress(0.0, text="Processing batch...")
        for index, sample in enumerate(samples):
            sample_result = context.process_pipeline_image(
                sample["image"],
                config,
                st.session_state.get("calibration_settings"),
            )
            sample_predictions = predict_tracks(
                model_bundle,
                sample_result["input_image"],
                sample_result["features"],
                sample_result["segmentation"].binary_mask,
            )
            sample_quality = assess_all_contours(
                sample_result["features"],
                sample_result["enhancement"].enhanced,
                sample_result["segmentation"].binary_mask,
                sample_result["segmentation"].parameters,
            )
            batch_results[sample["name"]] = {
                "image": sample["image"],
                "analysis_image": sample_result["input_image"],
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
        st.session_state[batch_state_key] = batch_results

    batch_results = st.session_state.get(batch_state_key, {})
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
                ) + sum(float(item["inference_time_ms"]) for item in item_predictions),
            )
            st.session_state["batch_reports"][name] = {
                "Input": name,
                "Source": entry["description"],
                "Tracks": item_summary["detected_contours"],
                "Dominant prediction": item_summary["dominant_prediction"],
                "Confident": item_summary["confident_classifications"],
                "Uncertain": item_summary["uncertain_classifications"],
                "Alpha": item_summary["class_counts"]["Alpha"],
                "Electron/Positron": item_summary["class_counts"]["Electron/Positron"],
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
                        image=entry["analysis_image"],
                        features=item_result["features"],
                        predictions=item_predictions,
                        confidence_threshold=confidence_threshold,
                    )
                    st.image(context.bgr_to_rgb(item_overlay), width=700)
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
        st.session_state[state_key] = selected_entry["predictions"]
        st.session_state[quality_state_key] = selected_entry["quality"]
        result = selected_entry["result"]

    predictions = st.session_state.get(state_key)
    if predictions is None:
        return
    if not predictions:
        st.warning("No segmented track is available for classification.")
        return

    quality_assessments = st.session_state.get(quality_state_key)
    if quality_assessments is None:
        quality_assessments = assess_all_contours(
            result["features"],
            result["enhancement"].enhanced,
            result["segmentation"].binary_mask,
            result["segmentation"].parameters,
        )
        st.session_state[quality_state_key] = quality_assessments

    overlay, report_rows = build_visual_report(
        image=result["input_image"],
        features=result["features"],
        predictions=predictions,
        confidence_threshold=confidence_threshold,
    )

    for row, track, prediction, quality in zip(
        report_rows,
        result["features"],
        predictions,
        quality_assessments,
        strict=True,
    ):
        row["Reporting decision"] = reporting_status(
            confidence=float(prediction["confidence"]),
            confidence_threshold=confidence_threshold,
            quality_score=int(quality["score"]),
        )
        if result["centimetres_per_pixel"] is not None:
            calibrated = context.feature_row(
                track, result["centimetres_per_pixel"]
            )
            row.update(
                {
                    name: value
                    for name, value in calibrated.items()
                    if name.endswith("(cm)") or name.endswith("(cm²)")
                }
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
        model_path=str(MODEL_PATH),
        model_classes=model_bundle["classes"],
        segmentation_parameters=result["segmentation"].parameters,
    )
    metadata["model_version"] = (
        f"{MODEL_PATH.stat().st_size}-{MODEL_PATH.stat().st_mtime_ns}"
    )
    metadata["calibration"] = {
        "enabled": result["centimetres_per_pixel"] is not None,
        "centimetres_per_pixel": result["centimetres_per_pixel"],
        "perspective_rectified": result["rectified"],
    }
    metadata["spatial_scaling"] = result["spatial_scaling"]
    metadata["rectification"] = result["rectification"]

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
    st.image(context.bgr_to_rgb(overlay))
    st.markdown(
        "**Legend:** 🟧 Alpha · 🟦 Electron/Positron · 🟩 Proton · "
        "🟨 Uncertain"
    )
    st.dataframe(report_rows, use_container_width=True, hide_index=True)

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

    safe_name = Path(st.session_state["input_name"]).stem
    st.subheader("Download reproducible report")
    downloads = st.columns(3)
    downloads[0].download_button(
        "Download annotated image",
        data=encode_report_png(overlay),
        file_name=f"{safe_name}_cnn_report.png",
        mime="image/png",
    )
    downloads[1].download_button(
        "Particle CSV",
        data=encode_report_csv(report_rows),
        file_name=f"{safe_name}_cnn_report.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "PDF summary",
        data=encode_pdf_report(overlay, {"metadata": metadata, "summary": summary}),
        file_name=f"{safe_name}_cnn_report.pdf",
        mime="application/pdf",
    )
