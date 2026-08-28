"""Streamlit dashboard for CNN evidence and live track classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from cloud_chamber.ml.member_models.cnn import (
    DISPLAY_NAMES,
    build_visual_report,
    crop_track_patch,
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
TRAINING_REPORT_PATH = Path("models/cnn_training_report.json")
TRAIN_COMMAND = "python -m cloud_chamber.ml.member_models.cnn"


def render(config: dict, context: PageContext) -> None:
    """Render the CNN panel in the same shared flow as the other model pages."""
    st.title("CNN Classifier")

    model_key = "cnn"
    state_key = f"{model_key}_predictions"
    quality_state_key = f"{model_key}_quality"
    batch_state_key = f"{model_key}_batch_results"

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
            current_result = st.session_state.get("pipeline_result")
            is_current_single_input = (
                len(samples) == 1
                and current_result is not None
                and sample["name"] == st.session_state.get("input_name")
            )
            sample_result = (
                current_result if is_current_single_input else context.process_pipeline_image(
                    sample["image"], config, st.session_state.get("calibration_settings")
                )
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
                        quality_assessments=item_quality,
                        instance_mask=item_result["segmentation"].binary_mask,
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
        image=result.get("original_image", result["input_image"]),
        features=result["features"],
        predictions=predictions,
        confidence_threshold=confidence_threshold,
        quality_assessments=quality_assessments,
        original_boxes=result.get("original_bounding_boxes"),
        instance_mask=result.get(
            "original_segmentation_mask", result["segmentation"].binary_mask
        ),
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
    track_reports = [
        {
            "track_id": track.track_id,
            "prediction": prediction,
            "quality": quality,
            "reporting_status": reporting_status(
                confidence=float(prediction["confidence"]),
                confidence_threshold=confidence_threshold,
                quality_score=int(quality["score"]),
            ),
            "features": context.feature_row(track, result["centimetres_per_pixel"]),
        }
        for track, prediction, quality in zip(
            result["features"], predictions, quality_assessments, strict=True
        )
    ]

    _render_final_test_evaluation()

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

    _render_particle_composition_chart(summary)
    st.subheader("CNN track-patch evidence")
    st.caption(
        "Each segmented track is cropped, masked and resized to 128 × 128 pixels "
        "before the CNN produces its class probabilities."
    )
    st.image(context.bgr_to_rgb(overlay))
    st.markdown(
        "**Legend:** 🟩 Alpha · 🟦 Electron/Positron · 🟥 Proton · "
        "🟪 V-track · Uncertain predictions retain their class colour"
    )
    decision_rows = _build_cnn_decision_rows(predictions, report_rows)
    _render_track_details(
        result["input_image"],
        result["segmentation"].binary_mask,
        result["features"],
        predictions,
        decision_rows,
        quality_assessments,
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

    complete_report = {
        "metadata": metadata,
        "summary": summary,
        "tracks": track_reports,
        "interpretation_note": (
            "Model confidence estimates class preference. Contour quality is "
            "an explainable heuristic and is not a correctness probability."
        ),
    }

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
        data=encode_pdf_report(overlay, complete_report),
        file_name=f"{safe_name}_cnn_report.pdf",
        mime="application/pdf",
    )


def _render_track_details(
    image: np.ndarray,
    binary_mask: np.ndarray,
    features: list,
    predictions: list[dict],
    decision_rows: list[dict],
    quality_assessments: list[dict],
) -> None:
    """Show each CNN input patch and decision in one collapsible panel."""
    st.subheader("Track-patch details")
    st.caption("Open a track to inspect its CNN input patch and prediction details.")
    for track, prediction, decision, quality in zip(
        features, predictions, decision_rows, quality_assessments, strict=True
    ):
        patch = crop_track_patch(image, track.bounding_box, binary_mask)
        with st.expander(
            f"{decision['Track']} · {decision['Predicted particle']} · "
            f"{decision['Confidence']:.0%} confidence",
            expanded=False,
        ):
            patch_column, detail_column = st.columns((1, 1.45))
            with patch_column:
                st.image(
                    patch,
                    clamp=True,
                    caption="Masked 128 × 128 patch supplied to the CNN",
                    use_container_width=True,
                )
            with detail_column:
                st.markdown("**CNN decision fields**")
                first_row = st.columns(2)
                first_row[0].metric("Predicted class", decision["Predicted particle"])
                first_row[1].metric("Confidence", f"{decision['Confidence']:.1%}")
                st.progress(
                    decision["Confidence"],
                    text=f"Predicted-class probability: {decision['Confidence']:.1%}",
                )
                second_row = st.columns(2)
                second_row[0].metric("Runner-up class", decision["Runner-up"])
                second_row[1].metric(
                    "Runner-up probability",
                    f"{decision['Runner-up probability']:.1%}",
                )
                st.progress(
                    decision["Runner-up probability"],
                    text=(
                        "Runner-up probability: "
                        f"{decision['Runner-up probability']:.1%}"
                    ),
                )
                st.caption(
                    f"Status: {decision['Status']} · "
                    f"Response time: {decision['Response time (ms)']:.2f} ms"
                )
                st.markdown("**Segmentation quality**")
                quality_row = st.columns(2)
                quality_row[0].metric("Contour quality", quality["grade"])
                quality_row[1].metric("Quality score", f"{quality['score']}/100")
                st.caption(f"Local contrast: {quality['local_contrast']:.1f}")
                for warning in quality["warnings"]:
                    st.warning(warning)


def _render_final_test_evaluation() -> None:
    """Show the saved final-test metrics for the trained CNN."""
    if not TRAINING_REPORT_PATH.exists():
        return

    try:
        training_report = json.loads(TRAINING_REPORT_PATH.read_text(encoding="utf-8"))
        metrics = training_report["final_test"]
        classification_report = metrics["classification_report"]
        confusion_matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
        class_names = list(metrics["class_names"])
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        st.info("Saved CNN final-test metrics are unavailable.")
        return

    total = int(confusion_matrix.sum())
    if total == 0 or len(class_names) != len(confusion_matrix):
        return

    mean_time = float(metrics["mean_inference_ms_per_track"])
    weighted = classification_report["weighted avg"]
    rows = [
        {
            "Predicted": "Overall (weighted)",
            "Accuracy": f"{float(metrics['accuracy']):.2%}",
            "Precision": f"{float(weighted['precision']):.2%}",
            "Recall": f"{float(weighted['recall']):.2%}",
            "F1-score": f"{float(weighted['f1-score']):.2%}",
            "Processing time": f"{mean_time:.4f} ms/track",
        }
    ]
    for index, class_name in enumerate(class_names):
        class_metrics = classification_report[class_name]
        true_positive = int(confusion_matrix[index, index])
        false_positive = int(confusion_matrix[:, index].sum()) - true_positive
        false_negative = int(confusion_matrix[index, :].sum()) - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        one_vs_rest_accuracy = (true_positive + true_negative) / total
        rows.append(
            {
                "Predicted": DISPLAY_NAMES.get(class_name, class_name),
                "Accuracy": f"{one_vs_rest_accuracy:.2%}",
                "Precision": f"{float(class_metrics['precision']):.2%}",
                "Recall": f"{float(class_metrics['recall']):.2%}",
                "F1-score": f"{float(class_metrics['f1-score']):.2%}",
                "Processing time": f"{mean_time:.4f} ms/track",
            }
        )

    st.subheader("CNN final-test evaluation")
    st.caption(
        "Metrics were recorded using the saved final-test set. Per-class accuracy "
        "uses one-versus-rest accuracy; processing time is the mean CNN inference "
        "time per track."
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _build_cnn_decision_rows(
    predictions: list[dict], report_rows: list[dict]
) -> list[dict]:
    """Build decision evidence from the CNN's top-two softmax probabilities."""
    rows = []
    for prediction, report_row in zip(predictions, report_rows, strict=True):
        ranked = sorted(
            prediction["probabilities"].items(), key=lambda item: item[1], reverse=True
        )
        winner, winner_probability = ranked[0]
        runner_up, runner_up_probability = ranked[1]
        rows.append(
            {
                "Track": f"T{prediction['track_id']}",
                "Predicted particle": DISPLAY_NAMES.get(winner, winner),
                "Confidence": float(winner_probability),
                "Runner-up": DISPLAY_NAMES.get(runner_up, runner_up),
                "Runner-up probability": float(runner_up_probability),
                "Status": report_row.get("Reporting decision", report_row["Status"]),
                "Response time (ms)": float(prediction["inference_time_ms"]),
            }
        )
    return rows


def _render_particle_composition_chart(summary: dict) -> None:
    """Show the predicted particle mix for the current image or frame."""
    st.markdown("**Predicted particle composition**")
    palette = {
        "Alpha": "#00B83F",
        "Electron/Positron": "#1976D2",
        "Proton": "#E53935",
        "V-track": "#9C27B0",
    }
    names = list(summary["class_counts"])
    figure = go.Figure(
        go.Pie(
            labels=names,
            values=[summary["class_counts"][name] for name in names],
            marker={"colors": [palette[name] for name in names]},
            textinfo="label+value+percent",
            textposition="outside",
            automargin=True,
            sort=False,
        )
    )
    figure.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend_title_text="Particle class",
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_batch_summary() -> None:
    """Show image and video-frame classification summaries for this session."""
    rows = list(st.session_state.get("batch_reports", {}).values())
    if rows:
        st.subheader("Batch and video-frame trace")
        st.caption(
            "Use this table to compare detected tracks, class counts and processing "
            "time across uploaded images or video frames."
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)
