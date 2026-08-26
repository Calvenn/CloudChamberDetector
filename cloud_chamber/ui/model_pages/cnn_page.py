from __future__ import annotations

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
    st.subheader("CNN track-patch evidence")
    st.caption(
        "Each segmented track is cropped, masked and resized to 128 × 128 pixels "
        "before the CNN produces its class probabilities."
    )
    st.image(context.bgr_to_rgb(overlay))
    st.markdown(
        "**Legend:** 🟩 Alpha · 🟦 Electron/Positron · 🟥 Proton · "
        "🟪 V-track · Uncertain predictions retain their class colour · "
        "Grey dashed: review segmentation"
    )
    _render_patch_atlas(
        result["input_image"],
        result["segmentation"].binary_mask,
        result["features"],
        predictions,
    )
    decision_rows = _build_cnn_decision_rows(predictions, report_rows)
    _render_probability_heatmap(predictions, model_bundle["classes"])
    _render_decision_margin(decision_rows)
    st.markdown("**CNN track decisions**")
    st.dataframe(
        decision_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent"
            ),
            "Decision margin": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent"
            ),
        },
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


def _render_patch_atlas(
    image: np.ndarray,
    binary_mask: np.ndarray,
    features: list,
    predictions: list[dict],
) -> None:
    """Display the exact masked 128 x 128 track patches supplied to the CNN."""
    st.markdown("**CNN input-patch atlas**")
    if not features:
        return
    columns = st.columns(4)
    for index, (track, prediction) in enumerate(zip(features, predictions, strict=True)):
        patch = crop_track_patch(image, track.bounding_box, binary_mask)
        with columns[index % len(columns)]:
            st.image(
                patch,
                clamp=True,
                caption=(
                    f"T{track.track_id} · {prediction['particle_type']}\n"
                    f"{float(prediction['confidence']):.0%} confidence"
                ),
                use_container_width=True,
            )


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
                "Decision margin": float(winner_probability - runner_up_probability),
                "Status": report_row.get("Reporting decision", report_row["Status"]),
                "Response time (ms)": float(prediction["inference_time_ms"]),
            }
        )
    return rows


def _render_probability_heatmap(
    predictions: list[dict], class_names: list[str]) -> None:
    """Visualise the complete CNN softmax distribution for every track."""
    st.markdown("**CNN probability map**")
    display_classes = [DISPLAY_NAMES.get(name, name) for name in class_names]
    probabilities = np.asarray(
        [
            [float(prediction["probabilities"].get(name, 0.0)) for name in class_names]
            for prediction in predictions
        ],
        dtype=float,
    )
    figure = go.Figure(
        go.Heatmap(
            z=probabilities,
            x=display_classes,
            y=[f"T{prediction['track_id']}" for prediction in predictions],
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            text=[[f"{value:.0%}" for value in row] for row in probabilities],
            texttemplate="%{text}",
            hovertemplate="Track: %{y}<br>Class: %{x}<br>Probability: %{z:.2%}<extra></extra>",
            colorbar={"title": "Probability"},
        )
    )
    figure.update_layout(
        height=max(300, 60 * len(predictions) + 130),
        xaxis_title="Particle class",
        yaxis_title="Detected track",
        margin={"l": 10, "r": 10, "t": 20, "b": 30},
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_decision_margin(rows: list[dict]) -> None:
    """Show how strongly the CNN separates its first and second choices."""
    st.markdown("**Top-two decision margin**")
    palette = {
        "Alpha": "#00B83F",
        "Electron/Positron": "#1976D2",
        "Proton": "#E53935",
        "V-track": "#9C27B0",
    }
    figure = go.Figure(
        go.Bar(
            x=[row["Decision margin"] for row in rows],
            y=[row["Track"] for row in rows],
            orientation="h",
            marker_color=[palette.get(row["Predicted particle"], "#808080") for row in rows],
            customdata=[(row["Predicted particle"], row["Runner-up"]) for row in rows],
            text=[f"{row['Decision margin']:.0%}" for row in rows],
            textposition="outside",
            hovertemplate=(
                "Track: %{y}<br>Predicted: %{customdata[0]}<br>"
                "Runner-up: %{customdata[1]}<br>Margin: %{x:.2%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis={"title": "Top probability minus runner-up probability", "range": [0, 1.08], "tickformat": ".0%"},
        yaxis={"autorange": "reversed", "title": "Track"},
        height=max(280, 48 * len(rows) + 100),
        margin={"l": 10, "r": 35, "t": 20, "b": 45},
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_batch_summary() -> None:
    """Show image and video-frame classification summaries for this session."""
    rows = list(st.session_state.get("batch_reports", {}).values())
    if rows:
        st.subheader("Batch and video-frame trace")
        figure = go.Figure()
        figure.add_bar(
            x=[row["Input"] for row in rows],
            y=[row["Confident"] for row in rows],
            name="Confident tracks",
            marker_color="#2E7D32",
        )
        figure.add_bar(
            x=[row["Input"] for row in rows],
            y=[row["Uncertain"] for row in rows],
            name="Uncertain tracks",
            marker_color="#F9A825",
        )
        figure.update_layout(
            title="Track-reporting status by image or video frame",
            barmode="stack",
            xaxis_title="Input / frame",
            yaxis_title="Detected tracks",
            margin={"l": 10, "r": 10, "t": 55, "b": 80},
        )
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(rows, use_container_width=True, hide_index=True)
