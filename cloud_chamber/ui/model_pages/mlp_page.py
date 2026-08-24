"""MLP Streamlit page and report components."""
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import cloud_chamber.ml.member_models.svm as svm_module
from cloud_chamber.ml.member_models.decision_tree import (build_visual_report as decision_tree_build_visual_report, encode_report_csv as decision_tree_encode_report_csv, encode_report_png as decision_tree_encode_report_png, load_model as decision_tree_load_model, predict_tracks as decision_tree_predict_tracks)
from cloud_chamber.ml.member_models.mlp import DISPLAY_NAMES, build_visual_report, encode_report_csv, encode_report_png, load_model, predict_tracks
from cloud_chamber.reporting import assess_all_contours, build_summary, encode_pdf_report, make_traceability_metadata, reporting_status
from .context import PageContext

def render(short_name: str, full_name: str, config: dict, context: PageContext) -> None:
    st.title(f"{short_name} Classifier")
    if short_name != "MLP":
        return

    model_key = {
        "MLP": "mlp",
        "SVM": "svm",
        "Decision Tree": "decision_tree",
    }[short_name]
    state_key = f"{model_key}_predictions"
    quality_state_key = f"{model_key}_quality"
    batch_state_key = f"{model_key}_batch_results"
    model_path = Path("models/mlp_classifier.joblib")
    train_command = "python scripts/train_mlp.py"
    model_loader = load_model
    model_predict = predict_tracks
    model_viz = build_visual_report
    png_encoder = encode_report_png
    csv_encoder = encode_report_csv
    if short_name == "SVM":
        model_path = Path("models/svm_classifier.joblib")
        train_command = "python scripts/train_svm.py"
        model_loader = svm_module.load_model
        model_predict = svm_module.predict_tracks
        model_viz = svm_module.build_visual_report
        png_encoder = svm_module.encode_report_png
        csv_encoder = svm_module.encode_report_csv
    elif short_name == "Decision Tree":
        model_path = Path("models/decision_tree_classifier.joblib")
        train_command = "python scripts/train_decision_tree.py"
        model_loader = decision_tree_load_model
        model_predict = decision_tree_predict_tracks
        model_viz = decision_tree_build_visual_report
        png_encoder = decision_tree_encode_report_png
        csv_encoder = decision_tree_encode_report_csv

    st.subheader(f"Run the trained {short_name}")
    result = st.session_state.get("pipeline_result")
    if result is None:
        st.warning(
            "Process an image or video frame on the Shared Processing "
            "Pipeline page first."
        )
        return
    if not model_path.exists():
        st.warning(f"Train the model first: `{train_command}`")
        return
    model_bundle = model_loader(model_path)
    _render_mlp_model_performance(Path("models/mlp_training_report.json"))
    confidence_threshold = st.slider(
        "Confidence reporting threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        key=f"{model_key}_confidence_threshold",
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
        f"Classify all {len(samples)} inputs and create {short_name} reports"
        if len(samples) > 1
        else f"Classify and create {short_name} report"
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
            if is_current_single_input:
                # Classification must consume the exact mask displayed by the
                # shared pipeline. Reprocessing here previously used different
                # rectification state and could create additional contours.
                sample_result = current_result
            else:
                sample_result = context.process_pipeline_image(
                    sample["image"],
                    config,
                    st.session_state.get("calibration_settings"),
                )
            sample_predictions = model_predict(
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
                    if short_name == "MLP":
                        item_overlay, _ = model_viz(
                            image=entry["analysis_image"],
                            features=item_result["features"],
                            predictions=item_predictions,
                            confidence_threshold=confidence_threshold,
                            quality_assessments=item_quality,
                            instance_mask=item_result["segmentation"].binary_mask,
                        )
                    else:
                        item_overlay, _ = model_viz(
                            entry["analysis_image"],
                            item_result["features"],
                            item_predictions,
                            confidence_threshold,
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

    if short_name == "MLP":
        overlay, report_rows = model_viz(
            image=result.get("original_image", result["input_image"]),
            features=result["features"],
            predictions=predictions,
            confidence_threshold=confidence_threshold,
            quality_assessments=quality_assessments,
            original_boxes=result.get("original_bounding_boxes"),
            instance_mask=result.get(
                "original_segmentation_mask",
                result["segmentation"].binary_mask,
            ),
        )
    else:
        overlay, report_rows = model_viz(
            result.get("original_image", result["input_image"]),
            result["features"],
            predictions,
            confidence_threshold,
            original_boxes=result.get("original_bounding_boxes"),
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
        model_path=str(model_path),
        model_classes=model_bundle["classes"],
        segmentation_parameters=result["segmentation"].parameters,
    )
    metadata["model_version"] = (
        f"{model_path.stat().st_size}-{model_path.stat().st_mtime_ns}"
    )
    metadata["calibration"] = {
        "enabled": result["centimetres_per_pixel"] is not None,
        "centimetres_per_pixel": result["centimetres_per_pixel"],
        "perspective_rectified": result["rectified"],
    }
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
                "features": context.feature_row(
                    track, result["centimetres_per_pixel"]
                ),
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

    _render_mlp_output_charts(
        summary=summary,
        predictions=predictions,
        confidence_threshold=confidence_threshold,
    )

    st.subheader("Annotated classification overview")
    particle_options = [
        DISPLAY_NAMES.get(str(class_name), str(class_name))
        for class_name in model_bundle["classes"]
    ]
    selected_particles = st.multiselect(
        "Show particle types",
        options=particle_options,
        default=particle_options,
        key=f"mlp_particle_filter_{st.session_state['input_name']}",
        help="The filter changes only this visualisation; model predictions remain unchanged.",
    )
    selected_indices = [
        index for index, prediction in enumerate(predictions)
        if prediction["particle_type"] in selected_particles
    ]
    filtered_features = [result["features"][index] for index in selected_indices]
    filtered_predictions = [predictions[index] for index in selected_indices]
    filtered_quality = [quality_assessments[index] for index in selected_indices]
    source_boxes = result.get("original_bounding_boxes")
    filtered_boxes = (
        [source_boxes[index] for index in selected_indices]
        if source_boxes is not None
        else None
    )
    filtered_overlay, filtered_rows = model_viz(
        image=result.get("original_image", result["input_image"]),
        features=filtered_features,
        predictions=filtered_predictions,
        confidence_threshold=confidence_threshold,
        quality_assessments=filtered_quality,
        original_boxes=filtered_boxes,
        instance_mask=result.get(
            "original_segmentation_mask",
            result["segmentation"].binary_mask,
        ),
    )
    st.image(
        context.bgr_to_rgb(filtered_overlay),
    )
    st.caption(
        f"Showing {len(filtered_rows)} of {len(report_rows)} classified tracks."
    )
    st.markdown(
        "**Legend:** 🟧 Alpha · 🟦 Electron/Positron · 🟩 Proton · "
        "🟪 V-track · Uncertain predictions retain their class colour · "
        "Grey dashed: review segmentation"
    )
    compact_rows = [
        {
            "Track": row["Track"],
            "Particle": row["Particle type"],
            "Confidence": row["Confidence"],
            "Decision": row.get("Reporting decision", row["Status"]),
            "Contour quality": row["Contour quality"],
        }
        for row in filtered_rows
    ]
    st.dataframe(
        compact_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="percent"
            )
        },
    )

    _render_particle_evidence(
        context,
        result["segmentation"].binary_mask,
        result["features"],
        predictions,
        quality_assessments,
        confidence_threshold,
        result["centimetres_per_pixel"],
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
        data=png_encoder(overlay),
        file_name=f"{safe_name}_{model_key}_report.png",
        mime="image/png",
    )
    downloads[1].download_button(
        "Particle CSV",
        data=csv_encoder(report_rows),
        file_name=f"{safe_name}_{model_key}_report.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "PDF summary",
        data=encode_pdf_report(overlay, complete_report),
        file_name=f"{safe_name}_{model_key}_report.pdf",
        mime="application/pdf",
    )


def _render_mlp_output_charts(
    summary: dict,
    predictions: list[dict],
    confidence_threshold: float,
) -> None:
    """Visualise complementary MLP output metrics for the current input.

    These charts describe model output only. They must not be interpreted as
    accuracy because the uploaded image does not provide ground-truth labels.
    """
    st.subheader("MLP output visualisations")
    st.caption(
        "These charts summarise the current predictions. Accuracy, precision, "
        "recall and F1-score must be calculated separately on labelled test data."
    )

    display_colours = {
        "Alpha": "#FFA500",
        "Electron/Positron": "#28C8FF",
        "Proton": "#00C800",
        "V-track": "#FF00FF",
    }
    class_names = list(summary["class_counts"])
    class_counts = [summary["class_counts"][name] for name in class_names]

    left, right = st.columns(2)
    with left:
        class_figure = go.Figure(
            go.Pie(
                labels=class_names,
                values=class_counts,
                hole=0.48,
                marker={"colors": [display_colours[name] for name in class_names]},
                textinfo="label+value+percent",
                sort=False,
            )
        )
        class_figure.update_layout(
            title="Predicted particle composition",
            margin={"l": 10, "r": 10, "t": 55, "b": 10},
            legend_title_text="Particle type",
        )
        st.plotly_chart(class_figure, use_container_width=True)

    with right:
        status_figure = go.Figure()
        status_figure.add_bar(
            y=["Detected tracks"],
            x=[summary["confident_classifications"]],
            name="Reliable candidate",
            orientation="h",
            marker_color="#2ECC71",
            text=[summary["confident_classifications"]],
            textposition="inside",
        )
        status_figure.add_bar(
            y=["Detected tracks"],
            x=[summary["uncertain_classifications"]],
            name="Uncertain",
            orientation="h",
            marker_color="#FFD700",
            text=[summary["uncertain_classifications"]],
            textposition="inside",
        )
        status_figure.update_layout(
            title=f"Reporting status at {confidence_threshold:.0%} confidence",
            barmode="stack",
            xaxis_title="Number of tracks",
            margin={"l": 10, "r": 10, "t": 55, "b": 10},
        )
        st.plotly_chart(status_figure, use_container_width=True)

    track_labels = [f"T{index + 1}" for index in range(len(predictions))]
    confidences = [float(item["confidence"]) for item in predictions]
    particle_types = [str(item["particle_type"]) for item in predictions]
    confidence_colours = [
        display_colours.get(particle_type, "#A0A0A0")
        for particle_type in particle_types
    ]
    confidence_figure = go.Figure(
        go.Bar(
            x=track_labels,
            y=confidences,
            marker_color=confidence_colours,
            customdata=particle_types,
            text=[f"{value:.0%}" for value in confidences],
            textposition="outside",
            hovertemplate=(
                "Track: %{x}<br>Prediction: %{customdata}<br>"
                "Confidence: %{y:.1%}<extra></extra>"
            ),
        )
    )
    confidence_figure.add_hline(
        y=confidence_threshold,
        line_dash="dash",
        line_color="#FF4B4B",
        annotation_text=f"Reporting threshold ({confidence_threshold:.0%})",
        annotation_position="top left",
    )
    confidence_figure.update_layout(
        title="Confidence for each detected track",
        xaxis_title="Track ID",
        yaxis_title="MLP confidence",
        yaxis={"range": [0, 1.08], "tickformat": ".0%"},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        showlegend=False,
    )
    st.plotly_chart(confidence_figure, use_container_width=True)


def _render_mlp_model_performance(report_path: Path) -> None:
    """Present labelled final-test performance separately from live predictions."""
    st.subheader("MLP model performance")
    if not report_path.exists():
        st.info("Train the MLP to generate its labelled final-test performance report.")
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        final_test = report["final_test"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        st.warning(f"The MLP training report cannot be read: {error}")
        return

    metrics = st.columns(5)
    metrics[0].metric("Final-test tracks", int(final_test["sample_count"]))
    metrics[1].metric("Accuracy", f"{float(final_test['accuracy']):.3f}")
    metrics[2].metric(
        "Balanced accuracy", f"{float(final_test['balanced_accuracy']):.3f}"
    )
    metrics[3].metric("Macro F1", f"{float(final_test['macro_f1']):.3f}")
    metrics[4].metric("Weighted F1", f"{float(final_test['weighted_f1']):.3f}")
    st.caption(
        "Macro F1 is the main result because each particle class contributes "
        "equally even when the dataset contains unequal class counts."
    )

    overall_figure = go.Figure(
        go.Bar(
            x=["Accuracy", "Balanced accuracy", "Macro F1", "Weighted F1"],
            y=[
                float(final_test["accuracy"]),
                float(final_test["balanced_accuracy"]),
                float(final_test["macro_f1"]),
                float(final_test["weighted_f1"]),
            ],
            text=[
                f"{float(final_test[key]):.3f}"
                for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
            ],
            textposition="outside",
            marker_color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B"],
        )
    )
    overall_figure.update_layout(
        title="Labelled final-test performance",
        yaxis={"title": "Score", "range": [0, 1.08]},
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    st.plotly_chart(overall_figure, use_container_width=True)

    classification = final_test.get("classification_report", {})
    class_rows = []
    for class_name in final_test.get("class_names", []):
        values = classification.get(class_name, {})
        class_rows.append(
            {
                "Particle": DISPLAY_NAMES.get(class_name, class_name),
                "Precision": float(values.get("precision", 0.0)),
                "Recall": float(values.get("recall", 0.0)),
                "F1": float(values.get("f1-score", 0.0)),
                "Test tracks": int(values.get("support", 0)),
            }
        )
    if class_rows:
        st.markdown("**Performance by particle class**")
        st.dataframe(
            class_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.3f"
                ),
            },
        )


def _render_particle_evidence(
    context: PageContext,
    binary_mask: np.ndarray,
    features: list,
    predictions: list[dict],
    quality_assessments: list[dict],
    confidence_threshold: float,
    centimetres_per_pixel: float | None,
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
                [context.feature_row(track, centimetres_per_pixel)],
                use_container_width=True,
                hide_index=True,
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
