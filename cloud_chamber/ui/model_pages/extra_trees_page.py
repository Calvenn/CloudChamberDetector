"""Extremely Randomized Trees Streamlit page."""
import json
import io
from pathlib import Path
import zipfile
import plotly.graph_objects as go
import streamlit as st
from cloud_chamber.calibration import detect_chamber_corners
from cloud_chamber.ml.member_models.extra_trees import (build_visual_report as build_extra_trees_visual_report, encode_report_csv as encode_extra_trees_report_csv, encode_report_png as encode_extra_trees_report_png, load_model as load_extra_trees_model, predict_tracks as predict_extra_trees_tracks, summarise_predictions as summarise_extra_trees_predictions)
from cloud_chamber.reporting import (
    assess_all_contours,
    build_summary,
    encode_batch_pdf_report,
    encode_json_report,
    encode_pdf_report,
    make_traceability_metadata,
    reporting_status,
)
from .context import PageContext

def render(config: dict, context: PageContext) -> None:
    """Beginner-friendly Extremely Randomized Trees classification page."""

    # =====================================================
    # PAGE TITLE
    # =====================================================

    st.title("Extremely Randomized Trees Classifier")

    st.write(
        "Extremely Randomized Trees classifies each detected particle track as "
        "**Alpha**, **Electron/Positron**, **Proton**, or **V-track** using "
        "the contours produced by the Shared Processing Pipeline."
    )

    project_root = Path(__file__).resolve().parents[3]

    model_path = (
        project_root
        / "models"
        / "extra_trees_classifier.joblib"
    )

    report_path = (
        project_root
        / "models"
        / "extra_trees_hybrid_training_report.json"
    )


    # =====================================================
    # 1. MODEL STATUS
    # =====================================================

    if not model_path.exists():

        st.warning(
            "Extremely Randomized Trees has not been trained yet."
        )

        st.code(
            "python scripts/train_extra_trees.py"
        )

        return


    st.caption(
        "Model ready"
        + (" · Saved evaluation report available" if report_path.exists() else "")
    )


    # =====================================================
    # 2. MODEL PERFORMANCE
    # =====================================================

    training_report = None

    if report_path.exists():

        training_report = _normalise_training_report(json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        ))

        validation = training_report["validation"]

        final_test = training_report["final_test"]

        selected_candidate = training_report.get(
            "selected_candidate"
        )


        st.subheader("Saved Model Evaluation")


        st.markdown("**Extremely Randomized Trees — Overall Performance**")
        processing_time = final_test.get("mean_inference_ms_per_track", 0.0)
        overall_rows = [
            {
                "Model Configuration": "Selected Extremely Randomized Trees",
                "Accuracy": f"{final_test['accuracy']:.2%}",
                "Precision": f"{final_test['macro_precision']:.2%}",
                "Recall": f"{final_test['macro_recall']:.2%}",
                "F1-score": f"{final_test['macro_f1']:.2%}",
                "Processing Time": f"{processing_time:.4f} ms/track",
            }
        ]
        st.dataframe(overall_rows, use_container_width=True, hide_index=True)

        per_class_container = st.expander(
            "View Per-Class Model Performance", expanded=False
        )
        class_labels = {
            "alpha": "Alpha",
            "electron_positron": "Electron",
            "proton": "Proton",
            "v_track": "V-track",
        }
        confusion_matrix = final_test["confusion_matrix"]
        sample_count = final_test["sample_count"]
        classification_report = final_test["classification_report"]
        class_rows = []
        for class_index, class_name in enumerate(final_test["class_names"]):
            true_positive = confusion_matrix[class_index][class_index]
            false_positive = sum(
                row[class_index]
                for row_index, row in enumerate(confusion_matrix)
                if row_index != class_index
            )
            false_negative = (
                sum(confusion_matrix[class_index]) - true_positive
            )
            true_negative = (
                sample_count - true_positive - false_positive - false_negative
            )
            class_result = classification_report[class_name]
            class_rows.append(
                {
                    "Predicted": class_labels.get(class_name, class_name),
                    "Accuracy": (
                        f"{(true_positive + true_negative) / sample_count:.2%}"
                    ),
                    "Precision": f"{class_result['precision']:.2%}",
                    "Recall": f"{class_result['recall']:.2%}",
                    "F1-score": f"{class_result['f1-score']:.2%}",
                    "Processing Time": f"{processing_time:.4f} ms/track",
                }
            )
        per_class_container.dataframe(
            class_rows, use_container_width=True, hide_index=True
        )
        per_class_container.caption(
            "Overall precision, recall and F1-score are macro averages. "
            "Per-class accuracy uses one-versus-rest calculation. The same "
            "overall mean processing time is shown for each class because "
            "timing was measured per track, not separately by class."
        )


        if selected_candidate:

            st.caption(
                "Selected Extremely Randomized Trees configuration: "
                f"**{selected_candidate['name']}**"
            )


        st.caption(
            f"Validation Macro F1: {validation['macro_f1']:.3f}. It is used "
            "to choose the best Extremely Randomized Trees "
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
                "Extremely Randomized Trees Candidate Comparison"
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

                        "Min Samples Split":
                            candidate.get("min_samples_split", 2),

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
            # Explain parameters
            # ---------------------------------------------

            with st.expander(
                "What Do the Extremely Randomized Trees Settings Mean?"
            ):

                st.markdown(
                    """
**Number of Trees**  
Controls how many decision trees are created. More trees can provide
more stable predictions but require more computation.

**Maximum Depth**  
Controls how deep an individual tree can grow. Limiting the depth can
reduce model complexity.

**Minimum Samples Split**
Controls the minimum number of samples required to divide a tree node.
A larger value makes each tree more conservative.

**Validation Macro F1**  
Measures classification performance while giving equal importance to
each particle class. The candidate with the highest Validation Macro F1
is selected.

All candidates use balanced-subsample class weights and a minimum leaf
size of 1.
"""
                )

        with st.expander(
            "View Feature Importance and Final-Test Analysis", expanded=False
        ):
            _render_model_level_analysis(training_report, model_path)


    # =====================================================
    # 3. CLASSIFY CURRENT IMAGE
    # =====================================================

    st.header("Classify Segmented Tracks")


    result = st.session_state.get(
        "pipeline_result"
    )

    image = result.get("input_image") if result is not None else None


    if result is None or image is None:

        st.info(
            "Go to **Shared Processing Pipeline**, upload an image or video "
            "and process it first."
        )

        return


    samples = st.session_state.get("input_batch") or [
        {
            "image": st.session_state["input_image"],
            "name": st.session_state["input_name"],
            "description": st.session_state["source_description"],
        }
    ]
    is_video = st.session_state.get("input_source_type") == "Video" or any(
        "_frame_" in sample["name"] for sample in samples
    )
    is_multi_input = len(samples) > 1
    st.write(
        f"Shared pipeline inputs ready for classification: **{len(samples)}**"
    )
    confidence_threshold = 0.60
    button_label = (
        f"Classify all {len(samples)} video frames"
        if is_video and is_multi_input
        else f"Classify all {len(samples)} images"
        if is_multi_input
        else "Classify Segmented Tracks"
    )

    if st.button(button_label, type="primary"):
        model = load_extra_trees_model(model_path)
        batch_results = {}
        pipeline_cache = st.session_state.setdefault(
            "pipeline_results_by_input", {}
        )
        progress = st.progress(
            0.0, text="Processing with Extremely Randomized Trees..."
        )
        for index, sample in enumerate(samples):
            current_result = st.session_state.get("pipeline_result")
            is_current_input = (
                current_result is not None
                and sample["name"] == st.session_state.get("input_name")
            )
            sample_result = pipeline_cache.get(sample["name"])
            if is_current_input:
                sample_result = current_result
            if sample_result is None:
                sample_result = context.process_pipeline_image(
                    sample["image"],
                    config,
                    {
                        "rectification_points": detect_chamber_corners(
                            sample["image"]
                        ).tolist()
                    },
                )
                pipeline_cache[sample["name"]] = sample_result
            sample_predictions = predict_extra_trees_tracks(
                model, sample_result["features"]
            )
            batch_results[sample["name"]] = {
                "image": sample["image"],
                "description": sample["description"],
                "result": sample_result,
                "predictions": sample_predictions,
            }
            progress.progress(
                (index + 1) / len(samples),
                text=f"Processed {index + 1} of {len(samples)} inputs",
            )
        progress.empty()
        st.session_state["extra_trees_batch_results"] = batch_results

    batch_results = st.session_state.get("extra_trees_batch_results", {})
    if batch_results and is_multi_input:
        st.subheader(
            "Video-Frame Results" if is_video else "Batch Image Results"
        )
        item_label = "Frame" if is_video else "Image"
        batch_rows = []
        for index, (name, entry) in enumerate(batch_results.items(), start=1):
            item_result = entry["result"]
            item_predictions = entry["predictions"]
            item_summary = summarise_extra_trees_predictions(
                item_predictions, confidence_threshold
            )
            batch_rows.append(
                {
                    "Input": name,
                    "Tracks": item_summary["Total"],
                    "Alpha": item_summary["Alpha"],
                    "Electron/Positron": item_summary["Electron/Positron"],
                    "Proton": item_summary["Proton"],
                    "V-track": item_summary["V-track"],
                    "Uncertain": item_summary["Uncertain"],
                }
            )
            with st.expander(
                f"{item_label} {index}: {name} — "
                f"{item_summary['Total']} tracks",
                expanded=False,
            ):
                if item_predictions:
                    item_overlay, _ = build_extra_trees_visual_report(
                        item_result.get("original_image", item_result["input_image"]),
                        item_result["features"],
                        item_predictions,
                        confidence_threshold,
                        original_boxes=item_result.get("original_bounding_boxes"),
                        instance_mask=item_result.get(
                            "original_segmentation_mask",
                            item_result["segmentation"].binary_mask,
                        ),
                    )
                    st.image(context.bgr_to_rgb(item_overlay), width=700)
                else:
                    st.warning("No segmented track was available to classify.")
        st.dataframe(batch_rows, use_container_width=True, hide_index=True)
        selected_name = st.selectbox(
            "Choose an input for the detailed report",
            list(batch_results),
            key="extra_trees_selected_batch_input",
        )
        selected_entry = batch_results[selected_name]
        st.session_state["input_image"] = selected_entry["image"]
        st.session_state["input_name"] = selected_name
        st.session_state["source_description"] = selected_entry["description"]
        st.session_state["pipeline_result"] = selected_entry["result"]
        st.session_state["extra_trees_predictions"] = selected_entry[
            "predictions"
        ]
        result = selected_entry["result"]
        image = result["input_image"]
    elif batch_results:
        selected_entry = next(iter(batch_results.values()))
        st.session_state["extra_trees_predictions"] = selected_entry[
            "predictions"
        ]
        result = selected_entry["result"]
        image = result["input_image"]

    features = result["features"]
    predictions = st.session_state.get("extra_trees_predictions")

    prediction_signature = st.session_state.get(
        "extra_trees_prediction_signature"
    )
    if predictions is not None and (
        prediction_signature != feature_signature
        or len(predictions) != len(features)
    ):
        st.session_state["extra_trees_predictions"] = None
        st.session_state["extra_trees_prediction_signature"] = None
        predictions = None
        st.info(
            "The image or segmentation tracks changed. Click "
            "**Classify Tracks** to refresh the Extra Trees predictions."
        )


    if predictions is None:

        st.caption(
            "Click **Classify Tracks** to start Extremely Randomized Trees classification."
        )

        return


    if not predictions:

        st.warning(
            "No segmented track is available for classification."
        )

        return

    quality_assessments = assess_all_contours(
        result["features"],
        result["enhancement"].enhanced,
        result["segmentation"].binary_mask,
        result["segmentation"].parameters,
    )

    classified_image, report_rows = (
        build_extra_trees_visual_report(
            result.get("original_image", image),
            features,
            predictions,
            confidence_threshold,
            quality_assessments=quality_assessments,
            original_boxes=result.get("original_bounding_boxes"),
            instance_mask=result.get(
                "original_segmentation_mask",
                result["segmentation"].binary_mask,
            ),
        )
    )


    # =====================================================
    # 4. CLASSIFICATION RESULT
    # =====================================================

    st.header("Classification Result")


    summary = summarise_extra_trees_predictions(
        predictions,
        confidence_threshold,
    )


    # -----------------------------------------------------
    # Particle counts
    # -----------------------------------------------------

    count_columns = st.columns(6)


    count_columns[0].metric(
        "Alpha",
        summary.get("Alpha", 0),
    )


    count_columns[1].metric(
        "Electron / Positron",
        summary.get("Electron/Positron", 0),
    )


    count_columns[2].metric(
        "Proton",
        summary.get("Proton", 0),
    )


    count_columns[3].metric(
        "V-track",
        summary.get("V-track", 0),
    )


    count_columns[4].metric(
        "Uncertain",
        summary.get("Uncertain", 0),
    )


    count_columns[5].metric(
        "Total",
        summary.get("Total", len(predictions)),
    )

    accepted_count = summary.get("Total", len(predictions)) - summary.get(
        "Uncertain", 0
    )
    if summary.get("Uncertain", 0):
        st.warning(
            f"{accepted_count} track(s) meet the {confidence_threshold:.0%} "
            f"threshold; {summary['Uncertain']} track(s) need review."
        )
    else:
        st.success(
            f"All {accepted_count} track(s) meet the "
            f"{confidence_threshold:.0%} confidence threshold."
        )


    # -----------------------------------------------------
    # Classified image
    # -----------------------------------------------------

    st.image(
        context.bgr_to_rgb(classified_image),
        caption=(
            "Extremely Randomized Trees classification on the Shared Processing Pipeline "
            "segmented contours"
        ),
    )
    st.markdown(
        "**Legend:** 🟩 Alpha · 🟦 Electron/Positron · "
        "🟥 Proton · 🟪 V-track · 🟨 Uncertain · "
        "Grey dashed = Review segmentation"
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

                "_confidence":
                    float(row["Confidence"]),

                "Status":
                    row["Status"],
            }
        )


    statuses = sorted({row["Status"] for row in simple_rows})
    controls = st.columns(2)
    status_filter = controls[0].selectbox(
        "Filter by status",
        ["All statuses", *statuses],
        key="extra_trees_status_filter",
    )
    sort_choice = controls[1].selectbox(
        "Sort track predictions",
        [
            "Track number",
            "Confidence: highest first",
            "Confidence: lowest first",
        ],
        key="extra_trees_prediction_sort",
    )
    visible_rows = [
        row
        for row in simple_rows
        if status_filter == "All statuses" or row["Status"] == status_filter
    ]
    if sort_choice == "Confidence: highest first":
        visible_rows.sort(key=lambda row: row["_confidence"], reverse=True)
    elif sort_choice == "Confidence: lowest first":
        visible_rows.sort(key=lambda row: row["_confidence"])
    else:
        visible_rows.sort(
            key=lambda row: int(str(row["Track"]).lstrip("Tt") or 0)
        )
    display_rows = [
        {key: value for key, value in row.items() if key != "_confidence"}
        for row in visible_rows
    ]

    st.dataframe(
        display_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.TextColumn(
                help="Highest class probability reported by Extremely Randomized Trees"
            ),
        },
    )
    st.caption(f"Showing {len(display_rows)} of {len(simple_rows)} tracks.")

    st.caption(
        "Confidence shows how strongly Extremely Randomized Trees supports its "
        f"prediction. Predictions below {confidence_threshold:.0%} confidence "
        "are marked as Uncertain."
    )

    _render_output_charts(summary, predictions, report_rows, confidence_threshold)


    # =====================================================
    # 5. ADVANCED DETAILS
    # =====================================================

    with st.expander(
        "Advanced Extremely Randomized Trees Details"
    ):

        # -------------------------------------------------
        # Per-class prediction probabilities
        # -------------------------------------------------

        st.subheader(
            "Prediction Probabilities"
        )

        st.write(
            "Extremely Randomized Trees calculates a probability for every particle "
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
            "Extremely Randomized Trees Classification Time"
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
            "This measures the Extremely Randomized Trees prediction stage only."
        )


    model_bundle = load_extra_trees_model(model_path)
    processing_time_ms = float(result["segmentation"].processing_time_ms) + sum(
        float(item["inference_time_ms"]) for item in predictions
    )
    research_summary = build_summary(
        predictions=predictions,
        quality_assessments=quality_assessments,
        confidence_threshold=confidence_threshold,
        processing_time_ms=processing_time_ms,
    )
    metadata = make_traceability_metadata(
        input_name=st.session_state["input_name"],
        source_description=st.session_state["source_description"],
        roi_profile=result.get("roi_profile", "automatic"),
        confidence_threshold=confidence_threshold,
        model_path=str(model_path),
        model_classes=model_bundle["classes"],
        segmentation_parameters=result["segmentation"].parameters,
    )
    metadata.update(
        {
            "model_name": "Extremely Randomized Trees",
            "model_version": f"{model_path.stat().st_size}-{model_path.stat().st_mtime_ns}",
            "model_configuration": model_bundle.get("selected_parameters", {}),
            "feature_columns": list(model_bundle.get("feature_columns", [])),
            "random_seed": model_bundle.get("random_seed"),
            "spatial_scaling": result.get("spatial_scaling", {}),
            "calibration": {
                "enabled": result.get("centimetres_per_pixel") is not None,
                "centimetres_per_pixel": result.get("centimetres_per_pixel"),
                "perspective_rectified": bool(result.get("rectified", False)),
            },
        }
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
                "features": context.feature_row(
                    track, result.get("centimetres_per_pixel")
                ),
            }
        )
    complete_report = {
        "metadata": metadata,
        "summary": research_summary,
        "tracks": track_reports,
        "interpretation_note": (
            "Model confidence indicates class preference, not correctness."
        ),
    }

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


        download_columns = st.columns(4)


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

        download_columns[2].download_button(
            "Download Research PDF",
            data=encode_pdf_report(classified_image, complete_report),
            file_name=f"{safe_name}_extremely_randomized_trees_report.pdf",
            mime="application/pdf",
        )

        download_columns[3].download_button(
            "Download Reproducibility JSON",
            data=encode_json_report(complete_report),
            file_name=f"{safe_name}_extremely_randomized_trees_report.json",
            mime="application/json",
        )

        if is_multi_input and batch_results:
            st.markdown("**Complete batch/video exports**")
            batch_signature = tuple(batch_results)
            export_package = st.session_state.get(
                "extra_trees_batch_export_package", {}
            )
            if export_package.get("signature") != batch_signature:
                with st.spinner("Preparing all annotated images and PDF pages..."):
                    export_package = {
                        "signature": batch_signature,
                        "zip": _encode_batch_classified_images(
                            batch_results, confidence_threshold
                        ),
                        "pdf": _encode_batch_research_pdf(
                            batch_results,
                            confidence_threshold,
                            metadata,
                            "Video frames" if is_video else "Batch images",
                        ),
                    }
                    st.session_state[
                        "extra_trees_batch_export_package"
                    ] = export_package
            batch_downloads = st.columns(2)
            batch_downloads[0].download_button(
                "Download All Classified Images",
                data=export_package["zip"],
                file_name="extremely_randomized_trees_classified_images.zip",
                mime="application/zip",
            )
            batch_downloads[1].download_button(
                "Download Complete Research PDF",
                data=export_package["pdf"],
                file_name=(
                    "extremely_randomized_trees_video_report.pdf"
                    if is_video
                    else "extremely_randomized_trees_batch_report.pdf"
                ),
                mime="application/pdf",
            )


def _render_model_level_analysis(training_report: dict, model_path: Path) -> None:
    """Show general trained-model evidence separately from live predictions."""
    st.info(
        "This section explains the trained model using labelled test data. "
        "It is not an analysis of the image currently being classified."
    )

    model_bundle = load_extra_trees_model(model_path)
    estimator = model_bundle["model"]
    importance_rows = sorted(
        zip(
            model_bundle.get("feature_columns", []),
            getattr(estimator, "feature_importances_", []),
            strict=True,
        ),
        key=lambda item: float(item[1]),
    )
    if importance_rows:
        st.markdown("**Feature Importance**")
        importance_figure = go.Figure(
            go.Bar(
                x=[float(value) for _, value in importance_rows],
                y=[name for name, _ in importance_rows],
                orientation="h",
                marker_color="#4C78A8",
                text=[f"{float(value):.1%}" for _, value in importance_rows],
                textposition="outside",
            )
        )
        importance_figure.update_layout(
            xaxis_title="Relative importance",
            yaxis_title="Track feature",
            xaxis_tickformat=".0%",
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(importance_figure, use_container_width=True)

    final_test = training_report["final_test"]
    display_names = {
        "alpha": "Alpha",
        "electron_positron": "Electron/Positron",
        "proton": "Proton",
        "v_track": "V-track",
    }
    labels = [
        display_names.get(name, name) for name in final_test["class_names"]
    ]
    matrix = final_test["confusion_matrix"]
    totals = [sum(row) for row in matrix]
    correct = [matrix[index][index] for index in range(len(labels))]
    missed = [totals[index] - correct[index] for index in range(len(labels))]

    st.markdown("**Final-Test Results: How Often Was Each Class Correct?**")
    st.caption(
        "Green represents correctly classified labelled tracks. Red represents "
        "tracks that were assigned to another particle class."
    )
    outcome_figure = go.Figure()
    for name, values, colour in (
        ("Correctly classified", correct, "#2ECC71"),
        ("Misclassified", missed, "#FF6B6B"),
    ):
        outcome_figure.add_bar(
            y=labels,
            x=[value / total for value, total in zip(values, totals, strict=True)],
            name=name,
            orientation="h",
            marker_color=colour,
            text=[
                f"{value}/{total} ({value / total:.1%})"
                for value, total in zip(values, totals, strict=True)
            ],
            textposition="inside",
        )
    outcome_figure.update_layout(
        barmode="stack",
        xaxis={"title": "Percentage of labelled tracks", "tickformat": ".0%"},
        yaxis_title="Actual particle class",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend_orientation="h",
    )
    st.plotly_chart(outcome_figure, use_container_width=True)

    confusion_rows = []
    for actual_index, actual_label in enumerate(labels):
        alternatives = [
            (count, labels[predicted_index])
            for predicted_index, count in enumerate(matrix[actual_index])
            if predicted_index != actual_index
        ]
        confused_count, confused_label = max(alternatives)
        confusion_rows.append(
            {
                "Actual particle": actual_label,
                "Correct": f"{correct[actual_index]}/{totals[actual_index]}",
                "Main incorrect prediction": confused_label if confused_count else "None",
                "Number misclassified this way": confused_count,
            }
        )
    st.markdown("**Most Common Classification Mistakes**")
    st.dataframe(confusion_rows, use_container_width=True, hide_index=True)


def _encode_batch_research_pdf(
    batch_results: dict,
    confidence_threshold: float,
    metadata: dict,
    input_type: str,
) -> bytes:
    """Build a complete readable PDF for all batch images or video frames."""
    entries = []
    for name, entry in batch_results.items():
        result = entry["result"]
        predictions = entry["predictions"]
        quality = assess_all_contours(
            result["features"],
            result["enhancement"].enhanced,
            result["segmentation"].binary_mask,
            result["segmentation"].parameters,
        )
        overlay, _ = build_extra_trees_visual_report(
            result.get("original_image", result["input_image"]),
            result["features"],
            predictions,
            confidence_threshold,
            quality_assessments=quality,
            original_boxes=result.get("original_bounding_boxes"),
            instance_mask=result.get(
                "original_segmentation_mask",
                result["segmentation"].binary_mask,
            ),
        )
        entries.append(
            {
                "name": name,
                "source": entry.get("description", ""),
                "overlay": overlay,
                "predictions": predictions,
                "processing_time_ms": float(
                    result["segmentation"].processing_time_ms
                )
                + sum(float(item["inference_time_ms"]) for item in predictions),
            }
        )
    batch_metadata = {**metadata, "input_type": input_type}
    return encode_batch_pdf_report(entries, batch_metadata)


def _encode_batch_classified_images(
    batch_results: dict,
    confidence_threshold: float,
) -> bytes:
    """Create an in-memory ZIP containing every annotated batch result."""
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for index, (name, entry) in enumerate(batch_results.items(), start=1):
            result = entry["result"]
            overlay, _ = build_extra_trees_visual_report(
                result.get("original_image", result["input_image"]),
                result["features"],
                entry["predictions"],
                confidence_threshold,
                original_boxes=result.get("original_bounding_boxes"),
                instance_mask=result.get(
                    "original_segmentation_mask",
                    result["segmentation"].binary_mask,
                ),
            )
            archive.writestr(
                f"{index:03d}_{Path(name).stem}_classified.png",
                encode_extra_trees_report_png(overlay),
            )
    return archive_buffer.getvalue()


def _render_output_charts(
    summary: dict,
    predictions: list[dict],
    report_rows: list[dict],
    confidence_threshold: float,
) -> None:
    """Visualise current Extremely Randomized Trees predictions."""
    st.subheader("Extremely Randomized Trees Output Visualisations")
    st.caption(
        "These charts summarise predictions for the current input. They do "
        "not represent accuracy because the uploaded input has no labels."
    )
    colours = {
        "Alpha": "#00C800",
        "Electron/Positron": "#28C8FF",
        "Proton": "#FF4B4B",
        "V-track": "#B400B4",
    }
    class_names = list(colours)
    class_counts = [int(summary.get(name, 0)) for name in class_names]
    left, right = st.columns(2)
    with left:
        composition = go.Figure(
            go.Pie(
                labels=class_names,
                values=class_counts,
                hole=0.48,
                marker={"colors": [colours[name] for name in class_names]},
                textinfo="label+value+percent",
                sort=False,
            )
        )
        composition.update_layout(
            title="Predicted particle composition",
            margin={"l": 10, "r": 10, "t": 55, "b": 10},
        )
        st.plotly_chart(composition, use_container_width=True)

    with right:
        status_counts = {
            status: sum(row["Status"] == status for row in report_rows)
            for status in ("Accepted", "Uncertain", "Review segmentation")
        }
        status_figure = go.Figure()
        for status, colour in (
            ("Accepted", "#2ECC71"),
            ("Uncertain", "#FFD700"),
            ("Review segmentation", "#A0A0A0"),
        ):
            if status_counts[status]:
                status_figure.add_bar(
                    y=["Detected tracks"],
                    x=[status_counts[status]],
                    name=status,
                    orientation="h",
                    marker_color=colour,
                    text=[status_counts[status]],
                    textposition="inside",
                )
        status_figure.update_layout(
            title=f"Reporting status at {confidence_threshold:.0%} confidence",
            barmode="stack",
            xaxis_title="Number of tracks",
            margin={"l": 10, "r": 10, "t": 55, "b": 10},
        )
        st.plotly_chart(status_figure, use_container_width=True)

    track_labels = [f"T{item['track_id']}" for item in predictions]
    confidences = [float(item["confidence"]) for item in predictions]
    particle_types = [str(item["particle_type"]) for item in predictions]
    confidence_figure = go.Figure(
        go.Bar(
            x=track_labels,
            y=confidences,
            marker_color=[colours.get(name, "#A0A0A0") for name in particle_types],
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
        yaxis_title="Extremely Randomized Trees confidence",
        yaxis={"range": [0, 1.08], "tickformat": ".0%"},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        showlegend=False,
    )
    st.plotly_chart(confidence_figure, use_container_width=True)


def _normalise_training_report(report: dict) -> dict:
    """Normalise supported Extremely Randomized Trees report schemas."""
    candidates = report.get("candidate_results", [])
    normalised_candidates = []
    for candidate in candidates:
        if "name" in candidate:
            normalised_candidates.append(candidate)
            continue
        parameters = candidate.get("parameters", {})
        validation = candidate.get("validation", {})
        normalised_candidates.append(
            {
                **candidate,
                **parameters,
                "name": (
                    f"Candidate {candidate.get('candidate', '?')}: "
                    f"{parameters.get('n_estimators', '?')} trees, "
                    f"depth={parameters.get('max_depth')}"
                ),
                "min_samples_leaf": parameters.get("min_samples_leaf", 1),
                "min_samples_split": parameters.get("min_samples_split", 2),
                "class_weight": parameters.get(
                    "class_weight", "balanced_subsample"
                ),
                "validation_macro_f1": validation.get("macro_f1", 0.0),
            }
        )
    report["candidate_results"] = normalised_candidates

    selected = report.get("selected_candidate")
    if not isinstance(selected, dict):
        selected = next(
            (
                candidate
                for candidate in normalised_candidates
                if candidate.get("candidate") == selected
            ),
            None,
        )
        report["selected_candidate"] = selected
    if "validation" not in report and selected is not None:
        report["validation"] = selected.get("validation", {})
    return report
