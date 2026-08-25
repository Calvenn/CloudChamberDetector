"""Extra Trees Streamlit page."""
import json
from pathlib import Path
import streamlit as st
from cloud_chamber.ml.member_models.extra_trees import (build_visual_report as build_extra_trees_visual_report, encode_report_csv as encode_extra_trees_report_csv, encode_report_png as encode_extra_trees_report_png, load_model as load_extra_trees_model, predict_tracks as predict_extra_trees_tracks, summarise_predictions as summarise_extra_trees_predictions)
from .context import PageContext

def render(context: PageContext) -> None:
    """Beginner-friendly Extra Trees classification page."""

    # =====================================================
    # PAGE TITLE
    # =====================================================

    st.title("Extra Trees Classifier")

    st.write(
        "Extra Trees classifies each detected particle track as "
        "**Alpha**, **Electron/Positron**, or **Proton**."
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

    image = result.get("input_image") if result is not None else None


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
        context.bgr_to_rgb(
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

def _normalise_training_report(report: dict) -> dict:
    """Accept both the original and hybrid Extra Trees report schemas."""
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
                    f"Hybrid candidate {candidate.get('candidate', '?')}: "
                    f"{parameters.get('n_estimators', '?')} trees, "
                    f"depth={parameters.get('max_depth')}"
                ),
                "min_samples_leaf": parameters.get("min_samples_leaf", 1),
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
