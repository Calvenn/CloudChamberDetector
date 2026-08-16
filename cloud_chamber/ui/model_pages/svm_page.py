"""SVM Streamlit page."""
from pathlib import Path
from time import perf_counter

import numpy as np
import streamlit as st
import cloud_chamber.ml.member_models.svm as svm_module
from .context import PageContext

def render(context: PageContext) -> None:
    st.title("SVM Classifier")
    st.subheader("Step 2: Configure SVM")
    result = st.session_state.get("pipeline_result")
    if result is None:
        st.warning(
            "Process an image or video frame on the Shared Processing "
            "Pipeline page first."
        )
        return
    model_path = Path("models/svm_classifier.joblib")
    if not model_path.exists():
        st.warning("Train the model first: `python scripts/train_svm.py`")
        return
    model_bundle = svm_module.load_model(model_path)

    with st.expander("Advanced Settings", expanded=False):
        use_probability = st.checkbox(
            "Use probability estimates",
            value=True,
            help="If checked, uses SVC's probability estimates to output classification confidence and apply the decision threshold.",
        )
        decision_threshold = st.slider(
            "Decision threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            disabled=not use_probability,
            help=(
                "Predictions below this probability remain visible but are marked "
                "Uncertain. This threshold is only applied when using probability estimates."
            ),
        )

    if st.button("Classify and create SVM report", type="primary"):
        st.session_state["svm_predictions"] = _predict_tracks_svm(
            model_bundle, result["features"], use_probability
        )
    predictions = st.session_state.get("svm_predictions")
    if predictions is None:
        return
    if not predictions:
        st.warning("No segmented track is available for classification.")
        return

    threshold = decision_threshold if use_probability else 0.0
    overlay, report_rows = svm_module.build_visual_report(
        st.session_state["input_image"],
        result["features"],
        predictions,
        threshold,
    )
    st.subheader("SVM Particle classification report")
    st.image(
        context.bgr_to_rgb(overlay),
        caption=(
            "SVM particle predictions. Yellow annotations are below the "
            "selected decision threshold."
        ),
    )
    st.dataframe(report_rows, use_container_width=True, hide_index=True)
    st.caption(
        "The label is the model's most likely class for each segmented "
        "contour. It is a prediction, not direct physical confirmation."
    )
    safe_name = Path(st.session_state["input_name"]).stem
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download annotated image",
        data=svm_module.encode_report_png(overlay),
        file_name=f"{safe_name}_svm_report.png",
        mime="image/png",
    )
    downloads[1].download_button(
        "Download classification CSV",
        data=svm_module.encode_report_csv(report_rows),
        file_name=f"{safe_name}_svm_report.csv",
        mime="text/csv",
    )


def _predict_tracks_svm(
    model_bundle: dict,
    features: list,
    use_probability: bool,
) -> list[dict]:
    matrix = svm_module.features_to_matrix(features)
    if matrix.shape[0] == 0:
        return []

    model = model_bundle["model"]
    started = perf_counter()

    if use_probability:
        probabilities = model.predict_proba(matrix)
        predictions = model.classes_[np.argmax(probabilities, axis=1)]
        elapsed_per_track = (perf_counter() - started) * 1000.0 / len(matrix)
        return [
            {
                "track_id": track.track_id,
                "predicted_class": str(label),
                "particle_type": svm_module.DISPLAY_NAMES.get(str(label), str(label)),
                "confidence": float(np.max(probability)),
                "inference_time_ms": elapsed_per_track,
                "probabilities": {
                    str(class_name): float(class_probability)
                    for class_name, class_probability in zip(
                        model.classes_, probability, strict=True
                    )
                },
            }
            for track, label, probability in zip(
                features, predictions, probabilities, strict=True
            )
        ]
    else:
        predictions = model.predict(matrix)
        elapsed_per_track = (perf_counter() - started) * 1000.0 / len(matrix)
        return [
            {
                "track_id": track.track_id,
                "predicted_class": str(label),
                "particle_type": svm_module.DISPLAY_NAMES.get(str(label), str(label)),
                "confidence": 1.0,
                "inference_time_ms": elapsed_per_track,
                "probabilities": {
                    str(class_name): 1.0 if str(class_name) == str(label) else 0.0
                    for class_name in model.classes_
                },
            }
            for track, label in zip(
                features, predictions, strict=True
            )
        ]
