"""Decision Tree Streamlit page."""
from pathlib import Path
import streamlit as st
from cloud_chamber.ml.member_models.decision_tree import (build_visual_report as decision_tree_build_visual_report, encode_report_csv as decision_tree_encode_report_csv, encode_report_png as decision_tree_encode_report_png, load_model as decision_tree_load_model, predict_tracks as decision_tree_predict_tracks)
from .context import PageContext

def render(context: PageContext) -> None:
    st.title("Decision Tree Classifier")
    st.subheader("Run the trained Decision Tree")
    result = st.session_state.get("pipeline_result")
    if result is None:
        st.warning(
            "Process an image or video frame on the Shared Processing "
            "Pipeline page first."
        )
        return
    model_path = Path("models/decision_tree_classifier.joblib")
    if not model_path.exists():
        st.warning(
            "Train the model first: `python scripts/train_decision_tree.py`"
        )
        return
    confidence_threshold = st.slider(
        "Reporting confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help=(
            "Predictions below this probability remain visible but are marked "
            "Uncertain. This threshold does not retrain the model."
        ),
    )
    if st.button("Classify and create Decision Tree report", type="primary"):
        model_bundle = decision_tree_load_model(model_path)
        st.session_state["decision_tree_predictions"] = (
            decision_tree_predict_tracks(model_bundle, result["features"])
        )
    predictions = st.session_state.get("decision_tree_predictions")
    if predictions is None:
        return
    if not predictions:
        st.warning("No segmented track is available for classification.")
        return

    overlay, report_rows = decision_tree_build_visual_report(
        st.session_state["input_image"],
        result["features"],
        predictions,
        confidence_threshold,
    )
    st.subheader("Particle classification report")
    st.image(
        context.bgr_to_rgb(overlay),
        caption=(
            "Decision Tree particle predictions. Yellow annotations are below "
            "the selected confidence threshold."
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
        data=decision_tree_encode_report_png(overlay),
        file_name=f"{safe_name}_decision_tree_report.png",
        mime="image/png",
    )
    downloads[1].download_button(
        "Download classification CSV",
        data=decision_tree_encode_report_csv(report_rows),
        file_name=f"{safe_name}_decision_tree_report.csv",
        mime="text/csv",
    )
