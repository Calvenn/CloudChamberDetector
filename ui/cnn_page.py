from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch

from cloud_chamber.ml.member_models.cnn import FeatureTrackCNN, build_class_mapping
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS


MODEL_PATH = Path("models/cnn_classifier.pth")
DISPLAY_LABELS = {
    "alpha": "Alpha",
    "electron_positron": "Electron / Positron (beta-like)",
    "proton": "Proton",
}


def render_cnn_page() -> None:
    st.subheader("CNN result and reporting")
    st.caption(
        "This page follows the shared feature-contract rule: the same contour "
        "features are used for prediction as in the other models, and the model "
        "output is reported in a concise, useful summary."
    )

    result = st.session_state.get("pipeline_result")
    if result is None:
        st.warning("Process an image or video frame on the Shared Processing Pipeline page first.")
        return

    if not MODEL_PATH.exists():
        st.warning("Train the CNN first with: `python -m cloud_chamber.ml.member_models.cnn`")
        return

    confidence_threshold = st.slider(
        "Reporting confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
    )

    if st.button("Classify with CNN", type="primary"):
        predictions = _predict_from_current_features(result["features"])
        st.session_state["cnn_predictions"] = predictions

    predictions = st.session_state.get("cnn_predictions")
    if predictions is None:
        return

    if not predictions:
        st.warning("No segmented track is available for CNN classification.")
        return

    st.subheader("CNN classification summary")
    report = _summarise_predictions(predictions, confidence_threshold)

    st.markdown("### Particle result overview")
    overview = st.columns(len(report["cards"])) if report["cards"] else st.columns(1)
    for idx, card in enumerate(report["cards"]):
        with overview[idx]:
            label = card["label"]
            display_name = DISPLAY_LABELS.get(label, label)
            colour = "#4caf50" if label in {"alpha", "proton"} else "#ffb703"
            st.markdown(
                f"<div style='border: 1px solid {colour}; border-radius: 10px; padding: 12px; background: rgba(255,255,255,0.02);'>"
                f"<h4 style='margin: 0 0 8px 0; color: {colour};'>{card['title']}</h4>"
                f"<p style='margin: 4px 0;'><b>Predicted:</b> {display_name}</p>"
                f"<p style='margin: 4px 0;'><b>Confidence:</b> {card['confidence']:.2%}</p>"
                f"<p style='margin: 4px 0;'><b>Status:</b> {card['status']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Class breakdown")
    class_summary = report["class_summary"]
    class_df = [
        {
            "Particle type": DISPLAY_LABELS.get(label, label),
            "Count": count,
            "Mean confidence": class_summary[label]["mean_confidence"],
        }
        for label, count in sorted(class_summary.items())
    ]
    st.dataframe(class_df, use_container_width=True, hide_index=True)

    st.markdown("### Detailed track-level result")
    display_table = []
    for row in report["table"]:
        display_row = dict(row)
        display_row["Predicted class"] = DISPLAY_LABELS.get(display_row["Predicted class"], display_row["Predicted class"])
        display_table.append(display_row)
    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.markdown("### Summary")
    st.json({
        "Total tracked objects": report["summary"]["Total tracked objects"],
        "Predicted class counts": {
            DISPLAY_LABELS.get(key, key): value for key, value in report["summary"]["Predicted class counts"].items()
        },
        "Mean confidence": report["summary"]["Mean confidence"],
        "Low-confidence predictions": report["summary"]["Low-confidence predictions"],
    })

    safe_name = Path(st.session_state.get("input_name", "cnn_result")).stem
    st.download_button(
        "Download CNN CSV report",
        data=report["csv"],
        file_name=f"{safe_name}_cnn_report.csv",
        mime="text/csv",
    )


def _predict_from_current_features(features) -> list[dict]:
    payload = torch.load(MODEL_PATH, map_location="cpu")
    class_mapping = build_class_mapping()
    model = FeatureTrackCNN(input_dim=len(FEATURE_COLUMNS), num_classes=len(class_mapping))
    model.load_state_dict(payload["model_state"])
    model.eval()

    rows = []
    inv_mapping = {value: key for key, value in class_mapping.items()}
    with torch.no_grad():
        for track in features:
            feature_values = []
            for column in FEATURE_COLUMNS:
                feature_values.append(float(getattr(track, column)))
            x = torch.tensor([feature_values], dtype=torch.float32)
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, index = torch.max(probabilities, dim=0)
            rows.append(
                {
                    "track_id": track.track_id,
                    "true_label": "unknown",
                    "predicted_label": inv_mapping[int(index.item())],
                    "confidence": round(float(confidence.item()), 4),
                    "probabilities": {
                        inv_mapping[i]: round(float(probabilities[i].item()), 4)
                        for i in range(len(probabilities))
                    },
                }
            )
    return rows


def _summarise_predictions(predictions: list[dict], threshold: float) -> dict:
    table_rows = []
    counts = {}
    cards = []

    for row in predictions:
        label = row["predicted_label"]
        counts[label] = counts.get(label, 0) + 1
        status = "Uncertain" if row["confidence"] < threshold else "Accepted"
        table_rows.append(
            {
                "Track": row["track_id"],
                "Predicted class": label,
                "Confidence": row["confidence"],
                "Status": status,
                **{f"P({name})": probability for name, probability in row["probabilities"].items()},
            }
        )

        cards.append(
            {
                "title": f"Track {row['track_id']}",
                "label": label,
                "predicted": label,
                "confidence": row["confidence"],
                "status": status,
            }
        )

    class_summary = {}
    for label, count in counts.items():
        class_scores = [row["confidence"] for row in predictions if row["predicted_label"] == label]
        class_summary[label] = {
            "count": count,
            "mean_confidence": round(float(np.mean(class_scores)) if class_scores else 0.0, 4),
        }

    summary = {
        "Total tracked objects": len(predictions),
        "Predicted class counts": counts,
        "Mean confidence": round(
            float(np.mean([row["confidence"] for row in predictions])) if predictions else 0.0,
            4,
        ),
        "Low-confidence predictions": sum(1 for row in predictions if row["confidence"] < threshold),
    }

    csv_text = "Track,Predicted class,Confidence,Status\n"
    for row in table_rows:
        csv_text += (
            f"{row['Track']},{row['Predicted class']},{row['Confidence']},{row['Status']}\n"
        )

    return {
        "table": table_rows,
        "summary": summary,
        "csv": csv_text.encode("utf-8-sig"),
        "cards": cards,
        "class_summary": class_summary,
    }
