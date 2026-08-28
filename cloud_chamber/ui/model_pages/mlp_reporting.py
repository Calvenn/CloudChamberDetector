"""Reusable analytical visualisations for the MLP Streamlit page.

Labelled final-test evidence and live, unlabelled prediction evidence are kept
separate so users do not mistake dashboard summaries for accuracy measures.
"""

from __future__ import annotations

from collections import Counter

import plotly.graph_objects as go
import streamlit as st

def render_ensemble_agreement(predictions: list[dict]) -> None:
    """Summarise agreement among the independently seeded MLP members."""
    agreements = [
        prediction.get("ensemble_agreement", {}) for prediction in predictions
    ]
    agreements = [item for item in agreements if item.get("total_members", 0)]
    if not agreements:
        return

    total_members = max(int(item["total_members"]) for item in agreements)
    counts = Counter(int(item["agreeing_members"]) for item in agreements)
    mean_ratio = sum(float(item["ratio"]) for item in agreements) / len(agreements)
    unanimous = counts.get(total_members, 0)

    st.markdown("**Five-member ensemble agreement**")
    columns = st.columns(3)
    columns[0].metric("Mean agreement", f"{mean_ratio:.1%}")
    columns[1].metric("Unanimous tracks", f"{unanimous}/{len(agreements)}")
    columns[2].metric("Ensemble members", total_members)

    possible = list(range(1, total_members + 1))
    figure = go.Figure(
        go.Bar(
            x=[f"{value}/{total_members}" for value in possible],
            y=[counts.get(value, 0) for value in possible],
            text=[counts.get(value, 0) for value in possible],
            textposition="outside",
            marker_color="#7E57C2",
        )
    )
    figure.update_layout(
        title="Agreement on the final predicted class",
        xaxis_title="Members selecting the ensemble class",
        yaxis_title="Detected tracks",
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)
def agreement_text(prediction: dict) -> str:
    """Return compact member-agreement text for one evidence card."""
    agreement = prediction.get("ensemble_agreement", {})
    agreeing = int(agreement.get("agreeing_members", 1))
    total = int(agreement.get("total_members", 1))
    return f"{agreeing}/{total} members selected this class"
