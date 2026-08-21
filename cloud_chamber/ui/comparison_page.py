"""Fair final-test comparison dashboard for all member classifiers."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st


REPORTS = {
    "CNN": Path("models/cnn_training_report.json"),
    "SVM": Path("models/svm_training_report.json"),
    "Decision Tree": Path("models/decision_tree_training_report.json"),
    "MLP": Path("models/mlp_training_report.json"),
    "Extra Trees": Path("models/extra_trees_training_report.json"),
}

DISPLAY_NAMES = {
    "alpha": "Alpha",
    "electron_positron": "Electron/Positron",
    "proton": "Proton",
    "v_track": "V-track",
}


def _load_results() -> tuple[dict[str, dict], dict[str, str]]:
    results: dict[str, dict] = {}
    problems: dict[str, str] = {}
    for model_name, path in REPORTS.items():
        if not path.exists():
            problems[model_name] = f"Missing {path.name}"
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            results[model_name] = report["final_test"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            problems[model_name] = f"Invalid report: {exc}"
    return results, problems


def _comparison_sample_count(results: dict[str, dict]) -> int | None:
    """Return the most commonly reported test size as the comparison cohort."""
    counts = [int(result["sample_count"]) for result in results.values()]
    return Counter(counts).most_common(1)[0][0] if counts else None


def render() -> None:
    st.title("Final Model Comparison")
    st.caption(
        "A fair comparison uses the same untouched final-test samples and the "
        "same metrics. Macro F1 is the primary ranking measure because every "
        "particle class contributes equally."
    )

    results, problems = _load_results()
    if not results:
        st.warning("No final-test reports are available yet.")
        return

    comparison_count = _comparison_sample_count(results)
    comparable = {
        name: result
        for name, result in results.items()
        if int(result["sample_count"]) == comparison_count
    }
    excluded = {
        name: result
        for name, result in results.items()
        if int(result["sample_count"]) != comparison_count
    }

    status_rows = []
    for model_name in REPORTS:
        if model_name in comparable:
            status = "Comparable"
            sample_count = comparison_count
        elif model_name in excluded:
            sample_count = int(excluded[model_name]["sample_count"])
            status = f"Excluded: expected {comparison_count} test tracks"
        else:
            sample_count = "—"
            status = problems.get(model_name, "Report unavailable")
        status_rows.append(
            {"Model": model_name, "Final-test tracks": sample_count, "Status": status}
        )

    st.subheader("Comparison validity")
    st.dataframe(status_rows, use_container_width=True, hide_index=True)

    winner_name, winner_result = max(
        comparable.items(), key=lambda item: float(item[1]["macro_f1"])
    )
    definitive = len(comparable) == len(REPORTS) and not problems and not excluded
    if definitive:
        st.success(
            f"Best overall model: {winner_name} — final-test Macro F1 "
            f"{winner_result['macro_f1']:.3f}."
        )
    else:
        st.info(
            f"Current provisional leader: {winner_name} — Macro F1 "
            f"{winner_result['macro_f1']:.3f} among the {len(comparable)} models "
            f"evaluated on {comparison_count} tracks. A definitive winner requires "
            "all five models to use exactly the same final-test set."
        )

    ranked = sorted(
        comparable.items(), key=lambda item: float(item[1]["macro_f1"]), reverse=True
    )
    ranking_rows = []
    for rank, (model_name, result) in enumerate(ranked, start=1):
        macro_report = result.get("classification_report", {}).get("macro avg", {})
        ranking_rows.append(
            {
                "Rank": rank,
                "Model": model_name,
                "Accuracy": float(result["accuracy"]),
                "Macro precision": float(macro_report.get("precision", 0.0)),
                "Macro recall": float(macro_report.get("recall", 0.0)),
                "Macro F1": float(result["macro_f1"]),
                "Time/track (ms)": result.get("mean_inference_ms_per_track", "Not recorded"),
            }
        )

    st.subheader("Final-test ranking")
    st.dataframe(
        ranking_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Macro precision": st.column_config.NumberColumn(format="%.3f"),
            "Macro recall": st.column_config.NumberColumn(format="%.3f"),
            "Macro F1": st.column_config.NumberColumn(format="%.3f"),
            "Time/track (ms)": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    _render_metric_chart(comparable)
    _render_recall_heatmap(comparable)
    _render_processing_chart(comparable)
    _render_confusion_matrices(comparable)


def _render_metric_chart(results: dict[str, dict]) -> None:
    figure = go.Figure()
    specifications = [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
    ]
    model_names = list(results)
    for label, key in specifications:
        figure.add_bar(
            name=label,
            x=model_names,
            y=[float(results[name][key]) for name in model_names],
            text=[f"{float(results[name][key]):.3f}" for name in model_names],
            textposition="outside",
        )
    figure.update_layout(
        title="Overall classification performance",
        barmode="group",
        yaxis={"title": "Score", "range": [0, 1.08]},
        xaxis_title="Model",
        legend_title_text="Metric",
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_recall_heatmap(results: dict[str, dict]) -> None:
    model_names = list(results)
    class_names = ["alpha", "electron_positron", "proton", "v_track"]
    values = [
        [
            float(results[model]["classification_report"][class_name]["recall"])
            for class_name in class_names
        ]
        for model in model_names
    ]
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=[DISPLAY_NAMES[name] for name in class_names],
            y=model_names,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            text=[[f"{value:.3f}" for value in row] for row in values],
            texttemplate="%{text}",
            colorbar={"title": "Recall"},
            hovertemplate="Model: %{y}<br>Class: %{x}<br>Recall: %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Per-class recall — ability to find each particle type",
        xaxis_title="True particle class",
        yaxis_title="Model",
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_processing_chart(results: dict[str, dict]) -> None:
    timed = [
        (name, result.get("mean_inference_ms_per_track"))
        for name, result in results.items()
        if result.get("mean_inference_ms_per_track") is not None
    ]
    st.subheader("Prediction speed")
    if not timed:
        st.warning("Processing time was not recorded by the available reports.")
        return
    figure = go.Figure(
        go.Bar(
            x=[item[0] for item in timed],
            y=[float(item[1]) for item in timed],
            text=[f"{float(item[1]):.4f} ms" for item in timed],
            textposition="outside",
            marker_color="#00A6A6",
        )
    )
    figure.update_layout(
        title="Mean model inference time per track (lower is better)",
        xaxis_title="Model",
        yaxis_title="Milliseconds per track",
    )
    st.plotly_chart(figure, use_container_width=True)
    missing = [
        name
        for name, result in results.items()
        if result.get("mean_inference_ms_per_track") is None
    ]
    if missing:
        st.caption(f"Timing not recorded for: {', '.join(missing)}.")


def _render_confusion_matrices(results: dict[str, dict]) -> None:
    st.subheader("Confusion matrices")
    st.caption("Rows are true classes; columns are predicted classes.")
    for model_name, result in results.items():
        labels = [DISPLAY_NAMES.get(name, name) for name in result["class_names"]]
        matrix = result["confusion_matrix"]
        with st.expander(model_name):
            figure = go.Figure(
                go.Heatmap(
                    z=matrix,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    text=matrix,
                    texttemplate="%{text}",
                    colorbar={"title": "Tracks"},
                )
            )
            figure.update_layout(
                xaxis_title="Predicted class",
                yaxis_title="True class",
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(figure, use_container_width=True)
