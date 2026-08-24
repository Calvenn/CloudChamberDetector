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
        "All five available final-test reports are compared descriptively. "
        "Macro F1 is the primary ranking measure because every particle class "
        "contributes equally."
    )

    results, problems = _load_results()
    if not results:
        st.warning("No final-test reports are available yet.")
        return

    _render_executive_overview(results)

    # The project report requests a descriptive comparison of every available
    # model. Unequal cohort sizes are disclosed below but do not hide a model.
    comparable = dict(results)

    class_signatures = {
        name: tuple(result.get("class_names", [])) for name, result in results.items()
    }
    fully_aligned = (
        len(results) == len(REPORTS)
        and len({int(item["sample_count"]) for item in results.values()}) == 1
        and len(set(class_signatures.values())) == 1
        and not problems
    )
    if not fully_aligned:
        st.warning(
            "All five models are shown, but this is a descriptive comparison: "
            "their final-test track counts differ, and CNN does not report V-track. "
            "The highest reported score may be stated, but the comparison is not "
            "a strictly controlled evaluation on one identical test cohort."
        )

    winner_name, winner_result = max(
        comparable.items(), key=lambda item: float(item[1]["macro_f1"])
    )
    definitive = fully_aligned
    if definitive:
        st.success(
            f"Best overall model: {winner_name} — final-test Macro F1 "
            f"{winner_result['macro_f1']:.3f}."
        )
    else:
        st.info(
            f"Highest reported result: {winner_name} — Macro F1 "
            f"{winner_result['macro_f1']:.3f} across the five available reports. "
            "Interpret this as a descriptive result because cohort sizes differ."
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
    _render_tradeoff_chart(results)
    _render_automatic_findings(results, comparable, definitive)
    _render_confusion_matrices(comparable)


def _render_executive_overview(results: dict[str, dict]) -> None:
    """Show every reported result while clearly separating it from fair ranking."""
    st.subheader("Five-model reported-results overview")
    st.caption(
        "This table is useful for checking the current outputs, but rows with "
        "different test counts or class sets must not be treated as a fair ranking."
    )
    rows = []
    for model_name in REPORTS:
        result = results.get(model_name)
        if result is None:
            continue
        rows.append(
            {
                "Model": model_name,
                "Tracks": int(result["sample_count"]),
                "Classes": len(result.get("class_names", [])),
                "Accuracy": float(result["accuracy"]),
                "Balanced accuracy": float(result["balanced_accuracy"]),
                "Macro F1": float(result["macro_f1"]),
                "Weighted F1": float(result["weighted_f1"]),
                "Time/track (ms)": result.get("mean_inference_ms_per_track"),
            }
        )
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
            "Balanced accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Macro F1": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
            "Weighted F1": st.column_config.NumberColumn(format="%.3f"),
            "Time/track (ms)": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    highest_reported = max(results.items(), key=lambda item: float(item[1]["macro_f1"]))
    fastest = [
        (name, result) for name, result in results.items()
        if result.get("mean_inference_ms_per_track") is not None
    ]
    columns = st.columns(3)
    columns[0].metric("Models reported", f"{len(results)} / {len(REPORTS)}")
    columns[1].metric(
        "Highest reported Macro F1",
        f"{highest_reported[1]['macro_f1']:.3f}",
        highest_reported[0],
    )
    if fastest:
        fastest_name, fastest_result = min(
            fastest, key=lambda item: float(item[1]["mean_inference_ms_per_track"])
        )
        columns[2].metric(
            "Fastest reported inference",
            f"{float(fastest_result['mean_inference_ms_per_track']):.4f} ms",
            fastest_name,
        )


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
            (
                float(results[model]["classification_report"][class_name]["recall"])
                if class_name in results[model].get("classification_report", {})
                else None
            )
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
            text=[
                [f"{value:.3f}" if value is not None else "N/A" for value in row]
                for row in values
            ],
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


def _render_tradeoff_chart(results: dict[str, dict]) -> None:
    """Relate balanced class performance to inference cost."""
    timed = {
        name: result for name, result in results.items()
        if result.get("mean_inference_ms_per_track") is not None
    }
    if not timed:
        return
    figure = go.Figure()
    for name, result in timed.items():
        figure.add_trace(
            go.Scatter(
                x=[float(result["mean_inference_ms_per_track"])],
                y=[float(result["macro_f1"])],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker={"size": 15},
                name=name,
                hovertemplate=(
                    f"{name}<br>Macro F1: %{{y:.3f}}<br>"
                    "Time/track: %{x:.4f} ms<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Reported accuracy–speed trade-off",
        xaxis_title="Mean inference time per track (ms; lower is better)",
        yaxis={"title": "Macro F1 (higher is better)", "range": [0, 1]},
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "This trade-off chart uses each report's current cohort and is descriptive "
        "until all five models are reevaluated on the same frozen final-test table."
    )


def _render_automatic_findings(
    results: dict[str, dict], comparable: dict[str, dict], definitive: bool
) -> None:
    """Translate the charts into short, evidence-based statements."""
    st.subheader("Key findings")
    reported_best_name, reported_best = max(
        results.items(), key=lambda item: float(item[1]["macro_f1"])
    )
    findings = [
        f"{reported_best_name} has the highest currently reported Macro F1 "
        f"({float(reported_best['macro_f1']):.3f}), but this is not yet proof "
        "that it is the best model because the evaluation cohorts differ."
    ]
    if comparable:
        cohort_best_name, cohort_best = max(
            comparable.items(), key=lambda item: float(item[1]["macro_f1"])
        )
        findings.append(
            f"Across all five reported results, {cohort_best_name} has the "
            f"highest Macro F1 ({float(cohort_best['macro_f1']):.3f})."
        )
    findings.append(
        "Macro F1 is the primary selection metric because it gives equal "
        "importance to Alpha, Electron/Positron, Proton and V-track despite "
        "their unequal sample frequencies."
    )
    findings.append(
        "For the university report, describe this as the highest reported model; "
        "also state that different sample counts limit direct experimental fairness."
    )
    for finding in findings:
        st.markdown(f"- {finding}")
    if definitive:
        st.success("The five reports are aligned; the displayed winner is definitive.")


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
