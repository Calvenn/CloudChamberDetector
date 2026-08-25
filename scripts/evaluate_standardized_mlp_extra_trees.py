"""Fair classifier-only comparison of MLP and Extra Trees.

Both models receive the identical final-test particles, measured from one COCO
ground-truth instance mask per annotation. This isolates classification from
automatic-segmentation recall and preserves annotation IDs for auditing.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.config import load_config
from cloud_chamber.ml.contour_dataset import build_feature_csv
from cloud_chamber.ml.member_models.extra_trees import load_model as load_extra_trees
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS, load_model as load_mlp


def _load_rows(path: Path, source: str) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source"] = source
    return rows


def _evaluate(bundle: dict, matrix: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    model = bundle["model"]
    started = perf_counter()
    predictions = np.asarray(model.predict(matrix), dtype=str)
    elapsed_ms = (perf_counter() - started) * 1000.0
    classes = sorted(set(labels.tolist()) | set(predictions.tolist()))
    metrics = {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mean_inference_ms_per_track": elapsed_ms / max(len(labels), 1),
        "class_names": classes,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=classes, output_dict=True, zero_division=0
        ),
    }
    return metrics, predictions


def main() -> int:
    config = load_config(PROJECT_ROOT / "config.yaml")
    allowed = set(config["classification"]["supported_classes"])
    output_dir = PROJECT_ROOT / "results" / "standardized_comparison"
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    sources = (
        (
            "external",
            PROJECT_ROOT / "dataset" / "external_dataset_split" / "final_test" / "annotations_coco.json",
            feature_dir / "external_final_test.csv",
        ),
        (
            "primary",
            PROJECT_ROOT / "dataset" / "primary_dataset_split" / "final_test_annotations_coco.json",
            feature_dir / "primary_final_test.csv",
        ),
    )
    rows: list[dict] = []
    for source, annotations, feature_path in sources:
        build_feature_csv(
            annotations,
            feature_path,
            config["enhancement"],
            allowed_labels=allowed,
            target_size=(
                int(config["spatial_scaling"]["processing_width"]),
                int(config["spatial_scaling"]["processing_height"]),
            ),
        )
        rows.extend(_load_rows(feature_path, source))

    rows.sort(key=lambda row: (row["source"], int(row["image_id"]), int(row["annotation_id"])))
    if not rows:
        raise ValueError("The standardised final-test set contains no valid particles")
    matrix = np.asarray(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([row["label"] for row in rows], dtype=str)

    bundles = {
        "MLP": load_mlp(PROJECT_ROOT / "models" / "mlp_classifier.joblib"),
        "Extra Trees": load_extra_trees(PROJECT_ROOT / "models" / "extra_trees_classifier.joblib"),
    }
    results = {}
    predictions_by_model = {}
    for name, bundle in bundles.items():
        if tuple(bundle["feature_columns"]) != tuple(FEATURE_COLUMNS):
            raise ValueError(f"{name} does not use the shared 16-feature contract")
        results[name], predictions_by_model[name] = _evaluate(bundle, matrix, labels)

    report = {
        "evaluation_type": "classifier-only, one ground-truth mask per COCO annotation",
        "feature_columns": list(FEATURE_COLUMNS),
        "sample_count": len(rows),
        "class_counts": dict(sorted(Counter(labels.tolist()).items())),
        "models": results,
    }
    report_path = output_dir / "mlp_extra_trees_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions_path = output_dir / "mlp_extra_trees_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["source", "image_id", "annotation_id", "actual", "mlp_prediction", "extra_trees_prediction"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "source": row["source"],
                    "image_id": row["image_id"],
                    "annotation_id": row["annotation_id"],
                    "actual": row["label"],
                    "mlp_prediction": predictions_by_model["MLP"][index],
                    "extra_trees_prediction": predictions_by_model["Extra Trees"][index],
                }
            )

    print(f"Standardised particles: {len(rows)}")
    for name, metrics in results.items():
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    print(f"Report: {report_path}")
    print(f"Predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
