"""Train, validate and save the contour-feature SVM classifier.

Parameter policy
----------------
The candidate values are deliberately small and interpretable. The validation
split selects the configuration by macro F1, so parameter choice is supported
by this project's data rather than an unsupported claim of universal optimality.
The final-test split is evaluated only after that choice has been made.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from sklearn.model_selection import ParameterGrid

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.config import load_config
from cloud_chamber.ml.contour_dataset import build_segmented_feature_csv, load_feature_csv
from cloud_chamber.ml.member_models.svm import FEATURE_COLUMNS


SPLITS = ("development", "validation", "final_test")

param_grid = [
    {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    {
        "kernel": ["rbf"],
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
    },
    {
        "kernel": ["poly"],
        "C": [0.1, 1.0, 10.0, 100.0],
        "degree": [2, 3],
        "gamma": ["scale", "auto"],
    },
]

# Generates a list of dictionaries compatible with your training loop
PARAMETER_CANDIDATES = list(ParameterGrid(param_grid))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contour-feature SVM")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "external_dataset_split",
    )
    parser.add_argument(
        "--primary-split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "primary_dataset_split",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "muller",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models" / "svm_classifier.joblib"
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def _primary_split(source_path: Path, split: str) -> Path:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    image_ids = {
        int(image["id"])
        for image in payload.get("images", [])
        if image.get("split") == split
    }
    if not image_ids:
        raise ValueError(f"Primary dataset contains no images for split={split!r}")
    filtered = {
        **payload,
        "images": [
            image for image in payload.get("images", [])
            if int(image["id"]) in image_ids
        ],
        "annotations": [
            annotation for annotation in payload.get("annotations", [])
            if int(annotation["image_id"]) in image_ids
        ],
    }
    output = source_path.with_name(f"{split}_annotations_coco.json")
    output.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    return output


def _features(
    annotations: Path,
    cache: Path,
    config: dict,
    labels: set[str],
    rebuild: bool,
    roi_profile: str,
) -> tuple[np.ndarray, np.ndarray]:
    if rebuild or not cache.exists():
        build_segmented_feature_csv(
            annotations,
            cache,
            config,
            allowed_labels=labels,
            roi_profile_name=roi_profile,
        )
    raw_matrix, labels_array = load_feature_csv(cache, labels)
    
    area = raw_matrix[:, 0]
    perimeter = raw_matrix[:, 1]
    major_axis = raw_matrix[:, 2]
    mean_width = raw_matrix[:, 3]
    aspect_ratio = raw_matrix[:, 5]
    solidity = raw_matrix[:, 6]
    rectangularity = raw_matrix[:, 7]
    thickness = raw_matrix[:, 8]
    mean_intensity = raw_matrix[:, 9]

    safe_major = np.where(major_axis > 0, major_axis, 1.0)
    line_density = np.where(major_axis > 0, (mean_intensity * area) / safe_major, 0.0)
    tortuosity = np.where(major_axis > 0, perimeter / (2.0 * safe_major), 1.0)

    svm_matrix = np.column_stack([
        area,
        perimeter,
        major_axis,
        mean_width,
        aspect_ratio,
        solidity,
        rectangularity,
        thickness,
        mean_intensity,
        line_density,
        tortuosity,
    ])
    
    return svm_matrix, labels_array


def make_pipeline(parameters: dict, random_seed: int):
    """Create a scaler and SVM as one indivisible training pipeline."""
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.calibration import CalibratedClassifierCV

    svc_kwargs = {
        "C": parameters["C"],
        "kernel": parameters["kernel"],
        "class_weight": {
            "alpha": 1.0,
            "electron_positron": 2.5,
            "proton": 0.6,
            "v_track": 2.0,
        },
        "random_state": random_seed,
    }
    if "gamma" in parameters:
        svc_kwargs["gamma"] = parameters["gamma"]

    return Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "svm",
                CalibratedClassifierCV(SVC(**svc_kwargs), method="isotonic", cv=5),
            ),
        ]
    )


def evaluate(model, matrix: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    started = perf_counter()
    predictions = model.predict(matrix)
    elapsed_ms = (perf_counter() - started) * 1000.0
    class_names = sorted(set(labels.tolist()) | set(predictions.tolist()))
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "mean_inference_ms_per_track": elapsed_ms / max(len(labels), 1),
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=class_names
        ).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=class_names, output_dict=True, zero_division=0
        ),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["random_seed"])
    allowed_labels = set(config["classification"]["supported_classes"])
    args.feature_dir.mkdir(parents=True, exist_ok=True)

    primary_source = args.primary_split_root / "annotations_coco.json"

    split_tables = {}
    class_counts = {}
    for split in SPLITS:
        external_x, external_y = _features(
            args.split_root / split / "annotations_coco.json",
            args.feature_dir / f"{split}.csv",
            config,
            allowed_labels,
            args.rebuild_features,
            "external_muller",
        )
        primary_x, primary_y = _features(
            _primary_split(primary_source, split),
            args.feature_dir / f"primary_{split}.csv",
            config,
            allowed_labels,
            args.rebuild_features,
            "primary_full_chamber",
        )
        matrix = np.concatenate((external_x, primary_x), axis=0)
        labels = np.concatenate((external_y, primary_y), axis=0)
        split_tables[split] = (matrix, labels)
        class_counts[split] = {
            "external": dict(sorted(Counter(external_y).items())),
            "primary": dict(sorted(Counter(primary_y).items())),
            "combined": dict(sorted(Counter(labels).items())),
        }
        print(
            f"{split}: {len(labels)} combined tracks "
            f"(external={len(external_y)}, primary={len(primary_y)})"
        )

    development_x, development_y = split_tables["development"]
    validation_x, validation_y = split_tables["validation"]
    candidates = []
    best = None
    for index, parameters in enumerate(PARAMETER_CANDIDATES, start=1):
        model = make_pipeline(parameters, seed)
        model.fit(development_x, development_y)
        balancing = "class_weight='balanced'"
        metrics = evaluate(model, validation_x, validation_y)
        record = {
            "candidate": index,
            "parameters": {
                **parameters,
            },
            "balancing": balancing,
            "validation": metrics,
        }
        candidates.append(record)
        print(
            f"candidate {index}: macro_f1={metrics['macro_f1']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )
        score = (metrics["macro_f1"], metrics["balanced_accuracy"])
        if best is None or score > best[0]:
            best = (score, parameters, model, balancing, index)

    assert best is not None
    _, best_parameters, _, balancing, best_index = best
    
    # Refit on combined (Development + Validation) data
    train_x = np.concatenate((development_x, validation_x), axis=0)
    train_y = np.concatenate((development_y, validation_y), axis=0)
    
    final_model = make_pipeline(best_parameters, seed)
    final_model.fit(train_x, train_y)
    
    final_x, final_y = split_tables["final_test"]
    final_metrics = evaluate(final_model, final_x, final_y)
    selected_validation = candidates[best_index - 1]["validation"]

    bundle = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": tuple(str(value) for value in final_model.classes_),
        "selected_parameters": best_parameters,
        "random_seed": seed,
        "balancing": balancing,
        "validation_metrics": selected_validation,
        "final_test_metrics": final_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)

    report = {
        "method": "RobustScaler + SVC (with Dev+Val Retraining)",
        "feature_source": (
            "combined primary and external automatic segmentation contours "
            "labelled by COCO-mask overlap"
        ),
        "selection_metric": "validation macro F1; balanced accuracy tie-breaker",
        "feature_columns": FEATURE_COLUMNS,
        "class_counts": class_counts,
        "candidate_results": candidates,
        "selected_candidate": best_index,
        "selected_parameters": best_parameters,
        "final_test": final_metrics,
        "model_path": str(args.output),
    }
    report_path = args.output.with_name("svm_training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}")
    print(f"Saved evidence report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
