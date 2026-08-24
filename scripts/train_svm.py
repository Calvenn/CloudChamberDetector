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

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.config import load_config
from cloud_chamber.ml.contour_dataset import build_segmented_feature_csv, load_feature_csv
from cloud_chamber.ml.member_models.svm import FEATURE_COLUMNS


SPLITS = ("development", "validation", "final_test")

PARAMETER_CANDIDATES = (
    {"C": 0.1, "kernel": "linear"},
    {"C": 1.0, "kernel": "linear"},
    {"C": 10.0, "kernel": "linear"},
    {"C": 0.1, "kernel": "rbf"},
    {"C": 1.0, "kernel": "rbf"},
    {"C": 10.0, "kernel": "rbf"},
)


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
    return load_feature_csv(cache, labels)


def make_pipeline(parameters: dict, random_seed: int):
    """Create a scaler and SVM as one indivisible training pipeline."""
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=parameters["C"],
                    kernel=parameters["kernel"],
                    probability=True,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Give each class equal total influence despite class imbalance."""
    counts = Counter(labels.tolist())
    sample_count = len(labels)
    class_count = len(counts)
    return np.asarray(
        [sample_count / (class_count * counts[label]) for label in labels],
        dtype=np.float64,
    )


def fit_with_balancing(model, matrix: np.ndarray, labels: np.ndarray, seed: int):
    """Use sample weights, with deterministic weighted resampling as fallback."""
    weights = balanced_sample_weights(labels)
    try:
        model.fit(matrix, labels, svm__sample_weight=weights)
        return "inverse-frequency sample weights"
    except TypeError:
        generator = np.random.default_rng(seed)
        probabilities = weights / weights.sum()
        indices = generator.choice(
            len(labels), size=len(labels), replace=True, p=probabilities
        )
        model.fit(matrix[indices], labels[indices])
        return "deterministic class-balanced resampling"


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
        balancing = fit_with_balancing(model, development_x, development_y, seed)
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
    _, best_parameters, best_model, balancing, best_index = best
    final_x, final_y = split_tables["final_test"]
    final_metrics = evaluate(best_model, final_x, final_y)
    selected_validation = candidates[best_index - 1]["validation"]

    bundle = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": tuple(str(value) for value in best_model.classes_),
        "selected_parameters": best_parameters,
        "random_seed": seed,
        "balancing": balancing,
        "validation_metrics": selected_validation,
        "final_test_metrics": final_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)

    report = {
        "method": "StandardScaler + SVC",
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
