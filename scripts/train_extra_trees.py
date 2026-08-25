"""Train Extremely Randomized Trees on ground-truth + segmented features."""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.config import load_config
from cloud_chamber.ml.contour_dataset import load_feature_csv
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS
from cloud_chamber.ml.shared_features import DEFAULT_FEATURE_DIR, ensure_shared_features


def _load_pair(root: Path, split: str, allowed: set[str], ground_truth: bool):
    if ground_truth:
        paths = (
            root / "ground_truth" / f"external_{split}.csv",
            root / "ground_truth" / f"primary_{split}.csv",
        )
    else:
        paths = (root / f"{split}.csv", root / f"primary_{split}.csv")
    tables = [load_feature_csv(path, allowed) for path in paths]
    return (
        np.concatenate([item[0] for item in tables], axis=0),
        np.concatenate([item[1] for item in tables], axis=0),
    )


def _metrics(model, matrix: np.ndarray, labels: np.ndarray) -> dict:
    started = perf_counter()
    predictions = model.predict(matrix)
    elapsed_ms = (perf_counter() - started) * 1000.0
    classes = sorted(set(labels.tolist()) | set(predictions.tolist()))
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mean_inference_ms_per_track": elapsed_ms / max(len(labels), 1),
        "class_names": classes,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=classes, output_dict=True, zero_division=0
        ),
    }


def _model(parameters: dict, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        **parameters,
        max_features="sqrt",
        min_samples_leaf=1,
        criterion="gini",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def main() -> int:
    config = load_config(PROJECT_ROOT / "config.yaml")
    seed = int(config["project"]["random_seed"])
    allowed = set(config["classification"]["supported_classes"])
    root = ensure_shared_features(
        config=config,
        external_split_root=PROJECT_ROOT / "dataset" / "external_dataset_split",
        primary_split_root=PROJECT_ROOT / "dataset" / "primary_dataset_split",
        feature_dir=DEFAULT_FEATURE_DIR,
        allowed_labels=allowed,
    )

    gt = {split: _load_pair(root, split, allowed, True) for split in ("development", "validation", "final_test")}
    segmented = {split: _load_pair(root, split, allowed, False) for split in ("development", "validation", "final_test")}
    train_x = np.concatenate((gt["development"][0], segmented["development"][0]), axis=0)
    train_y = np.concatenate((gt["development"][1], segmented["development"][1]), axis=0)
    validation_x, validation_y = gt["validation"]

    candidates = []
    best = None
    for index, values in enumerate(product((100, 300, 500, 1000), (None, 20, 30, 40), (2, 4, 8)), start=1):
        parameters = dict(zip(("n_estimators", "max_depth", "min_samples_split"), values, strict=True))
        model = _model(parameters, seed)
        model.fit(train_x, train_y)
        validation = _metrics(model, validation_x, validation_y)
        record = {"candidate": index, "parameters": parameters, "validation": validation}
        candidates.append(record)
        score = (validation["macro_f1"], validation["balanced_accuracy"], validation["accuracy"])
        if best is None or score > best[0]:
            best = (score, parameters, index)
        print(f"candidate {index}: macro_f1={validation['macro_f1']:.4f}")

    assert best is not None
    _, parameters, best_index = best
    refit_x = np.concatenate((train_x, gt["validation"][0], segmented["validation"][0]), axis=0)
    refit_y = np.concatenate((train_y, gt["validation"][1], segmented["validation"][1]), axis=0)
    model = _model(parameters, seed)
    model.fit(refit_x, refit_y)

    standardised_final = _metrics(model, *gt["final_test"])
    segmented_final = _metrics(model, *segmented["final_test"])
    output = PROJECT_ROOT / "models" / "extra_trees_classifier.joblib"
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": tuple(str(value) for value in model.classes_),
        "selected_parameters": parameters,
        "random_seed": seed,
        "validation_metrics": candidates[best_index - 1]["validation"],
        "final_test_metrics": standardised_final,
        "segmented_final_test_metrics": segmented_final,
        "training_representation": "ground truth + merged segmented augmentation",
    }
    joblib.dump(bundle, output)
    report = {
        "method": "ExtraTreesClassifier",
        "selection_metric": "validation macro F1; balanced accuracy and accuracy tie-breakers",
        "feature_columns": FEATURE_COLUMNS,
        "training_class_counts": dict(sorted(Counter(refit_y.tolist()).items())),
        "class_counts": {
            "final_test": dict(
                sorted(Counter(gt["final_test"][1].tolist()).items())
            )
        },
        "selected_candidate": best_index,
        "validation": candidates[best_index - 1]["validation"],
        "selected_parameters": parameters,
        "parameters": parameters,
        "candidate_results": candidates,
        "standardised_final_test": standardised_final,
        "final_test": standardised_final,
        "segmented_final_test": segmented_final,
        "model_path": str(output),
    }
    report_path = PROJECT_ROOT / "models" / "extra_trees_hybrid_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved candidate: {output}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
