"""Train, validate and save the contour-feature MLP classifier.

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
from cloud_chamber.ml.contour_dataset import build_feature_csv, load_feature_csv
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS


SPLITS = ("development", "validation", "final_test")

# 32 and 64 neurons are modest capacities for only ten input features. A
# second 32-neuron layer tests whether additional non-linearity helps without
# creating an unnecessarily large university-project model.
PARAMETER_CANDIDATES = (
    {"hidden_layer_sizes": (32,), "alpha": 0.0001},
    {"hidden_layer_sizes": (32,), "alpha": 0.001},
    {"hidden_layer_sizes": (64,), "alpha": 0.0001},
    {"hidden_layer_sizes": (64,), "alpha": 0.001},
    {"hidden_layer_sizes": (64, 32), "alpha": 0.0001},
    {"hidden_layer_sizes": (64, 32), "alpha": 0.001},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contour-feature MLP")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "external_dataset_split",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "muller",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models" / "mlp_classifier.joblib"
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def make_pipeline(parameters: dict, random_seed: int):
    """Create a scaler and MLP as one indivisible training pipeline."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            # Pixel measurements and ratios have very different magnitudes.
            # Zero-mean/unit-variance scaling stops large-valued area or
            # perimeter columns from dominating gradient updates.
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=parameters["hidden_layer_sizes"],
                    activation="relu",
                    solver="adam",
                    alpha=parameters["alpha"],
                    learning_rate_init=0.001,
                    # 500 is a ceiling. Early stopping monitors a stratified
                    # 10% subset of development data and retains the weights
                    # from the best internal validation score.
                    max_iter=500,
                    tol=0.0001,
                    n_iter_no_change=20,
                    early_stopping=True,
                    validation_fraction=0.1,
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
        model.fit(matrix, labels, mlp__sample_weight=weights)
        return "inverse-frequency sample weights"
    except TypeError:
        # Older scikit-learn releases did not accept MLP sample_weight. This
        # equivalent fallback samples training rows with replacement according
        # to those weights; validation and final-test rows remain untouched.
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

    split_tables = {}
    class_counts = {}
    for split in SPLITS:
        feature_path = args.feature_dir / f"{split}.csv"
        annotation_path = args.split_root / split / "annotations_coco.json"
        if args.rebuild_features or not feature_path.exists():
            print(f"Building labelled contour features: {split}")
            class_counts[split] = build_feature_csv(
                annotation_path,
                feature_path,
                config["enhancement"],
                allowed_labels=allowed_labels,
            )
        matrix, labels = load_feature_csv(feature_path, allowed_labels)
        split_tables[split] = (matrix, labels)
        class_counts.setdefault(split, dict(sorted(Counter(labels).items())))
        print(f"{split}: {len(labels)} tracks {class_counts[split]}")

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
                "hidden_layer_sizes": list(parameters["hidden_layer_sizes"]),
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
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
    # The untouched final-test split is used only after validation has selected
    # the candidate; it never influences model or parameter selection.
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
        "method": "StandardScaler + MLPClassifier",
        "selection_metric": "validation macro F1; balanced accuracy tie-breaker",
        "feature_columns": FEATURE_COLUMNS,
        "class_counts": class_counts,
        "candidate_results": candidates,
        "selected_candidate": best_index,
        "selected_parameters": {
            **best_parameters,
            "hidden_layer_sizes": list(best_parameters["hidden_layer_sizes"]),
        },
        "final_test": final_metrics,
        "model_path": str(args.output),
    }
    report_path = args.output.with_name("mlp_training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}")
    print(f"Saved evidence report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
