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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.core.config import load_config
from cloud_chamber.ml.contour_dataset import (
    load_feature_csv,
)
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS, SoftVotingMLPEnsemble
from cloud_chamber.ml.shared_features import DEFAULT_FEATURE_DIR, ensure_shared_features


SPLITS = ("development", "validation", "final_test")

# 32 and 64 neurons are modest capacities for only ten input features. A
# second 32-neuron layer tests whether additional non-linearity helps without
# creating an unnecessarily large university-project model.
PARAMETER_CANDIDATES = (
    *(
        {
            "hidden_layer_sizes": layers,
            "alpha": alpha,
            "activation": activation,
            "solver": solver,
            "balancing": balancing,
        }
        for layers in ((64,), (64, 32), (128, 64), (128, 64, 32))
        for alpha in (0.0001, 0.001, 0.01)
        for activation in ("relu", "tanh")
        # L-BFGS was substantially slower and did not win validation in the
        # expanded search; retain Adam for repeatable project retraining.
        for solver in ("adam",)
        for balancing in ("none", "sqrt_inverse_frequency")
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contour-feature MLP")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "external_dataset_split",
        help="External Müller dataset split root.",
    )
    parser.add_argument(
        "--primary-split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "primary_dataset_split",
        help="Primary dataset root containing annotations_coco.json.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
    )
    parser.add_argument(
        "--primary-roi-profile",
        default="primary_full_chamber",
        help="Segmentation profile applied to primary-dataset images.",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models" / "mlp_classifier.joblib"
    )
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--rebuild-segmented-features",
        action="store_true",
        help=(
            "Rebuild only automatic-segmentation feature CSVs while reusing "
            "unchanged ground-truth feature CSVs."
        ),
    )
    return parser.parse_args()


def primary_split_annotations(source_path: Path, split: str) -> Path:
    """Create a split-specific primary COCO file beside the source JSON.

    Keeping the filtered file beside the original preserves its relative image
    paths, such as ``development/images/example.jpg``.
    """
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
            image
            for image in payload.get("images", [])
            if int(image["id"]) in image_ids
        ],
        "annotations": [
            annotation
            for annotation in payload.get("annotations", [])
            if int(annotation["image_id"]) in image_ids
        ],
    }
    output_path = source_path.with_name(f"{split}_annotations_coco.json")
    output_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    return output_path


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
                    activation=parameters["activation"],
                    solver=parameters["solver"],
                    alpha=parameters["alpha"],
                    learning_rate_init=0.001,
                    # 500 is a ceiling. Early stopping monitors a stratified
                    # 10% subset of development data and retains the weights
                    # from the best internal validation score.
                    max_iter=1000,
                    tol=0.0001,
                    n_iter_no_change=20,
                    early_stopping=parameters["solver"] == "adam",
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


def fit_with_balancing(
    model, matrix: np.ndarray, labels: np.ndarray, seed: int, strategy: str
):
    """Use sample weights, with deterministic weighted resampling as fallback."""
    if strategy == "none":
        model.fit(matrix, labels)
        return "none"
    weights = balanced_sample_weights(labels)
    if strategy == "sqrt_inverse_frequency":
        # Moderate imbalance correction avoids making the rare V-track class
        # as influential as alpha despite having far fewer distinct examples.
        weights = np.sqrt(weights)
    try:
        model.fit(matrix, labels, mlp__sample_weight=weights)
        return strategy.replace("_", " ") + " sample weights"
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
    ensure_shared_features(
        config=config,
        external_split_root=args.split_root,
        primary_split_root=args.primary_split_root,
        feature_dir=args.feature_dir,
        allowed_labels=allowed_labels,
        rebuild=args.rebuild_features,
        rebuild_segmented=args.rebuild_segmented_features,
        primary_roi_profile=args.primary_roi_profile,
    )

    primary_annotations = args.primary_split_root / "annotations_coco.json"
    if not primary_annotations.is_file():
        raise FileNotFoundError(
            f"Primary annotation file not found: {primary_annotations}"
        )

    # Evaluation always uses one complete ground-truth particle per annotation.
    # Training additionally sees the corresponding automatic-segmentation
    # representation so deployment remains robust to realistic contour errors.
    split_tables = {}
    hybrid_training_tables = {}
    class_counts = {}
    for split in SPLITS:
        external_annotations = args.split_root / split / "annotations_coco.json"
        if not external_annotations.is_file():
            raise FileNotFoundError(
                f"External annotation file not found: {external_annotations}"
            )
        primary_filtered = primary_split_annotations(primary_annotations, split)

        external_segmented_x, external_segmented_y = load_feature_csv(
            args.feature_dir / f"{split}.csv", allowed_labels
        )
        primary_segmented_x, primary_segmented_y = load_feature_csv(
            args.feature_dir / f"primary_{split}.csv", allowed_labels
        )
        ground_truth_dir = args.feature_dir / "ground_truth"
        external_x, external_y = load_feature_csv(
            ground_truth_dir / f"external_{split}.csv", allowed_labels
        )
        primary_x, primary_y = load_feature_csv(
            ground_truth_dir / f"primary_{split}.csv", allowed_labels
        )
        evaluation_x = np.concatenate((external_x, primary_x), axis=0)
        evaluation_y = np.concatenate((external_y, primary_y), axis=0)
        segmented_x = np.concatenate(
            (external_segmented_x, primary_segmented_x), axis=0
        )
        segmented_y = np.concatenate(
            (external_segmented_y, primary_segmented_y), axis=0
        )
        split_tables[split] = (evaluation_x, evaluation_y)
        hybrid_training_tables[split] = (
            np.concatenate((evaluation_x, segmented_x), axis=0),
            np.concatenate((evaluation_y, segmented_y), axis=0),
        )
        class_counts[split] = {
            "ground_truth": dict(sorted(Counter(evaluation_y).items())),
            "segmented_augmentation": dict(sorted(Counter(segmented_y).items())),
            "hybrid_training": dict(
                sorted(Counter(hybrid_training_tables[split][1]).items())
            ),
        }
        print(
            f"{split}: {len(evaluation_y)} standardised particles; "
            f"{len(segmented_y)} segmented augmentations"
        )

    development_x, development_y = hybrid_training_tables["development"]
    validation_x, validation_y = split_tables["validation"]
    candidates = []
    best = None
    for index, parameters in enumerate(PARAMETER_CANDIDATES, start=1):
        model = make_pipeline(parameters, seed)
        balancing = fit_with_balancing(
            model, development_x, development_y, seed, parameters["balancing"]
        )
        metrics = evaluate(model, validation_x, validation_y)
        record = {
            "candidate": index,
            "parameters": {
                **parameters,
                "hidden_layer_sizes": list(parameters["hidden_layer_sizes"]),
                "learning_rate_init": 0.001,
                "max_iter": 1000,
                "early_stopping": parameters["solver"] == "adam",
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
        # Treat all particle classes equally. Raw accuracy is dominated by the
        # common alpha/proton records and previously selected models with very
        # weak V-track recall. Accuracy is retained only as the final tie-break.
        score = (
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            metrics["accuracy"],
        )
        if best is None or score > best[0]:
            best = (score, parameters, model, balancing, index)

    assert best is not None
    _, best_parameters, _, _, best_index = best
    # Hyperparameters are now fixed, so use all non-test labels for the final
    # fit. The held-out final-test split remains untouched until this point.
    validation_train_x, validation_train_y = hybrid_training_tables["validation"]
    refit_x = np.concatenate((development_x, validation_train_x), axis=0)
    refit_y = np.concatenate((development_y, validation_train_y), axis=0)
    ensemble_members = []
    balancing_methods = []
    for member_seed in range(seed, seed + 5):
        member = make_pipeline(best_parameters, member_seed)
        balancing_methods.append(
            fit_with_balancing(
                member,
                refit_x,
                refit_y,
                member_seed,
                best_parameters["balancing"],
            )
        )
        ensemble_members.append(member)
    best_model = SoftVotingMLPEnsemble(ensemble_members)
    balancing = balancing_methods[0]
    # The untouched final-test split is used only after validation has selected
    # the candidate; it never influences model or parameter selection.
    final_x, final_y = split_tables["final_test"]
    final_metrics = evaluate(best_model, final_x, final_y)
    selected_validation = candidates[best_index - 1]["validation"]

    baseline = next(
        (
            candidate
            for candidate in candidates
            if tuple(candidate["parameters"]["hidden_layer_sizes"]) == (64,)
            and candidate["parameters"]["activation"] == "relu"
            and candidate["parameters"]["balancing"] == "none"
        ),
        None,
    )
    comparison = [candidates[best_index - 1]]
    if baseline is not None and baseline["candidate"] != best_index:
        comparison.append(baseline)

    print("\nMLP setting comparison (validation evidence)")
    print(
        f"{'MLP setting':<24} {'Balancing':<31} "
        f"{'Macro F1':>10} {'V-track recall':>15}"
    )
    print("-" * 84)
    for candidate in comparison:
        parameters = candidate["parameters"]
        validation = candidate["validation"]
        layers = ", ".join(
            str(value) for value in parameters["hidden_layer_sizes"]
        )
        activation = (
            "ReLU"
            if parameters["activation"] == "relu"
            else str(parameters["activation"]).title()
        )
        balancing_label = (
            "Square-root inverse frequency"
            if parameters["balancing"] == "sqrt_inverse_frequency"
            else "None"
        )
        v_track_recall = validation["classification_report"]["v_track"][
            "recall"
        ]
        print(
            f"{f'MLP ({layers}), {activation}':<24} "
            f"{balancing_label:<31} "
            f"{validation['macro_f1']:>9.2%} {v_track_recall:>14.2%}"
        )

    bundle = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": tuple(str(value) for value in best_model.classes_),
        "selected_parameters": best_parameters,
        "random_seed": seed,
        "balancing": balancing,
        "ensemble_members": len(ensemble_members),
        "validation_metrics": selected_validation,
        "final_test_metrics": final_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)

    report = {
        "method": "StandardScaler + MLPClassifier",
        "feature_source": (
            "hybrid ground-truth particle masks and merged automatic "
            "segmentation contours; validation and final-test use one "
            "ground-truth record per annotation"
        ),
        "selection_metric": (
            "validation macro F1; balanced accuracy and accuracy tie-breakers"
        ),
        "final_fit_source": "development + validation after hyperparameter selection",
        "ensemble_members": len(ensemble_members),
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

