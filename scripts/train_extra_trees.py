"""Train, tune, evaluate, and save an Extra Trees contour classifier."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from cloud_chamber.config import load_config
from cloud_chamber.ml.contour_dataset import (
    build_feature_csv,
    load_feature_csv,
)
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS


# ============================================================
# EXTRA TREES CANDIDATES
# ============================================================
#
# Development data:
#   Used to train each candidate.
#
# Validation data:
#   Used to compare the candidates.
#
# Final-test data:
#   Used only after the best candidate is selected.
#

CANDIDATES = [
    {
        "name": "Model A - Baseline",
        "n_estimators": 100,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    {
        "name": "Model B - Balanced 300",
        "n_estimators": 300,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    },
    {
        "name": "Model C - Balanced 500 Depth 20",
        "n_estimators": 500,
        "max_features": "sqrt",
        "max_depth": 20,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    },
    {
        "name": "Model D - Balanced 500 Leaf 2",
        "n_estimators": 500,
        "max_features": "sqrt",
        "max_depth": 20,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_split(
    split_name: str,
    split_root: Path,
    feature_dir: Path,
    config: dict,
    allowed_labels: set[str],
):
    """
    Create the feature CSV if it does not already exist,
    then load its features and correct particle labels.
    """

    feature_path = feature_dir / f"{split_name}.csv"

    annotation_path = (
        split_root
        / split_name
        / "annotations_coco.json"
    )

    if not feature_path.exists():
        build_feature_csv(
            annotation_path,
            feature_path,
            config["enhancement"],
            allowed_labels=allowed_labels,
        )

    return load_feature_csv(
        feature_path,
        allowed_labels,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate(
    labels,
    predictions,
    class_names: list[str],
) -> dict:
    """
    Calculate classification results.

    These results are returned instead of being printed
    because they will be saved into the training report.
    """

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=class_names,
    )

    report = classification_report(
        labels,
        predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "sample_count": int(len(labels)),

        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_names,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_names,
                average="weighted",
                zero_division=0,
            )
        ),

        "class_names": class_names,

        "confusion_matrix": matrix.tolist(),

        "classification_report": report,
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main() -> int:

    # --------------------------------------------------------
    # 1. Load project configuration
    # --------------------------------------------------------

    config = load_config(
        PROJECT_ROOT / "config.yaml"
    )

    allowed_labels = set(
        config["classification"]["supported_classes"]
    )

    seed = int(
        config["project"]["random_seed"]
    )


    # --------------------------------------------------------
    # 2. Dataset locations
    # --------------------------------------------------------

    split_root = (
        PROJECT_ROOT
        / "dataset"
        / "external_dataset_split"
    )

    feature_dir = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "muller"
    )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # 3. Load development data
    # --------------------------------------------------------

    development_x, development_y = load_split(
        "development",
        split_root,
        feature_dir,
        config,
        allowed_labels,
    )


    # --------------------------------------------------------
    # 4. Load validation data
    # --------------------------------------------------------

    validation_x, validation_y = load_split(
        "validation",
        split_root,
        feature_dir,
        config,
        allowed_labels,
    )


    # --------------------------------------------------------
    # 5. Load final-test data
    # --------------------------------------------------------

    final_test_x, final_test_y = load_split(
        "final_test",
        split_root,
        feature_dir,
        config,
        allowed_labels,
    )


    # ========================================================
    # 6. TRAIN DIFFERENT EXTRA TREES CANDIDATES
    # ========================================================

    candidate_results = []

    best_model = None
    best_candidate = None
    best_macro_f1 = -1.0


    for candidate in CANDIDATES:

        # Create this candidate model
        current_model = ExtraTreesClassifier(
            n_estimators=candidate["n_estimators"],
            max_features=candidate["max_features"],
            max_depth=candidate["max_depth"],
            min_samples_leaf=candidate["min_samples_leaf"],
            class_weight=candidate["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )


        # ----------------------------------------------------
        # Train using DEVELOPMENT data
        # ----------------------------------------------------

        current_model.fit(
            development_x,
            development_y,
        )


        # ----------------------------------------------------
        # Check using VALIDATION data
        # ----------------------------------------------------

        validation_predictions = current_model.predict(
            validation_x
        )

        class_names = [
            str(value)
            for value in current_model.classes_
        ]


        # Main score used to compare models
        validation_macro_f1 = f1_score(
            validation_y,
            validation_predictions,
            labels=class_names,
            average="macro",
            zero_division=0,
        )


        validation_accuracy = accuracy_score(
            validation_y,
            validation_predictions,
        )


        validation_balanced_accuracy = (
            balanced_accuracy_score(
                validation_y,
                validation_predictions,
            )
        )


        # Save the result of this candidate
        candidate_results.append(
            {
                **candidate,

                "validation_accuracy":
                    float(validation_accuracy),

                "validation_balanced_accuracy":
                    float(validation_balanced_accuracy),

                "validation_macro_f1":
                    float(validation_macro_f1),
            }
        )


        # ----------------------------------------------------
        # Keep the candidate with highest validation Macro F1
        # ----------------------------------------------------

        if validation_macro_f1 > best_macro_f1:

            best_macro_f1 = validation_macro_f1

            best_model = current_model

            best_candidate = candidate


    # Make sure training succeeded
    if best_model is None or best_candidate is None:
        raise RuntimeError(
            "No Extra Trees model was successfully trained."
        )


    # ========================================================
    # 7. USE THE BEST MODEL
    # ========================================================

    model = best_model

    class_names = [
        str(value)
        for value in model.classes_
    ]


    # ========================================================
    # 8. VALIDATION RESULTS FOR SELECTED MODEL
    # ========================================================

    validation_predictions = model.predict(
        validation_x
    )

    validation_metrics = evaluate(
        validation_y,
        validation_predictions,
        class_names,
    )


    # ========================================================
    # 9. FINAL TEST
    # ========================================================
    #
    # The final-test split is used only after the best model
    # has already been selected using validation Macro F1.
    #

    final_predictions = model.predict(
        final_test_x
    )

    final_metrics = evaluate(
        final_test_y,
        final_predictions,
        class_names,
    )


    # ========================================================
    # 10. COUNT PARTICLE CLASSES
    # ========================================================

    class_counts = {

        "development": dict(
            sorted(
                Counter(
                    development_y.tolist()
                ).items()
            )
        ),

        "validation": dict(
            sorted(
                Counter(
                    validation_y.tolist()
                ).items()
            )
        ),

        "final_test": dict(
            sorted(
                Counter(
                    final_test_y.tolist()
                ).items()
            )
        ),
    }


    # ========================================================
    # 11. FEATURE IMPORTANCE
    # ========================================================

    feature_importance = sorted(
        (
            {
                "feature": feature,
                "importance": float(importance),
            }

            for feature, importance in zip(
                FEATURE_COLUMNS,
                model.feature_importances_,
                strict=True,
            )
        ),

        key=lambda item: item["importance"],

        reverse=True,
    )


    # ========================================================
    # 12. SAVE THE TRAINED MODEL
    # ========================================================

    output_path = (
        PROJECT_ROOT
        / "models"
        / "extra_trees_classifier.joblib"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    joblib.dump(
        {
            "model": model,

            "feature_columns":
                FEATURE_COLUMNS,

            "classes":
                tuple(model.classes_),

            "random_seed":
                seed,

            "selected_candidate":
                best_candidate,
        },
        output_path,
    )


    # ========================================================
    # 13. CREATE TRAINING REPORT
    # ========================================================

    report = {

        "method":
            "ExtraTreesClassifier",


        # ----------------------------------------------------
        # Dataset usage
        # ----------------------------------------------------

        "training_split":
            "development",

        "selection_split":
            "validation",

        "untouched_evaluation_split":
            "final_test",


        # ----------------------------------------------------
        # Tuning results
        # ----------------------------------------------------

        "candidate_results":
            candidate_results,

        "selected_candidate":
            best_candidate,

        "best_validation_macro_f1":
            float(best_macro_f1),


        # ----------------------------------------------------
        # Selected model
        # ----------------------------------------------------

        "parameters":
            model.get_params(
                deep=False
            ),

        "feature_columns":
            FEATURE_COLUMNS,

        "classes":
            class_names,


        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        "class_counts":
            class_counts,


        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        "validation":
            validation_metrics,

        "final_test":
            final_metrics,


        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        "feature_importance":
            feature_importance,


        # ----------------------------------------------------
        # Saved model
        # ----------------------------------------------------

        "model_path":
            str(output_path),
    }


    # ========================================================
    # 14. SAVE REPORT
    # ========================================================

    report_path = output_path.with_name(
        "extra_trees_training_report.json"
    )


    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


    return 0


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())