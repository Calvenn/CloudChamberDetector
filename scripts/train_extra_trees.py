"""Train, tune, evaluate, and save an Extra Trees contour classifier."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from itertools import product
from time import perf_counter

import cv2
import joblib
import numpy as np
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
    decode_uncompressed_rle,
    load_feature_csv,
)
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import extract_track_features
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

GRID_SEARCH_VALUES = {
    "n_estimators": [100, 300, 500, 1000],
    "max_depth": [None, 20, 30, 40],
    "min_samples_split": [2, 4, 8],
}

V_TRACK_EXTRA_MULTIPLIER = 5
PROTON_EXTRA_MULTIPLIER = 3


def build_grid_candidates() -> list[dict]:
    """Return the Extra Trees configurations evaluated on the validation split."""
    return [
        {
            "name": (
                f"Grid n_estimators={n_estimators}, max_depth={max_depth}, "
                f"min_samples_split={min_samples_split}"
            ),
            "n_estimators": n_estimators,
            "max_features": "sqrt",
            "max_depth": max_depth,
            "min_samples_leaf": 1,
            "min_samples_split": min_samples_split,
            "criterion": "gini",
            "class_weight": "balanced_subsample",
        }
        for n_estimators, max_depth, min_samples_split in product(
            GRID_SEARCH_VALUES["n_estimators"],
            GRID_SEARCH_VALUES["max_depth"],
            GRID_SEARCH_VALUES["min_samples_split"],
        )
    ]

# ============================================================
# LOAD DATASET
# ============================================================

def build_primary_feature_csv_in_memory(
    coco_data: dict,
    parent_dir: Path,
    output_path: Path,
    enhancement_settings: dict,
    allowed_labels: set[str] | None = None,
):
    """Build feature CSV directly from in-memory COCO data, avoiding writing COCO JSON files to disk."""
    images = {int(item["id"]): item for item in coco_data["images"]}
    categories = {int(item["id"]): item["name"] for item in coco_data["categories"]}
    grouped = defaultdict(list)
    for annotation in coco_data["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)

    rows = []
    for image_id, annotations in grouped.items():
        image_record = images[image_id]
        image_path = (parent_dir / image_record["file_name"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read linked image: {image_path}")
        enhanced = enhance_image(image, enhancement_settings).enhanced

        for annotation in annotations:
            mask = decode_uncompressed_rle(annotation["segmentation"])
            measured = extract_track_features(mask, enhanced, minimum_area=1.0)
            if not measured:
                continue
            track = max(measured, key=lambda item: item.area_pixels)
            label = categories[int(annotation["category_id"])]
            if allowed_labels is not None and label not in allowed_labels:
                continue
            row = {
                "image_id": image_id,
                "annotation_id": int(annotation["id"]),
                "label": label,
                **{column: getattr(track, column) for column in FEATURE_COLUMNS},
            }
            rows.append(row)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "annotation_id", "label", *FEATURE_COLUMNS]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_split(
    split_name: str,
    split_root: Path,
    feature_dir: Path,
    config: dict,
    allowed_labels: set[str],
    is_primary: bool = False,
    rebuild_features: bool = False,
):
    """
    Create the feature CSV if it does not already exist or if rebuild_features is True,
    then load its features and correct particle labels.
    """

    feature_path = feature_dir / f"{split_name}.csv"

    if rebuild_features or not feature_path.exists():
        if is_primary:
            coco_path = split_root / "annotations_coco.json"
            if not coco_path.exists():
                raise FileNotFoundError(f"Primary annotations not found at {coco_path}")

            with open(coco_path, "r", encoding="utf-8") as f:
                coco_data = json.load(f)

            split_images = [img for img in coco_data["images"] if img.get("split") == split_name]
            split_image_ids = {img["id"] for img in split_images}

            tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            image_cache = {}

            split_annotations = []
            for ann in coco_data["annotations"]:
                if ann["image_id"] in split_image_ids:
                    img_record = next(img for img in split_images if img["id"] == ann["image_id"])
                    w = img_record["width"]
                    h = img_record["height"]

                    if img_record["id"] not in image_cache:
                        image_path = (split_root / img_record["file_name"]).resolve()
                        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                        if image is None:
                            raise FileNotFoundError(f"Cannot read image: {image_path}")
                        image_cache[img_record["id"]] = enhance_image(image, config["enhancement"]).enhanced

                    enhanced = image_cache[img_record["id"]]

                    bbox = ann["bbox"]
                    x, y, bw, bh = [int(v) for v in bbox]
                    x1 = max(0, min(x, w - 1))
                    y1 = max(0, min(y, h - 1))
                    x2 = max(0, min(x + bw, w))
                    y2 = max(0, min(y + bh, h))

                    mask = np.zeros((h, w), dtype=np.uint8)
                    if x2 > x1 and y2 > y1:
                        roi = enhanced[y1:y2, x1:x2]
                        roi_th = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, tophat_kernel)
                        _, bin_roi = cv2.threshold(roi_th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        mask[y1:y2, x1:x2] = bin_roi
                    else:
                        mask[y1:y2, x1:x2] = 255

                    flat = mask.ravel(order='F')
                    counts = []
                    current_value = 0
                    current_count = 0
                    for val in flat:
                        if val == current_value:
                            current_count += 1
                        else:
                            counts.append(current_count)
                            current_value = val
                            current_count = 1
                    if current_count > 0:
                        counts.append(current_count)

                    split_ann = ann.copy()
                    split_ann["segmentation"] = {
                        "size": [h, w],
                        "counts": counts
                    }
                    split_annotations.append(split_ann)

            split_data = {
                "info": coco_data.get("info", {}),
                "categories": coco_data.get("categories", []),
                "images": split_images,
                "annotations": split_annotations,
            }

            build_primary_feature_csv_in_memory(
                split_data,
                split_root,
                feature_path,
                config["enhancement"],
                allowed_labels=allowed_labels,
            )
        else:
            annotation_path = (
                split_root
                / split_name
                / "annotations_coco.json"
            )
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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contour-feature Extra Trees classifier")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--split-root-external",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "external_dataset_split",
    )
    parser.add_argument(
        "--split-root-primary",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "primary_dataset_split",
    )
    parser.add_argument(
        "--feature-dir-external",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "muller_seg",
    )
    parser.add_argument(
        "--feature-dir-primary",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "primary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "extra_trees_classifier.joblib",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate(
    model,
    matrix: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Calculate classification results.

    These results are returned instead of being printed
    because they will be saved into the training report.
    """
    started = perf_counter()
    predictions = model.predict(matrix)
    elapsed_ms = (perf_counter() - started) * 1000.0

    cm = confusion_matrix(
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
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
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
        "mean_inference_ms_per_track": elapsed_ms / max(len(labels), 1),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main() -> int:

    # --------------------------------------------------------
    # 1. Load project configuration
    # --------------------------------------------------------

    args = parse_args()
    config = load_config(args.config)

    allowed_labels = set(
        config["classification"]["supported_classes"]
    )

    seed = int(
        config["project"]["random_seed"]
    )


    # --------------------------------------------------------
    # 2. Dataset locations
    # --------------------------------------------------------

    # External (Müller) Dataset
    split_root_external = args.split_root_external
    feature_dir_external = args.feature_dir_external
    feature_dir_external.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Primary Dataset
    split_root_primary = args.split_root_primary
    feature_dir_primary = args.feature_dir_primary
    feature_dir_primary.mkdir(
        parents=True,
        exist_ok=True,
    )

    # In-memory split will be loaded directly from main primary annotations if needed


    # --------------------------------------------------------
    # 3. Load development data
    # --------------------------------------------------------

    dev_x_ext, dev_y_ext = load_split(
        "development",
        split_root_external,
        feature_dir_external,
        config,
        allowed_labels,
        is_primary=False,
        rebuild_features=args.rebuild_features,
    )
    dev_x_pri, dev_y_pri = load_split(
        "development",
        split_root_primary,
        feature_dir_primary,
        config,
        allowed_labels,
        is_primary=True,
        rebuild_features=args.rebuild_features,
    )
    development_x = np.concatenate([dev_x_ext, dev_x_pri], axis=0)
    development_y = np.concatenate([dev_y_ext, dev_y_pri], axis=0)

    # Oversample minority classes (v_track and proton) in development data
    rng = np.random.RandomState(seed)
    v_indices = np.where(development_y == "v_track")[0]
    p_indices = np.where(development_y == "proton")[0]

    if len(v_indices) > 0 and len(p_indices) > 0:
        v_extra = rng.choice(
            v_indices,
            size=len(v_indices) * V_TRACK_EXTRA_MULTIPLIER,
            replace=True,
        )
        p_extra = rng.choice(
            p_indices,
            size=len(p_indices) * PROTON_EXTRA_MULTIPLIER,
            replace=True,
        )
        development_x_train = np.concatenate([development_x, development_x[v_extra], development_x[p_extra]], axis=0)
        development_y_train = np.concatenate([development_y, development_y[v_extra], development_y[p_extra]], axis=0)
    else:
        development_x_train = development_x
        development_y_train = development_y


    # --------------------------------------------------------
    # 4. Load validation data
    # --------------------------------------------------------

    val_x_ext, val_y_ext = load_split(
        "validation",
        split_root_external,
        feature_dir_external,
        config,
        allowed_labels,
        is_primary=False,
        rebuild_features=args.rebuild_features,
    )
    val_x_pri, val_y_pri = load_split(
        "validation",
        split_root_primary,
        feature_dir_primary,
        config,
        allowed_labels,
        is_primary=True,
        rebuild_features=args.rebuild_features,
    )
    validation_x = np.concatenate([val_x_ext, val_x_pri], axis=0)
    validation_y = np.concatenate([val_y_ext, val_y_pri], axis=0)


    # --------------------------------------------------------
    # 5. Load final-test data
    # --------------------------------------------------------

    test_x_ext, test_y_ext = load_split(
        "final_test",
        split_root_external,
        feature_dir_external,
        config,
        allowed_labels,
        is_primary=False,
        rebuild_features=args.rebuild_features,
    )
    test_x_pri, test_y_pri = load_split(
        "final_test",
        split_root_primary,
        feature_dir_primary,
        config,
        allowed_labels,
        is_primary=True,
        rebuild_features=args.rebuild_features,
    )
    final_test_x = np.concatenate([test_x_ext, test_x_pri], axis=0)
    final_test_y = np.concatenate([test_y_ext, test_y_pri], axis=0)


    # ========================================================
    # 6. TRAIN DIFFERENT EXTRA TREES CANDIDATES
    # ========================================================

    candidate_results = []

    best_model = None
    best_candidate = None
    best_macro_f1 = -1.0


    for candidate in build_grid_candidates():

        # Create this candidate model
        current_model = ExtraTreesClassifier(
            n_estimators=candidate["n_estimators"],
            max_features=candidate["max_features"],
            max_depth=candidate["max_depth"],
            min_samples_leaf=candidate["min_samples_leaf"],
            min_samples_split=candidate.get("min_samples_split", 2),
            criterion=candidate.get("criterion", "gini"),
            class_weight=candidate["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )


        # ----------------------------------------------------
        # Train using DEVELOPMENT data
        # ----------------------------------------------------

        current_model.fit(
            development_x_train,
            development_y_train,
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
        # Keep the candidate with the highest validation macro F1.
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

    validation_metrics = evaluate(
        model,
        validation_x,
        validation_y,
        class_names,
    )


    # ========================================================
    # 9. FINAL TEST
    # ========================================================
    #
    # The final-test split is used only after the best model
    # has already been selected using validation Macro F1.
    #

    final_metrics = evaluate(
        model,
        final_test_x,
        final_test_y,
        class_names,
    )


    # ========================================================
    # 10. COUNT PARTICLE CLASSES
    # ========================================================

    class_counts = {
        "development": dict(sorted(Counter(development_y.tolist()).items())),
        "validation": dict(sorted(Counter(validation_y.tolist()).items())),
        "final_test": dict(sorted(Counter(final_test_y.tolist()).items())),
    }


    # ========================================================
    # 11. FEATURE IMPORTANCE
    # ========================================================

    feat_importances = getattr(best_model, "feature_importances_", np.zeros(len(FEATURE_COLUMNS)))
    feature_importance = sorted(
        (
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(
                FEATURE_COLUMNS,
                feat_importances,
                strict=True,
            )
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )


    # ========================================================
    # 12. SAVE THE TRAINED MODEL
    # ========================================================

    output_path = args.output

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "classes": tuple(str(v) for v in model.classes_),
            "selected_parameters": best_candidate,
            "random_seed": seed,
            "validation_metrics": validation_metrics,
            "final_test_metrics": final_metrics,
        },
        output_path,
    )


    report = {
        "method": "ExtraTreesClassifier",
        "training_split": "development",
        "selection_split": "validation",
        "untouched_evaluation_split": "final_test",
        "candidate_results": candidate_results,
        "selected_candidate": best_candidate,
        "best_validation_macro_f1": float(best_macro_f1),
        "parameters": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in model.get_params(deep=False).items()
        },
        "feature_columns": FEATURE_COLUMNS,
        "classes": class_names,
        "class_counts": class_counts,
        "development_resampling": {
            "v_track_extra_multiplier": V_TRACK_EXTRA_MULTIPLIER,
            "proton_extra_multiplier": PROTON_EXTRA_MULTIPLIER,
        },
        "validation": validation_metrics,
        "final_test": final_metrics,
        "feature_importance": feature_importance,
        "model_path": str(output_path),
    }
    report_path = output_path.with_name("extra_trees_training_report.json")
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return 0


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())
