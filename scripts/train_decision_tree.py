"""Train, validate and save the contour-feature Decision Tree classifier.

This script is intentionally explicit about the dataset sources. The training
set is the union of the primary and external development splits, so the model is
fit on both dataset>primary_dataset_split>development and
dataset>external_dataset_split>development before validation is used to select
its parameter candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.config import load_config
from cloud_chamber.ml.contour_dataset import build_feature_csv, load_feature_csv
from cloud_chamber.ml.member_models.decision_tree import (
    PARAMETER_CANDIDATES,
    build_training_report,
    combine_feature_tables,
    evaluate,
    save_model,
    train_candidates,
)

def primary_development_annotations(
    source_annotations: Path,
    split_name: str,
) -> Path:
    """Return a temporary filtered COCO JSON if needed, otherwise the original path."""
    with source_annotations.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    image_ids = {
        int(image["id"])
        for image in payload.get("images", [])
        if image.get("split") == split_name
    }
    if not image_ids:
        raise ValueError(f"No images in {source_annotations} for split={split_name!r}")

    filtered = {
        **payload,
        "images": [
            image for image in payload.get("images", []) if int(image["id"]) in image_ids
        ],
        "annotations": [
            annotation
            for annotation in payload.get("annotations", [])
            if int(annotation["image_id"]) in image_ids
        ],
    }
    temp_path = source_annotations.with_name(f"{split_name}_annotations_coco.json")
    temp_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    return temp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contour-feature Decision Tree")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--primary-split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "primary_dataset_split",
    )
    parser.add_argument(
        "--external-split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "external_dataset_split",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "decision_tree",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "decision_tree_classifier.joblib",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def load_feature_table(
    annotation_path: Path,
    feature_path: Path,
    allowed_labels: set[str],
    *,
    rebuild_features: bool,
    config: dict,
) -> tuple[object, object]:
    """Build a feature CSV if needed and return its matrix and labels."""
    if rebuild_features or not feature_path.exists():
        print(f"Building labelled contour features: {annotation_path}")
        build_feature_csv(
            annotation_path,
            feature_path,
            config["enhancement"],
            allowed_labels=allowed_labels,
        )
    matrix, labels = load_feature_csv(feature_path, allowed_labels)
    return matrix, labels


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["random_seed"])
    allowed_labels = set(config["classification"]["supported_classes"])
    args.feature_dir.mkdir(parents=True, exist_ok=True)

    primary_annotations = args.primary_split_root / "annotations_coco.json"
    external_development = args.external_split_root / "development" / "annotations_coco.json"
    external_validation = args.external_split_root / "validation" / "annotations_coco.json"
    external_final_test = args.external_split_root / "final_test" / "annotations_coco.json"

    for required in (
        primary_annotations,
        external_development,
        external_validation,
        external_final_test,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required dataset annotation file not found: {required}")

    primary_development = primary_development_annotations(
        primary_annotations,
        "development",
    )

    primary_dev_matrix, primary_dev_labels = load_feature_table(
        primary_development,
        args.feature_dir / "primary_development.csv",
        allowed_labels,
        rebuild_features=args.rebuild_features,
        config=config,
    )
    external_dev_matrix, external_dev_labels = load_feature_table(
        external_development,
        args.feature_dir / "external_development.csv",
        allowed_labels,
        rebuild_features=args.rebuild_features,
        config=config,
    )
    validation_matrix, validation_labels = load_feature_table(
        external_validation,
        args.feature_dir / "external_validation.csv",
        allowed_labels,
        rebuild_features=args.rebuild_features,
        config=config,
    )
    final_test_matrix, final_test_labels = load_feature_table(
        external_final_test,
        args.feature_dir / "external_final_test.csv",
        allowed_labels,
        rebuild_features=args.rebuild_features,
        config=config,
    )

    development_x, development_y = combine_feature_tables(
        [
            (primary_dev_matrix, primary_dev_labels),
            (external_dev_matrix, external_dev_labels),
        ]
    )
    class_counts = {
        "primary_development": dict(sorted(Counter(primary_dev_labels).items())),
        "external_development": dict(sorted(Counter(external_dev_labels).items())),
        "development": dict(sorted(Counter(development_y).items())),
        "validation": dict(sorted(Counter(validation_labels).items())),
        "final_test": dict(sorted(Counter(final_test_labels).items())),
    }

    print(
        "Training split: primary_dataset_split/development + "
        "external_dataset_split/development"
    )
    print(
        f"development rows={len(development_y)} "
        f"primary={len(primary_dev_labels)} external={len(external_dev_labels)}"
    )
    print(f"validation rows={len(validation_labels)}")
    print(f"final_test rows={len(final_test_labels)}")

    candidates, best_index, best_model = train_candidates(
        development_x,
        development_y,
        validation_matrix,
        validation_labels,
        seed,
    )

    final_metrics = evaluate(best_model, final_test_matrix, final_test_labels)
    selected_validation = candidates[best_index - 1]["validation"]
    classes = tuple(str(value) for value in best_model.classes_)
    bundle = save_model(
        best_model,
        args.output,
        classes=classes,
        selected_parameters=candidates[best_index - 1]["parameters"],
        random_seed=seed,
        validation_metrics=selected_validation,
        final_test_metrics=final_metrics,
    )

    report = build_training_report(
        candidates,
        best_index,
        class_counts,
        final_metrics,
        args.output,
    )
    report_path = args.output.with_name("decision_tree_training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}")
    print(f"Saved evidence report: {report_path}")
    print(f"Selected candidate: {best_index}/{len(PARAMETER_CANDIDATES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
