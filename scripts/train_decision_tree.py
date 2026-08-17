"""Train, validate and save the contour-feature Decision Tree classifier.

All compared classifiers use the same external development, validation and
final-test contour tables. This prevents a misleading comparison caused by a
different feature extractor or a different number of final-test tracks.
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
from cloud_chamber.ml.contour_dataset import (
    build_segmented_feature_csv,
    load_feature_csv,
)
from cloud_chamber.ml.member_models.decision_tree import (
    PARAMETER_CANDIDATES,
    build_training_report,
    evaluate,
    save_model,
    train_candidates,
)

SPLITS = ("development", "validation", "final_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contour-feature Decision Tree")
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
        build_segmented_feature_csv(
            annotation_path,
            feature_path,
            config,
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

    split_tables = {}
    class_counts = {}
    for split in SPLITS:
        annotation_path = args.split_root / split / "annotations_coco.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Required dataset annotation file not found: {annotation_path}"
            )
        matrix, labels = load_feature_table(
            annotation_path,
            args.feature_dir / f"{split}.csv",
            allowed_labels,
            rebuild_features=args.rebuild_features,
            config=config,
        )
        split_tables[split] = (matrix, labels)
        class_counts[split] = dict(sorted(Counter(labels).items()))
        print(f"{split}: {len(labels)} tracks {class_counts[split]}")

    development_x, development_y = split_tables["development"]
    validation_matrix, validation_labels = split_tables["validation"]
    final_test_matrix, final_test_labels = split_tables["final_test"]

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
