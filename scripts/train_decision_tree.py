"""Train and evaluate Decision Tree using combined labelled datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.config import load_config
from cloud_chamber.ml.contour_dataset import (
    build_feature_csv,
    build_segmented_feature_csv,
    load_feature_csv,
)
from cloud_chamber.ml.member_models.decision_tree import (
    PARAMETER_CANDIDATES,
    augment_feature_matrix,
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "decision_tree_classifier.joblib",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def _primary_split(source_path: Path, split: str, output_dir: Path) -> Path:
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
            {
                **image,
                # The filtered COCO file lives in the Decision-Tree cache
                # directory, not next to the primary images.  Make paths
                # absolute so both segmented and ground-truth builders read
                # the source images rather than a nonexistent cache-relative
                # location.
                "file_name": str((source_path.parent / image["file_name"]).resolve()),
            }
            for image in payload.get("images", [])
            if int(image["id"]) in image_ids
        ],
        "annotations": [
            annotation for annotation in payload.get("annotations", [])
            if int(annotation["image_id"]) in image_ids
        ],
    }
    # Keep generated annotations specific to this training job.  In
    # particular, do not overwrite the shared split annotation files used by
    # the other member models.
    output = output_dir / f"decision_tree_primary_{split}_annotations_coco.json"
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


def _ground_truth_features(
    annotations: Path,
    cache: Path,
    config: dict,
    labels: set[str],
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load clean instance-mask features for the Decision Tree only."""
    if rebuild or not cache.exists():
        build_feature_csv(
            annotations,
            cache,
            config["enhancement"],
            allowed_labels=labels,
        )
    return load_feature_csv(cache, labels)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["random_seed"])
    allowed = set(config["classification"]["supported_classes"])
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    primary_source = args.primary_split_root / "annotations_coco.json"

    segmented_tables = {}
    ground_truth_tables = {}
    counts = {}
    for split in SPLITS:
        external_segmented_x, external_segmented_y = _features(
            args.split_root / split / "annotations_coco.json",
            args.feature_dir / f"decision_tree_external_{split}.csv",
            config,
            allowed,
            args.rebuild_features,
            "external_muller",
        )
        primary_annotations = _primary_split(primary_source, split, args.feature_dir)
        primary_segmented_x, primary_segmented_y = _features(
            primary_annotations,
            args.feature_dir / f"decision_tree_primary_{split}.csv",
            config,
            allowed,
            args.rebuild_features,
            "primary_full_chamber",
        )

        external_ground_truth_x, external_ground_truth_y = _ground_truth_features(
            args.split_root / split / "annotations_coco.json",
            args.feature_dir / f"decision_tree_ground_truth_external_{split}.csv",
            config,
            allowed,
            args.rebuild_features,
        )
        primary_ground_truth_x, primary_ground_truth_y = _ground_truth_features(
            primary_annotations,
            args.feature_dir / f"decision_tree_ground_truth_primary_{split}.csv",
            config,
            allowed,
            args.rebuild_features,
        )

        segmented_x = augment_feature_matrix(
            np.concatenate((external_segmented_x, primary_segmented_x), axis=0)
        )
        segmented_y = np.concatenate(
            (external_segmented_y, primary_segmented_y), axis=0
        )
        ground_truth_x = augment_feature_matrix(
            np.concatenate((external_ground_truth_x, primary_ground_truth_x), axis=0)
        )
        ground_truth_y = np.concatenate(
            (external_ground_truth_y, primary_ground_truth_y), axis=0
        )
        segmented_tables[split] = (segmented_x, segmented_y)
        ground_truth_tables[split] = (ground_truth_x, ground_truth_y)
        counts[split] = {
            "segmented": {
                "external": dict(sorted(Counter(external_segmented_y).items())),
                "primary": dict(sorted(Counter(primary_segmented_y).items())),
                "combined": dict(sorted(Counter(segmented_y).items())),
            },
            "ground_truth": {
                "external": dict(sorted(Counter(external_ground_truth_y).items())),
                "primary": dict(sorted(Counter(primary_ground_truth_y).items())),
                "combined": dict(sorted(Counter(ground_truth_y).items())),
            },
        }
        print(
            f"{split}: {len(segmented_y)} segmented + {len(ground_truth_y)} "
            "ground-truth tracks"
        )

    # Fit only on contours produced by the same automatic segmentation used by
    # the web application. Ground-truth mask features are retained below for
    # audit purposes, but must not teach the model ideal shapes or annotation
    # errors that will not exist at inference time.
    development_x, development_y = segmented_tables["development"]
    candidates, best_index, model = train_candidates(
        development_x,
        development_y,
        *segmented_tables["validation"],
        seed,
    )
    final_metrics = evaluate(model, *segmented_tables["final_test"])
    ground_truth_final_metrics = evaluate(model, *ground_truth_tables["final_test"])
    selected = candidates[best_index - 1]
    save_model(
        model,
        args.output,
        classes=tuple(str(value) for value in model.classes_),
        selected_parameters=selected["parameters"],
        random_seed=seed,
        validation_metrics=selected["validation"],
        final_test_metrics=final_metrics,
    )
    report = build_training_report(
        candidates,
        best_index,
        counts,
        final_metrics,
        args.output,
        ground_truth_final_metrics=ground_truth_final_metrics,
    )
    report_path = args.output.with_name("decision_tree_training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}")
    print(f"Saved evidence report: {report_path}")
    print(f"Selected candidate: {best_index}/{len(PARAMETER_CANDIDATES)}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
