"""Train and evaluate Decision Tree using combined labelled datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.core.config import load_config
from cloud_chamber.ml.contour_dataset import build_segmented_feature_csv, load_feature_csv
from cloud_chamber.ml.member_models.decision_tree import (
    FEATURE_COLUMNS,
    PARAMETER_CANDIDATES,
    build_training_report,
    evaluate,
    make_model,
    save_model,
    train_candidates,
)
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS as SHARED_FEATURE_COLUMNS
from cloud_chamber.ml.shared_features import DEFAULT_FEATURE_DIR, ensure_shared_features

SPLITS = ("development", "validation", "final_test")


def parse_args() -> argparse.Namespace:
    """Read Decision Tree data, model and feature-cache locations."""

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
        default=DEFAULT_FEATURE_DIR,
        help="Canonical shared ground-truth and production-segmented feature tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "decision_tree_classifier.joblib",
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


def _load_pair(
    root: Path, split: str, labels: set[str], ground_truth: bool
) -> tuple[np.ndarray, np.ndarray]:
    paths = (
        (
            root / "ground_truth" / f"external_{split}.csv",
            root / "ground_truth" / f"primary_{split}.csv",
        )
        if ground_truth
        else (root / f"{split}.csv", root / f"primary_{split}.csv")
    )
    tables = [load_feature_csv(path, labels) for path in paths]
    return (
        np.concatenate([_decision_tree_matrix(item[0]) for item in tables], axis=0),
        np.concatenate([item[1] for item in tables], axis=0),
    )


def _decision_tree_matrix(raw_matrix: np.ndarray) -> np.ndarray:
    """Project the shared 16-column table onto the tree's named 10 columns."""
    indices = [SHARED_FEATURE_COLUMNS.index(column) for column in FEATURE_COLUMNS]
    return raw_matrix[:, indices]


def main() -> int:
    """Select, retrain and save the contour-feature Decision Tree."""

    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["random_seed"])
    allowed = set(config["classification"]["supported_classes"])
    ensure_shared_features(
        config=config,
        external_split_root=args.split_root,
        primary_split_root=args.primary_split_root,
        feature_dir=args.feature_dir,
        allowed_labels=allowed,
        rebuild=args.rebuild_features,
    )
    tables = {}
    hybrid_tables = {}
    counts = {}
    for split in SPLITS:
        ground_truth_x, ground_truth_y = _load_pair(
            args.feature_dir, split, allowed, True
        )
        segmented_x, segmented_y = _load_pair(
            args.feature_dir, split, allowed, False
        )
        tables[split] = (ground_truth_x, ground_truth_y)
        hybrid_tables[split] = (
            np.concatenate((ground_truth_x, segmented_x), axis=0),
            np.concatenate((ground_truth_y, segmented_y), axis=0),
        )
        counts[split] = {
            "ground_truth": dict(sorted(Counter(ground_truth_y).items())),
            "segmented_augmentation": dict(sorted(Counter(segmented_y).items())),
            "hybrid_training": dict(sorted(Counter(hybrid_tables[split][1]).items())),
        }
        print(
            f"{split}: {len(ground_truth_y)} ground-truth particles; "
            f"{len(segmented_y)} segmented augmentations"
        )

    candidates, best_index, model = train_candidates(
        *hybrid_tables["development"], *tables["validation"], seed
    )
    selected = candidates[best_index - 1]
    refit_x = np.concatenate(
        (hybrid_tables["development"][0], hybrid_tables["validation"][0]), axis=0
    )
    refit_y = np.concatenate(
        (hybrid_tables["development"][1], hybrid_tables["validation"][1]), axis=0
    )
    model = make_model(selected["parameters"], seed)
    model.fit(refit_x, refit_y)
    final_metrics = evaluate(model, *tables["final_test"])
    segmented_final_metrics = evaluate(
        model, *_load_pair(args.feature_dir, "final_test", allowed, False)
    )
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
        candidates, best_index, counts, final_metrics, args.output
    )
    report["feature_source"] = (
        "hybrid ground-truth masks and production-segmented contours; "
        "validation and final-test use ground-truth records"
    )
    report["segmented_final_test"] = segmented_final_metrics
    report_path = args.output.with_name("decision_tree_training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}")
    print(f"Saved evidence report: {report_path}")
    print(f"Selected candidate: {best_index}/{len(PARAMETER_CANDIDATES)}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
