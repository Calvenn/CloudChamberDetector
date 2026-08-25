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
from cloud_chamber.ml.contour_dataset import build_segmented_feature_csv, load_feature_csv
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


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["random_seed"])
    allowed = set(config["classification"]["supported_classes"])
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    primary_source = args.primary_split_root / "annotations_coco.json"

    tables = {}
    counts = {}
    for split in SPLITS:
        external_x, external_y = _features(
            args.split_root / split / "annotations_coco.json",
            args.feature_dir / f"{split}.csv",
            config,
            allowed,
            args.rebuild_features,
            "external_muller",
        )
        primary_x, primary_y = _features(
            _primary_split(primary_source, split),
            args.feature_dir / f"primary_{split}.csv",
            config,
            allowed,
            args.rebuild_features,
            "primary_full_chamber",
        )
        matrix = np.concatenate((external_x, primary_x), axis=0)
        labels = np.concatenate((external_y, primary_y), axis=0)
        tables[split] = (matrix, labels)
        counts[split] = {
            "external": dict(sorted(Counter(external_y).items())),
            "primary": dict(sorted(Counter(primary_y).items())),
            "combined": dict(sorted(Counter(labels).items())),
        }
        print(
            f"{split}: {len(labels)} combined tracks "
            f"(external={len(external_y)}, primary={len(primary_y)})"
        )

    candidates, best_index, model = train_candidates(
        *tables["development"], *tables["validation"], seed
    )
    final_metrics = evaluate(model, *tables["final_test"])
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
        candidates, best_index, counts, final_metrics, args.output
    )
    report_path = args.output.with_name("decision_tree_training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}")
    print(f"Saved evidence report: {report_path}")
    print(f"Selected candidate: {best_index}/{len(PARAMETER_CANDIDATES)}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
