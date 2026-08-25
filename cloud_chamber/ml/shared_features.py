"""Shared contour-feature cache preparation for classical classifiers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from cloud_chamber.ml.contour_dataset import (
    build_feature_csv,
    build_segmented_feature_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "shared" / "muller"
LEGACY_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "mlp" / "muller"
SPLITS = ("development", "validation", "final_test")


def primary_split_annotations(source_path: Path, split: str) -> Path:
    """Create the split-specific primary COCO view used by all trainers."""
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
    output = source_path.with_name(f"{split}_annotations_coco.json")
    output.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    return output


def seed_from_legacy_cache(feature_dir: Path) -> bool:
    """Copy the old MLP-named cache once so existing work is not recomputed."""
    if feature_dir.exists() or not LEGACY_FEATURE_DIR.exists():
        return False
    feature_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LEGACY_FEATURE_DIR, feature_dir)
    print(f"Migrated shared feature cache: {LEGACY_FEATURE_DIR} -> {feature_dir}")
    return True


def ensure_shared_features(
    *,
    config: dict,
    external_split_root: Path,
    primary_split_root: Path,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    allowed_labels: set[str],
    rebuild: bool = False,
    rebuild_segmented: bool = False,
    primary_roi_profile: str = "primary_full_chamber",
) -> Path:
    """Build only missing/re requested canonical feature CSVs for every split."""
    feature_dir = Path(feature_dir)
    if not rebuild and not rebuild_segmented:
        seed_from_legacy_cache(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir = feature_dir / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    primary_annotations = primary_split_root / "annotations_coco.json"
    if not primary_annotations.is_file():
        raise FileNotFoundError(
            f"Primary annotation file not found: {primary_annotations}"
        )

    target_size = (
        int(config["spatial_scaling"]["processing_width"]),
        int(config["spatial_scaling"]["processing_height"]),
    )
    rebuild_segmented = rebuild or rebuild_segmented

    for split in SPLITS:
        external_annotations = external_split_root / split / "annotations_coco.json"
        if not external_annotations.is_file():
            raise FileNotFoundError(
                f"External annotation file not found: {external_annotations}"
            )
        primary_annotations_for_split = primary_split_annotations(
            primary_annotations, split
        )

        segmented_jobs = (
            (
                external_annotations,
                feature_dir / f"{split}.csv",
                "external_muller",
            ),
            (
                primary_annotations_for_split,
                feature_dir / f"primary_{split}.csv",
                primary_roi_profile,
            ),
        )
        for annotations, output, roi_profile in segmented_jobs:
            if rebuild_segmented or not output.exists():
                print(f"Building shared segmented features: {output.name}")
                build_segmented_feature_csv(
                    annotations,
                    output,
                    config,
                    allowed_labels=allowed_labels,
                    roi_profile_name=roi_profile,
                    merge_annotation_fragments=True,
                    use_production_pipeline=True,
                )

        ground_truth_jobs = (
            (
                external_annotations,
                ground_truth_dir / f"external_{split}.csv",
            ),
            (
                primary_annotations_for_split,
                ground_truth_dir / f"primary_{split}.csv",
            ),
        )
        for annotations, output in ground_truth_jobs:
            if rebuild or not output.exists():
                print(f"Building shared ground-truth features: {output.name}")
                build_feature_csv(
                    annotations,
                    output,
                    config["enhancement"],
                    allowed_labels=allowed_labels,
                    target_size=target_size,
                )

    return feature_dir
