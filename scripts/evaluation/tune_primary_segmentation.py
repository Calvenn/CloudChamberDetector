"""Tune only the primary-domain classical segmentation profile."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.ui.shared_pipeline.page import (
    process_pipeline_image as _process_tiled_pipeline_image,
)
from cloud_chamber.core.config import load_config
from cloud_chamber.ml.contour_dataset import annotation_to_mask


GEOMETRY_PRESETS = (
    {
        "minimum_thin_area": 250,
        "minimum_thin_perimeter": 140,
        "minimum_thin_major_axis": 100,
        "minimum_thin_aspect_ratio": 3.5,
    },
    {
        "minimum_thin_area": 180,
        "minimum_thin_perimeter": 110,
        "minimum_thin_major_axis": 85,
        "minimum_thin_aspect_ratio": 3.0,
    },
    {
        "minimum_thin_area": 120,
        "minimum_thin_perimeter": 80,
        "minimum_thin_major_axis": 70,
        "minimum_thin_aspect_ratio": 2.5,
    },
)


def _records(split: str, allowed: set[str]) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    annotation_path = (
        PROJECT_ROOT
        / "dataset"
        / "primary_dataset_split"
        / f"{split}_annotations_coco.json"
    )
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in payload["categories"]}
    images = {int(item["id"]): item for item in payload["images"]}
    grouped: dict[int, list[dict]] = {}
    for annotation in payload["annotations"]:
        if categories[int(annotation["category_id"])] in allowed:
            grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    records = []
    for image_id, annotations in grouped.items():
        image = cv2.imread(
            str((annotation_path.parent / images[image_id]["file_name"]).resolve()),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise FileNotFoundError(images[image_id]["file_name"])
        masks = [annotation_to_mask(item, image.shape[:2]) > 0 for item in annotations]
        records.append((image, masks))
    return records


def _evaluate(records: list[tuple[np.ndarray, list[np.ndarray]]], config: dict) -> dict:
    contour_count = 0
    matched_contours = 0
    annotation_count = 0
    matched_annotations = 0
    for image, truth_masks in records:
        result = _process_tiled_pipeline_image(image, config)
        matched_ids: set[int] = set()
        annotation_count += len(truth_masks)
        for contour in result["segmentation"].contours:
            x, y, width, height = cv2.boundingRect(contour)
            local = contour.copy()
            local[:, 0, 0] -= x
            local[:, 0, 1] -= y
            predicted = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(predicted, [local], -1, 255, cv2.FILLED)
            predicted_pixels = predicted > 0
            pixel_count = max(int(np.count_nonzero(predicted_pixels)), 1)
            overlaps = [
                np.count_nonzero(
                    predicted_pixels & truth[y:y + height, x:x + width]
                ) / pixel_count
                for truth in truth_masks
            ]
            contour_count += 1
            if overlaps and max(overlaps) >= 0.5:
                matched_contours += 1
                matched_ids.add(int(np.argmax(overlaps)))
        matched_annotations += len(matched_ids)
    precision = matched_contours / max(contour_count, 1)
    recall = matched_annotations / max(annotation_count, 1)
    beta_squared = 4.0
    f2 = (
        (1.0 + beta_squared) * precision * recall
        / max(beta_squared * precision + recall, 1e-12)
    )
    return {
        "images": len(records),
        "annotations": annotation_count,
        "contours": contour_count,
        "matched_contours": matched_contours,
        "matched_annotations": matched_annotations,
        "particle_precision": precision,
        "annotation_recall": recall,
        "f2": f2,
        "false_contours_per_image": (
            contour_count - matched_contours
        ) / max(len(records), 1),
    }


def main() -> int:
    """Select primary-domain segmentation settings using validation data."""

    base = load_config(PROJECT_ROOT / "config.yaml")
    allowed = set(base["classification"]["supported_classes"])
    validation_records = _records("validation", allowed)
    candidates = []
    for threshold in (35, 45, 55):
        for geometry in GEOMETRY_PRESETS:
            settings = {
                "threshold_offset": threshold,
                "hysteresis_low_threshold_offset": 0 if threshold < 55 else 5,
                **geometry,
            }
            config = copy.deepcopy(base)
            profile = config["segmentation"]["roi_profiles"]["primary_full_chamber"]
            profile.update(settings)
            profile["artifact_filter_enabled"] = True
            profile["artifact_filter_model"] = "models/artifact_filter_primary.joblib"
            metrics = _evaluate(validation_records, config)
            candidates.append({"settings": settings, "validation": metrics})
            print(
                f"threshold={threshold}, major={geometry['minimum_thin_major_axis']}: "
                f"recall={metrics['annotation_recall']:.4f}, "
                f"precision={metrics['particle_precision']:.4f}, "
                f"F2={metrics['f2']:.4f}",
                flush=True,
            )
    selected = max(
        candidates,
        key=lambda item: (
            item["validation"]["f2"],
            item["validation"]["annotation_recall"],
            item["validation"]["particle_precision"],
        ),
    )
    final_config = copy.deepcopy(base)
    final_profile = final_config["segmentation"]["roi_profiles"]["primary_full_chamber"]
    final_profile.update(selected["settings"])
    final_profile["artifact_filter_enabled"] = True
    final_profile["artifact_filter_model"] = "models/artifact_filter_primary.joblib"
    final_test = _evaluate(_records("final_test", allowed), final_config)
    report = {
        "selection_rule": "validation F2, then recall, then precision",
        "external_profile_modified": False,
        "candidates": candidates,
        "selected_settings": selected["settings"],
        "selected_validation": selected["validation"],
        "final_test_with_primary_filter": final_test,
    }
    output = PROJECT_ROOT / "results" / "segmentation_ablation" / "primary_profile_tuning.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "final_test": final_test}, indent=2))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
