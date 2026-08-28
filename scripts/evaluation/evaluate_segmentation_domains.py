"""Evaluate final segmentation separately on both held-out dataset domains.

The primary dataset supplies bounding-box ground truth, whereas the external
Muller dataset supplies pixel masks.  Results are therefore kept in separate
blocks.  A detection matches an annotation when at least half of the detected
contour pixels lie inside that annotation.  Greedy one-to-one assignment stops
several fragments of one particle from being counted as several true positives.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import _process_tiled_pipeline_image
from cloud_chamber.core.config import load_config
from cloud_chamber.ml.contour_dataset import annotation_to_mask


MATCH_THRESHOLD = 0.5
DATASETS = {
    "primary": {
        "annotation_path": (
            PROJECT_ROOT
            / "dataset"
            / "primary_dataset_split"
            / "final_test_annotations_coco.json"
        ),
        "profile": "primary_full_chamber",
        "annotation_type": "bounding boxes",
    },
    "external_muller": {
        "annotation_path": (
            PROJECT_ROOT
            / "dataset"
            / "external_dataset_split"
            / "final_test"
            / "annotations_coco.json"
        ),
        "profile": "external_muller",
        "annotation_type": "pixel masks",
    },
}


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    return (0, 0, 0, 0) if points is None else cv2.boundingRect(points)


def _intersects(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _prediction_mask(
    contour: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, width, height = cv2.boundingRect(contour)
    local = contour.copy()
    local[:, 0, 0] -= x
    local[:, 0, 1] -= y
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [local], -1, 255, cv2.FILLED)
    return mask > 0, (x, y, width, height)


def _evaluate_dataset(
    annotation_path: Path,
    config: dict,
    allowed_labels: set[str],
    *,
    profile: str,
    annotation_type: str,
) -> dict:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in payload["categories"]}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        label = categories[int(annotation["category_id"])]
        if label in allowed_labels:
            grouped[int(annotation["image_id"])].append(annotation)

    total_predictions = 0
    total_annotations = 0
    matched_pairs = 0
    overlap_sum = 0.0
    fragmented_annotations = 0
    annotations_with_candidates = 0
    candidate_detection_sum = 0
    per_class_annotations: dict[str, int] = defaultdict(int)
    per_class_matches: dict[str, int] = defaultdict(int)

    for image_record in payload["images"]:
        image_id = int(image_record["id"])
        image_path = (annotation_path.parent / image_record["file_name"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read linked image: {image_path}")

        annotations = grouped.get(image_id, [])
        truth = []
        for annotation in annotations:
            mask = annotation_to_mask(annotation, image.shape[:2])
            label = categories[int(annotation["category_id"])]
            truth.append((mask > 0, _bbox_from_mask(mask), label))
            per_class_annotations[label] += 1

        result = _process_tiled_pipeline_image(image, config)
        contours = result["segmentation"].contours
        total_predictions += len(contours)
        total_annotations += len(truth)

        candidates: list[tuple[float, int, int]] = []
        candidate_counts: dict[int, int] = defaultdict(int)
        for prediction_id, contour in enumerate(contours):
            predicted, prediction_box = _prediction_mask(contour)
            x, y, width, height = prediction_box
            predicted_pixels = max(int(np.count_nonzero(predicted)), 1)
            for truth_id, (truth_mask, truth_box, _) in enumerate(truth):
                if not _intersects(prediction_box, truth_box):
                    continue
                overlap = float(
                    np.count_nonzero(
                        predicted & truth_mask[y : y + height, x : x + width]
                    )
                    / predicted_pixels
                )
                if overlap >= MATCH_THRESHOLD:
                    candidates.append((overlap, prediction_id, truth_id))
                    candidate_counts[truth_id] += 1

        annotations_with_candidates += len(candidate_counts)
        fragmented_annotations += sum(count > 1 for count in candidate_counts.values())
        candidate_detection_sum += sum(candidate_counts.values())

        used_predictions: set[int] = set()
        used_truth: set[int] = set()
        for overlap, prediction_id, truth_id in sorted(candidates, reverse=True):
            if prediction_id in used_predictions or truth_id in used_truth:
                continue
            used_predictions.add(prediction_id)
            used_truth.add(truth_id)
            matched_pairs += 1
            overlap_sum += overlap
            per_class_matches[truth[truth_id][2]] += 1

    false_predictions = total_predictions - matched_pairs
    missed_annotations = total_annotations - matched_pairs
    precision = matched_pairs / max(total_predictions, 1)
    recall = matched_pairs / max(total_annotations, 1)
    beta_squared = 4.0
    f2 = (
        (1.0 + beta_squared) * precision * recall
        / max(beta_squared * precision + recall, 1e-12)
    )
    class_metrics = {}
    for label in sorted(per_class_annotations):
        annotations = per_class_annotations[label]
        matches = per_class_matches[label]
        class_metrics[label] = {
            "annotations": annotations,
            "matched_annotations": matches,
            "annotation_recall": matches / max(annotations, 1),
        }

    return {
        "profile": profile,
        "annotation_type": annotation_type,
        "matching_rule": (
            "greedy one-to-one; >= 50% of detection pixels inside annotation"
        ),
        "images": len(payload["images"]),
        "annotations": total_annotations,
        "detected_contours": total_predictions,
        "matched_pairs": matched_pairs,
        "false_contours": false_predictions,
        "missed_annotations": missed_annotations,
        "detection_precision": precision,
        "annotation_recall": recall,
        "f2": f2,
        "false_contours_per_image": false_predictions / max(len(payload["images"]), 1),
        "mean_detection_inside_annotation": overlap_sum / max(matched_pairs, 1),
        "fragmented_annotations": fragmented_annotations,
        "fragmentation_rate_among_detected_annotations": (
            fragmented_annotations / max(annotations_with_candidates, 1)
        ),
        "mean_candidate_detections_per_detected_annotation": (
            candidate_detection_sum / max(annotations_with_candidates, 1)
        ),
        "per_class": class_metrics,
    }


def main() -> int:
    config = load_config(PROJECT_ROOT / "config.yaml")
    allowed = set(config["classification"]["supported_classes"])
    results = {}
    for name, settings in DATASETS.items():
        print(f"Evaluating {name} final test...", flush=True)
        results[name] = _evaluate_dataset(
            settings["annotation_path"],
            config,
            allowed,
            profile=settings["profile"],
            annotation_type=settings["annotation_type"],
        )

    report = {
        "evaluation_scope": "held-out final-test segmentation by dataset domain",
        "combined_score_calculated": False,
        "reason_results_are_separate": (
            "Primary uses bounding boxes and external Muller uses pixel masks."
        ),
        "datasets": results,
    }
    print("\nFinal-test segmentation comparison")
    print(
        f"{'Dataset':<18} {'Precision':>10} {'Recall':>10} "
        f"{'F2':>10} {'False/image':>12}"
    )
    print("-" * 64)
    for name, metrics in report["datasets"].items():
        print(
            f"{name:<18} "
            f"{metrics['detection_precision'] * 100:>9.2f}% "
            f"{metrics['annotation_recall'] * 100:>9.2f}% "
            f"{metrics['f2'] * 100:>9.2f}% "
            f"{metrics['false_contours_per_image']:>12.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

