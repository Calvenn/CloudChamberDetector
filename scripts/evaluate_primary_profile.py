"""Select primary-image parameters using validation bounding-box labels."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_chamber.config import load_config
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.segmentation import segment_tracks


def overlap_fraction(box: tuple[int, int, int, int], truth: list[int]) -> float:
    """Return the proportion of a predicted box lying inside a labelled box."""
    x, y, width, height = box
    tx, ty, tw, th = truth
    intersection_width = max(0, min(x + width, tx + tw) - max(x, tx))
    intersection_height = max(0, min(y + height, ty + th) - max(y, ty))
    return intersection_width * intersection_height / max(width * height, 1)


def main() -> int:
    config = load_config()
    folder = ROOT / "dataset/primary_dataset_split/validation"
    labels = sorted((folder / "labels").glob("*.json"))
    # Keep the selected clean primary baseline fixed and test only conservative
    # thin-track exceptions. ``None`` represents no exception.
    candidates = [
        None,
        (150, 70, 30, 3.0),
        (100, 50, 25, 3.5),
        (120, 60, 30, 4.0),
        (140, 70, 35, 4.5),
        (160, 80, 40, 5.0),
        (180, 90, 45, 5.5),
    ]
    totals = {candidate: [0, 0, 0] for candidate in candidates}
    profile = config["segmentation"]["roi_profiles"]["primary_full_chamber"]

    for label_path in labels:
        record = json.loads(label_path.read_text(encoding="utf-8"))
        image = cv2.imread(str(folder / "images" / f"{label_path.stem}.jpg"))
        enhanced = enhance_image(image, config["enhancement"]).enhanced
        truth_boxes = [track["bbox"] for track in record["tracks"]]

        for candidate in candidates:
            settings = deepcopy(config["segmentation"])
            settings["threshold_offset"] = 40
            settings["closing_kernel"] = 3
            settings["minimum_object_area"] = 200
            settings["minimum_major_axis"] = 40
            settings["enable_thin_track_rule"] = candidate is not None
            if candidate is not None:
                (
                    settings["minimum_thin_area"],
                    settings["minimum_thin_perimeter"],
                    settings["minimum_thin_major_axis"],
                    settings["minimum_thin_aspect_ratio"],
                ) = candidate
            result = segment_tracks(enhanced, settings, profile)

            # Greedy one-to-one matching prevents several fragments from being
            # rewarded for overlapping one supplied particle box.
            possible = []
            for prediction_index, predicted in enumerate(result.bounding_boxes):
                for truth_index, truth in enumerate(truth_boxes):
                    overlap = overlap_fraction(predicted, truth)
                    if overlap >= 0.5:
                        possible.append((overlap, prediction_index, truth_index))
            matched_predictions = set()
            matched_truth = set()
            for _, prediction_index, truth_index in sorted(possible, reverse=True):
                if prediction_index not in matched_predictions and truth_index not in matched_truth:
                    matched_predictions.add(prediction_index)
                    matched_truth.add(truth_index)

            true_positive = len(matched_truth)
            totals[candidate][0] += true_positive
            totals[candidate][1] += len(result.bounding_boxes) - true_positive
            totals[candidate][2] += len(truth_boxes) - true_positive

    results = []
    for candidate, (tp, fp, fn) in totals.items():
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        results.append(
            {
                "thin_track_rule": candidate,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    results.sort(key=lambda row: (row["f1"], -row["false_positive"]), reverse=True)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
