"""Print contour diagnostics for one labelled external image."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.calibration import select_and_scale_roi
from cloud_chamber.config import load_config
from cloud_chamber.enhancement import enhance_image
from cloud_chamber.ml.contour_dataset import annotation_to_mask, _scale_label_mask
from cloud_chamber.segmentation import scale_pixel_parameters, segment_tracks


def _topology(mask: np.ndarray) -> tuple[int, int, int]:
    working = mask.copy()
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(working):
        eroded = cv2.erode(working, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(working, opened))
        working = eroded
    foreground = (skeleton > 0).astype(np.uint8)
    neighbours = cv2.filter2D(foreground, cv2.CV_16S, np.ones((3, 3), np.uint8)) - foreground
    endpoints = int(np.count_nonzero((foreground > 0) & (neighbours == 1)))
    branches = int(np.count_nonzero((foreground > 0) & (neighbours >= 3)))
    return int(np.count_nonzero(foreground)), endpoints, branches


def main() -> int:
    config = load_config(PROJECT_ROOT / "config.yaml")
    annotations_path = PROJECT_ROOT / "dataset/external_dataset_split/final_test/annotations_coco.json"
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_record = next(item for item in data["images"] if item["file_name"].endswith("2h24m19s 24979.jpg"))
    annotations = [item for item in data["annotations"] if int(item["image_id"]) == int(image_record["id"])]
    image_path = (annotations_path.parent / image_record["file_name"]).resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    source_shape = image.shape[:2]
    size = (int(config["spatial_scaling"]["processing_width"]), int(config["spatial_scaling"]["processing_height"]))
    normalisation = select_and_scale_roi(image, target_size=size)
    profile = config["segmentation"]["roi_profiles"]["external_muller"]
    settings = {**config["segmentation"], **{k: v for k, v in profile.items() if k not in ("left", "right", "top", "bottom")}}
    settings = scale_pixel_parameters(settings, int(config["spatial_scaling"]["pixel_parameter_reference_size"]), size)
    enhancement = enhance_image(normalisation.image, config["enhancement"], settings)
    segmented = segment_tracks(enhancement.segmentation_input, settings)
    truth_masks = [
        _scale_label_mask(annotation_to_mask(item, source_shape), normalisation.coordinates, size) > 0
        for item in annotations
    ]
    print("annotations", len(annotations), "detections", len(segmented.contours))
    for index, contour in enumerate(segmented.contours, start=1):
        mask = np.zeros_like(segmented.binary_mask)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        predicted = mask > 0
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        (_, _), (a, b), _ = cv2.minAreaRect(contour)
        aspect = max(a, b) / max(min(a, b), 1e-6)
        overlaps = [np.count_nonzero(predicted & truth) / max(np.count_nonzero(predicted), 1) for truth in truth_masks]
        coverages = [np.count_nonzero(predicted & truth) / max(np.count_nonzero(truth), 1) for truth in truth_masks]
        best = int(np.argmax(overlaps))
        skeleton, endpoints, branches = _topology(mask)
        print(
            index,
            "MATCH" if overlaps[best] >= 0.5 else "FALSE",
            f"inside={overlaps[best]:.2f}",
            f"coverage={coverages[best]:.2f}",
            f"area={area:.0f}",
            f"aspect={aspect:.2f}",
            f"solidity={area/max(hull_area,1):.2f}",
            f"complexity={perimeter/max(np.sqrt(area),1):.2f}",
            f"skeleton={skeleton}",
            f"endpoints={endpoints}",
            f"branches={branches}",
            f"box={cv2.boundingRect(contour)}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
