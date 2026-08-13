"""Create labelled contour-feature tables from Müller COCO instance masks."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cloud_chamber.enhancement import enhance_image
from cloud_chamber.features import extract_track_features
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS


def decode_uncompressed_rle(segmentation: dict) -> np.ndarray:
    """Decode the COCO column-major run-length format used by this dataset."""
    height, width = (int(value) for value in segmentation["size"])
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    foreground = False
    for count in segmentation["counts"]:
        end = position + int(count)
        if foreground:
            flat[position:end] = 255
        position = end
        foreground = not foreground
    if position != flat.size:
        raise ValueError("Invalid COCO RLE: runs do not match mask dimensions")
    # COCO RLE is column-major, but OpenCV drawing functions require a
    # C-contiguous destination buffer. ascontiguousarray preserves the decoded
    # pixels while changing only their in-memory layout.
    return np.ascontiguousarray(flat.reshape((height, width), order="F"))


def build_feature_csv(
    annotation_path: str | Path,
    output_path: str | Path,
    enhancement_settings: dict,
    allowed_labels: set[str] | None = None,
) -> dict[str, int]:
    """Measure one labelled feature row for every valid COCO instance."""
    annotation_path = Path(annotation_path).resolve()
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data["images"]}
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    grouped = defaultdict(list)
    for annotation in data["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)

    rows: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for image_id, annotations in grouped.items():
        image_record = images[image_id]
        image_path = (annotation_path.parent / image_record["file_name"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read linked image: {image_path}")
        enhanced = enhance_image(image, enhancement_settings).enhanced

        for annotation in annotations:
            mask = decode_uncompressed_rle(annotation["segmentation"])
            measured = extract_track_features(mask, enhanced, minimum_area=1.0)
            if not measured:
                continue
            # Each converted COCO annotation represents one connected instance.
            track = max(measured, key=lambda item: item.area_pixels)
            label = categories[int(annotation["category_id"])]
            # Exclude dataset-specific labels that are outside the project's
            # documented particle taxonomy; never silently relabel them.
            if allowed_labels is not None and label not in allowed_labels:
                continue
            row = {
                "image_id": image_id,
                "annotation_id": int(annotation["id"]),
                "label": label,
                **{column: getattr(track, column) for column in FEATURE_COLUMNS},
            }
            rows.append(row)
            counts[label] += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "annotation_id", "label", *FEATURE_COLUMNS]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return dict(sorted(counts.items()))


def load_feature_csv(
    path: str | Path,
    allowed_labels: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached labelled feature table in the fixed column order."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if allowed_labels is None or row["label"] in allowed_labels
    ]
    matrix = np.asarray(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in selected],
        dtype=np.float64,
    )
    labels = np.asarray([row["label"] for row in selected], dtype=object)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Feature table contains NaN or infinity: {path}")
    return matrix, labels
