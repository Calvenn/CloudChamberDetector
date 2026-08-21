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
from cloud_chamber.calibration import (
    DEFAULT_PROCESSING_SIZE,
    extract_roi,
    select_and_scale_roi,
)
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS
from cloud_chamber.segmentation import scale_pixel_parameters, segment_tracks


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


def annotation_to_mask(annotation: dict, image_shape: tuple[int, int]) -> np.ndarray:
    """Decode an instance mask or fall back to its COCO bounding box."""
    segmentation = annotation.get("segmentation")
    if segmentation is not None:
        return decode_uncompressed_rle(segmentation)

    if "bbox" not in annotation:
        raise KeyError(f"Annotation has neither segmentation nor bbox: {annotation}")

    x, y, width, height = [float(value) for value in annotation["bbox"]]
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x0 = int(np.clip(np.round(x), 0, image_shape[1] - 1))
    y0 = int(np.clip(np.round(y), 0, image_shape[0] - 1))
    x1 = int(np.clip(np.round(x + width), 0, image_shape[1]))
    y1 = int(np.clip(np.round(y + height), 0, image_shape[0]))
    if x1 <= x0 or y1 <= y0:
        return mask
    cv2.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 255, thickness=-1)
    return mask


def _scale_label_mask(
    mask: np.ndarray,
    coordinates,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Apply the image ROI transform to a categorical mask without blending."""
    selected, _ = extract_roi(mask, coordinates, target_size)
    return cv2.resize(selected, target_size, interpolation=cv2.INTER_NEAREST)


def build_feature_csv(
    annotation_path: str | Path,
    output_path: str | Path,
    enhancement_settings: dict,
    allowed_labels: set[str] | None = None,
    target_size: tuple[int, int] = DEFAULT_PROCESSING_SIZE,
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
        source_shape = image.shape
        normalisation = select_and_scale_roi(image, target_size=target_size)
        enhanced = enhance_image(
            normalisation.image, enhancement_settings
        ).enhanced

        for annotation in annotations:
            mask = annotation_to_mask(annotation, source_shape)
            mask = _scale_label_mask(
                mask, normalisation.coordinates, target_size
            )
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


def build_segmented_feature_csv(
    annotation_path: str | Path,
    output_path: str | Path,
    config: dict,
    allowed_labels: set[str] | None = None,
    minimum_label_overlap: float = 0.5,
    roi_profile_name: str = "external_muller",
) -> dict[str, int]:
    """Build labelled features from automatic contours, not perfect masks.

    Each automatically segmented contour receives the class of the COCO mask
    containing at least ``minimum_label_overlap`` of its pixels. Unmatched
    detections are excluded because the classifier has no background class;
    rejecting those regions remains the segmenter's responsibility.
    """
    annotation_path = Path(annotation_path).resolve()
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data["images"]}
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    grouped = defaultdict(list)
    for annotation in data["annotations"]:
        label = categories[int(annotation["category_id"])]
        if allowed_labels is None or label in allowed_labels:
            grouped[int(annotation["image_id"])].append(annotation)

    profiles = config["segmentation"]["roi_profiles"]
    if roi_profile_name not in profiles:
        raise ValueError(f"Unknown segmentation ROI profile: {roi_profile_name}")
    profile = profiles[roi_profile_name]
    source_margins = {side: 0.0 for side in ("left", "right", "top", "bottom")}
    segmentation_settings = {
        **config["segmentation"],
        **{
            name: value
            for name, value in profile.items()
            if name not in ("left", "right", "top", "bottom")
        },
    }
    scaling_settings = config.get("spatial_scaling", {})
    target_size = (
        int(scaling_settings["processing_width"]),
        int(scaling_settings["processing_height"]),
    )
    segmentation_settings = scale_pixel_parameters(
        segmentation_settings,
        int(scaling_settings.get("pixel_parameter_reference_size", 1920)),
        target_size,
    )
    feature_minimum_area = float(segmentation_settings["minimum_object_area"])
    if segmentation_settings.get("enable_thin_track_rule", True):
        feature_minimum_area = min(
            feature_minimum_area,
            float(segmentation_settings["minimum_thin_area"]),
        )

    rows: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for image_id, annotations in grouped.items():
        image_record = images[image_id]
        image_path = (annotation_path.parent / image_record["file_name"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read linked image: {image_path}")
        source_image_shape = image.shape[:2]
        normalisation = select_and_scale_roi(image, target_size=target_size)
        image = normalisation.image
        margins = source_margins
        enhanced = enhance_image(image, config["enhancement"])
        segmented = segment_tracks(
            enhanced.enhanced, segmentation_settings, margins
        )
        measured = extract_track_features(
            segmented.binary_mask,
            enhanced.enhanced,
            minimum_area=feature_minimum_area,
        )
        contour_by_box = {
            tuple(int(value) for value in cv2.boundingRect(contour)): contour
            for contour in segmented.contours
        }
        labelled_masks = [
            (
                annotation,
                _scale_label_mask(
                    annotation_to_mask(annotation, source_image_shape),
                    normalisation.coordinates,
                    target_size,
                ) > 0,
            )
            for annotation in annotations
        ]

        for track in measured:
            contour = contour_by_box.get(track.bounding_box)
            if contour is None:
                continue
            predicted_mask = np.zeros_like(segmented.binary_mask)
            cv2.drawContours(predicted_mask, [contour], -1, 255, cv2.FILLED)
            predicted_pixels = predicted_mask > 0
            predicted_count = int(np.count_nonzero(predicted_pixels))
            if predicted_count == 0:
                continue
            best_annotation = None
            best_overlap = 0.0
            for annotation, truth_mask in labelled_masks:
                overlap = float(
                    np.count_nonzero(predicted_pixels & truth_mask)
                    / predicted_count
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_annotation = annotation
            if best_annotation is None or best_overlap < minimum_label_overlap:
                continue
            label = categories[int(best_annotation["category_id"])]
            rows.append(
                {
                    "image_id": image_id,
                    "annotation_id": int(best_annotation["id"]),
                    "label_overlap": best_overlap,
                    "label": label,
                    **{
                        column: getattr(track, column)
                        for column in FEATURE_COLUMNS
                    },
                }
            )
            counts[label] += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "annotation_id",
        "label_overlap",
        "label",
        *FEATURE_COLUMNS,
    ]
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
