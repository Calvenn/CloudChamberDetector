"""Compare candidate longest-side resolutions on validation annotations."""

from __future__ import annotations

import copy
import csv
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_chamber.config import load_config
from app import _process_pipeline_image


CANDIDATES = tuple(int(value) for value in sys.argv[1:]) or (1312, 1600, 1920)
LENGTH_KEYS = (
    "top_hat_kernel", "closing_kernel", "directional_closing_length",
    "alignment_merge_gap", "minimum_major_axis", "minimum_thin_perimeter",
    "minimum_thin_major_axis",
)
AREA_KEYS = ("minimum_object_area", "minimum_thin_area")
ODD_KEYS = ("top_hat_kernel", "closing_kernel", "directional_closing_length")


def _odd(value: float) -> int:
    rounded = max(1, int(round(value)))
    return rounded if rounded % 2 else rounded + 1


def _config_for(candidate: int) -> dict:
    config = copy.deepcopy(load_config(ROOT / "config.yaml"))
    ratio = candidate / 1920.0
    config["spatial_normalisation"]["target_longest_side"] = candidate
    for profile in config["segmentation"]["roi_profiles"].values():
        for key in LENGTH_KEYS:
            if key in profile:
                value = float(profile[key]) * ratio
                profile[key] = _odd(value) if key in ODD_KEYS else max(1, int(round(value)))
        for key in AREA_KEYS:
            if key in profile:
                profile[key] = max(1, int(round(float(profile[key]) * ratio * ratio)))
    return config


def _iou(first, second) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _greedy_matches(predicted, truths, threshold=0.5):
    pairs = sorted(
        ((_iou(p, t["bbox"]), pi, ti) for pi, p in enumerate(predicted)
         for ti, t in enumerate(truths)), reverse=True
    )
    used_p, used_t, matched = set(), set(), []
    for score, pi, ti in pairs:
        if score < threshold or pi in used_p or ti in used_t:
            continue
        used_p.add(pi); used_t.add(ti); matched.append(ti)
    return matched


def _records(annotation_path: Path, allowed_ids=None):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(c["id"]): c["name"] for c in data["categories"]}
    grouped = {}
    for ann in data["annotations"]:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    for image in data["images"]:
        if allowed_ids is not None and Path(image["file_name"]).stem not in allowed_ids:
            continue
        yield annotation_path.parent / image["file_name"], image, grouped.get(int(image["id"]), []), categories


def main() -> None:
    chunk_start = int(os.environ.get("SPATIAL_CHUNK_START", "0"))
    chunk_end = int(os.environ.get("SPATIAL_CHUNK_END", "1000000"))
    manifest = ROOT / "dataset/primary_dataset_split/manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        primary_ids = {r["image_id"] for r in csv.DictReader(handle) if r["split"] == "validation"}
    sources = [
        (ROOT / "dataset/external_dataset_split/validation/annotations_coco.json", None),
        (ROOT / "dataset/primary_dataset_split/annotations_coco.json", primary_ids),
    ]
    results = []
    for candidate in CANDIDATES:
        config = _config_for(candidate)
        tp = fp = fn = thin_total = thin_matched = images = 0
        elapsed = 0.0
        record_index = 0
        for annotation_path, allowed_ids in sources:
            for image_path, record, annotations, categories in _records(annotation_path, allowed_ids):
                current_index = record_index
                record_index += 1
                if current_index < chunk_start or current_index >= chunk_end:
                    continue
                image = cv2.imread(str(image_path.resolve()))
                if image is None:
                    raise FileNotFoundError(image_path)
                scale = candidate / max(image.shape[1], image.shape[0])
                truths = []
                for ann in annotations:
                    x, y, w, h = (float(v) for v in ann["bbox"])
                    truths.append({"bbox": (x*scale, y*scale, w*scale, h*scale), "label": categories[int(ann["category_id"])]})
                started = perf_counter()
                output = _process_pipeline_image(image, config, None)
                elapsed += perf_counter() - started
                predicted = output["segmentation"].bounding_boxes
                matched = _greedy_matches(predicted, truths)
                matched_set = set(matched)
                tp += len(matched); fp += len(predicted) - len(matched); fn += len(truths) - len(matched); images += 1
                for index, truth in enumerate(truths):
                    if truth["label"] == "electron_positron":
                        thin_total += 1
                        thin_matched += int(index in matched_set)
                del output, image
                gc.collect()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append({"longest_side": candidate, "images": images, "true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1, "thin_track_recall": thin_matched / thin_total if thin_total else 0.0, "mean_processing_ms": elapsed * 1000 / images})
        print(results[-1])
    suffix = "_" + "_".join(str(value) for value in CANDIDATES)
    if chunk_start or chunk_end < 1000000:
        suffix += f"_{chunk_start}_{chunk_end}"
    output = ROOT / f"results/spatial_resolution_validation{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
