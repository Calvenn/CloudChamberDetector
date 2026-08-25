"""Train the post-segmentation particle-versus-artifact filter."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import _process_tiled_pipeline_image
from cloud_chamber.config import load_config
from cloud_chamber.ml.artifact_filter import FEATURE_COLUMNS, candidate_matrix
from cloud_chamber.ml.contour_dataset import annotation_to_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=("external", "primary"),
        default="external",
        help="Image domain whose candidate filter should be trained.",
    )
    parser.add_argument("--rebuild-candidates", action="store_true")
    return parser.parse_args()


def _annotation_path(split: str, domain: str) -> Path:
    if domain == "primary":
        return (
            PROJECT_ROOT
            / "dataset"
            / "primary_dataset_split"
            / f"{split}_annotations_coco.json"
        )
    return (
        PROJECT_ROOT
        / "dataset"
        / "external_dataset_split"
        / split
        / "annotations_coco.json"
    )


def _dataset(
    split: str,
    config: dict,
    domain: str,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = (
        PROJECT_ROOT
        / "results"
        / "artifact_filter"
        / f"{domain}_{split}_candidates.npz"
    )
    if cache_path.is_file() and not rebuild:
        cached = np.load(cache_path)
        print(f"{split}: loaded {len(cached['labels'])} cached candidates", flush=True)
        return cached["features"], cached["labels"]
    annotation_path = _annotation_path(split, domain)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    allowed = set(config["classification"]["supported_classes"])
    grouped: dict[int, list[dict]] = {}
    for annotation in data["annotations"]:
        if categories[int(annotation["category_id"])] in allowed:
            grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    rows = []
    labels = []
    images = {int(item["id"]): item for item in data["images"]}
    training_config = json.loads(json.dumps(config))
    training_config["segmentation"]["artifact_filter_enabled"] = False
    for image_index, (image_id, annotations) in enumerate(grouped.items(), start=1):
        record = images[image_id]
        image_path = (annotation_path.parent / record["file_name"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        result = _process_tiled_pipeline_image(image, training_config)
        contours = result["segmentation"].contours
        if not contours:
            continue
        matrix = candidate_matrix(contours, result["enhancement"].enhanced)
        truth_union = np.zeros(image.shape[:2], dtype=np.uint8)
        for annotation in annotations:
            truth_union = cv2.bitwise_or(
                truth_union,
                annotation_to_mask(annotation, image.shape[:2]),
            )
        for contour, vector in zip(contours, matrix, strict=True):
            x, y, width, height = cv2.boundingRect(contour)
            local_contour = contour.copy()
            local_contour[:, 0, 0] -= x
            local_contour[:, 0, 1] -= y
            predicted = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(predicted, [local_contour], -1, 255, cv2.FILLED)
            predicted_pixels = predicted > 0
            predicted_count = max(int(np.count_nonzero(predicted_pixels)), 1)
            truth_crop = truth_union[y:y + height, x:x + width] > 0
            overlap = np.count_nonzero(predicted_pixels & truth_crop) / predicted_count
            rows.append(vector)
            labels.append("particle" if overlap >= 0.5 else "artifact")
        if image_index % 10 == 0:
            print(f"{split}: processed {image_index}/{len(grouped)} images", flush=True)
    features = np.asarray(rows, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, labels=labels_array)
    return features, labels_array


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = np.where(probabilities >= threshold, "particle", "artifact")
    return {
        "sample_count": int(len(labels)),
        "threshold": float(threshold),
        "particle_precision": float(precision_score(labels, predictions, pos_label="particle", zero_division=0)),
        "particle_recall": float(recall_score(labels, predictions, pos_label="particle", zero_division=0)),
        "particle_f2": float(fbeta_score(labels, predictions, pos_label="particle", beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=["artifact", "particle"]).tolist(),
        "classification_report": classification_report(labels, predictions, labels=["artifact", "particle"], output_dict=True, zero_division=0),
    }


def main() -> int:
    args = parse_args()
    config = load_config(PROJECT_ROOT / "config.yaml")
    seed = int(config["project"]["random_seed"])
    tables = {
        split: _dataset(
            split,
            config,
            args.domain,
            args.rebuild_candidates,
        )
        for split in ("development", "validation", "final_test")
    }
    candidates = []
    best = None
    for index, values in enumerate(product((200, 500), (None, 20), (1, 3)), start=1):
        parameters = dict(zip(("n_estimators", "max_depth", "min_samples_leaf"), values, strict=True))
        model = ExtraTreesClassifier(**parameters, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=-1)
        model.fit(*tables["development"])
        particle_index = list(model.classes_).index("particle")
        probabilities = model.predict_proba(tables["validation"][0])[:, particle_index]
        for threshold in np.arange(0.25, 0.76, 0.05):
            metrics = _metrics(tables["validation"][1], probabilities, float(threshold))
            score = (metrics["particle_f2"], metrics["particle_precision"])
            if best is None or score > best[0]:
                best = (score, parameters, float(threshold), index, metrics)
        candidates.append({"candidate": index, "parameters": parameters})
    assert best is not None
    _, parameters, threshold, best_index, validation = best
    train_x = np.concatenate((tables["development"][0], tables["validation"][0]))
    train_y = np.concatenate((tables["development"][1], tables["validation"][1]))
    model = ExtraTreesClassifier(**parameters, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=-1)
    model.fit(train_x, train_y)
    particle_index = list(model.classes_).index("particle")
    final_probabilities = model.predict_proba(tables["final_test"][0])[:, particle_index]
    final_test = _metrics(tables["final_test"][1], final_probabilities, threshold)
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "acceptance_threshold": threshold,
        "selected_parameters": parameters,
        "random_seed": seed,
        "validation_metrics": validation,
        "final_test_metrics": final_test,
    }
    model_name = (
        "artifact_filter_primary.joblib"
        if args.domain == "primary"
        else "artifact_filter.joblib"
    )
    output = PROJECT_ROOT / "models" / model_name
    joblib.dump(bundle, output, compress=3)
    report = {
        "method": "Extremely Randomized Trees particle-versus-artifact candidate filter",
        "domain": args.domain,
        "selected_candidate": best_index,
        "selected_parameters": parameters,
        "acceptance_threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "class_counts": {split: dict(zip(*np.unique(labels, return_counts=True), strict=True)) for split, (_, labels) in tables.items()},
        "validation": validation,
        "final_test": final_test,
        "model_path": str(output),
    }
    report_name = (
        "artifact_filter_primary_training_report.json"
        if args.domain == "primary"
        else "artifact_filter_training_report.json"
    )
    report_path = PROJECT_ROOT / "models" / report_name
    report_path.write_text(json.dumps(report, indent=2, default=lambda value: int(value)), encoding="utf-8")
    print(json.dumps({"validation": validation, "final_test": final_test}, indent=2))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
