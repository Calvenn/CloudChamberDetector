"""Compare straight-only and curvature-aware segmentation on validation data."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.core.config import load_config
from cloud_chamber.ml.contour_dataset import build_segmented_feature_csv


def _summarise(path: Path, annotation_path: Path, allowed: set[str]) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    truth_ids = {
        int(item["id"])
        for item in data["annotations"]
        if categories[int(item["category_id"])] in allowed
    }
    counts = Counter(int(row["annotation_id"]) for row in rows)
    overlaps = [float(row["label_overlap"]) for row in rows]
    return {
        "ground_truth_particles": len(truth_ids),
        "matched_unique_particles": len(counts),
        "matched_particle_recall": len(counts) / max(len(truth_ids), 1),
        "segmented_records": len(rows),
        "fragmented_particles": sum(value > 1 for value in counts.values()),
        "fragmentation_rate_among_matched": (
            sum(value > 1 for value in counts.values()) / max(len(counts), 1)
        ),
        "mean_records_per_matched_particle": len(rows) / max(len(counts), 1),
        "mean_prediction_inside_annotation": sum(overlaps) / max(len(overlaps), 1),
    }


def main() -> int:
    """Measure the effect of curved-fragment linking on held-out tracks."""

    config = load_config(PROJECT_ROOT / "config.yaml")
    allowed = set(config["classification"]["supported_classes"])
    annotations = (
        PROJECT_ROOT
        / "dataset"
        / "external_dataset_split"
        / "validation"
        / "annotations_coco.json"
    )
    output_dir = PROJECT_ROOT / "results" / "segmentation_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, enabled in (("straight_only", False), ("curvature_aware", True)):
        candidate = deepcopy(config)
        candidate["segmentation"]["curved_alignment_merge"] = enabled
        feature_path = output_dir / f"{name}_validation.csv"
        build_segmented_feature_csv(
            annotations,
            feature_path,
            candidate,
            allowed_labels=allowed,
            roi_profile_name="external_muller",
            merge_annotation_fragments=False,
        )
        results[name] = _summarise(feature_path, annotations, allowed)

    report_path = output_dir / "curved_linking_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, metrics in results.items():
        print(name, json.dumps(metrics, sort_keys=True))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
