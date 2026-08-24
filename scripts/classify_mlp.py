"""Classify particle tracks in one cloud-chamber image with the trained MLP."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.config import load_config
from cloud_chamber.ml.member_models.mlp import (
    build_visual_report,
    load_model,
    predict_tracks,
)
from cloud_chamber.pipeline import analyse_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify tracks with the trained MLP")
    parser.add_argument("image", type=Path, help="Input cloud-chamber image")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models" / "mlp_classifier.joblib"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "mlp")
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")

    config = load_config(args.config)
    result = analyse_image(args.image, config)
    predictions = predict_tracks(load_model(args.model), result.features)

    # analyse_image returns features in the normalized processing coordinate space.
    analysis_image = result.enhancement.enhanced
    overlay, rows = build_visual_report(
        analysis_image,
        result.features,
        predictions,
        confidence_threshold=args.confidence_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.output_dir / f"{args.image.stem}_classified.png"
    csv_path = args.output_dir / f"{args.image.stem}_predictions.csv"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Unable to save overlay: {overlay_path}")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Detected tracks: {len(predictions)}")
    for item in predictions:
        print(
            f"Track {item['track_id']}: {item['particle_type']} "
            f"(confidence={item['confidence']:.3f})"
        )
    print(f"Annotated image: {overlay_path}")
    print(f"Prediction table: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
