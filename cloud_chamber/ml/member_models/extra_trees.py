"""Extra Trees inference and reporting for segmented particle tracks."""

from __future__ import annotations

from dataclasses import asdict
import csv
import io
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import joblib
import numpy as np

from cloud_chamber.features import TrackFeatures
from cloud_chamber.ml.member_models.mlp import (
    CLASS_COLOURS,
    DISPLAY_NAMES,
    FEATURE_COLUMNS,
)


def features_to_matrix(features: Iterable[TrackFeatures]) -> np.ndarray:
    """Convert accepted segmented tracks to the training feature order."""
    rows = []
    for track in features:
        values = asdict(track)
        rows.append([float(values[column]) for column in FEATURE_COLUMNS])
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_COLUMNS))


def load_model(model_path: str | Path) -> dict:
    """Load and validate the saved Extra Trees model bundle."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Extra Trees model not found: {path}. "
            "Run scripts/train_extra_trees.py first."
        )
    bundle = joblib.load(path)
    if tuple(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Saved Extra Trees model uses different feature columns")
    return bundle


def predict_tracks(
    model_bundle: dict, features: Iterable[TrackFeatures]
) -> list[dict]:
    """Predict a particle type and class probabilities for each track."""
    feature_list = list(features)
    matrix = features_to_matrix(feature_list)
    if matrix.shape[0] == 0:
        return []

    model = model_bundle["model"]
    started = perf_counter()
    probabilities = model.predict_proba(matrix)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    elapsed_per_track = (perf_counter() - started) * 1000.0 / len(matrix)

    return [
        {
            "track_id": track.track_id,
            "predicted_class": str(label),
            "particle_type": DISPLAY_NAMES.get(str(label), str(label)),
            "confidence": float(np.max(probability)),
            "inference_time_ms": elapsed_per_track,
            "probabilities": {
                str(class_name): float(class_probability)
                for class_name, class_probability in zip(
                    model.classes_, probability, strict=True
                )
            },
        }
        for track, label, probability in zip(
            feature_list, predictions, probabilities, strict=True
        )
    ]

def summarise_predictions(
    predictions: list[dict],
    confidence_threshold: float = 0.60,
) -> dict:
    """Summarise Extra Trees prediction results."""

    summary = {
        "Alpha": 0,
        "Electron/Positron": 0,
        "Proton": 0,
        "Uncertain": 0,
        "Total": len(predictions),
    }

    for prediction in predictions:
        particle_type = prediction["particle_type"]
        confidence = float(prediction["confidence"])

        if particle_type in summary:
            summary[particle_type] += 1

        if confidence < confidence_threshold:
            summary["Uncertain"] += 1

    return summary

def build_visual_report(
    image: np.ndarray,
    features: Iterable[TrackFeatures],
    predictions: list[dict],
    confidence_threshold: float = 0.60,
) -> tuple[np.ndarray, list[dict]]:
    """Draw the class predicted for every accepted segmented contour."""
    feature_list = list(features)
    if len(feature_list) != len(predictions):
        raise ValueError("Feature and prediction counts must be equal")

    overlay = image.copy()
    rows = []
    for track, prediction in zip(feature_list, predictions, strict=True):
        x, y, width, height = track.bounding_box
        confidence = float(prediction["confidence"])
        uncertain = confidence < confidence_threshold
        colour = (0, 255, 255) if uncertain else CLASS_COLOURS.get(
            prediction["predicted_class"], (255, 255, 255)
        )

        cv2.rectangle(overlay, (x, y), (x + width, y + height), colour, 2)
        label = (
            f"T{track.track_id}: {prediction['particle_type']} "
            f"{confidence:.0%}"
        )
        cv2.putText(
            overlay,
            label,
            (x, max(y - 7, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            1,
            cv2.LINE_AA,
        )
        rows.append(
            {
                "Track": track.track_id,
                "Particle type": prediction["particle_type"],
                "Confidence": confidence,
                "Status": "Uncertain" if uncertain else "Confident",
                "X": x,
                "Y": y,
                "Width": width,
                "Height": height,
                "Inference time (ms)": prediction["inference_time_ms"],
                **{
                    f"P({DISPLAY_NAMES.get(name, name)})": probability
                    for name, probability in prediction["probabilities"].items()
                },
            }
        )
    return overlay, rows


def encode_report_png(overlay: np.ndarray) -> bytes:
    """Encode an annotated classification report as PNG."""
    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise OSError("Unable to encode the Extra Trees visual report")
    return encoded.tobytes()


def encode_report_csv(rows: list[dict]) -> bytes:
    """Encode classification rows as spreadsheet-compatible CSV."""
    if not rows:
        return b""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")
