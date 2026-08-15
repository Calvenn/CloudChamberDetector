"""SVM classifier for contour-based track features.

The saved object is a scikit-learn Pipeline. It contains both StandardScaler
and SVC, preventing training/inference preprocessing differences.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import io
from pathlib import Path
from time import perf_counter
from typing import Iterable

import joblib
import cv2
import numpy as np

from cloud_chamber.features import TrackFeatures


FEATURE_COLUMNS = (
    "area_pixels",
    "perimeter_pixels",
    "major_axis_pixels",
    "mean_width_pixels",
    "orientation_degrees",
    "aspect_ratio",
    "solidity",
    "rectangularity",
    "thickness_pixels",
    "mean_intensity",
)

CLASS_COLOURS = {
    "alpha": (0, 165, 255),
    "electron_positron": (255, 120, 0),
    "proton": (0, 200, 0),
    "v_track": (180, 0, 180),
}
DISPLAY_NAMES = {
    "alpha": "Alpha",
    "electron_positron": "Electron/Positron",
    "proton": "Proton",
    "v_track": "V-track",
}


def features_to_matrix(features: Iterable[TrackFeatures]) -> np.ndarray:
    """Convert shared TrackFeatures objects into the fixed numerical matrix."""
    rows = []
    for track in features:
        values = asdict(track)
        rows.append([float(values[column]) for column in FEATURE_COLUMNS])
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_COLUMNS))


def load_model(model_path: str | Path):
    """Load the scaler-plus-SVM pipeline saved by the training script."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SVM model not found: {path}. Run scripts/train_svm.py first."
        )
    bundle = joblib.load(path)
    if tuple(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Saved SVM uses a different feature-column contract")
    return bundle


def predict_tracks(model_bundle: dict, features: Iterable[TrackFeatures]) -> list[dict]:
    """Classify segmented tracks and report probability and inference time."""
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


def build_visual_report(
    image: np.ndarray,
    features: Iterable[TrackFeatures],
    predictions: list[dict],
    confidence_threshold: float = 0.60,
) -> tuple[np.ndarray, list[dict]]:
    """Draw classified tracks and build the matching tabular report.

    A probability is model confidence, not proof of particle identity. Values
    below the chosen reporting threshold retain the most likely class but are
    visibly marked ``Uncertain`` to avoid overstating the result.
    """
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

        # The same track ID connects the picture, feature table and CSV row.
        cv2.rectangle(overlay, (x, y), (x + width, y + height), colour, 2)
        status = "Uncertain" if uncertain else "Accepted"
        label = (
            f"T{track.track_id}: {prediction['particle_type']} "
            f"{confidence:.0%}"
        )
        text_y = max(y - 7, 16)
        cv2.putText(
            overlay,
            label,
            (x, text_y),
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
                "Status": status,
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
    """Encode an annotated BGR image for the GUI download button."""
    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise OSError("Unable to encode the SVM visual report")
    return encoded.tobytes()


def encode_report_csv(rows: list[dict]) -> bytes:
    """Encode table rows as a spreadsheet-compatible UTF-8 CSV file."""
    if not rows:
        return b""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")
