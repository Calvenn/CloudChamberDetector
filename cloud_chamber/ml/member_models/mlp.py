"""Multilayer Perceptron classifier for contour-based track features.

The saved object is a scikit-learn Pipeline. It contains both StandardScaler
and MLPClassifier, preventing training/inference preprocessing differences.
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


# This order is the public interface between shared feature extraction and all
# MLP operations. Changing it would invalidate an already trained model.
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

# BGR colours are intentionally fixed so a class has the same visual identity
# in every exported report. Yellow is reserved for uncertain predictions.
CLASS_COLOURS = {
    "alpha": (0, 165, 255),
    "electron_positron": (255, 120, 0),
    "proton": (0, 200, 0),
    "v_track": (255, 0, 255),
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
    """Load the scaler-plus-MLP pipeline saved by the training script."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"MLP model not found: {path}. Run scripts/train_mlp.py first."
        )
    bundle = joblib.load(path)
    if tuple(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Saved MLP uses a different feature-column contract")
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
    quality_assessments: list[dict] | None = None,
    original_boxes: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Draw classified tracks and build the matching tabular report.

    A probability is model confidence, not proof of particle identity. Values
    below the chosen reporting threshold retain the most likely class but are
    visibly marked ``Uncertain`` to avoid overstating the result.
    """
    feature_list = list(features)
    if len(feature_list) != len(predictions):
        raise ValueError("Feature and prediction counts must be equal")
    quality_by_track = {
        int(item["track_id"]): item for item in (quality_assessments or [])
    }

    overlay = image.copy()
    rows = []
    for index, (track, prediction) in enumerate(
        zip(feature_list, predictions, strict=True)
    ):
        if original_boxes and index < len(original_boxes):
            box = original_boxes[index]
            x, y = int(round(box["x"])), int(round(box["y"]))
            width, height = int(round(box["width"])), int(round(box["height"]))
        else:
            x, y, width, height = track.bounding_box
        confidence = float(prediction["confidence"])
        uncertain = confidence < confidence_threshold
        quality = quality_by_track.get(track.track_id)
        low_quality = quality is not None and int(quality["score"]) < 50
        if low_quality:
            colour = (160, 160, 160)
        elif uncertain:
            colour = (0, 255, 255)
        else:
            colour = CLASS_COLOURS.get(
                prediction["predicted_class"], (255, 255, 255)
            )

        # The same track ID connects the picture, feature table and CSV row.
        if low_quality:
            _draw_dashed_rectangle(
                overlay, (x, y), (x + width, y + height), colour
            )
        else:
            cv2.rectangle(overlay, (x, y), (x + width, y + height), colour, 2)
        status = (
            "Review segmentation"
            if low_quality
            else "Uncertain"
            if uncertain
            else "Accepted"
        )
        quality_text = f" | Q:{quality['grade']}" if quality else ""
        label = (
            f"T{track.track_id}: {prediction['particle_type']} "
            f"{confidence:.0%}{quality_text}"
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
                "Contour quality": quality["grade"] if quality else "Not assessed",
                "Contour quality score": quality["score"] if quality else "",
                "Local contrast": quality["local_contrast"] if quality else "",
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


def _draw_dashed_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    colour: tuple[int, int, int],
    dash_length: int = 8,
) -> None:
    """Draw a dashed review box without hiding the underlying particle."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    for start in range(x1, x2, dash_length * 2):
        cv2.line(image, (start, y1), (min(start + dash_length, x2), y1), colour, 2)
        cv2.line(image, (start, y2), (min(start + dash_length, x2), y2), colour, 2)
    for start in range(y1, y2, dash_length * 2):
        cv2.line(image, (x1, start), (x1, min(start + dash_length, y2)), colour, 2)
        cv2.line(image, (x2, start), (x2, min(start + dash_length, y2)), colour, 2)


def encode_report_png(overlay: np.ndarray) -> bytes:
    """Encode an annotated BGR image for the GUI download button."""
    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise OSError("Unable to encode the MLP visual report")
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
