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
    DISPLAY_NAMES,
    FEATURE_COLUMNS,
)


# OpenCV BGR colour registry. Only classes present in the loaded model are used.
EXTRA_TREES_CLASS_COLOURS = {
    "proton": (0, 0, 255),
    "electron_positron": (255, 0, 0),
    "alpha": (0, 255, 0),
    "v_track": (180, 0, 180),
}


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


def explain_track_prediction(
    model: object,
    feature_columns: tuple[str, ...] | list[str],
    feature_vector: np.ndarray,
    top_n: int = 3,
) -> list[dict]:
    """Calculate key feature drivers for a single track prediction using decision path impurity drops."""
    n_features = len(feature_columns)
    contributions = np.zeros(n_features, dtype=np.float64)

    base_model = getattr(model, "estimator", getattr(model, "base_estimator", model))
    if hasattr(model, "calibrated_classifiers_"):
        estimators = []
        for cc in model.calibrated_classifiers_:
            estimators.extend(getattr(cc.estimator, "estimators_", []))
    else:
        estimators = list(getattr(base_model, "estimators_", []))

    if estimators:
        sample = feature_vector.reshape(1, -1)
        for tree in estimators:
            node_indicator = tree.decision_path(sample)
            leaf_id = tree.apply(sample)[0]
            feature = tree.tree_.feature
            impurity = tree.tree_.impurity
            n_node_samples = tree.tree_.n_node_samples
            node_indices = node_indicator.indices

            for node_id in node_indices:
                if leaf_id == node_id:
                    continue
                feat = feature[node_id]
                if feat >= 0:
                    left_child = tree.tree_.children_left[node_id]
                    right_child = tree.tree_.children_right[node_id]
                    drop = impurity[node_id] - (
                        (n_node_samples[left_child] / n_node_samples[node_id]) * impurity[left_child] +
                        (n_node_samples[right_child] / n_node_samples[node_id]) * impurity[right_child]
                    )
                    contributions[feat] += max(drop, 0.0)

    total = np.sum(contributions)
    if total > 0:
        contributions /= total
    else:
        feat_imp = getattr(base_model, "feature_importances_", None)
        if feat_imp is not None and len(feat_imp) == n_features:
            contributions = np.asarray(feat_imp, dtype=np.float64)
            if np.sum(contributions) > 0:
                contributions /= np.sum(contributions)

    driver_indices = np.argsort(contributions)[::-1][:top_n]
    return [
        {
            "feature": feature_columns[idx],
            "contribution": float(contributions[idx]),
            "value": float(feature_vector[idx]),
        }
        for idx in driver_indices
    ]


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
            "top_drivers": explain_track_prediction(
                model, FEATURE_COLUMNS, vector, top_n=3
            ),
            "probabilities": {
                str(class_name): float(class_probability)
                for class_name, class_probability in zip(
                    model.classes_, probability, strict=True
                )
            },
        }
        for track, label, probability, vector in zip(
            feature_list, predictions, probabilities, matrix, strict=True
        )
    ]

def summarise_predictions(
    predictions: list[dict],
    confidence_threshold: float = 0.60,
) -> dict:
    """Summarise Extra Trees prediction results."""

    summary = {
        DISPLAY_NAMES.get(class_name, class_name): 0
        for class_name in dict.fromkeys(
            prediction["predicted_class"] for prediction in predictions
        )
    }
    summary.update({"Uncertain": 0, "Total": len(predictions)})

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
    quality_assessments: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Draw the class predicted for every accepted segmented contour."""
    feature_list = list(features)
    if len(feature_list) != len(predictions):
        raise ValueError("Feature and prediction counts must be equal")
    quality_by_track = {
        int(item["track_id"]): item for item in (quality_assessments or [])
    }

    overlay = image.copy()
    rows = []
    for track, prediction in zip(feature_list, predictions, strict=True):
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
            colour = EXTRA_TREES_CLASS_COLOURS.get(
                prediction["predicted_class"], (255, 255, 255)
            )

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
        top_driver_str = (
            ", ".join(
                f"{d['feature']} ({d['contribution']:.0%})"
                for d in prediction.get("top_drivers", [])
            )
            if prediction.get("top_drivers")
            else "N/A"
        )
        rows.append(
            {
                "Track": track.track_id,
                "Particle type": prediction["particle_type"],
                "Confidence": confidence,
                "Status": status,
                "Top decision driver": top_driver_str,
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
    """Encode an annotated classification report as PNG."""
    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise OSError("Unable to encode the Extra Trees visual report")
    return encoded.tobytes()


def encode_report_csv(rows: list[dict]) -> bytes:
    """Encode full technical results with clear, analysis-friendly headings."""
    if not rows:
        return b""
    preferred_columns = (
        "Track",
        "Particle type",
        "Confidence",
        "Reporting decision",
        "Status",
        "Contour quality",
        "Contour quality score",
        "Local contrast",
        "X",
        "Y",
        "Width",
        "Height",
        "Inference time (ms)",
        "Top decision driver",
    )
    column_names = {
        "Track": "Track ID",
        "Particle type": "Predicted Particle Type",
        "Confidence": "Confidence (0-1)",
        "Reporting decision": "Reporting Decision",
        "Status": "Box Display Status",
        "Contour quality": "Contour Quality Grade",
        "Contour quality score": "Contour Quality Score (0-100)",
        "Local contrast": "Local Contrast",
        "X": "Bounding Box X (px)",
        "Y": "Bounding Box Y (px)",
        "Width": "Bounding Box Width (px)",
        "Height": "Bounding Box Height (px)",
        "Inference time (ms)": "Inference Time (ms)",
        "Top decision driver": "Top Decision Drivers",
    }
    source_columns = list(rows[0])
    ordered_columns = [name for name in preferred_columns if name in source_columns]
    ordered_columns.extend(name for name in source_columns if name not in ordered_columns)
    fieldnames = [column_names.get(name, name) for name in ordered_columns]
    export_rows = [
        {
            column_names.get(name, name): row.get(name, "")
            for name in ordered_columns
        }
        for row in rows
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(export_rows)
    return output.getvalue().encode("utf-8-sig")
