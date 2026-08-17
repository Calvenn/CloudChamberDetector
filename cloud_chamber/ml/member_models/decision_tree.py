"""Decision Tree classifier for contour-based track features.

Per project convention, training, validation, prediction and model-saving
functions for this classifier live only in this file. scripts/train_decision_tree.py
is a thin CLI wrapper: it parses arguments, builds or loads the shared
feature CSVs via cloud_chamber.ml.contour_dataset, calls the functions
defined here, and writes the resulting model and JSON report to disk. It
contains no Decision-Tree-specific logic of its own.

The saved bundle wraps a fitted scikit-learn DecisionTreeClassifier. Tree
splits compare raw feature values against learned thresholds rather than
distances or gradients, so unlike cloud_chamber.ml.member_models.mlp this
bundle needs no StandardScaler.
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


# Must stay byte-identical to cloud_chamber.ml.member_models.mlp.FEATURE_COLUMNS.
# cloud_chamber.ml.contour_dataset writes every cached feature CSV using that
# module's column order, so this tuple has to match it exactly or the columns
# loaded here will silently line up with the wrong feature names.
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

# Matches cloud_chamber.ml.member_models.mlp so a class keeps the same visual
# identity in every model's exported report. Yellow (applied in
# build_visual_report, not here) is reserved for uncertain predictions.
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

# max_depth caps tree growth against only ten input features; min_samples_leaf
# stops the tree from carving out a leaf for a single noisy track. criterion
# is fixed to Gini (scikit-learn's default) so only depth and leaf size are
# searched, keeping the candidate set as small and interpretable as the MLP's.
# The validation split (see train_candidates) selects between these; none is
# claimed to be universally optimal.
PARAMETER_CANDIDATES = (
    {"max_depth": 4, "min_samples_leaf": 5},
    {"max_depth": 4, "min_samples_leaf": 10},
    {"max_depth": 6, "min_samples_leaf": 5},
    {"max_depth": 6, "min_samples_leaf": 10},
    {"max_depth": 8, "min_samples_leaf": 5},
    {"max_depth": 8, "min_samples_leaf": 10},
)


def features_to_matrix(features: Iterable[TrackFeatures]) -> np.ndarray:
    """Convert shared TrackFeatures objects into the fixed numerical matrix."""
    rows = []
    for track in features:
        values = asdict(track)
        rows.append([float(values[column]) for column in FEATURE_COLUMNS])
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_COLUMNS))


def combine_feature_tables(
    tables: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a collection of feature matrices and labels while preserving order."""
    matrices = []
    labels = []
    for matrix, label_array in tables:
        if matrix.size == 0:
            continue
        matrices.append(matrix)
        labels.append(label_array)
    if not matrices:
        return (
            np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64),
            np.asarray([], dtype=object),
        )
    return np.vstack(matrices), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------


def make_model(parameters: dict, random_seed: int):
    """Create one Decision Tree estimator. No scaler is needed for splits."""
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        # Reweights classes inversely to frequency at fit time. Trees support
        # this natively, so no manual resampling fallback is needed here
        # (unlike the MLP's sample_weight / resampling logic).
        class_weight="balanced",
        random_state=random_seed,
    )


def evaluate(model, matrix: np.ndarray, labels: np.ndarray) -> dict:
    """Score a fitted model against a labelled split (validation or final-test)."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    started = perf_counter()
    predictions = model.predict(matrix)
    elapsed_ms = (perf_counter() - started) * 1000.0
    class_names = sorted(set(labels.tolist()) | set(predictions.tolist()))
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "mean_inference_ms_per_track": elapsed_ms / max(len(labels), 1),
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=class_names
        ).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=class_names, output_dict=True, zero_division=0
        ),
    }


def train_candidates(
    development_x: np.ndarray,
    development_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    random_seed: int,
) -> tuple[list[dict], int, object]:
    """Fit every declared candidate on development data and score it on validation.

    Returns every candidate's parameter/metric record, the 1-based index of
    the best candidate by validation macro F1 (balanced accuracy as
    tie-breaker), and that candidate's fitted model. The final-test split is
    deliberately not touched here; it is only evaluated once, after this
    selection is final.
    """
    candidates: list[dict] = []
    best = None
    for index, parameters in enumerate(PARAMETER_CANDIDATES, start=1):
        model = make_model(parameters, random_seed)
        model.fit(development_x, development_y)
        metrics = evaluate(model, validation_x, validation_y)
        record = {
            "candidate": index,
            "parameters": {
                **parameters,
                "criterion": "gini",
                "class_weight": "balanced",
            },
            "validation": metrics,
        }
        candidates.append(record)
        score = (metrics["macro_f1"], metrics["balanced_accuracy"])
        if best is None or score > best[0]:
            best = (score, model, index)

    assert best is not None
    _, best_model, best_index = best
    return candidates, best_index, best_model


# ---------------------------------------------------------------------------
# Model saving and loading
# ---------------------------------------------------------------------------


def save_model(
    model,
    output_path: str | Path,
    *,
    classes: tuple[str, ...],
    selected_parameters: dict,
    random_seed: int,
    validation_metrics: dict,
    final_test_metrics: dict,
) -> dict:
    """Persist the fitted Decision Tree as the bundle load_model expects."""
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": classes,
        "selected_parameters": selected_parameters,
        "random_seed": random_seed,
        "balancing": "class_weight='balanced'",
        "validation_metrics": validation_metrics,
        "final_test_metrics": final_test_metrics,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def build_training_report(
    candidates: list[dict],
    best_index: int,
    class_counts: dict,
    final_metrics: dict,
    model_path: str | Path,
) -> dict:
    """Assemble the JSON evidence report saved alongside the model."""
    return {
        "method": "DecisionTreeClassifier",
        "selection_metric": "validation macro F1; balanced accuracy tie-breaker",
        "feature_columns": FEATURE_COLUMNS,
        "class_counts": class_counts,
        "candidate_results": candidates,
        "selected_candidate": best_index,
        "selected_parameters": candidates[best_index - 1]["parameters"],
        "final_test": final_metrics,
        "model_path": str(model_path),
    }


def load_model(model_path: str | Path) -> dict:
    """Load the Decision Tree bundle saved by save_model / scripts/train_decision_tree.py."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Decision Tree model not found: {path}. "
            "Run scripts/train_decision_tree.py first."
        )
    bundle = joblib.load(path)
    if tuple(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(
            "Saved Decision Tree uses a different feature-column contract"
        )
    return bundle


# ---------------------------------------------------------------------------
# Prediction and reporting
# ---------------------------------------------------------------------------


def predict_tracks(model_bundle: dict, features: Iterable[TrackFeatures]) -> list[dict]:
    """Classify segmented tracks and report probability and inference time.

    Output schema matches cloud_chamber.ml.member_models.mlp.predict_tracks so
    the Streamlit report page and CSV/PNG export code behave identically
    across every member model.
    """
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
        raise OSError("Unable to encode the Decision Tree visual report")
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
