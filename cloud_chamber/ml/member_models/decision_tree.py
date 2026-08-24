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
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS


# Axis-aligned tree splits cannot readily express relationships such as
# perimeter squared divided by area.  These deterministic shape features are
# calculated from the existing feature contract for both training rows and live
# TrackFeatures, so they improve the tree without modifying shared extraction
# or dataset code.
DERIVED_FEATURE_COLUMNS = (
    "circularity_from_area_perimeter",
    "perimeter_to_major_axis",
    "orientation_sin_2x",
    "orientation_cos_2x",
    "major_axis_to_mean_width",
    "aspect_ratio_times_solidity",
    "intensity_area_product",
)
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

# OpenCV colours are BGR. These colours identify the predicted particle class
# independently of the model confidence.
CLASS_COLOURS = {
    "alpha": (0, 255, 0),
    "electron_positron": (255, 0, 0),
    "proton": (0, 0, 255),
    "v_track": (180, 0, 180),
}
DISPLAY_NAMES = {
    "alpha": "Alpha",
    "electron_positron": "Electron/Positron",
    "proton": "Proton",
    "v_track": "V-track",
}

# A wider, still interpretable search lets validation choose between shallow
# trees and carefully regularised deeper trees.  min_samples_leaf and pruning
# prevent individual segmentation artefacts from becoming decision rules.
PARAMETER_CANDIDATES = tuple(
    {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "ccp_alpha": ccp_alpha,
    }
    for criterion in ("gini", "entropy", "log_loss")
    for max_depth in (4, 6, 8, 10, None)
    for min_samples_leaf in (2, 5, 10, 20)
    for ccp_alpha in (0.0, 0.0005)
)


def features_to_matrix(features: Iterable[TrackFeatures]) -> np.ndarray:
    """Convert shared TrackFeatures objects into the shared raw feature matrix."""
    rows = []
    for track in features:
        values = asdict(track)
        rows.append([float(values[column]) for column in FEATURE_COLUMNS])
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_COLUMNS))


def augment_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    """Add stable geometry interactions that a single tree cannot infer by itself.

    This function deliberately accepts the unmodified shared CSV matrix.  It is
    used for both fitting and prediction, which keeps the saved model's feature
    contract explicit and prevents training/serving skew.
    """
    raw = np.asarray(matrix, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            "Decision Tree expects a two-dimensional raw feature matrix with "
            f"{len(FEATURE_COLUMNS)} columns"
        )
    if raw.shape[0] == 0:
        return np.empty((0, len(MODEL_FEATURE_COLUMNS)), dtype=np.float64)

    area = raw[:, 0]
    perimeter = raw[:, 1]
    major_axis = raw[:, 2]
    mean_width = raw[:, 3]
    orientation_radians = np.deg2rad(2.0 * raw[:, 4])
    aspect_ratio = raw[:, 5]
    solidity = raw[:, 6]
    mean_intensity = raw[:, 9]

    def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=np.abs(denominator) > np.finfo(np.float64).eps,
        )

    circularity = safe_divide(4.0 * np.pi * area, perimeter * perimeter)
    perimeter_to_major_axis = safe_divide(perimeter, major_axis)
    major_axis_to_mean_width = safe_divide(major_axis, mean_width)
    derived = np.column_stack(
        (
            circularity,
            perimeter_to_major_axis,
            np.sin(orientation_radians),
            np.cos(orientation_radians),
            major_axis_to_mean_width,
            aspect_ratio * solidity,
            area * mean_intensity,
        )
    )
    return np.column_stack((raw, derived))


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
        criterion=parameters["criterion"],
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        ccp_alpha=parameters["ccp_alpha"],
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
    development_sample_weight: np.ndarray | None = None,
) -> tuple[list[dict], int, object]:
    """Fit every declared candidate on development data and score it on validation.

    Returns every candidate's parameter/metric record, the 1-based index of
    the best candidate by validation macro F1 (balanced accuracy as
    tie-breaker), and that candidate's fitted model. The final-test split is
    deliberately not touched here; it is only evaluated once, after this
    selection is final.
    """
    if development_sample_weight is not None:
        development_sample_weight = np.asarray(
            development_sample_weight, dtype=np.float64
        )
        if development_sample_weight.shape != (len(development_y),):
            raise ValueError(
                "Decision Tree development sample weights must contain one "
                "positive value per training row"
            )
        if (
            not np.isfinite(development_sample_weight).all()
            or np.any(development_sample_weight <= 0.0)
        ):
            raise ValueError("Decision Tree development sample weights must be positive")

    candidates: list[dict] = []
    best = None
    for index, parameters in enumerate(PARAMETER_CANDIDATES, start=1):
        model = make_model(parameters, random_seed)
        model.fit(
            development_x,
            development_y,
            sample_weight=development_sample_weight,
        )
        metrics = evaluate(model, validation_x, validation_y)
        record = {
            "candidate": index,
            "parameters": {
                **parameters,
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
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "raw_feature_columns": FEATURE_COLUMNS,
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
    *,
    ground_truth_final_metrics: dict | None = None,
    ground_truth_weight: float = 1.0,
) -> dict:
    """Assemble the JSON evidence report saved alongside the model."""
    report = {
        "method": "DecisionTreeClassifier",
        "selection_metric": "validation macro F1; balanced accuracy tie-breaker",
        "training_sources": {
            "segmented_contours": True,
            "ground_truth_contours": False,
            "ground_truth_contours_used_for": "audit only",
        },
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "class_counts": class_counts,
        "candidate_results": candidates,
        "selected_candidate": best_index,
        "selected_parameters": candidates[best_index - 1]["parameters"],
        "final_test": final_metrics,
        "model_path": str(model_path),
    }
    if ground_truth_final_metrics is not None:
        report["ground_truth_final_test"] = ground_truth_final_metrics
    return report


def load_model(model_path: str | Path) -> dict:
    """Load the Decision Tree bundle saved by save_model / scripts/train_decision_tree.py."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Decision Tree model not found: {path}. "
            "Run scripts/train_decision_tree.py first."
        )
    bundle = joblib.load(path)
    saved_columns = tuple(bundle["feature_columns"])
    if saved_columns not in (FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS):
        raise ValueError(
            "Saved Decision Tree uses a different feature-column contract"
        )
    expected_count = len(saved_columns)
    model_count = getattr(bundle["model"], "n_features_in_", expected_count)
    if model_count != expected_count:
        raise ValueError(
            "Saved Decision Tree feature metadata does not match its estimator"
        )
    return bundle


# ---------------------------------------------------------------------------
# Geometry-based frame-artifact rejection
# ---------------------------------------------------------------------------

# Maximum fraction of image width or height a genuine particle bounding box
# may span. The chamber frame/border produces contours that are nearly
# as wide or tall as the full analysis image; real particle tracks do not.
_MAX_SPAN_FRACTION: float = 0.60

# Maximum fraction of total image area that a single particle bounding box
# may occupy. A border/rim contour routinely exceeds this; real particles do
# not even for the largest alpha blobs in the dataset.
_MAX_AREA_FRACTION: float = 0.28


def _is_frame_artifact(
    track: TrackFeatures,
    image_height: int,
    image_width: int,
) -> bool:
    """Return True when a contour's geometry strongly indicates a chamber frame.

    The decision tree has no background/reject class, so it is forced to
    assign every contour to one of the four particle classes. Frame and border
    contours share thin-elongated properties with electron/positron tracks and
    are therefore the most common mis-classification. This guard operates on
    raw image geometry rather than on learned features.
    """
    _x, _y, box_w, box_h = track.bounding_box
    width_span = box_w / max(image_width, 1)
    height_span = box_h / max(image_height, 1)
    box_area_fraction = (box_w * box_h) / max(image_width * image_height, 1)
    return (
        width_span >= _MAX_SPAN_FRACTION
        or height_span >= _MAX_SPAN_FRACTION
        or box_area_fraction >= _MAX_AREA_FRACTION
    )


def predict_tracks(
    model_bundle: dict,
    features: Iterable[TrackFeatures],
    image_shape: tuple[int, int] | None = None,
) -> list[dict]:
    """Classify segmented tracks and report probability and inference time.

    Parameters
    ----------
    model_bundle:
        The bundle returned by :func:`load_model`.
    features:
        Iterable of :class:`~cloud_chamber.features.TrackFeatures` produced
        by the shared segmentation pipeline.
    image_shape:
        Optional ``(height, width)`` of the analysis image passed through the
        pipeline.  When supplied, any contour whose bounding box spans more
        than :data:`_MAX_SPAN_FRACTION` of the image in either dimension, or
        whose bounding-box area exceeds :data:`_MAX_AREA_FRACTION` of the
        total image area, is rejected as a frame artifact before the tree
        evaluates it.  Rejected contours receive confidence ``0.0`` and the
        sentinel class ``"frame_artifact"`` so the UI confidence filter
        suppresses them automatically.
    """
    feature_list = list(features)
    if not feature_list:
        return []

    # Pre-filter: split into accepted/rejected *before* running the model so
    # border contours never receive a spurious high-confidence particle label.
    artifact_mask: list[bool] = [False] * len(feature_list)
    if image_shape is not None:
        img_h, img_w = int(image_shape[0]), int(image_shape[1])
        for i, track in enumerate(feature_list):
            if _is_frame_artifact(track, img_h, img_w):
                artifact_mask[i] = True

    # Build the matrix only for non-artifact contours so the DT never sees
    # out-of-distribution geometry values for frame pixels.
    accepted_indices = [i for i, flag in enumerate(artifact_mask) if not flag]
    accepted_features = [feature_list[i] for i in accepted_indices]

    null_probs = {str(c): 0.0 for c in model_bundle["model"].classes_}
    elapsed_per_track = 0.0
    class_predictions: dict[int, dict] = {}

    if accepted_features:
        raw_matrix = features_to_matrix(accepted_features)
        model = model_bundle["model"]
        saved_columns = tuple(model_bundle.get("feature_columns", FEATURE_COLUMNS))
        matrix = (
            augment_feature_matrix(raw_matrix)
            if saved_columns == MODEL_FEATURE_COLUMNS
            else raw_matrix
        )
        started = perf_counter()
        probabilities = model.predict_proba(matrix)
        labels = model.classes_[np.argmax(probabilities, axis=1)]
        elapsed_per_track = (
            (perf_counter() - started) * 1000.0 / len(matrix)
        )
        for orig_idx, label, probability in zip(
            accepted_indices, labels, probabilities, strict=True
        ):
            class_predictions[orig_idx] = {
                "predicted_class": str(label),
                "particle_type": DISPLAY_NAMES.get(str(label), str(label)),
                "confidence": float(np.max(probability)),
                "probabilities": {
                    str(class_name): float(class_probability)
                    for class_name, class_probability in zip(
                        model.classes_, probability, strict=True
                    )
                },
            }

    results = []
    for i, track in enumerate(feature_list):
        if artifact_mask[i]:
            results.append({
                "track_id": track.track_id,
                "predicted_class": "frame_artifact",
                "particle_type": "Frame artifact",
                "confidence": 0.0,
                "inference_time_ms": 0.0,
                "probabilities": null_probs,
            })
        else:
            entry = class_predictions[i]
            results.append({
                "track_id": track.track_id,
                "predicted_class": entry["predicted_class"],
                "particle_type": entry["particle_type"],
                "confidence": entry["confidence"],
                "inference_time_ms": elapsed_per_track,
                "probabilities": entry["probabilities"],
            })
    return results


def build_visual_report(
    image: np.ndarray,
    features: Iterable[TrackFeatures],
    predictions: list[dict],
    confidence_threshold: float = 0.60,
) -> tuple[np.ndarray, list[dict]]:
    """Draw and report only predictions meeting the confidence threshold."""
    feature_list = list(features)
    if len(feature_list) != len(predictions):
        raise ValueError("Feature and prediction counts must be equal")

    overlay = image.copy()
    rows = []
    for track, prediction in zip(feature_list, predictions, strict=True):
        x, y, width, height = track.bounding_box
        confidence = float(prediction["confidence"])
        if confidence < confidence_threshold:
            # The threshold is a display/export filter, not merely a status
            # label.  Keeping these predictions out of the overlay prevents
            # low-confidence boxes from being mistaken for detections.
            continue
        colour = CLASS_COLOURS.get(prediction["predicted_class"], (255, 255, 255))

        cv2.rectangle(overlay, (x, y), (x + width, y + height), colour, 2)
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
                "Status": "Accepted",
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
