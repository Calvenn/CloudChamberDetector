"""Shared, reproducible mask-level evaluation functions."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cloud_chamber.models import DetectionResult, EvaluationResult


def evaluate_mask(
    detection: DetectionResult,
    ground_truth_mask: np.ndarray,
) -> EvaluationResult:
    if detection.binary_mask.shape != ground_truth_mask.shape:
        raise ValueError("Prediction and ground-truth masks must have equal shapes")

    predicted = detection.binary_mask > 0
    expected = ground_truth_mask > 0

    true_positive = int(np.logical_and(predicted, expected).sum())
    false_positive = int(np.logical_and(predicted, ~expected).sum())
    false_negative = int(np.logical_and(~predicted, expected).sum())

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    union = int(np.logical_or(predicted, expected).sum())
    iou = _safe_divide(true_positive, union)
    dice = _safe_divide(
        2 * true_positive,
        int(predicted.sum()) + int(expected.sum()),
    )

    return EvaluationResult(
        method_name=detection.method_name,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        iou=iou,
        dice=dice,
        processing_time_ms=detection.processing_time_ms,
    )


def export_evaluations_csv(
    evaluations: list[EvaluationResult],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_name",
        "precision",
        "recall",
        "f1_score",
        "iou",
        "dice",
        "processing_time_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in evaluations)
    return path


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0

