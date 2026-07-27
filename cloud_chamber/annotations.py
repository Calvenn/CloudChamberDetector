"""Ground-truth mask and particle annotation loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUPPORTED_CLASSES = {
    "unknown",
    "alpha",
    "electron_positron",
    "proton",
}


def load_ground_truth_mask(
    path: str | Path,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    mask_path = Path(path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read ground-truth mask: {mask_path}")

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if expected_shape is not None and binary.shape != expected_shape:
        raise ValueError(
            f"Ground-truth shape {binary.shape} does not match {expected_shape}"
        )
    return binary


def load_particle_annotations(path: str | Path) -> dict[str, Any]:
    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    tracks = annotations.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("Annotation file must contain a 'tracks' list")

    for track in tracks:
        particle_class = track.get("class")
        if particle_class not in SUPPORTED_CLASSES:
            raise ValueError(f"Unsupported particle class: {particle_class}")
        bbox = track.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Each track must contain bbox [x, y, width, height]")

    return annotations

