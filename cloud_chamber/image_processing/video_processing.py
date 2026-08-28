"""Video-only preprocessing for faint, transient cloud-chamber tracks."""

from __future__ import annotations

import cv2
import numpy as np


def temporal_track_composite(
    frames: list[np.ndarray],
    *,
    reference_index: int | None = None,
    evidence_gain: float = 2.0,
    stabilise: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Boost transient bright tracks while retaining the reference frame.

    A temporal median estimates static chamber texture. Positive differences
    from that background are maximum-projected across the window, then added
    to the reference frame. This output can pass through the unchanged still-
    image segmentation and classification pipeline.
    """
    if not frames:
        raise ValueError("At least one video frame is required")
    if evidence_gain < 0:
        raise ValueError("evidence_gain must be non-negative")
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("Video frames must be BGR colour images")
    if any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames):
        raise ValueError("All video frames must have equal shape and uint8 dtype")

    centre = len(frames) // 2 if reference_index is None else int(reference_index)
    if centre < 0 or centre >= len(frames):
        raise ValueError("reference_index is outside the supplied frame window")

    aligned = _stabilise_to_reference(frames, centre) if stabilise else frames
    grey_stack = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in aligned]
    ).astype(np.float32)
    background = np.median(grey_stack, axis=0)
    positive_difference = np.maximum(grey_stack - background, 0.0)
    evidence = np.max(positive_difference, axis=0)
    evidence = cv2.GaussianBlur(evidence, (3, 3), 0)
    evidence_u8 = np.clip(evidence, 0, 255).astype(np.uint8)

    reference = aligned[centre]
    boosted = cv2.add(
        reference,
        cv2.cvtColor(
            np.clip(evidence * evidence_gain, 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        ),
    )
    return boosted, evidence_u8


def _stabilise_to_reference(
    frames: list[np.ndarray], reference_index: int
) -> list[np.ndarray]:
    """Align small translational camera movements using phase correlation."""
    reference = frames[reference_index]
    reference_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = reference_grey.shape
    maximum_shift = 0.05 * max(height, width)
    aligned: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if index == reference_index:
            aligned.append(frame)
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (shift_x, shift_y), response = cv2.phaseCorrelate(grey, reference_grey)
        if response < 0.05 or abs(shift_x) > maximum_shift or abs(shift_y) > maximum_shift:
            aligned.append(frame)
            continue
        transform = np.asarray(
            [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32
        )
        aligned.append(
            cv2.warpAffine(
                frame,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return aligned

