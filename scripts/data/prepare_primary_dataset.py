"""Convert the supplied colour-box annotations into training metadata.

The source ZIP remains unchanged. Raw images, annotation references, JSON
labels, review overlays, a CSV manifest, and COCO bounding-box metadata are
written into a new session-safe primary dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


SOURCE_PREFIX = "01 - Annotated Capture"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SESSION_SPLIT = {
    "VID_20220609_215401": "development",
    "VID_20220609_220549": "development",
    "VID_20220609_220753": "validation",
    "VID_20220609_221024": "final_test",
}
CLASS_IDS = {
    "alpha": 1,
    "electron_positron": 2,
}
CLASS_COLOURS_BGR = {
    "alpha": (0, 255, 255),
    "electron_positron": (0, 0, 255),
}
SNAPSHOT_PATTERN = re.compile(r"_snapshot_(\d{2})\.(\d{2})_")


@dataclass(frozen=True)
class ImagePair:
    session: str
    split: str
    sample_id: str
    video_path: str
    supplied_bw_path: str
    annotated_path: str
    annotated_time_seconds: int


def prepare_primary_dataset(
    archive_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Prepare all paired raw/annotated images and extract colour boxes."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    coco_images: list[dict[str, object]] = []
    coco_annotations: list[dict[str, object]] = []
    next_annotation_id = 1

    with zipfile.ZipFile(archive_path) as archive:
        pairs = pair_images(archive.namelist())
        with tempfile.TemporaryDirectory(prefix="cloud_chamber_videos_") as temp:
            captures = _open_source_videos(archive, pairs, Path(temp))
            try:
                for image_id, pair in enumerate(pairs, start=1):
                    annotated_bytes = archive.read(pair.annotated_path)
                    annotated_image = _decode_image(
                        annotated_bytes,
                        pair.annotated_path,
                    )
                    raw_image = _capture_video_frame(
                        captures[pair.session],
                        pair.annotated_time_seconds,
                        pair.video_path,
                    )
                    if raw_image.shape != annotated_image.shape:
                        raise ValueError(
                            f"Image dimensions differ for {pair.sample_id}: "
                            f"{raw_image.shape} vs {annotated_image.shape}"
                        )
                    raw_bytes = _encode_jpeg(raw_image)

                    boxes = extract_annotation_boxes(annotated_image)
                    height, width = raw_image.shape[:2]
                    image_relative = Path(pair.split) / "images" / f"{pair.sample_id}.jpg"
                    label_relative = Path(pair.split) / "labels" / f"{pair.sample_id}.json"
                    reference_relative = (
                        Path(pair.split)
                        / "annotation_reference"
                        / f"{pair.sample_id}.jpg"
                    )
                    overlay_relative = (
                        Path(pair.split)
                        / "review_overlays"
                        / f"{pair.sample_id}.jpg"
                    )

                    _write_bytes_if_changed(output_root / image_relative, raw_bytes)
                    _write_bytes_if_changed(
                        output_root / reference_relative,
                        annotated_bytes,
                    )
                    overlay = draw_review_overlay(raw_image, boxes)
                    _write_image(output_root / overlay_relative, overlay)

                    tracks = []
                    for track_number, box in enumerate(boxes, start=1):
                        x, y, box_width, box_height = box["bbox"]
                        class_name = str(box["class"])
                        tracks.append(
                            {
                                "track_id": track_number,
                                "class": class_name,
                                "class_id": CLASS_IDS[class_name],
                                "bbox": [x, y, box_width, box_height],
                                "annotation_colour": box["annotation_colour"],
                                "source": "supplied_colour_bounding_box",
                            }
                        )
                        coco_annotations.append(
                            {
                                "id": next_annotation_id,
                                "image_id": image_id,
                                "category_id": CLASS_IDS[class_name],
                                "bbox": [x, y, box_width, box_height],
                                "area": box_width * box_height,
                                "iscrowd": 0,
                            }
                        )
                        next_annotation_id += 1

                    label_document = {
                        "schema_version": "1.0",
                        "image_id": pair.sample_id,
                        "split": pair.split,
                        "session": pair.session,
                        "width": width,
                        "height": height,
                        "raw_image": image_relative.as_posix(),
                        "annotation_reference": reference_relative.as_posix(),
                        "tracks": tracks,
                        "needs_review": len(tracks) == 0,
                    }
                    _write_json(output_root / label_relative, label_document)

                    coco_images.append(
                        {
                            "id": image_id,
                            "file_name": image_relative.as_posix(),
                            "width": width,
                            "height": height,
                            "split": pair.split,
                            "session": pair.session,
                        }
                    )
                    records.append(
                        {
                            "image_id": pair.sample_id,
                            "split": pair.split,
                            "session": pair.session,
                            "video_time_seconds": pair.annotated_time_seconds,
                            "alpha_boxes": sum(
                                track["class"] == "alpha" for track in tracks
                            ),
                            "electron_positron_boxes": sum(
                                track["class"] == "electron_positron"
                                for track in tracks
                            ),
                            "total_boxes": len(tracks),
                            "needs_review": len(tracks) == 0,
                            "raw_image": image_relative.as_posix(),
                            "label": label_relative.as_posix(),
                            "annotation_reference": reference_relative.as_posix(),
                            "review_overlay": overlay_relative.as_posix(),
                            "original_video_path": pair.video_path,
                            "supplied_bw_path": pair.supplied_bw_path,
                            "original_annotated_path": pair.annotated_path,
                        }
                    )
            finally:
                for capture in captures.values():
                    capture.release()

    _validate_prepared_records(records)
    _write_manifest(output_root / "manifest.csv", records)
    _write_json(
        output_root / "annotations_coco.json",
        {
            "info": {
                "description": (
                    "Primary cloud-chamber dataset converted from supplied "
                    "red/yellow bounding-box annotations"
                ),
                "version": "1.0",
            },
            "categories": [
                {"id": 1, "name": "alpha"},
                {"id": 2, "name": "electron_positron"},
            ],
            "images": coco_images,
            "annotations": coco_annotations,
        },
    )
    _write_json(
        output_root / "classes.json",
        {
            "background": 0,
            **CLASS_IDS,
            "unknown": 3,
            "colour_mapping": {
                "yellow": "alpha",
                "red": "electron_positron",
            },
        },
    )

    summary = _build_summary(records)
    _write_json(output_root / "summary.json", summary)
    return summary


def pair_images(archive_names: list[str]) -> list[ImagePair]:
    """Pair raw and annotated images by session and chronological order."""
    pairs: list[ImagePair] = []
    for session, split in SESSION_SPLIT.items():
        raw_paths = sorted(
            (
                name
                for name in archive_names
                if _is_raw_session_image(name, session)
            ),
            key=_snapshot_sort_key,
        )
        annotated_paths = sorted(
            (
                name
                for name in archive_names
                if _is_annotated_session_image(name, session)
            ),
            key=_snapshot_sort_key,
        )
        if len(raw_paths) != len(annotated_paths):
            raise ValueError(
                f"Session {session} has {len(raw_paths)} raw images but "
                f"{len(annotated_paths)} annotated images"
            )

        video_path = (
            f"{SOURCE_PREFIX}/{session}/{session}.mp4"
        )
        duplicate_index: Counter[int] = Counter()
        for raw_path, annotated_path in zip(raw_paths, annotated_paths):
            annotated_seconds = _snapshot_seconds(annotated_path)
            raw_seconds = _snapshot_seconds(raw_path)
            if abs(raw_seconds - annotated_seconds) > 1:
                raise ValueError(
                    f"Unreliable pair in {session}: {raw_path} and "
                    f"{annotated_path}"
                )
            duplicate_index[annotated_seconds] += 1
            sample_id = (
                f"{session}_t{annotated_seconds:04d}_"
                f"{duplicate_index[annotated_seconds]:02d}"
            )
            pairs.append(
                ImagePair(
                    session=session,
                    split=split,
                    sample_id=sample_id,
                    video_path=video_path,
                    supplied_bw_path=raw_path,
                    annotated_path=annotated_path,
                    annotated_time_seconds=annotated_seconds,
                )
            )
    return pairs


def extract_annotation_boxes(
    annotated_image: np.ndarray,
) -> list[dict[str, object]]:
    """Extract axis-aligned red and yellow boxes from one annotated image."""
    masks = _annotation_colour_masks(annotated_image)
    boxes: list[dict[str, object]] = []
    for class_name, colour_name in (
        ("alpha", "yellow"),
        ("electron_positron", "red"),
    ):
        for bbox in _rectangles_from_mask(masks[colour_name]):
            boxes.append(
                {
                    "class": class_name,
                    "annotation_colour": colour_name,
                    "bbox": list(bbox),
                }
            )
    return sorted(
        boxes,
        key=lambda item: (
            CLASS_IDS[str(item["class"])],
            item["bbox"][1],
            item["bbox"][0],
        ),
    )


def draw_review_overlay(
    raw_image: np.ndarray,
    boxes: list[dict[str, object]],
) -> np.ndarray:
    """Draw extracted labels on the unmodified raw image for human review."""
    overlay = raw_image.copy()
    for box in boxes:
        x, y, width, height = box["bbox"]
        class_name = str(box["class"])
        colour = CLASS_COLOURS_BGR[class_name]
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width - 1, y + height - 1),
            colour,
            3,
        )
        cv2.putText(
            overlay,
            class_name,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )
    return overlay


def _annotation_colour_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    blue, green, red = cv2.split(image)
    red_float = red.astype(np.float32)
    green_float = green.astype(np.float32)
    blue_float = blue.astype(np.float32)

    red_mask = (
        (red > 150)
        & (red_float > green_float * 1.35)
        & (red_float > blue_float * 1.35)
    )
    yellow_mask = (
        (red > 150)
        & (green > 130)
        & (blue < 150)
        & (red_float > blue_float * 1.25)
        & (green_float > blue_float * 1.20)
    )
    return {
        "red": red_mask.astype(np.uint8) * 255,
        "yellow": yellow_mask.astype(np.uint8) * 255,
    }


def _rectangles_from_mask(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)

    horizontal_lines = _extract_lines(horizontal_mask, horizontal=True)
    vertical_lines = _extract_lines(vertical_mask, horizontal=False)
    horizontal_lines = _merge_collinear_lines(
        horizontal_lines,
        horizontal=True,
    )
    vertical_lines = _merge_collinear_lines(
        vertical_lines,
        horizontal=False,
    )

    rectangles: list[tuple[int, int, int, int]] = []
    for top_index, top in enumerate(horizontal_lines):
        top_position, top_start, top_end = top
        for bottom in horizontal_lines[top_index + 1 :]:
            bottom_position, bottom_start, bottom_end = bottom
            if bottom_position - top_position < 25:
                continue
            if abs(top_start - bottom_start) > 12:
                continue
            if abs(top_end - bottom_end) > 12:
                continue

            left = round((top_start + bottom_start) / 2)
            right = round((top_end + bottom_end) / 2)
            if not _has_supporting_vertical(
                vertical_lines,
                left,
                top_position,
                bottom_position,
            ):
                continue
            if not _has_supporting_vertical(
                vertical_lines,
                right,
                top_position,
                bottom_position,
            ):
                continue

            rectangle = (
                left,
                top_position,
                right - left + 1,
                bottom_position - top_position + 1,
            )
            rectangles.append(rectangle)

    # Intersecting boxes can merge their horizontal edges into one component.
    # Reconstructing the same candidates from pairs of vertical edges recovers
    # those cases without requiring the boxes to be manually redrawn.
    for left_index, left_line in enumerate(vertical_lines):
        left_position, left_start, left_end = left_line
        for right_line in vertical_lines[left_index + 1 :]:
            right_position, right_start, right_end = right_line
            if right_position - left_position < 25:
                continue
            if abs(left_start - right_start) > 12:
                continue
            if abs(left_end - right_end) > 12:
                continue

            top = round((left_start + right_start) / 2)
            bottom = round((left_end + right_end) / 2)
            if not _has_supporting_horizontal(
                horizontal_lines,
                top,
                left_position,
                right_position,
            ):
                continue
            if not _has_supporting_horizontal(
                horizontal_lines,
                bottom,
                left_position,
                right_position,
            ):
                continue
            rectangles.append(
                (
                    left_position,
                    top,
                    right_position - left_position + 1,
                    bottom - top + 1,
                )
            )
    return _deduplicate_rectangles(rectangles)


def _extract_lines(
    line_mask: np.ndarray,
    horizontal: bool,
) -> list[tuple[int, int, int]]:
    contours, _ = cv2.findContours(
        line_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    lines = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        length = width if horizontal else height
        thickness = height if horizontal else width
        if length < 25 or thickness > 15:
            continue
        if horizontal:
            lines.append((round(y + height / 2), x, x + width - 1))
        else:
            lines.append((round(x + width / 2), y, y + height - 1))
    return lines


def _merge_collinear_lines(
    lines: list[tuple[int, int, int]],
    horizontal: bool,
) -> list[tuple[int, int, int]]:
    del horizontal  # Both orientations use the same normalised representation.
    pending = sorted(lines)
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int]] = []
        while pending:
            position, start, end = pending.pop(0)
            match_index = None
            for index, candidate in enumerate(pending):
                other_position, other_start, other_end = candidate
                if abs(position - other_position) > 6:
                    continue
                if other_start > end + 15 or start > other_end + 15:
                    continue
                match_index = index
                position = round((position + other_position) / 2)
                start = min(start, other_start)
                end = max(end, other_end)
                break
            if match_index is not None:
                pending.pop(match_index)
                pending.append((position, start, end))
                pending.sort()
                changed = True
            else:
                merged.append((position, start, end))
        pending = sorted(merged)
    return pending


def _has_supporting_vertical(
    vertical_lines: list[tuple[int, int, int]],
    expected_x: int,
    top: int,
    bottom: int,
) -> bool:
    for x, start, end in vertical_lines:
        if abs(x - expected_x) > 12:
            continue
        if start <= top + 15 and end >= bottom - 15:
            return True
    return False


def _has_supporting_horizontal(
    horizontal_lines: list[tuple[int, int, int]],
    expected_y: int,
    left: int,
    right: int,
) -> bool:
    for y, start, end in horizontal_lines:
        if abs(y - expected_y) > 12:
            continue
        if start <= left + 15 and end >= right - 15:
            return True
    return False


def _deduplicate_rectangles(
    rectangles: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    unique: list[tuple[int, int, int, int]] = []
    for rectangle in sorted(rectangles, key=lambda item: item[2] * item[3]):
        if any(
            max(abs(a - b) for a, b in zip(rectangle, existing)) <= 12
            for existing in unique
        ):
            continue
        unique.append(rectangle)
    return sorted(unique, key=lambda item: (item[1], item[0]))


def _build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    split_images = Counter(str(record["split"]) for record in records)
    alpha_by_split = Counter()
    electron_by_split = Counter()
    for record in records:
        split = str(record["split"])
        alpha_by_split[split] += int(record["alpha_boxes"])
        electron_by_split[split] += int(record["electron_positron_boxes"])

    return {
        "schema_version": "1.0",
        "total_images": len(records),
        "total_boxes": sum(int(record["total_boxes"]) for record in records),
        "class_counts": {
            "alpha": sum(int(record["alpha_boxes"]) for record in records),
            "electron_positron": sum(
                int(record["electron_positron_boxes"]) for record in records
            ),
        },
        "split_image_counts": dict(sorted(split_images.items())),
        "split_class_counts": {
            split: {
                "alpha": alpha_by_split[split],
                "electron_positron": electron_by_split[split],
            }
            for split in sorted(split_images)
        },
        "images_needing_review": [
            str(record["image_id"])
            for record in records
            if bool(record["needs_review"])
        ],
        "notes": [
            "Clean inputs are frames captured from the original MP4 files.",
            "Supplied S/BW snapshots are excluded as model inputs.",
            "Red supplied boxes are mapped to electron_positron.",
            "Yellow supplied boxes are mapped to alpha.",
            "Bounding boxes are ground truth for object detection, not masks.",
            "The source ZIP remains unchanged.",
        ],
    }


def _validate_prepared_records(records: list[dict[str, object]]) -> None:
    if len(records) != 77:
        raise ValueError(f"Expected 77 paired images, found {len(records)}")
    image_ids = [str(record["image_id"]) for record in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Prepared image identifiers are not unique")
    sessions_by_split: dict[str, set[str]] = {}
    for record in records:
        session = str(record["session"])
        sessions_by_split.setdefault(session, set()).add(str(record["split"]))
    leaking = {
        session: splits
        for session, splits in sessions_by_split.items()
        if len(splits) > 1
    }
    if leaking:
        raise ValueError(f"Recording-session leakage detected: {leaking}")


def _write_manifest(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def _write_bytes_if_changed(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    path.write_bytes(content)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Unable to write image: {path}")


def _decode_image(content: bytes, source_name: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {source_name}")
    return image


def _encode_jpeg(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise OSError("Unable to encode captured video frame")
    return encoded.tobytes()


def _open_source_videos(
    archive: zipfile.ZipFile,
    pairs: list[ImagePair],
    temp_root: Path,
) -> dict[str, cv2.VideoCapture]:
    captures: dict[str, cv2.VideoCapture] = {}
    for pair in pairs:
        if pair.session in captures:
            continue
        video_file = temp_root / f"{pair.session}.mp4"
        video_file.write_bytes(archive.read(pair.video_path))
        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            raise ValueError(f"Unable to open source video: {pair.video_path}")
        captures[pair.session] = capture
    return captures


def _capture_video_frame(
    capture: cv2.VideoCapture,
    time_seconds: int,
    source_name: str,
) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
    success, frame = capture.read()
    if not success or frame is None:
        raise ValueError(
            f"Unable to capture {time_seconds}s frame from {source_name}"
        )
    return frame


def _is_raw_session_image(name: str, session: str) -> bool:
    path = PurePosixPath(name)
    return (
        len(path.parts) == 3
        and path.parts[0] == SOURCE_PREFIX
        and path.parts[1] == session
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _is_annotated_session_image(name: str, session: str) -> bool:
    path = PurePosixPath(name)
    return (
        len(path.parts) == 4
        and path.parts[0] == SOURCE_PREFIX
        and path.parts[1] == session
        and path.parts[2] == "Annotated"
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _snapshot_seconds(name: str) -> int:
    match = SNAPSHOT_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Snapshot time not found in filename: {name}")
    return int(match.group(1)) * 60 + int(match.group(2))


def _snapshot_sort_key(name: str) -> tuple[int, str]:
    return _snapshot_seconds(name), name


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the primary cloud-chamber dataset and convert supplied "
            "red/yellow boxes into JSON and COCO labels."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "primary_dataset",
    )
    arguments = parser.parse_args()
    summary = prepare_primary_dataset(arguments.archive, arguments.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

