"""Shared dataset discovery and image/video acquisition."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2

from cloud_chamber.core.contracts import ImageSample


DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
DEFAULT_EXCLUDED_FOLDERS = {"annotated", "bw"}


def load_image(path: str | Path) -> ImageSample:
    """Decode one colour image and retain its traceable source path."""
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return ImageSample(
        sample_id=image_path.stem,
        image=image,
        source_path=image_path.resolve(),
    )


def load_image_folder(
    folder: str | Path,
    extensions: set[str] | None = None,
    recursive: bool = False,
) -> list[ImageSample]:
    """Load supported images from one folder in deterministic name order."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Image folder not found: {folder_path}")

    allowed = {item.lower() for item in (extensions or DEFAULT_IMAGE_EXTENSIONS)}
    candidates = folder_path.rglob("*") if recursive else folder_path.iterdir()
    paths = sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in allowed
    )
    return [load_image(path) for path in paths]


def discover_raw_images(
    dataset_root: str | Path,
    extensions: set[str] | None = None,
    excluded_folder_names: set[str] | None = None,
) -> list[Path]:
    """Find unmodified image inputs while excluding Annotated and BW folders."""
    root = _require_directory(dataset_root, "Extracted dataset")
    allowed = {item.lower() for item in (extensions or DEFAULT_IMAGE_EXTENSIONS)}
    excluded = {
        name.casefold()
        for name in (excluded_folder_names or DEFAULT_EXCLUDED_FOLDERS)
    }

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and not _contains_excluded_folder(path, root, excluded)
    )


def discover_videos(
    dataset_root: str | Path,
    extensions: set[str] | None = None,
) -> list[Path]:
    """Find videos recursively inside an extracted dataset."""
    root = _require_directory(dataset_root, "Extracted dataset")
    allowed = {item.lower() for item in (extensions or DEFAULT_VIDEO_EXTENSIONS)}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )


def iter_video_frames(
    video_path: str | Path,
    frame_interval: int = 30,
) -> Iterator[ImageSample]:
    """Yield every ``frame_interval`` frame and always release the video."""
    if frame_interval < 1:
        raise ValueError("frame_interval must be at least 1")

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    try:
        frame_number = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_number % frame_interval == 0:
                yield ImageSample(
                    sample_id=f"{path.stem}_frame_{frame_number:06d}",
                    image=frame.copy(),
                    source_path=path.resolve(),
                    frame_number=frame_number,
                )
            frame_number += 1
    finally:
        capture.release()


def export_video_frames(
    video_path: str | Path,
    output_root: str | Path,
    frame_interval: int = 30,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> list[Path]:
    """Save regularly sampled video frames as reproducible JPG inputs."""
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    video = Path(video_path)
    destination = Path(output_root) / video.stem
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for sample in iter_video_frames(video, frame_interval=frame_interval):
        output_path = destination / f"{sample.sample_id}.jpg"
        if output_path.exists() and not overwrite:
            written.append(output_path)
            continue

        success = cv2.imwrite(
            str(output_path),
            sample.image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not success:
            raise OSError(f"Unable to write extracted frame: {output_path}")
        written.append(output_path)

    return written


def dataset_summary(
    dataset_root: str | Path,
    acquisition_settings: dict[str, Any],
) -> dict[str, Any]:
    """Summarise discoverable raw inputs without decoding the full dataset."""
    image_extensions = set(acquisition_settings["allowed_image_extensions"])
    video_extensions = set(acquisition_settings["allowed_video_extensions"])
    excluded = set(acquisition_settings.get("excluded_folder_names", []))
    images = discover_raw_images(
        dataset_root,
        extensions=image_extensions,
        excluded_folder_names=excluded,
    )
    videos = discover_videos(dataset_root, extensions=video_extensions)
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "raw_image_count": len(images),
        "video_count": len(videos),
        "first_raw_images": [str(path) for path in images[:5]],
        "first_videos": [str(path) for path in videos[:5]],
    }


def _require_directory(path: str | Path, label: str) -> Path:
    """Return a validated directory or raise an error with user guidance."""
    folder = Path(path)
    if not folder.is_dir():
        raise NotADirectoryError(
            f"{label} folder not found: {folder}. "
            "Extract dataset.zip before scanning or analysing its contents."
        )
    return folder


def _contains_excluded_folder(
    path: Path,
    root: Path,
    excluded: set[str],
) -> bool:
    """Report whether a path is inside a generated or annotated folder."""
    relative_parent_parts = path.relative_to(root).parent.parts
    return any(part.casefold() in excluded for part in relative_parent_parts)
