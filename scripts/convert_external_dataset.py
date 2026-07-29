"""Convert the Müller cloud-chamber masks to COCO instance annotations.

The public Müller dataset stores one semantic mask per image as a NumPy
``.npz`` array with shape ``(height, width, 5)``. The channel order, verified
against the authors' public code, is:

0. alpha
1. electron
2. proton
3. V track
4. background

Mask R-CNN requires individual object instances. This script treats each
8-connected region in a particle channel as one track instance and exports:

* a COCO JSON file containing class labels, bounding boxes and binary RLE masks;
* a CSV manifest with per-image instance counts;
* a JSON conversion summary for auditing.

The source images and masks are never modified. Semantic masks cannot identify
two touching tracks of the same class as separate instances, so the converted
annotations should be visually reviewed before final model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


MULLER_CHANNELS: tuple[tuple[int, int, str], ...] = (
    (0, 1, "alpha"),
    (1, 2, "electron_positron"),
    (2, 3, "proton"),
    (3, 4, "v_track"),
)
BACKGROUND_CHANNEL = 4
EXPECTED_CHANNELS = 5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Müller five-channel semantic masks into COCO instance "
            "segmentation annotations for Mask R-CNN."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/external_dataset"),
        help="Folder containing images/ and masks/ (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/external_dataset_coco"),
        help="Output folder for COCO annotations (default: %(default)s).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help=(
            "Discard connected regions smaller than this many pixels "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Value used to binarise each mask channel (default: %(default)s).",
    )
    parser.add_argument(
        "--strict-one-hot",
        action="store_true",
        help="Fail if pixels are assigned to zero or multiple mask channels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing generated annotation files.",
    )
    return parser.parse_args()


def resolve_from_project(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def collect_pairs(source: Path) -> list[ImageMaskPair]:
    image_dir = source / "images"
    mask_dir = source / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask folder not found: {mask_dir}")

    images: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for image_path in image_dir.glob(f"*{extension}"):
            if image_path.stem in images:
                raise ValueError(f"Duplicate image stem: {image_path.stem}")
            images[image_path.stem] = image_path

    masks = {mask_path.stem: mask_path for mask_path in mask_dir.glob("*.npz")}
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        messages = []
        if missing_masks:
            messages.append(
                f"{len(missing_masks)} images without masks, e.g. "
                f"{missing_masks[:3]}"
            )
        if missing_images:
            messages.append(
                f"{len(missing_images)} masks without images, e.g. "
                f"{missing_images[:3]}"
            )
        raise ValueError("; ".join(messages))
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")

    return [
        ImageMaskPair(images[stem], masks[stem])
        for stem in sorted(images, key=str.casefold)
    ]


def load_muller_mask(mask_path: Path) -> np.ndarray:
    with np.load(mask_path, allow_pickle=False) as archive:
        if archive.files != ["arr_0"]:
            raise ValueError(
                f"{mask_path.name}: expected NPZ key 'arr_0', "
                f"found {archive.files}"
            )
        mask = archive["arr_0"]

    if mask.ndim != 3 or mask.shape[2] != EXPECTED_CHANNELS:
        raise ValueError(
            f"{mask_path.name}: expected HxWx{EXPECTED_CHANNELS}, got {mask.shape}"
        )
    if not np.isfinite(mask).all():
        raise ValueError(f"{mask_path.name}: mask contains NaN or infinity")
    return mask


def encode_uncompressed_rle(binary_mask: np.ndarray) -> dict[str, Any]:
    """Return COCO's JSON-compatible, uncompressed column-major RLE."""
    pixels = np.asarray(binary_mask, dtype=np.uint8).reshape(-1, order="F")
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [pixels.size]))
    counts = np.diff(boundaries).astype(int).tolist()
    if pixels.size and pixels[0] != 0:
        counts.insert(0, 0)
    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": counts,
    }


def connected_instances(
    class_mask: np.ndarray,
    min_area: int,
) -> Iterable[tuple[np.ndarray, list[int], int]]:
    binary = np.asarray(class_mask, dtype=np.uint8)
    component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    for component_id in range(1, component_count):
        x, y, width, height, area = (
            int(value) for value in stats[component_id]
        )
        if area < min_area:
            continue
        instance_mask = component_map == component_id
        yield instance_mask, [x, y, width, height], area


def relative_image_name(image_path: Path, output: Path) -> str:
    """Create a portable COCO filename relative to the output dataset folder."""
    try:
        relative = image_path.resolve().relative_to(output.parent.resolve())
        return relative.as_posix()
    except ValueError:
        return image_path.resolve().as_posix()


def ensure_output_paths(output: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "coco": output / "annotations_coco.json",
        "manifest": output / "manifest.csv",
        "summary": output / "summary.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Generated files already exist: {names}. "
            "Use --overwrite to replace them."
        )
    output.mkdir(parents=True, exist_ok=True)
    return paths


def convert(
    source: Path,
    output: Path,
    min_area: int,
    threshold: float,
    strict_one_hot: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if min_area < 1:
        raise ValueError("--min-area must be at least 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    pairs = collect_pairs(source)
    output_paths = ensure_output_paths(output, overwrite)
    categories = [
        {"id": category_id, "name": class_name, "supercategory": "particle_track"}
        for _, category_id, class_name in MULLER_CHANNELS
    ]
    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    discarded_counts: Counter[str] = Counter()
    pixels_without_class = 0
    overlapping_channel_pixels = 0
    annotation_id = 1

    for image_id, pair in enumerate(pairs, start=1):
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {pair.image_path}")
        height, width = image.shape[:2]
        semantic_mask = load_muller_mask(pair.mask_path)
        if semantic_mask.shape[:2] != (height, width):
            raise ValueError(
                f"{pair.image_path.name}: image is {(height, width)}, but mask "
                f"is {semantic_mask.shape[:2]}"
            )

        binary_channels = semantic_mask >= threshold
        assignments = binary_channels.sum(axis=2)
        missing = int(np.count_nonzero(assignments == 0))
        overlapping = int(np.count_nonzero(assignments > 1))
        pixels_without_class += missing
        overlapping_channel_pixels += overlapping
        if strict_one_hot and (missing or overlapping):
            raise ValueError(
                f"{pair.mask_path.name}: one-hot violation: "
                f"{missing} unassigned and {overlapping} overlapping pixels"
            )

        image_record = {
            "id": image_id,
            "file_name": relative_image_name(pair.image_path, output),
            "width": width,
            "height": height,
            "source": "muller_2023",
        }
        coco_images.append(image_record)
        per_image_counts = {class_name: 0 for _, _, class_name in MULLER_CHANNELS}

        for channel, category_id, class_name in MULLER_CHANNELS:
            raw_component_count = 0
            kept_component_count = 0
            class_binary = binary_channels[:, :, channel]
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                np.asarray(class_binary, dtype=np.uint8),
                connectivity=8,
            )
            raw_component_count = component_count - 1
            discarded_counts[class_name] += sum(
                int(stats[index, cv2.CC_STAT_AREA]) < min_area
                for index in range(1, component_count)
            )

            for instance_mask, bbox, area in connected_instances(
                class_binary,
                min_area,
            ):
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "segmentation": encode_uncompressed_rle(instance_mask),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
                kept_component_count += 1

            if kept_component_count > raw_component_count:
                raise AssertionError("Kept more components than were detected")
            per_image_counts[class_name] = kept_component_count
            class_counts[class_name] += kept_component_count

        manifest_rows.append(
            {
                "image_id": image_id,
                "file_name": image_record["file_name"],
                "mask_file": pair.mask_path.name,
                **per_image_counts,
                "total_instances": sum(per_image_counts.values()),
            }
        )

    now = datetime.now(timezone.utc).isoformat()
    coco = {
        "info": {
            "description": (
                "Müller et al. cloud-chamber semantic masks converted to "
                "connected-component instances"
            ),
            "version": "1.0",
            "date_created": now,
            "conversion_note": (
                "Each 8-connected region is treated as one instance. Touching "
                "tracks of the same class cannot be separated automatically."
            ),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": categories,
    }
    summary = {
        "source": str(source),
        "output": str(output),
        "created_at": now,
        "source_format": "HxWx5 one-hot semantic NPZ mask",
        "channel_mapping": {
            "0": "alpha",
            "1": "electron_positron",
            "2": "proton",
            "3": "v_track",
            "4": "background",
        },
        "conversion": {
            "connectivity": 8,
            "threshold": threshold,
            "minimum_component_area_pixels": min_area,
            "segmentation_encoding": "COCO uncompressed RLE",
        },
        "image_count": len(coco_images),
        "annotation_count": len(annotations),
        "annotations_by_class": dict(class_counts),
        "discarded_small_components_by_class": dict(discarded_counts),
        "one_hot_audit": {
            "unassigned_pixels": pixels_without_class,
            "overlapping_channel_pixels": overlapping_channel_pixels,
        },
        "limitation": (
            "Connected components approximate instances. Manually review "
            "touching tracks and fragmented tracks before final training."
        ),
    }

    with output_paths["coco"].open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, indent=2)
    with output_paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with output_paths["manifest"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return summary


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source = resolve_from_project(args.source, project_root)
    output = resolve_from_project(args.output, project_root)
    try:
        summary = convert(
            source=source,
            output=output,
            min_area=args.min_area,
            threshold=args.threshold,
            strict_one_hot=args.strict_one_hot,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
