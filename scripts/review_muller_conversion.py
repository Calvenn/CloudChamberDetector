"""Generate visual checks for converted Müller COCO instance annotations.

Each output panel shows:

1. the original cloud-chamber image;
2. the annotation preview supplied with the Müller dataset;
3. the connected-component instances exported for Mask R-CNN.

The script samples images from development, validation and final-test splits,
while prioritising images containing the rarer proton and V-track classes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_COLOURS = {
    "alpha": (0, 180, 0),
    "electron_positron": (220, 80, 20),
    "proton": (20, 20, 230),
    "v_track": (180, 30, 180),
}
SPLITS = ("development", "validation", "final_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create side-by-side visual reviews of Müller annotations."
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("dataset/external_dataset_split"),
        help="Folder containing the three COCO splits (default: %(default)s).",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("dataset/external_dataset"),
        help="Folder containing images/ and inspection/ (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/external_dataset_review"),
        help="Output folder for review panels (default: %(default)s).",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=12,
        help="Number of panels per split (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace review panels that already exist.",
    )
    return parser.parse_args()


def project_path(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def decode_rle(segmentation: dict[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in segmentation["size"])
    counts = np.asarray(segmentation["counts"], dtype=np.int64)
    if np.any(counts < 0) or int(counts.sum()) != height * width:
        raise ValueError("Invalid uncompressed COCO RLE")
    values = np.arange(counts.size, dtype=np.uint8) % 2
    flat = np.repeat(values, counts)
    return flat.reshape((height, width), order="F").astype(bool)


def choose_images(
    images: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    category_names: dict[int, str],
    sample_count: int,
) -> list[dict[str, Any]]:
    """Choose diverse images, preferring rare-class examples first."""
    if sample_count >= len(images):
        return images

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    priorities = ("v_track", "proton", "alpha", "electron_positron")
    for class_name in priorities:
        candidates = [
            image
            for image in images
            if any(
                category_names[int(annotation["category_id"])] == class_name
                for annotation in annotations_by_image[int(image["id"])]
            )
        ]
        candidates.sort(
            key=lambda image: sum(
                category_names[int(annotation["category_id"])] == class_name
                for annotation in annotations_by_image[int(image["id"])]
            ),
            reverse=True,
        )
        for image in candidates[:3]:
            image_id = int(image["id"])
            if image_id not in selected_ids and len(selected) < sample_count:
                selected.append(image)
                selected_ids.add(image_id)

    remaining = [image for image in images if int(image["id"]) not in selected_ids]
    needed = sample_count - len(selected)
    if needed > 0 and remaining:
        indices = np.linspace(0, len(remaining) - 1, needed, dtype=int)
        for index in indices:
            selected.append(remaining[int(index)])
    return selected


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def draw_instances(
    original: np.ndarray,
    annotations: list[dict[str, Any]],
    category_names: dict[int, str],
) -> tuple[np.ndarray, Counter[str]]:
    overlay = original.copy()
    counts: Counter[str] = Counter()
    for annotation in annotations:
        class_name = category_names[int(annotation["category_id"])]
        colour = CLASS_COLOURS[class_name]
        mask = decode_rle(annotation["segmentation"])
        counts[class_name] += 1
        tinted = np.empty_like(overlay)
        tinted[:] = colour
        overlay[mask] = cv2.addWeighted(
            overlay[mask],
            0.35,
            tinted[mask],
            0.65,
            0,
        )
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, colour, 2)
        x, y, width, height = (int(value) for value in annotation["bbox"])
        cv2.rectangle(overlay, (x, y), (x + width, y + height), colour, 2)

    legend_y = 62
    for class_name in CLASS_COLOURS:
        text = f"{class_name}: {counts[class_name]}"
        colour = CLASS_COLOURS[class_name]
        cv2.putText(
            overlay,
            text,
            (12, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
        legend_y += 25
    return overlay, counts


def resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_contact_sheet(
    panels: list[np.ndarray],
    target_width: int = 1500,
    columns: int = 2,
) -> np.ndarray:
    if not panels:
        raise ValueError("No panels available for contact sheet")
    thumb_width = target_width // columns
    ratio = thumb_width / panels[0].shape[1]
    thumb_height = max(1, int(panels[0].shape[0] * ratio))
    thumbnails = [
        cv2.resize(panel, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        for panel in panels
    ]
    rows = []
    blank = np.zeros_like(thumbnails[0])
    for start in range(0, len(thumbnails), columns):
        row = thumbnails[start : start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def generate_reviews(
    split_root: Path,
    external_root: Path,
    output: Path,
    samples_per_split: int,
    overwrite: bool,
) -> dict[str, Any]:
    if samples_per_split < 1:
        raise ValueError("--samples-per-split must be at least 1")
    image_root = external_root / "images"
    inspection_root = external_root / "inspection"
    if not image_root.is_dir() or not inspection_root.is_dir():
        raise FileNotFoundError(
            "External dataset must contain images/ and inspection/"
        )

    report: dict[str, Any] = {"splits": {}}
    for split_name in SPLITS:
        coco_path = split_root / split_name / "annotations_coco.json"
        if not coco_path.is_file():
            raise FileNotFoundError(f"Missing split annotations: {coco_path}")
        with coco_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        category_names = {
            int(category["id"]): category["name"]
            for category in coco["categories"]
        }
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        chosen = choose_images(
            coco["images"],
            annotations_by_image,
            category_names,
            samples_per_split,
        )

        split_output = output / split_name
        split_output.mkdir(parents=True, exist_ok=True)
        contact_panels: list[np.ndarray] = []
        sampled_counts: Counter[str] = Counter()
        generated_files: list[str] = []

        for sequence, image_record in enumerate(chosen, start=1):
            image_name = PurePath(image_record["file_name"]).name
            original_path = image_root / image_name
            inspection_path = inspection_root / image_name
            original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
            inspection = cv2.imread(str(inspection_path), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError(f"Unable to read original image: {original_path}")
            if inspection is None:
                raise ValueError(
                    f"Unable to read supplied inspection image: {inspection_path}"
                )

            converted, counts = draw_instances(
                original,
                annotations_by_image[int(image_record["id"])],
                category_names,
            )
            sampled_counts.update(counts)
            height, width = original.shape[:2]
            inspection = resize_to(inspection, width, height)
            panel = np.hstack(
                (
                    label_panel(original, "Original"),
                    label_panel(inspection, "Muller supplied annotation"),
                    label_panel(converted, "Converted Mask R-CNN instances"),
                )
            )
            safe_stem = Path(image_name).stem.replace(" ", "_")
            panel_path = split_output / f"{sequence:02d}_{safe_stem}.jpg"
            if panel_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Review already exists; use --overwrite: {panel_path}"
                )
            if not cv2.imwrite(str(panel_path), panel):
                raise OSError(f"Unable to save review panel: {panel_path}")
            contact_panels.append(panel)
            generated_files.append(panel_path.name)

        contact_sheet_path = split_output / "contact_sheet.jpg"
        contact_sheet = make_contact_sheet(contact_panels)
        if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
            raise OSError(f"Unable to save contact sheet: {contact_sheet_path}")
        report["splits"][split_name] = {
            "sample_count": len(chosen),
            "sampled_instances_by_class": {
                name: sampled_counts[name] for name in CLASS_COLOURS
            },
            "contact_sheet": str(contact_sheet_path),
            "panels": generated_files,
        }

    output.mkdir(parents=True, exist_ok=True)
    with (output / "review_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


# pathlib.PurePath accepts Windows and POSIX-looking COCO paths for basename use.
from pathlib import PurePath  # noqa: E402  (kept near its explanatory comment)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    report = generate_reviews(
        split_root=project_path(args.split_root, project_root),
        external_root=project_path(args.external_root, project_root),
        output=project_path(args.output, project_root),
        samples_per_split=args.samples_per_split,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
