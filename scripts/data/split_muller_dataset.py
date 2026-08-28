"""Create leakage-safe Müller development, validation and final-test splits.

Frames are grouped by their original recording name. Entire recordings are
assigned to one split so visually similar neighbouring frames cannot appear in
both training and evaluation data.

The script does not copy or modify images. Each split's COCO ``file_name``
points back to ``dataset/external_dataset/images``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


CLASS_NAMES = ("alpha", "electron_positron", "proton", "v_track")

# Selected after auditing all six recordings. This gives every split examples
# of every class while keeping each complete recording in one split.
DEFAULT_SPLIT_GROUPS = {
    "development": {
        "2021-03-06_17-54-05",
        "2021-04-18_09-34-57",
        "4m2s",
    },
    "validation": {
        "2021-04-17",
    },
    "final_test": {
        "2h24m19s",
        "43m1s",
    },
}

KNOWN_GROUP_PREFIXES = tuple(
    sorted(
        set().union(*DEFAULT_SPLIT_GROUPS.values()),
        key=len,
        reverse=True,
    )
)


def parse_args() -> argparse.Namespace:
    """Read the converted annotation source and split output location."""

    parser = argparse.ArgumentParser(
        description="Split converted Müller COCO data by recording source."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("dataset/external_dataset_coco/annotations_coco.json"),
        help="Master converted COCO JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/external_dataset_split"),
        help="Output split folder (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of generated split metadata.",
    )
    return parser.parse_args()


def project_path(path: Path, project_root: Path) -> Path:
    """Resolve a relative path from the repository root."""

    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def recording_group(file_name: str) -> str:
    """Return the recording identifier used to prevent split leakage."""

    stem = PurePosixPath(file_name).stem
    for prefix in KNOWN_GROUP_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    raise ValueError(f"Unrecognised Müller recording name: {file_name}")


def group_to_split() -> dict[str, str]:
    """Map every configured recording group to exactly one partition."""

    mapping: dict[str, str] = {}
    for split_name, groups in DEFAULT_SPLIT_GROUPS.items():
        for group in groups:
            if group in mapping:
                raise AssertionError(f"Recording assigned twice: {group}")
            mapping[group] = split_name
    return mapping


def output_file_name(original_name: str) -> str:
    """Build the portable image path stored in the split COCO file."""

    image_name = PurePosixPath(original_name).name
    return f"../../external_dataset/images/{image_name}"


def prepare_outputs(output: Path, overwrite: bool) -> dict[str, dict[str, Path]]:
    """Create split destinations while protecting existing outputs."""

    paths: dict[str, dict[str, Path]] = {}
    for split_name in DEFAULT_SPLIT_GROUPS:
        split_dir = output / split_name
        split_paths = {
            "coco": split_dir / "annotations_coco.json",
            "manifest": split_dir / "manifest.csv",
        }
        existing = [path for path in split_paths.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"{split_name} outputs already exist; use --overwrite"
            )
        split_dir.mkdir(parents=True, exist_ok=True)
        paths[split_name] = split_paths
    summary = output / "summary.json"
    combined_manifest = output / "split_manifest.csv"
    if not overwrite and (summary.exists() or combined_manifest.exists()):
        raise FileExistsError("Split outputs already exist; use --overwrite")
    return paths


def split_dataset(
    annotations_path: Path,
    output: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Create COCO partitions without placing one recording in multiple splits."""

    if not annotations_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_path}")
    with annotations_path.open("r", encoding="utf-8") as handle:
        master = json.load(handle)

    images = master.get("images")
    annotations = master.get("annotations")
    categories = master.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Invalid COCO file: missing images or annotations list")
    if not isinstance(categories, list):
        raise ValueError("Invalid COCO file: missing categories list")

    category_names = {int(item["id"]): item["name"] for item in categories}
    missing_classes = set(CLASS_NAMES) - set(category_names.values())
    if missing_classes:
        raise ValueError(f"COCO categories are missing: {sorted(missing_classes)}")

    split_paths = prepare_outputs(output, overwrite)
    source_assignment = group_to_split()
    image_split: dict[int, str] = {}
    image_group: dict[int, str] = {}
    split_images: dict[str, list[dict[str, Any]]] = {
        name: [] for name in DEFAULT_SPLIT_GROUPS
    }

    for image in images:
        image_id = int(image["id"])
        group = recording_group(str(image["file_name"]))
        split_name = source_assignment[group]
        image_split[image_id] = split_name
        image_group[image_id] = group
        copied_image = dict(image)
        copied_image["file_name"] = output_file_name(str(image["file_name"]))
        copied_image["recording_group"] = group
        split_images[split_name].append(copied_image)

    split_annotations: dict[str, list[dict[str, Any]]] = {
        name: [] for name in DEFAULT_SPLIT_GROUPS
    }
    per_image_counts: dict[int, Counter[str]] = {
        image_id: Counter() for image_id in image_split
    }
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_split:
            raise ValueError(f"Annotation refers to missing image ID {image_id}")
        split_name = image_split[image_id]
        split_annotations[split_name].append(annotation)
        class_name = category_names[int(annotation["category_id"])]
        per_image_counts[image_id][class_name] += 1

    output_paths = prepare_outputs(output, True)
    all_manifest_rows: list[dict[str, Any]] = []
    summary_splits: dict[str, Any] = {}
    now = datetime.now(timezone.utc).isoformat()

    for split_name in DEFAULT_SPLIT_GROUPS:
        current_images = split_images[split_name]
        current_annotations = split_annotations[split_name]
        class_counts = Counter(
            category_names[int(annotation["category_id"])]
            for annotation in current_annotations
        )
        absent_classes = set(CLASS_NAMES) - {
            name for name, count in class_counts.items() if count > 0
        }
        if absent_classes:
            raise ValueError(
                f"{split_name} has no instances of {sorted(absent_classes)}"
            )

        split_coco = {
            "info": {
                **master.get("info", {}),
                "description": (
                    "Müller connected-component instances: "
                    f"{split_name} source-separated split"
                ),
                "date_created": now,
                "split": split_name,
                "leakage_control": "complete recording groups",
            },
            "licenses": master.get("licenses", []),
            "images": current_images,
            "annotations": current_annotations,
            "categories": categories,
        }
        with output_paths[split_name]["coco"].open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(split_coco, handle, indent=2)

        manifest_rows = []
        for image in current_images:
            image_id = int(image["id"])
            counts = per_image_counts[image_id]
            row = {
                "split": split_name,
                "recording_group": image_group[image_id],
                "image_id": image_id,
                "file_name": image["file_name"],
                **{name: counts[name] for name in CLASS_NAMES},
                "total_instances": sum(counts.values()),
            }
            manifest_rows.append(row)
            all_manifest_rows.append(row)

        with output_paths[split_name]["manifest"].open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

        summary_splits[split_name] = {
            "recording_groups": sorted(DEFAULT_SPLIT_GROUPS[split_name]),
            "image_count": len(current_images),
            "image_percentage": round(100 * len(current_images) / len(images), 2),
            "annotation_count": len(current_annotations),
            "annotations_by_class": {
                name: class_counts[name] for name in CLASS_NAMES
            },
        }

    seen_groups: dict[str, str] = {}
    for split_name, groups in DEFAULT_SPLIT_GROUPS.items():
        for group in groups:
            if group in seen_groups:
                raise AssertionError(
                    f"Leakage: {group} occurs in {seen_groups[group]} "
                    f"and {split_name}"
                )
            seen_groups[group] = split_name

    summary = {
        "created_at": now,
        "source_annotations": str(annotations_path),
        "output": str(output),
        "strategy": "recording-group split",
        "image_copying": False,
        "image_root": str(annotations_path.parent.parent / "external_dataset/images"),
        "total_images": len(images),
        "total_annotations": len(annotations),
        "splits": summary_splits,
        "leakage_audit": {
            "recording_group_overlap": False,
            "image_id_overlap": False,
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_manifest_rows[0]))
        writer.writeheader()
        writer.writerows(all_manifest_rows)
    return summary


def main() -> int:
    """Create leakage-safe Müller development, validation and test splits."""

    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    annotations = project_path(args.annotations, project_root)
    output = project_path(args.output, project_root)
    try:
        summary = split_dataset(annotations, output, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Split failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
