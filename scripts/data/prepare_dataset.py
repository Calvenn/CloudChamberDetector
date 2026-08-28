"""Create a leakage-safe raw-image split directly from dataset.zip."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
EXCLUDED_FOLDERS = {"annotated", "bw"}
SESSION_SPLIT = {
    "VID_20220609_215401": "development",
    "VID_20220609_220549": "development",
    "VID_20220609_220753": "validation",
    "VID_20220609_221024": "final_test",
}


def prepare_dataset(
    archive_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Extract usable primary images into recording-session-safe partitions."""

    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | int]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for information in archive.infolist():
            source = PurePosixPath(information.filename)
            if not _is_usable_raw_image(source):
                continue

            session = source.parts[1]
            split = SESSION_SPLIT.get(session)
            if split is None:
                continue

            destination = output_root / split / session / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with archive.open(information) as source_handle:
                    with destination.open("wb") as destination_handle:
                        shutil.copyfileobj(source_handle, destination_handle)

            records.append(
                {
                    "split": split,
                    "session": session,
                    "filename": source.name,
                    "relative_path": destination.relative_to(output_root).as_posix(),
                    "size_bytes": information.file_size,
                }
            )

    _validate_records(records)
    _write_manifest(records, output_root / "manifest.csv")

    split_counts = Counter(str(record["split"]) for record in records)
    session_counts = Counter(str(record["session"]) for record in records)
    summary: dict[str, object] = {
        "archive": str(archive_path.resolve()),
        "output_root": str(output_root.resolve()),
        "excluded_folders": sorted(EXCLUDED_FOLDERS),
        "total_raw_images": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "session_counts": dict(sorted(session_counts.items())),
        "split_percentages": {
            name: round(count / len(records) * 100.0, 2)
            for name, count in sorted(split_counts.items())
        },
        "session_split": SESSION_SPLIT,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _is_usable_raw_image(path: PurePosixPath) -> bool:
    if len(path.parts) < 3:
        return False
    if path.parts[0] != "01 - Annotated Capture":
        return False
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    parent_names = {part.casefold() for part in path.parts[2:-1]}
    return parent_names.isdisjoint(EXCLUDED_FOLDERS)


def _validate_records(records: list[dict[str, str | int]]) -> None:
    expected_counts = {
        "development": 34,
        "validation": 19,
        "final_test": 24,
    }
    actual_counts = Counter(str(record["split"]) for record in records)
    if dict(actual_counts) != expected_counts:
        raise ValueError(
            f"Unexpected split counts: {dict(actual_counts)}; "
            f"expected {expected_counts}"
        )

    filenames = [str(record["filename"]) for record in records]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate raw-image filenames were found")

    session_splits: dict[str, set[str]] = {}
    for record in records:
        session = str(record["session"])
        session_splits.setdefault(session, set()).add(str(record["split"]))
    leaking = {
        session: splits
        for session, splits in session_splits.items()
        if len(splits) != 1
    }
    if leaking:
        raise ValueError(f"Recording-session leakage detected: {leaking}")


def _write_manifest(
    records: list[dict[str, str | int]],
    output_path: Path,
) -> None:
    fields = [
        "split",
        "session",
        "filename",
        "relative_path",
        "size_bytes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only raw cloud-chamber images and create session-safe "
            "development, validation and final-test folders."
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
        default=Path(__file__).resolve().parents[2] / "dataset_clean",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            prepare_dataset(arguments.archive, arguments.output),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
