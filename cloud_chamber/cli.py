"""Command-line entry point for shared pipeline verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloud_chamber.acquisition import (
    dataset_summary,
    discover_videos,
    export_video_frames,
)
from cloud_chamber.annotations import load_ground_truth_mask
from cloud_chamber.config import load_config
from cloud_chamber.detectors import list_detectors
from cloud_chamber.evaluation import evaluate_mask, export_evaluations_csv
from cloud_chamber.pipeline import analyse_image, save_pipeline_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cloud Chamber Mode A comparison runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser("analyse", help="Run one detector")
    analyse.add_argument("image", type=Path)
    analyse.add_argument(
        "--detector",
        default="otsu_baseline",
        choices=list_detectors(),
    )
    analyse.add_argument("--ground-truth", type=Path)
    analyse.add_argument("--config", type=Path, default=Path("config.yaml"))

    compare = subparsers.add_parser(
        "compare",
        help="Run every registered detector",
    )
    compare.add_argument("image", type=Path)
    compare.add_argument("--ground-truth", type=Path)
    compare.add_argument("--config", type=Path, default=Path("config.yaml"))

    scan = subparsers.add_parser(
        "scan-dataset",
        help="Count suitable raw images and videos in the extracted dataset",
    )
    scan.add_argument("--root", type=Path)
    scan.add_argument("--config", type=Path, default=Path("config.yaml"))

    extract = subparsers.add_parser(
        "extract-frames",
        help="Capture regularly sampled JPG frames from one video",
    )
    extract.add_argument("video", type=Path)
    extract.add_argument("--output", type=Path)
    extract.add_argument("--interval", type=int)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--config", type=Path, default=Path("config.yaml"))

    extract_all = subparsers.add_parser(
        "extract-all-frames",
        help="Capture JPG frames from every dataset video",
    )
    extract_all.add_argument("--root", type=Path)
    extract_all.add_argument("--output", type=Path)
    extract_all.add_argument("--interval", type=int)
    extract_all.add_argument("--overwrite", action="store_true")
    extract_all.add_argument("--config", type=Path, default=Path("config.yaml"))

    subparsers.add_parser("list", help="List registered detectors")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list":
        print("\n".join(list_detectors()))
        return 0

    config = load_config(args.config)
    if args.command == "scan-dataset":
        root = args.root or Path(config["paths"]["dataset_root"])
        print(
            json.dumps(
                dataset_summary(root, config["acquisition"]),
                indent=2,
            )
        )
        return 0

    if args.command in {"extract-frames", "extract-all-frames"}:
        output = args.output or Path(config["paths"]["extracted_frames"])
        interval = args.interval or int(config["acquisition"]["frame_interval"])
        quality = int(config["acquisition"]["jpeg_quality"])
        videos = (
            [args.video]
            if args.command == "extract-frames"
            else discover_videos(
                args.root or Path(config["paths"]["dataset_root"]),
                extensions=set(
                    config["acquisition"]["allowed_video_extensions"]
                ),
            )
        )
        total = 0
        for video in videos:
            frames = export_video_frames(
                video,
                output,
                frame_interval=interval,
                jpeg_quality=quality,
                overwrite=args.overwrite,
            )
            total += len(frames)
            print(f"{video}: {len(frames)} frames")
        print(f"Saved or found {total} frame images in {output}")
        return 0

    names = (
        [args.detector]
        if args.command == "analyse"
        else list_detectors()
    )
    evaluations = []

    for detector_name in names:
        result = analyse_image(args.image, detector_name, config)
        output_folder = save_pipeline_images(
            result,
            config["paths"]["results"],
        )
        summary = {
            "sample_id": result.sample_id,
            "method_name": result.detection.method_name,
            "processing_time_ms": result.detection.processing_time_ms,
            "track_candidates": len(result.features),
            "output_folder": str(output_folder),
        }

        if args.ground_truth:
            ground_truth = load_ground_truth_mask(
                args.ground_truth,
                expected_shape=result.detection.binary_mask.shape,
            )
            evaluation = evaluate_mask(result.detection, ground_truth)
            evaluations.append(evaluation)
            summary["evaluation"] = evaluation.to_dict()

        print(json.dumps(summary, indent=2))

    if evaluations:
        export_evaluations_csv(
            evaluations,
            Path(config["paths"]["results"]) / "comparison.csv",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
