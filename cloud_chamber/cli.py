"""Command-line tools for acquisition and the shared processing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloud_chamber.acquisition import dataset_summary, export_video_frames
from cloud_chamber.config import load_config
from cloud_chamber.pipeline import analyse_image, save_pipeline_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud Chamber Mode A runner")
    commands = parser.add_subparsers(dest="command", required=True)
    analyse = commands.add_parser("analyse", help="Run the shared pipeline")
    analyse.add_argument("image", type=Path)
    analyse.add_argument("--config", type=Path, default=Path("config.yaml"))
    scan = commands.add_parser("scan-dataset", help="Count dataset media")
    scan.add_argument("--root", type=Path)
    scan.add_argument("--config", type=Path, default=Path("config.yaml"))
    extract = commands.add_parser("extract-frames", help="Extract video frames")
    extract.add_argument("video", type=Path)
    extract.add_argument("--output", type=Path)
    extract.add_argument("--interval", type=int)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "scan-dataset":
        root = args.root or Path(config["paths"]["dataset_root"])
        print(json.dumps(dataset_summary(root, config["acquisition"]), indent=2))
        return 0
    if args.command == "extract-frames":
        frames = export_video_frames(
            args.video,
            args.output or Path(config["paths"]["extracted_frames"]),
            frame_interval=args.interval
            or int(config["acquisition"]["frame_interval"]),
            jpeg_quality=int(config["acquisition"]["jpeg_quality"]),
            overwrite=args.overwrite,
        )
        print(f"Saved or found {len(frames)} frames")
        return 0

    result = analyse_image(args.image, config)
    output = save_pipeline_images(result, config["paths"]["results"])
    print(
        json.dumps(
            {
                "sample_id": result.sample_id,
                "track_candidates": len(result.features),
                "processing_time_ms": result.segmentation.processing_time_ms,
                "output_folder": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
