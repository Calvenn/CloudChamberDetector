"""Build the canonical contour-feature cache used by classical models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.core.config import load_config
from cloud_chamber.ml.shared_features import DEFAULT_FEATURE_DIR, ensure_shared_features


def parse_args() -> argparse.Namespace:
    """Read dataset and output locations for shared feature generation."""

    parser = argparse.ArgumentParser(description="Build shared contour features")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "external_dataset_split",
    )
    parser.add_argument(
        "--primary-split-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "primary_dataset_split",
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-segmented", action="store_true")
    parser.add_argument("--primary-roi-profile", default="primary_full_chamber")
    return parser.parse_args()


def main() -> int:
    """Generate the common contour-feature tables used by classifiers."""

    args = parse_args()
    config = load_config(args.config)
    ensure_shared_features(
        config=config,
        external_split_root=args.split_root,
        primary_split_root=args.primary_split_root,
        feature_dir=args.feature_dir,
        allowed_labels=set(config["classification"]["supported_classes"]),
        rebuild=args.rebuild,
        rebuild_segmented=args.rebuild_segmented,
        primary_roi_profile=args.primary_roi_profile,
    )
    print(f"Shared feature cache ready: {args.feature_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
