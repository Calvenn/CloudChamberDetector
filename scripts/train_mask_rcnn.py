"""Train the shared class-agnostic FCN mask head within Mask R-CNN."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--development-annotations",
        type=Path,
        default=Path(
            "dataset/external_dataset_split/development/annotations_coco.json"
        ),
    )
    parser.add_argument(
        "--validation-annotations",
        type=Path,
        default=Path(
            "dataset/external_dataset_split/validation/annotations_coco.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialise the backbone from COCO weights.",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable horizontal and vertical training flips.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit batches per epoch for smoke testing.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Limit validation batches for smoke testing.",
    )
    return parser.parse_args()


def resolve(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def set_reproducible_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> Any:
    from torch.utils.data import DataLoader

    from cloud_chamber.ml.muller_dataset import detection_collate

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=detection_collate,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def main() -> int:
    import torch

    from cloud_chamber.ml.muller_dataset import MullerTrackDataset
    from cloud_chamber.ml.training import (
        build_track_mask_rcnn,
        train_one_epoch,
        validation_losses,
    )

    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    if args.patience < 1:
        raise ValueError("Patience must be positive")

    project_root = PROJECT_ROOT
    development_path = resolve(args.development_annotations, project_root)
    validation_path = resolve(args.validation_annotations, project_root)
    output = resolve(args.output, project_root)
    output.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(args.seed)

    development = MullerTrackDataset(
        development_path,
        gaussian_kernel=5,
        gaussian_sigma=1.0,
        augment=not args.no_augmentation,
        class_agnostic=True,
    )
    validation = MullerTrackDataset(
        validation_path,
        gaussian_kernel=5,
        gaussian_sigma=1.0,
        augment=False,
        class_agnostic=True,
    )
    train_loader = make_loader(
        development,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    model = build_track_mask_rcnn(
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )

    best_path = output / "mask_rcnn_track_segmenter.pt"
    checkpoint_path = output / "mask_rcnn_last_checkpoint.pt"
    history_path = output / "mask_rcnn_training_history.json"
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    started = time.perf_counter()

    print(
        json.dumps(
            {
                "device": str(device),
                "development_images": len(development),
                "validation_images": len(validation),
                "class_agnostic_classes": 2,
                "enhancement": "grayscale + Gaussian(5, sigma=1.0)",
                "amp": amp_enabled,
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            amp_enabled=amp_enabled,
            max_batches=args.max_train_batches,
        )
        validation_loss = validation_losses(
            model,
            validation_loader,
            device,
            amp_enabled=amp_enabled,
            max_batches=args.max_validation_batches,
        )
        scheduler.step(validation_loss["total"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "duration_seconds": round(time.perf_counter() - epoch_started, 2),
        }
        history.append(record)

        improved = validation_loss["total"] < best_validation_loss
        if improved:
            best_validation_loss = validation_loss["total"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_loss": best_validation_loss,
                "history": history,
                "class_agnostic": True,
                "num_classes": 2,
            },
            checkpoint_path,
        )
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "device": str(device),
                    "development_images": len(development),
                    "validation_images": len(validation),
                    "best_validation_loss": best_validation_loss,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "epochs": history,
                },
                handle,
                indent=2,
            )
        print(json.dumps(record, indent=2), flush=True)

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {epoch} epochs; "
                f"best validation loss={best_validation_loss:.6f}",
                flush=True,
            )
            break

    print(f"Best model: {best_path}", flush=True)
    print(f"Training history: {history_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
