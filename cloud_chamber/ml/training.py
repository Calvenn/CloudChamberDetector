"""Mask R-CNN construction and training helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def build_track_mask_rcnn(
    *,
    pretrained: bool = True,
    min_size: int = 640,
    max_size: int = 1024,
    trainable_backbone_layers: int = 3,
) -> Any:
    """Build a class-agnostic Mask R-CNN with an FCN mask head."""
    try:
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except ImportError as error:
        raise RuntimeError(
            "PyTorch and torchvision are required for Mask R-CNN training."
        ) from error

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None if pretrained else None,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=(
            trainable_backbone_layers if pretrained else 5
        ),
    )

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_features,
        256,
        2,
    )
    return model


def move_targets_to_device(
    targets: list[dict[str, Any]],
    device: Any,
) -> list[dict[str, Any]]:
    return [
        {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in target.items()
            if key != "original_category_ids"
        }
        for target in targets
    ]


def mean_losses(
    totals: dict[str, float],
    batches: int,
) -> dict[str, float]:
    if batches == 0:
        raise ValueError("Data loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def train_one_epoch(
    model: Any,
    loader: Iterable[Any],
    optimizer: Any,
    device: Any,
    scaler: Any,
    *,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    import torch

    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            losses = model(images, targets)
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite training loss: {total_loss}")
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["total"] += float(total_loss.detach().cpu())
        for name, loss in losses.items():
            totals[name] += float(loss.detach().cpu())
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return mean_losses(totals, batches)


def validation_losses(
    model: Any,
    loader: Iterable[Any],
    device: Any,
    *,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Compute detection losses without gradient updates."""
    import torch

    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = move_targets_to_device(targets, device)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                losses = model(images, targets)
                total_loss = sum(losses.values())
            totals["total"] += float(total_loss.detach().cpu())
            for name, loss in losses.items():
                totals[name] += float(loss.detach().cpu())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    return mean_losses(totals, batches)
