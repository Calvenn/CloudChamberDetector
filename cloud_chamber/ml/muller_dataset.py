"""PyTorch dataset for the converted Müller Mask R-CNN annotations.

The shared segmenter is class-agnostic: every alpha, electron/positron, proton
and V-track instance is presented to Mask R-CNN as class ``1`` (particle
track). The original four-class COCO annotations remain unchanged and can be
used later by the independent classification modules.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def decode_uncompressed_rle(segmentation: dict[str, Any]) -> np.ndarray:
    """Decode a JSON-compatible uncompressed COCO RLE mask."""
    height, width = (int(value) for value in segmentation["size"])
    counts = np.asarray(segmentation["counts"], dtype=np.int64)
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("Invalid COCO RLE counts")
    if int(counts.sum()) != height * width:
        raise ValueError("COCO RLE length does not match its image dimensions")
    values = np.arange(counts.size, dtype=np.uint8) % 2
    flat = np.repeat(values, counts)
    return flat.reshape((height, width), order="F")


class MullerTrackDataset:
    """Load enhanced images and instance targets for torchvision Mask R-CNN."""

    def __init__(
        self,
        annotations_path: str | Path,
        *,
        gaussian_kernel: int = 5,
        gaussian_sigma: float = 1.0,
        augment: bool = False,
        class_agnostic: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required. Install requirements-ml.txt first."
            ) from error

        self._torch = torch
        self.annotations_path = Path(annotations_path).resolve()
        if not self.annotations_path.is_file():
            raise FileNotFoundError(
                f"COCO annotations not found: {self.annotations_path}"
            )
        if gaussian_kernel < 1 or gaussian_kernel % 2 == 0:
            raise ValueError("Gaussian kernel must be a positive odd number")

        with self.annotations_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        self.images = sorted(coco["images"], key=lambda item: int(item["id"]))
        self.categories = {
            int(category["id"]): category["name"]
            for category in coco["categories"]
        }
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(
            list
        )
        for annotation in coco["annotations"]:
            self.annotations_by_image[int(annotation["image_id"])].append(
                annotation
            )

        self.gaussian_kernel = int(gaussian_kernel)
        self.gaussian_sigma = float(gaussian_sigma)
        self.augment = bool(augment)
        self.class_agnostic = bool(class_agnostic)

    def __len__(self) -> int:
        return len(self.images)

    def _resolve_image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        return (self.annotations_path.parent / path).resolve()

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        torch = self._torch
        image_record = self.images[index]
        image_id = int(image_record["id"])
        image_path = self._resolve_image_path(str(image_record["file_name"]))
        colour = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if colour is None:
            raise ValueError(f"Unable to read training image: {image_path}")

        # Required shared enhancement: grayscale conversion followed by Gaussian
        # filtering. Repeating the channel preserves compatibility with the
        # ImageNet-pretrained ResNet-50 backbone.
        grayscale = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.GaussianBlur(
            grayscale,
            (self.gaussian_kernel, self.gaussian_kernel),
            self.gaussian_sigma,
        )
        image = torch.from_numpy(enhanced.copy()).float().div(255.0)
        image = image.unsqueeze(0).repeat(3, 1, 1)

        annotations = self.annotations_by_image[image_id]
        boxes = []
        labels = []
        masks = []
        areas = []
        crowds = []
        original_category_ids = []
        for annotation in annotations:
            x, y, width, height = (
                float(value) for value in annotation["bbox"]
            )
            if width <= 0 or height <= 0:
                continue
            mask = decode_uncompressed_rle(annotation["segmentation"])
            if mask.shape != enhanced.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match image "
                    f"{enhanced.shape}: {image_path.name}"
                )
            category_id = int(annotation["category_id"])
            boxes.append([x, y, x + width, y + height])
            labels.append(1 if self.class_agnostic else category_id)
            masks.append(mask)
            areas.append(float(annotation.get("area", int(mask.sum()))))
            crowds.append(int(annotation.get("iscrowd", 0)))
            original_category_ids.append(category_id)

        height, width = enhanced.shape
        if masks:
            masks_tensor = torch.as_tensor(
                np.stack(masks),
                dtype=torch.uint8,
            )
        else:
            masks_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
            "original_category_ids": torch.as_tensor(
                original_category_ids,
                dtype=torch.int64,
            ),
        }

        if self.augment:
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(2,))
                target["masks"] = torch.flip(target["masks"], dims=(2,))
                old_x1 = target["boxes"][:, 0].clone()
                old_x2 = target["boxes"][:, 2].clone()
                target["boxes"][:, 0] = width - old_x2
                target["boxes"][:, 2] = width - old_x1
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(1,))
                target["masks"] = torch.flip(target["masks"], dims=(1,))
                old_y1 = target["boxes"][:, 1].clone()
                old_y2 = target["boxes"][:, 3].clone()
                target["boxes"][:, 1] = height - old_y2
                target["boxes"][:, 3] = height - old_y1

        return image, target


def detection_collate(
    batch: list[tuple[Any, dict[str, Any]]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Keep variable-size detection targets as lists for torchvision."""
    images, targets = zip(*batch)
    return list(images), list(targets)
