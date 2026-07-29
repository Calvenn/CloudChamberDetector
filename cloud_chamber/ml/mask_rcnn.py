"""Inference adapter for the selected FCN mask head within Mask R-CNN."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cloud_chamber.ml.contracts import SegmentationResult, SegmentedInstance


class MaskRCNNSegmenter:
    """Class-agnostic track segmenter loaded from trained Torch weights."""

    name = "FCN Mask Head within Mask R-CNN"

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.confidence_threshold = float(confidence_threshold)
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                "Mask R-CNN weights are not available yet: "
                f"{self.weights_path}"
            )
        try:
            import torch
            from torchvision.models.detection import maskrcnn_resnet50_fpn
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-ml.txt before running Mask R-CNN."
            ) from error

        self._torch = torch
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
            min_size=640,
            max_size=1024,
        )
        state = torch.load(
            self.weights_path,
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(state)
        self._model.to(self._device).eval()

    def segment(self, enhanced_image: np.ndarray) -> SegmentationResult:
        if enhanced_image.ndim != 2 or enhanced_image.dtype != np.uint8:
            raise ValueError("Mask R-CNN input must be greyscale uint8")

        tensor = self._torch.from_numpy(enhanced_image).float().div(255.0)
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1).to(self._device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self._model([tensor])[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        instances: list[SegmentedInstance] = []
        for score, box, mask in zip(
            output["scores"],
            output["boxes"],
            output["masks"],
        ):
            confidence = float(score.detach().cpu())
            if confidence < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = (
                int(round(value))
                for value in box.detach().cpu().tolist()
            )
            binary_mask = (
                mask[0].detach().cpu().numpy() >= 0.5
            ).astype(np.uint8) * 255
            instances.append(
                SegmentedInstance(
                    instance_id=len(instances) + 1,
                    bounding_box=(x1, y1, x2 - x1, y2 - y1),
                    mask=binary_mask,
                    confidence=confidence,
                )
            )
        return SegmentationResult(self.name, instances, elapsed_ms)
