"""CNN member workspace for cloud-chamber particle classification.

Each labelled COCO annotation becomes a masked, cropped particle-track image
patch.  The CNN learns spatial features from those patches, while sharing the
MLP's four-class labels and leakage-safe dataset splits.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - dependency check for the user flow.
    raise RuntimeError(
        "PyTorch is required for the CNN implementation. Install it with pip install torch."
    ) from exc

from cloud_chamber.core.config import load_config
from cloud_chamber.feature_extraction.contour_features import TrackFeatures
from collections import Counter
from cloud_chamber.ml.contour_dataset import annotation_to_mask

CLASS_COLOURS = {
    # OpenCV uses BGR rather than RGB.
    "alpha": (0, 255, 0),
    "electron_positron": (255, 0, 0),
    "proton": (0, 0, 255),
    "v_track": (255, 0, 255),
}
DISPLAY_NAMES = {
    "alpha": "Alpha",
    "electron_positron": "Electron/Positron",
    "proton": "Proton",
    "v_track": "V-track",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "external_dataset_split"
SPLIT_NAMES = ("development", "validation", "final_test")
PATCH_SIZE = 128


def build_class_mapping() -> dict[str, int]:
    """Return the shared four-class mapping used by every project model."""
    return {
        "alpha": 0,
        "electron_positron": 1,
        "proton": 2,
        "v_track": 3,
    }


class TrackPatchCNN(nn.Module):
    """CNN that learns directly from cropped, masked particle-track images."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"Expected (batch, 1, height, width) image tensor, got {tuple(x.shape)}"
            )
        return self.classifier(self.features(x))


def crop_track_patch(
    image: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Return one square greyscale track patch, optionally masking background."""
    x, y, width, height = (int(value) for value in bounding_box)
    padding = max(8, int(round(max(width, height) * 0.15)))
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(image.shape[1], x + width + padding), min(image.shape[0], y + height + padding)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(f"Invalid track crop: {bounding_box}")
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if mask is not None:
        crop_mask = mask[top:bottom, left:right]
        grey = cv2.bitwise_and(grey, grey, mask=(crop_mask > 0).astype(np.uint8) * 255)
    return cv2.resize(grey, (patch_size, patch_size), interpolation=cv2.INTER_AREA)


def _patch_tensor(patch: np.ndarray, augment: bool = False) -> torch.Tensor:
    """Apply development-only light augmentation and produce a 1xHxW tensor."""
    if augment and np.random.random() < 0.5:
        patch = cv2.flip(patch, 1)
    if augment:
        angle = float(np.random.uniform(-12.0, 12.0))
        matrix = cv2.getRotationMatrix2D((PATCH_SIZE / 2, PATCH_SIZE / 2), angle, 1.0)
        patch = cv2.warpAffine(patch, matrix, (PATCH_SIZE, PATCH_SIZE), borderValue=0)
    return torch.from_numpy(np.ascontiguousarray(patch[None, ...])).float() / 255.0


@dataclass(frozen=True)
class CocoTrackSample:
    """One annotated source object before its image patch is prepared."""

    image_path: Path
    annotation: dict
    label: int


@dataclass(frozen=True)
class PreparedTrackSample:
    """A production-segmented patch prepared before CNN training."""

    patch: np.ndarray
    label: int


class CocoPatchDataset(Dataset):
    """One labelled particle-image crop per COCO instance annotation."""

    def __init__(
        self,
        samples: list[CocoTrackSample | PreparedTrackSample],
        augment: bool = False,
    ):
        self.samples = samples
        self.augment = augment
        # Crops are immutable training inputs.  Building them once avoids
        # re-opening and resizing the same source images on every epoch.
        self.patches = [self._load_patch(sample) for sample in samples]

    @staticmethod
    def _load_patch(sample: CocoTrackSample | PreparedTrackSample) -> np.ndarray:
        """Read one source image and prepare its labelled patch once."""
        if isinstance(sample, PreparedTrackSample):
            return sample.patch
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read training image: {sample.image_path}")
        mask = annotation_to_mask(sample.annotation, image.shape[:2])
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"
            )
        coordinates = cv2.boundingRect((mask > 0).astype(np.uint8))
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            coordinates = tuple(int(round(value)) for value in sample.annotation["bbox"])
        return crop_track_patch(image, coordinates, mask)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return _patch_tensor(self.patches[index], augment=self.augment), self.samples[index].label


def coco_track_samples(
    annotation_path: Path,
    class_mapping: dict[str, int],
    split: str | None = None,
) -> list[CocoTrackSample]:
    """Read supported COCO annotations, optionally filtering primary split."""
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    images = {
        int(item["id"]): item
        for item in payload["images"]
        if split is None or item.get("split") == split
    }
    samples = []
    for annotation in payload["annotations"]:
        image = images.get(int(annotation["image_id"]))
        if image is None:
            continue
        label_name = categories[int(annotation["category_id"])]
        if label_name not in class_mapping:
            continue
        samples.append(
            CocoTrackSample(
                image_path=(annotation_path.parent / image["file_name"]).resolve(),
                annotation=annotation,
                label=class_mapping[label_name],
            )
        )
    return samples


def segmented_track_samples(
    annotation_path: Path,
    class_mapping: dict[str, int],
    config: dict,
    split: str | None = None,
    minimum_label_overlap: float = 0.5,
) -> list[PreparedTrackSample]:
    """Build labelled patches from the exact production segmentation path."""
    from app import _process_tiled_pipeline_image

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    images = {
        int(item["id"]): item
        for item in payload["images"]
        if split is None or item.get("split") == split
    }
    grouped: dict[int, list[dict]] = {}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        label = categories[int(annotation["category_id"])]
        if image_id in images and label in class_mapping:
            grouped.setdefault(image_id, []).append(annotation)

    samples: list[PreparedTrackSample] = []
    for image_id, annotations in grouped.items():
        image_path = (annotation_path.parent / images[image_id]["file_name"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read training image: {image_path}")
        result = _process_tiled_pipeline_image(image, config)
        segmented = result["segmentation"]
        contour_by_box = {
            tuple(int(value) for value in cv2.boundingRect(contour)): contour
            for contour in segmented.contours
        }
        truth = [
            (
                annotation,
                annotation_to_mask(annotation, image.shape[:2]) > 0,
            )
            for annotation in annotations
        ]
        for track in result["features"]:
            contour = contour_by_box.get(track.bounding_box)
            if contour is None:
                continue
            x, y, width, height = track.bounding_box
            local_contour = contour.copy()
            local_contour[:, 0, 0] -= x
            local_contour[:, 0, 1] -= y
            predicted = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(predicted, [local_contour], -1, 255, cv2.FILLED)
            pixels = predicted > 0
            pixel_count = max(int(np.count_nonzero(pixels)), 1)
            overlaps = [
                np.count_nonzero(pixels & mask[y:y + height, x:x + width])
                / pixel_count
                for _, mask in truth
            ]
            if not overlaps or max(overlaps) < minimum_label_overlap:
                continue
            annotation = truth[int(np.argmax(overlaps))][0]
            label_name = categories[int(annotation["category_id"])]
            samples.append(
                PreparedTrackSample(
                    patch=crop_track_patch(
                        result["input_image"],
                        track.bounding_box,
                        segmented.binary_mask,
                    ),
                    label=class_mapping[label_name],
                )
            )
    return samples

def build_dataloaders(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    primary_dataset_root: str | Path = PROJECT_ROOT / "dataset" / "primary_dataset_split",
    batch_size: int = 16,
    random_seed: int = 42,
    config: dict | None = None,
) -> tuple[
    dict[str, DataLoader],
    dict[str, DataLoader],
    dict[str, int],
    dict[str, dict[str, dict]],
]:
    """Build reproducible loaders for ground-truth and segmented patches."""
    dataset_root = Path(dataset_root)
    primary_dataset_root = Path(primary_dataset_root)
    class_mapping = build_class_mapping()
    config = config or load_config(PROJECT_ROOT / "config.yaml")
    primary_annotations = primary_dataset_root / "annotations_coco.json"
    if not primary_annotations.is_file():
        raise FileNotFoundError(f"Primary annotation file not found: {primary_annotations}")

    class_counts = {}
    loaders = {}
    segmented_loaders = {}
    generator = torch.Generator().manual_seed(random_seed)
    inverse_mapping = {value: key for key, value in class_mapping.items()}
    for split in SPLIT_NAMES:
        external_annotations = dataset_root / split / "annotations_coco.json"
        if not external_annotations.is_file():
            raise FileNotFoundError(f"External annotation file not found: {external_annotations}")
        external_samples = coco_track_samples(external_annotations, class_mapping)
        primary_samples = coco_track_samples(primary_annotations, class_mapping, split=split)
        ground_truth_samples = external_samples + primary_samples
        segmented_samples = segmented_track_samples(
            external_annotations, class_mapping, config
        ) + segmented_track_samples(
            primary_annotations, class_mapping, config, split=split
        )
        samples = (
            ground_truth_samples + segmented_samples
            if split == "development"
            else ground_truth_samples
        )
        if not ground_truth_samples:
            raise ValueError(f"No supported image-patch samples found for {split}")
        def count_labels(items) -> dict[str, int]:
            """Count labels using the human-readable class names."""
            counts = Counter(inverse_mapping[item.label] for item in items)
            return dict(sorted(counts.items()))

        class_counts[split] = {
            "ground_truth": count_labels(ground_truth_samples),
            "segmented_augmentation": count_labels(segmented_samples),
            "hybrid_training": count_labels(ground_truth_samples + segmented_samples),
        }
        print(
            f"{split}: {len(ground_truth_samples)} ground-truth patches; "
            f"{len(segmented_samples)} segmented patches"
        )
        dataset = CocoPatchDataset(samples, augment=(split == "development"))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "development"),
            generator=generator if split == "development" else None,
        )
        segmented_loaders[split] = DataLoader(
            CocoPatchDataset(segmented_samples, augment=False),
            batch_size=batch_size,
            shuffle=False,
        )
    return loaders, segmented_loaders, class_mapping, class_counts

def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    class_mapping: dict[str, int],
) -> dict:
    """Measure predictions and class-aware metrics for one data split."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    model.eval()
    all_preds = []
    all_labels = []
    started = perf_counter()
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    elapsed_ms = (perf_counter() - started) * 1000.0
    
    inv_mapping = {v: str(k) for k, v in class_mapping.items()}
    str_labels = [inv_mapping[int(l)] for l in all_labels]
    str_preds = [inv_mapping[int(p)] for p in all_preds]
    class_names = sorted(set(str_labels) | set(str_preds))
    
    return {
        "sample_count": int(len(all_labels)),
        "accuracy": float(accuracy_score(str_labels, str_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(str_labels, str_preds)),
        "macro_f1": float(f1_score(str_labels, str_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(str_labels, str_preds, average="weighted", zero_division=0)),
        "mean_inference_ms_per_track": elapsed_ms / max(len(all_labels), 1),
        "class_names": class_names,
        "confusion_matrix": confusion_matrix(str_labels, str_preds, labels=class_names).tolist(),
        "classification_report": classification_report(
            str_labels, str_preds, labels=class_names, output_dict=True, zero_division=0
        ),
    }

def train_cnn(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    primary_dataset_root: str | Path = PROJECT_ROOT / "dataset" / "primary_dataset_split",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    config: dict | None = None,
) -> dict[str, Any]:
    """Train the CNN and retain the checkpoint selected on validation data."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    loaders, segmented_loaders, class_mapping, class_counts = build_dataloaders(
        dataset_root=dataset_root,
        primary_dataset_root=primary_dataset_root,
        batch_size=batch_size,
        random_seed=random_seed,
        config=config,
    )
    
    train_loader = loaders["development"]
    val_loader = loaders["validation"]
    test_loader = loaders["final_test"]
    
    model = TrackPatchCNN(input_dim=PATCH_SIZE, num_classes=len(class_mapping))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Match the MLP's inverse-frequency balancing policy so every class,
    # including sparse V-tracks, has equal total training influence.
    development_labels = torch.tensor(
        [sample.label for sample in train_loader.dataset.samples], dtype=torch.long
    )
    counts = torch.bincount(development_labels, minlength=len(class_mapping)).float()
    class_weights = counts.sum() / (len(class_mapping) * counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_score: tuple[float, float] | None = None
    best_state = None
    
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * features.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        validation_predictions = []
        validation_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                validation_predictions.extend(predictions.tolist())
                validation_labels.extend(labels.tolist())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total if total else 0.0
        from sklearn.metrics import balanced_accuracy_score, f1_score
        val_macro_f1 = float(
            f1_score(validation_labels, validation_predictions, average="macro", zero_division=0)
        )
        score = (
            val_macro_f1,
            float(balanced_accuracy_score(validation_labels, validation_predictions)),
        )
        if best_score is None or score > best_score:
            best_score = score
            import copy
            best_state = copy.deepcopy(model.state_dict())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_macro_f1"].append(val_macro_f1)

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_metrics = evaluate_loader(model, val_loader, class_mapping)
    final_test_metrics = evaluate_loader(model, test_loader, class_mapping)
    segmented_final_test_metrics = evaluate_loader(
        model, segmented_loaders["final_test"], class_mapping
    )

    return {
        "model": model,
        "class_mapping": class_mapping,
        "history": history,
        "dataset_root": str(Path(dataset_root)),
        "primary_dataset_root": str(Path(primary_dataset_root)),
        "class_counts": class_counts,
        "class_weights": class_weights.tolist(),
        "random_seed": random_seed,
        "validation_metrics": validation_metrics,
        "final_test_metrics": final_test_metrics,
        "segmented_final_test_metrics": segmented_final_test_metrics,
    }

def load_model(model_path: str | Path) -> dict:
    """Load a saved CNN bundle in the same container shape used by the other model pages."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CNN model not found: {path}. Run python -m cloud_chamber.ml.member_models.cnn first."
        )

    payload = torch.load(path, map_location="cpu")
    class_mapping = payload.get("class_mapping") or build_class_mapping()
    metadata = payload.get("metadata", {})
    if metadata.get("input_kind") != "track_image_patch":
        raise ValueError(
            "Saved CNN is not the image-patch model. Retrain it with "
            "python -m cloud_chamber.ml.member_models.cnn."
        )

    model = TrackPatchCNN(
        input_dim=PATCH_SIZE, num_classes=len(class_mapping)
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return {
        "model": model,
        "class_mapping": class_mapping,
        "classes": list(class_mapping.keys()),
        "patch_size": int(metadata.get("patch_size", PATCH_SIZE)),
    }


def predict_tracks(
    model_bundle: dict,
    image: np.ndarray,
    features: Iterable[TrackFeatures],
    binary_mask: np.ndarray,
) -> list[dict]:
    """Classify every segmented track crop in an image or video frame."""
    feature_list = list(features)
    if not feature_list:
        return []

    model = model_bundle["model"]
    model.eval()
    patch_size = int(model_bundle.get("patch_size", PATCH_SIZE))
    patches = [
        _patch_tensor(crop_track_patch(image, track.bounding_box, binary_mask, patch_size))
        for track in feature_list
    ]
    tensor = torch.stack(patches)
    started = perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    class_mapping = model_bundle.get("class_mapping") or build_class_mapping()
    inv_mapping = {value: key for key, value in class_mapping.items()}
    elapsed_per_track = (perf_counter() - started) * 1000.0 / len(feature_list)

    return [
        {
            "track_id": track.track_id,
            "predicted_class": inv_mapping[int(label.item())],
            "particle_type": DISPLAY_NAMES.get(inv_mapping[int(label.item())], inv_mapping[int(label.item())]),
            "confidence": float(probabilities[index, int(label.item())].item()),
            "inference_time_ms": elapsed_per_track,
            "probabilities": {
                inv_mapping[i]: float(probabilities[index, i].item())
                for i in range(probabilities.shape[1])
            },
        }
        for index, (track, label) in enumerate(zip(feature_list, predictions, strict=True))
    ]


def summarise_predictions(predictions: list[dict], confidence_threshold: float = 0.60) -> dict:
    """Summarise a batch of CNN predictions into a compact report."""
    summary = {
        "Alpha": 0,
        "Electron/Positron": 0,
        "Proton": 0,
        "Uncertain": 0,
        "Total": len(predictions),
    }
    for prediction in predictions:
        particle_type = prediction["particle_type"]
        if particle_type in summary:
            summary[particle_type] += 1
        if float(prediction["confidence"]) < confidence_threshold:
            summary["Uncertain"] += 1
    return summary


def build_visual_report(
    image: np.ndarray,
    features: Iterable[TrackFeatures],
    predictions: list[dict],
    confidence_threshold: float = 0.60,
    quality_assessments: list[dict] | None = None,
    original_boxes: list[dict] | None = None,
    instance_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Draw classified CNN tracks and build the matching tabular report."""
    feature_list = list(features)
    if len(feature_list) != len(predictions):
        raise ValueError("Feature and prediction counts must be equal")

    quality_by_track = {
        int(item["track_id"]): item for item in (quality_assessments or [])
    }
    overlay = image.copy()
    component_labels = None
    component_stats = None
    used_components: set[int] = set()
    if instance_mask is not None:
        if instance_mask.shape != image.shape[:2]:
            raise ValueError("Instance mask and report image must have equal size")
        _, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            (instance_mask > 0).astype(np.uint8), connectivity=8
        )
    rows = []
    for index, (track, prediction) in enumerate(
        zip(feature_list, predictions, strict=True)
    ):
        if original_boxes and index < len(original_boxes):
            box = original_boxes[index]
            x, y = int(round(box["x"])), int(round(box["y"]))
            width, height = int(round(box["width"])), int(round(box["height"]))
        else:
            x, y, width, height = track.bounding_box
        confidence = float(prediction["confidence"])
        uncertain = confidence < confidence_threshold
        quality = quality_by_track.get(track.track_id)
        low_quality = quality is not None and int(quality["score"]) < 50
        colour = (160, 160, 160) if low_quality else CLASS_COLOURS.get(
            prediction["predicted_class"], (255, 255, 255)
        )
        component_id = _match_component_to_box(
            component_stats, (x, y, width, height), used_components
        )
        if component_id is not None and component_labels is not None:
            used_components.add(component_id)
            region = component_labels == component_id
            colour_array = np.asarray(colour, dtype=np.float32)
            overlay[region] = np.clip(
                overlay[region].astype(np.float32) * 0.48 + colour_array * 0.52,
                0,
                255,
            ).astype(np.uint8)
            outlines, _ = cv2.findContours(
                region.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, outlines, -1, colour, 2, cv2.LINE_AA)
        elif low_quality:
            _draw_dashed_rectangle(overlay, (x, y), (x + width, y + height), colour)
        else:
            cv2.rectangle(overlay, (x, y), (x + width, y + height), colour, 2)
        status = (
            "Review segmentation" if low_quality else
            "Uncertain" if uncertain else "Accepted"
        )
        quality_text = f" | Q:{quality['grade']}" if quality else ""
        uncertainty_text = " | Uncertain" if uncertain and not low_quality else ""
        label = (
            f"T{track.track_id}: {prediction['particle_type']} {confidence:.0%}"
            f"{uncertainty_text}{quality_text}"
        )
        cv2.putText(
            overlay,
            label,
            (x, max(y - 7, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            1,
            cv2.LINE_AA,
        )
        rows.append(
            {
                "Track": track.track_id,
                "Particle type": prediction["particle_type"],
                "Confidence": confidence,
                "Status": status,
                "Contour quality": quality["grade"] if quality else "Not assessed",
                "Contour quality score": quality["score"] if quality else "",
                "Local contrast": quality["local_contrast"] if quality else "",
                "X": x,
                "Y": y,
                "Width": width,
                "Height": height,
                "Inference time (ms)": prediction["inference_time_ms"],
                **{
                    f"P({DISPLAY_NAMES.get(name, name)})": probability
                    for name, probability in prediction["probabilities"].items()
                },
            }
        )
    return overlay, rows


def _match_component_to_box(
    stats: np.ndarray | None,
    box: tuple[int, int, int, int],
    used_components: set[int],
) -> int | None:
    """Return the unused binary-mask component with the greatest box IoU."""
    if stats is None or len(stats) <= 1:
        return None
    x, y, width, height = box
    best_id = None
    best_iou = 0.0
    for component_id in range(1, len(stats)):
        if component_id in used_components:
            continue
        cx, cy, cw, ch, _ = (int(value) for value in stats[component_id])
        intersection_width = max(0, min(x + width, cx + cw) - max(x, cx))
        intersection_height = max(0, min(y + height, cy + ch) - max(y, cy))
        intersection = intersection_width * intersection_height
        union = width * height + cw * ch - intersection
        iou = intersection / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_id = component_id
    return best_id if best_iou > 0 else None


def _draw_dashed_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    colour: tuple[int, int, int],
    dash_length: int = 8,
) -> None:
    """Draw a low-quality fallback box without hiding the track."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    for start, end in ((x1, x2),):
        for x in range(start, end, dash_length * 2):
            cv2.line(image, (x, y1), (min(x + dash_length, end), y1), colour, 1)
            cv2.line(image, (x, y2), (min(x + dash_length, end), y2), colour, 1)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), colour, 1)
        cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), colour, 1)


def encode_report_png(overlay: np.ndarray) -> bytes:
    """Encode the CNN annotation overlay to PNG bytes."""
    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise OSError("Unable to encode the CNN visual report")
    return encoded.tobytes()


def encode_report_csv(rows: list[dict]) -> bytes:
    """Write rows to CSV using the same style as the other member models."""
    if not rows:
        return b""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def save_model(model: nn.Module, model_path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Save a trained CNN and optional metadata for later inference."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    class_mapping = (metadata or {}).get("class_mapping") or build_class_mapping()
    payload = {
        "model_state": model.state_dict(),
        "class_mapping": class_mapping,
        "classes": list(class_mapping.keys()),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def parse_args() -> argparse.Namespace:
    """Read CNN training paths and hyperparameters from the command line."""

    parser = argparse.ArgumentParser(description="Train a CNN for cloud-chamber particle classification")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="External dataset split root.")
    parser.add_argument("--primary-dataset-root", type=Path, default=PROJECT_ROOT / "dataset" / "primary_dataset_split", help="Primary dataset root containing annotations_coco.json.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "cnn_classifier.pth")
    return parser.parse_args()


def main() -> int:
    """Train the CNN and save its model bundle and evaluation report."""

    args = parse_args()
    config = load_config(PROJECT_ROOT / "config.yaml")
    random_seed = int(config["project"]["random_seed"])
    result = train_cnn(
        dataset_root=args.dataset_root,
        primary_dataset_root=args.primary_dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=random_seed,
        config=config,
    )
    
    model_path = save_model(result["model"], args.output, metadata={
        "class_mapping": result["class_mapping"],
        "classes": list(result["class_mapping"].keys()),
        "history": result["history"],
        "class_weights": result["class_weights"],
        "random_seed": result["random_seed"],
        "input_kind": "track_image_patch",
        "patch_size": PATCH_SIZE,
    })
    
    report = {
        "method": "TrackPatchCNN",
        "feature_source": (
            "hybrid ground-truth masked patches and production-segmented "
            "patches; validation and final-test use ground-truth patches"
        ),
        "selection_metric": "validation macro F1; balanced accuracy tie-breaker",
        "input": "masked greyscale particle-track patches resized to 128x128",
        "class_counts": result["class_counts"],
        "classes": list(result["class_mapping"].keys()),
        "class_mapping": result["class_mapping"],
        "balancing": "inverse-frequency class weights in CrossEntropyLoss",
        "random_seed": result["random_seed"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patch_size": PATCH_SIZE,
        },
        "selected_epoch": int(np.argmax(result["history"]["val_macro_f1"])) + 1,
        "history": result["history"],
        "validation": result["validation_metrics"],
        "final_test": result["final_test_metrics"],
        "segmented_final_test": result["segmented_final_test_metrics"],
        "model_path": str(model_path),
    }
    
    report_path = args.output.with_suffix("").with_name("cnn_training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    print(f"Saved CNN model: {model_path}")
    print(f"Saved evidence report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
