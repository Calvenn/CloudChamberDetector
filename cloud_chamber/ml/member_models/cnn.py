"""CNN member workspace for cloud-chamber particle classification.

This implementation follows the same shared feature-extraction contract as the
project's README and other model implementations: a fixed set of contour-based
features is used as the classifier input, and the CNN is used as a small
nonlinear model over that feature matrix rather than a separate patch-based
training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict
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

from cloud_chamber.config import load_config
from cloud_chamber.features import TrackFeatures
from cloud_chamber.ml.contour_dataset import build_feature_csv, load_feature_csv
from cloud_chamber.ml.member_models.mlp import FEATURE_COLUMNS

CLASS_COLOURS = {
    "alpha": (0, 165, 255),
    "electron_positron": (255, 120, 0),
    "proton": (0, 200, 0),
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
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "muller"
SPLIT_NAMES = ("development", "validation", "final_test")


def build_class_mapping() -> dict[str, int]:
    """Return the project label mapping used by the dataset annotations."""
    mapping = {
        "alpha": 0,
        "electron_positron": 1,
        "proton": 2,
    }
    return mapping


def features_to_matrix(features: Iterable[TrackFeatures]) -> np.ndarray:
    """Convert the shared contour-feature objects into the fixed CNN input matrix."""
    rows = []
    for track in features:
        values = asdict(track)
        rows.append([float(values[column]) for column in FEATURE_COLUMNS])
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_COLUMNS))


class FeatureTrackCNN(nn.Module):
    """Compact MLP-style CNN over the fixed contour-feature vector.

    The project README states that all models use the same shared feature columns
    and dataset splits. This model accepts the common 10-feature representation
    from ``cloud_chamber.ml.member_models.mlp.FEATURE_COLUMNS`` and applies a
    small feed-forward network, which preserves the project contract while still
    giving the CNN a distinct implementation.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureDataset(Dataset):
    """Dataset built from the shared contour-feature table for each split.
    
    Features are cached in CSV format. If the cache doesn't exist, it is built
    automatically from the COCO annotations using the shared enhancement and
    feature extraction pipeline, matching the project contract.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        feature_dir: str | Path = DEFAULT_FEATURE_DIR,
        feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
        class_mapping: dict[str, int] | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.feature_dir = Path(feature_dir)
        self.feature_columns = tuple(feature_columns)
        self.class_mapping = class_mapping or build_class_mapping()
        self.samples: list[tuple[np.ndarray, int]] = []
        self._ensure_features_exist()
        self._load_samples()

    def _ensure_features_exist(self) -> None:
        """Build feature CSV from COCO if it doesn't exist."""
        csv_path = self.feature_dir / f"{self.split}.csv"
        if csv_path.exists():
            return
        
        print(f"Building feature table for {self.split} split...")
        annotation_path = self.dataset_root / self.split / "annotations_coco.json"
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"COCO annotations not found: {annotation_path}. "
                f"Please verify the dataset structure."
            )
        
        config = load_config(PROJECT_ROOT / "config.yaml")
        allowed_labels = set(config["classification"]["supported_classes"])
        build_feature_csv(
            annotation_path,
            csv_path,
            config["enhancement"],
            allowed_labels=allowed_labels,
        )
        print(f"Saved features to {csv_path}")

    def _load_samples(self) -> None:
        csv_path = self.feature_dir / f"{self.split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Feature table not found: {csv_path}. "
                f"Run build_feature_csv or the training script with config."
            )
        
        matrix, labels = load_feature_csv(
            csv_path,
            allowed_labels=set(self.class_mapping.keys()),
        )
        for features, label in zip(matrix, labels, strict=True):
            self.samples.append((features.astype(np.float32), self.class_mapping.get(str(label), 0)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        features, label = self.samples[index]
        return torch.from_numpy(features), int(label)


def build_dataloaders(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    batch_size: int = 16,
    split: str = "development",
    validation_split: float = 0.2,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Create development and validation loaders from the dataset feature table."""
    dataset_root = Path(dataset_root)
    feature_dir = Path(feature_dir)
    class_mapping = build_class_mapping()
    full_dataset = FeatureDataset(
        dataset_root, 
        split=split, 
        feature_dir=feature_dir,
        class_mapping=class_mapping
    )
    if len(full_dataset) == 0:
        raise ValueError(f"No labelled feature rows were found under {dataset_root / split}")

    train_size = max(1, int(len(full_dataset) * (1.0 - validation_split)))
    val_size = max(1, len(full_dataset) - train_size)
    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_mapping


def train_cnn(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    split: str = "development",
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
) -> dict[str, Any]:
    """Train the CNN on the shared contour features from the selected split."""
    train_loader, val_loader, class_mapping = build_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        split=split,
        feature_dir=feature_dir,
    )
    model = FeatureTrackCNN(input_dim=len(FEATURE_COLUMNS), num_classes=len(class_mapping))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
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
        with torch.no_grad():
            for features, labels in val_loader:
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total if total else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    return {
        "model": model,
        "class_mapping": class_mapping,
        "history": history,
        "dataset_split": split,
        "dataset_root": str(Path(dataset_root)),
    }


def predict_tracks(model: nn.Module, dataset_root: str | Path, split: str = "development") -> list[dict[str, Any]]:
    """Predict labels for all feature rows in a chosen split and return simple results."""
    class_mapping = build_class_mapping()
    dataset = FeatureDataset(dataset_root, split=split, class_mapping=class_mapping)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            inv_mapping = {value: key for key, value in class_mapping.items()}
            for label, prediction, probability in zip(labels, preds, probabilities, strict=True):
                confidence = float(probability[int(prediction.item())].item())
                results.append(
                    {
                        "true_label": inv_mapping[int(label.item())],
                        "predicted_label": inv_mapping[int(prediction.item())],
                        "confidence": confidence,
                        "probabilities": {
                            inv_mapping[i]: float(probability[i].item())
                            for i in range(len(probability))
                        },
                    }
                )
    return results


def load_model(model_path: str | Path, num_classes: int = 3) -> dict:
    """Load a saved CNN bundle in the same container shape used by the other model pages."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CNN model not found: {path}. Run python -m cloud_chamber.ml.member_models.cnn first."
        )

    payload = torch.load(path, map_location="cpu")
    class_mapping = payload.get("class_mapping") or build_class_mapping()
    feature_columns = payload.get("feature_columns") or FEATURE_COLUMNS
    if tuple(feature_columns) != FEATURE_COLUMNS:
        raise ValueError("Saved CNN uses a different feature-column contract")

    model = FeatureTrackCNN(input_dim=len(FEATURE_COLUMNS), num_classes=num_classes)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "class_mapping": class_mapping,
        "classes": list(class_mapping.keys()),
    }


def predict_tracks(model_bundle: dict, features: Iterable[TrackFeatures]) -> list[dict]:
    """Classify segmented contours using the saved CNN and return probability-rich rows."""
    feature_list = list(features)
    matrix = features_to_matrix(feature_list)
    if matrix.shape[0] == 0:
        return []

    model = model_bundle["model"]
    model.eval()
    tensor = torch.tensor(matrix, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    class_mapping = model_bundle.get("class_mapping") or build_class_mapping()
    inv_mapping = {value: key for key, value in class_mapping.items()}
    elapsed_per_track = 0.0

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
) -> tuple[np.ndarray, list[dict]]:
    """Create the annotated CNN overlay and table rows used by the GUI."""
    feature_list = list(features)
    if len(feature_list) != len(predictions):
        raise ValueError("Feature and prediction counts must be equal")

    overlay = image.copy()
    rows = []
    for track, prediction in zip(feature_list, predictions, strict=True):
        x, y, width, height = track.bounding_box
        confidence = float(prediction["confidence"])
        uncertain = confidence < confidence_threshold
        colour = (0, 255, 255) if uncertain else CLASS_COLOURS.get(
            prediction["predicted_class"], (255, 255, 255)
        )
        cv2.rectangle(overlay, (x, y), (x + width, y + height), colour, 2)
        label = (
            f"T{track.track_id}: {prediction['particle_type']} {confidence:.0%}"
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
                "Status": "Uncertain" if uncertain else "Confident",
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
        "feature_columns": FEATURE_COLUMNS,
        "class_mapping": class_mapping,
        "classes": list(class_mapping.keys()),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN for cloud-chamber particle classification")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--split", choices=SPLIT_NAMES, default="development")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "cnn_classifier.pth")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_cnn(
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        split=args.split,
        feature_dir=args.feature_dir,
    )
    model_path = save_model(result["model"], args.output, metadata={
        "class_mapping": result["class_mapping"],
        "classes": list(result["class_mapping"].keys()),
        "history": result["history"],
        "dataset_split": result["dataset_split"],
    })
    print(f"Saved CNN model: {model_path}")
    print(json.dumps({
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "history": result["history"],
        "class_mapping": result["class_mapping"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
