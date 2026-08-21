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
from collections import Counter
from cloud_chamber.ml.contour_dataset import build_feature_csv, load_feature_csv, build_segmented_feature_csv
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



def primary_split_annotations(source_path: Path, split: str) -> Path:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    image_ids = {
        int(image["id"])
        for image in payload.get("images", [])
        if image.get("split") == split
    }
    if not image_ids:
        raise ValueError(f"Primary dataset contains no images for split={split!r}")
    filtered = {
        **payload,
        "images": [img for img in payload.get("images", []) if int(img["id"]) in image_ids],
        "annotations": [ann for ann in payload.get("annotations", []) if int(ann["image_id"]) in image_ids],
    }
    output_path = source_path.with_name(f"{split}_annotations_coco.json")
    output_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    return output_path

def load_or_build_features(
    annotation_path: Path,
    feature_path: Path,
    config: dict,
    allowed_labels: set[str],
    rebuild: bool,
    roi_profile_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if rebuild or not feature_path.exists():
        print(f"Building labelled contour features: {feature_path.name}")
        build_segmented_feature_csv(
            annotation_path,
            feature_path,
            config,
            allowed_labels=allowed_labels,
            roi_profile_name=roi_profile_name,
        )
    return load_feature_csv(feature_path, allowed_labels)

def build_dataloaders(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    primary_dataset_root: str | Path = PROJECT_ROOT / "dataset" / "primary_dataset_split",
    batch_size: int = 16,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    rebuild_features: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, int], dict[str, dict[str, dict]]]:
    dataset_root = Path(dataset_root)
    primary_dataset_root = Path(primary_dataset_root)
    feature_dir = Path(feature_dir)
    class_mapping = build_class_mapping()
    allowed_labels = set(class_mapping.keys())
    config = load_config(PROJECT_ROOT / "config.yaml")

    primary_annotations = primary_dataset_root / "annotations_coco.json"
    if not primary_annotations.is_file():
        raise FileNotFoundError(f"Primary annotation file not found: {primary_annotations}")

    split_tables = {}
    class_counts = {}
    loaders = {}

    for split in SPLIT_NAMES:
        external_annotations = dataset_root / split / "annotations_coco.json"
        if not external_annotations.is_file():
            raise FileNotFoundError(f"External annotation file not found: {external_annotations}")
            
        primary_filtered = primary_split_annotations(primary_annotations, split)

        external_x, external_y = load_or_build_features(
            external_annotations, feature_dir / f"{split}.csv", config, allowed_labels, rebuild_features, "external_muller"
        )
        primary_x, primary_y = load_or_build_features(
            primary_filtered, feature_dir / f"primary_{split}.csv", config, allowed_labels, rebuild_features, "primary_full_chamber"
        )
        
        matrix = np.concatenate((external_x, primary_x), axis=0)
        labels = np.concatenate((external_y, primary_y), axis=0)
        split_tables[split] = (matrix, labels)
        
        class_counts[split] = {
            "external": dict(sorted(Counter(external_y).items())),
            "primary": dict(sorted(Counter(primary_y).items())),
            "combined": dict(sorted(Counter(labels).items())),
        }
        print(f"{split}: {len(labels)} combined tracks (external={len(external_y)}, primary={len(primary_y)})")
        
        tensors_x = torch.tensor(matrix, dtype=torch.float32)
        int_labels = np.array([class_mapping.get(str(lbl), 0) for lbl in labels], dtype=np.int64)
        tensors_y = torch.tensor(int_labels, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(tensors_x, tensors_y)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "development"))

    return loaders, class_mapping, class_counts

def evaluate_loader(model: nn.Module, loader: DataLoader, class_mapping: dict[str, int]) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
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
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    rebuild_features: bool = False,
) -> dict[str, Any]:
    loaders, class_mapping, class_counts = build_dataloaders(
        dataset_root=dataset_root,
        primary_dataset_root=primary_dataset_root,
        batch_size=batch_size,
        feature_dir=feature_dir,
        rebuild_features=rebuild_features,
    )
    
    train_loader = loaders["development"]
    val_loader = loaders["validation"]
    test_loader = loaders["final_test"]
    
    model = FeatureTrackCNN(input_dim=len(FEATURE_COLUMNS), num_classes=len(class_mapping))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float('inf')
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            import copy
            best_state = copy.deepcopy(model.state_dict())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_metrics = evaluate_loader(model, val_loader, class_mapping)
    final_test_metrics = evaluate_loader(model, test_loader, class_mapping)

    return {
        "model": model,
        "class_mapping": class_mapping,
        "history": history,
        "dataset_root": str(Path(dataset_root)),
        "primary_dataset_root": str(Path(primary_dataset_root)),
        "class_counts": class_counts,
        "validation_metrics": validation_metrics,
        "final_test_metrics": final_test_metrics,
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
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="External dataset split root.")
    parser.add_argument("--primary-dataset-root", type=Path, default=PROJECT_ROOT / "dataset" / "primary_dataset_split", help="Primary dataset root containing annotations_coco.json.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "cnn_classifier.pth")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_cnn(
        dataset_root=args.dataset_root,
        primary_dataset_root=args.primary_dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        feature_dir=args.feature_dir,
        rebuild_features=args.rebuild_features,
    )
    
    model_path = save_model(result["model"], args.output, metadata={
        "class_mapping": result["class_mapping"],
        "classes": list(result["class_mapping"].keys()),
        "history": result["history"],
    })
    
    report = {
        "method": "FeatureTrackCNN",
        "feature_source": (
            "combined primary and external automatic segmentation contours "
            "labelled by COCO-mask overlap"
        ),
        "selection_metric": "validation loss",
        "feature_columns": FEATURE_COLUMNS,
        "class_counts": result["class_counts"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "validation": result["validation_metrics"],
        "final_test": result["final_test_metrics"],
        "model_path": str(model_path),
    }
    
    report_path = args.output.with_suffix("").with_name("cnn_training_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    print(f"Saved CNN model: {model_path}")
    print(f"Saved evidence report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
