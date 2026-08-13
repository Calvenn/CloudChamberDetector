"""Select threshold offset and closing size using validation masks only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    split = ROOT / "dataset/external_dataset_split/validation"
    images = json.loads((split / "annotations_coco.json").read_text())["images"]
    candidates = [(offset, closing) for offset in (0, 2, 4) for closing in (7, 9, 11, 15)]
    totals = {candidate: [0, 0, 0] for candidate in candidates}
    top_hat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

    for record in images:
        image_path = (split / record["file_name"]).resolve()
        grey = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        filtered = cv2.GaussianBlur(grey, (7, 7), 1.4)
        local_bright = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, top_hat_kernel)
        otsu, _ = cv2.threshold(
            local_bright, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        truth = np.any(
            np.load(ROOT / "dataset/external_dataset/masks" / f"{image_path.stem}.npz")[
                "arr_0"
            ][:, :, :4]
            >= 0.5,
            axis=2,
        )
        for offset, closing_size in candidates:
            _, mask = cv2.threshold(
                local_bright, min(255, otsu + offset), 255, cv2.THRESH_BINARY
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (closing_size, closing_size)
            )
            prediction = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)
            totals[(offset, closing_size)][0] += int(np.count_nonzero(prediction & truth))
            totals[(offset, closing_size)][1] += int(np.count_nonzero(prediction & ~truth))
            totals[(offset, closing_size)][2] += int(np.count_nonzero(~prediction & truth))

    results = []
    for (offset, closing), (tp, fp, fn) in totals.items():
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        results.append(
            {
                "offset": offset,
                "closing_kernel": closing,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": tp / (tp + fp + fn),
            }
        )
    results.sort(key=lambda row: (row["f1"], row["iou"]), reverse=True)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
