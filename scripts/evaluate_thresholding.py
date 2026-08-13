"""Compare permitted threshold settings on Müller validation masks only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_chamber.config import load_config


def main() -> int:
    config = load_config()
    split = PROJECT_ROOT / "dataset/external_dataset_split/validation"
    records = json.loads((split / "annotations_coco.json").read_text())["images"]
    candidates = [
        ("otsu", 0, 0),
        ("otsu", 0, 8),
        *[
            ("white_tophat", kernel, offset)
            for kernel in (11, 15, 21, 31, 41)
            for offset in (0, 4, 8)
        ],
        *[
            ("adaptive_gaussian", block, constant)
            for block in (31, 51, 81, 121)
            for constant in (-4, -2, 0, 2)
        ],
    ]
    totals = {candidate: [0, 0, 0] for candidate in candidates}
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for record in records:
        image_path = (split / record["file_name"]).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        filtered = cv2.GaussianBlur(image, (7, 7), 1.4)
        mask_path = PROJECT_ROOT / "dataset/external_dataset/masks" / (
            image_path.stem + ".npz"
        )
        labelled = np.load(mask_path)["arr_0"]
        truth = np.any(labelled[:, :, :4] >= 0.5, axis=2)

        for candidate in candidates:
            method, block_size, constant = candidate
            if method == "otsu":
                otsu, _ = cv2.threshold(
                    filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                _, prediction = cv2.threshold(
                    filtered, min(255, otsu + constant), 255, cv2.THRESH_BINARY
                )
            elif method == "adaptive_gaussian":
                prediction = cv2.adaptiveThreshold(
                    filtered,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    constant,
                )
            else:
                top_hat_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (block_size, block_size)
                )
                local_bright = cv2.morphologyEx(
                    filtered, cv2.MORPH_TOPHAT, top_hat_kernel
                )
                otsu, _ = cv2.threshold(
                    local_bright,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                _, prediction = cv2.threshold(
                    local_bright,
                    min(255, otsu + constant),
                    255,
                    cv2.THRESH_BINARY,
                )
            prediction = cv2.morphologyEx(
                prediction, cv2.MORPH_CLOSE, close_kernel
            )
            prediction = cv2.morphologyEx(
                prediction, cv2.MORPH_OPEN, open_kernel
            ).astype(bool)
            totals[candidate][0] += int(np.count_nonzero(prediction & truth))
            totals[candidate][1] += int(np.count_nonzero(prediction & ~truth))
            totals[candidate][2] += int(np.count_nonzero(~prediction & truth))

    results = []
    for candidate, (true_positive, false_positive, false_negative) in totals.items():
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
        iou = true_positive / (true_positive + false_positive + false_negative)
        results.append(
            {
                "method": candidate[0],
                "block_size": candidate[1],
                "constant_or_offset": candidate[2],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            }
        )
    results.sort(key=lambda item: (item["f1"], item["iou"]), reverse=True)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
