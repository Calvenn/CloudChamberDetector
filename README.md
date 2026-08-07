# Cloud Chamber Particle Classification

University-level BMDS2133 **Mode A** study. All five classifiers use exactly
the same acquisition, preprocessing, segmentation, contour features and data
splits so their results can be compared fairly.

## Selected methodology

1. Acquire a raw image or capture one frame from an MP4/AVI/MOV video.
2. Convert BGR/RGB to grayscale.
3. Apply Gaussian filtering.
4. Apply Otsu binary thresholding.
5. Refine the mask using morphological opening and closing.
6. Detect external contours.
7. Extract area, perimeter, length, width, aspect ratio, solidity,
   rectangularity, thickness, orientation and mean intensity.
8. Classify the common features using CNN, SVM, Decision Tree, MLP or
   Extremely Randomised Trees (Extra Trees).

CLAHE, background subtraction, edge detection, Hough transforms, watershed
and Mask R-CNN are not part of the selected methodology.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Do not run `python app.py`; Streamlit requires its own runner.

To test one image without the GUI:

```powershell
python -m cloud_chamber.cli analyse "path\to\image.jpg"
```

## Shared code

| File | Responsibility |
|---|---|
| `app.py` | Image/video acquisition, shared processing previews, five model pages and final-comparison placeholder. |
| `config.yaml` | Fixed Gaussian, threshold/morphology and dataset settings. |
| `cloud_chamber/acquisition.py` | Image loading, video discovery and frame extraction. |
| `cloud_chamber/enhancement.py` | Grayscale conversion followed by Gaussian filtering only. |
| `cloud_chamber/segmentation.py` | Otsu thresholding, opening, closing and contour detection. |
| `cloud_chamber/features.py` | Common contour-based feature extraction. |
| `cloud_chamber/pipeline.py` | Integrates all shared processing stages. |

Members must not duplicate or change the shared stages during model
comparison. Parameters are tuned using validation data and then fixed.

## Team-member implementation files

| Classifier | File |
|---|---|
| CNN | `cloud_chamber/ml/member_models/cnn.py` |
| SVM | `cloud_chamber/ml/member_models/svm.py` |
| Decision Tree | `cloud_chamber/ml/member_models/decision_tree.py` |
| MLP | `cloud_chamber/ml/member_models/mlp.py` |
| Extremely Randomised Trees | `cloud_chamber/ml/member_models/extra_trees.py` |

Each member adds training, validation, prediction and model-saving functions
only in their assigned file. Each model must use the same feature columns,
class mapping and split manifest.

## Dataset splits

- `development`: fit each classifier.
- `validation`: tune model hyperparameters and shared pipeline settings.
- `final_test`: run once after every choice is fixed.

The physical primary split is under `dataset/primary_dataset_split/`. The Müller external
split metadata is under `dataset/external_dataset_split/`; its COCO records
link to the source images rather than duplicating them. The existing prepared
datasets are retained even though the former Mask R-CNN model code was removed.

Use the same labels for all classifiers: alpha, electron/positron, proton and
V-track where the dataset supplies those classes. Report per-class precision,
recall and F1-score, macro F1-score, confusion matrix and processing time.
