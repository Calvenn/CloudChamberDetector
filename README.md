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

### Train and use the MLP

Install the updated dependencies, build labelled contour features, train all
declared MLP candidates and select the best one using validation macro F1:

```powershell
python -m pip install -r requirements.txt
python scripts/train_mlp.py
```

The command saves:

- `models/mlp_classifier.joblib`: fitted StandardScaler and MLP classifier.
- `models/mlp_training_report.json`: class counts, every candidate parameter
  result, selected candidate, confusion matrix and final-test metrics.
- `data/features/muller/*.csv`: reproducible labelled contour features.

Use `python scripts/train_mlp.py --rebuild-features` only after changing the
shared feature extraction. In the GUI, process an input on **Shared Processing
Pipeline**, open **MLP**, then select **Classify and create MLP report**.

The report draws a track ID, predicted particle type and confidence on the
input image. Its table contains probabilities for all four particle classes,
bounding-box coordinates and inference time. Predictions below the adjustable
confidence threshold are shown in yellow as `Uncertain`; the threshold changes
reporting only and does not retrain the model. Both the annotated PNG and the
detailed CSV can be downloaded from the MLP page.

### Small-dot rejection

The shared segmenter rejects noise before classification using explicit values
in `config.yaml`. A contour must have area of at least `80 px^2` and major-axis
length of at least `25 px` under the general-track rule. A second electron-like
rule accepts area `90 px^2`, perimeter `40 px`, major-axis length `17 px` and
aspect ratio `2.0`. This preserves an elongated electron-track path while
preventing irregular compact blobs with long boundaries from passing it.

To preserve fragmented tracks without adding another technique, Gaussian
filtering uses a `7x7` kernel with sigma `1.4`. Morphological white top-hat
(`41x41`) suppresses slowly varying background before Otsu thresholding with
an offset of `4`. On all Muller validation masks this achieved pixel F1
`0.4481` and IoU `0.2888`, compared with F1 `0.3734` and IoU `0.2295` for
global Otsu+8. Morphological closing (`15x15`) reconnects longer gaps; this
kernel achieved the best validation F1 (`0.4503`) among 7, 9, 11 and 15.
Morphological opening is disabled because
its erosion stage can erase tracks only one or two pixels wide; residual dots
are handled by contour area, length and aspect ratio instead. Near-frame-spanning
regions are rejected only when they are also extremely thin, which removes
straight chamber boundaries without rejecting an ordinary long track solely
because of its length. These settings must be tuned on validation data and
then fixed for all five models.

The parameter values are not described as universally optimal. The candidate
set tests 32, 64 and 64â†’32 hidden units with L2 strengths `0.0001` and `0.001`.
Adam's documented starting learning rate `0.001` is fixed, while architecture
and L2 strength are selected only from validation results. A maximum of 500
epochs is a safety ceiling. Adam early stopping monitors a stratified 10% of
the development data and stops after 20 epochs without sufficient improvement;
the separate validation split still selects the final candidate.

## Dataset splits

- `development`: fit each classifier.
- `validation`: tune model hyperparameters and shared pipeline settings.
- `final_test`: run once after every choice is fixed.

The physical primary split is under `dataset/primary_dataset_split/`. The MÃ¼ller external
split metadata is under `dataset/external_dataset_split/`; its COCO records
link to the source images rather than duplicating them. The existing prepared
datasets are retained even though the former Mask R-CNN model code was removed.

Use the same labels for all classifiers: alpha, electron/positron, proton and
Report per-class precision, recall and F1-score, macro F1-score, confusion
matrix and processing time.

The project taxonomy follows `Types of particle tracks.pdf`: alpha, proton and
electron/positron (with muon-like thin tracks discussed in the same visual
group). Figures showing a low-energy electron or a secondary electron describe
electron behaviour, not additional classes. MÃ¼ller `v_track` annotations are
excluded rather than incorrectly renamed. The primary dataset contributes
alpha and electron/positron labels; MÃ¼ller provides all three selected classes.

### Dataset-specific region of interest

Choose the image layout on the Shared Processing Pipeline page:

- **External MÃ¼ller / already cropped** uses the complete image.
- **Primary dataset / full chamber** excludes 6% from the left and right, 7%
  from the top and 12% from the bottom. It also uses Otsu offset `40`, closing
  `3x3`, minimum area `200 px^2` and minimum major axis `40 px`. The permissive
  thin-track exception is disabled because dense primary droplets otherwise
  create hundreds of false contours. On 19 primary validation images, this
  reduced unmatched contours from 400 to 56 and gave the best tested one-to-one
  box F1 (`0.239`).

The ROI is applied before morphological closing. Only accepted regions inside
it form the final mask. Contours are then read again from that exact final mask;
the same contours generate both yellow bounding boxes and feature vectors.

