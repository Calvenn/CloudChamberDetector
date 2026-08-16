# Cloud Chamber Particle Classification

University-level BMDS2133 **Mode A** study. All five classifiers use exactly
the same acquisition, preprocessing, segmentation, contour features and data
splits so their results can be compared fairly.

## Selected methodology

1. Acquire one image, a batch of images, or sampled frames from an
   MP4/AVI/MOV video.
2. Optionally calibrate the image using a known physical reference and apply
   four-corner perspective rectification when the chamber is viewed at an angle.
3. Convert BGR/RGB to grayscale.
4. Apply Gaussian filtering.
5. Apply Otsu binary thresholding.
6. Refine the mask using morphological opening and closing.
7. Detect external contours.
8. Extract area, perimeter, length, width, aspect ratio, solidity,
   rectangularity, thickness, orientation and mean intensity.
9. Classify the common features using CNN, SVM, Decision Tree, MLP or
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

### Image calibration

Spatial scaling is optional because a physical scale cannot be recovered from
an ordinary image without a known reference. Perspective rectification is an
independent control and can be enabled while measurements remain in pixels.
To obtain physical units, enable **spatial calibration**, enter a real reference length in
centimetres and the pixel coordinates of its two endpoints. The calculated
`cm/pixel` scale adds physical length, width, perimeter, thickness and area to
the feature tables and reports.

If the camera views the rectangular chamber at an angle, enable perspective
rectification. The application automatically fills editable top-left,
top-right, bottom-right and bottom-left coordinates and shows their boundary
on the image; correct the values if the preview does not follow the chamber.
The reference endpoints are transformed through the same homography before the
final scale is calculated. Do not enable calibration or claim centimetre
measurements when no reliable physical reference is visible.

On **Shared Processing Pipeline**, choose **Image** to upload several image
files together and press **Load image batch**. Choose **Video** to preview one
frame or set the start frame, sampling interval and maximum frame count before
pressing **Extract video frame batch**. Use the acquisition-batch selector to
choose the image/frame sent through enhancement, segmentation, feature
extraction and classification. The filename and video frame number are kept so
each result remains traceable to its source.

### Understanding the MLP report

After processing an input, open **MLP** and press **Classify and create MLP
report**. The report separates two different signals:

- **Classification confidence** is the MLP class probability.
- **Contour quality** is a transparent heuristic based on local contrast,
  contour shape, thin-track acceptance and analysis-boundary position. It is a
  review aid, not a correctness probability.

The page provides image-level summary cards, class counts, a colour-coded
overview, a particle table and collapsible evidence cards containing the exact
binary contour, all four class probabilities, contour measurements and
warnings. Yellow boxes are uncertain classifications; grey dashed boxes require
segmentation review. Results accumulated for different batch images or video
frames appear in the session batch table.

The MLP confidence slider defaults to `0.60`. Changing it affects only whether
a prediction is reported as confident or uncertain; it does not retrain the
model or change its most likely class. The threshold actually used is recorded
in the JSON and PDF traceability information.

Available downloads are an annotated PNG, particle CSV and multipage PDF
summary. The PDF report retains the input source, model details, confidence
threshold and important result information. Batch results remain visible in
the application summary table.

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
| `cloud_chamber/calibration.py` | Known-reference spatial scaling and optional perspective rectification. |
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

The physical primary split is under `dataset/primary_dataset_split/`.

### How the Müller split works

The Müller split is **metadata-based**. Its 483 images remain once in
`dataset/external_dataset/images/` to avoid making three unnecessary copies.
Members must not select or move those source images manually. Membership is
defined by the COCO file in each split folder:

| Purpose | COCO file used by the code | Images | Labelled instances |
| --- | --- | ---: | ---: |
| Train the model | `dataset/external_dataset_split/development/annotations_coco.json` | 330 | 5,336 |
| Select parameters | `dataset/external_dataset_split/validation/annotations_coco.json` | 108 | 1,624 |
| Report the final result once | `dataset/external_dataset_split/final_test/annotations_coco.json` | 45 | 1,484 |

Each COCO image record contains a relative link such as
`../../external_dataset/images/<image name>.jpg`. The loader resolves this link
automatically. `manifest.csv` is a human-readable list for checking which
recording and image belongs to a split; it is not the training input.

For the provided MLP implementation, members normally do not specify the files
one at a time. Run:

```powershell
.\.venv\Scripts\python.exe scripts\train_mlp.py --rebuild-features
```

`scripts/train_mlp.py` automatically loops over `development`, `validation`
and `final_test`, reads the correct `annotations_coco.json`, trains only on the
development feature table, selects the candidate on validation, and evaluates
the selected model on final test. Other model members should follow the same
split names and may reuse `cloud_chamber/ml/contour_dataset.py` to build or load
their labelled contour-feature tables.

The split is leakage-safe by recording: an entire video recording belongs to
only one split. See `dataset/external_dataset_split/summary.json` for the counts
and audit, which records `recording_group_overlap: false` and
`image_id_overlap: false`.

Use the same configured labels for all classifiers: `alpha`,
`electron_positron`, `proton` and Müller `v_track`. Report per-class precision,
recall and F1-score, macro F1-score, confusion matrix and processing time.

The project taxonomy follows `Types of particle tracks.pdf`: alpha, proton and
electron/positron (with muon-like thin tracks discussed in the same visual
group). Figures showing a low-energy electron or a secondary electron describe
electron behaviour, not additional classes. Müller `v_track` is retained as a
fourth external-dataset class and is never renamed or merged into alpha. The
primary dataset contributes alpha and electron/positron labels; Müller provides
all four configured classes.

### Dataset-specific region of interest

The application identifies the image layout automatically; no analysis-region
control is shown on the Shared Processing Pipeline page:

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

