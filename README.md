# Cloud Chamber Particle-Track Classification

This project detects particle tracks in cloud-chamber images and video frames,
extracts a representation for each segmented track, and classifies it as:

- Alpha
- Electron/Positron
- Proton
- V-track

The Streamlit application provides a shared image-processing pipeline,
individual dashboards for five classifiers, downloadable reports, and a final
model-comparison dashboard.

## System workflow

```text
Image or video
      ↓
Acquisition and optional temporal video enhancement
      ↓
Perspective rectification
      ↓
Overlapping tiling and spatial scaling
      ↓
Grayscale → Gaussian filter → white top-hat enhancement
      ↓
Otsu-guided hysteresis thresholding
      ↓
Morphological closing and fragment linking
      ↓
Contour filtering and learned artefact rejection
      ↓
Segmented tracks → classification → report
```

### Image processing

1. **Acquisition** accepts JPG, JPEG, PNG, TIF and TIFF images or MP4, AVI and
   MOV videos. Video samples retain their frame number, timestamp and source.
2. **Temporal enhancement** can compare neighbouring frames to strengthen
   transient tracks while suppressing stationary background information.
3. **Rectification** maps four chamber boundary points to a rectangle. It
   corrects perspective only on a processing copy; the source remains unchanged.
4. **Tiling** covers the rectified image with overlapping `800 × 800` tiles.
   The 15% overlap protects tracks crossing tile boundaries. Each tile is
   processed at `640 × 640` pixels.
5. **Enhancement** converts the tile to grayscale, applies a `7 × 7` Gaussian
   filter with sigma `1.4`, and uses white top-hat transformation to emphasise
   locally bright tracks over uneven backgrounds.
6. **Segmentation** uses Otsu to estimate an image-dependent threshold.
   Hysteresis retains faint pixels only when connected to strong track pixels.
7. **Refinement** uses closing, directional reconnection and aligned/curved
   fragment linking to reconnect broken track sections.
8. **Filtering** rejects contours that fail the general-track or thin-track
   rules. A separate Extra Trees artefact filter can reject scratches,
   reflections and background texture that survive the geometric rules.
9. Tile masks are mapped back to the complete image and merged. These final
   contours are passed to feature extraction and classification.

Dataset-specific segmentation profiles are stored in `config.yaml`. The
primary and Müller datasets use separate artefact filters because their image
conditions and annotation formats differ.

## Classifiers

| Model | Implementation | Input used by the current code |
|---|---|---|
| CNN | PyTorch `TrackPatchCNN` | `128 × 128` masked grayscale track patch |
| SVM | scikit-learn calibrated SVC | 11 selected contour features |
| Decision Tree | scikit-learn `DecisionTreeClassifier` | 10 contour features |
| MLP | Five scikit-learn MLP pipelines with soft voting | All 16 contour features |
| Extremely Randomized Trees | scikit-learn `ExtraTreesClassifier` | All 16 contour features |

The shared 16-feature record contains area, perimeter, major axis, mean width,
orientation, aspect ratio, solidity, rectangularity, thickness, mean intensity,
intensity standard deviation, circularity, convexity,
perimeter-to-major-axis ratio, and sine and cosine encodings of twice the
orientation.

The CNN uses the segmented shape directly rather than this numerical vector.
SVM and Decision Tree use selected subsets defined in their model modules.
Every saved model stores its expected feature order, preventing incompatible
runtime features from being silently accepted.

## Installation and startup

Python 3.10 or newer is required.

```powershell
cd C:\Coding\CloudChamberDetector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Do not use `python app.py`; Streamlit applications require the Streamlit
runner.

## Using the application

### 1. Process an input

Open **Shared Processing Pipeline** and choose **Image** or **Video**.

- For images, upload one or more files and select a sample from the batch.
- For video, configure the starting frame, interval and maximum frame count.
- Enable temporal enhancement when a track is clearer across neighbouring frames.
- Check the proposed rectification boundary and adjust it when necessary.
- Review the enhancement, binary mask, accepted contours and feature table.

The shared result is stored in the Streamlit session. All model pages therefore
classify the same accepted segmented regions for the current input.

### 2. Classify the tracks

Open a classifier page and run classification. A result can contain:

- Track identifier and segmented shape
- Predicted class and confidence
- Per-class probability evidence
- Segmentation-quality information
- Processing time and downloadable outputs

Low confidence marks a prediction as uncertain but does not change its most
likely class or colour. `Review segmentation` means the contour may be
incomplete, fragmented or merged; it is not another particle class.

### 3. Compare the models

Open **Final Model Comparison** to compare accuracy, balanced accuracy, macro
F1, weighted F1, per-class recall and inference time.

The official comparison reads `final_test` from each model report. This section
uses held-out ground-truth particle records. A report may also contain
`segmented_final_test`, which evaluates automatic segmentation output. These
sections use different cohorts and must not be presented as the same result.

## Dataset design

| Partition | Purpose |
|---|---|
| Development | Fit model parameters |
| Validation | Select model and processing settings |
| Final test | Evaluate once after all choices are fixed |

Partitions are organised by recording group rather than random images. This
prevents neighbouring frames with nearly identical backgrounds from appearing
in both training and testing data.

### Primary dataset

Prepared data are stored under `dataset/primary_dataset_split/`. Its supplied
annotations are bounding boxes and do not provide sufficient examples of all
four classes for an independent four-class comparison.

### External Müller dataset

The Müller data provide five-channel semantic masks: four particle channels and
one background channel. Conversion produces individual instances and COCO
annotations. Source images remain in one shared folder; split membership is
defined by each partition's `annotations_coco.json` and `manifest.csv`.

Do not guess which shared-folder image is unused. Use the final-test COCO file
or manifest to identify images reserved for testing.

## Training

Run commands from the project root:

```powershell
# Build or verify shared feature tables
python scripts/data/build_shared_features.py

# Feature-based classifiers
python scripts/training/train_svm.py
python scripts/training/train_decision_tree.py
python scripts/training/train_mlp.py
python scripts/training/train_extra_trees.py

# CNN
python -m cloud_chamber.ml.member_models.cnn

# Domain-specific particle-versus-artefact filters
python scripts/training/train_artifact_filter.py --domain external
python scripts/training/train_artifact_filter.py --domain primary
```

Use `--rebuild-features` or the relevant segmented-feature rebuild option only
after changing segmentation or feature extraction. Run a command with `--help`
to view its supported options.

Training uses ground-truth records together with production-segmented examples
where implemented. Official validation and final-test classifier metrics use
one ground-truth representation per annotation. Segmentation duplicates or
missed contours therefore do not redefine the official classifier test cohort.

The selected MLP is retrained as five independently seeded
StandardScaler → MLPClassifier pipelines. Soft voting averages their class
probabilities. Ensemble agreement measures stability between these members; it
does not prove correctness without a ground-truth label.

## Evaluation metrics

- **Accuracy:** proportion of all predictions that are correct.
- **Balanced accuracy:** average recall across classes.
- **Precision:** proportion of predictions for a class that are correct.
- **Recall:** proportion of the true class that the model finds.
- **F1-score:** balance between precision and recall.
- **Macro F1:** mean class F1 with equal importance for every class.
- **Weighted F1:** mean class F1 weighted by class frequency.
- **F2-score:** precision–recall score giving greater importance to recall.
- **IoU/Dice:** pixel overlap between predicted and ground-truth masks.

Accuracy alone can be influenced by the most frequent class. Model analysis
therefore also uses macro F1, balanced accuracy, per-class metrics and confusion
matrices.

## Project structure

```text
app.py                         Streamlit composition and navigation
config.yaml                    Dataset and processing settings
cloud_chamber/
  core/                        Configuration, contracts and validation
  image_processing/            Acquisition through segmentation
  feature_extraction/          Shared contour-feature calculations
  ml/
    member_models/             Five classifier implementations
    artifact_filter.py         Particle-versus-artefact model support
    contour_dataset.py         Labelled feature generation
  evaluation/                  Annotations and segmentation metrics
  reporting/                   Quality checks and report exports
  ui/
    shared_pipeline/           Shared processing interface
    model_pages/               Classifier dashboards
    comparison_page.py         Final-test comparison
scripts/
  data/                        Dataset and feature preparation
  training/                    Model training
  evaluation/                  Segmentation experiments
models/                        Saved models and JSON reports
results/                       Generated experimental results
```

See `FILE_GUIDE.md` for a detailed responsibility map.

## Reproducibility and checks

- The configured random seed is `42`.
- Feature order and classes are stored with each trained model.
- Recording-group splits reduce frame-level leakage.
- Parameters are selected using validation data, not final-test data.
- Segmentation or feature-contract changes require regenerated segmented
  features and retraining of affected classifiers.

```powershell
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m compileall -q app.py cloud_chamber scripts
```

## Limitations

- Segmentation sensitivity varies with illumination and background conditions.
- Missed tracks cannot be classified; fragmented or merged contours alter the
  representation supplied to the model.
- Pixel-based measurements depend on image scale because complete physical
  calibration is not part of the evaluated workflow.
- Proton and V-track results are less stable when their test counts are small.
- CNN and feature-based classifiers use different representations, so their
  comparison covers the complete representation-and-classification strategy.
