# Cloud Chamber Particle Analysis

BMDS2133 project using shared image enhancement, FCN-based Mask R-CNN
instance segmentation, automatic feature extraction and independently
developed machine-learning models.

## Run the current application

```powershell
cd C:\Coding\CloudChamberDetector
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The currently operational shared stages are image/video acquisition,
video-frame selection, greyscale conversion, Gaussian filtering and trained
Mask R-CNN instance segmentation.

## Selected architecture

```text
Image or video
  -> frame acquisition
  -> greyscale conversion
  -> Gaussian filtering
  -> class-agnostic Mask R-CNN instance segmentation
  -> automatic feature extraction
  -> independent member classification models
  -> common evaluation and comparison
```

Mask R-CNN uses a Fully Convolutional Network mask head. It returns one mask
and bounding box for each `particle_track`. The shared segmenter is frozen
before member-model comparison.

## Datasets and split rules

Members may use both the primary dataset and the approved Müller external
dataset. Never train on validation or final-test data.

| Split | Permitted use |
|---|---|
| `development` | Training, implementation and data augmentation |
| `validation` | Hyperparameter tuning, early stopping and model selection |
| `final_test` | One unbiased evaluation after the model and settings are fixed |

### Primary dataset

The physically separated primary data are stored under:

```text
dataset/primary_dataset_split/
├── development/
│   ├── images/
│   └── labels/
├── validation/
│   ├── images/
│   └── labels/
└── final_test/
    ├── images/
    └── labels/
```

The `images` folders contain model inputs. The per-image JSON files in `labels`
contain bounding-box ground truth. `annotation_reference` and
`review_overlays`, when present, are audit images and must not be used as model
inputs.

Primary-dataset limitations:

- 101 electron/positron boxes;
- 2 alpha boxes;
- no proton or V-track boxes;
- bounding boxes only, with no pixel-level instance masks.

Use the primary dataset for acquisition/enhancement testing, bounding-box
experiments, domain testing on the project's own chamber and supplementary
evaluation. Do not use its audit images as additional training data.

### Müller external dataset

The approved Müller dataset provides pixel masks and four particle-track
categories:

```text
alpha
electron_positron
proton
v_track
```

Its original files are stored once:

```text
dataset/external_dataset/
├── images/       # 483 original cloud-chamber images
├── masks/        # Original five-channel Müller semantic NPZ masks
└── inspection/   # Optional supplied annotation previews
```

The training splits are stored as COCO instance-segmentation metadata:

```text
dataset/external_dataset_split/
├── development/
│   ├── annotations_coco.json
│   └── manifest.csv
├── validation/
│   ├── annotations_coco.json
│   └── manifest.csv
└── final_test/
    ├── annotations_coco.json
    └── manifest.csv
```

The external split does not duplicate images. Each `annotations_coco.json`
contains relative `file_name` paths that point back to
`dataset/external_dataset/images`. This is a local path reference, not an
internet link or Windows shortcut. Therefore, do not move, rename or delete the
shared external `images` folder.

Training and evaluation scripts import `MullerTrackDataset` from
`cloud_chamber/ml/muller_dataset.py` as a bridge between the COCO files and
PyTorch. The bridge resolves the linked image paths, decodes the COCO RLE
instance masks, applies the shared greyscale-plus-Gaussian enhancement and
returns Mask R-CNN targets. It is a library module and is not run separately.

Choose the required split by changing the folder in the COCO annotation path.
Do not select or rename individual images.

```python
from cloud_chamber.ml.muller_dataset import MullerTrackDataset

# Use development for model training.
development_dataset = MullerTrackDataset(
    "dataset/external_dataset_split/development/annotations_coco.json",
    augment=True,
    class_agnostic=True,
)

# Use validation for tuning, early stopping and model selection.
validation_dataset = MullerTrackDataset(
    "dataset/external_dataset_split/validation/annotations_coco.json",
    augment=False,
    class_agnostic=True,
)

# Use final_test only after the model and settings have been finalised.
final_test_dataset = MullerTrackDataset(
    "dataset/external_dataset_split/final_test/annotations_coco.json",
    augment=False,
    class_agnostic=True,
)
```

Normally, training code uses `development_dataset` and
`validation_dataset`. The final evaluation code uses `final_test_dataset` and
must evaluate all final-test images rather than manually choosing favourable
examples.

### External dataset label mapping

The original Müller COCO annotations use these category IDs:

| COCO label | Particle category |
|---:|---|
| `0` | Background; not stored as an object annotation |
| `1` | Alpha |
| `2` | Electron/positron |
| `3` | Proton |
| `4` | V track |

There are two ways to load these labels:

```python
# Shared Mask R-CNN segmentation: track versus background.
segmentation_dataset = MullerTrackDataset(
    "dataset/external_dataset_split/development/annotations_coco.json",
    class_agnostic=True,
)

# Four-class member model: retain alpha/electron/proton/V-track labels.
classification_dataset = MullerTrackDataset(
    "dataset/external_dataset_split/development/annotations_coco.json",
    class_agnostic=False,
)
```

When `class_agnostic=True`, the shared segmentation loader remaps every
particle category to:

| Segmentation label | Meaning |
|---:|---|
| `0` | Background |
| `1` | Particle track of any type |

Therefore, labels `2`, `3` and `4` do not appear in the shared Mask R-CNN
training targets. Their original values are preserved in
`original_category_ids` and in the COCO JSON for the independent member
classification models.

When `class_agnostic=False`, labels `1–4` retain the original Müller particle
categories shown in the first table.

The external data were split by complete recording group rather than random
frames. Members must not alter the supplied split because nearby video frames
could otherwise leak between training and evaluation.

## Shared code

| File | Responsibility |
|---|---|
| `app.py` | GUI and workflow presentation |
| `cloud_chamber/acquisition.py` | Image/video loading and frame extraction |
| `cloud_chamber/enhancement.py` | Greyscale conversion and Gaussian filtering only |
| `cloud_chamber/ml/contracts.py` | Standard segmentation and classification outputs |
| `cloud_chamber/ml/base.py` | Interfaces every segmenter/model follows |
| `cloud_chamber/ml/mask_rcnn.py` | Mask R-CNN inference adapter |
| `config.yaml` | Shared paths and fixed parameters |

## Independent member work

All independent machine-learning code must be written inside:

```text
cloud_chamber/ml/member_models/
```

Each member works in the assigned algorithm file:

```text
cloud_chamber/ml/member_models/cnn.py
cloud_chamber/ml/member_models/neuro_explicit.py
cloud_chamber/ml/member_models/yolov5.py
cloud_chamber/ml/member_models/modified_unet.py
cloud_chamber/ml/member_models/mask_rcnn_model.py
```

Do not place training logic directly in `app.py`. Keep the model, feature
extraction, training and prediction logic in the assigned `ml/member_models`
file. A small command-line training entry point may be added under `scripts/`
when required. The GUI page should only collect inputs, call the member module
and display its result.

Members develop their modules independently. Common integration outputs will
be connected after the individual implementations are completed. Every member
model must follow the shared contracts in `cloud_chamber/ml/contracts.py` and
`cloud_chamber/ml/base.py`.

Members may choose different learning architectures and training parameters.
They may not change the shared data split, enhancement settings, segmentation
input, class names or result contract. Clearly document which dataset and split
are used. Development data may be used for training, validation only for model
selection, and final test only for the final reported result.

## Optional ML dependencies

Acquisition and enhancement can run without PyTorch. Mask R-CNN segmentation,
training and member deep-learning models require the CUDA-enabled ML
dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

The trained shared segmenter is already stored at:

```text
models/mask_rcnn_track_segmenter.pt
```

Do not commit virtual environments, model caches or temporary training output.
