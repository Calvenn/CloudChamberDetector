# Project File Guide

This guide explains the responsibility of every source file. Individual members
should keep algorithm code inside the assigned detector file and avoid changing
the shared contracts without team agreement.

## Project entry points

| File | Purpose |
|---|---|
| `app.py` | Streamlit GUI for image/video input, frame selection, enhancement previews, technique pages, and final comparison. |
| `config.yaml` | Shared paths and fixed experimental parameters. Members should read settings from here instead of hardcoding values. |
| `requirements.txt` | Python packages required to run the project. |
| `pyproject.toml` | Installable Python-project metadata and dependency declarations. |
| `README.md` | Setup instructions, dataset location, commands, and team workflow. |
| `FILE_GUIDE.md` | Describes the responsibility of each project file. |

## Shared package

| File | Purpose |
|---|---|
| `cloud_chamber/__init__.py` | Marks `cloud_chamber` as a Python package and stores its version. |
| `cloud_chamber/config.py` | Loads `config.yaml` and rejects missing or invalid settings. |
| `cloud_chamber/models.py` | Defines shared contracts such as `ImageSample`, `EnhancementResult`, `DetectionResult`, and `EvaluationResult`. |
| `cloud_chamber/acquisition.py` | Loads images, discovers raw dataset files, reads videos, and exports sampled frames. It excludes `Annotated` and `BW`. |
| `cloud_chamber/enhancement.py` | Implements greyscale conversion, optional reference-frame subtraction, Gaussian filtering, CLAHE, and the Otsu preview. |
| `cloud_chamber/annotations.py` | Loads and validates particle annotations and binary ground-truth masks. |
| `cloud_chamber/features.py` | Measures common geometric and intensity features in pixels. |
| `cloud_chamber/evaluation.py` | Calculates precision, recall, F1-score, IoU, Dice score, and exports CSV results. |
| `cloud_chamber/validation.py` | Checks every member returns a valid same-sized binary mask, bounding boxes, and processing time. |
| `cloud_chamber/pipeline.py` | Connects acquisition, enhancement, one detector, validation, feature extraction, and result saving. |
| `cloud_chamber/cli.py` | Command-line alternative for dataset scanning, frame extraction, image analysis, and detector comparison. |

## Individual detector package

| File | Purpose |
|---|---|
| `cloud_chamber/detectors/base.py` | Defines the interface every individual detector must follow. |
| `cloud_chamber/detectors/registry.py` | Makes completed detectors discoverable by the CLI and GUI. |
| `cloud_chamber/detectors/baseline.py` | Working Otsu baseline for verifying the shared pipeline. It is not automatically the final method. |
| `cloud_chamber/detectors/thresholding.py` | Member area for Otsu, Triangle, Yen, Niblack, and the thresholding winner. |
| `cloud_chamber/detectors/edge_based.py` | Member area for Sobel, Laplacian of Gaussian, Canny, and the edge winner. |
| `cloud_chamber/detectors/morphological.py` | Member area for watershed, directional opening, reconstruction, and the morphology winner. |
| `cloud_chamber/detectors/contour_shape.py` | Member area for contour descriptors, Hu moments, skeleton analysis, and the shape winner. |
| `cloud_chamber/detectors/hough.py` | Member area for standard, probabilistic, randomised Hough, and the Hough winner. |
| `cloud_chamber/detectors/__init__.py` | Imports completed detectors so their registration decorators execute. |

## Dataset preparation

| File or folder | Purpose |
|---|---|
| `scripts/prepare_dataset.py` | Creates session-safe development, validation, and final-test folders from `dataset.zip`, excluding `Annotated` and `BW`. |
| `scripts/prepare_primary_dataset.py` | Captures clean frames from the original MP4s, extracts red/yellow supplied boxes into per-image JSON and COCO labels, creates session-safe splits, and writes review overlays. |
| `primary_dataset/*/images/` | Clean MP4 frames used as model input. Enhancement is applied later by the shared pipeline. |
| `primary_dataset/*/labels/` | Per-image bounding-box labels for alpha and electron/positron tracks. |
| `primary_dataset/*/annotation_reference/` | Original coloured-box images retained for audit only; never use these as model inputs. |
| `primary_dataset/*/review_overlays/` | Extracted boxes drawn on clean frames for visual label checking. |
| `primary_dataset/annotations_coco.json` | Combined standard COCO object-detection annotations. |
| `primary_dataset/manifest.csv` | Lists each clean frame, label, session, split, source video time, and class counts. |
| `primary_dataset/summary.json` | Records image and bounding-box counts by split and class. |
| `dataset_clean/manifest.csv` | Lists every raw image, recording session, and assigned split. |
| `dataset_clean/summary.json` | Records split counts, percentages, exclusions, and session assignments. |
| `data/ground_truth/annotations/` | Stores machine-readable particle labels and bounding boxes. |
| `data/ground_truth/masks/` | Stores manually prepared binary track masks. |
| `data/frames/` | Stores JPG frames extracted from MP4 videos. |
| `results/` | Stores generated masks, measurements, and metrics. It must not be used as training input. |

## Tests

| File | Purpose |
|---|---|
| `tests/test_acquisition.py` | Verifies discovery, processed-folder exclusion, and frame export. |
| `tests/test_config.py` | Verifies shared configuration loading. |
| `tests/test_contracts.py` | Verifies shared detector-result rules. |
| `tests/test_evaluation.py` | Verifies comparison metrics. |
| `tests/test_gui_helpers.py` | Verifies GUI image and mask decoding. |
| `tests/test_pipeline.py` | Verifies enhancement stages and the baseline detector contract. |
| `tests/test_prepare_dataset.py` | Verifies raw-image inclusion and `Annotated`/`BW` exclusion. |
| `tests/test_primary_dataset.py` | Verifies red/yellow annotation extraction, including intersecting boxes. |
