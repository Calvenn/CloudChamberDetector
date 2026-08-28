# Project File Guide

The shared runtime flow is acquisition → calibration/tiling → grayscale and
Gaussian enhancement → white top-hat enhancement → Otsu-guided hysteresis →
morphological refinement → contour filtering → shared feature extraction.

| File | What it does |
|---|---|
| `app.py` | Small composition root: configures Streamlit, navigation, and page dependencies. |
| `config.yaml` | Stores fixed paths and experiment parameters. |
| `cloud_chamber/image_processing/acquisition.py` | Loads images and captures traceable video frames. |
| `cloud_chamber/image_processing/calibration.py` | Selects/scales regions and performs perspective rectification. |
| `cloud_chamber/image_processing/enhancement.py` | Performs grayscale, Gaussian and white top-hat enhancement. |
| `cloud_chamber/image_processing/segmentation.py` | Performs thresholding, morphology, fragment linking and contour filtering. |
| `cloud_chamber/image_processing/tiling.py` | Splits, scales and recombines overlapping image tiles. |
| `cloud_chamber/image_processing/video_processing.py` | Builds temporally enhanced evidence from neighbouring video frames. |
| `cloud_chamber/image_processing/pipeline.py` | Provides non-interactive orchestration and reproducible image export. |
| `cloud_chamber/feature_extraction/contour_features.py` | Defines and calculates the common 16-feature particle-track vector. |
| `cloud_chamber/core/contracts.py` | Defines shared data contracts exchanged between processing, ML and reporting. |
| `cloud_chamber/core/config.py` | Loads fixed paths and experiment parameters. |
| `cloud_chamber/core/validation.py` | Validates shared detector output contracts. |
| `cloud_chamber/evaluation/annotations.py` | Loads ground-truth masks and particle annotations. |
| `cloud_chamber/evaluation/metrics.py` | Calculates mask precision, recall, F1, IoU and Dice. |
| `cloud_chamber/reporting/quality.py` | Assesses segmentation quality independently of model confidence. |
| `cloud_chamber/reporting/summaries.py` | Builds image summaries and traceability metadata. |
| `cloud_chamber/reporting/exports.py` | Encodes JSON, CSV, single-image PDF and batch PDF reports. |
| `cloud_chamber/ui/feature_rows.py` | Applies shared column names, rounding, and physical-unit conversion to feature tables. |
| `cloud_chamber/ui/shared_pipeline/` | Owns shared acquisition, calibration, enhancement, segmentation, and feature-review UI. |
| `cloud_chamber/ui/model_pages/*.py` | Keeps each classifier interface independent and maintainable. |
| `cloud_chamber/ml/member_models/*.py` | Separate team workspaces for the five classifiers. |

## Script organisation

The `scripts/` directory contains command-line utilities only; Streamlit imports
the reusable packages above directly.

| Folder | Purpose |
|---|---|
| `scripts/data/` | Prepare, convert and split datasets, and build shared feature tables. |
| `scripts/training/` | Train the five classifiers, the artefact filter, and run model inference utilities. |
| `scripts/evaluation/` | Evaluate segmentation/model results and tune segmentation parameters. |

Run a script from the project root, for example:
`python scripts/training/train_mlp.py`.

## Development quality checks

Install the optional tools with `pip install -e .[dev]`, then run:

```text
ruff check .
pytest
python -m compileall -q app.py cloud_chamber scripts
```

Generated datasets, reports, and trained model artefacts are deliberately kept
outside runtime modules. They should not be edited during structural refactoring.

The root contains only application-level utilities. Processing, feature
extraction, contracts and reporting are kept in their dedicated packages.

Dataset preparation scripts remain separate from runtime code because they
reconstruct the labelled splits. Prepared images and annotations must not be
deleted merely because an earlier segmentation model was removed.
