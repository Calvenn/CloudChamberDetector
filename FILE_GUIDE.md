# Project File Guide

The selected shared flow is acquisition → grayscale → Gaussian filter → Otsu
threshold → morphological opening/closing → contours → contour features.

| File | What it does |
|---|---|
| `app.py` | Runs the Streamlit interface and displays every shared stage. |
| `config.yaml` | Stores fixed paths and experiment parameters. |
| `cloud_chamber/acquisition.py` | Loads images and captures video frames. |
| `cloud_chamber/calibration.py` | Converts pixel measurements to physical units and optionally corrects perspective. |
| `cloud_chamber/enhancement.py` | Performs grayscale conversion and Gaussian filtering. |
| `cloud_chamber/segmentation.py` | Performs thresholding, morphology and contour detection. |
| `cloud_chamber/features.py` | Converts contours into the common numerical feature vector. |
| `cloud_chamber/pipeline.py` | Runs the shared stages together and saves their images. |
| `cloud_chamber/models.py` | Defines shared result data structures. |
| `cloud_chamber/evaluation.py` | Calculates mask metrics where ground truth masks exist. |
| `cloud_chamber/cli.py` | Tests acquisition and processing without the GUI. |
| `cloud_chamber/ml/member_models/*.py` | Separate team workspaces for the five classifiers. |

Dataset preparation scripts remain separate from runtime code because they
reconstruct the labelled splits. Prepared images and annotations must not be
deleted merely because an earlier segmentation model was removed.
