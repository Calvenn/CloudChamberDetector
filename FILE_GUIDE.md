# Project File Guide

| File or folder | Purpose |
|---|---|
| `app.py` | Combined acquisition/enhancement/segmentation page, five empty algorithm pages and final comparison |
| `config.yaml` | Fixed dataset, enhancement, segmentation and classification settings |
| `requirements.txt` | Lightweight acquisition/UI dependencies |
| `requirements-ml.txt` | Optional PyTorch and TorchVision dependencies |
| `cloud_chamber/acquisition.py` | Image/video acquisition and frame extraction |
| `cloud_chamber/enhancement.py` | Selected greyscale and Gaussian enhancement |
| `cloud_chamber/models.py` | General image-processing data contracts |
| `cloud_chamber/ml/contracts.py` | Segmented instance and member-model prediction contracts |
| `cloud_chamber/ml/base.py` | `Segmenter` and `ParticleModel` protocols |
| `cloud_chamber/ml/mask_rcnn.py` | FCN mask head within Mask R-CNN inference adapter |
| `cloud_chamber/ml/member_models/` | Separate CNN, neuro-explicit, YOLOv5, modified U-Net and Mask R-CNN member files |
| `primary_dataset/*/images/` | Clean model inputs |
| `primary_dataset/*/labels/` | Bounding-box ground truth |
| `primary_dataset/*/annotation_reference/` | Supplied visual annotations for audit only |
| `primary_dataset/*/review_overlays/` | Extracted boxes drawn on clean frames for review |
| `primary_dataset/annotations_coco.json` | Combined COCO bounding-box annotations |

The previous classical detector modules are legacy placeholders and are not
part of the newly selected machine-learning comparison.
