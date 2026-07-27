# Cloud Chamber Particle Track Analysis

University-level BMDS2133 Mode A project for comparing image-processing
techniques used to detect particle tracks in cloud-chamber images and videos.

The project provides one shared workflow for:

1. Image or video acquisition
2. Shared image enhancement
3. Individual particle-track detection techniques
4. Feature measurement and optional classification
5. Quantitative comparison and final result presentation

Each team member develops a detector in their assigned file. Do not create a
separate application or duplicate the acquisition and enhancement code.

If VS Code reports `Import "streamlit" could not be resolved`, select this
Python interpreter:

```text
Image Processing\.venv\Scripts\python.exe
```

## 1. How to run the system

Run the Streamlit graphical interface:

```powershell/cmd
python -m streamlit run app.py
```

Do not run the GUI with `python app.py`. Doing that causes the
`missing ScriptRunContext` warning because Streamlit requires its own runner.

## 2. Dataset to use

The original source archive is:

```text
dataset.zip
```

Do not edit or delete this ZIP. It is the unchanged source backup.

The prepared and labelled dataset is:

```text
primary_dataset/
|-- development/
|   |-- images/
|   |-- labels/
|   |-- annotation_reference/
|   `-- review_overlays/
|-- validation/
|-- final_test/
|-- annotations_coco.json
|-- classes.json
|-- manifest.csv
`-- summary.json
```

The three splits have different purposes:

| Split | Images | Correct use |
|---|---:|---|
| `development` | 34 | Develop or train a method. |
| `validation` | 19 | Tune parameters and select the best version. |
| `final_test` | 24 | Evaluate the completed system once. Do not tune with it. |

Inside each split:

| Folder | Purpose | Use as model input? |
|---|---|---|
| `images` | Clean frames captured from the original MP4 files. | Yes |
| `labels` | Matching JSON bounding-box labels. | Yes, as ground truth |
| `annotation_reference` | Original supplied images containing coloured boxes. | No |
| `review_overlays` | Extracted labels drawn on clean frames for checking. | No |

The red supplied boxes are mapped to `electron_positron`; yellow boxes are
mapped to `alpha`.

Current primary-data limitation:

- 101 electron/positron bounding boxes
- 2 alpha bounding boxes
- 0 proton bounding boxes

## 3. Shared files

Members may use these files but should not change their contracts without team
agreement:

| File | Purpose |
|---|---|
| `app.py` | Streamlit GUI containing acquisition, enhancement, technique pages and final comparison. |
| `config.yaml` | Dataset paths and shared experimental parameters. |
| `cloud_chamber/acquisition.py` | Image loading, video reading, dataset discovery and frame extraction. |
| `cloud_chamber/enhancement.py` | Greyscale, Gaussian filtering, CLAHE, optional background subtraction and threshold preview. |
| `cloud_chamber/models.py` | Shared input/output data structures. |
| `cloud_chamber/pipeline.py` | Connects enhancement, detection, validation, feature extraction and result saving. |
| `cloud_chamber/features.py` | Common track measurements and calibrated measurements. |
| `cloud_chamber/evaluation.py` | Precision, recall, F1, IoU, Dice and CSV comparison output. |
| `cloud_chamber/validation.py` | Rejects invalid detector outputs. |
| `cloud_chamber/detectors/registry.py` | Makes completed detectors available to the GUI and CLI. |
| `FILE_GUIDE.md` | More detailed responsibility of every project file. |

## 6. Where each member implements a technique

Each member works only in the appropriate detector file:

| Technique category | Implementation file | Possible algorithms |
|---|---|---|
| Thresholding | `cloud_chamber/detectors/thresholding.py` | Otsu, Triangle, Yen, Niblack |
| Edge based | `cloud_chamber/detectors/edge_based.py` | Sobel, Laplacian of Gaussian, Canny |
| Morphological | `cloud_chamber/detectors/morphological.py` | Opening, reconstruction, watershed |
| Contour and shape | `cloud_chamber/detectors/contour_shape.py` | Contours, shape descriptors, skeletons |
| Hough transform | `cloud_chamber/detectors/hough.py` | Standard, probabilistic or randomised Hough |
