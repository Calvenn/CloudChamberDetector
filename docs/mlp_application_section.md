# 3.3 Application of the Multilayer Perceptron Algorithm

The Multilayer Perceptron (MLP) is applied as the classification stage of the cloud-chamber particle-track analysis system. Before classification, each input image passes through the shared image-processing pipeline: grayscale conversion, Gaussian filtering, intensity-based thresholding, morphological refinement, contour detection and feature extraction. Each detected contour is represented by ten numerical features: area, perimeter, major-axis length, mean width, orientation, aspect ratio, solidity, rectangularity, thickness and mean intensity. The trained MLP uses these measurements to classify a contour as an alpha, electron/positron, proton or V-shaped track.

During training, labelled contours from the development dataset are converted into feature vectors. A `StandardScaler` standardises every feature so measurements with large numerical values, such as area and perimeter, do not dominate smaller-valued ratios. Candidate MLP structures contain one or two hidden layers with ReLU activation. The Adam optimiser updates network weights, while early stopping limits overfitting. Class imbalance is handled using inverse-frequency sample weights, with deterministic balanced resampling as a compatibility fallback. The candidate with the highest validation macro F1-score is selected; balanced accuracy is used as the tie-breaker. Only after selection is the model evaluated once on the held-out final-test set. The scaler and trained classifier are saved together to ensure identical preprocessing during classification.

During classification, the same preprocessing and feature-extraction procedures are applied to a new cloud-chamber image. The saved pipeline standardises the feature vector and calculates one probability for each particle class. The class with the highest probability becomes the predicted particle type. A confidence threshold of 0.60 is used for reporting: lower-confidence results retain their most likely class but are marked as uncertain for manual review. The application produces an annotated image containing each track ID, predicted class and confidence, together with a CSV table containing the detailed probabilities and inference time.

## Sample input and output

Sample input: a cloud-chamber image containing one or more segmented particle tracks. For example, a thick, bright and comparatively straight contour may produce a feature vector with high area, width and mean intensity.

Sample command:

```text
python scripts/classify_mlp.py dataset/primary_dataset_split/final_test/images/<image-name>.jpg
```

Sample console output:

```text
Detected tracks: 2
Track 1: Alpha (confidence=0.913)
Track 2: Electron/Positron (confidence=0.782)
Annotated image: results/mlp/<image-name>_classified.png
Prediction table: results/mlp/<image-name>_predictions.csv
```

The values above demonstrate the output format; the actual predictions depend on the selected image and trained model.

## Training pseudocode

```text
INPUT: labelled development, validation and final-test images
OUTPUT: saved scaler-and-MLP model and evaluation report

FOR each dataset split
    preprocess every image
    segment particle-track contours
    match each contour to its ground-truth label
    extract the ten numerical features
END FOR

FOR each candidate MLP configuration
    fit StandardScaler using development features
    balance the development classes
    train MLP using ReLU, Adam and early stopping
    predict validation labels
    calculate validation macro F1 and balanced accuracy
END FOR

select candidate with best validation macro F1
use balanced accuracy to break a tie
evaluate selected model on held-out final-test data
save scaler, MLP, feature order, classes and metrics
```

## Classification pseudocode

```text
INPUT: new cloud-chamber image and saved MLP model
OUTPUT: annotated image and prediction table

load configuration and saved scaler-and-MLP pipeline
read input image
convert image to grayscale and apply Gaussian filtering
segment tracks using thresholding and morphology
extract one ten-feature vector for every detected contour

FOR each feature vector
    standardise features using the saved scaler
    calculate MLP class probabilities
    select class with highest probability
    IF confidence is below 0.60
        mark prediction as uncertain
    ELSE
        mark prediction as accepted
    END IF
    draw track ID, class and confidence on output image
END FOR

save annotated image and CSV prediction table
```

## Python files

- `scripts/train_mlp.py` trains, validates, evaluates and saves the MLP.
- `scripts/classify_mlp.py` loads the saved model and classifies a new image.
