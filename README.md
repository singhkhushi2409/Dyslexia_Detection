# NeuroLearn Dyslexia Handwriting Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-modeling-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

NeuroLearn is an educational and research-oriented Streamlit module that analyzes a handwriting image with OpenCV, extracts 13 geometric and structural features, and estimates whether the sample contains patterns associated with the dyslexic class in the training data. It is a screening aid for experimentation, not a diagnostic instrument.

## Table of Contents

- [Overview](#overview)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Feature Definitions and Clinical Rationale](#feature-definitions-and-clinical-rationale)
- [Model and Benchmarks](#model-and-benchmarks)
- [Repository Layout](#repository-layout)
- [Setup and Installation](#setup-and-installation)
- [Run the Application](#run-the-application)
- [Retrain the Model](#retrain-the-model)
- [Input Quality Guidance](#input-quality-guidance)
- [Troubleshooting](#troubleshooting)
- [Clinical and Responsible-Use Disclaimer](#clinical-and-responsible-use-disclaimer)

## Overview

The Writing tab accepts `.jpg`, `.jpeg`, and `.png` files. The application then:

1. Converts the image to grayscale, applies a $3 \times 3$ Gaussian blur, and uses inverted Otsu thresholding.
2. Cleans the binary foreground with a $2 \times 2$ morphological opening and removes detected long horizontal notebook rules.
3. Finds 8-connected components and keeps components with area at least 20 pixels, height at least 5 pixels, and width at least 2 pixels.
4. Groups components by writing row before calculating baseline, spacing, margin, and collision features.
5. Calculates the 13 features documented below.
6. Loads `handwriting_model.pkl`, creates a named feature row, and returns a class prediction plus the model probability for class `1` (`Dyslexic`).

The model is loaded at prediction time from the repository root. It is not trained when the Streamlit app starts.

## Capabilities

- Upload and preview a handwriting image.
- Extract and display the 13 calculated feature values.
- Produce a binary model verdict and dyslexic-class probability.
- Regenerate `handwriting_features.csv` from the labeled image folders and train candidate tree-based classifiers with a stratified 80/20 split.
- Persist the selected estimator to `handwriting_model.pkl` with Joblib.

## Architecture

```text
+------------------+    +----------------------+    +----------------------+
| Handwriting      | -> | PIL upload / OpenCV | -> | Gray + blur + Otsu   |
| JPG / PNG / JPEG |    | image ingestion      |    | inverse threshold     |
+------------------+    +----------------------+    +----------+-----------+
                                                               |
                                                               v
                                                   +----------------------+
                                                   | Morphological opening|
                                                   | and 8-connected CCA  |
                                                   +----------+-----------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | 13 feature vector    |
                                                   | geometry + ink +     |
                                                   | orientation           |
                                                   +----------+-----------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | handwriting_model.pkl |
                                                   | scikit-learn estimator|
                                                   +----------+-----------+
                                                              |
                                                              v
                                                   +----------------------+
                                                   | Prediction, class-   |
                                                   | 1 probability, table |
                                                   +----------------------+

Training path:
handwriting_features.csv -> stratified train/test split -> candidate models
                           -> 5-fold threshold calibration -> best estimator + threshold
                           -> handwriting_model.pkl
```

## Feature Definitions and Clinical Rationale

Let the cleaned binary image have height $H$ and width $W$. Let $N$ be the number of retained connected components. For component $i$, let its bounding box be $(x_i, y_i, w_i, h_i)$ and its foreground area be $a_i$. Unless stated otherwise, standard deviations use NumPy's population convention (`ddof=0`). These are observable handwriting correlates, not clinical biomarkers.

| # | Feature | Mathematical definition | Clinical / behavioral rationale |
|---:|---|---|---|
| 1 | Ink Density (%) | $100 \times \frac{\operatorname{card}\{(x,y): B(x,y)>0\}}{H W}$ | Captures how much of the page is occupied by foreground ink. It can reflect writing compactness and instrument or scan conditions; unusually high or low values may accompany inconsistent motor execution, but are highly sensitive to cropping and pen choice. |
| 2 | Connected Components | $N = \operatorname{card}\{i: a_i \ge 20, h_i \ge 5, w_i \ge 2\}$ | Counts separated foreground regions after cleaning. Fragmentation, disconnected marks, or segmentation into many pieces can reflect irregular formation, though printed text, punctuation, and image noise also change this count. |
| 3 | Mean Letter Height | $\mu_h = \frac{1}{N}\sum_{i=1}^{N} h_i$ | Measures the typical vertical scale of detected components. Writing size is relevant to graphomotor development and spatial planning, but depends strongly on image resolution and the writer's style. |
| 4 | Height Variation (%) | $100 \times \frac{\sigma_h}{\mu_h + 10^{-6}}$ | Normalized size inconsistency. Variable component heights can indicate difficulty maintaining a stable writing zone; ascenders, descenders, mixed case, and segmentation artifacts are important confounders. |
| 5 | Mean Letter Width | $\mu_w = \frac{1}{N}\sum_{i=1}^{N} w_i$ | Describes typical horizontal extent. It provides a coarse measure of spatial scaling and motor planning rather than a direct measure of letter quality. |
| 6 | Width Variation (%) | $100 \times \frac{\sigma_w}{\mu_w + 10^{-6}}$ | Quantifies inconsistency in horizontal sizing. Elevated variation may be compatible with unstable spacing or motor control, but language, handwriting style, and letter composition also affect it. |
| 7 | Baseline Drift | $\sigma_b$, where $b_i = y_i + h_i$ | Measures variation in the bottom edge of components. Difficulty maintaining a consistent baseline is a commonly discussed handwriting organization signal and may relate to visual-spatial or graphomotor demands. Camera perspective and unruled paper can create the same pattern. |
| 8 | Mean Spacing | $\mu_g = \frac{1}{M}\sum_{j=1}^{M} g_j$, where $g_j = x_{j+1}-(x_j+w_j) \ge 0$ | Estimates average horizontal gap between x-sorted components. It approximates writing rhythm and separation, but the implementation does not distinguish character gaps from word gaps. |
| 9 | Spacing Variation (%) | $100 \times \frac{\sigma_g}{\mu_g + 10^{-6}}$ | Measures spacing regularity. Erratic gaps can accompany reduced automaticity or planning load; it is also affected by cursive joins, punctuation, crop boundaries, and component segmentation. |
| 10 | Estimated Stroke Width | $2 \times \operatorname{mean}(D(p): B(p)>0)$, where $D$ is the Euclidean distance transform | Approximates local ink thickness from the distance to background pixels. It may loosely reflect pen pressure and stroke construction, but it is not a pressure sensor and varies with resolution, ink, blur, and thresholding. |
| 11 | Slant Variation | $\sigma_\theta$, with $\theta_i = \frac{1}{2}\arctan2(2\mu_{11},\mu_{20}-\mu_{02})$ for each component | Quantifies consistency of component orientation using second-order central moments. Variable slant can indicate directional or motor inconsistency; isolated components, letter shape, and threshold artifacts can dominate the estimate. |
| 12 | Margin Alignment | $\sigma_x$, the standard deviation of component left edges $x_i$ | Describes consistency of starting positions relative to the image's left edge. It is a rough spatial-alignment measure, not true page-margin detection, and changes with line layout, cropping, and multiple text lines. |
| 13 | Overlap / Collision (%) | $100 \times \frac{\operatorname{card}\{i: x_{i+1}<x_i+w_i\}}{N-1}$ | Counts adjacent x-sorted bounding boxes whose horizontal extents overlap. Collisions can signal crowded or joined writing, but touching letters, cursive writing, and imperfect connected-component segmentation can produce the same result. |

### Interpretation boundary

The rationale above explains why a feature may be worth investigating in handwriting research. It does not establish that any single feature, or this 13-feature combination, identifies dyslexia. Dyslexia is primarily a language-based neurodevelopmental learning disorder; handwriting findings may overlap with dysgraphia, developmental stage, motor differences, vision, fatigue, and writing instruction.

## Model and Benchmarks

`train_handwriting.py` compares HistGradientBoosting, Random Forest, Extra Trees, and a soft-voting ensemble. It uses a stratified 80/20 split with `random_state=42`, then calibrates a classification threshold from 5-fold out-of-fold training probabilities over thresholds from 0.35 to 0.65. The reported benchmark is experimental and should be reproduced on a separately governed evaluation set before deployment.

| Evaluated architecture | 5-fold CV mean F1 | Test accuracy | ROC-AUC | Dyslexic recall |
|---|---:|---:|---:|---:|
| Default Random Forest | 0.8689 | 77.78% | 0.8667 | 76% |
| GridSearch Tuned RF | 0.8877 | 77.78% | 0.8667 | 76% |
| Tuned Extra Trees | 0.8675 | 79.37% | 0.9081 | 82% |
| **HistGradientBoosting (Final)** | **0.8781** | **80.95%** | **0.8707** | **82%** |

The final reported model therefore achieved **80.95% accuracy**, **0.8707 ROC-AUC**, and **82% recall for the dyslexic class**. Accuracy and recall are split-dependent estimates, not guarantees for new populations. The probability displayed by the app is the estimator's class probability; it is not a calibrated clinical risk score.

## Repository Layout

```text
.
|-- app.py                      # Streamlit UI and inference-time feature extraction
|-- handwriting_features.csv    # 13 feature columns plus label
|-- handwriting_model.pkl       # Runtime model artifact (generated by training)
|-- requirements.txt            # Python dependencies
|-- data/                       # Source and intermediate data assets
`-- train_handwriting.py        # Candidate training, evaluation, and model export
```

## Setup and Installation

Use Python 3.10 or newer. Run commands from the repository root. The commands below create an isolated environment for each operating system.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once or activate with `.\.venv\Scripts\activate.bat` from Command Prompt.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Debian or Ubuntu, install the virtual-environment package first if `venv` is unavailable: `sudo apt update && sudo apt install python3-venv`.

## Run the Application

The app expects `handwriting_model.pkl` beside `app.py`.

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, select the **Writing** tab, upload a clear handwriting image, and choose **Predict**. The app writes a temporary `temp_handwriting.jpg` in the repository root during an upload session; do not commit that file.

## Retrain the Model

With the virtual environment active:

```bash
python generate_handwriting_features.py
python train_handwriting.py
```

The first script regenerates features with the current preprocessing from `data/non_dyslexic` and `data/dyslexic`. The second script uses the 13 columns in the exact schema below, prints candidate metrics and a classification report, saves the selected model as `handwriting_model.pkl`, and stores the selected classification threshold in the model artifact.

```text
ink_density, connected_components, mean_height, height_variation,
mean_width, width_variation, baseline_drift, mean_spacing,
spacing_variation, estimated_stroke_width, slant_variation,
margin_alignment, overlap_ratio
```

Do not change column names or order without updating both `train_handwriting.py` and `app.py`. Any model artifact must be evaluated for data leakage, subgroup performance, calibration, and external validity before use beyond research.

## Input Quality Guidance

- Use a well-lit, high-contrast image with the complete writing sample visible.
- Keep the page as flat and front-facing as practical; avoid shadows, glare, ruled backgrounds, and heavy compression.
- Provide enough writing for at least two retained components. Images with fewer than two components are rejected.
- Use the same capture conventions when comparing samples. Resolution, pen, paper, language, age, and writing task all affect the extracted values.
- Treat the displayed probability as model output, not a diagnosis or a measure of intelligence.

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `handwriting_model.pkl was not found` | The runtime artifact has not been generated or is in another directory. | Run `python train_handwriting.py` from the repository root and confirm the file is beside `app.py`. |
| `Unable to read handwriting image` | OpenCV cannot decode the upload or the temporary file. | Re-export the image as a valid JPG or PNG and upload it again. |
| `Not enough handwriting detected` | Thresholding and component filters found fewer than two usable regions. | Upload a larger, clearer, higher-contrast sample with more writing and less background clutter. |
| `ModuleNotFoundError` during startup | Dependencies were installed outside the active environment. | Activate `.venv`, then run `python -m pip install -r requirements.txt`; verify with `python -m pip list`. |
| Streamlit opens but prediction fails | The model feature schema does not match the artifact. | Retrain with the current `MODEL_FEATURE_COLUMNS`; do not rename CSV columns independently. |

## Clinical and Responsible-Use Disclaimer

NeuroLearn is **not a medical device, diagnostic test, or substitute for a professional evaluation**. The benchmark reflects a limited dataset and a single experimental split; it does not establish clinical sensitivity, specificity, fairness, or generalization. Handwriting alone cannot diagnose dyslexia, and a model result must not be used to label, exclude, place, discipline, or deny services to a learner.

Clinical or educational decisions should be made by qualified professionals using validated, age-appropriate, culturally and linguistically appropriate assessments, plus relevant developmental, educational, sensory, and medical history. A high model score warrants human review, not a diagnosis; a low score must not be used to rule dyslexia out. Obtain informed consent, minimize collection of identifiable handwriting samples, protect uploaded data, and provide an accessible human appeal or review path.

For production use, establish dataset governance, independent external validation, subgroup and language analysis, probability calibration, audit logging, model/version controls, retention limits, and a documented clinical safety review before exposing predictions to learners, families, educators, or clinicians.
