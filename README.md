---

# 🧠 Brain Tumor MRI Classification using EfficientNetB0 and TensorFlow

## Overview

This project is a deep learning-based medical image classification system designed to detect and classify brain tumors from MRI scans. It leverages transfer learning with **EfficientNetB0** to classify MRI images into four distinct categories with **95% accuracy**.

The model is deployed via a **Streamlit** web application, allowing users to upload MRI images and receive real-time predictions along with confidence scores.

---

## Dataset

**Source:** [Kaggle – Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- **Size:** ~7,000 MRI images (Training: ~5,600 | Testing: 1,600)
- **Classes:** 4 distinct categories (400 images per class in test set)
- **Preprocessing:** Resized to 224×224 pixels, converted to 3-channel RGB for EfficientNet compatibility.

### Classes

| Label | Description |
| --- | --- |
| **Glioma** | Malignant brain tumor originating from glial cells |
| **Meningioma** | Tumor arising from the meninges (brain's outer membrane) |
| **Pituitary** | Tumor in the pituitary gland at the brain's base |
| **No Tumor** | Healthy brain MRI (no abnormality detected) |

---

## Model Architecture

The model uses a hybrid approach: a frozen/pre-trained feature extractor followed by a custom, regularized classification head.

- **Base Model:** EfficientNetB0 (Pretrained on ImageNet, excluding top layers)
- **Custom Classification Head:**
  1. `GlobalAveragePooling2D`
  2. `Dropout` (rate = 0.3)
  3. `Dense` (128 neurons, ReLU activation)
  4. `Dropout` (rate = 0.2)
  5. `Dense` (64 neurons, ReLU activation)
  6. `BatchNormalization`
  7. `Dropout` (rate = 0.3)
  8. `Dense` (4 neurons, Softmax activation)

---

## Techniques Used

- **Transfer Learning:** Leveraging pre-trained ImageNet weights for robust feature extraction.
- **Two-Phase Training Strategy:** Feature extraction followed by targeted fine-tuning.
- **Architectural Regularization:** Compressed Dense layers (128 → 64) and strategic Dropout to prevent overfitting on medical data.
- **Data Augmentation:** Rotation (±10°), zoom (±10%), width/height shifts (±10%), and horizontal flipping via `ImageDataGenerator`.
- **Callbacks:** `EarlyStopping` (patience=5), `ReduceLROnPlateau` (factor=0.2, patience=3), and `ModelCheckpoint`.
- **Domain-Specific Preprocessing:** EfficientNet's native `preprocess_input` function.

---

## Training Strategy

### Phase 1: Feature Extraction (Warm-up)
- **Base Model:** Entire `EfficientNetB0` frozen (`trainable = False`).
- **Optimizer:** Adam with learning rate = `1e-3`.
- **Loss:** `sparse_categorical_crossentropy`.
- **Duration:** 10 epochs.
- **Goal:** Train only the randomly initialized custom Dense head without destroying pre-trained weights (prevents catastrophic forgetting).

### Phase 2: Fine-Tuning
- **Base Model:** Unfrozen, but only the **last 20 layers** are set to `trainable = True`.
- **Optimizer:** Adam with a reduced learning rate = `1e-4`.
- **Loss:** `sparse_categorical_crossentropy`.
- **Duration:** Up to 20 epochs (with EarlyStopping).
- **Goal:** Gently adapt the high-level, task-specific features of EfficientNet to MRI textures while the custom head fine-tunes alongside it.

---

## Performance

### Overall Metrics

| Metric | Score |
| --- | --- |
| **Training Accuracy** | ~98.9% |
| **Validation Accuracy** | **95.0%** |
| **Validation Loss** | ~0.25 |
| **Macro Avg F1-Score** | **0.95** |

### Confusion Matrix (Test Set: 1,600 images)

```
Predicted →     Glioma  Meningioma  NoTumor  Pituitary
Actual ↓
Glioma           330       39         29        2
Meningioma         2      396          0        2
No Tumor           0        0        400        0
Pituitary          0        0          0      400
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Glioma | 0.99 | 0.82 | 0.90 | 400 |
| Meningioma | 0.91 | 0.99 | 0.95 | 400 |
| No Tumor | 0.93 | 1.00 | 0.97 | 400 |
| Pituitary | 0.99 | 1.00 | 1.00 | 400 |
| **Accuracy** | | | **0.95** | **1600** |

### Key Observations
- **No Tumor & Pituitary:** Perfect 100% recall — zero misclassifications.
- **Meningioma:** Near-perfect at 99% recall (only 4 errors).
- **Glioma:** Slightly lower recall (82%) — 39 gliomas confused with meningioma, 29 with no tumor. This is expected due to the visual variability of gliomas across stages.

---

## Technologies Used

- **Core:** Python, TensorFlow 2.x, Keras
- **Architecture:** EfficientNetB0, NumPy, Pillow
- **Evaluation:** Scikit-learn (confusion matrix, classification report)
- **Deployment:** Streamlit
- **Visualization:** Matplotlib, Seaborn

---

## Features

- 🖼️ Upload MRI images through an intuitive web interface.
- ⚡ Real-time, low-latency predictions.
- 📊 Confidence score display for all 4 classes.
- 🧠 Multi-class brain tumor classification (Glioma, Meningioma, Pituitary, No Tumor).
- 🚀 Lightweight and computationally efficient architecture (~28M parameters).

---

## Project Files

| File | Description |
| --- | --- |
| `Brain_Tumor_MRI_using_transfer_learning_.ipynb` | Complete Jupyter/Colab Notebook with training, fine-tuning, and evaluation code |
| `app.py` | Streamlit application entry point |
| `brain_tumor_classifier.keras` | Trained TensorFlow Keras model weights |
| `best_model.keras` | Best model checkpoint (lowest val_loss) saved during training |
| `class_indices.json` | Class label mapping used for prediction output |
| `requirements.txt` | Python dependencies required to run the app |

---

## Setup & Installation

1. **Clone or download** the project repository:
   ```bash
   git clone https://github.com/zaidkhan8241/Brain-Tumor-MRI-Prediction.git
   cd Brain-Tumor-MRI-Prediction
   ```

2. **Create and activate** a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify required files are present:**
   - `brain_tumor_classifier.keras` (or `best_model.keras`)
   - `class_indices.json`

---

## Usage

### Run the Streamlit Web App
```bash
streamlit run app.py
```
This launches a local web server (typically at `http://localhost:8501`). Upload an MRI image through the browser interface to receive a prediction with confidence scores.

### Run Training (Google Colab / Jupyter)
Open `Brain_Tumor_MRI_using_transfer_learning_.ipynb` in Google Colab or Jupyter Notebook and execute all cells sequentially. The notebook handles:
1. Dataset download (via `opendatasets`)
2. Data augmentation & generator setup
3. Phase 1: Feature extraction (10 epochs)
4. Phase 2: Fine-tuning (up to 20 epochs with callbacks)
5. Evaluation (confusion matrix + classification report)
6. Model export & download

> ⚠️ **Important:** When evaluating the model, always set `shuffle=False` in the evaluation generator to ensure predictions align correctly with true labels.

---

## License

This project is for educational and research purposes only. It is **not** a certified medical device and should **not** be used for clinical diagnosis.

---

## Acknowledgments

- Dataset: [Masoud Nickparvar – Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Architecture: [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)

---

### Summary of Changes Made:

| Section | What Changed |
|---|---|
| Fine-tuning LR | `1e-5` → **`1e-4`** (matches your actual training logs) |
| Performance table | Removed the "please verify" note; added **actual numbers** from your run |
| Confusion Matrix | Added the **real 4×4 matrix** from your evaluation |
| Classification Report | Added **full per-class table** with precision/recall/F1 |
| Key Observations | Added medical context for the Glioma confusion pattern |
| Setup section | **Completed** the truncated installation steps |
| Usage section | Added Streamlit launch command + Colab instructions |
| Evaluation note | Added the critical `shuffle=False` warning |
| Project Files | Added `best_model.keras` entry |
| License | Added medical disclaimer |
| Acknowledgments | Added dataset and paper citations |
