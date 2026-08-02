# 🧠 Brain Tumor MRI Classification using EfficientNetB0 and TensorFlow

## Overview

This project is a deep learning-based medical image classification system designed to detect and classify brain tumors from MRI scans. It leverages transfer learning with **EfficientNetB0** to classify MRI images into four distinct categories with high accuracy. 

The model is deployed via a **Streamlit** web application, allowing users to upload MRI images and receive real-time predictions along with confidence scores.

---

## Dataset

**Source:** Kaggle – Brain Tumor MRI Dataset

- **Size:** ~7,000 MRI images
- **Classes:** 4 distinct categories
- **Preprocessing:** Resized to 224×224 pixels, converted to 3-channel RGB for EfficientNet compatibility.

### Classes

| Label | Description |
| --- | --- |
| **Glioma** | Glioma Tumor |
| **Meningioma** | Meningioma Tumor |
| **Pituitary** | Pituitary Tumor |
| **No Tumor** | Healthy Brain MRI |

---

## Model Architecture

The model uses a hybrid approach: a frozen/pre-trained feature extractor followed by a custom, regularized classification head.

- **Base Model:** EfficientNetB0 (Pretrained on ImageNet, excluding top layers)
- **Custom Classification Head:**
  1. `GlobalAveragePooling2D`
  2. `Dropout` (rate = 0.3)
  3. `Dense` (128 neurons, ReLU activation)
  4. `Dropout` (rate = 0.2)
  5. `Dense` (64 neurons, ReLU activation) *(Optimized for reduced overfitting)*
  6. `BatchNormalization`
  7. `Dropout` (rate = 0.3)
  8. `Dense` (4 neurons, Softmax activation)

---

## Techniques Used

- **Transfer Learning:** Leveraging pre-trained ImageNet weights for robust feature extraction.
- **Two-Phase Training Strategy:** Feature extraction followed by targeted fine-tuning.
- **Architectural Regularization:** Compressed Dense layers (128 → 64) and strategic Dropout to prevent overfitting on medical data.
- **Data Augmentation:** Rotation, zoom, width/height shifts, and horizontal flipping via `ImageDataGenerator`.
- **Callbacks:** `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`.
- **Domain-Specific Preprocessing:** EfficientNet's native `preprocess_input` function.

---

## Training Strategy

### Phase 1: Feature Extraction (Warm-up)
- **Base Model:** Entire `EfficientNetB0` frozen (`trainable = False`).
- **Optimizer:** Adam with learning rate = `1e-3`.
- **Duration:** 10 epochs.
- **Goal:** Train only the randomly initialized custom Dense head without destroying pre-trained weights (prevents catastrophic forgetting).

### Phase 2: Fine-Tuning
- **Base Model:** Unfrozen, but only the **last 20 layers** are set to `trainable = True`.
- **Optimizer:** Adam with a significantly reduced learning rate = `1e-5`.
- **Duration:** 20 epochs.
- **Goal:** Gently adapt the high-level, task-specific features of EfficientNet to MRI textures while the custom head fine-tunes alongside it.

---

## Performance

*(Note: Please verify these exact numbers match your latest training run after the Dense(64) update)*

| Metric | Score |
| --- | --- |
| **Training Accuracy** | ~99.0% |
| **Validation Accuracy** | ~95.0% |
| **Validation Loss** | ~0.28 |

The model achieves strong generalization performance with a minimized train-validation gap, demonstrating effective regularization for a relatively small medical dataset.

---

## Technologies Used

- **Core:** Python, TensorFlow, Keras
- **Architecture:** EfficientNetB0, NumPy, Pillow
- **Deployment:** Streamlit
- **Visualization:** Matplotlib, Seaborn

---

## Features

- 🖼️ Upload MRI images through an intuitive web interface.
- ⚡ Real-time, low-latency predictions.
- 📊 Confidence score display for all 4 classes.
- 🧠 Multi-class brain tumor classification.
- 🚀 Lightweight and computationally efficient architecture.

---

## Project Files

- `Brain_Tumor_MRI_using_transfer_learning_.ipynb` - Complete Jupyter Notebook with training, fine-tuning, and evaluation code.
- `app.py` - Streamlit application entry point.
- `brain_tumor_classifier.keras` - Trained TensorFlow Keras model weights.
- `class_indices.json` - Class label mapping used for prediction output.
- `requirements.txt` - Python dependencies required to run the app.

---

## Setup & Installation

1. Clone or download the project repository.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
