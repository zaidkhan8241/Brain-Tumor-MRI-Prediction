# 🧠 Brain Tumor MRI Classification using EfficientNetB0 and TensorFlow

## Overview

This project is a deep learning-based medical image classification system designed to detect and classify brain tumors from MRI images. It uses transfer learning with EfficientNetB0 to classify MRI scans into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The model is deployed through a Streamlit web application, allowing users to upload MRI images and receive predictions along with confidence scores.

---

## Dataset

**Source:** Kaggle – Brain Tumor MRI Dataset

- Approximately 7,000 MRI images
- 4 classes
- RGB images resized to 224×224

### Classes

| Label | Description |
| --- | --- |
| Glioma | Glioma Tumor |
| Meningioma | Meningioma Tumor |
| Pituitary | Pituitary Tumor |
| No Tumor | Healthy Brain MRI |

---

## Model Architecture

- EfficientNetB0 (Pretrained on ImageNet)
- GlobalAveragePooling2D
- Dropout (0.3)
- Dense Layer (128 neurons, ReLU)
- Dropout (0.2)
- Dense Output Layer (4 neurons, Softmax)

---

## Techniques Used

- Transfer Learning
- Fine-Tuning of the last 20 layers
- Data Augmentation
- Early Stopping
- ReduceLROnPlateau
- Model Checkpointing
- EfficientNet Preprocessing
- TensorFlow/Keras
- Streamlit Deployment

---

## Training Strategy

### Phase 1: Feature Extraction

- EfficientNetB0 frozen
- Adam optimizer with learning rate = 1e-3
- Trained for 10 epochs

### Phase 2: Fine-Tuning

- Last 20 layers unfrozen
- Adam optimizer with learning rate = 1e-5
- Trained for 20 epochs

---

## Performance

| Metric | Score |
| --- | --- |
| Training Accuracy | 99.18% |
| Validation Accuracy | 95.00% |
| Validation Loss | 0.2845 |

The model achieved strong generalization performance with minimal overfitting, making it suitable for medical image classification tasks.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pillow
- Streamlit
- Matplotlib

---

## Features

- Upload MRI images through a web interface
- Real-time prediction
- Confidence score display
- Multi-class brain tumor classification
- Lightweight and efficient model
- Transfer learning-based architecture

---

## Results

The model successfully classifies brain MRI images into four categories with approximately **95% validation accuracy**. Transfer learning with EfficientNetB0 and fine-tuning of the last layers significantly improved performance while maintaining computational efficiency.

---

## Project Files

- `app.py` - Streamlit application entry point
- `brain_tumor_classifier.keras` - trained TensorFlow Keras model
- `class_indices.json` - class label mapping used for prediction output
- `requirements.txt` - Python dependencies required to run the app

---

## Setup

1. Clone or download the project.
2. Create and activate a Python virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Run the Application

From the project folder, launch Streamlit:

```bash
streamlit run app.py
```

Then open the local URL provided by Streamlit in your browser.

---

## Notes

- The model expects images resized to `224x224` pixels.
- Grayscale input images are converted to RGB.
- Images with alpha channels are trimmed to RGB.

---

Developed as a brain tumor classification demo using TensorFlow, EfficientNetB0, and Streamlit.
