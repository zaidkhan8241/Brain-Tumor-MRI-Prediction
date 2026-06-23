import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_my_model():
    model = load_model("brain_tumor_classifier.keras")
    return model

model = load_my_model()

# ---------------------------------------------------
# Load Class Names
# ---------------------------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Convert {"glioma":0,...} -> {0:"glioma",...}
class_names = {v: k for k, v in class_indices.items()}

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("🧠 Brain Tumor MRI Classification")
st.markdown(
    """
Upload an MRI image and the model will predict whether it belongs to:

- Glioma
- Meningioma
- No Tumor
- Pituitary

Model: **EfficientNetB0 + Transfer Learning**
"""
)

# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
def predict_image(image):

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(image)

    # Convert grayscale image to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Remove alpha channel if present
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return predicted_class, confidence, prediction[0]

# ---------------------------------------------------
# Display Result
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    with st.spinner("Analyzing MRI Image..."):

        predicted_class, confidence, probabilities = predict_image(image)

        label = class_names[predicted_class]

        with col2:

            st.subheader("Prediction")

            st.success(f"Detected Class: {label.upper()}")

            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )

    st.markdown("---")

    st.subheader("Class Probabilities")

    for idx, prob in enumerate(probabilities):
        st.write(
            f"{class_names[idx].capitalize()} : {prob * 100:.2f}%"
        )
        st.progress(float(prob))

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption(
    "Developed using TensorFlow, EfficientNetB0, Transfer Learning, and Streamlit."
)