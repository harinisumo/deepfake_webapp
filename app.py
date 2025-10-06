import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# --- CONFIG ---
MODEL_PATH = "efficient_bo_best.keras"   # Local model file
IMAGE_SIZE = (128, 128)      # Set this to your training input size
FILE_ID ="1FfqPPnJUlRMmztkQmSp7QA_TW0BTM4Bg"  # Only if downloading model from Google Drive
# ---------------

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Image Detection")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).resize(IMAGE_SIZE)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    pred = model.predict(img_array)[0][0]

    label = "FAKE" if pred > 0.5 else "REAL"
    st.markdown(f"### Prediction: ✅ **{label}**")
