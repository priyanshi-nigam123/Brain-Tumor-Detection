import io
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

# ---- Config (must match training) ----
IMAGE_SIZE = 128
CLASS_LABELS = ['pituitary', 'notumor', 'meningioma', 'glioma']  # same order as training

HF_REPO_ID = "Priyanshii123/brain-tumor-model"
HF_FILENAME = "model.keras"

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")


# ============================================
# ✅ FIX #1: ADD AUGMENTATION FUNCTION
# ============================================
def augment_image(image):
    """Apply same augmentation as training pipeline"""
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.8, 1.2))
    image = np.array(image) / 255.0
    return image


@st.cache_resource
def get_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return load_model(model_path, compile=False)


model = get_model()

st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI scan to classify it as **no tumor**, **glioma**, "
    "**meningioma**, or **pituitary** tumor. Model: fine-tuned VGG16."
)
st.caption("⚠️ For educational/demo purposes only — not a medical diagnostic tool.")

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save to a temp path so we can use keras' load_img exactly like the notebook did
    temp_path = "temp_upload.png"
    display_image = Image.open(uploaded_file).convert("RGB")
    display_image.save(temp_path)

    st.image(display_image, caption="Uploaded MRI", use_container_width=True)

    # ============================================
    # ✅ FIX #2: APPLY AUGMENTATION
    # ============================================
    img = load_img(temp_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    
    # Apply same augmentation as during training
    img_array = augment_image(img_array)  # ✅ THIS WAS MISSING!
    
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = predictions[predicted_index] * 100

    if predicted_label == "notumor":
        st.success(f"**No Tumor Detected** (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"**Tumor Detected: {predicted_label}** (Confidence: {confidence:.2f}%)")

    st.subheader("All class probabilities")
    for label, score in zip(CLASS_LABELS, predictions):
        display_label = "No Tumor" if label == "notumor" else f"Tumor: {label}"
        st.write(f"{display_label}: {score*100:.2f}%")
        st.progress(float(score))
        display_label = "No Tumor" if label == "notumor" else f"Tumor: {label}"
        st.write(f"{display_label}: {score*100:.2f}%")
        st.progress(float(score))
