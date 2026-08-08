import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C3AED, #4F46E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.6rem;
    }
    .upload-card, .preview-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 1.4rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #EEF2FF, #F5F3FF);
        border: 1px solid #C7D2FE;
        border-left: 6px solid #7C3AED;
        margin-bottom: 1rem;
    }
    .prediction-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #4C1D95;
    }
    .confidence-text {
        font-size: 1rem;
        color: #7C3AED;
        font-weight: 700;
    }
    .class-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 600;
        color: #374151;
        margin-top: 0.6rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7C3AED, #4F46E5);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 700;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: white;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7C3AED, #4F46E5);
    }
    section[data-testid="stSidebar"] {
        background-color: #FAF5FF;
    }
    </style>
""", unsafe_allow_html=True)

HF_REPO_ID = "Priyanshii123/Brain_Tumor_Detection"
HF_MODEL_FILENAME = "brain_tumor_efficientnet_model.keras"

IMAGE_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
VALID_EXT = (".jpg", ".jpeg", ".png")

CLASS_INFO = {
    "glioma": "A tumor that arises from glial cells in the brain or spine.",
    "meningioma": "A tumor that forms on membranes covering the brain and spinal cord.",
    "notumor": "No tumor detected in the scan.",
    "pituitary": "A tumor that forms in the pituitary gland."
}

CLASS_COLORS = {
    "glioma": "#DC2626",
    "meningioma": "#D97706",
    "notumor": "#16A34A",
    "pituitary": "#2563EB"
}

EXAMPLE_DIR = "example_images"

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image: Image.Image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    return CLASS_NAMES[pred_idx], preds

def safe_open(path_or_file):
    img = Image.open(path_or_file)
    img.load()
    return img.convert("RGB")

with st.sidebar:
    st.markdown("### 🧠 About")
    st.write(
        "This app classifies brain MRI scans into four categories using a "
        "fine-tuned **EfficientNetB0** model."
    )
    st.markdown("---")

    st.markdown("### 📊 Classes")
    for cname, desc in CLASS_INFO.items():
        st.markdown(f"**{cname.capitalize()}** — {desc}")

    st.markdown("---")
    st.markdown("### 🖼️ Try an Example")
    st.caption("Click a thumbnail to test the model without uploading your own image.")

    selected_example = None
    for cname in CLASS_NAMES:
        class_dir = os.path.join(EXAMPLE_DIR, cname)
        if os.path.isdir(class_dir):
            files = [f for f in sorted(os.listdir(class_dir)) if f.lower().endswith(VALID_EXT)][:5]
            if files:
                st.markdown(f"**{cname.capitalize()}**")
                cols = st.columns(5)
                for i, fname in enumerate(files):
                    fpath = os.path.join(class_dir, fname)
                    with cols[i]:
                        try:
                            if st.button("▫", key=f"{cname}_{i}", help=fname):
                                selected_example = fpath
                            st.image(fpath, width="stretch")
                        except Exception:
                            pass

st.markdown('<div class="main-header">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a brain MRI scan or pick an example from the '
    'sidebar to classify it as Glioma, Meningioma, Pituitary tumor, or No tumor.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="large")

image_to_predict = None

with col1:
    st.markdown('<div class="section-title">Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    try:
        if uploaded_file is not None:
            image_to_predict = safe_open(uploaded_file)
        elif selected_example is not None:
            image_to_predict = safe_open(selected_example)
    except Exception as e:
        st.error(f"Couldn't read this image: {e}")
        image_to_predict = None

    if image_to_predict is not None:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.image(image_to_predict, caption="Selected Scan", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        run = st.button("✨ Analyze Scan", width="stretch")
    else:
        st.info("Upload an image above or select an example from the sidebar to begin.")
        run = False

with col2:
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    if image_to_predict is not None and run:
        with st.spinner("Analyzing scan..."):
            model = load_model()
            pred_class, probs = predict(model, image_to_predict)
            confidence = float(np.max(probs)) * 100

        display_label = "No Tumor" if pred_class == "notumor" else pred_class.capitalize()
        st.markdown(f"""
            <div class="result-card">
                <div class="prediction-title">{display_label}</div>
                <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                <p style="margin-top:0.5rem; color:#4B5563;">{CLASS_INFO[pred_class]}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Confidence by Class")
        for cname, p in zip(CLASS_NAMES, probs):
            label = "No Tumor" if cname == "notumor" else cname.capitalize()
            st.markdown(
                f'<div class="class-row"><span>{label}</span><span>{p*100:.2f}%</span></div>',
                unsafe_allow_html=True
            )
            st.progress(float(p))
    else:
        st.write("Prediction results will appear here after you analyze a scan.")

st.markdown("---")
