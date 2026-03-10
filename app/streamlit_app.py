# app/streamlit_app.py

import streamlit as st
import numpy as np
from PIL import Image

from inference import CrackModel, calculate_crack_score
from postprocess import overlay_mask


st.set_page_config(page_title="Concrete Crack Detection", layout="wide")

st.title("Concrete Crack Segmentation")
st.write("Upload a concrete surface image to detect cracks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return CrackModel()

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image_np, use_container_width=True)

    mask = model.predict(image)
    ratio, severity = calculate_crack_score(mask)

    overlay = overlay_mask(image_np, mask)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Crack Mask")
        st.image(mask, use_container_width=True)

    with col2:
        st.subheader("Crack Overlay")
        st.image(overlay, use_container_width=True)

    st.subheader("Crack Analysis")
    st.write(f"Crack Ratio: {ratio:.4f}")
    st.write(f"Severity Level: {severity}")