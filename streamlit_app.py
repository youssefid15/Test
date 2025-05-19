import streamlit as st
from PIL import Image

st.title("Test image upload")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.write("Upload received")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image")
