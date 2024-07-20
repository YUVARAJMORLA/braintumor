import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
# import cv2

# Load the trained model
model = load_model('tumor.h5', compile=False)

# Set up the Streamlit interface
st.title('Brain Tumor Detection')

st.write("Upload an MRI image to predict if it has a brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Make a prediction
    pred = np.argmax(model.predict(x), axis=1)
    result = "Yes" if pred[0] == 1 else "No"

    st.image(img, caption='Uploaded MRI Image', use_column_width=True)
    st.write("Prediction: ", result)

#pipreqs --encoding=utf8