import streamlit as st
# import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.16.1"])
    import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit UI
st.title("CIFAR-10 Image Classifier 🚀")
st.write("Upload an image (32x32) and the model will predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=200)

    # Preprocess the image (resize to 32x32, normalize)
    image = image.resize((32, 32))
    image = np.array(image) / 255.0  # Normalize

    # Ensure 3 channels (RGB) even if grayscale
    if image.shape[-1] != 3:
        image = np.stack((image.squeeze(),)*3, axis=-1)

    # Add batch dimension (model expects [batch, 32, 32, 3])
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f}%)")