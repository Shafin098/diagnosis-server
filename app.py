import streamlit as st
import os
import numpy as np
from tensorflow import keras
from PIL import Image
import tensorflow as tf

MODEL_PATH = './model.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model inference.
    Converts RGBA to RGB if needed and resizes to target size.
    """
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(target_size)
    img_array = np.array(image)

    if len(img_array.shape) != 3 or img_array.shape[-1] != 3:
        raise ValueError(
            f"Image has incorrect number of channels: {img_array.shape}")

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    assert img_array.shape == (
        1, 224, 224, 3), f"Incorrect shape: {img_array.shape}"

    return img_array


@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        with open("error.txt", "a") as error_file:
            error_file.write(f"Error loading model: {str(e)}\n")
        return None


def main():
    st.title("Disease Classification App")
    st.write("Upload an image to classify the disease")

    model = load_model()

    if model is None:
        st.error("Model could not be loaded. Please check the logs.")
        return

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=list(ALLOWED_EXTENSIONS)
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        if st.button('Classify Disease'):
            try:
                processed_image = preprocess_image(image)

                predictions = model.predict(processed_image)

                results = {
                    'Chickenpox': float(predictions[0][0]),
                    'Cowpox': float(predictions[0][1]),
                    'HFMD': float(predictions[0][2]),
                    'Healthy': float(predictions[0][3]),
                    'Measles': float(predictions[0][4]),
                    'Monkeypox': float(predictions[0][5])
                }

                st.subheader("Classification Results:")

                st.bar_chart(results)

                for disease, probability in results.items():
                    st.write(f"{disease}: {probability:.2%}")

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")


if __name__ == '__main__':
    main()
