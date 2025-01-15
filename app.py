import streamlit as st
import os
import numpy as np
from tensorflow import keras
from PIL import Image
import tensorflow as tf

import requests
import re
from urllib.parse import parse_qs, urlparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os


def download_file_from_google_drive(url, output_path=None):
    """
    Download a public file from Google Drive using its URL.

    Args:
        url (str): The Google Drive file URL
        output_path (str, optional): Path where the file should be saved.
            If not provided, saves in current directory with original filename.

    Returns:
        str: Path to the downloaded file

    Raises:
        ValueError: If the URL is invalid or file is not publicly accessible
        ConnectionError: If there are network issues
        IOError: If there are issues writing the file
    """
    try:
        if 'drive.google.com' not in url:
            raise ValueError("Not a valid Google Drive URL")

        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            parsed = urlparse(url)
            file_id = parse_qs(parsed.query)['id'][0]
        else:
            raise ValueError("Could not extract file ID from URL")

        download_url = f"https://drive.google.com/uc?id={file_id}"
        session = requests.Session()

        response = session.get(download_url, stream=True)
        response.raise_for_status()

        if "login.google.com" in response.url:
            raise ValueError("File is not publicly accessible")

        if 'content-disposition' in response.headers:
            filename = re.findall("filename=(.+)",
                                  response.headers['content-disposition'])[0]
        else:
            filename = f"gdrive_file_{file_id}"

        if output_path is None:
            output_path = filename.strip('"')

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        with open(output_path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)

        return output_path

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Network error occurred: {str(e)}")
    except IOError as e:
        raise IOError(f"Error writing file: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


MODEL_PATH = 'model.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

st.set_page_config(
    page_title="Medical Disease Classification System", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        padding: 2rem;
    }

    .stTitle {
        color: #1e293b;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 1rem !important;
    }

    .subtitle {
        color: #475569;
        font-size: 1.1rem;
        text-align: center;
        margin: 0 auto;
        line-height: 1.6;
    }

    .disease-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
        margin: 1.5rem 0;
    }

    .disease-tag {
        background-color: #e2e8f0;
        padding: 0.8rem 1.5rem;
        border-radius: 20px;
        color: #475569;
        min-width: 120px;
        text-align: center;
    }

    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 2rem;
    }

    .accuracy-bar {
        height: 8px;
        background-color: #e2e8f0;
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }

    .accuracy-fill {
        height: 100%;
        background-color: #2563eb;
        transition: width 1s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)


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
        url = "https://drive.google.com/file/d/1zU1Nqd0KaOR5Hn-e6LTeU4Dd0XUWDX4W/view?usp=drive_link"
        try:
            file_path = download_file_from_google_drive(url)
            print(f"File downloaded successfully to: {file_path}")
            with open(file_path, 'r') as file:
                html_content = file.read()
                # print(html_content)

                soup = BeautifulSoup(html_content, 'html.parser')

                form = soup.find('form')
                action_url = form['action']
                data = {input_tag['name']: input_tag['value'] for input_tag in form.find_all(
                    'input') if input_tag.get('name')}

                response = requests.get(action_url, params=data)

                if response.status_code == 200:
                    with open(MODEL_PATH, 'wb') as file:
                        file.write(response.content)
                    print("File downloaded successfully!")
                else:
                    print(
                        f"Failed to download. Status code: {response.status_code}")

        except Exception as e:
            print(f"Error: {e}")
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        with open("error.txt", "a") as error_file:
            error_file.write(f"Error loading model: {str(e)}\n")
        return None


def main():
    st.logo("swe.png")
    st.markdown("""<h1 class="stTitle">
                    Medical Disease Classification System
                </h1>""",
                unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">Advanced AI-powered system for identifying and classifying various skin conditions. Upload or capture an image for instant analysis.</h3>', unsafe_allow_html=True)

    diseases = ['Chickenpox', 'Cowpox',
                'HFMD', 'Healthy', 'Measles', 'Monkeypox']
    disease_tags = '<div class="disease-tags">' + \
        ''.join(
            [f'<span class="disease-tag">{disease}</span>' for disease in diseases]) + '</div>'
    st.markdown(disease_tags, unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.error("Model could not be loaded. Please check the logs.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style='text-align: center; padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);'>
                <i class="fas fa-upload" style='font-size: 2.5rem; color: #2563eb;'></i>
                <h3 style='font-size: 1.2rem; margin: 1rem 0;'>Upload Image</h3>
                <p style='color: #475569;'>Select an image from your device</p>
            </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=list(
            ALLOWED_EXTENSIONS), key="file_uploader", label_visibility="collapsed")

    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);'>
                <i class="fas fa-camera" style='font-size: 2.5rem; color: #2563eb;'></i>
                <h3 style='font-size: 1.2rem; margin: 1rem 0;'>Take Photo</h3>
                <p style='color: #475569;'>Use your device's camera</p>
            </div>
        """, unsafe_allow_html=True)
        camera_photo = st.camera_input(
            "Take a photo", label_visibility="collapsed")

    if uploaded_file is not None or camera_photo is not None:
        image = Image.open(uploaded_file if uploaded_file else camera_photo)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Uploaded Image', use_container_width=True)

        try:
            with st.spinner('Processing...'):
                processed_image = preprocess_image(image)

                predictions = model.predict(processed_image)

                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                predicted_disease = diseases[predicted_class_idx]

                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">
                            <div style="background-color: #059669; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;">
                                <i class="fas fa-check"></i>
                            </div>
                            <div>
                                <h3 style="font-size: 1.2rem; margin: 0;">Analysis Result</h3>
                                <p style="color: #475569;">Classification complete</p>
                            </div>
                        </div>
                        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; margin-top: 1rem;">
                            <div style="font-size: 1.4rem; font-weight: 600; text-align: center; margin-bottom: 1rem;">{predicted_disease}</div>
                            <p>The image has been analyzed and classified with high confidence.</p>
                            <div class="accuracy-bar">
                                <div class="accuracy-fill" style="width: {confidence * 100}%;"></div>
                            </div>
                            <div style="text-align: right; color: #475569; font-size: 0.9rem;">
                                Confidence Score: {confidence:.2%}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # st.subheader("Detailed Classification Results:")
                # results = {disease: float(
                #     predictions[0][i]) for i, disease in enumerate(diseases)}
                # st.bar_chart(results)

                # for disease, probability in results.items():
                #     st.write(f"{disease}: {probability:.2%}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    st.markdown(
        f"""<p class="subtitle">© {datetime.now().year} Daffodil International University, Department of Software Engineering. All rights reserved</p>""", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
