from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow import keras
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = './model.keras'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image for model inference.
    Converts RGBA to RGB if needed and resizes to target size.
    """
    img = Image.open(image_path)

    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(target_size)

    img_array = np.array(img)

    if len(img_array.shape) != 3 or img_array.shape[-1] != 3:
        raise ValueError(
            f"Image has incorrect number of channels: {img_array.shape}")

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    assert img_array.shape == (
        1, 224, 224, 3), f"Incorrect shape: {img_array.shape}"

    return img_array


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'photo' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['photo']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            processed_image = preprocess_image(filepath)

            predictions = model.predict(processed_image)
            predictionList = predictions.tolist()
            results = {
                'predictions': {
                    'Chickenpox': predictionList[0][0],
                    'Cowpox': predictionList[0][1],
                    'HFMD': predictionList[0][2],
                    'Healthy': predictionList[0][3],
                    'Measles': predictionList[0][4],
                    'Monkeypox': predictionList[0][5],
                },
                'filename': filename
            }

            os.remove(filepath)

            return jsonify(results)

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        with open("error.txt", "a") as error_file:  # "a" appends instead of overwriting
            error_file.write(f"Error loading model: {str(e)}\n")
        model = None

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
