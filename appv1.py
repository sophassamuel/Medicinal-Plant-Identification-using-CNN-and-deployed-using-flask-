import os
import io
import pickle
import numpy as np
from PIL import Image
import base64
from flask import Flask, jsonify, request, render_template
from keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

app = Flask(__name__,
            template_folder='templates')
model = None

class_dict = ["Rasna",
              "Arive-Dantu",
              "Jackfruit",
              "Neem",
              "Basale",
              "Indian Mustard",
              "Karanda",
              "Lemon",
              "Roxburgh fig",
              "Peepal Tree",
              "Hibiscus",
              "Jasmine",
              "Mango",
              "Mint",
              "Drumstick",
              "Jamaica Cherry-Gasagase",
              "Curry",
              "Oleander",
              "Parijata",
              "Tulsi",
              "Betel",
              "Mexican Mint",
              "Indian Beech",
              "Guava",
              "Pomegranate",
              "Sandalwood",
              "Jamun",
              "Rose Apple",
              "Crape Jasmine",
              "Fenugreek"]


def load_ml_model():
    global model
    model = load_model("MedicinalPlant.h5")
    print("Model loaded.")


def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image = image.copy()  # make a copy of the input array
    image = preprocess_input(image)
    return image


@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index1.html', message='No file found')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No file selected')

    if file and allowed_file(file.filename):
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = prepare_image(image, target_size=(256, 256))
        prediction = model.predict(processed_image)
        result = np.argmax(prediction, axis=1)
        new = []
        output = []
        for i in range(len(result)):
            output.append(class_dict[result[i]])
        return render_template('result.html', new = output)

    else:
        return render_template('index1.html', message='Invalid file type')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


if __name__ == '__main__':
    load_ml_model()
    app.run(debug=True, port=os.environ.get('PORT', 5000))
