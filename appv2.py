import os
import io
import base64
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

class_dict = ["Rasna, It is a medicinal plant",
              "Arive-Dantu, It is a medicinal plant",
              "Jackfruit, It is a medicinal plant",
              "Neem, It is a medicinal plant",
              "Basale, It is a medicinal plant",
              "Indian Mustard, It is a medicinal plant",
              "Karanda, It is a medicinal plant",
              "Lemon, It is a medicinal plant",
              "Roxburgh fig, It is a medicinal plant",
              "Peepal Tree, It is a medicinal plant",
              "Hibiscus, It is a medicinal plant",
              "Jasmine, It is a medicinal plant",
              "Mango,It is a medicinal plant",
              "Mint, It is a medicinal plant",
              "Drumstick, It is a medicinal plant",
              "Jamaica Cherry-Gasagase, It is a medicinal plant",
              "Curry, It is a medicinal plant",
              "Oleander, It is a medicinal plant",
              "Parijata, It is a medicinal plant",
              "Tulsi, It is a medicinal plant",
              "Betel, It is a medicinal plant",
              "Mexican Mint, It is a medicinal plant",
              "Indian Beech, It is a medicinal plant",
              "Guava, It is a medicinal plant",
              "Pomegranate, It is a medicinal plant",
              "Sandalwood, Not a medicinal plant",
              "Jamun, It is a medicinal plant",
              "Rose Apple, It is a medicinal plant",
              "Crape Jasmine, It is a medicinal plant",
              "Fenugreek, It is a medicinal plant"]


def load_ml_model():
    global model
    model = load_model("MedicinalPlant.h5")
    print("Model loaded.")


def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image.copy())
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image = image.copy()  # make a copy of the input array
    image = preprocess_input(image)
    return image

@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html')


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
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        output = []
        for i in range(len(result)):
            output.append(class_dict[result[i]])
        return render_template('index2.html', new = output, image_data = image_base64)

    else:
        return render_template('index1.html', message='Invalid file type')
""" def imgview(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    im = Image.open(io.BytesIO(image))
    encoded = base64.b64encode(im.getvalue())
    return render_template('index2.html', image_data=)
 """
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


if __name__ == '__main__':
    load_ml_model()
    app.run(debug=True, port=os.environ.get('PORT', 5000))
