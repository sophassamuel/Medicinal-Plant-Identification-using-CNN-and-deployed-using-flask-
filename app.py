import os
import io
import pickle
import numpy as np
from PIL import Image
import base64
from flask import Flask, jsonify, request, render_template
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

def init_app(self, app):
    app.template_folder = "views"
    app.config.setdefault("FLASK_MVC_DIR", "app")

    self.root_path = Path(app.config["FLASK_MVC_DIR"])

    self._register_router(app)


def load_ml_model():
    global model
    model = load_model("C:/Users/sopha/Projects/flaskapp/MedicinalPlant.h5")
    print("Model loaded.")


def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    resized = tf.image.resize(image, target_size)
    padded = tf.image.pad_to_bounding_box(
        resized, 0, 0, target_size[0], target_size[1])
    image = img_to_array(padded)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['img']
    encoded = message.split(',')[1]
    decoded = io.BytesIO(base64.b64decode(encoded))
    image = Image.open(decoded)
    processed_image = prepare_image(image, target_size=(256, 256))
    prediction = model.predict(processed_image)
    result = np.argmax(prediction, axis=1)
    pred = class_dict[result[0]]
    return render_template('result.html', prediction=pred)


if __name__ == '__main__':
    load_ml_model()
    app.run(debug=True, port=os.environ.get('PORT', 5000))
