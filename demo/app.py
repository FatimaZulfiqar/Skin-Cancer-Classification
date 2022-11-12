from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import cv2
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import  decode_predictions
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'E:/Python Projects/Skin cancer (disease) detection/Saved Model/Skin Disease classification using inceptionresnetv2.h5'

# Load your trained model
model = load_model(MODEL_PATH)


print('Model loaded. Check http://127.0.0.1:5000/')

classes = {
    0:'benign',
    1:'malignant'
}

def model_predict(img_path, model):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

    image = np.array(image)
    image = image.astype("float32")
    image /= 255
    image = np.expand_dims(image, axis=0)

    preds = model.predict_classes([image])[0]
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction

        preds = model_predict(file_path, model)
        sign = classes[preds]
        result = str(sign)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

