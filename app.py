from __future__ import division, print_function
# coding=utf-8
from keras.models import load_model
from tensorflow.keras.preprocessing import image


import sys
import os
import glob
import re
import numpy as np
import random

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from PIL import  Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import model
# Define a flask app
app = Flask(__name__)
account_sid = ''
auth_token = ''

print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
   
    return render_template('index.html')
def process_image(file):
    image = Image.open(file)
    image = image.resize((64, 64))
    image = np.array(image)
    if len(image.shape) == 2:  # if image has only one channel, repeat along the channel axis
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    else:
        image = image[:, :, :3]  # take only the first 3 channels if there are more than 3
    image = image / 255.0
    return image[np.newaxis, ...]

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
        image=process_image(file_path)
        tempr=random.randint(50,101)
        fire = model.predict(file_path)
        print(fire)
        if fire:
            if tempr>70: 
                prediction = 'fire'           
            else:
                prediction = 'fire'
        else:
            prediction = 'no fire'

        return prediction
  
    return None


if __name__ == '__main__':
    app.run(port=5000, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
