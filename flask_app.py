from flask import Flask, render_template, request, redirect, url_for

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import re
import base64
import matplotlib.pyplot as plt

json_file = open('/home/steve/Desktop/digit/flask_api/model.json','r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("/home/steve/Desktop/digit/flask_api/weights.h5")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
graph = tf.Graph()

def prediction(path):
    x = cv2.imread(path,0)
    x = cv2.resize(x, (28, 28))
    # plt.imshow(x.reshape(28,28),cmap=plt.cm.binary)
    # plt.show()
    x = x.reshape(1,28,28,1)
    x = np.array(x).astype('float32')
    out = model.predict(x)
    # print(np.argmax(out, axis=1))
    # convert the response to a string
    response = np.argmax(out, axis=1)
    return str(response[0])

def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1))
    imgstr = imgstr.group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)

@app.route('/predict/')
def saket():
    return render_template('predict.html')

@app.route('/p')
def ans():
    path = '/home/steve/Desktop/digit/flask_api/output.png'
    ans = prediction(path)
    return render_template('p.html', title=ans)

if __name__ == "__main__":
    app.run()
    # run the app locally on the given port
# optional if we want to run in debugging mode
