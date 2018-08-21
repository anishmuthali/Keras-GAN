from flask import Flask, send_file
from keras.models import *
import numpy as np
import os
from PIL import Image
import time
import tensorflow as tf
import keras.backend as K

app = Flask(__name__)

model = load_model("batch_25501.h5")
graph = tf.get_default_graph()

def predict():
    rand_val = np.random.uniform(-1.0, 1.0, size=(1,100))
    with graph.as_default():
        img = model.predict(rand_val)
    return img

@app.route("/gen_img")
def generate_image():
    img = predict()
    img = np.reshape(img, (28,28)) * 255
    img = np.array(img, dtype='uint8')
    filename = str((int)(time.time()*100)) + ".png"
    img = Image.fromarray(img)
    img.save(filename)
    return send_file(filename, mimetype='image/png')

    
if __name__ == "__main__":
    app.run(threaded=True)
