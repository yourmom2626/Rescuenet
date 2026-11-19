
import tensorflow as tf
import numpy as np
import json

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("classifier.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

def predict_disease(image_path):

    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    predictions = model.predict(img, verbose=0)

    index = np.argmax(predictions[0])

    disease = class_names[index]
    confidence = float(predictions[0][index] * 100)

    return disease, confidence
