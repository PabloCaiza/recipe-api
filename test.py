import tensorflow as tf

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import boto3
import json




PATH_TO_SAVED_MODEL = "saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index={
 1: {'id': 1, 'name': 'egg'},
 2: {'id': 2, 'name': 'cheese'}, 
 3: {'id': 3, 'name': 'milk'},
 4: {'id': 4, 'name': 'lemon'},
 5: {'id': 5, 'name': 'onion'}, 
 6: {'id': 6, 'name': 'garlic'}, 
 7: {'id': 7, 'name': 'potatoe'}, 
 8: {'id': 8, 'name': 'green banana'}, 
 9: {'id': 9, 'name': 'tomato'}, 
 10: {'id': 10, 'name': 'chicken'}}

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def detectIngredients(image):
    image_np = load_image_into_numpy_array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    indexItems = [index for index in range(len(detections['detection_scores'])) if
                  detections['detection_scores'][index] >= 0.6]
    categories = dict()
    for index in indexItems:
        if detections['detection_classes'][index] in categories:
            categories[detections['detection_classes'][index]] += 1
        else:
            categories[detections['detection_classes'][index]] = 1
    final_categories = dict()
    for c in categories:
        for ci in category_index:
            if c == category_index[ci]['id']:
                final_categories[category_index[ci]['name']] = categories[c]
    return final_categories

app = Flask(__name__)


@app.route('/predictIngredients', methods=['POST'])
def predictIngredientes():
    uploaded_file = request.files['file']
    print(uploaded_file)
    ingredients = detectIngredients(uploaded_file)
    print(ingredients)
    diccionario = {
        "status": 200,
        "data": ingredients
    }
    return jsonify(diccionario)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
    print(__name__)