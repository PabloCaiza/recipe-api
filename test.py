import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from bson import json_util
from dotenv import load_dotenv
from google.cloud import translate
import json
import os

load_dotenv()
app = Flask(__name__)
PATH_TO_SAVED_MODEL = "saved_model"
project_id = os.getenv("PROJECT_ID", "")
parent = f"projects/{project_id}"
client = translate.TranslationServiceClient()

app.config["MONGO_URI"] = "mongodb+srv://{}:{}@cluster0.uwltrza.mongodb.net/mineria".format(os.getenv("MONGO_USERNAME"),
                                                                                            os.getenv("MONGO_PASSWORD"))
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = {
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


def translate_text(target_language_code, content):
    response = client.translate_text(contents=content,
                                     target_language_code=target_language_code,
                                     parent=parent)
    translated_items = []
    for translation in response.translations:
        translated_items.append(translation.translated_text)
    return translated_items


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


@app.route('/predictIngredients', methods=['POST'])
def predictIngredientes():
    # uploaded_file = request.files['file']
    ingredients = {}
    for file in request.files:
        uploaded_file = request.files[file]
        print(uploaded_file)
        ingredients_detected = detectIngredients(uploaded_file)
        ingredients.update(ingredients_detected)
    print(ingredients)
    recipes = mongo.db.recipes.find()
    response = json_util.dumps(recipes)
    recipes = json.loads(response)
    possibleRecipes = []
    for r in recipes:
        hasAnyIngredient = False
        count = 0
        for i in ingredients:
            if i in r["ingredients"]:
                hasAnyIngredient = True
                count = count + 1
        if hasAnyIngredient:
            r["countedElements"] = count
            possibleRecipes.append(r)

    response = sorted(possibleRecipes, key=lambda d: d['countedElements'], reverse=True)
    response = response[:10]
    for r in response:
        r['ingredients'] = translate_text('es', r['ingredients'])
    diccionario = {
        "status": 200,
        "data": response,
        "ingredients": translate_text('es', list(ingredients.keys()))
    }
    return jsonify(diccionario)


mongo = PyMongo(app)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    print(__name__)
