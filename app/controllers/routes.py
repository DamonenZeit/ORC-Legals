import os
from flask import render_template, request, redirect, url_for, jsonify, Flask

import app
from . import main
from core.service.classification_legals import classification_legals_service
from core.service.rotation_crop_identification import prediction
from core.load_models.yolo_labeling import detector, model_insight


@main.route('/')
@main.route('/index')
def index():
    return render_template('index.html')


@main.route('/test', methods=['POST'])
def test():
    # Assuming you send the person's name and role in the request
    message = request.form.get('message')
    print(message)
    return jsonify({'status': 'Success', 'message': f'{message} chao xìn'})


@main.route('/test_image', methods=['POST'])
def test_file():
    image_base64 = request.json.get('image_base64')
    return classification_legals_service(image_base64)


@main.route('/legals', methods=['POST'])
def test_predict():
    try:
        image_base64 = request.json.get('image_base64')
        return prediction(image_base64=image_base64, model=model_insight, vietOCR_detector=detector)
    except Exception as e:
        print(e)
        return jsonify({'status': 'Error', 'message': str(e)})