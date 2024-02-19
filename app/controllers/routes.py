import os
from flask import render_template, request, redirect, url_for, jsonify
from . import main
from app.service.test_file import test_file_service


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


@main.route('/test_file', methods=['POST'])
def test_file():
    file = request.files.get('file')
    return test_file_service(file)
