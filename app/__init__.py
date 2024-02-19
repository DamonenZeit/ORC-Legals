from flask import Flask

from app.controllers import main as main_blueprint

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'bnkhcm_ocr_legal_documents'


app.register_blueprint(main_blueprint)