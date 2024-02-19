from flask import Blueprint

main = Blueprint('controllers', __name__)

from . import routes
