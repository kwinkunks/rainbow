# -*- coding: utf-8 -*-
"""
Simple application to get data from images.

by Matt Hall, Agile Geoscience, 2016
"""
from io import BytesIO
import base64

from flask import Flask
from flask import make_response
from flask import request, jsonify, render_template

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from errors import InvalidUsage

application = Flask(__name__)

@application.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

#
# Rainbow.
#
@application.route('/rainbow')
def rainbow():
    # Params from inputs.
    url = request.args.get('url')

    # Get image.
    requests_something = True
    
    # Unweave the rainbow.
    result = image_to_data(img)

    pass


@application.route('/')
def main():
    return render_template('index.html', title='Home')

if __name__ == "__main__":
    application.debug = True
    application.run()
