# -*- coding: utf-8 -*-
"""
Simple application to get data from images.

by Matt Hall, Agile Geoscience, 2016
"""
from io import BytesIO
import base64
import urllib
import requests

from flask import Flask
from flask import make_response
from flask import request, jsonify, render_template
from flask import send_file

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from errors import InvalidUsage
import utils

#
# Set up.
#
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
    ncolours = request.args.get('ncolours') or '128'
    interval = request.args.get('interval') or '0,1'
    region = request.args.get('region')

    # Condition parameters.
    n_colours = int(ncolours)
    interval = [int(n) for n in interval.split(',')]
    if region:
        region = [int(n) for n in region.split(',')]
    else:
        region = []

    # Fetch and crop image.
    try:
        r = requests.get(url)
        im = Image.open(BytesIO(r.content))
    except Exception:
        result = {'job_uuid': uuid.uuid1()}
        result['status'] = 'failed'
        m = 'Error. Unable to open image from target URI. '
        result['message'] = m
        result['parameters'] = utils.build_params(method, avg,
                                                  t_min, t_max,
                                                  region,
                                                  trace_spacing,
                                                  url=url)
        return jsonify(result)

    if region:
        try:
            im = im.crop(region)
        except Exception:
            m = 'Improper crop parameters '
            raise InvalidUsage(m+region, status_code=410)

    if utils.is_greyscale(im):
        m = "The image appears to be greyscale already."
        raise InvalidUsage(m, status_code=410)

    # Unweave the rainbow.
    result = utils.image_to_data(im, n_colours=n_colours, interval=interval)
    imgout = Image.fromarray(np.uint8(result*255))

    return utils.serve_pil_image(imgout)


@application.route('/')
def main():
    return render_template('index.html', title='Home')

if __name__ == "__main__":
    application.debug = True
    application.run()
