# -*- coding: utf-8 -*-
"""
Simple application to get data from images.

by Matt Hall, Agile Geoscience, 2016
"""
from io import BytesIO
import base64
import urllib
import requests
import uuid

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
import mycarta_imtools as mci

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
@application.route('/api')
def api():
    params = {}
    result = {}
    crop = []
    find_data = False
    success = True
    m = 'Thanks for using keats!'

    # Params from inputs.
    params['url'] = request.args.get('url')
    params['n_colours'] = request.args.get('n_colours') or '128'
    params['interval'] = request.args.get('interval') or '0,1'
    params['region'] = request.args.get('region') or 'all'
    params['recover'] = request.args.get('recover') or ''
    params['format'] = request.args.get('format') or 'PNG'
    params['return_cmap'] = request.args.get('return_cmap') or ''

    # Condition parameters.
    params['n_colours'] = int(params['n_colours'])
    params['recover'] = False if params['recover'].lower() in ['false', 'no', '0'] else True
    params['return_cmap'] = True if params['return_cmap'].lower() in ['true', 'yes', '1'] else False
    params['interval'] = [int(n) for n in params['interval'].split(',')]
    if params['region'].lower() == 'auto':
        find_data = True
    elif params['region'] is not '':
        try:
            crop = [int(n) for n in params['region'].split(',')]
        except:
            crop = []
    else:
        pass

    # Fetch and crop image.
    try:
        r = requests.get(params['url'])
        img = Image.open(BytesIO(r.content))
    except Exception:
        result['status'] = 'failed'
        m = 'Error. Unable to open image from target URI. '
        result['message'] = m
        return jsonify(result)

    if crop:
        try:
            img = img.crop(region)
        except Exception:
            m = 'Improper crop parameters. '
            raise InvalidUsage(m+crop, status_code=410)

    if find_data:
        img = mci.find_data(img)

    recover = params['recover']
    if utils.is_greyscale(img):
        success = False
        m = "The image appears to be greyscale already."
        recover = False

    # Unweave the rainbow.
    if recover:
        data, cmap = utils.image_to_data(img,
                                         n_colours=params['n_colours'],
                                         interval=params['interval'])
        imgout = Image.fromarray(np.uint8(data*255))
    else:
        data = np.asarray(img)[..., :3] / 255.
        cmap = np.array([])
        imgout = img
    # except Exception:
    #     result['status'] = 'failed'
    #     m = 'Error. There was a problem converting this image. '
    #     result['message'] = m
    #     return jsonify(result)


    databytes = BytesIO()
    if params['format'].lower() in ['numpy', 'npy', 'np', 'array', 'ndarray', 'bin', 'binary']:
        params['format'] = 'NumPy binary'
        ext = 'npy'
        np.save(databytes, data)
    elif params['format'].lower() in ['text', 'txt', 'ascii', 'utf8']:
        params['format'] = 'NumPy text'
        ext = 'txt'
        np.savetxt(databytes, data)
    elif params['format'].lower() in ['png', 'jpg', 'jpeg', 'tiff']:
        if params['format'] == 'jpg':
            ext = 'jpg'
            params['format'] = 'jpeg'
        elif params['format'] == 'jpeg':
            ext = 'jpg'
        else:
            ext = params['format'].lower()
        imgout.save(databytes, params['format'])
        params['format'] = params['format'].upper() + ' image'
    else:
        result['status'] = 'failed'
        m = 'Error. Target format not recognized. '
        result['message'] = m
        return jsonify(result)

    databytes.seek(0)
    uuid1 = str(uuid.uuid1())
    file_link = utils.get_url(databytes, ext, uuid1)

    result['parameters'] = params
    result['uuid'] = uuid1
    result['message'] = m
    if success:
        result['status'] = 'success'
    else:
        result['status'] = 'failed'

    result['result'] = {}
    result['result']['image'] = file_link

    if params['recover']:
        result['result']['cmap'] = cmap.tolist() if params['return_cmap'] else []
        result['result']['colours'] = cmap.shape[0]

    #return utils.serve_pil_image(imgout)
    return jsonify(result)

@application.route('/')
def main():
    return render_template('index.html', title='Home')

if __name__ == "__main__":
    application.debug = True
    application.run()
