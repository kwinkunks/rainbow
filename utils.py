# -*- coding: utf-8 -*-
"""
Various functions.

by Matt Hall, Agile Geoscience, 2016
"""
from io import BytesIO, StringIO
import base64

import numpy as np
from PIL import Image
from PIL import ImageStat
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from pytsp import run, dumps_matrix

from flask import send_file


def serve_pil_image(img):
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


def is_greyscale(img):
    """
    Decide if an image is greyscale or not.
    """
    stat = ImageStat.Stat(img)
    if sum(stat.sum[:3])/3 == stat.sum[0]:
        return True
    return False


def get_imarray(img):
    """
    Turns a PIL image into an array in [0, 1] with shape (h*w, 3).

    Args:
        img (Image): a PIL image.

    Returns:
        ndarray.
    """
    return np.asarray(img)[..., :3] / 255.


def get_quanta(imarray, n_colours=256):
    """
    Reduces the colours in the image array down to some specified number,
    default 256. Usually you'll want at least 100, at most 500. Returns
    an unsorted colour table (codebook) for the colours.

    Call via get_codebook.

    Args:
        imarray (ndarray): The array from ``get_imarray()``.
        n_colours (int): The number of colours to reduce to.

    Returns:
        ndarray. An array of size (n_colours, 3).
    """
    h, w, c = imarray.shape
    im_ = imarray.reshape((w * h, c))

    # Define training set.
    n = min(h*w//50, n_colours*10)
    sample = shuffle(im_, random_state=0)[:n]
    kmeans = KMeans(n_clusters=n_colours).fit(sample)

    quanta = kmeans.cluster_centers_

    # Regularization.
    quanta[quanta > 1] = 1
    quanta[quanta < 0] = 0

    return quanta


def get_distances(quanta, zero_point=None):
    """
    Makes the complete distance matrix that the TSP solver needs. The
    adjustments are (1) adding the cool-point to start at, and (2) adding
    the zero-point to avoid creating a closed loop and make a path instead.

    Call via get_codebook.

    Args:
        quanta (ndarray): The array from ``get_quanta()``.
        zero_point (ndarray): The point to use as the starting point, e.g.
            [[0.25, 0, 0.5]], or [[0,0,0]], or even [[1,1,1]].

    Returns:
        ndarray. A matrix of size (n_colours+2, n_colours+2).
    """
    # Add cool-point.
    zero_point = zero_point or [[0.0, 0.00, 0.50]]
    p = np.vstack([zero_point, quanta])

    # Make distance matrix.
    dists = squareform(pdist(p, 'euclidean'))

    # The values in `dists` are floats in the range 0 to sqrt(3). 
    # Normalize the values to int16s.
    d = 32767 * dists / np.sqrt(3)
    d = d.astype(np.int16)

    # To use a TSP algo to solve the shortest Hamiltonian path problem,
    # we need to add a point that is zero units from every other point.
    row, col = d.shape
    d = np.insert(d, row, 0, axis=0)
    d = np.insert(d, col, 0, axis=1)

    return d


def sort_quanta(distances):
    """
    Solves the travelling salesman problem, with a magic zero-point, to
    find the shortest Hamiltonian path through the points. Returns the
    indices of the points in their sorted order.

    Call via get_codebook.

    Args:
        distances (ndarray): The distance matrix from ``get_distances()``.

    Returns:
        ndarray. A 1D array of size (n_colours).
    """
    # Set up the file describing the problem.
    outf = "/tmp/route.tsp"
    with open(outf, 'w') as f:
        f.write(dumps_matrix(distances, name="Route"))

    # Run the solver.
    tour = run(outf, start=0, solver="LKH")
    result = np.array(tour['tour'])

    # Slice off the initial value and the last value to account for the added
    # colours. Then subtract one to shift indices back to proper range.
    return result[1:-1] - 1


def get_codebook(imarray, n_colours=256, cool_point=None):
    """
    Finds and then sorts the colour table (aka codebook or palette). Wraps
    get_quanta, get_distances, and sort_quanta.

    Args:
        imarray (ndarray): The image array from ``get_imarray()``.
        n_colours (int): The number of colours to reduce to.
        cool_point (ndarray): The point to use as the starting point, e.g.
            [[0.25, 0, 0.5]], or [[0,0,0]], or even [[1,1,1]].

    Returns:
        ndarray. A matrix of size (n_colours+2, n_colours+2).
    """
    q = get_quanta(imarray, n_colours)
    s = sort_quanta(get_distances(q, cool_point))
    return q[s]


def make_cmap(colours):
    """
    Makes a Matplotlib colourmap from the colours provided.

    Args:
        colours (ndarray): The array of RGB triples to convert.

    Returns:
        matplotlib.colors.LinearSegmentedColormap.
    """
    # setting up color arrays
    r1 = np.array(c)[:, 0] # value of Red for the nth sample
    g1 = np.array(c)[:, 1] # value of Green for the nth sample
    b1 = np.array(c)[:, 2] # value of Blue for the nth sample

    r2 = r1 # value of Red at the nth sample
    r0 = np.linspace(0, 1, len(r1)) # position of the nth Red sample within the range 0 to 1

    g2 = g1 # value of Green at the nth sample
    g0 = np.linspace(0, 1, len(g1)) # position of the nth Green sample within the range 0 to 1

    b2 = b1 # value of Blue at the nth sample
    b0 = np.linspace(0, 1, len(b1)) # position of the nth Blue sample within the range 0 to 1

    # creating lists
    R = zip(r0, r1, r2)
    G = zip(g0, g1, g2)
    B = zip(b0, b1, b2)

    # creating list of above lists and transposing
    RGB = zip(R, G, B)
    rgb = zip(*RGB)
    #print rgb

    # creating dictionary
    k = ['red', 'green', 'blue'] # makes list of keys
    data_dict = dict(zip(k,rgb)) # makes a dictionary from list of keys and list of values

    return clr.LinearSegmentedColormap('my_colourmap', data_dict)


def recover_data(imarray, colours):
    """
    Given a sorted colour table, convert an image array into a data array in
    the closed interval [0, 1].

    Args:
        imarray (ndarray): The array of pixel data, as RGB triples.
        colours (ndarray): The array of sorted RGB triples.

    Returns:
        ndarray. The recovered data, the same shape as the input imaray.
    """
    # Imarray should be h x w x 3 array.
    kdtree = cKDTree(colours)
    _, ix = kdtree.query(imarray)

    # Scale.
    out = ix.astype(np.float)
    out /= np.amax(out)

    return out


def scale_data(data, interval):
    """
    Scale data to a new interval.

    Args:
        data (ndarray): The data to scale, in the closed interval [0,1].
        interval (tuple): A tuple of numbers to scale to.

    Returns:
        ndarray. The same shape as the input data.
    """
    mi, ma = interval
    return data * (ma-mi) + mi


def image_to_data(img, n_colours=128, interval=None):
    """
    Does everything.

    Args:
        img (Image): The image to convert.
        n_colours (int): The number of colours to reduce to.

    Returns:
        ndarray. The recovered data.
    """
    interval = interval or [0, 1]
    imarray = get_imarray(img)
    colours = get_codebook(imarray, n_colours=n_colours)
    recovered = recover_data(imarray, colours)
    return scale_data(recovered, interval)
