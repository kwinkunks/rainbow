# -*- coding: utf-8 -*-
"""
Various functions.

by Matt Hall, Agile Geoscience, 2016
"""
from io import BytesIO
import base64

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import matplotlib.colors as clr
from pytsp import run, dumps_matrix


def get_imarray(img):
    return np.asarray(img)[..., :3] / 255.

def get_quanta(imarray, n_colours=256):
    h, w, d = im.shape
    im_ = im.reshape((w * h, d))

    # Define training set.
    n = min(im.size/100, n_colours*10)
    sample = shuffle(im_, random_state=0)[:n]
    kmeans = KMeans(n_clusters=n_colours).fit(sample)

    quanta = kmeans.cluster_centers_

    # Regularization.
    quanta[quanta > 1] = 1
    quanta[quanta < 0] = 0

    return quanta

def get_distances(quanta):
    # Add black point.
    p = np.vstack([[[0,0,0]], quanta])

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

    outf = "/tmp/route.tsp"
    with open(outf, 'w') as f:
        f.write(dumps_matrix(d, name="Route"))

    tour = run(outf, start=0, solver="LKH")
    result = np.array(tour['tour'])

    return result[1:-1]

def get_codebook(imarray, n_colours=256):
    q = get_quanta(imarray, n_colours)
    s = sort_quanta(get_distances(q))
    return q[s]

def make_cmap(colours):
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
    # Imarray should be h x w x 3 array.
    kdtree = cKDTree(colours)
    _, ix = kdtree.query(imarray)

    # Scale.
    out = ix.astype(np.float)
    out /= np.amax(out)

    return out

def image_to_data(img):
    imarray = get_imarray(img)
    colours = get_codebook(imarray)
    return recover_data(imarray, colours)
