# -*- coding: utf-8 -*-
"""
Various functions.

by Matt Hall, Agile Geoscience, 2016
"""
from io import BytesIO
import datetime
import sys

import numpy as np
from PIL import ImageStat
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.neighbors import BallTree
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from pytsp import run, dumps_matrix
import boto3
from obspy.core import Trace, Stream
from obspy.io.segy.segy import SEGYTraceHeader
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYBinaryFileHeader

from flask import send_file


def get_params(request):

    # Get raw parameters.
    params = {'url': request.args.get('url')}
    params['n_colours'] = request.args.get('n_colours') or '128'
    params['interval'] = request.args.get('interval') or '0,1'
    params['region'] = request.args.get('region') or ''
    params['sampling'] = request.args.get('sampling') or 'random'
    params['cool_point'] = request.args.get('cool_point') or None
    params['prod'] = request.args.get('prod') or ''
    params['recover'] = request.args.get('recover') or ''
    params['format'] = request.args.get('format') or 'PNG'
    params['return_cmap'] = request.args.get('return_cmap') or ''
    params['hull'] = request.args.get('hull') or ''

    # Condition parameters.
    params['n_colours'] = int(params['n_colours'])
    params['prod'] = False if params['prod'].lower() in ['false', 'no', '0'] else True
    params['recover'] = False if params['recover'].lower() in ['false', 'no', '0'] else True
    params['hull'] = False if params['hull'].lower() in ['false', 'no', '0'] else True
    params['return_cmap'] = True if params['return_cmap'].lower() in ['true', 'yes', '1', 'on', 'ok'] else False
    params['interval'] = [int(n) for n in params['interval'].split(',')]

    if params['cool_point'] is not None:
        try:
            cool_point = [int(n) for n in params['cool_point'].split(',')]
        except:
            cool_point = None
        params['cool_point'] = cool_point

    return params


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
    rgbimg = img.convert('RGB')
    return np.asarray(rgbimg)[..., :3] / 255.


def mask_colours(a, tree, colours, tolerance=1e-6, leave=0):
    """
    Remove particular colours from the palette.

    TODO
        Only remove them if they aren't in the colourmap... i.e. if they are
        out on their own.
    """
    target = tree.query_radius(colours, tolerance)
    mask = np.ones(a.shape[0], dtype=bool)
    end = None if leave < 2 else 1 - leave
    print(leave, end)
    for t in target:
        mask[t[leave:end]] = False
    return a[mask]


def isolate_black(a, tree):
    count = min([6, a.shape[0]//15])
    distances, indices = tree.query([[0,0,0]], count)
    for (dist, idx) in zip(distances, indices):
        tol = np.diff(dist[1:]).mean()
        if dist[0] < tol / 3:
            # Then there's effectively a point
            # at the target colour.
            if dist[1] > 3 * tol:
                # Point is prolly not in cbar
                # so we eliminate.
                a = np.delete(a, idx[0], axis=0)
            else:
                # Colour is part of colourbar.
                # If it's right at black, eliminate it.
                if dist[0] < tol / 30:
                    a = np.delete(a, idx[0], axis=0)
        else:
            # There's no point that colour. Add one.
            pass
    return a


def isolate_white(a, tree):
    count = min([6, a.shape[0]//15])
    distances, indices = tree.query([[1, 1, 1]], count)
    for (dist, idx) in zip(distances, indices):
        tol = np.diff(dist[1:]).mean()
        if dist[0] < tol / 3:
            # Then there's effectively a point
            # at the target colour.
            if dist[1] > 3 * tol:
                # Point is prolly not in cbar
                # so we eliminate.
                a = np.delete(a, idx[0], axis=0)
    return a


def remove_duplicates(a, tree, tolerance=1e-6):
    """
    Remove all duplicate points, within the given tolerance.
    """
    for c in a:
        a = mask_colours(a, tree, [c], leave=1)
    return a


def remove_isolates(a, tree, min_neighbours):
    """
    Remove all points with fewer than 2r neighbours in a radius of r,
    where r is the median of all nearest neighbour distances.
    """
    radius = (min_neighbours + 1) / 2
    d, _ = tree.query(a, 2)
    tol = np.median(d[:, 1]) * radius
    i = tree.query_radius(a, tol)
    indices_of_isolates = [j for j, k in enumerate(i) if k.size < 2*radius]
    return np.delete(a, indices_of_isolates, axis=0)


def get_quanta(imarray, n_colours=256, sampling=None, min_neighbours=6):
    """
    Reduces the colours in the image array down to some specified number,
    default 256. Usually you'll want at least 100, at most 500. Returns
    an unsorted colour table (codebook) for the colours.

    Call via get_codebook.

    Args:
        imarray (ndarray): The array from ``get_imarray()``.
        n_colours (int): The number of colours to reduce to.
        sampling (str): 'random' for random pixels from the image.
            'columns' for random columns (eg for seismic data).
            'rows' for random rows.
        min_neighbours (int): The minimum number of neighbours a point
            should have.

    Returns:
        ndarray. An array of size (n_colours, 3).
    """
    h, w, c = imarray.shape

    # Define training set.
    n = min(h*w//10, n_colours*100)

    if sampling == 'rows':
        nrow = n // imarray.shape[0]  # How many traces do we need?
        data = imarray[np.random.randint(0, imarray.shape[1], nrow)]
    elif sampling == 'columns':
        ntr = n // imarray.shape[1]  # How many traces do we need?
        data = imarray[np.random.randint(0, imarray.shape[0], ntr)]
    else:  # random
        im_ = imarray.reshape((-1, c))
        data = shuffle(im_, random_state=0)[:n]

    sample = data.reshape((-1, c))

    # Fit the data.
    kmeans = KMeans(n_clusters=n_colours).fit(sample)

    quanta = kmeans.cluster_centers_

    # Regularization.
    # For some reason this first bit seems to be necessary sometimes
    quanta[quanta > 1] = 1
    quanta[quanta < 0] = 0

    # tree = BallTree(quanta)
    # quanta = remove_duplicates(quanta, tree)

    # tree = BallTree(quanta)
    # quanta = remove_isolates(quanta, tree, min_neighbours)

    return quanta


def get_distances(quanta, cool_point=None):
    """
    Makes the complete distance matrix that the TSP solver needs. The
    adjustments are (1) adding the cool-point to start at, and (2) adding
    the zero-point to avoid creating a closed loop and make a path instead.

    Call via get_codebook.

    Args:
        quanta (ndarray): The array from ``get_quanta()``.
        cool_point (ndarray): The point to use as the starting point, e.g.
            [[0, 0, 0.5]], [[0.25, 0, 0.5]], or [[0, 0, 0]], or even [[1, 1, 1]].

    Returns:
        ndarray. A matrix of size (n_colours+2, n_colours+2).
    """
    # Add cool-point.
    cool_point = cool_point or [[0.0, 0.0, 0.0]]
    p = np.vstack([cool_point, quanta])

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


def get_codebook(imarray, n_colours=128, sampling=None, cool_point=None):
    """
    Finds and then sorts the colour table (aka codebook or palette). Wraps
    get_quanta, get_distances, and sort_quanta.

    Args:
        imarray (ndarray): The image array from ``get_imarray()``.
        n_colours (int): The number of colours to reduce to.
        sampling (str): The way to sample the image (default 'random')
        cool_point (ndarray): The point to use as the starting point.

    Returns:
        ndarray. A matrix of size (n_colours+2, n_colours+2).
    """
    q = get_quanta(imarray, n_colours,  sampling)
    d = get_distances(q, cool_point)
    r = sort_quanta(d)
    print(len(r), r)

    # Compute the dataspace.
    dataspace = np.concatenate([[0], np.cumsum([d[p, q] for p, q in zip(r, r[1:])])])

    return q[r], dataspace


def recover_data(imarray, colours, dataspace=None):
    """
    Given a sorted colour table, convert an image array into a data array in
    the closed interval [0, 1].

    Args:
        imarray (ndarray): The array of pixel data, as RGB triples.
        colours (ndarray): The array of sorted RGB triples.

    Returns:
        ndarray. The recovered data, the same shape as the input imarray.
    """
    if dataspace is None:
        dataspace = np.arange(0, len(colours), dtype=np.float)

    kdtree = cKDTree(colours)
    dx, ix = kdtree.query(imarray)
    data = dataspace[ix]
    print(dataspace)

    # Scale.
    out = data.astype(np.float)
    out /= np.amax(out)

    # Remove anything that maps too far.
    out[dx > np.sqrt(3)/8] = np.nan

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


def image_to_data(img, n_colours=128, sampling=None, cool_point=None, interval=None):
    """
    Does everything.

    Args:
        img (Image): The image to convert.
        n_colours (int): The number of colours to reduce to.

    Returns:
        ndarray. The recovered data. [0-1]
    """
    interval = interval or [0, 1]
    imarray = get_imarray(img)
    colours, dataspace = get_codebook(imarray,
                                      n_colours=n_colours,
                                      sampling=sampling,
                                      cool_point=cool_point
                                      )
    recovered = recover_data(imarray, colours, dataspace)
    return scale_data(recovered, interval), colours


def get_url(databytes, ext, uuid1):
    """
    Upload to AWS S3 storage and collect URL.
    """
    file_link = ''
    now = datetime.datetime.now()
    expires = now + datetime.timedelta(minutes=240)
    success = False

    try:
        from secrets import KEY, SECRET
        session = boto3.session.Session(aws_access_key_id=KEY,
                                        aws_secret_access_key=SECRET,
                                        region_name='us-east-1'
                                        )
        client = session.client('s3')
        key = uuid1 + '.' + ext
        bucket = 'keats'
        acl = 'public-read'  # For public file.
        params = {'Body': databytes,
                  'Expires': expires,
                  'Bucket': bucket,
                  'Key': key,
                  'ACL': acl,
                  }
        r = client.put_object(**params)
        success = r['ResponseMetadata']['HTTPStatusCode'] == 200
    except:
        print('Upload to S3 failed')

    if success:
        # Only do this if successfully uploaded, because
        # you always get a link, even if no file.
        if acl == 'public-read':
            file_link = 'https://s3.amazonaws.com/{}/{}'.format(bucket, key)
        else:
            try:
                params = {'Bucket': bucket,
                          'Key': key}
                file_link = client.generate_presigned_url('get_object',
                                                          Params=params,
                                                          ExpiresIn=3600)
            except:
                print('Retrieval of S3 link failed')

    return file_link


def write_segy(f, data):
    """
    Write a 2D NumPY array to an open file handle f.
    """
    stream = Stream()

    # Data is in [0, 1] so rescale to 8-bit.
    # USING 16-bit because can't save as 8-bit int in ObsPy.
    data = np.int16((data-0.5)*255)

    for i, trace in enumerate(data):

        # Make the trace.
        tr = Trace(trace)

        # Add required data.
        tr.stats.delta = 0.004

        # Add yet more to the header (optional).
        tr.stats.segy = {'trace_header': SEGYTraceHeader()}
        tr.stats.segy.trace_header.trace_sequence_number_within_line = i + 1
        tr.stats.segy.trace_header.receiver_group_elevation = 0

        # Append the trace to the stream.
        stream.append(tr)

    # Text header.
    stream.stats = AttribDict()
    stream.stats.textual_file_header = '{:80s}'.format('Generated by Keats.').encode()
    stream.stats.textual_file_header += '{:80s}'.format('Sample interval unknown.').encode()
    stream.stats.textual_file_header += '{:80s}'.format('IEEE floats.').encode()

    # Binary header.
    stream.stats.binary_file_header = SEGYBinaryFileHeader()
    stream.stats.binary_file_header.trace_sorting_code = 4
    stream.stats.binary_file_header.seg_y_format_revision_number = 0x0100

    # Write the data.
    # Encoding should be 8, but that doesn't work.
    stream.write(f, format='SEGY', data_encoding=3, byteorder=sys.byteorder)

    return f
