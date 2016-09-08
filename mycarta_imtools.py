# -*- coding: utf-8 -*-
"""
Various functions.

by Matteo Niccoli, 2016
github.com/mycarta
"""
from io import StringIO

import requests
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, exposure
from skimage.morphology import disk
from skimage.morphology import opening
from skimage.morphology import remove_small_objects
from skimage.morphology import convex_hull_image

def find_data(img, min_int=0.03, max_int=0.97, disk_sz=2, hull=True):
    """
    Find the map in an image (using morphological operations) and return it.
    Heuristic assumption the map is the largest object in the map.
    
    Args:
    	img: (M, N, 3) or (M, N, 4). An RGB or RGBA image.
    	min_int : threshold value to eliminate ~black background.
            If min_int is not given, a default value of 0.03 is uded.
    	max_int : threshold value to eliminate ~white background.
            If max_int is not given, a default value of 0.97 is uded. 
    	disk_sz : size of disk-shaped structuring element for opening.
            If disk_sz is not given, a default value of 2 is uded.
    	hull : if True, the convex hull of the largest detected object
            is returned.
    
    Returns:  
        Image. (M, N) RGB image. An image with only the main map.    
    """
    rgbimg = img.convert('RGB')
    img = np.asarray(rgbimg)[:,:,:3]

    # Stretch contrast, convert to binary, and apply mild opening.
    p2, p98 = np.percentile(img, (2, 98))
    rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    binary = np.logical_and(color.rgb2gray(rescale) > min_int, color.rgb2gray(rescale) < max_int)
    binary = opening(binary, disk(disk_sz))

    # Keep only largest white object.
    label_objects, nb_labels = ndi.label(binary)
    sizes = np.bincount(label_objects.ravel())   
    sizes[0] = 0   
    if nb_labels < 2: # background not included in the count
        binary_objects = binary # in case the image already contained only the map
    else:
        binary_objects = remove_small_objects(binary, max(sizes))  

    # Remove holes.
    binary_holes = ndi.morphology.binary_fill_holes(binary_objects) 

    # Get convex hull image (smallest convex polygon that surround all white pixels).
    if hull:
        binary_holes = convex_hull_image(binary_holes)
    
    # Use it to make 3D mask.
    mask3 = np.zeros(img.shape)
    mask3[:,:,0] = binary_holes
    mask3[:,:,1] = binary_holes
    mask3[:,:,2] = binary_holes
    
    # Use mask to get only map in original image.
    final = np.ma.masked_where(mask3 ==0, img)
    final = final.filled(0)
    
    # Crop zero columns and zero rows, plus a few more to counter the opening.
    # see http://stackoverflow.com/a/31402351/1034648
    non_empty = np.where(final != 0)
    out = final[np.min(non_empty[0]) : np.max(non_empty[0]), 
                   np.min(non_empty[1]) : np.max(non_empty[1])][disk_sz:-disk_sz, disk_sz:-disk_sz]

    return Image.fromarray(np.uint8(out))
