# -*- coding: utf-8 -*-
"""

@author: Nathan Cai
"""

import numpy as np
from Util import print_progress as pb
import Util
import cv2


def slic(image, K, m, iteration = 10):
    """

    Parameters
    ----------
    image : Numpy Matrix or OpenCV image
        Image to process
    K : Int
        Number of centers
    m : Int
        Compactness Factor

    Returns
    -------
    TYPE
        Superpixel Image

    """
    # Step 1: Initialize cluster centers
    print('Initialize cluster centers')
    centers = []
    height = image.shape[0]
    width = image.shape[1]
    S = width * height
    N = int(np.sqrt(S/K))
    dx = int(width/N)
    dy = int(height/N)
    for i in range(N//2, height, dy):
        for j in range(N//2, width, dx):
            center = [j, i, image[i][j][0], image[i][j][1], image[i][j][2]]
            centers.append(center)
    centers = np.array(centers)

    # Step 2: Assign pixels to clusters
    print('Assign pixels to clusters')
    labels = np.zeros((height, width), dtype=np.int64)
    distances = np.full((height, width), np.inf)
    for k in range(iteration):
        for i in range(height):
            for j in range(width):        
                pb(j, width)
                for c in range(len(centers)):
                    dc = np.sqrt((j-centers[c][0])**2 + (i-centers[c][1])**2)
                    ds = np.sqrt((image[i][j][0]-centers[c][2])**2 + (image[i][j][1]-centers[c][3])**2 + (image[i][j][2]-centers[c][4])**2)
                    d = np.sqrt(dc**2 + (ds/m)**2)
                    if d < distances[i][j]:
                        distances[i][j] = d
                        labels[i][j] = c

        # Step 3: Update cluster centers
        print('Update cluster centers [%v]', k)
        new_centers = np.zeros_like(centers)
        counts = np.zeros((K,))
        for i in range(height):
            for j in range(width):
                c = labels[i][j]
                new_centers[c][0] += j
                new_centers[c][1] += i
                new_centers[c][2] += image[i][j][0]
                new_centers[c][3] += image[i][j][1]
                new_centers[c][4] += image[i][j][2]
                counts[c] += 1
        for c in range(K):
            if counts[c] > 0:
                new_centers[c] /= counts[c]
        centers = new_centers

    # Step 4: Generate superpixels
    print('Generate superpixels')
    superpixels = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            superpixels[i][j] = centers[labels[i][j]][2:5]
    return superpixels.astype(np.uint8)

from joblib import Parallel, delayed

def slic_parallel(image, K, m, n_jobs=1, iteration=5):
    # Divide the image into smaller sub-regions
    sub_regions = Util.divide_image(image, n_jobs)

    # Process each sub-region independently in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(slic)(sub_region, K, m, iteration) for sub_region in sub_regions)

    # Merge the results from each sub-region to obtain the final superpixels
    superpixels = Util.merge_results(results)

    return results