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
    
    print('Assign pixels to clusters')
    labels = np.zeros((height, width), dtype=np.int64)
    distances = np.full((height, width), np.inf)
    for k in range(iteration):
        print('Iteration: ', k+1)
        for i in range(height):
            for j in range(width):        
                for c in range(K):
                    dc = np.sqrt((j-centers[c][0])**2 + (i-centers[c][1])**2)
                    ds = np.sqrt((image[i][j][0]-centers[c][2])**2 + (image[i][j][1]-centers[c][3])**2 + (image[i][j][2]-centers[c][4])**2)
                    d = np.sqrt(dc**2 + (ds/m)**2)
                    if d < distances[i][j]:
                        distances[i][j] = d
                        labels[i][j] = c

        print('Update cluster centers')
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

    print('Generate superpixels')
    superpixels = np.zeros_like(image)
    
    colors = np.random.rand(K, 3)
    
    for i in range(height):
        for j in range(width):
            #superpixels[i][j] = centers[labels[i][j]][2:5] * 255    
            superpixels[i][j] = colors[labels[i][j]][0:3] * 255
    
    superpixels = Draw_dots(superpixels, centers[0:K], K, 3)
    superpixels = superpixels.astype(np.uint8)
    
    return superpixels

def Draw_dots(image, points, k, radius):
    height = image.shape[0]
    width = image.shape[1]
    for p in range(k):
        for x in range(int(points[p][0]) - radius, int(points[p][0]) + radius):
            for y in range(int(points[p][1]) - radius, int(points[p][1]) + radius):
                if (x >= 0 and x < width-1) and (y >= 0 and y < height-1):
                    image[y][x] = (1-points[p][2:5])*255
    return image


