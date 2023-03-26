# -*- coding: utf-8 -*-
"""
@author: Nathan Cai

Just some fun stuff for visualization
"""
import numpy as np
import cv2

# Print iterations progress
def print_progress(iteration, total):
    percent = int(100 * (iteration / total))
    bar_length = 20
    filled_length = int(bar_length * iteration // total)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {percent}% ', end='', flush=True)
    
def resize_img(img, scale_precent):
    width = int(img.shape[1] * scale_precent / 100)
    height = int(img.shape[0] * scale_precent / 100)
    dim = (width, height)
      
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
    
