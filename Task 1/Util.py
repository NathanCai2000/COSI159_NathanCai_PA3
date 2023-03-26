# -*- coding: utf-8 -*-
"""
@author: Nathan Cai

Just some fun stuff for visualization
"""
import numpy as np

# Print iterations progress
def print_progress(iteration, total):
    percent = int(100 * (iteration / total))
    bar_length = 20
    filled_length = int(bar_length * iteration // total)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {percent}% ', end='', flush=True)
        
def divide_image(image, n_jobs):
    height, width, channels = image.shape
    sub_height = height // n_jobs
    sub_width = width
    sub_regions = []
    for i in range(n_jobs):
        start_y = i * sub_height
        end_y = start_y + sub_height
        sub_region = image[start_y:end_y, :, :]
        sub_regions.append(sub_region)
    return sub_regions

def merge_results(results):
    # Combine the labels and centroids from each sub-region
    all_labels = []
    all_centroids = []
    for r in results:
        labels, centroids = r
        all_labels.append(labels)
        all_centroids.append(centroids)

    # Merge the labels and centroids into a single output image
    height, width, __, __ = all_labels[0].shape
    output_image = np.zeros((height, width), dtype=np.uint32)
    for i in range(len(all_labels)):
        output_image += (i + 1) * all_labels[i]

    return output_image
