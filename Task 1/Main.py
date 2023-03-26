# -*- coding: utf-8 -*-
"""

@author: Nathan Cai
"""
import SLIC
import cv2
import matplotlib.pyplot as plt

def main():
    
    img1 = cv2.imread('campus-spring.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.normalize(img1, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    img3 = SLIC.slic_parallel(img2, 6 , 25, iteration=5, n_jobs=6)
    
    list_imgs = [img1, img2]
    titles = ['original', 'Normalized', 'SLIC', 'Scikit']
    
    for x in range(4):
        plt.subplot(2,2,x+1)
        plt.title(titles[x])
        plt.axis('off')
        plt.imshow(list_imgs[x])
    
    return

if __name__ == "__main__":
    main()