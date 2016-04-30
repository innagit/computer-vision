import cv2
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter
from skimage.measure import LineModel, ransac
import matplotlib.cm as cm

img1 = cv2.imread('chapel1.png', 0)  
img2 = cv2.imread('chapel2.png', 0)
N, M = img1.shape

harris1 = cv2.cornerHarris(np.float32(img1),2,3,0.04)
harris1 = cv2.dilate(harris1,None)
thd1 = harris1.max() * 0.02
harris2 = cv2.cornerHarris(np.float32(img2),2,3,0.04)
harris2 = cv2.dilate(harris2,None)
thd2 = harris2.max() * 0.04

feature1 = []
feature2 = []
for n in range(N):
    for m in range(M):
        if harris1[n][m] > thd1:
            img1[n][m]=255
            feature1.append([n,m,harris1[n][m]])
        if harris2[n][m] > thd2:
            img2[n][m]=255
            feature2.append([n,m,harris2[n][m]])

plt.subplot(1,2,1)       
plt.imshow(img1, cmap = cm.Greys_r)
plt.subplot(1,2,2)       
plt.imshow(img2, cmap = cm.Greys_r)
plt.show()
