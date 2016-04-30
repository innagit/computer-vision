from matplotlib import pyplot as plt
import numpy as np
import cv2

from skimage.util import img_as_float
from skimage.feature import corner_harris, corner_peaks, plot_matches
from skimage.transform import  AffineTransform
from skimage.measure import ransac

img1 = img_as_float(cv2.imread('chapel1.png', 0))
img2 = img_as_float(cv2.imread('chapel2.png', 0))

corner_img1 = corner_peaks(corner_harris(img1), threshold_rel=0.01, min_distance=5)
corner_img2 = corner_peaks(corner_harris(img2), threshold_rel=0.01, min_distance=5)

src = []
dst = []
wid = 5
sigma = 3
for corner in corner_img1:
    src.append(corner)

    r, c = np.round(corner).astype(np.intp)
    wid_img1 = img1[r-wid:r+wid+1, c-wid:c+wid+1]

    y, x = np.mgrid[-wid:wid+1, -wid:wid+1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    weights = g

    SSEs = []
    for r2, c2 in corner_img2:
        wid_img2 = img2[r2-wid:r2+wid+1, c2-wid:c2+wid+1]
        SSEs.append(np.sum(weights * (wid_img1 - wid_img2)**2))
    dst.append(corner_img2[np.argmin(SSEs)])
    
src = np.array(src)
dst = np.array(dst)

# estimate affine transform model with RANSAC
model, inliers = ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=20, max_trials=100)

inlier_idxs = np.nonzero(inliers)[0]
fmat = cv2.findFundamentalMat(src[inlier_idxs],dst[inlier_idxs],cv2.FM_LMEDS)[0]

print (model.scale, model.translation, model.rotation)
print (fmat)

# visualize 
fig, ax = plt.subplots(nrows=1, ncols=1)
plot_matches(ax, img1, img2, src, dst, np.column_stack((inlier_idxs, inlier_idxs)), matches_color='r')
plt.show()

