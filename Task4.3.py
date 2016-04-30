import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

img1 = cv2.imread('chapel1.png', 0)
img2 = cv2.imread('chapel2.png', 0)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)             # interesting points for img1   (4.3.a)
kp2, des2 = orb.detectAndCompute(img2,None)             #                    for img2

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)                   # match interesting points
print matches

DIST_LIMIT = 150                                        # distance criterion
inliers = []
outliers = []
for match1, match2 in matches:
    if match1.distance < DIST_LIMIT:                    # decide inlier vs. outlier by distance  (4.3.b.I)
        inliers.append(match1)
    else:
        outliers.append(match1)

src = []
dst = []
for inlier in inliers:                                  # inlier points src(img1), dst(img2)
    x1, y1 = kp1[inlier.queryIdx].pt
    x2, y2 = kp2[inlier.trainIdx].pt
    src.append([int(x1), int(y1)])
    dst.append([int(x2), int(y2)])
src = np.array(src)
dst = np.array(dst)

out = []
for outlier in outliers:                                # outlier points of img1
    x1, y1 = kp1[outlier.queryIdx].pt
    out.append([int(x1), int(y1)])

fmat = cv2.findFundamentalMat(src,dst,cv2.FM_LMEDS)[0]  # fundamental matrix by inlier points (src, dst)
fmat = normalize(fmat, axis=0, norm='l2')               # normalized     (4.3.b.II)

rgb1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)            # gray to rgb 
for o in out:
    cv2.circle(rgb1, tuple(o), 2, (0,255,0), -1)        # draw outliers with green dots on img1 (4.4.b.III)
plt.imshow(rgb1)
plt.show()

img3 = cv2.drawMatches(img1,kp1,img2,kp2,               # Plot the corresponding epipolar lines (4.3.c)
        inliers,None,matchColor=(255,0,0),singlePointColor=(0,255,0), flags=2)
plt.imshow(img3)
plt.show()
