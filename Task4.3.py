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

matSrc = []                                             # sort by distance
for n, (match1, match2) in enumerate(matches):
    matSrc.append([match1.queryIdx, match1.trainIdx, match1.distance])
matSrc = sorted(matSrc, key=lambda tup: tup[2])    

def isSeparated(m):                                     # is separated to points selected already 
    for l in sel:
        x1, y1 = kp1[matSrc[m][0]].pt
        x2, y2 = kp1[matSrc[l][0]].pt
        if (x1-x2)**2 + (y1-y2)**2 < 1600:  # 40**2     # eucledean distance of 40 pixels
            return False
    return True

sel = []
src = []
dst = []
outliers = []
for n in range(len(matSrc)):                            # select 8 points separated order by distance
    if len(src) >= 8:                                   # for fundamental matrix F (fmat)
        for m in range(n+1, len(matSrc)):               # not selected -> outliers
            x1, y1 = kp1[matSrc[m][0]].pt
            outliers.append([int(x1), int(y1)])
        break
    if isSeparated(n):
        sel.append(n)
        x1, y1 = kp1[matSrc[n][0]].pt
        x2, y2 = kp2[matSrc[n][1]].pt
        src.append([int(x1), int(y1)])
        dst.append([int(x2), int(y2)])
src = np.array(src)
dst = np.array(dst)

fmat = cv2.findFundamentalMat(src,dst,cv2.FM_LMEDS)[0]  # fundamental matrix by inlier points (src, dst)
fmat = normalize(fmat, axis=0, norm='l2')               # normalized     (4.3.b.II)
print fmat

rgb1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)            # gray to rgb 
for outlier in outliers:
    cv2.circle(rgb1, tuple(outlier), 2, (0,255,0), -1)  # draw outliers with green dots on img1 (4.4.b.III)
plt.imshow(rgb1)
plt.show()
        
N, M = img1.shape
img3 = np.hstack((img1,img2))
rgb3 = cv2.cvtColor(img3,cv2.COLOR_GRAY2RGB)            # gray to rgb 
for n in range(7):                                      # 7 sets of matching points
    cv2.circle(rgb3, tuple(src[n]), 5, (0,255,0))       
    cv2.circle(rgb3, tuple((dst[n][0]+M,dst[n][1])), 5, (0,255,0))
    cv2.line(rgb3, tuple(src[n]), tuple((dst[n][0]+M,dst[n][1])), (255,0,0), 1)
    
plt.imshow(rgb3)
plt.show()
