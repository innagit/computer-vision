import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

img1 = cv2.imread('chapel1.png', 0)
img2 = cv2.imread('chapel2.png', 0)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)             # interest points for img1   (4.3.a)
kp2, des2 = orb.detectAndCompute(img2,None)             # for img2

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)                   # match interest points

matDist = []                                            # sort by dist
for n, (match1, match2) in enumerate(matches):
    matDist.append([match1.queryIdx, match1.trainIdx, match1.distance])
matDist = sorted(matDist, key=lambda tup: tup[2])

def isSeparated(m):                                     # find points well-separated from other points selected already 
    for l in sel:
        x1, y1 = kp1[matDist[m][0]].pt
        x2, y2 = kp1[matDist[l][0]].pt
        if (x1-x2)**2 + (y1-y2)**2 < 1600:  # 40**2     # eucledean distof 40 pixels
            return False
    return True

sel = []
outliers = []
src = np.zeros((8,2)).astype(int)
dst = np.zeros((8,2)).astype(int)
A89 = np.zeros((8,9))                                    # 8x9 matrix for fundamental matrix F
row = 0
for n in range(len(matDist)):                            # select 8 points separated by distance
    if row >= 8:
        for m in range(n+1, len(matDist)):               # not selected -> outliers
            x1, y1 = kp1[matDist[m][0]].pt
            outliers.append([int(x1), int(y1)])
        break
    if isSeparated(n):
        sel.append(n)
        x1, y1 = kp1[matDist[n][0]].pt
        x2, y2 = kp2[matDist[n][1]].pt
        src[row] = [int(x1), int(y1)]
        dst[row] = [int(x2), int(y2)]
        A89[row] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]    # row of A89 matrix
        row +=1

tol = 0.000001
U, s, V = np.linalg.svd(A89)
rank = (s > tol*s[0]).sum()
fmat = V[rank:].T.copy()                                     # null vector of A89xfmat = [0]
F = normalize(fmat[:,0].reshape(3,3), axis=0, norm='l2')     # normalized     (4.3.b.II)
print F

rgb1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
for outlier in outliers:
    cv2.circle(rgb1, tuple(outlier), 2, (0,255,0), -1)  # draw outliers with green dots on img1 (4.3.b.III)
plt.imshow(rgb1)
plt.show()

N, M = img1.shape
img3 = np.hstack((img1,img2))
rgb3 = cv2.cvtColor(img3,cv2.COLOR_GRAY2RGB)
for n in range(7):                                      # 7 sets of matching points
    cv2.circle(rgb3, tuple(src[n]), 5, (0,255,0))
    cv2.circle(rgb3, tuple((dst[n][0]+M,dst[n][1])), 5, (0,255,0))
    cv2.line(rgb3, tuple(src[n]), tuple((dst[n][0]+M,dst[n][1])), (255,0,0), 1)

plt.imshow(rgb3)
plt.show()

