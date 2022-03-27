import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('photos/college.jpg')

blank = np.zeros(img.shape[:2], dtype='uint8')

#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

circle =  cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(img,img)
cv.imshow('masked',masked)

"""
#grayscale histogram
gray_hist = cv.calcHist([gray], [0], masked, [256], [0,256])

plt.figure()
plt.title("histogram")
plt.xlabel("bins")
plt.ylabel("pixels")
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()
"""

#Color histogram
colors = ('b', 'g', 'r')
plt.figure()
plt.title("histogram")
plt.xlabel("bins")
plt.ylabel("pixels")
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    
plt.show()

cv.waitKey(0)
