import cv2 as cv
import numpy as np


img = cv.imread('photos/college.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Laplacion Method

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacion', lap)


# Sobel Method
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobelx', sobelx)
cv.imshow('Sobely', sobely)
cv.imshow('sobel', combined_sobel)


cv.waitKey(0)