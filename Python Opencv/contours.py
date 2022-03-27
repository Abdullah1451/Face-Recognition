import cv2 as cv
import numpy as np

#reading image
img = cv.imread('photos/college.jpg')

blank = np.zeros(img.shape, dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)

canny_edge = cv.Canny(blur, 125, 175)

#another way is using threshold() function rather than canny()

ret,thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

cv.imshow('thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

print(f"{len(contours)} contours")

cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('contours', blank)

cv.waitKey(0)